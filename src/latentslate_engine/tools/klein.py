from __future__ import annotations

import importlib.util
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
    AssetInput,
    ChoiceOption,
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    ToolRequirement,
    WorkflowKind,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.klein import (
    KLEIN_SIZE_PRESETS,
    KleinRuntime,
    KleinVariant,
    resolve_klein_runtime_plan,
)
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext

KLEIN4B_TEXT_TO_IMAGE_ID = UUID("077f54e4-14f9-5aaf-973b-5d89d0214214")
KLEIN4B_IMAGE_TO_IMAGE_ID = UUID("6e52c99c-35f3-5eba-ba32-4a800756beed")
KLEIN9B_TEXT_TO_IMAGE_ID = UUID("e329a7d2-c145-4299-96ef-f2b70376d499")
KLEIN9B_IMAGE_TO_IMAGE_ID = UUID("3333a6bd-8e71-4236-9372-bad407161803")

_KLEIN_ATTENTION_MODES = frozenset(
    {"native", "flash_hub", "flash3_hub", "flash4_hub", "sage_hub"}
)
_KLEIN_OFFLOAD_MODES = frozenset(
    {"none", "model", "sequential", "group_block", "group_leaf"}
)
_KLEIN_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})
_KLEIN_VAE_MODES = frozenset({"on", "off"})
_KLEIN_CACHE_MODES = frozenset({"none", "prompt", "media"})


def _runtime_availability(variant: KleinVariant) -> tuple[bool, str | None]:
    modules = ["torch", "diffusers", "transformers", "accelerate", "PIL", "peft"]
    if variant == "klein9b":
        modules.extend(("modelopt", "torchao"))
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        return False, f"Run `uv sync`; missing Klein runtime packages: {', '.join(missing)}"
    return True, None


def _prompt_input() -> ToolInput:
    return ToolInput(
        key="prompt",
        label="Prompt",
        type=InputType.TEXT,
        role=InputRole.PROMPT,
        required=True,
        ui=InputUi(
            group="Prompt",
            multiline=True,
            placeholder="Describe the image to generate or the edit to apply.",
        ),
    )


def _seed_input() -> ToolInput:
    return ToolInput(
        key="seed",
        label="Seed",
        type=InputType.INTEGER,
        role=InputRole.SEED,
        required=True,
        default=0,
        ui=InputUi(group="Advanced", advanced=True, min=0, step=1),
    )


def _size_input(*, include_source: bool, default: str) -> ToolInput:
    values = list(KLEIN_SIZE_PRESETS)
    if not include_source:
        values.remove("source")
    return ToolInput(
        key="size",
        label="Size",
        type=InputType.CHOICE,
        required=True,
        default=default,
        options=[ChoiceOption(value=value, label=value) for value in values],
        ui=InputUi(group="Output"),
    )


def _reference_inputs() -> list[ToolInput]:
    return [
        ToolInput(
            key="source_image",
            label="Input Image 1",
            type=InputType.IMAGE,
            role=InputRole.SOURCE_IMAGE,
            required=True,
            ui=InputUi(group="Input"),
        ),
        ToolInput(
            key="reference_image_2",
            label="Input Image 2",
            type=InputType.IMAGE,
            required=False,
            ui=InputUi(group="Input", advanced=True),
        ),
        ToolInput(
            key="reference_image_3",
            label="Input Image 3",
            type=InputType.IMAGE,
            required=False,
            ui=InputUi(group="Input", advanced=True),
        ),
    ]


class _KleinBase(Tool):
    variant: KleinVariant
    model_label: str
    bundle_id: str

    def model_family(self) -> str:
        return self.variant

    def execution_capabilities(self) -> ExecutionCapabilities:
        quantization = {"native", "bf16", "int8"}
        if self.variant == "klein9b":
            quantization.add("nvfp4")
        return ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset({"safetensors"}),
            attention_modes=_KLEIN_ATTENTION_MODES,
            offload_modes=_KLEIN_OFFLOAD_MODES,
            quantization_modes=frozenset(quantization),
            compile_modes=_KLEIN_COMPILE_MODES,
            compile_fullgraph=True,
            compile_dynamic=True,
            vae_tiling_modes=_KLEIN_VAE_MODES,
            vae_slicing_modes=_KLEIN_VAE_MODES,
            cache_modes=_KLEIN_CACHE_MODES,
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        optimizations = request.optimizations
        quantization = str(optimizations.get("quantization", "inherit"))
        offload = str(optimizations.get("offload", "inherit"))
        attention = str(optimizations.get("attention", "inherit"))
        compile_enabled = bool(optimizations.get("compile", False))
        use_stream = bool(optimizations.get("group_offload_use_stream", False))
        blocks = optimizations.get("group_offload_blocks")
        vae_tiling = str(optimizations.get("vae_tiling", "inherit"))

        if attention in {"flash_hub", "flash3_hub", "flash4_hub", "sage_hub"} and (
            importlib.util.find_spec("kernels") is None
        ):
            errors.append(
                f"Klein attention mode {attention!r} requires the Hugging Face "
                "`kernels` package on this Engine host"
            )
        if compile_enabled and request.loras:
            errors.append(
                "Klein LoRA switching is not supported on a compiled transformer; "
                "use a non-compiled variant or bake/fuse the adapter into a separate model"
            )
        if compile_enabled and offload in {"sequential", "group_block", "group_leaf"}:
            errors.append(
                f"Klein compile is not supported with offload mode {offload!r}; "
                "use offload='none' or offload='model'"
            )
        if request.loras and quantization in {"int8", "nvfp4"}:
            errors.append(
                f"Klein LoRAs are not yet supported with quantization {quantization!r}"
            )
        if self.variant == "klein9b" and request.loras and quantization == "inherit":
            errors.append(
                "Klein 9B LoRA variants must explicitly choose quantization='native' "
                "or quantization='bf16'; the inherited consumer profile is NVFP4"
            )
        if request.loras and offload in {"group_block", "group_leaf"}:
            errors.append("Klein LoRA switching is not yet supported with group offloading")
        if quantization == "nvfp4" and request.model_override:
            errors.append(
                "Klein NVFP4 currently supports only the built-in 9B consumer checkpoint, "
                "not an arbitrary model override"
            )
        if (
            self.variant == "klein9b"
            and not request.model_override
            and quantization in {"native", "bf16", "int8"}
        ):
            errors.append(
                f"Klein 9B quantization {quantization!r} requires a complete Diffusers "
                "model override; the built-in consumer bundle stores an NVFP4 transformer"
            )
        if self.variant == "klein9b" and request.model_override and quantization == "inherit":
            errors.append(
                "Klein 9B model overrides must explicitly select native, BF16, or INT8 "
                "quantization instead of inheriting the built-in NVFP4 profile"
            )
        if offload in {"group_block", "group_leaf"} and use_stream and vae_tiling == "on":
            errors.append(
                "streamed group offload with VAE tiling needs a model-specific warmup "
                "forward and is intentionally disabled in the safe Klein adapter"
            )
        if offload == "group_block" and use_stream and blocks not in {None, 1}:
            errors.append(
                "streamed block-level group offload requires group_offload_blocks = 1"
            )
        return errors

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_klein_runtime_plan(
            context.settings,
            self.variant,
            context.execution,
        )

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> KleinRuntime:
        key = ("flux2_klein", self.variant, plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: KleinRuntime(context.settings, self.variant, plan),
        )

    def _generate(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        source_assets: list[AssetInput],
    ) -> list[StoredArtifact]:
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        runtime = self._runtime(context, plan)
        output_path = context.storage.artifact_path(context.job_id, "output.png")
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                size_name=str(inputs["size"]),
                seed=int(inputs["seed"]),
                image_paths=[context.resolve_asset(asset.asset_id) for asset in source_assets],
                reference_keys=[str(asset.asset_id) for asset in source_assets],
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "loras": metadata["loras"],
                    "cache": metadata["cache"],
                }
            )
        finally:
            if not plan.keep_pipeline_loaded:
                unloaded = RUNTIME_MANAGER.unload_runtime(runtime)
                context.record_provenance(runtime_unloaded_after_job=unloaded)
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output_path.name,
                content_type="image/png",
                path=output_path,
                role="primary",
                media_type="image",
                metadata=metadata,
            )
        ]

    @staticmethod
    def _source_assets(inputs: dict[str, Any]) -> list[AssetInput]:
        assets = [AssetInput.model_validate(inputs["source_image"])]
        for key in ("reference_image_2", "reference_image_3"):
            if value := inputs.get(key):
                assets.append(AssetInput.model_validate(value))
        return assets

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers",
            "pipeline": "Flux2KleinPipeline",
            "model_family": f"flux2_{self.variant}",
        }


class Klein4BTextToImageTool(_KleinBase):
    variant = "klein4b"
    model_label = "Klein 4B"
    bundle_id = "klein4b-basic"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability(self.variant)
        return ToolDescriptor(
            id=KLEIN4B_TEXT_TO_IMAGE_ID,
            key="flux2_klein4b.text_to_image",
            schema_revision=1,
            name="Klein 4B Text to Image",
            description=(
                "Fast four-step text-to-image generation with FLUX.2 Klein 4B, "
                "the consumer-GPU model used by the imported LatentSlate workflows."
            ),
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                _size_input(include_source=False, default="512x512"),
                _seed_input(),
            ],
            requirements=[ToolRequirement(bundle_id=self.bundle_id)],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(context, inputs, source_assets=[])


class Klein4BImageToImageTool(_KleinBase):
    variant = "klein4b"
    model_label = "Klein 4B"
    bundle_id = "klein4b-basic"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability(self.variant)
        return ToolDescriptor(
            id=KLEIN4B_IMAGE_TO_IMAGE_ID,
            key="flux2_klein4b.image_to_image",
            schema_revision=1,
            name="Klein 4B Image to Image (1-3 refs)",
            description=(
                "Edit or compose one to three reference images with the fast "
                "four-step FLUX.2 Klein 4B model."
            ),
            workflow_kind=WorkflowKind.IMAGE_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                *_reference_inputs(),
                _size_input(include_source=True, default="source"),
                _seed_input(),
            ],
            requirements=[ToolRequirement(bundle_id=self.bundle_id)],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(
            context,
            inputs,
            source_assets=self._source_assets(inputs),
        )


class KleinTextToImageTool(_KleinBase):
    """Stable V0 Klein 9B text-to-image tool."""

    variant = "klein9b"
    model_label = "Klein 9B"
    bundle_id = "klein9b-basic"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability(self.variant)
        return ToolDescriptor(
            id=KLEIN9B_TEXT_TO_IMAGE_ID,
            key="flux2_klein9b.text_to_image",
            schema_revision=2,
            name="Klein 9B Text to Image",
            description="Generate an image from text with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                _size_input(include_source=False, default="1024x1024"),
                _seed_input(),
            ],
            requirements=[ToolRequirement(bundle_id=self.bundle_id)],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(context, inputs, source_assets=[])


class KleinImageToImageTool(_KleinBase):
    """Stable V0 Klein 9B image-to-image tool."""

    variant = "klein9b"
    model_label = "Klein 9B"
    bundle_id = "klein9b-basic"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability(self.variant)
        return ToolDescriptor(
            id=KLEIN9B_IMAGE_TO_IMAGE_ID,
            key="flux2_klein9b.image_to_image",
            schema_revision=2,
            name="Klein 9B Image to Image (1-3 refs)",
            description="Edit or compose one to three images with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.IMAGE_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                *_reference_inputs(),
                _size_input(include_source=True, default="source"),
                _seed_input(),
            ],
            requirements=[ToolRequirement(bundle_id=self.bundle_id)],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(
            context,
            inputs,
            source_assets=self._source_assets(inputs),
        )
