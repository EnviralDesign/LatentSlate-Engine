from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
    AssetInput,
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
from ..resources import ArtifactPrecision, ArtifactQuantization, ResourceDescriptor, ResourceFormat
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.klein import (
    KLEIN_CANVAS,
    KleinRuntime,
    KleinVariant,
    resolve_klein_runtime_plan,
)
from ..runtime.klein_support import klein_runtime_support
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext
from .canvas import dimension_tool_inputs

KLEIN4B_TEXT_TO_IMAGE_ID = UUID("077f54e4-14f9-5aaf-973b-5d89d0214214")
KLEIN4B_IMAGE_TO_IMAGE_ID = UUID("6e52c99c-35f3-5eba-ba32-4a800756beed")
KLEIN9B_TEXT_TO_IMAGE_ID = UUID("e329a7d2-c145-4299-96ef-f2b70376d499")
KLEIN9B_IMAGE_TO_IMAGE_ID = UUID("3333a6bd-8e71-4236-9372-bad407161803")

_KLEIN_ATTENTION_MODES = frozenset({"native", "flash_hub", "flash3_hub", "flash4_hub", "sage_hub"})
_KLEIN_OFFLOAD_MODES = frozenset(
    {"none", "model", "sequential", "group_block", "group_leaf", "staged"}
)
_KLEIN_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})
_KLEIN_VAE_MODES = frozenset({"on", "off"})
_KLEIN_CACHE_MODES = frozenset({"none", "prompt", "media"})


def _runtime_availability(variant: KleinVariant) -> tuple[bool, str | None]:
    support = klein_runtime_support()
    return support.core_available, support.core_reason


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


def _dimension_inputs(*, default_width: int, default_height: int) -> list[ToolInput]:
    """Publish the exact Klein /16 canvas; the engine never rewrites it."""

    return dimension_tool_inputs(
        KLEIN_CANVAS,
        default_width=default_width,
        default_height=default_height,
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

    def variant_base_availability(self) -> tuple[bool, str | None]:
        support = klein_runtime_support()
        return support.core_available, support.core_reason

    def model_resource_components(self) -> frozenset[str]:
        return frozenset({"transformer"})

    def execution_capabilities(self) -> ExecutionCapabilities:
        support = klein_runtime_support()
        if not support.core_available:
            return ExecutionCapabilities()

        attention = {"native"}
        if support.kernels_available:
            attention.update({"flash_hub", "flash3_hub", "flash4_hub", "sage_hub"})
        formats = {"diffusers"}
        quantization = {"native", "bf16"}
        formats.add("safetensors")
        quantization.update({"fp8", "nvfp4"})
        return ExecutionCapabilities(
            model_formats=frozenset(formats),
            recipe_types=frozenset(
                {"klein4_stored" if self.variant == "klein4b" else "klein9_stored"}
            ),
            lora_formats=(frozenset({"safetensors"}) if support.peft_available else frozenset()),
            attention_modes=frozenset(attention),
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
        offload = str(optimizations.get("offload", "inherit"))
        attention = str(optimizations.get("attention", "inherit"))
        compile_enabled = bool(optimizations.get("compile", False))
        use_stream = bool(optimizations.get("group_offload_use_stream", False))
        blocks = optimizations.get("group_offload_blocks")
        vae_tiling = str(optimizations.get("vae_tiling", "inherit"))
        support = klein_runtime_support()
        stored_format = "safetensors" in request.model_formats
        requested_quantization = str(optimizations.get("quantization", "inherit"))
        stored_quantization = requested_quantization in {"fp8", "nvfp4"}
        component_recipe = request.recipe_type in {"klein4_stored", "klein9_stored"}
        stored_request = (
            stored_format or stored_quantization or offload == "staged" or component_recipe
        )

        if requested_quantization == "nvfp4" and not component_recipe:
            errors.append("Klein NVFP4 is available only through its typed component recipe")

        if attention in {"flash_hub", "flash3_hub", "flash4_hub", "sage_hub"} and (
            not support.kernels_available
        ):
            errors.append(
                support.kernels_reason
                or f"Klein attention mode {attention!r} requires the Hugging Face kernels package"
            )
        if request.loras and not support.peft_available:
            errors.append(support.peft_reason or "Klein LoRAs require PEFT")
        if stored_request:
            if self.variant != "klein4b" and not component_recipe:
                errors.append(
                    "standalone stored quantized SafeTensors execution is supported only "
                    "for Klein 4B; Klein 9B requires its typed component recipe"
                )
            if stored_format and requested_quantization not in {
                "inherit",
                "fp8",
                "nvfp4",
            }:
                errors.append("Klein stored SafeTensors requires quantization='fp8' or 'nvfp4'")
            if stored_quantization and not stored_format and not component_recipe:
                errors.append(
                    f"Klein quantization={requested_quantization!r} requires a stored "
                    "SafeTensors model override"
                )
            if offload == "staged" and not (
                stored_format or stored_quantization or component_recipe
            ):
                errors.append(
                    "Engine-owned staged residency is reserved for a stored quantized transformer"
                )
            if component_recipe and request.model_override:
                errors.append("Klein component recipes cannot also declare a model override")
            if attention not in {"inherit", "native"}:
                errors.append("Klein stored quantized execution supports native attention only")
            if offload not in {"inherit", "staged"}:
                errors.append(
                    "Klein stored quantized execution requires Engine-owned staged residency"
                )
            if compile_enabled:
                errors.append("Klein stored quantized execution does not yet support torch.compile")
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
        if request.loras and offload in {"group_block", "group_leaf"}:
            errors.append("Klein LoRA switching is not yet supported with group offloading")
        if offload in {"group_block", "group_leaf"} and use_stream and vae_tiling == "on":
            errors.append(
                "streamed group offload with VAE tiling needs a model-specific warmup "
                "forward and is intentionally disabled in the safe Klein adapter"
            )
        if offload == "group_block" and use_stream and blocks not in {None, 1}:
            errors.append("streamed block-level group offload requires group_offload_blocks = 1")
        return errors

    def validate_model_resource(
        self,
        resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        if resource.component is not None and resource.format != ResourceFormat.SAFETENSORS:
            return [
                "Klein transformer component promotion requires a standalone SafeTensors artifact"
            ]
        if resource.format != ResourceFormat.SAFETENSORS:
            return []
        if self.variant != "klein4b":
            return ["standalone SafeTensors model resources are supported only for Klein 4B"]
        stored_fp8 = (
            resource.precision == ArtifactPrecision.FP8
            and resource.quantization == ArtifactQuantization.NATIVE
        )
        if not stored_fp8:
            return [
                (
                    "Klein standalone model overrides require precision='fp8' and "
                    "quantization='native'; NVFP4 is available only through its typed recipe"
                )
            ]
        try:
            from ..runtime.klein_stored_adapter import plan_klein_stored_transformer

            plan_klein_stored_transformer(path).require_available()
        except (OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

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
                width=inputs.get("width"),
                height=inputs.get("height"),
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
                    "quantized_dispatch": metadata.get("quantized_dispatch"),
                    "text_encoder_quantized_dispatch": metadata.get(
                        "text_encoder_quantized_dispatch"
                    ),
                    "residency_policy": metadata.get("residency_policy"),
                    "reference_preprocessing": metadata.get("reference_preprocessing"),
                    "loras": metadata["loras"],
                    "lora_dispatch": metadata.get("lora_dispatch"),
                    "cache": metadata["cache"],
                }
            )
        except BaseException:
            if runtime.residency_poisoned():
                evicted = RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
                context.record_provenance(runtime_evicted_after_residency_failure=evicted)
            raise
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
            schema_revision=4,
            name="Klein 4B Text to Image",
            description=(
                "Fast four-step text-to-image generation with FLUX.2 Klein 4B, "
                "the consumer-GPU model used by the imported LatentSlate workflows."
            ),
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                *_dimension_inputs(default_width=512, default_height=512),
                _seed_input(),
            ],
            canvas=KLEIN_CANVAS,
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
            schema_revision=4,
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
                *_dimension_inputs(default_width=512, default_height=512),
                _seed_input(),
            ],
            canvas=KLEIN_CANVAS,
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
            schema_revision=4,
            name="Klein 9B Text to Image",
            description="Generate an image from text with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                *_dimension_inputs(default_width=1024, default_height=1024),
                _seed_input(),
            ],
            canvas=KLEIN_CANVAS,
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
            schema_revision=4,
            name="Klein 9B Image to Image (1-3 refs)",
            description="Edit or compose one to three images with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.IMAGE_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                *_reference_inputs(),
                *_dimension_inputs(default_width=512, default_height=512),
                _seed_input(),
            ],
            canvas=KLEIN_CANVAS,
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
