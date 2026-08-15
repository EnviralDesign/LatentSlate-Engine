from __future__ import annotations

import importlib.util
import os
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
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.manager import RUNTIME_MANAGER
from ..runtime.wan5_kitchen_managed import ManagedWan5KitchenRuntime
from ..runtime.wan22 import (
    WAN22_MAX_DURATION_SECONDS,
    WAN22_MIN_DURATION_SECONDS,
    Wan22Runtime,
    frames_for_duration,
    resolve_wan22_runtime_plan,
)
from ..runtime.wan22_support import wan22_runtime_support
from ..storage import StoredArtifact
from ..wan5_kitchen_recipe import Wan5KitchenRuntimeRequest
from .base import (
    ExecutionCapabilities,
    ExecutionRequest,
    Tool,
    ToolContext,
)

TEXT_TO_VIDEO_ID = UUID("e2a558e6-533d-5e34-9231-f6388ef2ea20")
IMAGE_TO_VIDEO_ID = UUID("076a3810-36de-55ac-8be2-c2a83a00c389")

_WAN22_OFFLOAD_MODES = frozenset({"sequential", "group_leaf", "model"})
_WAN22_CACHE_MODES = frozenset({"none", "prompt"})


def _runtime_availability() -> tuple[bool, str | None]:
    support = wan22_runtime_support()
    return support.core_available, support.core_reason


def _kitchen_runtime_availability() -> tuple[bool, str | None]:
    """Return parent-process prerequisites without probing CUDA or importing torch.

    Worker startup owns CUDA/kernel proof.  Catalog construction is deliberately
    light so it can advertise an accepted recipe without paying a driver/runtime
    initialization cost or retaining an accelerator context in the API parent.
    """

    if os.name != "nt":
        return False, "Engine-native Wan 5B Kitchen execution requires Windows Job Objects"
    available, reason = _runtime_availability()
    if not available:
        return available, reason
    if importlib.util.find_spec("comfy_kitchen") is None:
        return False, "Install the direct Kitchen runtime dependency"
    return True, None


def _kitchen_request(context: ToolContext) -> object | None:
    execution = context.execution
    return None if execution is None else execution.recipe


def _inputs() -> list[ToolInput]:
    return [
        ToolInput(
            key="prompt",
            label="Prompt",
            type=InputType.TEXT,
            role=InputRole.PROMPT,
            required=True,
            ui=InputUi(
                group="Prompt",
                multiline=True,
                placeholder="Describe the shot, motion, camera, and visual style.",
            ),
        ),
        ToolInput(
            key="width",
            label="Width",
            type=InputType.INTEGER,
            role=InputRole.WIDTH,
            required=True,
            default=1280,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="height",
            label="Height",
            type=InputType.INTEGER,
            role=InputRole.HEIGHT,
            required=True,
            default=704,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="duration_seconds",
            label="Duration",
            type=InputType.NUMBER,
            role=InputRole.DURATION_SECONDS,
            required=True,
            default=5.0,
            ui=InputUi(
                group="Output",
                min=WAN22_MIN_DURATION_SECONDS,
                max=WAN22_MAX_DURATION_SECONDS,
                step=0.5,
                unit="seconds",
            ),
        ),
        ToolInput(
            key="seed",
            label="Seed",
            type=InputType.INTEGER,
            role=InputRole.SEED,
            required=True,
            default=0,
            ui=InputUi(group="Advanced", advanced=True, min=0, step=1),
        ),
    ]


class Wan22TextToVideoTool(Tool):
    _kitchen_operation = "wan5_t2v"

    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def variant_recipe_availability(self, recipe_type: str | None) -> tuple[bool, str | None]:
        if recipe_type == "wan5_kitchen":
            return _kitchen_runtime_availability()
        return _runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        support = wan22_runtime_support()
        if not support.core_available:
            return ExecutionCapabilities()

        return ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset(),
            recipe_types=frozenset({"wan5_kitchen"}),
            attention_modes=frozenset({"native"}),
            offload_modes=_WAN22_OFFLOAD_MODES | {"staged"},
            quantization_modes=frozenset({"bf16", "fp8"}),
            compile_modes=frozenset(),
            compile_fullgraph=False,
            compile_dynamic=False,
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            cache_modes=_WAN22_CACHE_MODES,
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        optimizations = request.optimizations
        if request.recipe_type == "wan5_kitchen":
            expected = {
                "attention": "native",
                "offload": "staged",
                "quantization": "fp8",
                "cache": "none",
            }
            for key, value in expected.items():
                if optimizations.get(key) != value:
                    errors.append(f"Wan 5B stored recipes require {key}={value}")
            if request.model_override or request.loras:
                errors.append("Wan 5B stored recipes own their fixed component closure")
            return errors
        quantization = str(optimizations.get("quantization", "inherit"))
        offload = str(optimizations.get("offload", "inherit"))
        attention = str(optimizations.get("attention", "inherit"))
        use_stream = bool(optimizations.get("group_offload_use_stream", False))
        record_stream = bool(optimizations.get("group_offload_record_stream", False))
        if (quantization == "inherit") != (offload == "inherit"):
            errors.append(
                "Wan 2.2 quantization and offload must either both inherit the "
                "configured profile or both be explicit"
            )
        if attention not in {"inherit", "native"}:
            errors.append("Wan 2.2 recovery variants currently support only native attention")
        if use_stream or record_stream:
            errors.append(
                "Wan 2.2 recovery variants disable group-offload streams to preserve "
                "the lowest predictable VRAM peak"
            )
        if offload == "model" and quantization in {"inherit", "bf16"}:
            errors.append(
                "Wan 2.2 BF16 model offload is not advertised as a 16 GB recovery "
                "path; use sequential or group_leaf"
            )
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="wan22.text_to_video",
            schema_revision=2,
            name="Text to Video",
            description=("Generate video with the dense Wan 2.2 TI2V-5B model in text-only mode."),
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            requirements=[ToolRequirement(bundle_id="wan22-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_wan22_runtime_plan(context.settings, context.execution)

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> Wan22Runtime:
        key = ("wan22_ti2v_5b", plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: Wan22Runtime(context.settings, plan),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        if isinstance(_kitchen_request(context), Wan5KitchenRuntimeRequest):
            return self._run_kitchen(context, inputs)
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        runtime = self._runtime(context, plan)
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "staged_text_encoder": metadata["staged_text_encoder"],
                    "prompt_stage": metadata["prompt_stage"],
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
                content_type="video/mp4",
                path=output_path,
                role="primary",
                media_type="video",
                metadata=metadata,
            )
        ]

    def _run_kitchen(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        start_image_path: Path | None = None,
    ) -> list[StoredArtifact]:
        request = _kitchen_request(context)
        if not isinstance(request, Wan5KitchenRuntimeRequest):
            raise TypeError("Wan 5B stored execution requires a typed runtime request")
        if request.operation != self._kitchen_operation:
            raise ValueError(
                f"tool {self.descriptor.key!r} requires Wan 5B operation "
                f"{self._kitchen_operation!r}"
            )
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        key = ("wan5_kitchen", request.operation, request.fingerprint)
        runtime = RUNTIME_MANAGER.activate(key, lambda: ManagedWan5KitchenRuntime(request))
        context.record_provenance(
            runtime_plan={
                "runtime": "engine-native/wan5-kitchen-disposable-worker",
                "operation": request.operation,
                "request_fingerprint": request.fingerprint,
                "component_fingerprint": request.component_fingerprint,
                "components": request.public_component_manifest(),
            }
        )
        result = runtime.generate(
            prompt=str(inputs["prompt"]),
            output_path=output_path,
            width=int(inputs["width"]),
            height=int(inputs["height"]),
            num_frames=frames_for_duration(float(inputs["duration_seconds"])),
            seed=int(inputs["seed"]),
            start_image_path=start_image_path,
            progress=context.progress,
            check_cancelled=context.check_cancelled,
        )
        status = runtime.status()
        context.record_provenance(
            runtime_result={
                **result.metadata,
                "worker": status["last_worker"],
                "cleanup_errors": status["cleanup_errors"],
            }
        )
        return [
            StoredArtifact(
                id=uuid4(),
                filename=result.output_path.name,
                content_type="video/mp4",
                path=result.output_path,
                role="primary",
                media_type="video",
                metadata=result.metadata,
            )
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers",
            "pipeline": "WanPipeline",
            "model_family": "wan_2_2_ti2v_5b",
            "mode": "text_to_video",
            "prompt_stage": "isolated_cpu_subprocess",
        }

    def variant_provenance(self, recipe_type: str | None) -> dict[str, Any]:
        if recipe_type == "wan5_kitchen":
            return {
                "runtime": "engine-native/wan5-kitchen-disposable-worker",
                "pipeline": (
                    "WanPipeline"
                    if self._kitchen_operation == "wan5_t2v"
                    else "WanImageToVideoPipeline"
                ),
                "model_family": "wan_2_2_ti2v_5b",
                "artifact_contract": "typed_stored_components_direct_kitchen",
                "cache": "none",
            }
        return self.provenance()

    def variant_requirements(self, recipe_type: str | None):
        if recipe_type == "wan5_kitchen":
            return []
        return super().variant_requirements(recipe_type)


class Wan22ImageToVideoTool(Wan22TextToVideoTool):
    _kitchen_operation = "wan5_i2v"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=IMAGE_TO_VIDEO_ID,
            key="wan22.image_to_video",
            schema_revision=1,
            name="Image to Video",
            description="Generate Wan 2.2 TI2V 5B video from a required first frame.",
            workflow_kind=WorkflowKind.IMAGE_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[
                _inputs()[0],
                ToolInput(
                    key="start_image",
                    label="First Frame",
                    type=InputType.IMAGE,
                    role=InputRole.START_IMAGE,
                    required=True,
                    ui=InputUi(group="Keyframes"),
                ),
                *_inputs()[1:],
            ],
            requirements=[],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def variant_recipe_availability(self, recipe_type: str | None) -> tuple[bool, str | None]:
        if recipe_type == "wan5_kitchen":
            return _kitchen_runtime_availability()
        return False, "Wan 2.2 5B I2V is available only through the stored Engine recipe"

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        request = _kitchen_request(context)
        if not isinstance(request, Wan5KitchenRuntimeRequest):
            raise TypeError("Wan 2.2 5B I2V requires its typed stored recipe")
        start = AssetInput.model_validate(inputs["start_image"])
        return self._run_kitchen(
            context,
            inputs,
            start_image_path=context.resolve_asset(start.asset_id),
        )
