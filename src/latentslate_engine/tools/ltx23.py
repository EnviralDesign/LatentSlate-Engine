from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..ltx23_kitchen_recipe import LTX23KitchenRuntimeRequest
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
from ..resources import ResourceDescriptor
from ..runtime.diffusers_repository import (
    LTX23_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.ltx23 import (
    LTX23_CANVAS,
    LTX23_MAX_DURATION_SECONDS,
    LTX23_MIN_DURATION_SECONDS,
    resolve_ltx23_runtime_plan,
)
from ..runtime.ltx23_kitchen_managed import ManagedLTX23KitchenRuntime
from ..runtime.ltx23_managed import ManagedLTX23Runtime
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext
from .canvas import dimension_tool_inputs

TEXT_TO_VIDEO_ID = UUID("46bdb57c-3b19-5397-8949-4e20ffe757c9")
FIRST_FRAME_TO_VIDEO_ID = UUID("5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7")
FIRST_LAST_FRAME_TO_VIDEO_ID = UUID("1a8f9c0b-410e-56e4-90de-23bcb9d644ca")
LTX23_ENGINE_DEFAULT_WIDTH = 768
LTX23_ENGINE_DEFAULT_HEIGHT = 512
LTX23_PINNED_WORKFLOW_DEFAULT_WIDTH = 1280
LTX23_PINNED_WORKFLOW_DEFAULT_HEIGHT = 720


def _runtime_availability() -> tuple[bool, str | None]:
    missing = [
        module
        for module in (
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "sentencepiece",
            "av",
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Install the ltx23 extra; missing: {', '.join(missing)}"
    return True, None


def _kitchen_runtime_availability() -> tuple[bool, str | None]:
    if os.name != "nt":
        return False, "Engine-native LTX 2.3 Kitchen execution requires Windows Job Objects"
    missing = [
        module
        for module in (
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "sentencepiece",
            "av",
            "comfy_kitchen",
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return (
            False,
            f"Install the LTX 2.3 Kitchen runtime dependencies; missing: {', '.join(missing)}",
        )
    return True, None


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
                placeholder="Describe the shot, motion, camera, and sound.",
            ),
        ),
        *dimension_tool_inputs(
            LTX23_CANVAS,
            default_width=LTX23_ENGINE_DEFAULT_WIDTH,
            default_height=LTX23_ENGINE_DEFAULT_HEIGHT,
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
                min=LTX23_MIN_DURATION_SECONDS,
                max=LTX23_MAX_DURATION_SECONDS,
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


def _kitchen_request(context: ToolContext) -> object | None:
    execution = context.execution
    return None if execution is None else execution.recipe


class LTX23TextToVideoTool(Tool):
    _worker_operation = "t2v"
    _kitchen_operation = "ltx23_dev_t2v"

    def model_family(self) -> str:
        return "ltx23"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def variant_recipe_availability(
        self,
        recipe_type: str | None,
    ) -> tuple[bool, str | None]:
        if recipe_type == "ltx23_kitchen":
            return _kitchen_runtime_availability()
        return _runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _reason = _runtime_availability()
        if not available:
            return ExecutionCapabilities()
        return ExecutionCapabilities(
            # LTX2Pipeline.from_pretrained consumes a complete repository.  A
            # standalone SafeTensors or GGUF resource is not a substitutable
            # pipeline, even when its name contains LTX.
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset(),
            recipe_types=frozenset({"ltx23_kitchen"}),
            attention_modes=frozenset({"native"}),
            offload_modes=frozenset({"sequential", "model", "staged", "none"}),
            quantization_modes=frozenset({"bf16", "fp8"}),
            compile_modes=frozenset(),
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            # The Engine-native Kitchen worker retains only exact
            # request-bound components; it has no cross-job prompt/media cache.
            cache_modes=frozenset({"none"}),
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        optimizations = request.optimizations
        if request.recipe_type == "ltx23_kitchen":
            expected = {
                "attention": "native",
                "offload": "staged",
                "quantization": "fp8",
                "cache": "none",
            }
            for key, value in expected.items():
                if optimizations.get(key) != value:
                    errors.append(f"LTX 2.3 Kitchen recipes require {key}={value}")
            if request.model_override or request.loras:
                errors.append("LTX 2.3 Kitchen recipes own their complete fixed component closure")
        elif str(optimizations.get("quantization", "inherit")) not in {"inherit", "bf16"}:
            errors.append("LTX 2.3 Reference execution accepts BF16 only")
        return errors

    def validate_model_resource(
        self,
        _resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        try:
            validate_diffusers_repository(path, LTX23_REPOSITORY_CONTRACT)
        except (OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="ltx23.text_to_video",
            schema_revision=2,
            name="Text to Video",
            description="Generate synchronized video and audio with LTX 2.3.",
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            canvas=LTX23_CANVAS,
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_ltx23_runtime_plan(context.settings, context.execution)

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> ManagedLTX23Runtime:
        key = ("ltx23", self._worker_operation, plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedLTX23Runtime(context.settings, plan, operation=self._worker_operation),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        if isinstance(_kitchen_request(context), LTX23KitchenRuntimeRequest):
            return self._run_kitchen(context, inputs)
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
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
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            # The supervisor retains no model tensors in the Engine process.
            pass
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
        end_image_path: Path | None = None,
    ) -> list[StoredArtifact]:
        request = _kitchen_request(context)
        if not isinstance(request, LTX23KitchenRuntimeRequest):
            raise TypeError("LTX 2.3 Kitchen execution requires a typed runtime request")
        if request.operation != self._kitchen_operation:
            raise ValueError(
                f"tool {self.descriptor.key!r} requires LTX operation {self._kitchen_operation!r}"
            )
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        key = ("ltx23_kitchen", request.operation, request.fingerprint)
        runtime = RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedLTX23KitchenRuntime(request),
        )
        context.record_provenance(
            runtime_plan={
                "runtime": "engine-native/ltx23-kitchen-persistent-worker",
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
            duration_seconds=float(inputs["duration_seconds"]),
            seed=int(inputs["seed"]),
            start_image_path=start_image_path,
            end_image_path=end_image_path,
            progress=context.progress,
            check_cancelled=context.check_cancelled,
        )
        status = runtime.status()
        context.record_provenance(
            runtime_result={
                **result.metadata,
                "worker": status["last_worker"],
                "pipeline_warm": result.metadata.get("cache", {}).get("pipeline_warm", False),
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
            "runtime": "diffusers_disposable_worker",
            "pipeline": "LTX2Pipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }

    def variant_provenance(self, recipe_type: str | None) -> dict[str, Any]:
        if recipe_type == "ltx23_kitchen":
            return {
                "runtime": "engine-native/ltx23-kitchen-persistent-worker",
                "model_family": "ltx_2_3",
                "artifact_contract": "typed_stored_components_direct_kitchen",
                "cache": "persistent-components",
                "engine_default_dimensions": [
                    LTX23_ENGINE_DEFAULT_WIDTH,
                    LTX23_ENGINE_DEFAULT_HEIGHT,
                ],
                "pinned_workflow_default_dimensions": [
                    LTX23_PINNED_WORKFLOW_DEFAULT_WIDTH,
                    LTX23_PINNED_WORKFLOW_DEFAULT_HEIGHT,
                ],
                "dimension_default_deviation": (
                    "intentional 16GB acceptance preset; pinned 1280x720 is retained as "
                    "source evidence but is not divisible by the two-stage /64 runtime grid"
                ),
            }
        return self.provenance()

    def variant_requirements(self, recipe_type: str | None):
        if recipe_type == "ltx23_kitchen":
            return []
        return super().variant_requirements(recipe_type)


class LTX23ImageToVideoTool(LTX23TextToVideoTool):
    """LTX 2.3 video with explicitly required first and last frames."""

    _worker_operation = "first_last"
    _kitchen_operation = "ltx23_distilled_flf"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        media_inputs = [
            ToolInput(
                key="start_image",
                label="First Frame",
                type=InputType.IMAGE,
                role=InputRole.START_IMAGE,
                required=True,
                ui=InputUi(group="Keyframes"),
            ),
            ToolInput(
                key="end_image",
                label="Last Frame",
                type=InputType.IMAGE,
                role=InputRole.END_IMAGE,
                required=False,
                ui=InputUi(group="Keyframes"),
            ),
        ]
        return ToolDescriptor(
            id=FIRST_LAST_FRAME_TO_VIDEO_ID,
            key="ltx23.first_last_frame_to_video",
            schema_revision=2,
            name="First and Last Frame to Video",
            description=(
                "Generate synchronized video and audio with LTX 2.3 from required first "
                "and last-frame anchors."
            ),
            workflow_kind=WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[
                _inputs()[0],
                media_inputs[0],
                media_inputs[1].model_copy(update={"required": True}),
                *_inputs()[1:],
            ],
            canvas=LTX23_CANVAS,
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> ManagedLTX23Runtime:
        # Conditions alter pipeline class and operation binding; never reuse the
        # text-to-video supervisor even when both point to the same model folder.
        key = ("ltx23_condition", self._worker_operation, plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedLTX23Runtime(context.settings, plan, operation=self._worker_operation),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start = AssetInput.model_validate(inputs["start_image"])
        end = AssetInput.model_validate(inputs["end_image"])
        if isinstance(_kitchen_request(context), LTX23KitchenRuntimeRequest):
            return self._run_kitchen(
                context,
                inputs,
                start_image_path=context.resolve_asset(start.asset_id),
                end_image_path=context.resolve_asset(end.asset_id),
            )
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                start_image_path=context.resolve_asset(start.asset_id),
                end_image_path=context.resolve_asset(end.asset_id),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "conditioning": metadata["conditioning"],
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            pass
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

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers_disposable_worker",
            "pipeline": "LTX2ConditionPipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }


class LTX23FirstFrameToVideoTool(LTX23ImageToVideoTool):
    """LTX 2.3 public first-frame-only operation.

    This is intentionally not an optional-endpoint schema.  A first+last
    request is a separate tool and recipe so authored callers cannot silently
    select a different conditioning topology by omitting a field.
    """

    _worker_operation = "first_frame"
    _kitchen_operation = "ltx23_dev_i2v"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=FIRST_FRAME_TO_VIDEO_ID,
            key="ltx23.image_to_video",
            schema_revision=2,
            name="First Frame to Video",
            description="Generate synchronized video and audio with LTX 2.3 from a first frame.",
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
            canvas=LTX23_CANVAS,
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start = AssetInput.model_validate(inputs["start_image"])
        if isinstance(_kitchen_request(context), LTX23KitchenRuntimeRequest):
            return self._run_kitchen(
                context,
                inputs,
                start_image_path=context.resolve_asset(start.asset_id),
            )
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                start_image_path=context.resolve_asset(start.asset_id),
                end_image_path=None,
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "conditioning": metadata["conditioning"],
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            pass
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
