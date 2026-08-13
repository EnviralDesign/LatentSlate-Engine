from __future__ import annotations

import importlib.util
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
from ..resources import ResourceDescriptor
from ..runtime.diffusers_repository import (
    LTX23_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.ltx23 import (
    LTX23_MAX_DURATION_SECONDS,
    LTX23_MIN_DURATION_SECONDS,
    LTX23ConditionRuntime,
    LTX23Runtime,
    resolve_ltx23_runtime_plan,
)
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, Tool, ToolContext

TEXT_TO_VIDEO_ID = UUID("46bdb57c-3b19-5397-8949-4e20ffe757c9")
FIRST_FRAME_TO_VIDEO_ID = UUID("5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7")
FIRST_LAST_FRAME_TO_VIDEO_ID = UUID("1a8f9c0b-410e-56e4-90de-23bcb9d644ca")


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
        ToolInput(
            key="width",
            label="Width",
            type=InputType.INTEGER,
            role=InputRole.WIDTH,
            required=True,
            default=768,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="height",
            label="Height",
            type=InputType.INTEGER,
            role=InputRole.HEIGHT,
            required=True,
            default=512,
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


class LTX23TextToVideoTool(Tool):
    def model_family(self) -> str:
        return "ltx23"

    def variant_base_availability(self) -> tuple[bool, str | None]:
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
            attention_modes=frozenset({"native"}),
            offload_modes=frozenset({"sequential", "model", "none"}),
            quantization_modes=frozenset({"bf16"}),
            compile_modes=frozenset(),
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            cache_modes=frozenset({"none", "prompt"}),
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

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
    ) -> LTX23Runtime:
        key = ("ltx23", plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: LTX23Runtime(context.settings, plan),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
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

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers",
            "pipeline": "LTX2Pipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }


class LTX23ImageToVideoTool(LTX23TextToVideoTool):
    """LTX 2.3 video with explicitly required first and last frames."""

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
            schema_revision=1,
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
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> LTX23ConditionRuntime:
        # Conditions alter pipeline class and model residency; never reuse the
        # text-to-video wrapper even when both point to the same model folder.
        key = ("ltx23_condition", plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: LTX23ConditionRuntime(context.settings, plan),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start = AssetInput.model_validate(inputs["start_image"])
        end = AssetInput.model_validate(inputs["end_image"])
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

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers",
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
            inputs=[_inputs()[0], ToolInput(
                key="start_image",
                label="First Frame",
                type=InputType.IMAGE,
                role=InputRole.START_IMAGE,
                required=True,
                ui=InputUi(group="Keyframes"),
            ), *_inputs()[1:]],
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start = AssetInput.model_validate(inputs["start_image"])
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
