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
    H3_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from ..runtime.h3 import (
    H3_CANVAS,
    H3_DEFAULT_HEIGHT,
    H3_DEFAULT_STEPS,
    H3_DEFAULT_WIDTH,
    H3_FIRST_LAST_WORKFLOW,
    H3_MAX_DURATION_SECONDS,
    H3_MAX_STEPS,
    H3_MIN_STEPS,
    H3_TEXT_WORKFLOW,
    H3Runtime,
    resolve_h3_dimensions,
    resolve_h3_runtime_plan,
    validate_h3_steps,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext
from .canvas import dimension_tool_inputs

TEXT_TO_VIDEO_ID = UUID("369a630e-4d64-4e3c-8f15-1809757a10e5")
FIRST_LAST_VIDEO_ID = UUID("8c038628-e5bd-4954-80e3-32956321089b")


def _runtime_availability() -> tuple[bool, str | None]:
    missing = [
        module
        for module in ("torch", "diffusers", "transformers", "av")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Install the h3 extra; missing: {', '.join(missing)}"
    return True, None


def _common_inputs() -> list[ToolInput]:
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
                placeholder=(
                    "Describe the video to generate, including motion, camera, and sound."
                ),
            ),
        ),
        *dimension_tool_inputs(
            H3_CANVAS,
            default_width=H3_DEFAULT_WIDTH,
            default_height=H3_DEFAULT_HEIGHT,
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
                min=5.0,
                max=H3_MAX_DURATION_SECONDS,
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
        ToolInput(
            key="steps",
            label="Steps",
            type=InputType.INTEGER,
            required=True,
            default=H3_DEFAULT_STEPS,
            ui=InputUi(
                group="Advanced",
                advanced=True,
                min=H3_MIN_STEPS,
                max=H3_MAX_STEPS,
                step=1,
            ),
        ),
    ]


class _H3Base(Tool):
    workflow: str

    def model_family(self) -> str:
        return "h3"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        if not _runtime_availability()[0]:
            return ExecutionCapabilities()
        return ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            attention_modes=frozenset({"native"}),
            quantization_modes=frozenset({"bf16"}),
            residency_policy=True,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        quantization = str(request.optimizations.get("quantization", "inherit"))
        attention = str(request.optimizations.get("attention", "inherit"))
        if quantization not in {"inherit", "bf16"}:
            errors.append(
                "H3 supports only pre-existing BF16 artifacts; quantized loaders "
                "are not implemented"
            )
        if attention not in {"inherit", "native"}:
            errors.append("H3 supports native attention only")
        return errors

    def validate_model_resource(
        self,
        _resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        try:
            validate_diffusers_repository(path, H3_REPOSITORY_CONTRACT)
        except (OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_h3_runtime_plan(
            context.settings,
            context.execution,
            workflow=self.workflow,
        )

    def _runtime(self, context: ToolContext, plan: ResolvedRuntimePlan) -> H3Runtime:
        key = (
            "minimax_h3",
            plan.pipeline_fingerprint,
        )
        return RUNTIME_MANAGER.activate(
            key,
            lambda: H3Runtime(context.settings, plan),
        )

    def _generate(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        image_asset: AssetInput | None,
        last_image_asset: AssetInput | None,
    ) -> list[StoredArtifact]:
        # Reject invalid public geometry and step budgets before a repository is
        # resolved or the persistent pipeline can be activated.
        resolve_h3_dimensions(inputs.get("width"), inputs.get("height"))
        validate_h3_steps(inputs.get("steps"))
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=inputs["width"],
                height=inputs["height"],
                steps=inputs["steps"],
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                image_path=(context.resolve_asset(image_asset.asset_id) if image_asset else None),
                last_image_path=(
                    context.resolve_asset(last_image_asset.asset_id) if last_image_asset else None
                ),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
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

    def provenance(self) -> dict[str, Any]:
        return {"runtime": "diffusers_modular", "workflow": self.workflow}


class H3TextToVideoTool(_H3Base):
    workflow = H3_TEXT_WORKFLOW

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="h3.text_to_video",
            schema_revision=3,
            name="Text to Video",
            description=(
                "Generate a short MiniMax-H3 video with synchronized stereo audio from text."
            ),
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_common_inputs(),
            canvas=H3_CANVAS,
            requirements=[ToolRequirement(bundle_id="h3-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(context, inputs, image_asset=None, last_image_asset=None)


class H3FirstLastFrameTool(_H3Base):
    workflow = H3_FIRST_LAST_WORKFLOW

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        media_inputs = [
            ToolInput(
                key="start_image",
                label="First Frame",
                type=InputType.IMAGE,
                role=InputRole.START_IMAGE,
                required=False,
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
            id=FIRST_LAST_VIDEO_ID,
            key="h3.first_last_frame_video",
            schema_revision=3,
            name="First/Last Frame Video",
            description=(
                "Generate a short MiniMax-H3 video with synchronized audio from a first "
                "frame, a last frame, or both."
            ),
            workflow_kind=WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[_common_inputs()[0], *media_inputs, *_common_inputs()[1:]],
            canvas=H3_CANVAS,
            requirements=[ToolRequirement(bundle_id="h3-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start_value = inputs.get("start_image")
        start = AssetInput.model_validate(start_value) if start_value else None
        end_value = inputs.get("end_image")
        end = AssetInput.model_validate(end_value) if end_value else None
        if start is None and end is None:
            raise ValueError("H3 first/last-frame generation requires a first or last frame")
        return self._generate(context, inputs, image_asset=start, last_image_asset=end)
