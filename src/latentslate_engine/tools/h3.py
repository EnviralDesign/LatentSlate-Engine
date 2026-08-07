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
from ..runtime.h3 import H3_MAX_DURATION_SECONDS, H3Runtime, PRESETS
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import Tool, ToolContext


TEXT_TO_VIDEO_ID = UUID("369a630e-4d64-4e3c-8f15-1809757a10e5")
FIRST_LAST_VIDEO_ID = UUID("8c038628-e5bd-4954-80e3-32956321089b")


def _runtime_availability() -> tuple[bool, str | None]:
    missing = [
        module
        for module in ("torch", "diffusers", "transformers", "torchao", "av")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Install the h3 extra; missing: {', '.join(missing)}"
    return True, None


def _quality_input() -> ToolInput:
    return ToolInput(
        key="quality",
        label="Quality",
        type=InputType.CHOICE,
        required=True,
        default="draft",
        options=[
            ChoiceOption(value="draft", label="Draft", description="832×480, 16 steps"),
            ChoiceOption(
                value="balanced",
                label="Balanced",
                description="960×544, 20 steps",
            ),
            ChoiceOption(value="final", label="Final", description="960×544, 30 steps"),
        ],
        ui=InputUi(group="Output"),
    )


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
        _quality_input(),
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
    ]


class _H3Base(Tool):
    def _runtime(self, context: ToolContext) -> H3Runtime:
        settings = context.settings
        key = (
            "minimax_h3",
            settings.h3_model_id,
            settings.h3_profile,
            settings.h3_device,
        )
        return RUNTIME_MANAGER.activate(key, lambda: H3Runtime(settings))

    def _generate(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        image_asset: AssetInput | None,
        last_image_asset: AssetInput | None,
    ) -> list[StoredArtifact]:
        quality = str(inputs["quality"])
        if quality not in PRESETS:
            raise ValueError(f"Unknown quality preset {quality!r}")
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        metadata = self._runtime(context).generate(
            prompt=str(inputs["prompt"]),
            output_path=output_path,
            preset_name=quality,
            duration_seconds=float(inputs["duration_seconds"]),
            seed=int(inputs["seed"]),
            image_path=context.resolve_asset(image_asset.asset_id) if image_asset else None,
            last_image_path=(
                context.resolve_asset(last_image_asset.asset_id) if last_image_asset else None
            ),
            progress=context.progress,
            check_cancelled=context.check_cancelled,
        )
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
        return {"runtime": "diffusers_modular", "workflow": "fl2va"}


class H3TextToVideoTool(_H3Base):
    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="h3.text_to_video",
            schema_revision=1,
            name="Text to Video",
            description=(
                "Generate a short MiniMax-H3 video with synchronized stereo audio "
                "from text."
            ),
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_common_inputs(),
            requirements=[ToolRequirement(bundle_id="h3-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(context, inputs, image_asset=None, last_image_asset=None)


class H3FirstLastFrameTool(_H3Base):
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
            id=FIRST_LAST_VIDEO_ID,
            key="h3.first_last_frame_video",
            schema_revision=1,
            name="First/Last Frame Video",
            description=(
                "Generate a short MiniMax-H3 video with synchronized audio from a first "
                "frame and an optional last frame."
            ),
            workflow_kind=WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[_common_inputs()[0], *media_inputs, *_common_inputs()[1:]],
            requirements=[ToolRequirement(bundle_id="h3-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        start = AssetInput.model_validate(inputs["start_image"])
        end_value = inputs.get("end_image")
        end = AssetInput.model_validate(end_value) if end_value else None
        return self._generate(context, inputs, image_asset=start, last_image_asset=end)
