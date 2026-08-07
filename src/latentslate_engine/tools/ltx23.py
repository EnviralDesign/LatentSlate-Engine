from __future__ import annotations

import importlib.util
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
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
from ..runtime.ltx23 import (
    LTX23_MAX_DURATION_SECONDS,
    LTX23_MIN_DURATION_SECONDS,
    LTX23_SIZE_PRESETS,
    LTX23Runtime,
)
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import Tool, ToolContext


TEXT_TO_VIDEO_ID = UUID("46bdb57c-3b19-5397-8949-4e20ffe757c9")


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
            key="size",
            label="Size",
            type=InputType.CHOICE,
            required=True,
            default="768x512",
            options=[
                ChoiceOption(value=value, label=value)
                for value in LTX23_SIZE_PRESETS
            ],
            ui=InputUi(group="Output"),
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
    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="ltx23.text_to_video",
            schema_revision=1,
            name="Text to Video",
            description="Generate synchronized video and audio with LTX 2.3.",
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _runtime(self, context: ToolContext) -> LTX23Runtime:
        settings = context.settings
        key = (
            "ltx23",
            settings.ltx23_model_id,
            settings.ltx23_profile,
            settings.ltx23_device,
        )
        return RUNTIME_MANAGER.activate(key, lambda: LTX23Runtime(settings))

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        metadata = self._runtime(context).generate(
            prompt=str(inputs["prompt"]),
            output_path=output_path,
            size_name=str(inputs["size"]),
            duration_seconds=float(inputs["duration_seconds"]),
            seed=int(inputs["seed"]),
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
        return {
            "runtime": "diffusers",
            "pipeline": "LTX2Pipeline",
            "model_family": "ltx_2_3",
        }
