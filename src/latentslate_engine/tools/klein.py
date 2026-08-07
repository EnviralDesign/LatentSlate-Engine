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
from ..runtime.klein import KLEIN_SIZE_PRESETS, KleinRuntime
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import Tool, ToolContext


TEXT_TO_IMAGE_ID = UUID("e329a7d2-c145-4299-96ef-f2b70376d499")
IMAGE_TO_IMAGE_ID = UUID("3333a6bd-8e71-4236-9372-bad407161803")


def _runtime_availability() -> tuple[bool, str | None]:
    missing = [
        module
        for module in ("torch", "diffusers", "transformers", "accelerate", "PIL")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Install the klein extra; missing: {', '.join(missing)}"
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


def _size_input(*, include_source: bool) -> ToolInput:
    values = list(KLEIN_SIZE_PRESETS)
    if not include_source:
        values.remove("source")
    default = "source" if include_source else "1024x1024"
    return ToolInput(
        key="size",
        label="Size",
        type=InputType.CHOICE,
        required=True,
        default=default,
        options=[ChoiceOption(value=value, label=value) for value in values],
        ui=InputUi(group="Output"),
    )


class _KleinBase(Tool):
    def _runtime(self, context: ToolContext) -> KleinRuntime:
        settings = context.settings
        key = (
            "flux2_klein9b",
            settings.klein_model_id,
            settings.klein_profile,
            settings.klein_device,
            settings.klein_transformer_model_id,
            settings.klein_transformer_filename,
            settings.klein_text_encoder_model_id,
        )
        return RUNTIME_MANAGER.activate(key, lambda: KleinRuntime(settings))

    def _generate(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        source_asset: AssetInput | None,
    ) -> list[StoredArtifact]:
        output_path = context.storage.artifact_path(context.job_id, "output.png")
        metadata = self._runtime(context).generate(
            prompt=str(inputs["prompt"]),
            output_path=output_path,
            size_name=str(inputs["size"]),
            seed=int(inputs["seed"]),
            image_path=context.resolve_asset(source_asset.asset_id) if source_asset else None,
            progress=context.progress,
            check_cancelled=context.check_cancelled,
        )
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

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers",
            "pipeline": "Flux2KleinPipeline",
            "model_family": "flux2_klein_9b",
        }


class KleinTextToImageTool(_KleinBase):
    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_IMAGE_ID,
            key="flux2_klein9b.text_to_image",
            schema_revision=1,
            name="Text to Image",
            description="Generate an image from text with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[_prompt_input(), _size_input(include_source=False), _seed_input()],
            requirements=[ToolRequirement(bundle_id="klein9b-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._generate(context, inputs, source_asset=None)


class KleinImageToImageTool(_KleinBase):
    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=IMAGE_TO_IMAGE_ID,
            key="flux2_klein9b.image_to_image",
            schema_revision=1,
            name="Image to Image",
            description="Edit or transform an image with FLUX.2 Klein 9B.",
            workflow_kind=WorkflowKind.IMAGE_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                _prompt_input(),
                ToolInput(
                    key="source_image",
                    label="Source Image",
                    type=InputType.IMAGE,
                    role=InputRole.SOURCE_IMAGE,
                    required=True,
                    ui=InputUi(group="Input"),
                ),
                _size_input(include_source=True),
                _seed_input(),
            ],
            requirements=[ToolRequirement(bundle_id="klein9b-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        source = AssetInput.model_validate(inputs["source_image"])
        return self._generate(context, inputs, source_asset=source)
