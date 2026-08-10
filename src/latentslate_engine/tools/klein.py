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
from ..runtime.klein import KLEIN_SIZE_PRESETS, KleinRuntime, KleinVariant
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import Tool, ToolContext

KLEIN4B_TEXT_TO_IMAGE_ID = UUID("077f54e4-14f9-5aaf-973b-5d89d0214214")
KLEIN4B_IMAGE_TO_IMAGE_ID = UUID("6e52c99c-35f3-5eba-ba32-4a800756beed")
KLEIN9B_TEXT_TO_IMAGE_ID = UUID("e329a7d2-c145-4299-96ef-f2b70376d499")
KLEIN9B_IMAGE_TO_IMAGE_ID = UUID("3333a6bd-8e71-4236-9372-bad407161803")


def _runtime_availability(variant: KleinVariant) -> tuple[bool, str | None]:
    modules = ["torch", "diffusers", "transformers", "accelerate", "PIL"]
    if variant == "klein9b":
        modules.extend(("modelopt", "torchao"))
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
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

    def _runtime(self, context: ToolContext) -> KleinRuntime:
        settings = context.settings
        if self.variant == "klein4b":
            key = (
                "flux2_klein4b",
                settings.klein4b_model_id,
                settings.klein4b_profile,
                settings.klein4b_device,
            )
        else:
            key = (
                "flux2_klein9b",
                settings.klein_model_id,
                settings.klein_profile,
                settings.klein_device,
                settings.klein_transformer_model_id,
                settings.klein_transformer_filename,
                settings.klein_text_encoder_model_id,
            )
        return RUNTIME_MANAGER.activate(
            key,
            lambda: KleinRuntime(settings, self.variant),
        )

    def _generate(
        self,
        context: ToolContext,
        inputs: dict[str, Any],
        *,
        source_assets: list[AssetInput],
    ) -> list[StoredArtifact]:
        output_path = context.storage.artifact_path(context.job_id, "output.png")
        metadata = self._runtime(context).generate(
            prompt=str(inputs["prompt"]),
            output_path=output_path,
            size_name=str(inputs["size"]),
            seed=int(inputs["seed"]),
            image_paths=[context.resolve_asset(asset.asset_id) for asset in source_assets],
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

    def _source_assets(self, inputs: dict[str, Any]) -> list[AssetInput]:
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
