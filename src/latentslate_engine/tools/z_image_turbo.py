"""Hidden base tool for the single official Z-Image Turbo T2I recipe."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..protocol import (
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from ..z_image_turbo_recipe import (
    ZImageTurboRuntimeRequest,
    revalidate_z_image_turbo_runtime_request,
)
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext

Z_IMAGE_TURBO_ID = UUID("966f6431-5e34-5cab-9b79-8efed1652fca")
Z_IMAGE_TURBO_KEY = "zimage.turbo_text_to_image"
Z_IMAGE_TURBO_RECIPE_TYPE = "z_image_turbo_t2i"


class ZImageTurboTextToImageTool(Tool):
    """Typed execution seam, intentionally gated until native GPU acceptance.

    Keeping the base in the catalog lets resource/recipe validation exercise the
    same public path.  ``run`` refuses instead of producing a dequantized or
    generic Diffusers result before the actual native graph is qualified.
    """

    def model_family(self) -> str:
        return "zimage"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return False, "CPU/source qualified only: native GPU dispatch has not yet been accepted"

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            recipe_types=frozenset({Z_IMAGE_TURBO_RECIPE_TYPE}), residency_policy=True
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != Z_IMAGE_TURBO_RECIPE_TYPE:
            errors.append("Z-Image Turbo requires the exact official Turbo T2I component recipe")
        if request.model_override:
            errors.append(
                "Z-Image Turbo selects immutable explicit components, not a model override"
            )
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=Z_IMAGE_TURBO_ID,
            key=Z_IMAGE_TURBO_KEY,
            schema_revision=1,
            name="Z-Image Turbo Text to Image",
            description="Workflow-derived Engine INT8 ConvRot T2I contract (GPU acceptance pending).",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(
                    key="prompt",
                    label="Prompt",
                    type=InputType.TEXT,
                    role=InputRole.PROMPT,
                    required=True,
                    ui=InputUi(group="Prompt", multiline=True),
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
            ],
            requirements=[],
            available=False,
            unavailable_reason="CPU/source qualified only: native GPU dispatch has not yet been accepted",
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]):
        context.check_cancelled()
        recipe = context.execution.recipe if context.execution is not None else None
        if not isinstance(
            recipe, ZImageTurboRuntimeRequest
        ) or not revalidate_z_image_turbo_runtime_request(recipe):
            raise ValueError("Z-Image execution requires a revalidated immutable Turbo T2I request")
        raise RuntimeError(
            "Z-Image Turbo native execution is intentionally unavailable until GPU dispatch acceptance; no dense fallback exists"
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "native",
            "pipeline": "ZImageTurboNative",
            "model_family": "z_image_turbo",
            "mode": "text_to_image",
            "conversion": False,
            "fallback": "forbidden",
        }
