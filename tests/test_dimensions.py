from uuid import UUID

import pytest
from pydantic import ValidationError

from latentslate_engine.protocol import (
    CanvasContract,
    InputRole,
    MediaType,
    ToolDescriptor,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.runtime.dimensions import require_dimensions
from latentslate_engine.tools import variant_base_tools
from latentslate_engine.tools.canvas import dimension_tool_inputs


def test_require_dimensions_accepts_an_exact_legal_canvas() -> None:
    canvas = CanvasContract(alignment=64, min_side=64, max_pixels=942_080)
    dimensions = require_dimensions(1024, 576, canvas)
    assert (dimensions.width, dimensions.height) == (1024, 576)
    assert dimensions.metadata()["requested_dimensions"] == {"width": 1024, "height": 576}
    assert dimensions.metadata()["effective_dimensions"] == {"width": 1024, "height": 576}


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (960, 540, "divisible by 64"),
        (32, 64, "at least 64"),
        (1280, 768, "pixel budget"),
    ],
)
def test_require_dimensions_never_rewrites_an_illegal_canvas(width, height, message) -> None:
    canvas = CanvasContract(alignment=64, min_side=64, max_pixels=942_080)
    with pytest.raises(ValueError, match=message):
        require_dimensions(width, height, canvas)


def test_catalog_dimension_tools_publish_a_matching_canvas() -> None:
    for tool in variant_base_tools():
        descriptor = tool.descriptor
        roles = {item.role for item in descriptor.inputs}
        if InputRole.WIDTH not in roles and InputRole.HEIGHT not in roles:
            assert descriptor.canvas is None
            continue
        assert InputRole.WIDTH in roles and InputRole.HEIGHT in roles
        assert descriptor.canvas is not None
        width = next(item for item in descriptor.inputs if item.role == InputRole.WIDTH)
        height = next(item for item in descriptor.inputs if item.role == InputRole.HEIGHT)
        assert width.ui is not None and height.ui is not None
        assert int(width.ui.step) == descriptor.canvas.alignment
        assert int(width.ui.min) == descriptor.canvas.min_side
        assert int(height.ui.step) == descriptor.canvas.alignment
        hashed = descriptor.with_schema_hash()
        assert hashed.schema_hash == descriptor.schema_hash


def test_dimension_tool_inputs_reject_a_mismatched_default_pair() -> None:
    canvas = CanvasContract(alignment=16, min_side=64)
    with pytest.raises(ValueError, match="supplied as a pair"):
        dimension_tool_inputs(canvas, default_width=512, default_height=None)


def test_canvas_min_side_must_sit_on_the_alignment_grid() -> None:
    with pytest.raises(ValidationError, match="min_side must be divisible"):
        CanvasContract(alignment=64, min_side=96)


def test_tool_without_dimension_roles_may_omit_canvas() -> None:
    ToolDescriptor(
        id=UUID("f51cb180-7570-428a-a173-4e0b060437ef"),
        key="test.no_canvas",
        schema_revision=1,
        name="No canvas",
        workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
        output=ToolOutput(type=MediaType.IMAGE),
        inputs=[],
    )
