"""Catalog helpers for exact width/height canvas contracts."""

from __future__ import annotations

from ..protocol import CanvasContract, InputRole, InputType, InputUi, ToolInput


def dimension_tool_inputs(
    canvas: CanvasContract,
    *,
    default_width: int | None,
    default_height: int | None,
    required: bool = True,
    unit: str = "pixels",
) -> list[ToolInput]:
    """Publish a matching width/height pair for ``canvas``."""

    if (default_width is None) != (default_height is None):
        raise ValueError("canvas dimension defaults must be supplied as a pair")
    if required and (default_width is None or default_height is None):
        raise ValueError("required canvas inputs need defaults")
    ui = InputUi(
        group="Output",
        min=canvas.min_side,
        max=canvas.max_side,
        step=canvas.alignment,
        unit=unit,
    )
    return [
        ToolInput(
            key="width",
            label="Width",
            type=InputType.INTEGER,
            role=InputRole.WIDTH,
            required=required,
            default=default_width,
            ui=ui.model_copy(),
        ),
        ToolInput(
            key="height",
            label="Height",
            type=InputType.INTEGER,
            role=InputRole.HEIGHT,
            required=required,
            default=default_height,
            ui=ui.model_copy(),
        ),
    ]
