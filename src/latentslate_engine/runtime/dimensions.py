"""Fail-closed spatial input validation for curated runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol import CanvasContract


@dataclass(frozen=True, slots=True)
class Dimensions:
    """The client request and the canvas the runtime will actually receive.

    After the catalog contract, these two sizes are always identical: the
    engine never rewrites a requested canvas onto a nearby legal grid.
    """

    requested_width: int
    requested_height: int
    width: int
    height: int

    def metadata(self) -> dict[str, dict[str, int]]:
        return {
            "requested_dimensions": {
                "width": self.requested_width,
                "height": self.requested_height,
            },
            "effective_dimensions": {"width": self.width, "height": self.height},
        }


def require_dimensions(width: int, height: int, canvas: CanvasContract) -> Dimensions:
    """Accept a canvas only when it already satisfies the public contract."""

    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < canvas.min_side:
            raise ValueError(f"{name} must be at least {canvas.min_side}")
        if canvas.max_side is not None and value > canvas.max_side:
            raise ValueError(f"{name} must be at most {canvas.max_side}")
        if value % canvas.alignment:
            raise ValueError(
                f"width and height must be divisible by {canvas.alignment} pixels; "
                f"received {width}x{height}"
            )
    if width * height > (canvas.max_pixels or width * height):
        raise ValueError(
            "dimensions exceed this model family's pixel budget "
            f"({width}x{height} > {canvas.max_pixels} pixels)"
        )
    if canvas.max_aspect is not None and (
        width > height * canvas.max_aspect or height > width * canvas.max_aspect
    ):
        raise ValueError(
            "dimensions must stay within a "
            f"1:{canvas.max_aspect:g} to {canvas.max_aspect:g}:1 aspect ratio"
        )
    return Dimensions(width, height, width, height)
