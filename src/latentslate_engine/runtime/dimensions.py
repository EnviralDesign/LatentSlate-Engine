"""Shared, fail-closed spatial input normalization for curated runtimes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dimensions:
    """The client request and the canvas the runtime will actually receive."""

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


def align_dimensions(
    width: int,
    height: int,
    *,
    alignment: int,
    min_side: int,
    max_pixels: int,
) -> Dimensions:
    """Validate a requested canvas and align it to the nearest supported grid.

    The pixel limit is intentionally checked *after* alignment, because a request
    just below a grid boundary can become too large when given to the pipeline.
    """

    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < min_side:
            raise ValueError(f"{name} must be at least {min_side}")
    if alignment <= 0:
        raise ValueError("dimension alignment must be positive")

    effective_width = _nearest_multiple(width, alignment)
    effective_height = _nearest_multiple(height, alignment)
    if effective_width < min_side or effective_height < min_side:
        raise ValueError(f"aligned dimensions must be at least {min_side}")
    if effective_width * effective_height > max_pixels:
        raise ValueError(
            "aligned dimensions exceed this model family's pixel budget "
            f"({effective_width}x{effective_height} > {max_pixels} pixels)"
        )
    return Dimensions(width, height, effective_width, effective_height)


def floor_source_dimensions(
    width: int,
    height: int,
    *,
    alignment: int,
    min_side: int,
    max_pixels: int,
) -> Dimensions:
    """Mirror source I2I preprocessing without claiming the raw source canvas.

    The pinned Diffusers image processor floors visible source dimensions to the
    model grid when no explicit output canvas is supplied.  Metadata therefore
    retains the EXIF-oriented source request separately from the effective canvas.
    """

    for name, value in (("source width", width), ("source height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if alignment <= 0:
        raise ValueError("dimension alignment must be positive")
    effective_width = width // alignment * alignment
    effective_height = height // alignment * alignment
    if effective_width < min_side or effective_height < min_side:
        raise ValueError(
            f"source dimensions floor below the minimum supported side of {min_side}"
        )
    if effective_width * effective_height > max_pixels:
        raise ValueError(
            "effective source dimensions exceed this model family's pixel budget "
            f"({effective_width}x{effective_height} > {max_pixels} pixels)"
        )
    return Dimensions(width, height, effective_width, effective_height)


def _nearest_multiple(value: int, alignment: int) -> int:
    """Round half-way canvases upward rather than relying on banker's rounding."""

    return ((value + alignment // 2) // alignment) * alignment
