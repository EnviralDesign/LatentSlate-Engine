"""Import-light canvas contract shared by Wan 2.2 tools and runtimes."""

from __future__ import annotations

from ..protocol import CanvasContract

WAN_I2V_MAX_PIXELS = 1280 * 704
WAN_I2V_MAX_DIMENSION = 1280
WAN_I2V_CANVAS = CanvasContract(
    alignment=16,
    min_side=64,
    max_side=WAN_I2V_MAX_DIMENSION,
    max_pixels=WAN_I2V_MAX_PIXELS,
)
