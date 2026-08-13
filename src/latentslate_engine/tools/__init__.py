from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import UUID

from ..config import Settings
from ..resources import ResourceInventory, discover_resources
from .base import Tool
from .h3 import H3FirstLastFrameTool, H3TextToVideoTool
from .klein import (
    Klein4BImageToImageTool,
    Klein4BTextToImageTool,
    KleinImageToImageTool,
    KleinTextToImageTool,
)
from .ltx23 import LTX23FirstFrameToVideoTool, LTX23ImageToVideoTool, LTX23TextToVideoTool
from .wan22 import Wan22ImageToVideoTool, Wan22TextToVideoTool
from .wan22_native import NativeWan14BI2VTool
from .wan22_native_flf import NativeWan14BFLFTool
from .wan22_native_t2v import NativeWan14BT2VTool
from .z_image_turbo import ZImageTurboTextToImageTool

if TYPE_CHECKING:
    from ..variants import VariantCatalogEntry


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool],
        *,
        resources: ResourceInventory | None = None,
        variants: list[VariantCatalogEntry] | None = None,
        variant_errors: list[str] | None = None,
    ):
        tool_list = list(tools)
        self._tools = {tool.descriptor.id: tool for tool in tool_list}
        if len(self._tools) != len(tool_list):
            raise ValueError("Tool IDs must be unique")
        keys = [tool.descriptor.key for tool in tool_list]
        if len(keys) != len(set(keys)):
            raise ValueError("Tool keys must be unique")
        self.resources = resources or ResourceInventory()
        self.variants = variants or []
        self.variant_errors = variant_errors or []

    def tools(self) -> list[Tool]:
        """Return registered executable tools without exposing mutable internals."""

        return list(self._tools.values())

    def descriptors(self):
        return [tool.descriptor for tool in self._tools.values()]

    def get(self, tool_id: UUID) -> Tool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {tool_id}") from exc


def variant_base_tools() -> list[Tool]:
    """Return the curated runtime adapters available to authored recipes.

    The same exact instances are not retained between calls. This description is
    an authoring/runtime capability seam, not a plugin discovery mechanism.
    """

    return [
        H3TextToVideoTool(),
        H3FirstLastFrameTool(),
        LTX23TextToVideoTool(),
        LTX23FirstFrameToVideoTool(),
        LTX23ImageToVideoTool(),
        Wan22TextToVideoTool(),
        Wan22ImageToVideoTool(),
        Klein4BTextToImageTool(),
        Klein4BImageToImageTool(),
        KleinTextToImageTool(),
        KleinImageToImageTool(),
        NativeWan14BI2VTool(),
        NativeWan14BFLFTool(),
        NativeWan14BT2VTool(),
        ZImageTurboTextToImageTool(),
    ]


# The registry is explicit by design. Data-defined variants may wrap these curated
# implementations, but arbitrary Python plugin discovery is intentionally unsupported.
def default_registry(
    settings: Settings | None = None,
    *,
    emit_warnings: bool = True,
) -> ToolRegistry:
    # Import lazily so the package can expose Tool/ToolRegistry without creating a
    # tools -> variants -> tools import cycle during module initialization.
    from ..variants import load_variant_tools

    settings = settings or Settings.from_env()
    resources = discover_resources(settings)
    variants = load_variant_tools(
        settings,
        variant_base_tools(),
        resources,
    )
    all_errors = [*resources.errors, *variants.errors]
    if emit_warnings:
        for error in all_errors:
            print(f"LatentSlate Engine catalog warning: {error}")
    return ToolRegistry(
        variants.tools,
        resources=resources,
        variants=variants.entries,
        variant_errors=all_errors,
    )
