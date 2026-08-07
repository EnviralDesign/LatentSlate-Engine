from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from .base import Tool
from .h3 import H3FirstLastFrameTool, H3TextToVideoTool
from .klein import KleinImageToImageTool, KleinTextToImageTool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        tool_list = list(tools)
        self._tools = {tool.descriptor.id: tool for tool in tool_list}
        if len(self._tools) != len(tool_list):
            raise ValueError("Tool IDs must be unique")

    def descriptors(self):
        return [tool.descriptor for tool in self._tools.values()]

    def get(self, tool_id: UUID) -> Tool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {tool_id}") from exc


# The registry is explicit by design. This engine is curated, not a plugin-discovery host.
def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            H3TextToVideoTool(),
            H3FirstLastFrameTool(),
            KleinTextToImageTool(),
            KleinImageToImageTool(),
        ]
    )
