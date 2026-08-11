from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..bundles import configured_bundles
from ..config import Settings
from ..model_store import owned_repository_directory
from ..protocol import BundleStatus, ToolDescriptor
from ..resources import ResourceInventory, discover_resources
from ..runtime.diffusers_repository import (
    H3_REPOSITORY_CONTRACT,
    KLEIN4B_REPOSITORY_CONTRACT,
    LTX23_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext
from .h3 import H3FirstLastFrameTool, H3TextToVideoTool
from .klein import (
    Klein4BImageToImageTool,
    Klein4BTextToImageTool,
    KleinImageToImageTool,
    KleinTextToImageTool,
)
from .ltx23 import LTX23TextToVideoTool
from .wan22 import Wan22TextToVideoTool
from .wan22_native import NativeWan14BI2VTool

if TYPE_CHECKING:
    from ..resources import ResourceDescriptor
    from ..variants import VariantCatalogEntry


_EXACT_CANONICAL_REPOSITORIES = {
    "h3-basic": H3_REPOSITORY_CONTRACT,
    "ltx23-basic": LTX23_REPOSITORY_CONTRACT,
    "klein4b-basic": KLEIN4B_REPOSITORY_CONTRACT,
}


class _CatalogAvailabilityTool(Tool):
    """Bind one curated tool to the canonical bundles present at catalog build time."""

    def __init__(self, base: Tool, descriptor: ToolDescriptor) -> None:
        self._base = base
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._base.run(context, inputs)

    def provenance(self) -> dict[str, Any]:
        return self._base.provenance()

    def model_family(self) -> str | None:
        return self._base.model_family()

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return self._base.variant_base_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        return self._base.execution_capabilities()

    def model_resource_components(self) -> frozenset[str]:
        return self._base.model_resource_components()

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        return self._base.validate_execution_request(request)

    def validate_model_resource(
        self,
        resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        return self._base.validate_model_resource(resource, path)


def _bind_canonical_availability(settings: Settings, tool: Tool) -> Tool:
    descriptor = tool.descriptor
    if not descriptor.available:
        return _CatalogAvailabilityTool(tool, descriptor)

    bundles = configured_bundles(settings)
    for requirement in descriptor.requirements:
        if not requirement.required:
            continue
        bundle = bundles.get(requirement.bundle_id)
        if bundle is None:
            reason = f"Required model bundle {requirement.bundle_id!r} is not configured"
            return _CatalogAvailabilityTool(
                tool,
                descriptor.model_copy(update={"available": False, "unavailable_reason": reason}),
            )
        status = bundle.status(settings.model_root)
        if status != BundleStatus.INSTALLED:
            reason = (
                f"Required model bundle {bundle.id!r} is {status.value}. "
                f"Run `latentslate-engine bundles install {bundle.id}`."
            )
            return _CatalogAvailabilityTool(
                tool,
                descriptor.model_copy(update={"available": False, "unavailable_reason": reason}),
            )

        contract = _EXACT_CANONICAL_REPOSITORIES.get(bundle.id)
        if contract is None:
            continue
        repository = owned_repository_directory(
            settings.model_root,
            bundle.id,
            bundle.repo_id,
        )
        try:
            validate_diffusers_repository(repository, contract)
        except (OSError, TypeError, ValueError) as exc:
            reason = (
                f"Required model bundle {bundle.id!r} is incompatible: {exc}. "
                f"Reinstall it with `latentslate-engine bundles install {bundle.id}`."
            )
            return _CatalogAvailabilityTool(
                tool,
                descriptor.model_copy(update={"available": False, "unavailable_reason": reason}),
            )

    return _CatalogAvailabilityTool(tool, descriptor)


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

    def descriptors(self):
        return [tool.descriptor for tool in self._tools.values()]

    def get(self, tool_id: UUID) -> Tool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {tool_id}") from exc


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
    variant_bases: list[Tool] = [
        H3TextToVideoTool(),
        H3FirstLastFrameTool(),
        LTX23TextToVideoTool(),
        Wan22TextToVideoTool(),
        Klein4BTextToImageTool(),
        Klein4BImageToImageTool(),
        KleinTextToImageTool(),
        KleinImageToImageTool(),
    ]
    resources = discover_resources(settings)
    variants = load_variant_tools(
        settings,
        [*variant_bases, NativeWan14BI2VTool()],
        resources,
    )
    base_tools = [_bind_canonical_availability(settings, tool) for tool in variant_bases]
    all_errors = [*resources.errors, *variants.errors]
    if emit_warnings:
        for error in all_errors:
            print(f"LatentSlate Engine catalog warning: {error}")
    return ToolRegistry(
        [*base_tools, *variants.tools],
        resources=resources,
        variants=variants.entries,
        variant_errors=all_errors,
    )
