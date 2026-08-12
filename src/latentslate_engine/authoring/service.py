"""Shared custom catalog authoring service for CLI and HTTP clients."""

from .lifecycle import catalog_disk_revision, catalog_status
from .recipe_authoring import (
    authoring_capabilities,
    publish_recipe_draft,
    save_recipe_draft,
    validate_recipe,
    validate_recipe_file,
)
from .resource_authoring import (
    add_resource,
    inspect_resource_source,
    validate_resource_catalog,
)
from .service_types import CatalogAuthoringError

__all__ = [
    "CatalogAuthoringError",
    "add_resource",
    "authoring_capabilities",
    "catalog_disk_revision",
    "catalog_status",
    "inspect_resource_source",
    "publish_recipe_draft",
    "save_recipe_draft",
    "validate_recipe",
    "validate_recipe_file",
    "validate_resource_catalog",
]
