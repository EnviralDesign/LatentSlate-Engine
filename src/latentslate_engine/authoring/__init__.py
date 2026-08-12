"""Shared custom catalog authoring service for CLI and HTTP clients."""

from .models import (
    RecipeDraftRequest,
    RecipePublishRequest,
    ResourceAddRequest,
    ResourceInspectRequest,
)
from .service import (
    CatalogAuthoringError,
    add_resource,
    authoring_capabilities,
    inspect_resource_source,
    publish_recipe_draft,
    save_recipe_draft,
    validate_recipe,
    validate_resource_catalog,
)

__all__ = [
    "CatalogAuthoringError",
    "RecipeDraftRequest",
    "RecipePublishRequest",
    "ResourceAddRequest",
    "ResourceInspectRequest",
    "add_resource",
    "authoring_capabilities",
    "inspect_resource_source",
    "publish_recipe_draft",
    "save_recipe_draft",
    "validate_recipe",
    "validate_resource_catalog",
]
