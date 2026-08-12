"""Authenticated HTTP authoring endpoints over the shared headless service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, status

from ..acquisition.deployment_install import DeploymentInstallError
from ..acquisition.resource_install import ResourceInstallResult, install_resource
from ..config import Settings
from ..tools import default_registry
from .inspection import _resolve_source_type
from .models import (
    AuthoringCapabilitiesResponse,
    AuthoringSourceType,
    CatalogStatus,
    RecipeDraftRequest,
    RecipeDraftResult,
    RecipePublicationResult,
    RecipePublishRequest,
    RecipeValidationResult,
    ResourceAddRequest,
    ResourceCatalogValidationResult,
    ResourceDeletionRequest,
    ResourceDeletionResult,
    ResourceEditorCatalogResponse,
    ResourceEditorResource,
    ResourceIdSuggestionRequest,
    ResourceIdSuggestionResult,
    ResourceInspectionResult,
    ResourceInspectRequest,
    ResourcePublicationPreview,
    ResourcePublicationResult,
    ResourceUpdateRequest,
)
from .service import (
    CatalogAuthoringError,
    add_resource,
    authoring_capabilities,
    catalog_status,
    delete_resource,
    inspect_resource_source,
    preview_resource,
    publish_recipe_draft,
    resource_editor_catalog,
    resource_editor_resource,
    save_recipe_draft,
    suggest_resource_id,
    update_resource,
    validate_recipe,
    validate_resource_catalog,
)


def register_authoring_routes(
    app: FastAPI,
    settings: Settings,
    *,
    dependencies: list[Any],
    loaded_revision: str,
) -> None:
    """Register administrative authoring routes on one Engine application."""

    @app.get(
        "/v1/authoring/capabilities",
        response_model=AuthoringCapabilitiesResponse,
        dependencies=dependencies,
    )
    async def get_authoring_capabilities() -> AuthoringCapabilitiesResponse:
        return authoring_capabilities()

    @app.get(
        "/v1/authoring/status",
        response_model=CatalogStatus,
        dependencies=dependencies,
    )
    async def get_authoring_status() -> CatalogStatus:
        return catalog_status(settings, loaded_revision)

    @app.post(
        "/v1/authoring/resources/inspect",
        response_model=ResourceInspectionResult,
        dependencies=dependencies,
    )
    async def inspect_resource(request: ResourceInspectRequest) -> ResourceInspectionResult:
        try:
            return inspect_resource_source(
                settings,
                _http_inspection_request(request),
                allow_local=False,
                allow_direct_https=False,
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/v1/authoring/resources",
        response_model=ResourcePublicationResult,
        status_code=status.HTTP_201_CREATED,
        dependencies=dependencies,
    )
    async def publish_resource(request: ResourceAddRequest) -> ResourcePublicationResult:
        try:
            return add_resource(
                settings,
                request.model_copy(
                    update={"inspection": _http_inspection_request(request.inspection)}
                ),
                allow_local=False,
                allow_direct_https=False,
                activation_action="restart_engine",
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.get(
        "/v1/authoring/resources",
        response_model=ResourceEditorCatalogResponse,
        dependencies=dependencies,
    )
    async def list_resources() -> ResourceEditorCatalogResponse:
        return resource_editor_catalog(settings)

    @app.post(
        "/v1/authoring/resources/suggest-id",
        response_model=ResourceIdSuggestionResult,
        dependencies=dependencies,
    )
    async def suggest_id(
        request: ResourceIdSuggestionRequest,
    ) -> ResourceIdSuggestionResult:
        try:
            return suggest_resource_id(settings, request)
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/v1/authoring/resources/preview",
        response_model=ResourcePublicationPreview,
        dependencies=dependencies,
    )
    async def preview_resource_publication(
        request: ResourceUpdateRequest,
        existing_resource_id: str | None = Query(default=None),
    ) -> ResourcePublicationPreview:
        try:
            if request.inspection is not None:
                request = request.model_copy(
                    update={"inspection": _http_inspection_request(request.inspection)}
                )
            return preview_resource(
                settings,
                request,
                existing_resource_id=existing_resource_id,
                allow_local=False,
                allow_direct_https=False,
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.get(
        "/v1/authoring/resources/validate",
        response_model=ResourceCatalogValidationResult,
        dependencies=dependencies,
    )
    async def validate_resources(
        resource_id: str | None = Query(default=None),
    ) -> ResourceCatalogValidationResult:
        return validate_resource_catalog(settings, resource_id)

    @app.post(
        "/v1/authoring/resources/fetch",
        response_model=ResourceInstallResult,
        dependencies=dependencies,
    )
    async def fetch_resource(resource_id: str = Query(min_length=1)) -> ResourceInstallResult:
        try:
            registry = default_registry(settings, emit_warnings=False)
            return install_resource(settings, registry, resource_id)
        except (DeploymentInstallError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/v1/authoring/resources/{resource_id:path}",
        response_model=ResourceEditorResource,
        dependencies=dependencies,
    )
    async def get_resource(resource_id: str) -> ResourceEditorResource:
        try:
            return resource_editor_resource(settings, resource_id)
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.put(
        "/v1/authoring/resources/{resource_id:path}",
        response_model=ResourcePublicationResult,
        dependencies=dependencies,
    )
    async def replace_resource(
        resource_id: str,
        request: ResourceUpdateRequest,
    ) -> ResourcePublicationResult:
        try:
            return update_resource(
                settings,
                resource_id,
                request.model_copy(
                    update={
                        "inspection": (
                            _http_inspection_request(request.inspection)
                            if request.inspection is not None
                            else None
                        )
                    }
                ),
                allow_local=False,
                allow_direct_https=False,
                activation_action="restart_engine",
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.delete(
        "/v1/authoring/resources/{resource_id:path}",
        response_model=ResourceDeletionResult,
        dependencies=dependencies,
    )
    async def remove_resource(
        resource_id: str,
        request: ResourceDeletionRequest,
    ) -> ResourceDeletionResult:
        try:
            return delete_resource(
                settings,
                resource_id,
                delete_artifact=request.delete_artifact,
                activation_action="restart_engine",
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/v1/authoring/recipes/validate",
        response_model=RecipeValidationResult,
        dependencies=dependencies,
    )
    async def validate_recipe_draft(
        request: RecipeDraftRequest,
    ) -> RecipeValidationResult:
        try:
            return validate_recipe(settings, request)
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/v1/authoring/recipes/drafts",
        response_model=RecipeDraftResult,
        status_code=status.HTTP_201_CREATED,
        dependencies=dependencies,
    )
    async def create_recipe_draft(request: RecipeDraftRequest) -> RecipeDraftResult:
        try:
            return save_recipe_draft(settings, request)
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/v1/authoring/recipes/drafts/{recipe_key}/publish",
        response_model=RecipePublicationResult,
        dependencies=dependencies,
    )
    async def publish_recipe(
        recipe_key: str,
        request: RecipePublishRequest,
    ) -> RecipePublicationResult:
        try:
            return publish_recipe_draft(
                settings,
                recipe_key,
                request,
                activation_action="restart_engine",
            )
        except CatalogAuthoringError as exc:
            raise _http_error(exc) from exc


def _http_error(exc: CatalogAuthoringError) -> HTTPException:
    code = status.HTTP_409_CONFLICT if exc.code == "catalog_conflict" else 422
    return HTTPException(status_code=code, detail=str(exc))


def _http_inspection_request(request: ResourceInspectRequest) -> ResourceInspectRequest:
    """Limit remote metadata lookups to the provider token intended for that provider."""

    if request.token_env is None:
        return request
    source_type = _resolve_source_type(request)
    allowed_token_env = {
        AuthoringSourceType.HUGGINGFACE: "HF_TOKEN",
        AuthoringSourceType.CIVITAI: "CIVITAI_TOKEN",
    }.get(source_type)
    if request.token_env != allowed_token_env:
        raise CatalogAuthoringError(
            "HTTP authoring accepts only HF_TOKEN for Hugging Face or CIVITAI_TOKEN for CivitAI",
            code="invalid_authoring_request",
        )
    return request
