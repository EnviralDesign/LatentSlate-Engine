from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..acquisition import deployment_install as installer
from ..config import Settings
from ..model_store import MODEL_FAMILIES
from ..recipes import build_recipe_selection_plan
from ..resources import ResourceDescriptor, ResourceKind, discover_resources
from ..tools import ToolRegistry, default_registry, variant_base_tools
from ..variants import OptimizationConfig, VariantDefinition, VariantTool
from .lifecycle import _activation
from .models import (
    AuthoringCapabilitiesResponse,
    BaseToolAuthoringCapability,
    RecipeDraftRequest,
    RecipeDraftResult,
    RecipePublicationResult,
    RecipePublishRequest,
    RecipeValidationResult,
    ResourceAuthoringCapability,
)
from .publication import (
    _dedupe,
    _owned_path,
    _publish_text_file,
    _rollback_text_file,
)
from .service_types import (
    _INSTALL_STATE_TOKENS,
    ActivationAction,
    CatalogAuthoringError,
)
from .toml import load_recipe_file, render_recipe_toml


def authoring_capabilities() -> AuthoringCapabilitiesResponse:
    entries: list[BaseToolAuthoringCapability] = []
    for tool in variant_base_tools():
        family = tool.model_family()
        if family is None:
            continue
        available, reason = tool.variant_base_availability()
        execution = _json_safe(asdict(tool.execution_capabilities()))
        entries.append(
            BaseToolAuthoringCapability(
                descriptor=tool.descriptor,
                family=family,
                runtime_available=available,
                runtime_unavailable_reason=reason,
                execution=execution,
                model_resource_components=sorted(tool.model_resource_components()),
            )
        )
    entries.sort(
        key=lambda item: (item.family, item.descriptor.workflow_kind.value, item.descriptor.key)
    )
    return AuthoringCapabilitiesResponse(
        recipe_schema=VariantDefinition.model_json_schema(),
        optimization_schema=OptimizationConfig.model_json_schema(),
        resource_schema=ResourceDescriptor.model_json_schema(),
        resource_authoring=ResourceAuthoringCapability(
            families=list(MODEL_FAMILIES),
            kinds=list(ResourceKind),
            source_unchanged=True,
        ),
        base_tools=entries,
    )


def validate_recipe(
    settings: Settings,
    request: RecipeDraftRequest,
    *,
    registry: ToolRegistry | None = None,
) -> RecipeValidationResult:
    settings.ensure_directories()
    definition = request.definition
    rendered = render_recipe_toml(definition)
    errors: list[str] = []
    warnings: list[str] = []
    inventory = discover_resources(settings)
    registry = registry or default_registry(settings, emit_warnings=False)

    existing = {entry.key: entry for entry in registry.variants}.get(definition.key)
    if existing is not None:
        local_path = _local_recipe_path(settings, definition.key)
        if not request.replace:
            errors.append(f"recipe key {definition.key!r} already exists")
        elif local_path is None:
            errors.append(
                f"recipe key {definition.key!r} is not owned by the local recipes catalog"
            )

    base_tools = {tool.descriptor.key: tool for tool in variant_base_tools()}
    base = base_tools.get(definition.base_tool)
    draft_tool: VariantTool | None = None
    entry = None
    if base is None:
        errors.append(f"unknown base_tool {definition.base_tool!r}")
    elif base.model_family() != definition.family:
        errors.append(
            f"recipe family {definition.family!r} does not match base_tool family "
            f"{base.model_family()!r}"
        )
    else:
        try:
            draft_tool = VariantTool(
                definition=definition,
                source_path=Path("drafts") / f"{definition.key}.toml",
                base_tool=base,
                inventory=inventory,
                settings=settings,
            )
            entry = draft_tool.catalog_entry()
        except Exception as exc:  # noqa: BLE001 - surface exact typed authoring error
            errors.append(str(exc))

    resolved_resources = []
    if entry is not None:
        for reference in entry.fixed_resources:
            try:
                resolved_resources.append(inventory.resolve(reference, include_components=True))
            except (KeyError, ValueError) as exc:
                errors.append(f"resource {reference!r}: {exc}")

    if draft_tool is not None and base is not None:
        base_available, base_reason = base.variant_base_availability()
        if not base_available:
            warnings.append(
                base_reason
                or "the selected runtime adapter is not installed in this environment; "
                "combination validation is deferred"
            )
        else:
            try:
                execution_request = draft_tool._execution_request(base.descriptor)
                errors.extend(base.validate_execution_request(execution_request))
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        resource_diagnostics = draft_tool._validate_resources()
        recipe_diagnostics = draft_tool._validate_recipe()
        for diagnostic in [*resource_diagnostics, *recipe_diagnostics]:
            if _is_install_state_diagnostic(diagnostic):
                warnings.append(diagnostic)
            else:
                errors.append(diagnostic)
        descriptor = draft_tool.descriptor
        if not descriptor.available and descriptor.unavailable_reason:
            warnings.append(descriptor.unavailable_reason)

    closure = None
    if draft_tool is not None and entry is not None and not errors:
        try:
            kept_tools = [
                tool
                for tool in _registry_tools(registry)
                if tool.descriptor.key != definition.key and tool.descriptor.id != entry.id
            ]
            kept_entries = [
                item
                for item in registry.variants
                if item.key != definition.key and item.id != entry.id
            ]
            transient = ToolRegistry(
                [*kept_tools, draft_tool],
                resources=inventory,
                variants=[*kept_entries, entry],
                variant_errors=list(registry.variant_errors),
            )
            closure = build_recipe_selection_plan(settings, transient, [definition.key])
            warnings.extend(closure.warnings)
            if closure.missing_resources:
                errors.append(
                    "recipe closure contains undeclared resources: "
                    + ", ".join(closure.missing_resources)
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"recipe closure: {exc}")

    return RecipeValidationResult(
        valid=not errors,
        definition=definition,
        toml=rendered,
        errors=_dedupe(errors),
        warnings=_dedupe(warnings),
        closure=closure,
    )


def save_recipe_draft(
    settings: Settings,
    request: RecipeDraftRequest,
    *,
    registry: ToolRegistry | None = None,
) -> RecipeDraftResult:
    validation = validate_recipe(settings, request, registry=registry)
    destination = _recipe_draft_path(settings, request.definition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_root = settings.temp_dir / "catalog-authoring" / uuid4().hex
    installer._mkdir_safe(stage_root, settings.home.resolve())
    stage = stage_root / "draft.toml"
    stage.write_text(validation.toml, encoding="utf-8")
    try:
        _publish_text_file(
            stage,
            destination,
            replace=request.replace,
            label=request.definition.key,
        )
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return RecipeDraftResult(
        draft_key=request.definition.key,
        draft_path=str(destination),
        validation=validation,
    )


def publish_recipe_draft(
    settings: Settings,
    recipe_key: str,
    request: RecipePublishRequest,
    *,
    registry: ToolRegistry | None = None,
    activation_action: ActivationAction = "next_cli_invocation",
) -> RecipePublicationResult:
    draft_path = _find_recipe_draft(settings, recipe_key)
    if draft_path is None:
        raise CatalogAuthoringError(f"unknown recipe draft {recipe_key!r}")
    try:
        definition = load_recipe_file(draft_path)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CatalogAuthoringError(f"invalid recipe draft {recipe_key!r}: {exc}") from exc
    if definition.key != recipe_key:
        raise CatalogAuthoringError("draft key does not match its requested identity")
    draft_request = RecipeDraftRequest(definition=definition, replace=request.replace)
    registry = registry or default_registry(settings, emit_warnings=False)
    validation = validate_recipe(settings, draft_request, registry=registry)
    if not validation.valid:
        raise CatalogAuthoringError("recipe publication refused: " + "; ".join(validation.errors))

    existing_path = _local_recipe_path(settings, recipe_key)
    if existing_path is not None:
        destination = existing_path
    else:
        destination = settings.recipes_root / definition.family / f"{definition.key}.toml"
    destination = _owned_path(settings.recipes_root, destination, "recipe publication")
    if existing_path is not None and not request.replace:
        raise CatalogAuthoringError(
            f"recipe key {recipe_key!r} already exists; pass replace explicitly",
            code="catalog_conflict",
        )

    before_errors = set(registry.variant_errors)
    stage_root = settings.temp_dir / "catalog-authoring" / uuid4().hex
    installer._mkdir_safe(stage_root, settings.home.resolve())
    stage = stage_root / "recipe.toml"
    stage.write_text(validation.toml, encoding="utf-8")
    state: tuple[str, bytes | None] | None = None
    try:
        installer._mkdir_safe(destination.parent, settings.home.resolve())
        state = _publish_text_file(
            stage,
            destination,
            replace=request.replace,
            label=recipe_key,
        )
        refreshed = default_registry(settings, emit_warnings=False)
        new_errors = [error for error in refreshed.variant_errors if error not in before_errors]
        if new_errors:
            raise CatalogAuthoringError(
                "published recipe failed catalog validation: " + "; ".join(new_errors)
            )
        published = next((entry for entry in refreshed.variants if entry.key == recipe_key), None)
        if published is None:
            raise CatalogAuthoringError("published recipe was not rediscovered")
        return RecipePublicationResult(
            recipe_key=recipe_key,
            recipe_path=str(destination),
            validation=validation,
            activation=_activation(settings, activation_action),
        )
    except Exception:
        if state is not None:
            _rollback_text_file(destination, state, stage_root)
        raise
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def validate_recipe_file(
    settings: Settings,
    path: Path,
    *,
    replace: bool = False,
    registry: ToolRegistry | None = None,
) -> RecipeValidationResult:
    try:
        definition = load_recipe_file(path)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CatalogAuthoringError(str(exc)) from exc
    return validate_recipe(
        settings,
        RecipeDraftRequest(definition=definition, replace=replace),
        registry=registry,
    )


def _recipe_draft_path(settings: Settings, definition: VariantDefinition) -> Path:
    path = settings.home / "drafts" / "recipes" / definition.family / f"{definition.key}.toml"
    return _owned_path(settings.home / "drafts" / "recipes", path, "recipe draft")


def _find_recipe_draft(settings: Settings, recipe_key: str) -> Path | None:
    root = settings.home / "drafts" / "recipes"
    if not root.exists():
        return None
    matches = []
    for path in sorted(root.rglob("*.toml")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            definition = load_recipe_file(path)
        except Exception:  # noqa: BLE001, S112 - invalid drafts are reported when selected
            continue
        if definition.key == recipe_key:
            matches.append(path)
    if len(matches) > 1:
        raise CatalogAuthoringError(f"multiple drafts claim recipe key {recipe_key!r}")
    return matches[0] if matches else None


def _local_recipe_path(settings: Settings, recipe_key: str) -> Path | None:
    root = settings.recipes_root
    if not root.exists():
        return None
    matches = []
    for path in sorted(root.rglob("*.toml")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            definition = load_recipe_file(path)
        except Exception:  # noqa: BLE001, S112
            continue
        if definition.key == recipe_key:
            matches.append(path)
    if len(matches) > 1:
        raise CatalogAuthoringError(f"multiple local recipes claim recipe key {recipe_key!r}")
    return matches[0] if matches else None


def _registry_tools(registry: ToolRegistry) -> list[Any]:
    accessor = getattr(registry, "tools", None)
    if callable(accessor):
        return list(accessor())
    return list(getattr(registry, "_tools", {}).values())


def _is_install_state_diagnostic(value: str) -> bool:
    lowered = value.casefold()
    return any(token in lowered for token in _INSTALL_STATE_TOKENS)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, tuple)):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
