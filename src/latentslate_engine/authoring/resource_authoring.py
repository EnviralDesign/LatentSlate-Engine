from __future__ import annotations

import shutil
import tomllib
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from ..acquisition import deployment_install as installer
from ..config import Settings
from ..model_store import MODEL_FAMILIES
from ..resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
    _declared_resource_path,
    _resource_id,
    discover_resources,
)
from .inspection import SourceInspectionError, inspect_source, stage_import
from .lifecycle import _activation
from .models import (
    AuthoringSourceType,
    ResourceAddRequest,
    ResourceCatalogValidationResult,
    ResourceDeclarationOrigin,
    ResourceEditorCatalogResponse,
    ResourceEditorGroup,
    ResourceEditorResource,
    ResourceIdSuggestionRequest,
    ResourceIdSuggestionResult,
    ResourceInspectionResult,
    ResourcePublicationPreview,
    ResourcePublicationResult,
    ResourceUpdateRequest,
)
from .publication import (
    _dedupe,
    _publish_text_file,
    _rollback_text_file,
)
from .service_types import _SAFE_FILENAME, ActivationAction, CatalogAuthoringError
from .toml import render_resource_toml


def inspect_resource_source(
    settings: Settings,
    request: Any,
    *,
    allow_local: bool = True,
    allow_direct_https: bool = True,
) -> ResourceInspectionResult:
    try:
        return inspect_source(
            request,
            settings,
            allow_local=allow_local,
            allow_direct_https=allow_direct_https,
        )
    except (SourceInspectionError, ValidationError) as exc:
        raise CatalogAuthoringError(str(exc)) from exc


def add_resource(
    settings: Settings,
    request: ResourceAddRequest | ResourceUpdateRequest,
    *,
    allow_local: bool = True,
    allow_direct_https: bool = True,
    activation_action: ActivationAction = "next_cli_invocation",
    existing_descriptor: ResourceDescriptor | None = None,
) -> ResourcePublicationResult:
    settings.ensure_directories()
    before = discover_resources(settings)
    if request.inspection is None:
        if existing_descriptor is None:
            raise CatalogAuthoringError("resource inspection is required when creating a resource")
        inspection = _retained_source_inspection(existing_descriptor)
        descriptor = _descriptor_from_retained_source(existing_descriptor, request)
    else:
        inspection = inspect_resource_source(
            settings,
            request.inspection,
            allow_local=allow_local,
            allow_direct_https=allow_direct_https,
        )
        descriptor = _resource_descriptor(request, inspection)
        if existing_descriptor is not None:
            descriptor = _merge_update_descriptor(existing_descriptor, descriptor, request)
    if request.family not in MODEL_FAMILIES:
        raise CatalogAuthoringError(
            f"unknown resource family {request.family!r}; expected one of {MODEL_FAMILIES}"
        )
    if inspection.candidates and inspection.exact_source is None:
        raise CatalogAuthoringError(
            "source selection is ambiguous; choose one exact candidate file_id"
        )

    stage_root = settings.temp_dir / "catalog-authoring" / uuid4().hex
    installer._mkdir_safe(stage_root, settings.home.resolve())
    artifact_stage = stage_root / "artifact"
    created_artifact = False
    declaration_state: tuple[str, bytes | None] | None = None
    target: Path | None = None
    declaration_path: Path | None = None
    try:
        if inspection.source_type == AuthoringSourceType.LOCAL and request.inspection is not None:
            try:
                inspection = stage_import(request.inspection, inspection, artifact_stage)
            except (SourceInspectionError, installer.DeploymentInstallError, OSError) as exc:
                raise CatalogAuthoringError(str(exc)) from exc

        target = _declared_resource_path(settings, descriptor)
        _validate_discovery_identity(settings, descriptor, target)
        local_declaration = _local_resource_declaration(settings, descriptor.id)
        declaration_path = (
            local_declaration
            if request.replace and local_declaration is not None
            else _resource_declaration_path(settings, descriptor.id)
        )
        _check_resource_conflicts(
            settings,
            before,
            descriptor,
            target,
            declaration_path,
            replace=request.replace,
        )

        try:
            rendered = render_resource_toml(descriptor)
        except (TypeError, ValueError) as exc:
            raise CatalogAuthoringError(f"resource cannot be serialized to TOML: {exc}") from exc
        declaration_stage = stage_root / "declaration.toml"
        declaration_stage.write_text(rendered, encoding="utf-8")
        _validate_staged_resource_declaration(declaration_stage, descriptor)

        if artifact_stage.exists():
            installer._mkdir_safe(target.parent, settings.home.resolve())
            if installer._exists(target):
                candidate = ResourceInventory(
                    resources=[descriptor],
                    paths={descriptor.id: target},
                )
                if not candidate.is_installed(descriptor.id):
                    raise CatalogAuthoringError(
                        f"artifact target already exists with different or incomplete content: {target}",
                        code="catalog_conflict",
                    )
                artifact_stage.unlink()
            else:
                installer._publish_file_no_clobber(artifact_stage, target, descriptor.id)
                created_artifact = True

        installer._mkdir_safe(declaration_path.parent, settings.home.resolve())
        declaration_state = _publish_text_file(
            declaration_stage,
            declaration_path,
            replace=request.replace,
            label=descriptor.id,
        )

        after = discover_resources(settings)
        _verify_resource_publication(before, after, descriptor)
        published = after.resolve(descriptor.id, include_components=True)
        return ResourcePublicationResult(
            resource=published,
            declaration_path=str(declaration_path),
            artifact_path=str(after.path_for(published.id)),
            inspection=inspection,
            activation=_activation(settings, activation_action),
        )
    except Exception:
        if declaration_path is not None and declaration_state is not None:
            _rollback_text_file(declaration_path, declaration_state, stage_root)
        if created_artifact and target is not None and installer._exists(target):
            installer._safe_unlink(target, target.parent)
        raise
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def update_resource(
    settings: Settings,
    resource_id: str,
    request: ResourceUpdateRequest,
    *,
    allow_local: bool = True,
    allow_direct_https: bool = True,
    activation_action: ActivationAction = "next_cli_invocation",
) -> ResourcePublicationResult:
    """Replace one local declaration without allowing its identity or path to move."""

    if request.resource_id != resource_id:
        raise CatalogAuthoringError(
            "resource ID in the request must match the resource being updated",
            code="catalog_conflict",
        )
    inventory = discover_resources(settings)
    existing = inventory.by_id().get(resource_id)
    if existing is None:
        raise CatalogAuthoringError(
            f"resource ID {resource_id!r} does not exist to replace",
            code="catalog_conflict",
        )
    if _local_resource_declaration(settings, resource_id) is None:
        raise CatalogAuthoringError(
            f"resource ID {resource_id!r} is not owned by the local declaration catalog",
            code="catalog_conflict",
        )
    if request.relative_path is not None and request.relative_path != existing.relative_path:
        raise CatalogAuthoringError(
            "resource replacement cannot move an existing artifact path",
            code="catalog_conflict",
        )
    return add_resource(
        settings,
        request.model_copy(update={"relative_path": existing.relative_path, "replace": True}),
        allow_local=allow_local,
        allow_direct_https=allow_direct_https,
        activation_action=activation_action,
        existing_descriptor=existing,
    )


def preview_resource(
    settings: Settings,
    request: ResourceAddRequest | ResourceUpdateRequest,
    *,
    existing_resource_id: str | None = None,
    allow_local: bool = True,
    allow_direct_https: bool = True,
) -> ResourcePublicationPreview:
    """Validate a create or local update without writing catalog or artifact state."""

    try:
        inventory = discover_resources(settings)
        existing = None
        update_mode = existing_resource_id is not None
        if existing_resource_id is not None:
            if request.resource_id != existing_resource_id:
                raise CatalogAuthoringError(
                    "resource ID in the request must match the resource being updated",
                    code="catalog_conflict",
                )
            existing = inventory.by_id().get(existing_resource_id)
            if existing is None:
                raise CatalogAuthoringError(
                    f"resource ID {existing_resource_id!r} does not exist to replace",
                    code="catalog_conflict",
                )
            if _local_resource_declaration(settings, existing_resource_id) is None:
                raise CatalogAuthoringError(
                    f"resource ID {existing_resource_id!r} is not owned by the local declaration catalog",
                    code="catalog_conflict",
                )
            if (
                request.relative_path is not None
                and request.relative_path != existing.relative_path
            ):
                raise CatalogAuthoringError(
                    "resource replacement cannot move an existing artifact path",
                    code="catalog_conflict",
                )
        if request.inspection is None:
            if existing is None:
                raise CatalogAuthoringError(
                    "resource inspection is required when creating a resource"
                )
            inspection = _retained_source_inspection(existing)
            descriptor = _descriptor_from_retained_source(existing, request)
        else:
            inspection = inspect_resource_source(
                settings,
                request.inspection,
                allow_local=allow_local,
                allow_direct_https=allow_direct_https,
            )
            descriptor = _resource_descriptor(request, inspection)
            if existing is not None:
                descriptor = _merge_update_descriptor(existing, descriptor, request)
        target = _declared_resource_path(settings, descriptor)
        _validate_discovery_identity(settings, descriptor, target)
        local_declaration = _local_resource_declaration(settings, descriptor.id)
        declaration_path = (
            local_declaration
            if update_mode and local_declaration is not None
            else _resource_declaration_path(settings, descriptor.id)
        )
        _check_resource_conflicts(
            settings,
            inventory,
            descriptor,
            target,
            declaration_path,
            replace=update_mode or request.replace,
        )
        rendered = render_resource_toml(descriptor)
        _validate_rendered_resource_toml(rendered, descriptor)
        return ResourcePublicationPreview(
            valid=True,
            resource=descriptor,
            toml=rendered,
            inspection=inspection,
            warnings=list(inspection.warnings),
        )
    except (CatalogAuthoringError, ValidationError, TypeError, ValueError) as exc:
        return ResourcePublicationPreview(valid=False, errors=[str(exc)])


def resource_editor_catalog(settings: Settings) -> ResourceEditorCatalogResponse:
    """Discover the catalog on demand, annotating declarations with their owner."""

    inventory = discover_resources(settings)
    resources = [
        _resource_editor_entry(settings, resource)
        for resource in sorted(inventory.resources, key=lambda item: item.id)
    ]
    grouped: dict[tuple[ResourceKind, str], list[str]] = {}
    for resource in resources:
        grouped.setdefault((resource.kind, resource.family), []).append(resource.id)
    groups = [
        ResourceEditorGroup(kind=kind, family=family, resource_ids=resource_ids)
        for (kind, family), resource_ids in sorted(
            grouped.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    ]
    return ResourceEditorCatalogResponse(
        resources=resources,
        groups=groups,
        errors=list(inventory.errors),
    )


def resource_editor_resource(
    settings: Settings,
    resource_id: str,
) -> ResourceEditorResource:
    """Return one freshly discovered resource editor entry."""

    inventory = discover_resources(settings)
    resource = inventory.by_id().get(resource_id)
    if resource is None:
        raise CatalogAuthoringError(f"unknown resource ID {resource_id!r}")
    return _resource_editor_entry(settings, resource)


def suggest_resource_id(
    settings: Settings,
    request: ResourceIdSuggestionRequest,
) -> ResourceIdSuggestionResult:
    """Generate a readable ID whose source fingerprint makes unrelated imports stable."""

    if request.family not in MODEL_FAMILIES:
        raise CatalogAuthoringError(
            f"unknown resource family {request.family!r}; expected one of {MODEL_FAMILIES}"
        )
    readable = _resource_id_slug(request.name)
    source_fingerprint = sha256(request.source.strip().encode("utf-8")).hexdigest()[:8]
    base_resource_id = f"{request.kind.value}:{request.family}:{readable}-{source_fingerprint}"
    used_ids = set(discover_resources(settings).by_id())
    resource_id = base_resource_id
    collision_index = 0
    while resource_id in used_ids:
        collision_index += 1
        resource_id = f"{base_resource_id}-{collision_index + 1}"
    return ResourceIdSuggestionResult(
        resource_id=resource_id,
        base_resource_id=base_resource_id,
        collision_index=collision_index,
    )


def validate_resource_catalog(
    settings: Settings,
    resource_id: str | None = None,
) -> ResourceCatalogValidationResult:
    inventory = discover_resources(settings)
    selected = None
    errors = list(inventory.errors)
    if resource_id is not None:
        try:
            selected = inventory.resolve(resource_id, include_components=True)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
    return ResourceCatalogValidationResult(
        valid=not errors,
        resource_id=resource_id,
        resource=selected,
        errors=_dedupe(errors),
        search_paths=[str(path) for _, path in settings.resource_declaration_roots()],
    )


def _resource_descriptor(
    request: ResourceAddRequest | ResourceUpdateRequest,
    inspection: ResourceInspectionResult,
) -> ResourceDescriptor:
    source = inspection.exact_source
    if source is None:
        raise CatalogAuthoringError("source has no exact declaration or imported artifact")
    facts = inspection.facts
    if facts.size_bytes is None or facts.size_bytes <= 0:
        raise CatalogAuthoringError("resource declaration requires an exact positive byte size")
    resource_format = request.format or facts.format
    if (
        resource_format == ResourceFormat.UNKNOWN
        and inspection.source_type == AuthoringSourceType.HUGGINGFACE
        and facts.filename is None
    ):
        resource_format = ResourceFormat.DIRECTORY
    if request.kind == ResourceKind.LORA and resource_format in {
        ResourceFormat.DIFFUSERS,
        ResourceFormat.DIRECTORY,
    }:
        raise CatalogAuthoringError("LoRA resources must be individual files")
    if request.kind == ResourceKind.LORA and not (request.base_model or "").strip():
        raise CatalogAuthoringError(
            "authored LoRA resources require base_model",
        )
    relative_path = request.relative_path or _default_relative_path(
        request.kind,
        request.family,
        request.resource_id,
        inspection,
        resource_format,
    )
    metadata = dict(request.metadata)
    metadata.update(
        {
            "authoring_source_type": inspection.source_type.value,
            "authoring_canonical_source": (
                inspection.canonical_source
                if inspection.source_type != AuthoringSourceType.LOCAL
                else "local-import"
            ),
        }
    )
    metadata.update(inspection.detected)
    if facts.safetensors:
        metadata["schema_sha256"] = facts.safetensors.schema_sha256
        metadata["tensor_count"] = facts.safetensors.tensor_count
        metadata["tensor_dtypes"] = facts.safetensors.dtypes
    try:
        return ResourceDescriptor(
            id=request.resource_id,
            kind=request.kind,
            family=request.family,
            name=request.name,
            relative_path=relative_path,
            format=resource_format,
            precision=request.precision or facts.precision or ArtifactPrecision.UNKNOWN,
            quantization=request.quantization or facts.quantization or ArtifactQuantization.UNKNOWN,
            size_bytes=facts.size_bytes,
            description=request.description,
            tags=list(dict.fromkeys(request.tags)),
            base_model=request.base_model,
            component=request.component,
            metadata=metadata,
            sources=[source],
        )
    except ValidationError as exc:
        raise CatalogAuthoringError(str(exc)) from exc


def _descriptor_from_retained_source(
    existing: ResourceDescriptor,
    request: ResourceUpdateRequest,
) -> ResourceDescriptor:
    """Apply editable fields while retaining an already-published acquisition identity."""

    updates: dict[str, Any] = {
        "kind": request.kind,
        "family": request.family,
        "name": request.name,
        "relative_path": existing.relative_path,
    }
    for field in (
        "format",
        "precision",
        "quantization",
        "base_model",
        "component",
        "description",
    ):
        if field in request.model_fields_set:
            updates[field] = getattr(request, field)
    if "tags" in request.model_fields_set:
        updates["tags"] = list(request.tags)
    if "metadata" in request.model_fields_set:
        updates["metadata"] = dict(request.metadata)
    descriptor = existing.model_copy(update=updates)
    if descriptor.kind == ResourceKind.LORA and not (descriptor.base_model or "").strip():
        raise CatalogAuthoringError("authored LoRA resources require base_model")
    return descriptor


def _merge_update_descriptor(
    existing: ResourceDescriptor,
    descriptor: ResourceDescriptor,
    request: ResourceUpdateRequest,
) -> ResourceDescriptor:
    """Keep declaration fields outside the add form and omission-sensitive collections."""

    updates: dict[str, Any] = {
        "trigger_words": existing.trigger_words,
        "default_strength": existing.default_strength,
        "config": existing.config,
    }
    if "tags" not in request.model_fields_set:
        updates["tags"] = existing.tags
    if "metadata" not in request.model_fields_set:
        updates["metadata"] = existing.metadata
    return descriptor.model_copy(update=updates)


def _retained_source_inspection(existing: ResourceDescriptor) -> ResourceInspectionResult:
    """Describe an unchanged declaration source without reading a local browser path."""

    source = existing.sources[0] if existing.sources else None
    source_type = (
        AuthoringSourceType.LOCAL
        if source is None or source.type.value == "manual"
        else AuthoringSourceType(source.type.value)
    )
    return ResourceInspectionResult(
        source_type=source_type,
        canonical_source="existing-declaration",
        facts={
            "filename": Path(existing.relative_path).name,
            "size_bytes": existing.size_bytes,
            "sha256": source.sha256 if source is not None else None,
            "format": existing.format,
            "precision": existing.precision,
            "quantization": existing.quantization,
        },
        exact_source=source,
        warnings=["existing acquisition source retained without re-inspection"],
    )


def _default_relative_path(
    kind: ResourceKind,
    family: str,
    resource_id: str,
    inspection: ResourceInspectionResult,
    resource_format: ResourceFormat,
) -> str:
    root = "models" if kind == ResourceKind.MODEL else "loras"
    prefix = f"{kind.value}:{family}:"
    if not resource_id.startswith(prefix):
        raise CatalogAuthoringError(
            f"resource ID must start with {prefix!r} so file-drop discovery preserves identity"
        )
    resource_key = resource_id.removeprefix(prefix).strip("/")
    if not resource_key:
        raise CatalogAuthoringError("resource ID must contain a stable key after its family")
    filename = inspection.facts.filename
    if filename is None:
        filename = str(inspection.recommended.get("name") or "resource")
    if resource_format in {ResourceFormat.DIFFUSERS, ResourceFormat.DIRECTORY}:
        relative_key = Path(resource_key)
    else:
        suffix = Path(filename).suffix.casefold()
        relative_key = Path(
            resource_key
            if suffix and resource_key.casefold().endswith(suffix)
            else resource_key + suffix
        )
    return (Path(root) / family / relative_key).as_posix()


def _validate_discovery_identity(
    settings: Settings,
    descriptor: ResourceDescriptor,
    target: Path,
) -> None:
    kind_root = settings.model_root if descriptor.kind == ResourceKind.MODEL else settings.lora_root
    family_root = kind_root / descriptor.family
    resource_key = target.relative_to(family_root.resolve()).as_posix()
    discovered_id = _resource_id(
        descriptor.kind,
        descriptor.family,
        resource_key,
    )
    if discovered_id != descriptor.id:
        raise CatalogAuthoringError(
            "resource relative_path would be rediscovered under a different ID: "
            f"{discovered_id!r}; choose a path matching {descriptor.id!r}"
        )


def _resource_declaration_path(settings: Settings, resource_id: str) -> Path:
    safe = _SAFE_FILENAME.sub("--", resource_id).strip(".-")
    if not safe:
        raise CatalogAuthoringError("resource ID cannot produce a declaration filename")
    return settings.resource_declarations_root / f"{safe}.toml"


def _check_resource_conflicts(
    settings: Settings,
    inventory: ResourceInventory,
    descriptor: ResourceDescriptor,
    target: Path,
    declaration_path: Path,
    *,
    replace: bool,
) -> None:
    by_id = inventory.by_id()
    existing = by_id.get(descriptor.id)
    local_declaration = _local_resource_declaration(settings, descriptor.id)
    if installer._exists(target):
        candidate = ResourceInventory(resources=[descriptor], paths={descriptor.id: target})
        if not candidate.is_installed(descriptor.id):
            raise CatalogAuthoringError(
                f"artifact target exists but does not match the proposed declaration: {target}; "
                "remove or relocate it before changing source identity",
                code="catalog_conflict",
            )
    if existing is not None:
        if not replace:
            raise CatalogAuthoringError(
                f"resource ID {descriptor.id!r} already exists",
                code="catalog_conflict",
            )
        if local_declaration is None:
            raise CatalogAuthoringError(
                f"resource ID {descriptor.id!r} is not owned by the local declaration catalog",
                code="catalog_conflict",
            )
        if inventory.path_for(existing.id).resolve() != target.resolve():
            raise CatalogAuthoringError(
                "resource replacement cannot move an existing artifact path",
                code="catalog_conflict",
            )
    elif replace:
        raise CatalogAuthoringError(
            f"resource ID {descriptor.id!r} does not exist to replace",
            code="catalog_conflict",
        )
    for resource in inventory.resources:
        if resource.id == descriptor.id:
            continue
        try:
            existing_target = inventory.path_for(resource.id).resolve()
        except KeyError:
            continue
        if existing_target == target.resolve():
            raise CatalogAuthoringError(
                f"artifact path is already owned by resource {resource.id!r}",
                code="catalog_conflict",
            )
    if installer._exists(declaration_path) and local_declaration is None:
        raise CatalogAuthoringError(
            f"declaration path already exists: {declaration_path}",
            code="catalog_conflict",
        )


def _validate_staged_resource_declaration(path: Path, descriptor: ResourceDescriptor) -> None:
    _validate_rendered_resource_toml(path.read_text(encoding="utf-8"), descriptor)


def _validate_rendered_resource_toml(rendered: str, descriptor: ResourceDescriptor) -> None:
    raw = tomllib.loads(rendered)
    parsed = ResourceDescriptor.model_validate(raw.get("resource", raw))
    if parsed.model_dump(exclude={"available", "unavailable_reason"}) != descriptor.model_dump(
        exclude={"available", "unavailable_reason"}
    ):
        raise CatalogAuthoringError("generated resource TOML failed deterministic round-trip")


def _verify_resource_publication(
    before: ResourceInventory,
    after: ResourceInventory,
    descriptor: ResourceDescriptor,
) -> None:
    before_errors = set(before.errors)
    new_errors = [error for error in after.errors if error not in before_errors]
    if new_errors:
        raise CatalogAuthoringError("published resource failed discovery: " + "; ".join(new_errors))
    try:
        after.resolve(descriptor.id, include_components=True)
    except (KeyError, ValueError) as exc:
        raise CatalogAuthoringError("published resource was not rediscovered") from exc


def _local_resource_declaration(settings: Settings, resource_id: str) -> Path | None:
    return _resource_declaration_in_root(
        settings.resource_declarations_root,
        resource_id,
        label="local",
    )


def _resource_editor_entry(
    settings: Settings,
    resource: ResourceDescriptor,
) -> ResourceEditorResource:
    local_path = _local_resource_declaration(settings, resource.id)
    if local_path is not None:
        return ResourceEditorResource(
            **resource.model_dump(),
            editable=True,
            declaration_origin=ResourceDeclarationOrigin.LOCAL,
            declaration_path=_declaration_display_path(
                settings,
                local_path,
                ResourceDeclarationOrigin.LOCAL,
            ),
        )
    builtin_path = _resource_declaration_in_root(
        settings.builtin_resource_declarations_root,
        resource.id,
        label="builtin",
    )
    if builtin_path is not None:
        return ResourceEditorResource(
            **resource.model_dump(),
            editable=False,
            declaration_origin=ResourceDeclarationOrigin.BUILTIN,
            declaration_path=_declaration_display_path(
                settings,
                builtin_path,
                ResourceDeclarationOrigin.BUILTIN,
            ),
        )
    return ResourceEditorResource(
        **resource.model_dump(),
        editable=False,
        declaration_origin=ResourceDeclarationOrigin.DISCOVERED,
    )


def _resource_declaration_in_root(
    root: Path,
    resource_id: str,
    *,
    label: str,
) -> Path | None:
    if not root.exists():
        return None
    matches = []
    for path in sorted(root.rglob("*.toml")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            value = raw.get("resource", raw)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            continue
        if isinstance(value, dict) and value.get("id") == resource_id:
            matches.append(path)
    if len(matches) > 1:
        raise CatalogAuthoringError(
            f"multiple {label} declarations claim resource ID {resource_id!r}"
        )
    return matches[0] if matches else None


def _declaration_display_path(
    settings: Settings,
    path: Path,
    origin: ResourceDeclarationOrigin,
) -> str:
    if origin == ResourceDeclarationOrigin.LOCAL:
        return path.relative_to(settings.home).as_posix()
    return (
        Path("builtin_resource_declarations")
        / path.relative_to(settings.builtin_resource_declarations_root)
    ).as_posix()


def _resource_id_slug(value: str) -> str:
    """Turn a human display name into the readable component of an authored ID."""

    characters = [
        character if character.isascii() and character.isalnum() else "-"
        for character in value.casefold()
    ]
    slug = "".join(characters).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "resource"
