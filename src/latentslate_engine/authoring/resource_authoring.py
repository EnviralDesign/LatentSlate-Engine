from __future__ import annotations

import shutil
import tomllib
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
    discover_resources,
)
from .inspection import SourceInspectionError, inspect_source, stage_import
from .lifecycle import _activation
from .models import (
    AuthoringSourceType,
    ResourceAddRequest,
    ResourceCatalogValidationResult,
    ResourceInspectionResult,
    ResourcePublicationResult,
)
from .publication import (
    _dedupe,
    _publish_text_file,
    _rollback_text_file,
)
from .service_types import ActivationAction, CatalogAuthoringError, _SAFE_FILENAME
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
    request: ResourceAddRequest,
    *,
    allow_local: bool = True,
    allow_direct_https: bool = True,
    activation_action: ActivationAction = "next_cli_invocation",
) -> ResourcePublicationResult:
    settings.ensure_directories()
    before = discover_resources(settings)
    inspection = inspect_resource_source(
        settings,
        request.inspection,
        allow_local=allow_local,
        allow_direct_https=allow_direct_https,
    )
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
        if inspection.source_type == AuthoringSourceType.LOCAL:
            try:
                inspection = stage_import(request.inspection, inspection, artifact_stage)
            except (SourceInspectionError, installer.DeploymentInstallError, OSError) as exc:
                raise CatalogAuthoringError(str(exc)) from exc

        descriptor = _resource_descriptor(request, inspection)
        target = _declared_resource_path(settings, descriptor)
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
    request: ResourceAddRequest,
    inspection: ResourceInspectionResult,
) -> ResourceDescriptor:
    source = inspection.exact_source
    if source is None:
        raise CatalogAuthoringError("source has no exact declaration or imported artifact")
    facts = inspection.facts
    if facts.size_bytes is None or facts.size_bytes <= 0:
        raise CatalogAuthoringError("resource declaration requires an exact positive byte size")
    resource_format = request.format or facts.format
    if resource_format == ResourceFormat.UNKNOWN and inspection.source_type == AuthoringSourceType.HUGGINGFACE and facts.filename is None:
        resource_format = ResourceFormat.DIRECTORY
    if request.kind == ResourceKind.LORA and resource_format in {
        ResourceFormat.DIFFUSERS,
        ResourceFormat.DIRECTORY,
    }:
        raise CatalogAuthoringError("LoRA resources must be individual files")
    relative_path = request.relative_path or _default_relative_path(
        request.kind,
        request.family,
        inspection,
        resource_format,
    )
    metadata = {
        "authoring_source_type": inspection.source_type.value,
        "authoring_canonical_source": (
            inspection.canonical_source
            if inspection.source_type != AuthoringSourceType.LOCAL
            else "local-import"
        ),
        **inspection.detected,
        **request.metadata,
    }
    if facts.safetensors:
        metadata.setdefault("schema_sha256", facts.safetensors.schema_sha256)
        metadata.setdefault("tensor_count", facts.safetensors.tensor_count)
        metadata.setdefault("tensor_dtypes", facts.safetensors.dtypes)
    try:
        return ResourceDescriptor(
            id=request.resource_id,
            kind=request.kind,
            family=request.family,
            name=request.name or str(inspection.recommended.get("name") or request.resource_id),
            relative_path=relative_path,
            format=resource_format,
            precision=request.precision or facts.precision or ArtifactPrecision.UNKNOWN,
            quantization=request.quantization
            or facts.quantization
            or ArtifactQuantization.UNKNOWN,
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


def _default_relative_path(
    kind: ResourceKind,
    family: str,
    inspection: ResourceInspectionResult,
    resource_format: ResourceFormat,
) -> str:
    root = "models" if kind == ResourceKind.MODEL else "loras"
    filename = inspection.facts.filename
    if filename is None:
        filename = str(inspection.recommended.get("name") or "resource")
    safe = _SAFE_FILENAME.sub("-", filename).strip(".-")
    if not safe:
        safe = "resource"
    if resource_format in {ResourceFormat.DIFFUSERS, ResourceFormat.DIRECTORY}:
        safe = Path(safe).stem
    return (Path(root) / family / "custom" / safe).as_posix()


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
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
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
        raise CatalogAuthoringError(
            "published resource failed discovery: " + "; ".join(new_errors)
        )
    try:
        after.resolve(descriptor.id, include_components=True)
    except (KeyError, ValueError) as exc:
        raise CatalogAuthoringError("published resource was not rediscovered") from exc


def _local_resource_declaration(settings: Settings, resource_id: str) -> Path | None:
    root = settings.resource_declarations_root
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
            f"multiple local declarations claim resource ID {resource_id!r}"
        )
    return matches[0] if matches else None
