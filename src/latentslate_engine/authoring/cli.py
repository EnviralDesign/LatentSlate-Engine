"""CLI grammar and Rich presentation for custom catalog authoring."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import RenderableType
from rich.text import Text

from ..acquisition.deployment_install import DeploymentInstallError
from ..acquisition.resource_install import ResourceInstallResult, install_resource
from ..cli_install_progress import HumanInstallProgress
from ..cli_presentation import (
    bullet_list,
    data_table,
    engine_command,
    identifier,
    key_values,
    next_action,
    page,
    panel,
    status,
)
from ..cli_product import concise_cli_error
from ..config import Settings
from ..resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceFormat,
    ResourceKind,
)
from ..tools import ToolRegistry
from .models import (
    AuthoringCapabilitiesResponse,
    AuthoringSourceType,
    RecipeDraftRequest,
    RecipeDraftResult,
    RecipePublicationResult,
    RecipePublishRequest,
    RecipeValidationResult,
    ResourceAddRequest,
    ResourceCatalogValidationResult,
    ResourceInspectRequest,
    ResourceInspectionResult,
    ResourcePublicationResult,
)
from .service import (
    CatalogAuthoringError,
    add_resource,
    authoring_capabilities,
    inspect_resource_source,
    publish_recipe_draft,
    save_recipe_draft,
    validate_recipe_file,
    validate_resource_catalog,
)


def configure_resource_authoring_cli(
    resource_commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    inspect_parser = resource_commands.add_parser(
        "inspect",
        help="Inspect one source without changing the resource catalog",
    )
    _add_source_arguments(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true")

    add_parser = resource_commands.add_parser(
        "add",
        help="Publish a validated resource declaration and import local files",
    )
    _add_source_arguments(add_parser)
    add_parser.add_argument("--id", dest="resource_id", required=True)
    add_parser.add_argument(
        "--kind",
        choices=[value.value for value in ResourceKind],
        required=True,
    )
    add_parser.add_argument("--family", required=True)
    add_parser.add_argument("--name")
    add_parser.add_argument("--relative-path")
    add_parser.add_argument("--format", choices=[value.value for value in ResourceFormat])
    add_parser.add_argument(
        "--precision", choices=[value.value for value in ArtifactPrecision]
    )
    add_parser.add_argument(
        "--quantization", choices=[value.value for value in ArtifactQuantization]
    )
    add_parser.add_argument("--base-model")
    add_parser.add_argument("--component")
    add_parser.add_argument("--description")
    add_parser.add_argument("--tag", action="append", default=[])
    add_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Add transparent declaration metadata; values accept JSON or plain text",
    )
    add_parser.add_argument("--replace", action="store_true")
    add_parser.add_argument("--json", action="store_true")

    validate_parser = resource_commands.add_parser(
        "validate",
        help="Validate one or all local resource declarations",
    )
    validate_parser.add_argument("resource_id", nargs="?")
    validate_parser.add_argument("--json", action="store_true")

    fetch_parser = resource_commands.add_parser(
        "fetch",
        help="Materialize one already-declared exact resource",
    )
    fetch_parser.add_argument("resource_id")
    fetch_parser.add_argument("--json", action="store_true")


def configure_recipe_authoring_cli(
    recipe_commands: argparse._SubParsersAction[argparse.ArgumentParser],
    recipe_validate: argparse.ArgumentParser,
) -> None:
    recipe_validate.add_argument(
        "--file",
        type=Path,
        help="Validate one draft TOML/JSON file instead of the published catalogs",
    )
    recipe_validate.add_argument(
        "--replace",
        action="store_true",
        help="Validate the file as an intentional replacement of a local recipe",
    )

    capabilities = recipe_commands.add_parser(
        "capabilities",
        help="Describe supported operations, inputs, resources, and runtime modes",
    )
    capabilities.add_argument("--json", action="store_true")

    create = recipe_commands.add_parser(
        "create",
        help="Validate a typed TOML/JSON recipe file and save it as an editable draft",
    )
    create.add_argument("file", type=Path)
    create.add_argument("--replace", action="store_true")
    create.add_argument("--publish", action="store_true")
    create.add_argument("--json", action="store_true")

    publish = recipe_commands.add_parser(
        "publish",
        help="Atomically publish one validated recipe draft",
    )
    publish.add_argument("recipe_key")
    publish.add_argument("--replace", action="store_true")
    publish.add_argument("--json", action="store_true")


def handle_resource_authoring_cli(
    args: argparse.Namespace,
    settings: Settings,
    registry: ToolRegistry,
    parser: argparse.ArgumentParser,
) -> bool:
    command = args.resource_command
    if command not in {"inspect", "add", "validate", "fetch"}:
        return False
    try:
        if command == "inspect":
            payload: Any = inspect_resource_source(settings, _inspection_request(args))
            renderable = format_resource_inspection(payload)
        elif command == "add":
            payload = add_resource(
                settings,
                ResourceAddRequest(
                    inspection=_inspection_request(args),
                    resource_id=args.resource_id,
                    kind=ResourceKind(args.kind),
                    family=args.family,
                    name=args.name,
                    relative_path=args.relative_path,
                    format=ResourceFormat(args.format) if args.format else None,
                    precision=ArtifactPrecision(args.precision) if args.precision else None,
                    quantization=(
                        ArtifactQuantization(args.quantization)
                        if args.quantization
                        else None
                    ),
                    base_model=args.base_model,
                    component=args.component,
                    description=args.description,
                    tags=args.tag,
                    metadata=_metadata(args.metadata),
                    replace=args.replace,
                ),
            )
            renderable = format_resource_publication(payload)
        elif command == "validate":
            payload = validate_resource_catalog(settings, args.resource_id)
            renderable = format_resource_validation(payload)
        else:
            if args.json:
                with redirect_stdout(sys.stderr):
                    payload = install_resource(settings, registry, args.resource_id)
            else:
                with HumanInstallProgress() as progress:
                    payload = install_resource(
                        settings,
                        registry,
                        args.resource_id,
                        progress=progress,
                    )
            renderable = format_resource_install(payload)
    except (CatalogAuthoringError, DeploymentInstallError, ValidationError, OSError) as exc:
        parser.error(concise_cli_error(str(exc)))
    _print_payload(payload, renderable, args.json)
    if command == "validate" and not payload.valid:
        raise SystemExit(1)
    return True


def handle_recipe_authoring_cli(
    args: argparse.Namespace,
    settings: Settings,
    registry: ToolRegistry,
    parser: argparse.ArgumentParser,
) -> bool:
    command = args.recipe_command
    if command == "validate" and args.file is not None:
        try:
            payload: Any = validate_recipe_file(
                settings,
                args.file,
                replace=args.replace,
                registry=registry,
            )
        except (CatalogAuthoringError, ValidationError, OSError) as exc:
            parser.error(concise_cli_error(str(exc)))
        _print_payload(payload, format_recipe_authoring_validation(payload), args.json)
        if not payload.valid:
            raise SystemExit(1)
        return True
    if command not in {"capabilities", "create", "publish"}:
        return False
    try:
        if command == "capabilities":
            payload = authoring_capabilities()
            renderable = format_authoring_capabilities(payload)
        elif command == "create":
            validation = validate_recipe_file(
                settings,
                args.file,
                replace=args.replace,
                registry=registry,
            )
            payload = save_recipe_draft(
                settings,
                RecipeDraftRequest(
                    definition=validation.definition,
                    replace=args.replace,
                ),
                registry=registry,
            )
            renderable = format_recipe_draft(payload)
            if args.publish:
                publication = publish_recipe_draft(
                    settings,
                    payload.draft_key,
                    RecipePublishRequest(replace=args.replace),
                    registry=registry,
                )
                payload = publication
                renderable = format_recipe_publication(publication)
        else:
            payload = publish_recipe_draft(
                settings,
                args.recipe_key,
                RecipePublishRequest(replace=args.replace),
                registry=registry,
            )
            renderable = format_recipe_publication(payload)
    except (CatalogAuthoringError, ValidationError, OSError) as exc:
        parser.error(concise_cli_error(str(exc)))
    _print_payload(payload, renderable, args.json)
    return True


def format_resource_inspection(payload: ResourceInspectionResult) -> RenderableType:
    facts = payload.facts
    overview = key_values(
        (
            ("Source", payload.canonical_source),
            ("Type", payload.source_type.value),
            ("File", facts.filename or "repository snapshot"),
            ("Format", facts.format.value),
            ("Size", str(facts.size_bytes) if facts.size_bytes is not None else "unknown"),
            ("SHA-256", facts.sha256 or "not supplied by source"),
            ("Exact declaration", "yes" if payload.exact_source else "selection required"),
        )
    )
    sections: list[RenderableType] = [panel("Detected facts", overview)]
    if payload.detected:
        sections.append(panel("Detected metadata", _mapping_table(payload.detected)))
    if payload.recommended:
        sections.append(panel("Recommended defaults", _mapping_table(payload.recommended)))
    if payload.candidates:
        table = data_table("ID", "File", "Bytes", "SHA-256", ratio=(1, 3, 1, 3))
        for candidate in payload.candidates:
            table.add_row(
                candidate.id,
                candidate.filename or candidate.label,
                str(candidate.size_bytes or "unknown"),
                candidate.sha256 or "unknown",
            )
        sections.append(panel("Explicit selection required", table, style="yellow"))
    if payload.warnings:
        sections.append(panel("Warnings", bullet_list(payload.warnings), style="yellow"))
    sections.append(next_action(engine_command("resources", "add", payload.canonical_source)))
    return page("Resource inspection", *sections)


def format_resource_publication(payload: ResourcePublicationResult) -> RenderableType:
    activation = payload.activation
    return page(
        f"Resource published · {payload.resource.name}",
        panel(
            "Catalog",
            key_values(
                (
                    ("Resource", identifier(payload.resource.id)),
                    ("Declaration", payload.declaration_path),
                    ("Artifact", payload.artifact_path),
                    ("Disk revision", activation.disk_revision),
                    ("Activation", activation.required_action.replace("_", " ")),
                )
            ),
            style="green",
        ),
        next_action(engine_command("resources", "fetch", payload.resource.id)),
    )


def format_resource_validation(payload: ResourceCatalogValidationResult) -> RenderableType:
    sections: list[RenderableType] = [
        panel(
            "Status",
            status("VALID" if payload.valid else "INVALID", "ok" if payload.valid else "bad"),
            style="green" if payload.valid else "red",
        )
    ]
    if payload.resource is not None:
        sections.append(panel("Resource", identifier(payload.resource.id)))
    if payload.errors:
        sections.append(panel("Errors", bullet_list(payload.errors), style="red"))
    sections.append(panel("Catalog roots", bullet_list(payload.search_paths)))
    return page("Resource declaration validation", *sections)


def format_resource_install(payload: ResourceInstallResult) -> RenderableType:
    return page(
        f"Resource fetch · {payload.resource.name}",
        panel(
            "Result",
            key_values(
                (
                    ("Resource", identifier(payload.resource.id)),
                    ("Status", payload.status.replace("_", " ")),
                    ("Artifact", payload.artifact_path),
                )
            ),
            style="green",
        ),
        next_action(engine_command("resources", "show", payload.resource.id)),
    )


def format_recipe_authoring_validation(payload: RecipeValidationResult) -> RenderableType:
    sections: list[RenderableType] = [
        panel(
            "Status",
            status("VALID" if payload.valid else "INVALID", "ok" if payload.valid else "bad"),
            style="green" if payload.valid else "red",
        ),
        panel("Generated TOML", Text(payload.toml)),
    ]
    if payload.errors:
        sections.append(panel("Errors", bullet_list(payload.errors), style="red"))
    if payload.warnings:
        sections.append(panel("Warnings", bullet_list(payload.warnings), style="yellow"))
    if payload.closure is not None:
        sections.append(
            panel(
                "Closure",
                key_values(
                    (
                        ("Resources", str(len(payload.closure.resources))),
                        ("Total bytes", str(payload.closure.total_bytes)),
                        ("Incremental bytes", str(payload.closure.incremental_bytes)),
                        (
                            "Provisionable",
                            "yes" if payload.closure.remote_provisionable else "no",
                        ),
                    )
                ),
            )
        )
    return page(f"Recipe validation · {payload.definition.key}", *sections)


def format_recipe_draft(payload: RecipeDraftResult) -> RenderableType:
    sections = [
        panel(
            "Draft",
            key_values(
                (
                    ("Recipe", identifier(payload.draft_key)),
                    ("Path", payload.draft_path),
                    (
                        "Validation",
                        status(
                            "VALID" if payload.validation.valid else "NEEDS WORK",
                            "ok" if payload.validation.valid else "warn",
                        ),
                    ),
                )
            ),
        )
    ]
    if payload.validation.errors:
        sections.append(panel("Errors", bullet_list(payload.validation.errors), style="red"))
    if payload.validation.warnings:
        sections.append(panel("Warnings", bullet_list(payload.validation.warnings), style="yellow"))
    sections.append(next_action(engine_command("recipes", "publish", payload.draft_key)))
    return page(f"Recipe draft · {payload.draft_key}", *sections)


def format_recipe_publication(payload: RecipePublicationResult) -> RenderableType:
    return page(
        f"Recipe published · {payload.recipe_key}",
        panel(
            "Catalog",
            key_values(
                (
                    ("Recipe", identifier(payload.recipe_key)),
                    ("Path", payload.recipe_path),
                    ("Activation", payload.activation.required_action.replace("_", " ")),
                    ("Disk revision", payload.activation.disk_revision),
                )
            ),
            style="green",
        ),
        next_action(engine_command("recipes", "show", payload.recipe_key)),
    )


def format_authoring_capabilities(
    payload: AuthoringCapabilitiesResponse,
) -> RenderableType:
    table = data_table("Family", "Operation", "Base tool", "Runtime", "Recipe types")
    for entry in payload.base_tools:
        recipe_types = entry.execution.get("recipe_types") or []
        table.add_row(
            entry.family,
            entry.descriptor.workflow_kind.value,
            identifier(entry.descriptor.key),
            status("READY" if entry.runtime_available else "SCHEMA ONLY", "ok" if entry.runtime_available else "warn"),
            ", ".join(recipe_types) or "fixed model",
        )
    return page(
        "Recipe authoring capabilities",
        table,
        next_action(engine_command("recipes", "create", "<recipe.toml>")),
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source")
    parser.add_argument(
        "--source-type",
        choices=[value.value for value in AuthoringSourceType],
        default=AuthoringSourceType.AUTO.value,
    )
    parser.add_argument("--revision")
    parser.add_argument("--filename")
    parser.add_argument("--file-id", type=int)
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--token-env")
    parser.add_argument("--requires-auth", action="store_true")
    parser.add_argument("--size-bytes", type=int)
    parser.add_argument("--sha256")


def _inspection_request(args: argparse.Namespace) -> ResourceInspectRequest:
    return ResourceInspectRequest(
        source=args.source,
        source_type=AuthoringSourceType(args.source_type),
        revision=args.revision,
        filename=args.filename,
        file_id=args.file_id,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        token_env=args.token_env,
        requires_auth=args.requires_auth,
        expected_size_bytes=args.size_bytes,
        expected_sha256=args.sha256,
    )


def _metadata(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise CatalogAuthoringError("metadata entries must use KEY=JSON")
        key = key.strip()
        if key in result:
            raise CatalogAuthoringError(f"duplicate metadata key {key!r}")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw
    return result


def _mapping_table(values: dict[str, Any]) -> RenderableType:
    table = data_table("Field", "Value", ratio=(1, 3))
    for key in sorted(values):
        table.add_row(key, json.dumps(values[key], sort_keys=True, ensure_ascii=False))
    return table


def _print_payload(payload: Any, renderable: RenderableType, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(payload.model_dump(mode="json"), indent=2))
    else:
        from ..cli_presentation import print_human

        print_human(renderable)
