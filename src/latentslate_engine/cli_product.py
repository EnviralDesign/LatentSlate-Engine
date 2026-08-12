"""Human-first, deterministic views for the recipe and resource CLI.

These helpers only consume already-discovered catalog data.  They deliberately
avoid importing a model runtime so listing, planning, and inspection work in a
protocol-only installation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from rich.console import RenderableType
from rich.text import Text

from . import __version__
from .cli_presentation import (
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
from .deployment_summary import format_iec_bytes
from .recipes import (
    DeploymentPlan,
    DeploymentProfileCatalogResponse,
    RecipeCatalogResponse,
    build_recipe_selection_plan,
)
from .resources import ResourceDescriptor

if TYPE_CHECKING:
    from .config import Settings
    from .tools import ToolRegistry


def recipe_detail_payload(
    settings: Settings,
    registry: ToolRegistry,
    recipe_key: str,
) -> dict[str, Any]:
    """Return an inspectable, JSON-safe view of one runnable-recipe definition."""

    entries = {entry.key: entry for entry in registry.variants}
    try:
        entry = entries[recipe_key]
    except KeyError as exc:
        raise KeyError(f"Unknown recipe {recipe_key!r}") from exc
    descriptors = {descriptor.key: descriptor for descriptor in registry.descriptors()}
    descriptor = descriptors.get(recipe_key)
    plan = build_recipe_selection_plan(settings, registry, [recipe_key])
    resources = {resource.id: resource for resource in plan.resources}

    required_resources: list[dict[str, Any]] = []
    used_references: set[str] = set()
    if entry.model_resource:
        required_resources.append(
            _recipe_resource_payload("model", entry.model_resource, registry, resources)
        )
        used_references.add(entry.model_resource)
    for role, reference in sorted(entry.recipe_resources.items()):
        required_resources.append(_recipe_resource_payload(role, reference, registry, resources))
        used_references.add(reference)
    for reference in entry.fixed_resources:
        if reference not in used_references:
            required_resources.append(
                _recipe_resource_payload("required_resource", reference, registry, resources)
            )

    operation: dict[str, Any] = {
        "base_tool": entry.base_tool,
        "recipe_type": entry.recipe_type,
    }
    if descriptor is not None:
        operation.update(
            {
                "workflow_kind": descriptor.workflow_kind.value,
                "output": descriptor.output.model_dump(mode="json"),
            }
        )

    return {
        "engine_version": __version__,
        "recipe": entry.model_dump(mode="json"),
        "identity": {
            "id": str(entry.id),
            "schema_hash": descriptor.schema_hash if descriptor else None,
            "source_path": entry.source_path,
        },
        "operation": operation,
        "execution": entry.optimizations,
        "required_resources": required_resources,
        "dynamic_resource_slots": entry.dynamic_resource_slots,
        "state": {
            "recipe_available": entry.available,
            "locally_runnable": plan.locally_runnable,
            "automatic_provisionable": plan.remote_provisionable,
            "blockers": _plan_blockers(plan),
            "warnings": plan.warnings,
        },
    }


def resource_detail_payload(registry: ToolRegistry, resource_id: str) -> dict[str, Any]:
    """Return one resource with disk state and the recipes that require it."""

    resources = registry.resources.by_id()
    try:
        resource = resources[resource_id]
    except KeyError as exc:
        raise KeyError(f"Unknown resource {resource_id!r}") from exc
    exact_sources = [source for source in resource.sources if source.is_exact()]
    required_secrets = sorted(
        {secret for source in exact_sources if (secret := source.required_secret()) is not None}
    )
    referenced_by = []
    for entry in sorted(registry.variants, key=lambda item: item.key):
        roles = _resource_roles(entry, resource, registry.resources)
        if roles:
            referenced_by.append({"recipe_key": entry.key, "roles": roles})

    return {
        "engine_version": __version__,
        "resource": resource.model_dump(mode="json"),
        "artifact_path": str(registry.resources.path_for(resource.id)),
        "state": {
            "installed": registry.resources.is_installed(resource.id),
            "automatic_provisionable": bool(exact_sources) and resource.size_bytes > 0,
            "required_secrets": required_secrets,
        },
        "referenced_by": referenced_by,
    }


def format_recipe_catalog(payload: RecipeCatalogResponse) -> RenderableType:
    """Format a compact recipe catalog without machine diagnostics."""

    table = data_table(
        "Family",
        "Status",
        "Tier",
        "Recipe",
        "Name",
        ratio=(1, 1, 1, 3, 3),
    )
    for entry in sorted(payload.recipes, key=lambda item: (item.family.casefold(), item.key)):
        label, kind = _recipe_status(entry)
        table.add_row(
            entry.family,
            status(label, kind),
            _recipe_tier(entry.tags),
            identifier(entry.key),
            entry.name,
        )
    actions = panel(
        "Next",
        Text(
            f"Inspect  {engine_command('recipes', 'show', '<recipe-key>')}\n"
            f"Plan     {engine_command('recipes', 'plan', '<recipe-key>...')}\n"
            f"Install  {engine_command('recipes', 'install', '<recipe-key>...')}",
            style="command",
        ),
        style="green",
    )
    return page(
        f"Recipes · {len(payload.recipes)}", table, *_catalog_errors(payload.errors), actions
    )


def _recipe_tier(tags: list[str]) -> Text:
    normalized = {tag.casefold() for tag in tags}
    if "recommended" in normalized:
        return status("RECOMMENDED", "ok")
    if "fallback" in normalized:
        return status("FALLBACK", "warn")
    if "experimental" in normalized:
        return status("EXPERIMENTAL", "warn")
    if "reference" in normalized or "source-of-truth" in normalized:
        return identifier("REFERENCE")
    if "quality-alternate" in normalized:
        return Text("ALTERNATE")
    return Text("—", style="muted")


def format_recipe_validation(payload: RecipeCatalogResponse) -> RenderableType:
    """Summarize recipe authoring validation while retaining its exit semantics."""

    if not payload.errors:
        return page(
            f"Recipe catalog · {len(payload.recipes)} recipe(s)",
            panel("Status", status("VALID", "ok"), style="green"),
            next_action(engine_command("recipes", "list")),
        )
    return page(
        f"Recipe catalog · {len(payload.errors)} authoring error(s)",
        panel("Status", status("INVALID", "bad"), style="red"),
        *_catalog_errors(payload.errors),
        next_action("Rerun with --json for full validation diagnostics.", label="Details"),
    )


def format_resource_catalog(
    resources: list[ResourceDescriptor], errors: list[str]
) -> RenderableType:
    """Format resources as installed artifacts or declared acquisition targets."""

    table = data_table("Status", "Resource", "Name", "Family", "Format", "Size", "Acquisition")
    for resource in sorted(
        resources, key=lambda item: (item.family, item.name.casefold(), item.id)
    ):
        label = "INSTALLED" if resource.available else "MISSING"
        provisioning = _resource_provisioning(resource)
        table.add_row(
            status(label, "ok" if resource.available else "warn"),
            identifier(resource.id),
            resource.name,
            resource.family,
            resource.format.value,
            format_iec_bytes(resource.size_bytes),
            provisioning,
        )
    return page(
        f"Resources · {len(resources)}",
        table,
        *_catalog_errors(errors),
        next_action(engine_command("resources", "show", "<resource-id>"), label="Inspect"),
    )


def format_deployment_profiles(payload: DeploymentProfileCatalogResponse) -> RenderableType:
    """Format saved, reusable recipe selections."""

    table = data_table("Profile", "Name", "Recipes", "Target", "Source", ratio=(1, 2, 1, 1, 2))
    for profile in payload.profiles:
        table.add_row(
            identifier(profile.key),
            profile.name,
            str(len(profile.recipes)),
            profile.target or "—",
            profile.source_path,
        )
    return page(
        f"Deployment profiles · {len(payload.profiles)} saved recipe selections",
        table,
        *_catalog_errors(payload.errors),
        next_action(engine_command("deployments", "plan", "<profile-key>"), label="Plan"),
    )


def format_recipe_detail(payload: dict[str, Any]) -> RenderableType:
    """Format one recipe with its operational resource closure."""

    recipe = payload["recipe"]
    identity = payload["identity"]
    operation = payload["operation"]
    state = payload["state"]
    overview = key_values(
        (
            ("Recipe", identifier(recipe["key"])),
            ("Name", recipe["name"]),
            ("Source", identity["source_path"]),
            ("Identity", f"{identity['id']} · schema {identity['schema_hash'] or 'unresolved'}"),
            ("Family", recipe["family"]),
            (
                "Operation",
                (
                    f"{operation.get('workflow_kind') or operation.get('recipe_type') or 'unspecified'} "
                    f"via {operation['base_tool']}"
                ),
            ),
        )
    )
    execution = data_table("Setting", "Value", ratio=(1, 3))
    if payload["execution"]:
        for key in sorted(payload["execution"]):
            execution.add_row(key, str(payload["execution"][key]))
    else:
        execution.add_row("Runtime", "default settings")
    state_rows = key_values(
        (
            (
                "Recipe",
                status(
                    "RUNNABLE" if state["recipe_available"] else "UNAVAILABLE",
                    "ok" if state["recipe_available"] else "bad",
                ),
            ),
            (
                "Local",
                status(
                    "YES" if state["locally_runnable"] else "NO",
                    "ok" if state["locally_runnable"] else "warn",
                ),
            ),
            (
                "Automatic provisioning",
                status(
                    "YES" if state["automatic_provisionable"] else "NO",
                    "ok" if state["automatic_provisionable"] else "warn",
                ),
            ),
        )
    )
    resource_table = data_table("Role", "Resource", "Status", ratio=(1, 3, 2))
    for item in payload["required_resources"]:
        resource = item.get("resource")
        if resource is None:
            resource_table.add_row(
                item["role"], identifier(item["reference"]), status("UNRESOLVED", "bad")
            )
            continue
        resource_table.add_row(
            item["role"],
            Text.assemble(resource["name"], "\n", identifier(resource["id"])),
            _resource_plan_status(resource),
        )
    sections: list[RenderableType] = [
        panel("Recipe", overview),
        panel("Execution", execution),
        panel("State", state_rows, style="green" if state["locally_runnable"] else "yellow"),
        panel("Required resources", resource_table),
    ]
    if payload["dynamic_resource_slots"]:
        sections.append(
            panel("Dynamic choices", bullet_list(payload["dynamic_resource_slots"]), style="yellow")
        )
    if state["blockers"]:
        sections.append(
            panel(
                "Blockers",
                bullet_list(_human_reason(item) for item in state["blockers"]),
                style="red",
            )
        )
    if state["automatic_provisionable"] and not state["locally_runnable"]:
        sections.append(next_action(engine_command("recipes", "install", recipe["key"])))
    elif state["locally_runnable"]:
        sections.append(next_action(engine_command("recipes", "validate")))
    else:
        sections.append(next_action(engine_command("recipes", "plan", recipe["key"]), label="Plan"))
    return page(f"Recipe · {recipe['name']}", *sections)


def format_resource_detail(payload: dict[str, Any]) -> RenderableType:
    """Format resource state without serializing raw source internals."""

    resource = payload["resource"]
    state = payload["state"]
    details = key_values(
        (
            ("Resource", identifier(resource["id"])),
            ("Name", resource["name"]),
            ("Artifact path", payload["artifact_path"]),
            (
                "Type",
                f"{resource['kind']} · {resource['family']} · {resource['format']} · {resource['precision']}/{resource['quantization']}",
            ),
            ("Size", format_iec_bytes(resource["size_bytes"])),
            ("Sources", _source_summary(resource["sources"])),
        )
    )
    state_rows = key_values(
        (
            (
                "Installed",
                status(
                    "YES" if state["installed"] else "NO", "ok" if state["installed"] else "warn"
                ),
            ),
            (
                "Automatic provisioning",
                status(
                    "YES" if state["automatic_provisionable"] else "NO",
                    "ok" if state["automatic_provisionable"] else "warn",
                ),
            ),
        )
    )
    sections: list[RenderableType] = [panel("Resource", details), panel("State", state_rows)]
    if state["required_secrets"]:
        sections.append(
            panel(
                "Required environment secrets",
                Text(", ".join(state["required_secrets"]), style="status.warn"),
                style="yellow",
            )
        )
    if payload["referenced_by"]:
        references = data_table("Recipe", "Roles", ratio=(2, 1))
        for item in payload["referenced_by"]:
            references.add_row(identifier(item["recipe_key"]), ", ".join(item["roles"]))
        sections.append(panel("Required by recipes", references))
    recipe_key = (
        payload["referenced_by"][0]["recipe_key"] if payload["referenced_by"] else "<recipe-key>"
    )
    sections.append(next_action(engine_command("recipes", "plan", recipe_key)))
    return page(f"Resource · {resource['name']}", *sections)


def format_recipe_install(payload: Any, recipe_keys: list[str]) -> RenderableType:
    """Format a completed installation without embedding a full lock on stdout."""

    plan = payload.deployment_plan
    changes = data_table("Result", "Resources", ratio=(1, 4))
    if payload.installed_resource_ids:
        changes.add_row("Installed", "\n".join(payload.installed_resource_ids))
    if payload.skipped_resource_ids:
        changes.add_row("Already installed", "\n".join(payload.skipped_resource_ids))
    if not payload.installed_resource_ids and not payload.skipped_resource_ids:
        changes.add_row("Result", "No resource changes were required.")
    state = key_values(
        (
            (
                "Recipe selection runnable",
                status(
                    "YES" if plan.locally_runnable else "NO",
                    "ok" if plan.locally_runnable else "warn",
                ),
            ),
        )
    )
    return page(
        f"Recipe installation · {', '.join(recipe_keys)}",
        panel("Resource changes", changes),
        panel("State", state),
        next_action(engine_command("recipes", "list")),
    )


def format_deployment_install(payload: Any, profile_key: str) -> RenderableType:
    """Format a completed saved-profile installation."""

    plan = payload.deployment_plan
    changes = data_table("Result", "Resources", ratio=(1, 4))
    if payload.installed_resource_ids:
        changes.add_row("Installed", "\n".join(payload.installed_resource_ids))
    if payload.skipped_resource_ids:
        changes.add_row("Already installed", "\n".join(payload.skipped_resource_ids))
    if not payload.installed_resource_ids and not payload.skipped_resource_ids:
        changes.add_row("Result", "No resource changes were required.")
    return page(
        f"Deployment profile installation · {profile_key}",
        panel("Resource changes", changes),
        panel(
            "State",
            key_values(
                (
                    (
                        "Profile runnable",
                        status(
                            "YES" if plan.locally_runnable else "NO",
                            "ok" if plan.locally_runnable else "warn",
                        ),
                    ),
                )
            ),
        ),
        next_action(engine_command("deployments", "profiles")),
    )


def concise_cli_error(value: str) -> str:
    """Return an actionable parser error without nested filesystem noise."""

    return _human_reason(value)


def _recipe_resource_payload(
    role: str,
    reference: str,
    registry: ToolRegistry,
    planned_resources: dict[str, Any],
) -> dict[str, Any]:
    try:
        resource = registry.resources.resolve(reference, include_components=True)
    except (KeyError, ValueError) as exc:
        return {"role": role, "reference": reference, "error": str(exc)}
    planned = planned_resources.get(resource.id)
    return {
        "role": role,
        "reference": reference,
        "resource": planned.model_dump(mode="json") if planned else None,
    }


def _resource_roles(entry: Any, resource: ResourceDescriptor, inventory: Any) -> list[str]:
    """Resolve recipe references exactly as the catalog does before comparing IDs."""

    def matches(reference: str | None) -> bool:
        if reference is None:
            return False
        try:
            resolved = inventory.resolve(
                reference,
                kind=resource.kind,
                family=resource.family,
                include_components=True,
            )
        except (KeyError, ValueError):
            return False
        return resolved.id == resource.id

    roles: list[str] = []
    if matches(entry.model_resource):
        roles.append("model")
    roles.extend(
        role for role, reference in sorted(entry.recipe_resources.items()) if matches(reference)
    )
    if any(matches(reference) for reference in entry.fixed_resources) and not roles:
        roles.append("required_resource")
    return roles


def _plan_blockers(plan: DeploymentPlan) -> list[str]:
    blockers: list[str] = []
    if plan.missing_resources:
        blockers.append("missing resource declarations: " + ", ".join(plan.missing_resources))
    for recipe in plan.recipes:
        if not recipe.available:
            blockers.append(recipe.unavailable_reason or f"recipe {recipe.key} is unavailable")
        if recipe.dynamic_resource_slots:
            blockers.append(
                f"dynamic resource choices: {recipe.key} ({', '.join(recipe.dynamic_resource_slots)})"
            )
    for resource in plan.resources:
        if not resource.installed:
            if resource.provisionable:
                continue
            blockers.append(
                f"{resource.id} requires manual staging or a precise source declaration"
            )
    return list(dict.fromkeys(blockers))


def _recipe_status(entry: Any) -> tuple[str, str]:
    if entry.available:
        return "RUNNABLE", "ok"
    reason = (entry.unavailable_reason or "").casefold()
    if any(token in reason for token in ("not installed", "missing", "incomplete", "unavailable")):
        return "MISSING", "warn"
    return "BLOCKED", "bad"


def _resource_provisioning(resource: ResourceDescriptor) -> str:
    if resource.available:
        return "installed"
    if any(source.is_exact() for source in resource.sources) and resource.size_bytes > 0:
        return "automatic install"
    return "manual staging"


def _resource_plan_status(resource: dict[str, Any]) -> str:
    if resource["installed"]:
        return "installed"
    if resource["provisionable"]:
        return "missing; automatic install available"
    return "missing; manual staging required"


def _source_summary(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "no declared remote source"
    labels = []
    for source in sources:
        source_type = source["type"]
        if source_type == "huggingface":
            labels.append(f"Hugging Face {source.get('repo_id') or 'source'}")
        elif source_type == "civitai":
            labels.append("Civitai exact file" if _source_is_exact(source) else "Civitai source")
        else:
            labels.append("manual")
    return "; ".join(labels)


def _source_is_exact(source: dict[str, Any]) -> bool:
    return bool(
        (source.get("repo_id") and source.get("revision"))
        or (source.get("model_version_id") and source.get("file_id"))
        or (source.get("url") and source.get("sha256"))
    )


def _catalog_errors(errors: list[str]) -> list[RenderableType]:
    if not errors:
        return []
    return [
        panel(
            f"Catalog authoring errors · {len(errors)}",
            bullet_list(_human_reason(error) for error in errors),
            style="red",
        )
    ]


def _human_reason(value: str) -> str:
    compact = " ".join(value.split())
    lowered = compact.casefold()
    if any(token in lowered for token in ("winerror", "oserror", "errno")):
        return "required artifact is missing or incomplete; repair or reinstall it"
    compact = re.sub(r"(?i)(token|secret|authorization)=[^\s&]+", r"\1=<redacted>", compact)
    return compact if len(compact) <= 180 else compact[:179].rstrip() + "…"
