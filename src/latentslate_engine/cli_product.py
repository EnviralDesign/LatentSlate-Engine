"""Human-first, deterministic views for the recipe and resource CLI.

These helpers only consume already-discovered catalog data.  They deliberately
avoid importing a model runtime so listing, planning, and inspection work in a
protocol-only installation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from . import __version__
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


def format_recipe_catalog(payload: RecipeCatalogResponse) -> str:
    """Format a compact recipe catalog without machine diagnostics."""

    lines = [f"Recipes ({len(payload.recipes)}):"]
    families = sorted({entry.family for entry in payload.recipes}, key=str.casefold)
    for family in families:
        entries = [entry for entry in payload.recipes if entry.family == family]
        lines.append(f"  {family} ({len(entries)}):")
        for entry in entries:
            lines.append(f"    {_recipe_status(entry):<10} {entry.key} — {entry.name}")
    lines.extend(_catalog_errors(payload.errors))
    lines.extend(
        (
            "",
            "Inspect: uv run latentslate-engine recipes show <recipe-key>",
            "Plan:    uv run latentslate-engine recipes plan <recipe-key>...",
        )
    )
    return "\n".join(lines)


def format_recipe_validation(payload: RecipeCatalogResponse) -> str:
    """Summarize recipe authoring validation while retaining its exit semantics."""

    if not payload.errors:
        return (
            f"Recipe catalog valid: {len(payload.recipes)} recipe(s).\n"
            "Next: uv run latentslate-engine recipes list"
        )
    lines = [f"Recipe catalog invalid: {len(payload.errors)} authoring error(s)."]
    lines.extend(_catalog_errors(payload.errors))
    lines.append("Details: rerun with --json for full validation diagnostics.")
    return "\n".join(lines)


def format_resource_catalog(resources: list[ResourceDescriptor], errors: list[str]) -> str:
    """Format resources as installed artifacts or declared acquisition targets."""

    lines = [f"Resources ({len(resources)}):"]
    for resource in sorted(
        resources, key=lambda item: (item.family, item.name.casefold(), item.id)
    ):
        status = "Installed" if resource.available else "Missing"
        provisioning = _resource_provisioning(resource)
        lines.append(
            f"  {status:<10} {resource.id} — {resource.name} "
            f"({resource.family}; {resource.format.value}; {format_iec_bytes(resource.size_bytes)}; "
            f"{provisioning})"
        )
    lines.extend(_catalog_errors(errors))
    lines.extend(
        (
            "",
            "Inspect: uv run latentslate-engine resources show <resource-id>",
        )
    )
    return "\n".join(lines)


def format_deployment_profiles(payload: DeploymentProfileCatalogResponse) -> str:
    """Format saved, reusable recipe selections."""

    lines = [f"Deployment profiles ({len(payload.profiles)} saved recipe selections):"]
    for profile in payload.profiles:
        target = f"; target {profile.target}" if profile.target else ""
        lines.append(
            f"  {profile.key} — {profile.name} "
            f"({len(profile.recipes)} recipe(s){target}; {profile.source_path})"
        )
    lines.extend(_catalog_errors(payload.errors))
    lines.extend(
        (
            "",
            "Plan: uv run latentslate-engine deployments plan <profile-key>",
        )
    )
    return "\n".join(lines)


def format_recipe_detail(payload: dict[str, Any]) -> str:
    """Format one recipe with its operational resource closure."""

    recipe = payload["recipe"]
    identity = payload["identity"]
    operation = payload["operation"]
    state = payload["state"]
    lines = [
        f"Recipe: {recipe['name']} ({recipe['key']})",
        f"Source: {identity['source_path']}",
        f"Identity: {identity['id']} · schema {identity['schema_hash'] or 'unresolved'}",
        f"Family: {recipe['family']}",
        (
            "Operation: "
            f"{operation.get('workflow_kind') or operation.get('recipe_type') or 'unspecified'} "
            f"via {operation['base_tool']}"
        ),
        "Execution: " + _settings_line(payload["execution"]),
        (
            "State: "
            f"runnable {_yes_no(state['locally_runnable'])}; "
            f"automatic provisioning {_yes_no(state['automatic_provisionable'])}"
        ),
        "Required resources:",
    ]
    for item in payload["required_resources"]:
        resource = item.get("resource")
        if resource is None:
            lines.append(f"  {item['role']}: {item['reference']} — unresolved")
            continue
        lines.append(
            f"  {item['role']}: {resource['name']} ({resource['id']}) — "
            f"{_resource_plan_status(resource)}"
        )
    if payload["dynamic_resource_slots"]:
        lines.append("Dynamic choices: " + ", ".join(payload["dynamic_resource_slots"]))
    if state["blockers"]:
        lines.append("Blockers:")
        lines.extend(f"  - {_human_reason(blocker)}" for blocker in state["blockers"])
    lines.append("")
    if state["automatic_provisionable"] and not state["locally_runnable"]:
        lines.append(f"Next: uv run latentslate-engine recipes install {recipe['key']}")
    elif state["locally_runnable"]:
        lines.append("Next: uv run latentslate-engine recipes validate")
    else:
        lines.append(f"Plan: uv run latentslate-engine recipes plan {recipe['key']}")
    return "\n".join(lines)


def format_resource_detail(payload: dict[str, Any]) -> str:
    """Format resource state without serializing raw source internals."""

    resource = payload["resource"]
    state = payload["state"]
    lines = [
        f"Resource: {resource['name']} ({resource['id']})",
        f"Artifact path: {payload['artifact_path']}",
        (
            f"Type: {resource['kind']} · {resource['family']} · {resource['format']} · "
            f"{resource['precision']}/{resource['quantization']}"
        ),
        f"Size: {format_iec_bytes(resource['size_bytes'])}",
        (
            "State: "
            f"installed {_yes_no(state['installed'])}; "
            f"automatic provisioning {_yes_no(state['automatic_provisionable'])}"
        ),
        "Sources: " + _source_summary(resource["sources"]),
    ]
    if state["required_secrets"]:
        lines.append("Required environment secrets: " + ", ".join(state["required_secrets"]))
    if payload["referenced_by"]:
        lines.append("Required by recipes:")
        for item in payload["referenced_by"]:
            lines.append(f"  {item['recipe_key']}: {', '.join(item['roles'])}")
    lines.extend(
        (
            "",
            f"Next: uv run latentslate-engine recipes plan {' '.join(item['recipe_key'] for item in payload['referenced_by'][:1]) or '<recipe-key>'}",
        )
    )
    return "\n".join(lines)


def format_recipe_install(payload: Any, recipe_keys: list[str]) -> str:
    """Format a completed installation without embedding a full lock on stdout."""

    plan = payload.deployment_plan
    lines = [f"Recipe installation: {', '.join(recipe_keys)}"]
    if payload.installed_resource_ids:
        lines.append("Installed resources:")
        lines.extend(f"  - {resource_id}" for resource_id in payload.installed_resource_ids)
    if payload.skipped_resource_ids:
        lines.append("Already installed:")
        lines.extend(f"  - {resource_id}" for resource_id in payload.skipped_resource_ids)
    if not payload.installed_resource_ids and not payload.skipped_resource_ids:
        lines.append("No resource changes were required.")
    lines.extend(
        (
            "",
            f"Recipe selection runnable: {_yes_no(plan.locally_runnable)}",
            "Next: uv run latentslate-engine recipes list",
        )
    )
    return "\n".join(lines)


def format_deployment_install(payload: Any, profile_key: str) -> str:
    """Format a completed saved-profile installation."""

    return format_recipe_install(payload, [profile_key]).replace(
        "Recipe installation:", "Deployment profile installation:", 1
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


def _recipe_status(entry: Any) -> str:
    if entry.available:
        return "Runnable"
    reason = (entry.unavailable_reason or "").casefold()
    if any(token in reason for token in ("not installed", "missing", "incomplete", "unavailable")):
        return "Missing"
    return "Blocked"


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


def _catalog_errors(errors: list[str]) -> list[str]:
    if not errors:
        return []
    return [
        "",
        f"Catalog authoring errors: {len(errors)} (rerun with --json for full diagnostics).",
    ]


def _settings_line(settings: dict[str, Any]) -> str:
    if not settings:
        return "default runtime settings"
    return ", ".join(f"{key}={settings[key]}" for key in sorted(settings))


def _human_reason(value: str) -> str:
    compact = " ".join(value.split())
    lowered = compact.casefold()
    if any(token in lowered for token in ("winerror", "oserror", "errno")):
        return "required artifact is missing or incomplete; repair or reinstall it"
    compact = re.sub(r"(?i)(token|secret|authorization)=[^\s&]+", r"\1=<redacted>", compact)
    return compact if len(compact) <= 180 else compact[:179].rstrip() + "…"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
