"""Deterministic human-readable summaries for deployment plans."""

from __future__ import annotations

from .recipes import DeploymentPlan, DeploymentRecipePlan, DeploymentResourcePlan


def format_iec_bytes(size_bytes: int) -> str:
    """Format a non-negative byte count with IEC units for CLI output."""

    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(size_bytes)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{size_bytes} B"
    if value >= 100 or value.is_integer():
        rendered = f"{value:.0f}"
    else:
        rendered = f"{value:.1f}"
    return f"{rendered} {units[unit_index]}"


def format_deployment_plan(plan: DeploymentPlan) -> str:
    """Return a concise, deterministic summary without filesystem error noise."""

    return _format_plan(
        plan,
        heading="Deployment plan",
        install_command=f"uv run latentslate-engine deployments install {plan.profile_key}",
    )


def format_recipe_selection_plan(plan: DeploymentPlan) -> str:
    """Format an explicit recipe selection over the same exact closure."""

    recipe_keys = " ".join(recipe.key for recipe in plan.recipes)
    return _format_plan(
        plan,
        heading="Recipe plan",
        install_command=f"uv run latentslate-engine recipes install {recipe_keys}",
        display_name=", ".join(recipe.key for recipe in plan.recipes),
        show_profile_key=False,
    )


def _format_plan(
    plan: DeploymentPlan,
    *,
    heading: str,
    install_command: str,
    display_name: str | None = None,
    show_profile_key: bool = True,
) -> str:
    """Render an exact closure with a caller-specific selection label."""

    profile = display_name or plan.profile_name
    if show_profile_key:
        profile = f"{profile} ({plan.profile_key})"
    if plan.target:
        profile = f"{profile}; target: {plan.target}"
    lines = [f"{heading}: {profile}", "", f"Recipes ({len(plan.recipes)}):"]
    resources = {resource.id: resource for resource in plan.resources}
    lines.extend(_format_recipe(recipe, resources) for recipe in plan.recipes)
    lines.extend(("", f"Resources ({len(plan.resources)} unique):"))
    lines.extend(_format_resource(resource) for resource in plan.resources)
    lines.extend(
        (
            "",
            (
                "Footprint: "
                f"total {format_iec_bytes(plan.total_bytes)}; "
                f"incremental download {format_iec_bytes(plan.incremental_bytes)}"
            ),
            f"Local runnable: {_yes_no(plan.locally_runnable)}",
            (f"Automatic/remote provisioning: {_yes_no(plan.remote_provisionable)}"),
            "Required secrets: " + (", ".join(plan.required_secrets) or "none"),
        )
    )

    blockers = _blockers(plan)
    if blockers:
        lines.append("")
        lines.append("Blockers:")
        lines.extend(f"  - {blocker}" for blocker in blockers)

    warnings = _additional_warnings(plan)
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)

    lines.append("")
    if plan.remote_provisionable and not plan.locally_runnable:
        lines.append(f"Next: {install_command}")
    elif plan.locally_runnable:
        lines.append("Next: uv run latentslate-engine recipes validate")
    else:
        lines.append("Details: rerun with --json for the full structured diagnostics.")
    return "\n".join(lines)


def _format_recipe(
    recipe: DeploymentRecipePlan,
    resources: dict[str, DeploymentResourcePlan],
) -> str:
    if recipe.available:
        status = "runnable"
    elif any(
        resource_id in resources and not resources[resource_id].installed
        for resource_id in recipe.fixed_resources
    ):
        status = "not runnable; required resources are missing or incomplete"
    else:
        status = f"not runnable; {_recipe_unavailable_status(recipe)}"
    return f"  - {recipe.key}: {status}"


def _format_resource(resource: DeploymentResourcePlan) -> str:
    if resource.installed:
        status = "installed"
        if not resource.provisionable:
            status += _nonprovisionable_resource_suffix(resource)
    elif resource.provisionable:
        status = "missing; automatic install available"
    elif _has_exact_source(resource):
        status = "missing; positive size declaration required before automatic install"
    else:
        status = "missing; manual acquisition or a source declaration required"
    return f"  - {resource.name} ({resource.id}): {status}; {format_iec_bytes(resource.size_bytes)}"


def _recipe_unavailable_status(recipe: DeploymentRecipePlan) -> str:
    detail = recipe.unavailable_reason or ""
    reason = detail.casefold()
    if "not present in the recipe catalog" in reason:
        return "not present in the recipe catalog"
    if any(token in reason for token in ("winerror", "oserror", "errno")):
        return "required artifact could not be inspected; repair or reinstall it"
    if any(token in reason for token in ("not installed", "incomplete", "unavailable")):
        return "required artifact is missing or incomplete"
    if detail:
        return _truncate_warning(detail)
    return "check the resource closure or rerun with --json for validation details"


def _blockers(plan: DeploymentPlan) -> list[str]:
    blockers: list[str] = []
    if plan.missing_resources:
        blockers.append("Missing resource declarations: " + ", ".join(plan.missing_resources))

    unresolved_recipes = [
        recipe.key for recipe in plan.recipes if not recipe.id or not recipe.schema_hash
    ]
    if unresolved_recipes:
        blockers.append(
            "Recipes without a complete lock identity: " + ", ".join(unresolved_recipes)
        )

    dynamic_slots = [
        f"{recipe.key} ({', '.join(recipe.dynamic_resource_slots)})"
        for recipe in plan.recipes
        if recipe.dynamic_resource_slots
    ]
    if dynamic_slots:
        blockers.append(
            "Dynamic resource slots prevent automatic provisioning: " + ", ".join(dynamic_slots)
        )

    missing_sizes = [
        resource.id
        for resource in plan.resources
        if not resource.provisionable and resource.size_bytes <= 0 and _has_exact_source(resource)
    ]
    if missing_sizes:
        blockers.append("Resources without a positive declared size: " + ", ".join(missing_sizes))

    source_less = [
        resource.id
        for resource in plan.resources
        if not resource.provisionable and not _has_exact_source(resource)
    ]
    if source_less:
        blockers.append(
            "Resources without an immutable automatic source: " + ", ".join(source_less)
        )

    return blockers


def _additional_warnings(plan: DeploymentPlan) -> list[str]:
    known_prefixes = (
        "recipe ",
        "resource ",
    )
    warnings: list[str] = []
    for warning in plan.warnings:
        normalized = warning.casefold()
        if normalized.startswith(known_prefixes):
            continue
        if any(token in normalized for token in ("winerror", "oserror", "errno")):
            warnings.append(
                "Artifact inspection failed; repair or reinstall the affected resource."
            )
        else:
            warnings.append(_truncate_warning(warning))
    return warnings


def _truncate_warning(warning: str, *, limit: int = 160) -> str:
    compact = " ".join(warning.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _has_exact_source(resource: DeploymentResourcePlan) -> bool:
    return any(source.is_exact() for source in resource.sources)


def _nonprovisionable_resource_suffix(resource: DeploymentResourcePlan) -> str:
    if _has_exact_source(resource) and resource.size_bytes <= 0:
        return "; positive size declaration required for remote provisioning"
    return "; no immutable remote source"
