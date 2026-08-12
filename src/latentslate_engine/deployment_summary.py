"""Deterministic human-readable summaries for deployment plans."""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text

from .cli_presentation import (
    bullet_list,
    data_table,
    identifier,
    key_values,
    next_action,
    page,
    panel,
    status,
)
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


def format_deployment_plan(plan: DeploymentPlan) -> RenderableType:
    """Return a concise, deterministic summary without filesystem error noise."""

    return _format_plan(
        plan,
        heading="Deployment plan",
        install_command=f"uv run latentslate-engine deployments install {plan.profile_key}",
    )


def format_recipe_selection_plan(plan: DeploymentPlan) -> RenderableType:
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
) -> RenderableType:
    """Render an exact closure with a caller-specific selection label."""

    profile = display_name or plan.profile_name
    overview_rows: list[tuple[str, str | Text]] = [("Selection", profile)]
    if show_profile_key:
        overview_rows.append(("Profile", identifier(plan.profile_key)))
    if plan.target:
        overview_rows.append(("Target", plan.target))
    overview_rows.extend(
        (
            ("Total footprint", format_iec_bytes(plan.total_bytes)),
            ("Incremental download", format_iec_bytes(plan.incremental_bytes)),
            (
                "Local runnable",
                status(
                    "YES" if plan.locally_runnable else "NO",
                    "ok" if plan.locally_runnable else "warn",
                ),
            ),
            (
                "Automatic provisioning",
                status(
                    "YES" if plan.remote_provisionable else "NO",
                    "ok" if plan.remote_provisionable else "warn",
                ),
            ),
            ("Required secrets", ", ".join(plan.required_secrets) or "none"),
        )
    )
    recipe_table = data_table("Recipe", "Status", ratio=(2, 3))
    resources = {resource.id: resource for resource in plan.resources}
    for recipe in plan.recipes:
        message, kind = _format_recipe(recipe, resources)
        recipe_table.add_row(identifier(recipe.key), status(message, kind))

    resource_table = data_table("Resource", "Status", "Size", ratio=(3, 3, 1))
    for resource in plan.resources:
        message, kind = _format_resource(resource)
        resource_table.add_row(
            Text.assemble(resource.name, "\n", identifier(resource.id)),
            status(message, kind),
            format_iec_bytes(resource.size_bytes),
        )
    sections: list[RenderableType] = [
        panel("Overview", key_values(overview_rows)),
        panel(f"Recipes · {len(plan.recipes)}", recipe_table),
        panel(f"Resources · {len(plan.resources)} unique", resource_table),
    ]

    blockers = _blockers(plan)
    if blockers:
        sections.append(panel("Blockers", bullet_list(blockers), style="red"))

    warnings = _additional_warnings(plan)
    if warnings:
        sections.append(panel("Warnings", bullet_list(warnings), style="yellow"))

    if plan.remote_provisionable and not plan.locally_runnable:
        sections.append(next_action(install_command))
    elif plan.locally_runnable:
        sections.append(next_action("uv run latentslate-engine recipes validate"))
    else:
        sections.append(
            next_action("Rerun with --json for the full structured diagnostics.", label="Details")
        )
    return page(heading, *sections)


def _format_recipe(
    recipe: DeploymentRecipePlan,
    resources: dict[str, DeploymentResourcePlan],
) -> tuple[str, str]:
    if recipe.available:
        return "RUNNABLE", "ok"
    elif any(
        resource_id in resources and not resources[resource_id].installed
        for resource_id in recipe.fixed_resources
    ):
        if recipe.unavailable_reason:
            return f"MISSING RESOURCES · {_recipe_unavailable_status(recipe)}", "warn"
        return "MISSING RESOURCES", "warn"
    return _recipe_unavailable_status(recipe).upper(), "bad"


def _format_resource(resource: DeploymentResourcePlan) -> tuple[str, str]:
    if resource.installed:
        label = "INSTALLED"
        if not resource.provisionable:
            label += _nonprovisionable_resource_suffix(resource).upper()
        return label, "ok"
    elif resource.provisionable:
        return "MISSING · AUTO INSTALL", "warn"
    elif _has_exact_source(resource):
        return "MISSING · SIZE REQUIRED", "bad"
    else:
        return "MISSING · MANUAL STAGING", "bad"


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


def _has_exact_source(resource: DeploymentResourcePlan) -> bool:
    return any(source.is_exact() for source in resource.sources)


def _nonprovisionable_resource_suffix(resource: DeploymentResourcePlan) -> str:
    if _has_exact_source(resource) and resource.size_bytes <= 0:
        return "; positive size declaration required for remote provisioning"
    return "; no immutable remote source"
