from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .config import Settings
from .resources import ResourceDescriptor, ResourceSource
from .variants import VariantCatalogEntry

if TYPE_CHECKING:
    from .tools import ToolRegistry


class RecipeCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    recipes: list[VariantCatalogEntry]
    errors: list[str] = Field(default_factory=list)
    search_paths: list[str] = Field(default_factory=list)


class DeploymentProfileDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    name: str = Field(min_length=1)
    recipes: list[str] = Field(min_length=1)
    target: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_recipes(self) -> DeploymentProfileDefinition:
        if len(self.recipes) != len(set(self.recipes)):
            raise ValueError("deployment profile recipe keys must be unique")
        return self


class DeploymentProfileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    recipes: list[str]
    target: str | None = None
    notes: str | None = None
    source_path: str


class DeploymentProfileCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: list[DeploymentProfileEntry]
    errors: list[str] = Field(default_factory=list)
    search_paths: list[str] = Field(default_factory=list)


class DeploymentRecipePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    id: str
    schema_hash: str | None = None
    available: bool
    unavailable_reason: str | None = None
    fixed_resources: list[str] = Field(default_factory=list)
    dynamic_resource_slots: list[str] = Field(default_factory=list)


class DeploymentResourcePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    kind: str
    name: str
    format: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    installed: bool
    provisionable: bool
    required_secrets: list[str] = Field(default_factory=list)
    sources: list[ResourceSource] = Field(default_factory=list)


class DeploymentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    profile_key: str
    profile_name: str
    target: str | None = None
    recipes: list[DeploymentRecipePlan]
    resources: list[DeploymentResourcePlan]
    total_bytes: int = Field(ge=0)
    incremental_bytes: int = Field(ge=0)
    locally_runnable: bool
    remote_provisionable: bool
    required_secrets: list[str] = Field(default_factory=list)
    missing_resources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DeploymentLockRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    id: str
    schema_hash: str | None = None


class DeploymentLockResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    kind: str
    format: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    installed: bool
    sources: list[ResourceSource] = Field(default_factory=list)


class DeploymentLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    engine_version: str
    generated_at: datetime
    profile_key: str
    target: str | None = None
    recipes: list[DeploymentLockRecipe]
    resources: list[DeploymentLockResource]
    required_secrets: list[str] = Field(default_factory=list)
    total_bytes: int = Field(ge=0)
    incremental_bytes: int = Field(ge=0)
    remote_provisionable: bool


@dataclass(slots=True)
class DeploymentProfileLoadResult:
    definitions: dict[str, DeploymentProfileDefinition]
    entries: list[DeploymentProfileEntry]
    errors: list[str]


def recipe_catalog(settings: Settings, registry: ToolRegistry) -> RecipeCatalogResponse:
    return RecipeCatalogResponse(
        engine_version=__version__,
        recipes=registry.variants,
        errors=registry.variant_errors,
        search_paths=[f"{label}:{path}" for label, path in settings.recipe_catalog_roots()],
    )


def load_deployment_profiles(settings: Settings) -> DeploymentProfileLoadResult:
    definitions: dict[str, DeploymentProfileDefinition] = {}
    entries: list[DeploymentProfileEntry] = []
    errors: list[str] = []
    seen_paths: set[Path] = set()

    for label, root in settings.deployment_profile_roots():
        if not root.exists():
            continue
        resolved_root = root.resolve()
        for path in sorted(resolved_root.rglob("*.toml")):
            if path.name.startswith("."):
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                relative = resolved.relative_to(resolved_root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
                definition = DeploymentProfileDefinition.model_validate(raw.get("profile", raw))
                if definition.key in definitions:
                    raise ValueError(f"duplicate deployment profile key {definition.key!r}")
                definitions[definition.key] = definition
                entries.append(
                    DeploymentProfileEntry(
                        key=definition.key,
                        name=definition.name,
                        recipes=definition.recipes,
                        target=definition.target,
                        notes=definition.notes,
                        source_path=(Path(label) / relative).as_posix(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report every authoring error
                errors.append(f"{resolved}: {exc}")

    entries.sort(key=lambda entry: (entry.name.casefold(), entry.key))
    return DeploymentProfileLoadResult(definitions, entries, errors)


def deployment_profile_catalog(settings: Settings) -> DeploymentProfileCatalogResponse:
    loaded = load_deployment_profiles(settings)
    return DeploymentProfileCatalogResponse(
        profiles=loaded.entries,
        errors=loaded.errors,
        search_paths=[f"{label}:{path}" for label, path in settings.deployment_profile_roots()],
    )


def build_deployment_plan(
    settings: Settings,
    registry: ToolRegistry,
    profile_key: str,
) -> DeploymentPlan:
    profiles = load_deployment_profiles(settings)
    if profiles.errors:
        raise ValueError("deployment profile catalog has errors: " + "; ".join(profiles.errors))
    try:
        profile = profiles.definitions[profile_key]
    except KeyError as exc:
        raise KeyError(f"Unknown deployment profile {profile_key!r}") from exc

    entries = {entry.key: entry for entry in registry.variants}
    descriptors = {descriptor.key: descriptor for descriptor in registry.descriptors()}
    recipe_plans: list[DeploymentRecipePlan] = []
    resources: dict[str, DeploymentResourcePlan] = {}
    missing_resources: set[str] = set()
    warnings: list[str] = []

    for recipe_key in profile.recipes:
        try:
            entry = entries[recipe_key]
        except KeyError:
            warnings.append(f"recipe {recipe_key!r} is not present in the recipe catalog")
            recipe_plans.append(
                DeploymentRecipePlan(
                    key=recipe_key,
                    id="",
                    available=False,
                    unavailable_reason="recipe is not present in the catalog",
                )
            )
            continue

        descriptor = descriptors.get(recipe_key)
        recipe_plan = DeploymentRecipePlan(
            key=entry.key,
            id=str(entry.id),
            schema_hash=descriptor.schema_hash if descriptor else None,
            available=entry.available,
            unavailable_reason=entry.unavailable_reason,
            fixed_resources=entry.fixed_resources,
            dynamic_resource_slots=entry.dynamic_resource_slots,
        )
        recipe_plans.append(recipe_plan)
        if not recipe_plan.id or not recipe_plan.schema_hash:
            warnings.append(
                f"recipe {entry.key!r} does not resolve to a lockable id/schema"
            )
        if not entry.available:
            warnings.append(
                f"recipe {entry.key!r} is unavailable: "
                f"{entry.unavailable_reason or 'unknown reason'}"
            )
        if entry.dynamic_resource_slots:
            warnings.append(
                f"recipe {entry.key!r} has dynamic resource slots that cannot be locked: "
                + ", ".join(entry.dynamic_resource_slots)
            )

        for reference in entry.fixed_resources:
            try:
                resource = _resolve_resource(registry, reference)
            except (KeyError, ValueError) as exc:
                missing_resources.add(reference)
                warnings.append(f"recipe {entry.key!r} resource {reference!r}: {exc}")
                continue
            resources.setdefault(resource.id, _resource_plan(registry, resource))

    resource_list = sorted(resources.values(), key=lambda item: item.id)
    total_bytes = sum(resource.size_bytes for resource in resource_list)
    incremental_bytes = sum(
        resource.size_bytes for resource in resource_list if not resource.installed
    )
    required_secrets = sorted(
        {
            secret
            for resource in resource_list
            for secret in resource.required_secrets
        }
    )
    locally_runnable = (
        not missing_resources
        and all(recipe.available for recipe in recipe_plans)
        and all(resource.installed for resource in resource_list)
    )
    recipe_identity_complete = all(
        bool(recipe.id) and bool(recipe.schema_hash) for recipe in recipe_plans
    )
    remote_provisionable = (
        not missing_resources
        and recipe_identity_complete
        and all(not recipe.dynamic_resource_slots for recipe in recipe_plans)
        and all(resource.provisionable for resource in resource_list)
    )
    for resource in resource_list:
        has_exact_source = any(source.is_exact() for source in resource.sources)
        if not resource.installed and has_exact_source and resource.size_bytes <= 0:
            warnings.append(
                f"resource {resource.id!r} is missing and must declare positive size_bytes "
                "for remote provisioning"
            )
        elif resource.installed and not resource.provisionable:
            warnings.append(
                f"resource {resource.id!r} is runnable locally but has no immutable remote source"
            )
        elif not resource.installed and not resource.provisionable:
            warnings.append(
                f"resource {resource.id!r} is missing and has no immutable remote source"
            )

    return DeploymentPlan(
        engine_version=__version__,
        profile_key=profile.key,
        profile_name=profile.name,
        target=profile.target,
        recipes=recipe_plans,
        resources=resource_list,
        total_bytes=total_bytes,
        incremental_bytes=incremental_bytes,
        locally_runnable=locally_runnable,
        remote_provisionable=remote_provisionable,
        required_secrets=required_secrets,
        missing_resources=sorted(missing_resources),
        warnings=warnings,
    )


def build_deployment_lock(
    settings: Settings,
    registry: ToolRegistry,
    profile_key: str,
    *,
    generated_at: datetime | None = None,
) -> DeploymentLock:
    plan = build_deployment_plan(settings, registry, profile_key)
    if not plan.remote_provisionable:
        reasons: list[str] = []
        unresolved_recipes = [
            recipe.key for recipe in plan.recipes if not recipe.id or not recipe.schema_hash
        ]
        if unresolved_recipes:
            reasons.append("unresolved recipes: " + ", ".join(unresolved_recipes))
        if plan.missing_resources:
            reasons.append("missing resource declarations: " + ", ".join(plan.missing_resources))
        dynamic_slots = [
            f"{recipe.key} ({', '.join(recipe.dynamic_resource_slots)})"
            for recipe in plan.recipes
            if recipe.dynamic_resource_slots
        ]
        if dynamic_slots:
            reasons.append("dynamic resource slots: " + ", ".join(dynamic_slots))
        missing_sizes = [
            resource.id
            for resource in plan.resources
            if not resource.installed
            and resource.size_bytes <= 0
            and any(source.is_exact() for source in resource.sources)
        ]
        if missing_sizes:
            reasons.append(
                "resources without positive declared size: " + ", ".join(missing_sizes)
            )
        nonprovisionable = [
            resource.id
            for resource in plan.resources
            if not resource.provisionable
            and resource.id not in missing_sizes
        ]
        if nonprovisionable:
            reasons.append("resources without immutable sources: " + ", ".join(nonprovisionable))
        detail = "; ".join(reasons) or "deployment plan is not exactly provisionable"
        raise ValueError(f"deployment lock cannot be generated: {detail}")

    return DeploymentLock(
        engine_version=plan.engine_version,
        generated_at=generated_at or datetime.now(UTC),
        profile_key=plan.profile_key,
        target=plan.target,
        recipes=[
            DeploymentLockRecipe(
                key=recipe.key,
                id=recipe.id,
                schema_hash=recipe.schema_hash,
            )
            for recipe in plan.recipes
        ],
        resources=[
            DeploymentLockResource(
                id=resource.id,
                family=resource.family,
                kind=resource.kind,
                format=resource.format,
                relative_path=resource.relative_path,
                size_bytes=resource.size_bytes,
                installed=resource.installed,
                sources=[source for source in resource.sources if source.is_exact()],
            )
            for resource in plan.resources
        ],
        required_secrets=plan.required_secrets,
        total_bytes=plan.total_bytes,
        incremental_bytes=plan.incremental_bytes,
        remote_provisionable=True,
    )


def _resolve_resource(registry: ToolRegistry, reference: str) -> ResourceDescriptor:
    by_id = registry.resources.by_id()
    if reference in by_id:
        return by_id[reference]
    return registry.resources.resolve(reference, include_components=True)


def _resource_plan(
    registry: ToolRegistry,
    resource: ResourceDescriptor,
) -> DeploymentResourcePlan:
    installed = registry.resources.is_installed(resource.id)
    exact_sources = [source for source in resource.sources if source.is_exact()]
    required_secrets = sorted(
        {
            secret
            for source in exact_sources
            if (secret := source.required_secret()) is not None
        }
    )
    return DeploymentResourcePlan(
        id=resource.id,
        family=resource.family,
        kind=resource.kind.value,
        name=resource.name,
        format=resource.format.value,
        relative_path=resource.relative_path,
        size_bytes=resource.size_bytes,
        installed=installed,
        provisionable=bool(exact_sources) and resource.size_bytes > 0,
        required_secrets=required_secrets,
        sources=resource.sources,
    )
