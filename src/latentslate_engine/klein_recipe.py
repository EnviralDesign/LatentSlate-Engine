"""Typed, inventory-owned stored Klein component-recipe validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from .resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)
from .runtime.klein_components import (
    KLEIN_BASE_TRANSFORMER_SCHEMA_SHA256,
    KLEIN_DISTILLED_TRANSFORMER_SCHEMA_SHA256,
    plan_klein_pipeline_support,
    plan_klein_small_vae,
    plan_klein_text_encoder,
    plan_klein_vae,
    revalidate_klein_dense_component,
    revalidate_klein_pipeline_support,
)
from .runtime.klein_contracts import (
    KLEIN4B_CONFIG,
    KLEIN9_QWEN_MIXED_ARCHITECTURE,
    KLEIN9_QWEN_MIXED_CONTRACT,
    KLEIN9B_CONFIG,
)

KleinRecipeMode = Literal["base", "distilled"]
_KLEIN_STORED_FP8_CONTRACT = "comfy_quant/float8_e4m3fn_global"
_KLEIN_STORED_NVFP4_CONTRACT = "comfy_quant/nvfp4_tensorcore"
_ROLES = frozenset({"pipeline_support", "transformer", "text_encoder", "vae"})
_SCHEDULES: dict[str, tuple[int, float]] = {
    "base": (20, 5.0),
    "distilled": (4, 1.0),
}
_TRANSFORMER_SCHEMAS = {
    "base": KLEIN_BASE_TRANSFORMER_SCHEMA_SHA256,
    "distilled": KLEIN_DISTILLED_TRANSFORMER_SCHEMA_SHA256,
}
_KLEIN_DISTILLED_NVFP4_SCHEMA_SHA256 = (
    "c6683e31192ed861a3068673e41d89555caacdad2e4a3a7357e5e576dcaea9d6"
)
_KLEIN9_DISTILLED_FP8_SCHEMA_SHA256 = (
    "c25cec508eb68835ccd5833bb3a9886a1dea9cfb652ecf98b1ecf4d6d332940d"
)
_KLEIN9_DISTILLED_NVFP4_SCHEMA_SHA256 = (
    "a222d48e4d796bfdb027b0c8e0eb3c8dc655d0901dbe9a7fdab41b434fb036f8"
)


@dataclass(frozen=True, slots=True)
class Klein4RecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class KleinStoredRecipe:
    mode: KleinRecipeMode
    base_model: str
    steps: int
    guidance_scale: float
    pipeline_support: Klein4RecipeComponent
    transformer: Klein4RecipeComponent
    text_encoder: Klein4RecipeComponent
    vae: Klein4RecipeComponent
    family: Literal["klein4b", "klein9b"] = "klein4b"


@dataclass(frozen=True, slots=True)
class Klein4RecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, Klein4RecipeComponent]
    support_plan: Any | None = None
    adapter_plans: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Klein4RuntimeRequest:
    schema_version: int
    family: str
    mode: KleinRecipeMode
    base_model: str
    steps: int
    guidance_scale: float
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity]
    support_plan: Any = field(repr=False, compare=False)
    adapter_plans: Mapping[str, Any] = field(repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        components = MappingProxyType(
            {role: MappingProxyType(dict(value)) for role, value in self.components.items()}
        )
        identities = MappingProxyType(dict(self.identities))
        adapter_plans = MappingProxyType(dict(self.adapter_plans))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "adapter_plans", adapter_plans)
        payload = {
            "schema_version": self.schema_version,
            "family": self.family,
            "mode": self.mode,
            "base_model": self.base_model,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        namespace = "klein4-stored-recipe" if self.family == "klein4b" else "klein9-stored-recipe"
        object.__setattr__(self, "fingerprint", f"{namespace}:sha256:{digest}")

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {
            role: {key: value for key, value in component.items() if key != "path"}
            for role, component in self.components.items()
        }


def validate_klein_stored_recipe(
    recipe: KleinStoredRecipe,
    inventory: ResourceInventory,
    *,
    include_adapter_plans: bool = True,
) -> Klein4RecipeValidation:
    errors: list[str] = []
    resolved: dict[str, Klein4RecipeComponent] = {}
    plans: dict[str, Any] = {}
    support_plan = None
    family = recipe.family

    if family == "klein9b" and recipe.mode != "distilled":
        errors.append("Klein 9B component recipes currently support Distilled mode only")

    expected_schedule = _SCHEDULES.get(recipe.mode)
    if expected_schedule != (recipe.steps, recipe.guidance_scale):
        errors.append(
            f"{recipe.mode} Klein requires immutable schedule "
            f"steps={expected_schedule[0]}, guidance_scale={expected_schedule[1]}"
        )

    for role in sorted(_ROLES):
        requested = getattr(recipe, role)
        component = _resolve_component(inventory, requested, role, errors)
        if component is None:
            continue
        resolved[role] = component
        resource = component.resource
        if resource.kind != ResourceKind.MODEL or not resource.available:
            errors.append(f"{role} must be an available model resource")
        shared_small_vae = bool(
            family == "klein9b"
            and role == "vae"
            and resource.family == "klein4b"
            and resource.metadata.get("architecture") == "flux2_small_decoder_full_encoder"
        )
        if (resource.family != family and not shared_small_vae) or resource.component != role:
            errors.append(f"{role} must declare family={family!r} and component={role!r}")

    if support := resolved.get("pipeline_support"):
        if support.resource.format != ResourceFormat.DIRECTORY or not support.path.is_dir():
            errors.append("pipeline_support must be a bounded directory resource")
        else:
            try:
                support_plan = plan_klein_pipeline_support(
                    support.path,
                    recipe.mode,
                    family,
                )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"pipeline_support contract failed: {exc}")

    transformer = resolved.get("transformer")
    if transformer is not None:
        nvfp4 = transformer.resource.quantization == ArtifactQuantization.NVFP4
        if nvfp4 and recipe.mode != "distilled":
            errors.append("Klein NVFP4 is supported only for Distilled recipes")
        _validate_descriptor(
            transformer.resource,
            role="transformer",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.FP4 if nvfp4 else ArtifactPrecision.FP8,
            quantization=(ArtifactQuantization.NVFP4 if nvfp4 else ArtifactQuantization.NATIVE),
            contract=(_KLEIN_STORED_NVFP4_CONTRACT if nvfp4 else _KLEIN_STORED_FP8_CONTRACT),
            architecture=f"flux2_klein_{'4b' if family == 'klein4b' else '9b'}_{recipe.mode}",
            base_model=recipe.base_model,
            errors=errors,
        )
        try:
            probe = probe_artifact(transformer.path)
            if family == "klein9b":
                expected_schema = (
                    _KLEIN9_DISTILLED_NVFP4_SCHEMA_SHA256
                    if nvfp4
                    else _KLEIN9_DISTILLED_FP8_SCHEMA_SHA256
                )
                adapter_config = KLEIN9B_CONFIG
            else:
                expected_schema = (
                    _KLEIN_DISTILLED_NVFP4_SCHEMA_SHA256
                    if nvfp4
                    else _TRANSFORMER_SCHEMAS[recipe.mode]
                )
                adapter_config = KLEIN4B_CONFIG
            if probe.schema_sha256 != expected_schema:
                raise ValueError("transformer schema does not match its declared Klein mode")
            if include_adapter_plans:
                from .runtime.klein_stored_adapter import (
                    plan_bfl_klein_nvfp4_transformer,
                    plan_klein_stored_transformer,
                )

                adapter = (
                    plan_bfl_klein_nvfp4_transformer(
                        transformer.path,
                        adapter_config,
                    )
                    if nvfp4
                    else plan_klein_stored_transformer(
                        transformer.path,
                        adapter_config,
                    )
                )
                adapter.require_available()
                plans["transformer"] = adapter
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"transformer contract failed: {exc}")

    text_encoder = resolved.get("text_encoder")
    if text_encoder is not None:
        mixed_qwen = family == "klein9b"
        _validate_descriptor(
            text_encoder.resource,
            role="text_encoder",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.FP4 if mixed_qwen else ArtifactPrecision.BF16,
            quantization=(
                ArtifactQuantization.NVFP4 if mixed_qwen else ArtifactQuantization.NATIVE
            ),
            contract=KLEIN9_QWEN_MIXED_CONTRACT if mixed_qwen else "native/bf16",
            architecture=KLEIN9_QWEN_MIXED_ARCHITECTURE if mixed_qwen else "qwen3_4b",
            base_model=None,
            errors=errors,
        )
        try:
            if mixed_qwen:
                from .runtime.klein_quantized_text import plan_klein_mixed_text_encoder

            plans["text_encoder"] = (
                plan_klein_mixed_text_encoder(text_encoder.path)
                if mixed_qwen
                else plan_klein_text_encoder(text_encoder.path)
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"text_encoder contract failed: {exc}")

    vae = resolved.get("vae")
    if vae is not None:
        use_small_vae = recipe.mode == "base" or family == "klein9b"
        vae_architecture = "flux2_small_decoder_full_encoder" if use_small_vae else "flux2_vae"
        _validate_descriptor(
            vae.resource,
            role="vae",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.FP32,
            quantization=ArtifactQuantization.NATIVE,
            contract="native/fp32",
            architecture=vae_architecture,
            base_model=None,
            errors=errors,
        )
        try:
            plans["vae"] = (
                plan_klein_small_vae(vae.path) if use_small_vae else plan_klein_vae(vae.path)
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"vae contract failed: {exc}")

    canonical_paths = [component.path.resolve(strict=False) for component in resolved.values()]
    ids = [component.resource.id for component in resolved.values()]
    if len(canonical_paths) != len(set(canonical_paths)) or len(ids) != len(set(ids)):
        errors.append("all Klein recipe roles must use distinct resources and paths")
    if set(resolved) != _ROLES:
        errors.append("Klein recipe did not resolve all four exact component roles")

    return Klein4RecipeValidation(
        not errors,
        tuple(errors),
        MappingProxyType(resolved),
        support_plan,
        MappingProxyType(plans),
    )


def build_klein_stored_runtime_request(
    recipe: KleinStoredRecipe,
    inventory: ResourceInventory,
) -> Klein4RuntimeRequest:
    validation = validate_klein_stored_recipe(recipe, inventory)
    if not validation.available or validation.support_plan is None:
        raise ValueError(
            f"Stored {recipe.family} recipe is unavailable: " + "; ".join(validation.errors)
        )

    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in ("transformer", "text_encoder", "vae"):
        component = validation.resolved[role]
        plan = validation.adapter_plans[role]
        identity = plan.identity
        identities[role] = identity
        components[role] = {
            "resource_id": component.resource.id,
            "path": str(component.path.resolve()),
            "format": component.resource.format.value,
            "component": role,
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
            "schema_sha256": probe_artifact(component.path).schema_sha256,
            "quantization_contract": str(component.resource.metadata["quantization_contract"]),
        }
    support = validation.resolved["pipeline_support"]
    components["pipeline_support"] = {
        "resource_id": support.resource.id,
        "path": str(support.path.resolve()),
        "format": support.resource.format.value,
        "component": "pipeline_support",
        "support_fingerprint": validation.support_plan.fingerprint,
        "file_count": len(validation.support_plan.files),
    }
    return Klein4RuntimeRequest(
        1,
        recipe.family,
        recipe.mode,
        recipe.base_model,
        recipe.steps,
        recipe.guidance_scale,
        components,
        identities,
        validation.support_plan,
        validation.adapter_plans,
    )


def revalidate_klein4_runtime_request(request: Klein4RuntimeRequest) -> bool:
    if set(request.components) != _ROLES or set(request.identities) != {
        "transformer",
        "text_encoder",
        "vae",
    }:
        return False
    if _SCHEDULES.get(request.mode) != (request.steps, request.guidance_scale):
        return False
    if not revalidate_klein_pipeline_support(request.support_plan):
        return False
    support = request.components["pipeline_support"]
    if (
        support.get("path") != str(request.support_plan.root)
        or support.get("support_fingerprint") != request.support_plan.fingerprint
    ):
        return False
    for role, identity in request.identities.items():
        component = request.components[role]
        if (
            component.get("path") != str(identity.path)
            or component.get("size_bytes") != identity.size_bytes
            or component.get("mtime_ns") != identity.mtime_ns
            or component.get("header_sha256") != identity.header_sha256
            or not revalidate_artifact(identity)
        ):
            return False
        plan = request.adapter_plans.get(role)
        if plan is None or plan.identity != identity:
            return False
        if role == "text_encoder":
            if request.family == "klein9b":
                from .runtime.klein_quantized_text import (
                    revalidate_klein_mixed_text_encoder,
                )

                if not revalidate_klein_mixed_text_encoder(plan):
                    return False
            elif not revalidate_klein_dense_component(plan):
                return False
        elif role == "vae" and not revalidate_klein_dense_component(plan):
            return False
    return True


def _resolve_component(
    inventory: ResourceInventory,
    requested: Klein4RecipeComponent,
    role: str,
    errors: list[str],
) -> Klein4RecipeComponent | None:
    actual = inventory.by_id().get(requested.resource.id)
    if actual is None:
        errors.append(f"{role} resource is not owned by this inventory")
        return None
    try:
        path = inventory.path_for(actual.id).resolve(strict=True)
    except (KeyError, OSError) as exc:
        errors.append(f"{role} inventory path is unavailable: {exc}")
        return None
    if actual != requested.resource or path != requested.path.resolve(strict=False):
        errors.append(f"{role} descriptor/path differs from inventory ownership")
        return None
    return Klein4RecipeComponent(actual, path)


def _validate_descriptor(
    resource: ResourceDescriptor,
    *,
    role: str,
    format: ResourceFormat,
    precision: ArtifactPrecision,
    quantization: ArtifactQuantization,
    contract: str,
    architecture: str,
    base_model: str | None,
    errors: list[str],
) -> None:
    if resource.format != format:
        errors.append(f"{role} format must be {format.value!r}")
    if resource.precision != precision or resource.quantization != quantization:
        errors.append(f"{role} stored precision/quantization metadata is incorrect")
    if resource.metadata.get("quantization_contract") != contract:
        errors.append(f"{role} quantization_contract must be {contract!r}")
    if resource.metadata.get("architecture") != architecture:
        errors.append(f"{role} architecture must be {architecture!r}")
    if base_model is not None and resource.base_model != base_model:
        errors.append(f"{role} base_model does not match recipe base_model")
