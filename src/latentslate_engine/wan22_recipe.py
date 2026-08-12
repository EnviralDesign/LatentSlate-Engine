"""CPU-only validation for explicit, inventory-owned Wan 2.2 I2V recipes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifacts import ArtifactIdentity, ArtifactProbe, probe_artifact, revalidate_artifact
from .resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)

_PROBE_SIGNATURE = "wan22_14b_36ch_40block_out16"
_DECLARED_ARCHITECTURES = {"wan2.2_i2v_14b": _PROBE_SIGNATURE, "wan": _PROBE_SIGNATURE}
_ARTIFACT_ROLES = frozenset(
    {"transformer_high_noise", "transformer_low_noise", "text_encoder", "vae"}
)
_NATIVE_REQUIRED_ROLES = frozenset({"pipeline_support", *_ARTIFACT_ROLES})
_NATIVE_ROLE_CONTRACTS = {
    "transformer_high_noise": frozenset(
        {
            "comfy_quant/float8_e4m3fn",
            "comfy_legacy/scaled_fp8_e4m3fn",
            "comfy_quant/int8_tensorwise_convrot",
        }
    ),
    "transformer_low_noise": frozenset(
        {
            "comfy_quant/float8_e4m3fn",
            "comfy_legacy/scaled_fp8_e4m3fn",
            "comfy_quant/int8_tensorwise_convrot",
        }
    ),
    "text_encoder": frozenset(
        {
            "comfy_legacy/scaled_fp8_e4m3fn",
            "comfy_quant/int8_tensorwise_convrot",
        }
    ),
    "vae": frozenset({"native/bf16"}),
}


@dataclass(frozen=True, slots=True)
class Wan22RecipeComponent:
    """A requested component reference; validation resolves it through inventory."""

    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class Wan22I2VRecipe:
    base_model: str
    high_noise: Wan22RecipeComponent
    low_noise: Wan22RecipeComponent
    text_encoder: Wan22RecipeComponent
    vae: Wan22RecipeComponent
    pipeline_support: Wan22RecipeComponent | None = None


@dataclass(frozen=True, slots=True)
class Wan22RecipeValidation:
    available: bool
    errors: tuple[str, ...]
    probes: tuple[ArtifactProbe, ...]
    resolved: dict[str, Wan22RecipeComponent]
    support_plan: Any | None = None
    adapter_plans: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Wan22RuntimeRequest:
    """Validated paths and identities to re-check immediately before execution."""

    schema_version: int
    family: str
    architecture: str
    base_model: str
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity]
    support_plan: Any | None = field(default=None, repr=False, compare=False)
    adapter_plans: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        frozen_components = MappingProxyType(
            {role: MappingProxyType(dict(component)) for role, component in self.components.items()}
        )
        frozen_identities = MappingProxyType(dict(self.identities))
        frozen_adapter_plans = MappingProxyType(dict(self.adapter_plans))
        object.__setattr__(self, "components", frozen_components)
        object.__setattr__(self, "identities", frozen_identities)
        object.__setattr__(self, "adapter_plans", frozen_adapter_plans)
        payload = {
            "schema_version": self.schema_version,
            "family": self.family,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "components": {
                role: dict(component)
                for role, component in sorted(frozen_components.items())
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "fingerprint", f"wan22-i2v-recipe:sha256:{digest}")

    def to_json_dict(self) -> dict[str, object]:
        """Return a deep-copyable JSON-safe internal execution manifest."""

        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "fingerprint": self.fingerprint,
            "components": {role: dict(component) for role, component in self.components.items()},
        }

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        """Return resource identities/contracts without exposing host filesystem paths."""

        return {
            role: {
                key: value
                for key, value in component.items()
                if key != "path"
            }
            for role, component in self.components.items()
        }


def validate_wan22_i2v_14b_recipe(
    recipe: Wan22I2VRecipe,
    inventory: ResourceInventory,
    *,
    include_support_plan: bool = True,
) -> Wan22RecipeValidation:
    """Validate the executor-neutral four-artifact recipe and optional support.

    This preserves the original portable recipe contract. Native execution adds its
    stricter five-role/exact-adapter checks in
    :func:`validate_native_wan22_i2v_14b_recipe`.
    """

    errors: list[str] = []
    resolved: dict[str, Wan22RecipeComponent] = {}
    probes: dict[str, ArtifactProbe] = {}
    support_plan: Any | None = None

    if recipe.pipeline_support is not None:
        support = _resolve_inventory_component(
            inventory,
            recipe.pipeline_support,
            "pipeline support",
            errors,
        )
        if support is not None:
            resolved["pipeline_support"] = support
            resource = support.resource
            if resource.kind != ResourceKind.MODEL or not resource.available:
                errors.append("pipeline support must be an available component resource")
            if resource.family != "wan22" or resource.component != "pipeline_support":
                errors.append(
                    "pipeline support must declare family='wan22' and "
                    "component='pipeline_support'"
                )
            if resource.format != ResourceFormat.DIRECTORY or not support.path.is_dir():
                errors.append("pipeline support must be a directory resource, not a model artifact")
            if include_support_plan:
                try:
                    support_plan = _plan_pipeline_support(support.path)
                except (ImportError, OSError, TypeError, ValueError) as exc:
                    errors.append(f"pipeline support validation failed: {exc}")

    requested = (
        ("transformer_high_noise", "high-noise transformer", recipe.high_noise, "high"),
        ("transformer_low_noise", "low-noise transformer", recipe.low_noise, "low"),
        ("text_encoder", "text encoder", recipe.text_encoder, None),
        ("vae", "VAE", recipe.vae, None),
    )
    for role, label, requested_component, stage in requested:
        component = _resolve_inventory_component(inventory, requested_component, label, errors)
        if component is None:
            continue
        resolved[role] = component
        resource = component.resource
        if resource.kind != ResourceKind.MODEL or not resource.available:
            errors.append(f"{label} must be an available model resource")
        if resource.family != "wan22" or resource.component != role:
            errors.append(f"{label} must declare family='wan22' and component={role!r}")
        if role.startswith("transformer") and resource.base_model != recipe.base_model:
            errors.append(f"{label} base_model does not match recipe base_model")
        if stage is not None and resource.metadata.get("noise_stage") != stage:
            errors.append(f"{label} must declare noise_stage={stage!r}")
        try:
            probe = probe_artifact(component.path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{label} probe failed: {exc}")
            continue
        probes[role] = probe
        if probe.format != resource.format.value:
            errors.append(f"{label} declared format does not match artifact container")
        _validate_contract(resource, probe, label, errors)
        _validate_role_architecture(role, resource, probe, label, errors)

    high, low = resolved.get("transformer_high_noise"), resolved.get("transformer_low_noise")
    if high is not None and low is not None:
        if high.resource.id == low.resource.id or high.path.resolve() == low.path.resolve():
            errors.append("high- and low-noise transformers must be distinct resources")
        high_contract = high.resource.metadata.get("quantization_contract")
        low_contract = low.resource.metadata.get("quantization_contract")
        if high.resource.format != low.resource.format or high_contract != low_contract:
            errors.append("high- and low-noise transformers must use one matching format and contract")
        declared = {
            _DECLARED_ARCHITECTURES.get(str(item.resource.metadata.get("architecture")))
            for item in (high, low)
        }
        if declared != {_PROBE_SIGNATURE}:
            errors.append(
                "high- and low-noise transformers must declare a mapped canonical architecture"
            )
        transformer_signals = {
            probe.architecture_signals
            for role, probe in probes.items()
            if role.startswith("transformer")
        }
        if transformer_signals != {(_PROBE_SIGNATURE,)}:
            errors.append(
                "high- and low-noise headers must expose the same exact architecture signature"
            )
        if (
            probes.get("transformer_high_noise") is not None
            and probes.get("transformer_low_noise") is not None
            and probes["transformer_high_noise"].schema_sha256
            != probes["transformer_low_noise"].schema_sha256
        ):
            errors.append(
                "high- and low-noise transformers must share one topology/schema fingerprint"
            )

    required_paths = [component.path.resolve() for component in resolved.values()]
    required_ids = [component.resource.id for component in resolved.values()]
    if len(required_paths) != len(set(required_paths)) or len(required_ids) != len(set(required_ids)):
        errors.append("all required Wan roles must resolve to distinct resources and canonical paths")
    required_roles = set(_ARTIFACT_ROLES)
    if recipe.pipeline_support is not None:
        required_roles.add("pipeline_support")
    if set(resolved) != required_roles:
        missing = sorted(required_roles - set(resolved))
        if missing:
            errors.append("Wan recipe is missing resolved roles: " + ", ".join(missing))

    return Wan22RecipeValidation(
        not errors,
        tuple(errors),
        tuple(probes.values()),
        resolved,
        support_plan,
    )


def validate_native_wan22_i2v_14b_recipe(
    recipe: Wan22I2VRecipe,
    inventory: ResourceInventory,
    *,
    include_adapter_plans: bool = True,
) -> Wan22RecipeValidation:
    """Validate the exact five-role recipe executable by NativeWanI2VRuntime.

    Catalog inspection may omit adapter materialization so listing an unavailable
    recipe does not import the CUDA execution stack. Runtime request construction
    retains the default and therefore always performs the complete validation.
    """

    generic = validate_wan22_i2v_14b_recipe(
        recipe,
        inventory,
        include_support_plan=include_adapter_plans,
    )
    errors = list(generic.errors)
    adapter_plans: dict[str, Any] = {}
    if include_adapter_plans and (
        recipe.pipeline_support is None or generic.support_plan is None
    ):
        errors.append("native Wan execution requires pipeline support")
    if set(generic.resolved) != _NATIVE_REQUIRED_ROLES:
        missing = sorted(_NATIVE_REQUIRED_ROLES - set(generic.resolved))
        if missing:
            errors.append("native Wan recipe is missing resolved roles: " + ", ".join(missing))

    if include_adapter_plans:
        planners = _native_adapter_planners()
        for role in sorted(_ARTIFACT_ROLES):
            component = generic.resolved.get(role)
            if component is None:
                continue
            contract = component.resource.metadata.get("quantization_contract")
            if component.resource.format != ResourceFormat.SAFETENSORS:
                errors.append(f"native {role} requires a SafeTensors artifact")
                continue
            if contract not in _NATIVE_ROLE_CONTRACTS[role]:
                errors.append(
                    f"native {role} does not support stored contract {contract!r}"
                )
                continue
            try:
                plan = planners[role](component.path)
                plan.require_available()
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"native {role} adapter is unavailable: {exc}")
                continue
            probe = next((item for item in generic.probes if item.path == component.path), None)
            if probe is None or plan.identity != probe.identity:
                errors.append(f"native {role} adapter identity does not match recipe probe")
                continue
            adapter_plans[role] = plan

        if set(adapter_plans) != _ARTIFACT_ROLES:
            missing = sorted(_ARTIFACT_ROLES - set(adapter_plans))
            if missing:
                errors.append("native Wan adapter plans are missing roles: " + ", ".join(missing))
        high = adapter_plans.get("transformer_high_noise")
        low = adapter_plans.get("transformer_low_noise")
        if (
            high is not None
            and low is not None
            and high.artifact_contract != low.artifact_contract
        ):
            errors.append("native high/low transformer storage contracts must match")

    return Wan22RecipeValidation(
        not errors,
        tuple(errors),
        generic.probes,
        generic.resolved,
        generic.support_plan,
        MappingProxyType(adapter_plans),
    )


def build_wan22_i2v_14b_runtime_request(
    recipe: Wan22I2VRecipe, inventory: ResourceInventory
) -> Wan22RuntimeRequest:
    validation = validate_wan22_i2v_14b_recipe(recipe, inventory)
    if not validation.available:
        raise ValueError("Wan 2.2 I2V recipe is unavailable: " + "; ".join(validation.errors))
    probe_by_path = {probe.path: probe for probe in validation.probes}
    components = {
        role: _runtime_component(component, probe_by_path[component.path])
        for role, component in validation.resolved.items()
        if role in _ARTIFACT_ROLES
    }
    if validation.support_plan is not None:
        support = validation.resolved["pipeline_support"]
        components["pipeline_support"] = _runtime_support_component(
            support,
            validation.support_plan,
        )
    identities = {
        role: probe_by_path[component.path].identity
        for role, component in validation.resolved.items()
        if role in _ARTIFACT_ROLES
    }
    return Wan22RuntimeRequest(
        2 if validation.support_plan is not None else 1,
        "wan22",
        _PROBE_SIGNATURE,
        recipe.base_model,
        components,
        identities,
        validation.support_plan,
    )


def build_native_wan22_i2v_14b_runtime_request(
    recipe: Wan22I2VRecipe,
    inventory: ResourceInventory,
) -> Wan22RuntimeRequest:
    """Build the exact identity- and adapter-bound request for the native runtime."""

    validation = validate_native_wan22_i2v_14b_recipe(recipe, inventory)
    if not validation.available or validation.support_plan is None:
        raise ValueError(
            "Native Wan 2.2 I2V recipe is unavailable: " + "; ".join(validation.errors)
        )
    probe_by_path = {probe.path: probe for probe in validation.probes}
    components = {
        role: _runtime_component(component, probe_by_path[component.path])
        for role, component in validation.resolved.items()
        if role in _ARTIFACT_ROLES
    }
    components["pipeline_support"] = _runtime_support_component(
        validation.resolved["pipeline_support"],
        validation.support_plan,
    )
    identities = {
        role: validation.adapter_plans[role].identity for role in _ARTIFACT_ROLES
    }
    return Wan22RuntimeRequest(
        3,
        "wan22",
        _PROBE_SIGNATURE,
        recipe.base_model,
        components,
        identities,
        validation.support_plan,
        validation.adapter_plans,
    )


def revalidate_runtime_request(request: Wan22RuntimeRequest) -> bool:
    """Re-check every serialized artifact and optional support plan."""

    expected_components = set(_ARTIFACT_ROLES)
    if request.support_plan is not None:
        expected_components.add("pipeline_support")
    if set(request.components) != expected_components or set(request.identities) != _ARTIFACT_ROLES:
        return False
    if request.support_plan is not None:
        support_component = request.components.get("pipeline_support", {})
        if (
            support_component.get("path") != str(request.support_plan.root)
            or support_component.get("support_fingerprint") != request.support_plan.fingerprint
            or support_component.get("tokenizer_sha256") != request.support_plan.tokenizer_sha256
            or not _revalidate_pipeline_support(request.support_plan)
        ):
            return False
    if request.adapter_plans and set(request.adapter_plans) != _ARTIFACT_ROLES:
        return False
    for role, identity in request.identities.items():
        component = request.components.get(role, {})
        if (
            component.get("path") != str(identity.path)
            or component.get("size_bytes") != identity.size_bytes
            or component.get("mtime_ns") != identity.mtime_ns
            or component.get("header_sha256") != identity.header_sha256
            or not revalidate_artifact(identity)
        ):
            return False
        adapter_plan = request.adapter_plans.get(role)
        if adapter_plan is not None and adapter_plan.identity != identity:
            return False
    return True


def _plan_pipeline_support(path: Path) -> Any:
    # Lazy import keeps CPU-only protocol installs usable when no native recipe is present.
    from .runtime.wan22_i2v_support import plan_wan_i2v_support

    return plan_wan_i2v_support(path)


def _revalidate_pipeline_support(plan: Any) -> bool:
    try:
        from .runtime.wan22_i2v_support import revalidate_wan_i2v_support

        return bool(revalidate_wan_i2v_support(plan))
    except (ImportError, OSError, TypeError, ValueError):
        return False


def _native_adapter_planners() -> dict[str, Any]:
    # Lazy imports keep protocol-only installs and non-native catalogs cheap.
    from .runtime.umt5_stored_adapter import plan_comfy_umt5_encoder
    from .runtime.wan21_vae_adapter import plan_comfy_wan21_vae
    from .runtime.wan22_stored_adapter import plan_comfy_wan_transformer

    return {
        "transformer_high_noise": plan_comfy_wan_transformer,
        "transformer_low_noise": plan_comfy_wan_transformer,
        "text_encoder": plan_comfy_umt5_encoder,
        "vae": plan_comfy_wan21_vae,
    }


def _resolve_inventory_component(
    inventory: ResourceInventory,
    requested: Wan22RecipeComponent,
    label: str,
    errors: list[str],
) -> Wan22RecipeComponent | None:
    actual = inventory.by_id().get(requested.resource.id)
    if actual is None:
        errors.append(f"{label} resource is not owned by this inventory")
        return None
    try:
        path = inventory.path_for(actual.id).resolve(strict=True)
    except (KeyError, OSError) as exc:
        errors.append(f"{label} inventory path is unavailable: {exc}")
        return None
    if actual != requested.resource or path != requested.path.resolve(strict=False):
        errors.append(f"{label} descriptor/path does not match the inventory mapping")
        return None
    return Wan22RecipeComponent(actual, path)


def _validate_contract(
    resource: ResourceDescriptor, probe: ArtifactProbe, label: str, errors: list[str]
) -> None:
    declared = resource.metadata.get("quantization_contract")
    if not isinstance(declared, str) or not declared or probe.quantization_contract != declared:
        errors.append(f"{label} requires one exact proven quantization_contract")
        return
    expected = {
        "comfy_quant/int8_tensorwise_convrot": (
            ArtifactPrecision.UNKNOWN,
            ArtifactQuantization.INT8,
        ),
        "comfy_quant/float8_e4m3fn": (
            ArtifactPrecision.FP8,
            ArtifactQuantization.NATIVE,
        ),
        "comfy_legacy/scaled_fp8_e4m3fn": (
            ArtifactPrecision.FP8,
            ArtifactQuantization.NATIVE,
        ),
        "native/bf16": (ArtifactPrecision.BF16, ArtifactQuantization.NATIVE),
        "native/fp16": (ArtifactPrecision.FP16, ArtifactQuantization.NATIVE),
        "native/fp32": (ArtifactPrecision.FP32, ArtifactQuantization.NATIVE),
        "gguf/q5_k_m": (ArtifactPrecision.UNKNOWN, ArtifactQuantization.GGUF),
    }.get(declared)
    if expected is None or (resource.precision, resource.quantization) != expected:
        errors.append(
            f"{label} descriptor precision/quantization does not match its proven contract"
        )


def _validate_role_architecture(
    role: str,
    resource: ResourceDescriptor,
    probe: ArtifactProbe,
    label: str,
    errors: list[str],
) -> None:
    expected = (
        _PROBE_SIGNATURE
        if role.startswith("transformer")
        else ("umt5_xxl" if role == "text_encoder" else "wan_vae_2_1")
    )
    declared = (
        _DECLARED_ARCHITECTURES.get(str(resource.metadata.get("architecture")))
        if role.startswith("transformer")
        else resource.metadata.get("architecture")
    )
    if declared != expected or expected not in probe.architecture_signals:
        errors.append(f"{label} declared architecture does not match its exact header signature")


def _runtime_component(
    component: Wan22RecipeComponent, probe: ArtifactProbe
) -> dict[str, str | int]:
    identity = probe.identity
    return {
        "resource_id": component.resource.id,
        "path": str(component.path),
        "format": component.resource.format.value,
        "component": component.resource.component or "",
        "quantization_contract": str(
            component.resource.metadata["quantization_contract"]
        ),
        "size_bytes": identity.size_bytes,
        "mtime_ns": identity.mtime_ns,
        "header_sha256": identity.header_sha256,
        "schema_sha256": probe.schema_sha256,
    }


def _runtime_support_component(
    component: Wan22RecipeComponent,
    support_plan: Any,
) -> dict[str, str | int]:
    return {
        "resource_id": component.resource.id,
        "path": str(component.path),
        "format": component.resource.format.value,
        "component": "pipeline_support",
        "support_fingerprint": support_plan.fingerprint,
        "tokenizer_sha256": support_plan.tokenizer_sha256,
        "file_count": len(support_plan.files),
    }
