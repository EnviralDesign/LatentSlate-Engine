"""CPU-only validation for explicit, inventory-owned Wan 2.2 I2V recipes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifacts import ArtifactIdentity, ArtifactProbe, probe_artifact, revalidate_artifact
from .lora import ConfiguredLora
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
    operation: str = "comfy_i2v_base"
    lora_stage_by_slot: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Wan22StageLora:
    """One active adapter with an explicit high/low Wan expert binding."""

    slot: str
    stage: str
    resource_id: str
    path: Path
    strength: float
    identity: ArtifactIdentity
    schema_sha256: str

    def public_dict(self) -> dict[str, str | float | int]:
        return {
            "slot": self.slot,
            "stage": self.stage,
            "resource_id": self.resource_id,
            "strength": self.strength,
            "size_bytes": self.identity.size_bytes,
            "mtime_ns": self.identity.mtime_ns,
            "header_sha256": self.identity.header_sha256,
            "schema_sha256": self.schema_sha256,
        }


_I2V_OPERATIONS: Mapping[str, Mapping[str, str | int | float]] = MappingProxyType(
    {
        "comfy_i2v_base": MappingProxyType(
            {
                "steps": 20,
                "stage_policy": "comfy_split",
                "high_guidance": 3.5,
                "low_guidance": 3.5,
                "sampler": "euler",
                "scheduler": "simple",
                "shift": 5.0,
                "fps": 16,
            }
        ),
        "comfy_i2v_lightx2v_4step": MappingProxyType(
            {
                "steps": 4,
                "stage_policy": "comfy_split",
                "high_guidance": 1.0,
                "low_guidance": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "shift": 5.0,
                "fps": 16,
            }
        ),
    }
)
_LIGHTX2V_REQUIRED_LORAS = MappingProxyType(
    {
        "high_noise": "lora:wan22:comfy-org/wan22-14b-i2v-lightx2v-4step-high-noise",
        "low_noise": "lora:wan22:comfy-org/wan22-14b-i2v-lightx2v-4step-low-noise",
    }
)


def wan22_i2v_operation(operation: str) -> Mapping[str, str | int | float]:
    """Return one pinned built-in operation, never a caller-selectable sampler."""

    try:
        return _I2V_OPERATIONS[operation]
    except KeyError as exc:
        raise ValueError(f"unknown native Wan I2V operation {operation!r}") from exc


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
    operation: str = "comfy_i2v_base"
    configured_loras: tuple[dict[str, str | float | bool], ...] = ()
    active_loras: tuple[Wan22StageLora, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        frozen_components = MappingProxyType(
            {role: MappingProxyType(dict(component)) for role, component in self.components.items()}
        )
        frozen_identities = MappingProxyType(dict(self.identities))
        frozen_adapter_plans = MappingProxyType(dict(self.adapter_plans))
        operation = str(self.operation)
        wan22_i2v_operation(operation)
        configured_loras = tuple(MappingProxyType(dict(item)) for item in self.configured_loras)
        active_loras = tuple(self.active_loras)
        if any(item.stage not in {"high", "low"} for item in active_loras):
            raise ValueError("Wan active LoRAs must target high or low stage")
        if len({item.slot for item in active_loras}) != len(active_loras):
            raise ValueError("Wan active LoRA slots must be unique")
        object.__setattr__(self, "components", frozen_components)
        object.__setattr__(self, "identities", frozen_identities)
        object.__setattr__(self, "adapter_plans", frozen_adapter_plans)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "configured_loras", configured_loras)
        object.__setattr__(self, "active_loras", active_loras)
        payload = {
            "schema_version": self.schema_version,
            "family": self.family,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "components": {
                role: dict(component) for role, component in sorted(frozen_components.items())
            },
            "operation": operation,
            "configured_loras": [dict(item) for item in configured_loras],
            "active_loras": [item.public_dict() for item in active_loras],
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
            "operation": self.operation,
            "configured_loras": [dict(item) for item in self.configured_loras],
            "active_loras": [
                {
                    **item.public_dict(),
                    "path": str(item.path),
                }
                for item in self.active_loras
            ],
        }

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        """Return resource identities/contracts without exposing host filesystem paths."""

        return {
            role: {key: value for key, value in component.items() if key != "path"}
            for role, component in self.components.items()
        }


def rehydrate_native_wan22_i2v_14b_runtime_request(
    payload: Mapping[str, Any],
) -> Wan22RuntimeRequest:
    """Rebuild a native request from its canonical worker manifest.

    A disposable native-Wan worker receives no live Python plans from its
    supervisor.  It rebuilds every support and adapter plan from this small JSON
    manifest, then compares the resulting canonical representation byte-for-byte
    (including the recipe fingerprint).  This deliberately rejects permissive
    "almost a recipe" input rather than letting a child reinterpret an arbitrary
    component graph.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("native Wan worker request must be an object")
    expected_top = {
        "schema_version",
        "family",
        "architecture",
        "base_model",
        "fingerprint",
        "components",
        "operation",
        "configured_loras",
        "active_loras",
    }
    if set(payload) != expected_top:
        raise ValueError("native Wan worker request fields are not canonical")
    if payload["schema_version"] != 4:
        raise ValueError("native Wan worker request schema_version must be 4")
    if payload["family"] != "wan22" or payload["architecture"] != _PROBE_SIGNATURE:
        raise ValueError("native Wan worker request family/architecture is invalid")
    base_model = payload["base_model"]
    fingerprint = payload["fingerprint"]
    components_payload = payload["components"]
    operation = payload["operation"]
    operation_contract = wan22_i2v_operation(operation) if isinstance(operation, str) else None
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("native Wan worker request base_model is invalid")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("wan22-i2v-recipe:sha256:"):
        raise ValueError("native Wan worker request fingerprint is invalid")
    if operation_contract is None:
        raise ValueError("native Wan worker request operation is invalid")
    if (
        not isinstance(components_payload, Mapping)
        or set(components_payload) != _NATIVE_REQUIRED_ROLES
    ):
        raise ValueError("native Wan worker request must contain the exact five component roles")

    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in sorted(_ARTIFACT_ROLES):
        raw = components_payload[role]
        if not isinstance(raw, Mapping):
            raise TypeError(f"native Wan worker component {role!r} is invalid")
        expected_fields = {
            "resource_id",
            "path",
            "format",
            "component",
            "quantization_contract",
            "size_bytes",
            "mtime_ns",
            "header_sha256",
            "schema_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError(f"native Wan worker component {role!r} fields are not canonical")
        path_value = raw["path"]
        if not isinstance(path_value, str):
            raise TypeError(f"native Wan worker component {role!r} path is invalid")
        path = Path(path_value).resolve(strict=True)
        identity = ArtifactIdentity(
            path=path,
            size_bytes=_strict_int(raw["size_bytes"], f"{role}.size_bytes"),
            mtime_ns=_strict_int(raw["mtime_ns"], f"{role}.mtime_ns"),
            header_sha256=_strict_sha256(raw["header_sha256"], f"{role}.header_sha256"),
        )
        probe = probe_artifact(path)
        if probe.identity != identity:
            raise ValueError(f"native Wan worker component {role!r} artifact identity changed")
        if (
            raw["format"] != "safetensors"
            or raw["component"] != role
            or raw["quantization_contract"] not in _NATIVE_ROLE_CONTRACTS[role]
            or raw["schema_sha256"] != probe.schema_sha256
            or not isinstance(raw["resource_id"], str)
        ):
            raise ValueError(f"native Wan worker component {role!r} contract is invalid")
        components[role] = {key: value for key, value in raw.items()}
        identities[role] = identity

    raw_support = components_payload["pipeline_support"]
    if not isinstance(raw_support, Mapping):
        raise TypeError("native Wan worker pipeline support is invalid")
    expected_support_fields = {
        "resource_id",
        "path",
        "format",
        "component",
        "support_fingerprint",
        "tokenizer_sha256",
        "file_count",
    }
    if set(raw_support) != expected_support_fields:
        raise ValueError("native Wan worker pipeline support fields are not canonical")
    support_path_value = raw_support["path"]
    if (
        not isinstance(support_path_value, str)
        or raw_support["format"] != "directory"
        or raw_support["component"] != "pipeline_support"
        or not isinstance(raw_support["resource_id"], str)
    ):
        raise ValueError("native Wan worker pipeline support contract is invalid")
    support_plan = _plan_pipeline_support(Path(support_path_value).resolve(strict=True))
    if (
        raw_support["support_fingerprint"] != support_plan.fingerprint
        or raw_support["tokenizer_sha256"] != support_plan.tokenizer_sha256
        or raw_support["file_count"] != len(support_plan.files)
    ):
        raise ValueError("native Wan worker pipeline support identity changed")
    components["pipeline_support"] = {key: value for key, value in raw_support.items()}

    planners = _native_adapter_planners()
    adapter_plans: dict[str, Any] = {}
    for role, identity in identities.items():
        plan = planners[role](identity.path)
        plan.require_available()
        if plan.identity != identity:
            raise ValueError(f"native Wan worker {role} adapter identity changed")
        adapter_plans[role] = plan
    configured_loras = _rehydrate_configured_loras(payload["configured_loras"])
    active_loras = _rehydrate_active_loras(payload["active_loras"])
    request = Wan22RuntimeRequest(
        4,
        "wan22",
        _PROBE_SIGNATURE,
        base_model,
        components,
        identities,
        support_plan,
        adapter_plans,
        operation,
        configured_loras,
        active_loras,
    )
    if request.fingerprint != fingerprint or request.to_json_dict() != dict(payload):
        raise ValueError("native Wan worker request fingerprint is not canonical")
    return request


def _rehydrate_configured_loras(value: object) -> tuple[dict[str, str | float | bool], ...]:
    if not isinstance(value, list):
        raise TypeError("native Wan worker configured_loras must be a list")
    configured: list[dict[str, str | float | bool]] = []
    slots: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "slot", "stage", "resource_reference", "strength", "active"
        }:
            raise ValueError("native Wan worker configured LoRA fields are not canonical")
        slot, stage, reference, strength, active = (
            item["slot"], item["stage"], item["resource_reference"], item["strength"], item["active"]
        )
        if (
            not isinstance(slot, str)
            or not slot
            or slot in slots
            or stage not in {"high", "low"}
            or (reference is not None and not isinstance(reference, str))
            or isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
            or not isinstance(active, bool)
        ):
            raise ValueError("native Wan worker configured LoRA is invalid")
        if active != (reference is not None and float(strength) != 0.0):
            raise ValueError("native Wan worker configured LoRA active state is invalid")
        slots.add(slot)
        configured.append(
            {
                "slot": slot,
                "stage": stage,
                "resource_reference": reference,
                "strength": float(strength),
                "active": active,
            }
        )
    return tuple(configured)


def _rehydrate_active_loras(value: object) -> tuple[Wan22StageLora, ...]:
    if not isinstance(value, list):
        raise TypeError("native Wan worker active_loras must be a list")
    active: list[Wan22StageLora] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "slot", "stage", "resource_id", "path", "strength", "size_bytes", "mtime_ns",
            "header_sha256", "schema_sha256"
        }:
            raise ValueError("native Wan worker active LoRA fields are not canonical")
        path_value = item["path"]
        if not isinstance(path_value, str):
            raise TypeError("native Wan worker active LoRA path is invalid")
        path = Path(path_value).resolve(strict=True)
        identity = ArtifactIdentity(
            path=path,
            size_bytes=_strict_int(item["size_bytes"], "lora.size_bytes"),
            mtime_ns=_strict_int(item["mtime_ns"], "lora.mtime_ns"),
            header_sha256=_strict_sha256(item["header_sha256"], "lora.header_sha256"),
        )
        probe = probe_artifact(path)
        if (
            probe.identity != identity
            or probe.format != "safetensors"
            or item["schema_sha256"] != probe.schema_sha256
            or item["stage"] not in {"high", "low"}
            or not isinstance(item["slot"], str)
            or not isinstance(item["resource_id"], str)
            or isinstance(item["strength"], bool)
            or not isinstance(item["strength"], (int, float))
            or not math.isfinite(float(item["strength"]))
            or float(item["strength"]) == 0.0
        ):
            raise ValueError("native Wan worker active LoRA identity or contract is invalid")
        active.append(
            Wan22StageLora(
                slot=item["slot"], stage=item["stage"], resource_id=item["resource_id"], path=path,
                strength=float(item["strength"]), identity=identity, schema_sha256=probe.schema_sha256,
            )
        )
    if len({item.slot for item in active}) != len(active):
        raise ValueError("native Wan worker active LoRA slots must be unique")
    return tuple(active)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"native Wan worker {label} must be a nonnegative integer")
    return value


def _strict_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"native Wan worker {label} must be a SHA256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"native Wan worker {label} must be a SHA256 hex string") from exc
    return value


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
                    "pipeline support must declare family='wan22' and component='pipeline_support'"
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
            errors.append(
                "high- and low-noise transformers must use one matching format and contract"
            )
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
    if len(required_paths) != len(set(required_paths)) or len(required_ids) != len(
        set(required_ids)
    ):
        errors.append(
            "all required Wan roles must resolve to distinct resources and canonical paths"
        )
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
    if include_adapter_plans and (recipe.pipeline_support is None or generic.support_plan is None):
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
                errors.append(f"native {role} does not support stored contract {contract!r}")
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
        if high is not None and low is not None and high.artifact_contract != low.artifact_contract:
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
    *,
    loras: tuple[Any, ...] = (),
    configured_loras: tuple[ConfiguredLora, ...] = (),
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
    identities = {role: validation.adapter_plans[role].identity for role in _ARTIFACT_ROLES}
    active_loras, configured_lora_manifest = _build_stage_loras(
        loras,
        configured_loras,
        recipe.lora_stage_by_slot,
        recipe.operation,
    )
    return Wan22RuntimeRequest(
        4,
        "wan22",
        _PROBE_SIGNATURE,
        recipe.base_model,
        components,
        identities,
        validation.support_plan,
        validation.adapter_plans,
        recipe.operation,
        configured_lora_manifest,
        active_loras,
    )


def _build_stage_loras(
    loras: tuple[Any, ...],
    configured_loras: tuple[ConfiguredLora, ...],
    stage_by_slot: Mapping[str, str],
    operation: str = "comfy_i2v_base",
) -> tuple[tuple[Wan22StageLora, ...], tuple[dict[str, str | float | bool], ...]]:
    """Bind active generic selections to fixed high/low recipe stages.

    A configured zero-strength slot remains visible as declarative provenance but
    is deliberately not resolved, probed, fingerprinted, or handed to a worker.
    """

    mapping = dict(stage_by_slot)
    if any(stage not in {"high", "low"} for stage in mapping.values()):
        raise ValueError("Wan recipe LoRA slots must target high or low stage")
    active_by_slot = {str(item.slot): item for item in loras}
    if len(active_by_slot) != len(loras):
        raise ValueError("Wan active LoRA slots must be unique")
    if set(active_by_slot) - set(mapping):
        raise ValueError("Wan active LoRA slot is not declared by this recipe")
    configured: list[dict[str, str | float | bool]] = []
    seen_slots: set[str] = set()
    for item in configured_loras:
        if item.slot not in mapping:
            raise ValueError("Wan configured LoRA slot is not declared by this recipe")
        if item.slot in seen_slots:
            raise ValueError("Wan configured LoRA slots must be unique")
        seen_slots.add(item.slot)
        configured.append(
            {
                "slot": item.slot,
                "stage": mapping[item.slot],
                "resource_reference": item.resource_reference,
                "strength": float(item.strength),
                "active": bool(item.active),
            }
        )
    if set(active_by_slot) != {item["slot"] for item in configured if item["active"]}:
        raise ValueError("Wan active LoRA stack does not match its configured slots")
    configured_by_slot = {str(item["slot"]): item for item in configured}
    for item in loras:
        configured_item = configured_by_slot[str(item.slot)]
        if (
            configured_item["resource_reference"] != str(item.resource_id)
            or configured_item["strength"] != float(item.strength)
            or configured_item["active"] is not True
        ):
            raise ValueError("Wan active LoRA does not match its configured slot")
    if operation == "comfy_i2v_lightx2v_4step":
        expected_slots = set(_LIGHTX2V_REQUIRED_LORAS)
        if mapping != {"high_noise": "high", "low_noise": "low"}:
            raise ValueError("LightX2V requires fixed high/low stage bindings")
        if {item["slot"] for item in configured} != expected_slots:
            raise ValueError("LightX2V requires its exact paired LoRA slots")
        for item in configured:
            if (
                item["resource_reference"] != _LIGHTX2V_REQUIRED_LORAS[item["slot"]]
                or item["strength"] != 1.0
                or item["active"] is not True
            ):
                raise ValueError("LightX2V requires the exact official pair at strength 1")
    active: list[Wan22StageLora] = []
    for item in loras:
        if float(item.strength) == 0.0:
            raise ValueError("zero-strength Wan LoRA must not reach runtime planning")
        path = Path(item.path).resolve(strict=True)
        probe = probe_artifact(path)
        if probe.format != "safetensors":
            raise ValueError("native Wan LoRA must use SafeTensors")
        expected_schema = getattr(item, "expected_schema_sha256", None)
        if expected_schema is not None and probe.schema_sha256 != expected_schema:
            raise ValueError("native Wan LoRA schema changed after selection")
        active.append(
            Wan22StageLora(
                slot=str(item.slot),
                stage=mapping[str(item.slot)],
                resource_id=str(item.resource_id),
                path=path,
                strength=float(item.strength),
                identity=probe.identity,
                schema_sha256=probe.schema_sha256,
            )
        )
    if operation == "comfy_i2v_lightx2v_4step" and {
        item.slot: item.resource_id for item in active
    } != dict(_LIGHTX2V_REQUIRED_LORAS):
        raise ValueError("LightX2V active LoRAs do not match the official pair")
    return tuple(active), tuple(configured)


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
    try:
        wan22_i2v_operation(request.operation)
    except ValueError:
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
    for lora in request.active_loras:
        try:
            probe = probe_artifact(lora.path)
        except (OSError, TypeError, ValueError):
            return False
        if (
            probe.identity != lora.identity
            or probe.format != "safetensors"
            or probe.schema_sha256 != lora.schema_sha256
            or lora.strength == 0.0
        ):
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
        "quantization_contract": str(component.resource.metadata["quantization_contract"]),
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
