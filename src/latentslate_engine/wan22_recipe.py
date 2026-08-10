"""CPU-only validation for explicit, inventory-owned Wan 2.2 I2V recipes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .artifacts import ArtifactIdentity, ArtifactProbe, probe_artifact, revalidate_artifact
from .resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceInventory,
    ResourceKind,
)

_PROBE_SIGNATURE = "wan22_14b_36ch_40block_out16"
_DECLARED_ARCHITECTURES = {"wan2.2_i2v_14b": _PROBE_SIGNATURE, "wan": _PROBE_SIGNATURE}


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


@dataclass(frozen=True, slots=True)
class Wan22RecipeValidation:
    available: bool
    errors: tuple[str, ...]
    probes: tuple[ArtifactProbe, ...]
    resolved: dict[str, Wan22RecipeComponent]


@dataclass(frozen=True, slots=True)
class Wan22RuntimeRequest:
    """Validated paths and identities to re-check immediately before execution."""

    schema_version: int
    family: str
    architecture: str
    base_model: str
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            MappingProxyType(
                {role: MappingProxyType(dict(component)) for role, component in self.components.items()}
            ),
        )
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))

    def to_json_dict(self) -> dict[str, object]:
        """Return a deep-copyable JSON-safe execution manifest."""

        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "components": {role: dict(component) for role, component in self.components.items()},
        }


def validate_wan22_i2v_14b_recipe(
    recipe: Wan22I2VRecipe, inventory: ResourceInventory
) -> Wan22RecipeValidation:
    """Resolve inventory resources then validate headers, contracts, and identities.

    The inventory binding closes path injection. It cannot eliminate a filesystem
    replacement after validation, so callers must call ``revalidate_runtime_request``
    immediately before opening model files for execution.
    """

    errors: list[str] = []
    requested = (
        ("transformer_high_noise", "high-noise transformer", recipe.high_noise, "high"),
        ("transformer_low_noise", "low-noise transformer", recipe.low_noise, "low"),
        ("text_encoder", "text encoder", recipe.text_encoder, None),
        ("vae", "VAE", recipe.vae, None),
    )
    resolved: dict[str, Wan22RecipeComponent] = {}
    probes: dict[str, ArtifactProbe] = {}
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
            errors.append("high- and low-noise transformers must declare a mapped canonical architecture")
        if {probe.architecture_signals for role, probe in probes.items() if role.startswith("transformer")} != {(_PROBE_SIGNATURE,)}:
            errors.append("high- and low-noise headers must expose the same exact architecture signature")
        if (
            probes.get("transformer_high_noise") is not None
            and probes.get("transformer_low_noise") is not None
            and probes["transformer_high_noise"].schema_sha256
            != probes["transformer_low_noise"].schema_sha256
        ):
            errors.append("high- and low-noise transformers must share one topology/schema fingerprint")
    required_paths = [component.path.resolve() for component in resolved.values()]
    required_ids = [component.resource.id for component in resolved.values()]
    if len(required_paths) != len(set(required_paths)) or len(required_ids) != len(set(required_ids)):
        errors.append("all required Wan roles must resolve to distinct resources and canonical paths")
    return Wan22RecipeValidation(not errors, tuple(errors), tuple(probes.values()), resolved)


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
    }
    identities = {role: probe_by_path[component.path].identity for role, component in validation.resolved.items()}
    return Wan22RuntimeRequest(1, "wan22", _PROBE_SIGNATURE, recipe.base_model, components, identities)


def revalidate_runtime_request(request: Wan22RuntimeRequest) -> bool:
    """Perform the required pre-open TOCTOU check for every serialized artifact."""

    if set(request.components) != set(request.identities):
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
    return True


def _resolve_inventory_component(
    inventory: ResourceInventory, requested: Wan22RecipeComponent, label: str, errors: list[str]
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
        "comfy_quant/int8_tensorwise_convrot": (ArtifactPrecision.UNKNOWN, ArtifactQuantization.INT8),
        "comfy_quant/float8_e4m3fn": (ArtifactPrecision.FP8, ArtifactQuantization.NATIVE),
        "comfy_legacy/scaled_fp8_e4m3fn": (ArtifactPrecision.FP8, ArtifactQuantization.NATIVE),
        "native/bf16": (ArtifactPrecision.BF16, ArtifactQuantization.NATIVE),
        "native/fp16": (ArtifactPrecision.FP16, ArtifactQuantization.NATIVE),
        "native/fp32": (ArtifactPrecision.FP32, ArtifactQuantization.NATIVE),
        "gguf/q5_k_m": (ArtifactPrecision.UNKNOWN, ArtifactQuantization.GGUF),
    }.get(declared)
    if expected is None or (resource.precision, resource.quantization) != expected:
        errors.append(f"{label} descriptor precision/quantization does not match its proven contract")


def _validate_role_architecture(
    role: str, resource: ResourceDescriptor, probe: ArtifactProbe, label: str, errors: list[str]
) -> None:
    expected = _PROBE_SIGNATURE if role.startswith("transformer") else ("umt5_xxl" if role == "text_encoder" else "wan_vae_2_1")
    declared = _DECLARED_ARCHITECTURES.get(str(resource.metadata.get("architecture"))) if role.startswith("transformer") else resource.metadata.get("architecture")
    if declared != expected or expected not in probe.architecture_signals:
        errors.append(f"{label} declared architecture does not match its exact header signature")


def _runtime_component(component: Wan22RecipeComponent, probe: ArtifactProbe) -> dict[str, str | int]:
    identity = probe.identity
    return {"resource_id": component.resource.id, "path": str(component.path), "format": component.resource.format.value, "component": component.resource.component or "", "quantization_contract": str(component.resource.metadata["quantization_contract"]), "size_bytes": identity.size_bytes, "mtime_ns": identity.mtime_ns, "header_sha256": identity.header_sha256, "schema_sha256": probe.schema_sha256}
