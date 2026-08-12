"""Exact official Comfy component recipe for Wan 2.2 TI2V 5B."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from .artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from .resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)

WAN5_COMFY_SOURCE_REVISION = "725e6ec60621c6f001af04769173e7dbb3c53541"
WAN5_COMFY_RUNTIME_REVISION = "eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f"
WAN5_COMFY_EXAMPLES_REVISION = "f9431bb000ce792094ff345446e22cac1ea6cef3"
WAN5_T2V_WORKFLOW_SHA256 = "e7913b6b2c8f7d82a6a6f9940289bf6e7513cc908bbf455e4553de9804c6f571"
WAN5_TRANSFORMER_SCHEMA_SHA256 = "5317bf88f8ab6a8acdc58e697c954a43aceecc7b658735e81dccc308af59ef90"
WAN5_TEXT_ENCODER_SCHEMA_SHA256 = "06886ca9d814dd3e89d5d1a90811eef984dbb796440ec37d726b75d89ae2bbe3"
WAN5_VAE_SCHEMA_SHA256 = "f01c9f6cada88c48a74a8b14f129bc75c3d1b7e36a3c3aeaf45ff4f9b1b1b8e9"

Wan5Operation = Literal["text_to_video"]
_ROLES = frozenset({"transformer", "text_encoder", "vae"})


@dataclass(frozen=True, slots=True)
class Wan5RecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class Wan5ComfyRecipe:
    operation: Wan5Operation
    base_model: str
    transformer: Wan5RecipeComponent
    text_encoder: Wan5RecipeComponent
    vae: Wan5RecipeComponent


@dataclass(frozen=True, slots=True)
class Wan5RecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, Wan5RecipeComponent]
    probes: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Wan5RuntimeRequest:
    schema_version: int
    family: str
    operation: Wan5Operation
    base_model: str
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity] = field(repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        components = MappingProxyType(
            {role: MappingProxyType(dict(value)) for role, value in self.components.items()}
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        payload = {
            "schema_version": self.schema_version,
            "family": self.family,
            "operation": self.operation,
            "base_model": self.base_model,
            "comfy_source_revision": WAN5_COMFY_SOURCE_REVISION,
            "comfy_runtime_revision": WAN5_COMFY_RUNTIME_REVISION,
            "workflow_revision": WAN5_COMFY_EXAMPLES_REVISION,
            "workflow_sha256": workflow_sha256(self.operation),
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "fingerprint", f"wan22-ti2v5b-comfy:sha256:{digest}")

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {
            role: {key: value for key, value in value.items() if key != "path"}
            for role, value in self.components.items()
        }


def workflow_sha256(operation: Wan5Operation) -> str:
    if operation != "text_to_video":
        raise ValueError(f"unsupported Wan 5B operation: {operation}")
    return WAN5_T2V_WORKFLOW_SHA256


def validate_wan5_comfy_recipe(
    recipe: Wan5ComfyRecipe,
    inventory: ResourceInventory,
) -> Wan5RecipeValidation:
    errors: list[str] = []
    resolved: dict[str, Wan5RecipeComponent] = {}
    probes: dict[str, object] = {}
    expected = {
        "transformer": (
            ArtifactPrecision.FP16,
            "native/fp16",
            "wan22_ti2v_5b_48ch_30block",
            WAN5_TRANSFORMER_SCHEMA_SHA256,
        ),
        "text_encoder": (
            ArtifactPrecision.FP8,
            "comfy_legacy/scaled_fp8_e4m3fn",
            "umt5_xxl",
            WAN5_TEXT_ENCODER_SCHEMA_SHA256,
        ),
        "vae": (
            ArtifactPrecision.FP16,
            "native/fp16",
            "wan_vae_2_2_48ch",
            WAN5_VAE_SCHEMA_SHA256,
        ),
    }
    for role in sorted(_ROLES):
        requested = getattr(recipe, role)
        actual = inventory.by_id().get(requested.resource.id)
        if actual is None:
            errors.append(f"{role} resource is not owned by this inventory")
            continue
        try:
            path = inventory.path_for(actual.id).resolve(strict=True)
        except (KeyError, OSError) as exc:
            errors.append(f"{role} inventory path is unavailable: {exc}")
            continue
        if actual != requested.resource or path != requested.path.resolve(strict=False):
            errors.append(f"{role} descriptor/path does not match the inventory mapping")
            continue
        resolved[role] = Wan5RecipeComponent(actual, path)
        precision, contract, architecture, schema = expected[role]
        if (
            actual.kind != ResourceKind.MODEL
            or actual.family != "wan22"
            or actual.component != role
            or actual.format != ResourceFormat.SAFETENSORS
            or actual.precision != precision
            or actual.quantization != ArtifactQuantization.NATIVE
            or actual.metadata.get("quantization_contract") != contract
            or actual.metadata.get("architecture") != architecture
            or actual.base_model != recipe.base_model
        ):
            errors.append(f"{role} descriptor does not match the Wan 5B Comfy contract")
            continue
        try:
            probe = probe_artifact(path)
            probes[role] = probe
            if probe.quantization_contract != contract:
                errors.append(f"{role} stored precision contract does not match its descriptor")
            if architecture not in probe.architecture_signals:
                errors.append(f"{role} header does not expose the required architecture")
            if role not in probe.component_signals:
                errors.append(f"{role} header does not expose the required component role")
            if probe.schema_sha256 != schema:
                errors.append(f"{role} schema fingerprint is not the pinned official artifact")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{role} probe failed: {exc}")
    if set(resolved) != _ROLES:
        errors.append("Wan 5B recipe did not resolve all three exact component roles")
    paths = [component.path for component in resolved.values()]
    if len(paths) != len(set(paths)):
        errors.append("Wan 5B recipe roles must resolve to distinct files")
    return Wan5RecipeValidation(
        not errors,
        tuple(errors),
        MappingProxyType(resolved),
        MappingProxyType(probes),
    )


def build_wan5_comfy_runtime_request(
    recipe: Wan5ComfyRecipe,
    inventory: ResourceInventory,
) -> Wan5RuntimeRequest:
    validation = validate_wan5_comfy_recipe(recipe, inventory)
    if not validation.available:
        raise ValueError("Wan 5B Comfy recipe is unavailable: " + "; ".join(validation.errors))
    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in sorted(_ROLES):
        component = validation.resolved[role]
        probe = validation.probes[role]
        identity = probe.identity
        identities[role] = identity
        components[role] = {
            "resource_id": component.resource.id,
            "path": str(component.path),
            "format": "safetensors",
            "component": role,
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
            "schema_sha256": probe.schema_sha256,
            "quantization_contract": str(component.resource.metadata["quantization_contract"]),
        }
    return Wan5RuntimeRequest(1, "wan22", recipe.operation, recipe.base_model, components, identities)


def revalidate_wan5_runtime_request(request: Wan5RuntimeRequest) -> bool:
    if set(request.components) != _ROLES or set(request.identities) != _ROLES:
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
    return True
