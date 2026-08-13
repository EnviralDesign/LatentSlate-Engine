"""Exact official Comfy LTX 2.3 optimized-component contracts.

This module deliberately models the Comfy graphs as separate operations.  The
Dev-FP8-plus-Distilled-LoRA T2V/I2V topology is not interchangeable with the
Distilled-FP8 first+last-frame topology, and neither is a converted form of the
native Diffusers BF16 reference closure.
"""

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

LTX23_COMFY_TEMPLATE_REVISION = "8b2c08f297c63ffc73ce93f938b0f5139c0ed73f"
LTX23_COMFY_RUNTIME_REVISION = "eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f"
LTX23_COMFY_T2V_TEMPLATE_SHA256 = "75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f"
LTX23_COMFY_I2V_TEMPLATE_SHA256 = "91dd8e44926fd37f6d9307789484370fa333582b14e53ed771d63ed805379ee4"
LTX23_COMFY_FLF_TEMPLATE_SHA256 = "168bc2584ef117133e76341f04e001aab2641b72b75d81b66b5c0b66e56c24a5"

LTX23_COMFY_FP8_REVISION = "1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1"
LTX23_COMFY_LORA_REVISION = "e14c0e7f46a1d68384214a3c6e6b309a382016dd"
LTX23_COMFY_TEXT_REVISION = "bd5f9c87fcb0360ae7112f9784562670894d9492"
LTX23_COMFY_UPSCALER_REVISION = "6f3520585aa27248020550da2f453aa0c572398c"

LTX23_COMFY_FPS = 24
LTX23_COMFY_MODEL_LORA_STRENGTH = 0.5
LTX23_COMFY_GUIDE_STRENGTH = 0.7
LTX23_COMFY_MAIN_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX23_COMFY_UPSCALE_SIGMAS = (0.85, 0.725, 0.4219, 0.0)

LTX23ComfyOperation = Literal["comfy_dev_t2v", "comfy_dev_i2v", "comfy_distilled_flf"]
_DEV_ROLES = frozenset({"checkpoint", "model_lora", "text_encoder", "text_lora", "latent_upscaler"})
_FLF_ROLES = frozenset({"checkpoint", "text_encoder"})


@dataclass(frozen=True, slots=True)
class LTX23ComfyRecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class LTX23ComfyRecipe:
    operation: LTX23ComfyOperation
    base_model: str
    components: Mapping[str, LTX23ComfyRecipeComponent]


@dataclass(frozen=True, slots=True)
class LTX23ComfyRecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, LTX23ComfyRecipeComponent]
    probes: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LTX23ComfyRuntimeRequest:
    schema_version: int
    family: str
    operation: LTX23ComfyOperation
    base_model: str
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity] = field(repr=False)
    fingerprint: str = field(init=False)
    component_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        components = MappingProxyType(
            {role: MappingProxyType(dict(value)) for role, value in self.components.items()}
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        component_payload = {
            "schema_version": self.schema_version,
            "family": self.family,
            "base_model": self.base_model,
            "template_revision": LTX23_COMFY_TEMPLATE_REVISION,
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        component_hash = hashlib.sha256(
            json.dumps(component_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "component_fingerprint", f"ltx23-comfy-components:sha256:{component_hash}")
        payload = {**component_payload, "operation": self.operation, "template_sha256": template_sha256(self.operation)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        object.__setattr__(self, "fingerprint", f"ltx23-comfy:sha256:{digest}")

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {role: {key: value for key, value in item.items() if key != "path"} for role, item in self.components.items()}


def template_sha256(operation: LTX23ComfyOperation) -> str:
    return {
        "comfy_dev_t2v": LTX23_COMFY_T2V_TEMPLATE_SHA256,
        "comfy_dev_i2v": LTX23_COMFY_I2V_TEMPLATE_SHA256,
        "comfy_distilled_flf": LTX23_COMFY_FLF_TEMPLATE_SHA256,
    }[operation]


def required_roles(operation: LTX23ComfyOperation) -> frozenset[str]:
    return _FLF_ROLES if operation == "comfy_distilled_flf" else _DEV_ROLES


def validate_ltx23_comfy_recipe(recipe: LTX23ComfyRecipe, inventory: ResourceInventory) -> LTX23ComfyRecipeValidation:
    expected = _expected_components(recipe.operation)
    errors: list[str] = []
    resolved: dict[str, LTX23ComfyRecipeComponent] = {}
    probes: dict[str, object] = {}
    if set(recipe.components) != set(expected):
        errors.append(f"LTX 2.3 Comfy {recipe.operation} requires exactly {', '.join(sorted(expected))}")
    for role, contract in expected.items():
        requested = recipe.components.get(role)
        if requested is None:
            continue
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
        resolved[role] = LTX23ComfyRecipeComponent(actual, path)
        kind, component, precision, schema = contract
        if (
            actual.kind != kind
            or actual.family != "ltx23"
            or actual.component != component
            or actual.format != ResourceFormat.SAFETENSORS
            or actual.precision != precision
            or actual.quantization != ArtifactQuantization.NATIVE
            or actual.base_model != recipe.base_model
        ):
            errors.append(f"{role} descriptor does not match the exact LTX 2.3 Comfy contract")
            continue
        try:
            probe = probe_artifact(path)
            probes[role] = probe
            if probe.schema_sha256 != schema:
                errors.append(f"{role} schema fingerprint is not the pinned official artifact")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{role} probe failed: {exc}")
    if set(resolved) != set(expected):
        errors.append("LTX 2.3 Comfy recipe did not resolve every exact component")
    paths = [item.path for item in resolved.values()]
    if len(paths) != len(set(paths)):
        errors.append("LTX 2.3 Comfy recipe roles must resolve to distinct artifacts")
    return LTX23ComfyRecipeValidation(not errors, tuple(errors), MappingProxyType(resolved), MappingProxyType(probes))


def build_ltx23_comfy_runtime_request(recipe: LTX23ComfyRecipe, inventory: ResourceInventory) -> LTX23ComfyRuntimeRequest:
    validation = validate_ltx23_comfy_recipe(recipe, inventory)
    if not validation.available:
        raise ValueError("LTX 2.3 Comfy recipe is unavailable: " + "; ".join(validation.errors))
    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in sorted(validation.resolved):
        item = validation.resolved[role]
        probe = validation.probes[role]
        identities[role] = probe.identity
        components[role] = {
            "resource_id": item.resource.id,
            "path": str(item.path),
            "size_bytes": probe.identity.size_bytes,
            "mtime_ns": probe.identity.mtime_ns,
            "header_sha256": probe.identity.header_sha256,
            "schema_sha256": probe.schema_sha256,
        }
    return LTX23ComfyRuntimeRequest(1, "ltx23", recipe.operation, recipe.base_model, components, identities)


def revalidate_ltx23_comfy_runtime_request(request: LTX23ComfyRuntimeRequest) -> bool:
    if (
        request.schema_version != 1
        or request.family != "ltx23"
        or request.base_model != "Lightricks/LTX-2.3"
        or request.operation not in {"comfy_dev_t2v", "comfy_dev_i2v", "comfy_distilled_flf"}
        or set(request.components) != required_roles(request.operation)
        or set(request.identities) != required_roles(request.operation)
    ):
        return False
    expected = _expected_components(request.operation)
    for role, identity in request.identities.items():
        component = request.components[role]
        if (
            component.get("path") != str(identity.path)
            or component.get("size_bytes") != identity.size_bytes
            or component.get("mtime_ns") != identity.mtime_ns
            or component.get("header_sha256") != identity.header_sha256
            or component.get("schema_sha256") != expected[role][3]
            or not revalidate_artifact(identity)
        ):
            return False
    canonical = LTX23ComfyRuntimeRequest(
        request.schema_version, request.family, request.operation, request.base_model,
        request.components, request.identities,
    )
    return (
        request.component_fingerprint == canonical.component_fingerprint
        and request.fingerprint == canonical.fingerprint
    )


def _expected_components(operation: LTX23ComfyOperation) -> dict[str, tuple[ResourceKind, str, ArtifactPrecision, str]]:
    common = {
        "text_encoder": (ResourceKind.MODEL, "text_encoder", ArtifactPrecision.FP4, "ddf523b18b1a724da6d4a3b0a97d4305ad3ad02a89ab7ada299663a9047040cd"),
    }
    if operation == "comfy_distilled_flf":
        return {"checkpoint": (ResourceKind.MODEL, "checkpoint", ArtifactPrecision.FP8, "124c441187373cba2d758847ec2254fa28d3e6fc6f9bec292d905139732a5d73"), **common}
    if operation not in {"comfy_dev_t2v", "comfy_dev_i2v"}:
        raise ValueError(f"unsupported LTX 2.3 Comfy operation {operation!r}")
    return {
        "checkpoint": (ResourceKind.MODEL, "checkpoint", ArtifactPrecision.FP8, "0a69321952b31131924aef3b568f759cb7c25d3d2738467973976bbf2061e746"),
        "model_lora": (ResourceKind.LORA, "model_lora", ArtifactPrecision.BF16, "f5d65b851a5e6fe5eb7ad4e0e4e2051ff9d36bcea75557e72923f606de51134f"),
        "text_lora": (ResourceKind.LORA, "text_lora", ArtifactPrecision.BF16, "601c8857a7d830f05f80792e044f97df6df8ff125079d5a305f3de5a2999d027"),
        "latent_upscaler": (ResourceKind.MODEL, "latent_upscaler", ArtifactPrecision.BF16, "ccd7989113ce24be08ef9cfcaf135ff85594f3f44e34be231dd4b426d4482c34"),
        **common,
    }
