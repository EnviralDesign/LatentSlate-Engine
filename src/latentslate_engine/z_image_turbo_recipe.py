"""Fail-closed component contract for the official Z-Image Turbo INT8 graph.

This is deliberately a *contract* rather than a generic Z-Image loader.  The
only supported operation is the three-file pinned Turbo text-to-image contract
pinned in ``docs/model-roadmaps/Z_IMAGE_TURBO.md``.  In particular, an INT8
file is not allowed to become a dense/dequantized fallback merely because its
header is superficially readable.
"""

from __future__ import annotations

import hashlib
import json
import struct
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
from .stored_quant import StoredQuantizedLayer

ZImageOperation = Literal["zimage_turbo_t2i_int8_convrot"]
Z_IMAGE_OPERATION: ZImageOperation = "zimage_turbo_t2i_int8_convrot"
Z_IMAGE_TRANSFORMER_CONTRACT = "comfy_quant/int8_tensorwise_convrot"
Z_IMAGE_MIXED_QWEN_CONTRACT = "comfy_quant/qwen3_4b_fp8_mixed"
Z_IMAGE_VAE_CONTRACT = "native/fp32"
_ROLES = frozenset({"transformer", "text_encoder", "vae"})
_SCHEDULE = {
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "guidance_scale": 1.0,
    "sampling": "auraflow_shift_3",
    "sampler": "res_multistep",
    "scheduler": "simple",
    "negative_conditioning": "zero_out_positive",
}
_IMMUTABLE_COMPONENTS = {
    "transformer": (
        6201001296,
        "d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e",
        "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
        "be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635",
    ),
    "text_encoder": (
        5631994051,
        "2f862278568d3f0a83167a16e5f11094da6dee72",
        "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15",
    ),
    "vae": (
        335304388,
        "93fae7d7f6189cc408fdd7cec36c91447b8506a2",
        "split_files/vae/ae.safetensors",
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
    ),
}
_Z_TRANSFORMER_HEADER_SHA256 = "01e93cae3aa75eb2106025889f1a78df19628a95c433b45d9447562b04907814"
_Z_QWEN_HEADER_SHA256 = "7537b0cd31f4fc963d334b4f997cedee6f51c62aa8518b7b7a852b182144aed9"
_Z_TRANSFORMER_INT8_LAYERS = 202


@dataclass(frozen=True, slots=True)
class ZImageTurboRecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class ZImageTurboRecipe:
    base_model: str
    transformer: ZImageTurboRecipeComponent
    text_encoder: ZImageTurboRecipeComponent
    vae: ZImageTurboRecipeComponent
    operation: ZImageOperation = Z_IMAGE_OPERATION
    width: int = 1024
    height: int = 1024
    steps: int = 8
    guidance_scale: float = 1.0
    sampling: str = "auraflow_shift_3"
    sampler: str = "res_multistep"
    scheduler: str = "simple"


@dataclass(frozen=True, slots=True)
class ZImageTransformerPlan:
    """Header-bound INT8 layer closure; no converted checkpoint is produced."""

    identity: ArtifactIdentity
    schema_sha256: str
    source_to_target: Mapping[str, str]
    stored_layers: Mapping[str, StoredQuantizedLayer]

    @property
    def stored_layer_count(self) -> int:
        return len(self.stored_layers)

    def require_stored_layout(self) -> None:
        if len(self.stored_layers) != _Z_TRANSFORMER_INT8_LAYERS or dict(self.source_to_target) != {
            key: key for key in self.stored_layers
        }:
            raise ValueError(
                "Z-Image transformer does not retain the complete stored ConvRot mapping"
            )
        if not self.stored_layers:
            raise ValueError("Z-Image transformer has no stored ConvRot layers")
        if any(
            layer.contract != Z_IMAGE_TRANSFORMER_CONTRACT for layer in self.stored_layers.values()
        ):
            raise ValueError("Z-Image transformer has an unsupported stored ConvRot layout")


@dataclass(frozen=True, slots=True)
class ZImageDensePlan:
    identity: ArtifactIdentity
    schema_sha256: str
    role: Literal["text_encoder", "vae"]
    tensor_count: int


@dataclass(frozen=True, slots=True)
class ZImageTurboRecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, ZImageTurboRecipeComponent]
    plans: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ZImageTurboRuntimeRequest:
    schema_version: int
    base_model: str
    operation: ZImageOperation
    schedule: Mapping[str, str | int | float]
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity]
    plans: Mapping[str, Any] = field(repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        components = MappingProxyType(
            {role: MappingProxyType(dict(value)) for role, value in self.components.items()}
        )
        payload = {
            "schema_version": self.schema_version,
            "base_model": self.base_model,
            "operation": self.operation,
            "schedule": dict(sorted(self.schedule.items())),
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        object.__setattr__(self, "plans", MappingProxyType(dict(self.plans)))
        object.__setattr__(self, "fingerprint", f"z-image-turbo:sha256:{digest}")

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {
            role: {key: value for key, value in item.items() if key != "path"}
            for role, item in self.components.items()
        }


def z_image_turbo_schedule(recipe: ZImageTurboRecipe) -> dict[str, str | int | float]:
    actual = {
        "width": recipe.width,
        "height": recipe.height,
        "steps": recipe.steps,
        "guidance_scale": recipe.guidance_scale,
        "sampling": recipe.sampling,
        "sampler": recipe.sampler,
        "scheduler": recipe.scheduler,
        "negative_conditioning": _SCHEDULE["negative_conditioning"],
    }
    if actual != _SCHEDULE:
        raise ValueError(
            "Z-Image Turbo requires the exact pinned schedule: 1024x1024, 8 steps, CFG 1, AuraFlow shift 3, res_multistep/simple"
        )
    return actual


def validate_z_image_turbo_recipe(
    recipe: ZImageTurboRecipe, inventory: ResourceInventory, *, include_plans: bool = True
) -> ZImageTurboRecipeValidation:
    errors: list[str] = []
    resolved: dict[str, ZImageTurboRecipeComponent] = {}
    plans: dict[str, Any] = {}
    try:
        z_image_turbo_schedule(recipe)
    except ValueError as exc:
        errors.append(str(exc))
    if recipe.operation != Z_IMAGE_OPERATION:
        errors.append("Z-Image Turbo supports only the pinned Turbo T2I operation")
    for role in sorted(_ROLES):
        component = _resolve_component(inventory, getattr(recipe, role), role, errors)
        if component is None:
            continue
        resolved[role] = component
        expected = _descriptor_contract(role, recipe.base_model)
        _validate_descriptor(component.resource, role, expected, errors)
        if include_plans and component.resource.available:
            try:
                plans[role] = _plan_component(role, component.path)
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"{role} contract failed: {exc}")
    paths = [component.path.resolve(strict=False) for component in resolved.values()]
    if len(paths) != len(set(paths)) or len(
        {item.resource.id for item in resolved.values()}
    ) != len(resolved):
        errors.append("Z-Image component roles must use distinct resources and paths")
    if set(resolved) != _ROLES:
        errors.append("Z-Image recipe did not resolve all three exact component roles")
    return ZImageTurboRecipeValidation(
        not errors, tuple(errors), MappingProxyType(resolved), MappingProxyType(plans)
    )


def build_z_image_turbo_runtime_request(
    recipe: ZImageTurboRecipe, inventory: ResourceInventory
) -> ZImageTurboRuntimeRequest:
    validation = validate_z_image_turbo_recipe(recipe, inventory)
    if not validation.available or set(validation.plans) != _ROLES:
        raise ValueError("Z-Image Turbo recipe is unavailable: " + "; ".join(validation.errors))
    identities: dict[str, ArtifactIdentity] = {}
    components: dict[str, dict[str, str | int]] = {}
    for role in sorted(_ROLES):
        component = validation.resolved[role]
        plan = validation.plans[role]
        identities[role] = plan.identity
        source = component.resource.sources[0]
        components[role] = {
            "resource_id": component.resource.id,
            "path": str(component.path.resolve()),
            "format": component.resource.format.value,
            "component": role,
            "size_bytes": plan.identity.size_bytes,
            "mtime_ns": plan.identity.mtime_ns,
            "header_sha256": plan.identity.header_sha256,
            "schema_sha256": plan.schema_sha256,
            "quantization_contract": str(component.resource.metadata["quantization_contract"]),
        }
        components[role].update(
            {
                "source_revision": str(source.revision),
                "source_sha256": str(source.sha256),
                "source_filename": str(source.filename),
            }
        )
    return ZImageTurboRuntimeRequest(
        1,
        recipe.base_model,
        recipe.operation,
        z_image_turbo_schedule(recipe),
        components,
        identities,
        validation.plans,
    )


def revalidate_z_image_turbo_runtime_request(request: ZImageTurboRuntimeRequest) -> bool:
    if (
        request.operation != Z_IMAGE_OPERATION
        or set(request.components) != _ROLES
        or set(request.identities) != _ROLES
        or set(request.plans) != _ROLES
    ):
        return False
    try:
        if dict(request.schedule) != _SCHEDULE:
            return False
    except (TypeError, ValueError):
        return False
    for role, identity in request.identities.items():
        component = request.components[role]
        plan = request.plans[role]
        _, revision, filename, sha256 = _IMMUTABLE_COMPONENTS[role]
        if (
            component.get("path") != str(identity.path)
            or component.get("header_sha256") != identity.header_sha256
            or plan.identity != identity
            or not revalidate_artifact(identity)
        ):
            return False
        if (
            component.get("source_revision") != revision
            or component.get("source_filename") != filename
            or component.get("source_sha256") != sha256
        ):
            return False
        if role == "transformer":
            if not isinstance(plan, ZImageTransformerPlan):
                return False
            try:
                plan.require_stored_layout()
                refreshed = _plan_transformer(identity.path)
            except (OSError, TypeError, ValueError):
                return False
            if refreshed != plan:
                return False
        elif role == "text_encoder":
            from .runtime.z_image_mixed_qwen import revalidate_z_image_mixed_qwen

            if not revalidate_z_image_mixed_qwen(plan):
                return False
    return True


def _resolve_component(
    inventory: ResourceInventory,
    requested: ZImageTurboRecipeComponent,
    role: str,
    errors: list[str],
) -> ZImageTurboRecipeComponent | None:
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
    return ZImageTurboRecipeComponent(actual, path)


def _descriptor_contract(role: str, base_model: str) -> dict[str, Any]:
    precision = (
        ArtifactPrecision.FP8
        if role == "text_encoder"
        else ArtifactPrecision.FP32
        if role == "vae"
        else ArtifactPrecision.UNKNOWN
    )
    quantization = (
        ArtifactQuantization.INT8 if role == "transformer" else ArtifactQuantization.NATIVE
    )
    contract = (
        Z_IMAGE_TRANSFORMER_CONTRACT
        if role == "transformer"
        else Z_IMAGE_MIXED_QWEN_CONTRACT
        if role == "text_encoder"
        else Z_IMAGE_VAE_CONTRACT
    )
    return {
        "precision": precision,
        "quantization": quantization,
        "contract": contract,
        "architecture": f"z_image_turbo_{role}",
        "base_model": base_model if role == "transformer" else None,
    }


def _validate_descriptor(
    resource: ResourceDescriptor, role: str, expected: Mapping[str, Any], errors: list[str]
) -> None:
    if resource.kind != ResourceKind.MODEL or not resource.available:
        errors.append(f"{role} must be an available model resource")
    if resource.family != "zimage" or resource.component != role:
        errors.append(f"{role} must declare family='zimage' and component={role!r}")
    if (
        resource.format != ResourceFormat.SAFETENSORS
        or resource.precision != expected["precision"]
        or resource.quantization != expected["quantization"]
    ):
        errors.append(f"{role} stored format/precision/quantization metadata is incorrect")
    if (
        resource.metadata.get("quantization_contract") != expected["contract"]
        or resource.metadata.get("architecture") != expected["architecture"]
    ):
        errors.append(f"{role} immutable metadata contract is incorrect")
    if expected["base_model"] is not None and resource.base_model != expected["base_model"]:
        errors.append(f"{role} base_model does not match recipe base_model")
    size, revision, filename, sha256 = _IMMUTABLE_COMPONENTS[role]
    if resource.size_bytes != size:
        errors.append(f"{role} size does not match the pinned immutable artifact")
    if len(resource.sources) != 1 or not resource.sources[0].is_exact():
        errors.append(f"{role} must have one exact immutable source")
    else:
        source = resource.sources[0]
        if (source.repo_id, source.revision, source.filename, source.sha256) != (
            "Comfy-Org/z_image_turbo",
            revision,
            filename,
            sha256,
        ):
            errors.append(f"{role} source does not match the pinned immutable artifact")


def _plan_component(role: str, path: Path) -> Any:
    if role == "transformer":
        return _plan_transformer(path)
    if role == "text_encoder":
        from .runtime.z_image_mixed_qwen import plan_z_image_mixed_qwen

        return plan_z_image_mixed_qwen(path)
    return _plan_dense(role, path)


def _plan_transformer(path: Path) -> ZImageTransformerPlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors":
        raise ValueError("transformer is not SafeTensors")
    raw_header, header = _read_z_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    if hashlib.sha256(raw_header).hexdigest() != _Z_TRANSFORMER_HEADER_SHA256:
        raise ValueError("transformer header differs from the exact official ConvRot mapping")
    if header.get("__metadata__") != {}:
        raise ValueError(
            "official transformer requires empty global metadata; ConvRot facts are per-layer"
        )
    int8_weights = tuple(
        sorted(
            key
            for key, value in header.items()
            if key.endswith(".weight") and isinstance(value, dict) and value.get("dtype") == "I8"
        )
    )
    if len(int8_weights) != _Z_TRANSFORMER_INT8_LAYERS:
        raise ValueError("transformer does not have the complete 202-layer INT8 ConvRot mapping")
    layers: dict[str, StoredQuantizedLayer] = {}
    for key in int8_weights:
        stem = key.removesuffix(".weight")
        scale = header.get(stem + ".weight_scale")
        marker = header.get(stem + ".comfy_quant")
        weight = header[key]
        if (
            not isinstance(scale, dict)
            or not isinstance(marker, dict)
            or not isinstance(weight, dict)
        ):
            raise TypeError(f"transformer ConvRot sidecars are incomplete: {stem}")
        rows = weight.get("shape", [None])[0]
        if scale.get("dtype") != "F32" or scale.get("shape") != [rows, 1]:
            raise ValueError(f"transformer ConvRot scale geometry is invalid: {stem}")
        marker_payload = _read_u8_payload(
            probe.identity.path, probe.identity.size_bytes, raw_header, marker
        )
        try:
            marker_value = json.loads(marker_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"transformer ConvRot marker is invalid JSON: {stem}") from exc
        group_size = (
            marker_value.get("convrot_groupsize") if isinstance(marker_value, dict) else None
        )
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or marker_value.get("format") != "int8_tensorwise"
            or marker_value.get("convrot") is not True
            or group_size <= 0
            or not isinstance(weight.get("shape"), list)
            or weight["shape"][1] % group_size
        ):
            raise ValueError(f"transformer ConvRot marker/group geometry is invalid: {stem}")
        layers[key] = StoredQuantizedLayer(
            probe.identity.path,
            probe.identity,
            key,
            Z_IMAGE_TRANSFORMER_CONTRACT,
            stem + ".weight_scale",
            stem + ".comfy_quant",
            group_size,
        )
    if len(layers) != _Z_TRANSFORMER_INT8_LAYERS:
        raise ValueError("transformer ConvRot mapping is incomplete")
    mapping = MappingProxyType({key: key for key in sorted(layers)})
    return ZImageTransformerPlan(
        probe.identity,
        probe.schema_sha256,
        mapping,
        MappingProxyType(dict(layers)),
    )


def _plan_dense(role: str, path: Path) -> ZImageDensePlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors" or probe.tensor_count <= 0:
        raise ValueError(f"{role} is not a readable non-empty SafeTensors artifact")
    raw_header, header = _read_z_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    dtypes = {value.get("dtype") for value in header.values() if isinstance(value, dict)}
    if role == "text_encoder":
        # Comfy calls this file mixed FP8.  It must be structurally explicit, not
        # guessed from its filename: at least one F8 weight plus its sidecars.
        if hashlib.sha256(raw_header).hexdigest() != _Z_QWEN_HEADER_SHA256:
            raise ValueError("mixed Qwen header differs from the exact official storage mapping")
        weights = [
            value
            for key, value in header.items()
            if key.endswith(".weight") and isinstance(value, dict)
        ]
        counts = {
            dtype: sum(value.get("dtype") == dtype for value in weights)
            for dtype in ("BF16", "F8_E4M3", "U8")
        }
        if counts != {"BF16": 209, "F8_E4M3": 177, "U8": 12}:
            raise ValueError(
                "mixed Qwen BF16/FP8/NVFP4 role count differs from the official header"
            )
        fp8 = [
            key
            for key, value in header.items()
            if key.endswith(".weight")
            and isinstance(value, dict)
            and value.get("dtype") == "F8_E4M3"
        ]
        nvfp4 = [
            key
            for key, value in header.items()
            if key.endswith(".weight") and isinstance(value, dict) and value.get("dtype") == "U8"
        ]
        if not all(_fp8_sidecar_geometry(header, key) for key in fp8) or not all(
            _nvfp4_sidecar_geometry(header, key) for key in nvfp4
        ):
            raise ValueError(
                "mixed Qwen sidecar/scale geometry differs from the official storage contract"
            )
    elif role == "vae" and not dtypes.intersection({"F32", "BF16", "F16"}):
        raise ValueError("VAE has no supported native dense tensor precision")
    return ZImageDensePlan(probe.identity, probe.schema_sha256, role, probe.tensor_count)


def _read_z_safetensors_header(path: Path, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Read one bounded SafeTensors header without importing a model runtime."""

    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("SafeTensors header is truncated")
        length = struct.unpack("<Q", prefix)[0]
        if length <= 0 or length > 64 * 1024 * 1024 or length > size_bytes - 8:
            raise ValueError("SafeTensors header exceeds bounded file extent")
        raw = stream.read(length)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SafeTensors header is invalid JSON") from exc
    if not isinstance(header, dict):
        raise TypeError("SafeTensors header must be an object")
    return raw, header


def _read_u8_payload(
    path: Path, size_bytes: int, raw_header: bytes, entry: Mapping[str, Any]
) -> bytes:
    if (
        entry.get("dtype") != "U8"
        or not isinstance(entry.get("shape"), list)
        or not isinstance(entry.get("data_offsets"), list)
    ):
        raise ValueError("SafeTensors marker entry is invalid")
    shape = entry["shape"]
    offsets = entry["data_offsets"]
    if (
        len(shape) != 1
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in (*shape, *offsets))
    ):
        raise ValueError("SafeTensors marker geometry is invalid")
    start, end = offsets
    if (
        start < 0
        or end != start + shape[0]
        or 8 + len(raw_header) + end > size_bytes
        or shape[0] > 1024
    ):
        raise ValueError("SafeTensors marker payload is out of bounds")
    with path.open("rb") as stream:
        stream.seek(8 + len(raw_header) + start)
        payload = stream.read(shape[0])
    if len(payload) != shape[0]:
        raise ValueError("SafeTensors marker payload is truncated")
    return payload


def _fp8_sidecar_geometry(header: Mapping[str, Any], key: str) -> bool:
    scale = header.get(key.removesuffix(".weight") + ".weight_scale")
    marker = header.get(key.removesuffix(".weight") + ".comfy_quant")
    return (
        isinstance(scale, dict)
        and scale.get("dtype") == "F32"
        and scale.get("shape") == []
        and isinstance(marker, dict)
        and marker.get("dtype") == "U8"
        and marker.get("shape") == [27]
    )


def _nvfp4_sidecar_geometry(header: Mapping[str, Any], key: str) -> bool:
    stem = key.removesuffix(".weight")
    weight = header.get(key)
    block_scale = header.get(stem + ".weight_scale")
    tensor_scale = header.get(stem + ".weight_scale_2")
    marker = header.get(stem + ".comfy_quant")
    if not all(isinstance(value, dict) for value in (weight, block_scale, tensor_scale, marker)):
        return False
    shape = weight.get("shape")
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and shape[1] % 8 == 0
        and block_scale.get("dtype") == "F8_E4M3"
        and block_scale.get("shape") == [shape[0], shape[1] // 8]
        and tensor_scale.get("dtype") == "F32"
        and tensor_scale.get("shape") == []
        and marker.get("dtype") == "U8"
        and marker.get("shape") == [19]
    )
