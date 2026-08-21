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
from collections import Counter
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
from .stored_quant import read_safetensors_header_bytes as _read_z_safetensors_header
from .stored_quant import read_safetensors_u8_payload as _read_u8_payload

ZImageOperation = Literal["zimage_turbo_t2i_int8_convrot"]
Z_IMAGE_OPERATION: ZImageOperation = "zimage_turbo_t2i_int8_convrot"
Z_IMAGE_TRANSFORMER_CONTRACT = "comfy_quant/int8_tensorwise_convrot"
Z_IMAGE_MIXED_QWEN_CONTRACT = "comfy_quant/qwen3_4b_fp8_mixed"
Z_IMAGE_VAE_CONTRACT = "native/fp32"
Z_IMAGE_PIPELINE_SUPPORT_CONTRACT = "tongyi-mai/z-image-turbo-prompt-support-v1"
Z_IMAGE_FIXED_LORA_CONTRACT = "engine-native/zimage-70s-horror-bf16-additive-v1"
_BASE_ROLES = frozenset({"pipeline_support", "transformer", "text_encoder", "vae"})
_STYLE_LORA_ROLE = "style_lora"
_SCHEDULE = {
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "guider": "basic",
    "sampling": "auraflow_shift_3",
    "sampler": "res_multistep",
    "scheduler": "simple",
}
_EXECUTION_CONTRACT = {
    "guider": "basic_positive_only",
    "noise": "cpu_fp32_manual_seed_then_transfer",
    "sampler": "res_multistep",
    "scheduler": "simple",
    "shift": 3,
    "output": "png_rgb_1024",
}
_IMMUTABLE_COMPONENTS = {
    "pipeline_support": (4459144, "f332072aa78be7aecdf3ee76d5c247082da564a6", "support-only", None),
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
    "style_lora": (
        85094800,
        "203460b92b193b3a112010ea1c22d1bfcec6dd6d",
        "70s-Horror-Movie-b.safetensors",
        "c50285bd237c3b6f022aafd1b47ebed75a7137466c228ff516b061bede3c5236",
    ),
}
_Z_TRANSFORMER_HEADER_SHA256 = "01e93cae3aa75eb2106025889f1a78df19628a95c433b45d9447562b04907814"
_Z_TRANSFORMER_INT8_LAYERS = 202
_Z_TRANSFORMER_STORED_CATEGORY_COUNTS = MappingProxyType(
    {
        "attention.qkv": 34,
        "attention.out": 34,
        "feed_forward.w1": 34,
        "feed_forward.w2": 34,
        "feed_forward.w3": 34,
        "adaLN_modulation": 32,
    }
)
_Z_PIPELINE_SUPPORT_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    {
        "text_encoder/config.json": (
            726,
            "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
        ),
        "tokenizer/merges.txt": (
            1_671_853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        "tokenizer/vocab.json": (
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
        "tokenizer/tokenizer_config.json": (
            9_732,
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ZImageTurboRecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class ZImageTurboRecipe:
    base_model: str
    pipeline_support: ZImageTurboRecipeComponent
    transformer: ZImageTurboRecipeComponent
    text_encoder: ZImageTurboRecipeComponent
    vae: ZImageTurboRecipeComponent
    style_lora: ZImageTurboRecipeComponent | None = None
    operation: ZImageOperation = Z_IMAGE_OPERATION
    width: int = 1024
    height: int = 1024
    steps: int = 8
    guider: str = "basic"
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
        categories = Counter(
            _z_image_transformer_stored_category(key) for key in self.stored_layers
        )
        if dict(categories) != dict(_Z_TRANSFORMER_STORED_CATEGORY_COUNTS):
            raise ValueError("Z-Image transformer stored ConvRot module categories differ from pin")


@dataclass(frozen=True, slots=True)
class ZImageDensePlan:
    identity: ArtifactIdentity
    schema_sha256: str
    role: Literal["text_encoder", "vae"]
    tensor_count: int


@dataclass(frozen=True, slots=True)
class ZImagePipelineSupportPlan:
    root: Path
    files: Mapping[str, tuple[int, str]]
    fingerprint: str


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

    def to_json_dict(self) -> dict[str, object]:
        """Return the canonical, plan-free child-worker capability payload.

        Paths and immutable header identities are included only for the private
        authenticated worker IPC.  Plans are deliberately rebuilt by the child;
        no parent-side Python object is trusted across the process boundary.
        """

        return {
            "schema_version": self.schema_version,
            "family": "zimage",
            "operation": self.operation,
            "base_model": self.base_model,
            "execution_contract": dict(_EXECUTION_CONTRACT),
            "schedule": dict(sorted(self.schedule.items())),
            "components": {role: dict(value) for role, value in sorted(self.components.items())},
            "fingerprint": self.fingerprint,
        }


def z_image_turbo_schedule(recipe: ZImageTurboRecipe) -> dict[str, str | int | float]:
    actual = {
        "width": recipe.width,
        "height": recipe.height,
        "steps": recipe.steps,
        "guider": recipe.guider,
        "sampling": recipe.sampling,
        "sampler": recipe.sampler,
        "scheduler": recipe.scheduler,
    }
    if actual != _SCHEDULE:
        raise ValueError(
            "Z-Image Turbo requires the exact pinned schedule: 1024x1024, 8 steps, BasicGuider, AuraFlow shift 3, res_multistep/simple"
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
    roles = _BASE_ROLES | ({_STYLE_LORA_ROLE} if recipe.style_lora is not None else set())
    for role in sorted(roles):
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
    if set(resolved) != roles:
        errors.append("Z-Image recipe did not resolve its exact component role closure")
    return ZImageTurboRecipeValidation(
        not errors, tuple(errors), MappingProxyType(resolved), MappingProxyType(plans)
    )


def build_z_image_turbo_runtime_request(
    recipe: ZImageTurboRecipe, inventory: ResourceInventory
) -> ZImageTurboRuntimeRequest:
    validation = validate_z_image_turbo_recipe(recipe, inventory)
    roles = _BASE_ROLES | ({_STYLE_LORA_ROLE} if recipe.style_lora is not None else set())
    if not validation.available or set(validation.plans) != roles:
        raise ValueError("Z-Image Turbo recipe is unavailable: " + "; ".join(validation.errors))
    identities: dict[str, ArtifactIdentity] = {}
    components: dict[str, dict[str, str | int]] = {}
    for role in sorted(roles):
        component = validation.resolved[role]
        plan = validation.plans[role]
        source = component.resource.sources[0]
        if role == "pipeline_support":
            components[role] = {
                "resource_id": component.resource.id,
                "path": str(component.path.resolve()),
                "format": component.resource.format.value,
                "component": role,
                "file_count": len(plan.files),
                "support_fingerprint": plan.fingerprint,
                "source_revision": str(source.revision),
            }
            continue
        identities[role] = plan.identity
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
    roles = frozenset(request.components)
    if (
        request.operation != Z_IMAGE_OPERATION
        or roles not in {_BASE_ROLES, _BASE_ROLES | {_STYLE_LORA_ROLE}}
        or set(request.identities) != roles - {"pipeline_support"}
        or set(request.plans) != roles
    ):
        return False
    try:
        if dict(request.schedule) != _SCHEDULE:
            return False
    except (TypeError, ValueError):
        return False
    support = request.plans.get("pipeline_support")
    support_component = request.components.get("pipeline_support", {})
    if (
        not isinstance(support, ZImagePipelineSupportPlan)
        or not revalidate_z_image_pipeline_support(support)
        or support_component.get("support_fingerprint") != support.fingerprint
        or support_component.get("file_count") != len(support.files)
        or support_component.get("path") != str(support.root)
        or support_component.get("source_revision") != _IMMUTABLE_COMPONENTS["pipeline_support"][1]
    ):
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
            from .runtime.z_image_qwen_checkpoint import revalidate_z_image_mixed_qwen

            if not revalidate_z_image_mixed_qwen(plan):
                return False
        elif role == "vae":
            from .runtime.z_image_vae import revalidate_z_image_flux_ae

            if not revalidate_z_image_flux_ae(plan):
                return False
        elif role == _STYLE_LORA_ROLE:
            from .runtime.z_image_stored_lora import (
                ZImageFixedLoraPlan,
                revalidate_z_image_70s_horror_lora,
            )

            if not isinstance(plan, ZImageFixedLoraPlan) or not revalidate_z_image_70s_horror_lora(
                plan
            ):
                return False
    return True


def rehydrate_z_image_turbo_runtime_request(
    value: Mapping[str, object],
) -> ZImageTurboRuntimeRequest:
    """Rebuild and prove an exact Z request from authenticated canonical JSON.

    This is intentionally independent of the catalog: the private worker proves
    current on-disk support, header, quantization and VAE plans before importing
    any heavyweight loader.
    """

    expected = {
        "schema_version",
        "family",
        "operation",
        "base_model",
        "execution_contract",
        "schedule",
        "components",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Z-Image worker recipe is not canonical")
    if (
        value["schema_version"] != 1
        or value["family"] != "zimage"
        or value["operation"] != Z_IMAGE_OPERATION
        or not isinstance(value["base_model"], str)
        or not isinstance(value["schedule"], Mapping)
        or dict(value["schedule"]) != _SCHEDULE
        or not isinstance(value["execution_contract"], Mapping)
        or dict(value["execution_contract"]) != _EXECUTION_CONTRACT
        or not isinstance(value["components"], Mapping)
        or not isinstance(value["fingerprint"], str)
    ):
        raise ValueError("Z-Image worker recipe contract is invalid")
    raw_components = value["components"]
    roles = frozenset(raw_components)
    if roles not in {_BASE_ROLES, _BASE_ROLES | {_STYLE_LORA_ROLE}} or not all(
        isinstance(component, Mapping) for component in raw_components.values()
    ):
        raise ValueError("Z-Image worker components are invalid")
    components = {role: dict(component) for role, component in raw_components.items()}
    plans: dict[str, Any] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in sorted(roles):
        component = components[role]
        path = component.get("path")
        if not isinstance(path, str):
            raise TypeError(f"Z-Image worker {role} path is invalid")
        plan = _plan_component(role, Path(path).resolve(strict=True))
        plans[role] = plan
        if role != "pipeline_support":
            identities[role] = plan.identity
    request = ZImageTurboRuntimeRequest(
        1,
        value["base_model"],
        Z_IMAGE_OPERATION,
        dict(value["schedule"]),
        components,
        identities,
        plans,
    )
    if request.fingerprint != value["fingerprint"] or not revalidate_z_image_turbo_runtime_request(
        request
    ):
        raise ValueError("Z-Image worker recipe identity differs from its authenticated request")
    return request


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
        None
        if role == "pipeline_support"
        else ArtifactPrecision.BF16
        if role == _STYLE_LORA_ROLE
        else ArtifactPrecision.FP8
        if role == "text_encoder"
        else ArtifactPrecision.FP32
        if role == "vae"
        else ArtifactPrecision.UNKNOWN
    )
    quantization = (
        None
        if role == "pipeline_support"
        else ArtifactQuantization.NATIVE
        if role == _STYLE_LORA_ROLE
        else ArtifactQuantization.INT8
        if role == "transformer"
        else ArtifactQuantization.NATIVE
    )
    contract = (
        Z_IMAGE_PIPELINE_SUPPORT_CONTRACT
        if role == "pipeline_support"
        else Z_IMAGE_FIXED_LORA_CONTRACT
        if role == _STYLE_LORA_ROLE
        else Z_IMAGE_TRANSFORMER_CONTRACT
        if role == "transformer"
        else Z_IMAGE_MIXED_QWEN_CONTRACT
        if role == "text_encoder"
        else Z_IMAGE_VAE_CONTRACT
    )
    return {
        "precision": precision,
        "quantization": quantization,
        "contract": contract,
        "architecture": (
            "z_image_turbo_70s_horror_fixed_lora"
            if role == _STYLE_LORA_ROLE
            else f"z_image_turbo_{role}"
        ),
        "base_model": base_model if role in {"transformer", _STYLE_LORA_ROLE} else None,
    }


def _validate_descriptor(
    resource: ResourceDescriptor, role: str, expected: Mapping[str, Any], errors: list[str]
) -> None:
    expected_kind = ResourceKind.LORA if role == _STYLE_LORA_ROLE else ResourceKind.MODEL
    if resource.kind != expected_kind or not resource.available:
        errors.append(f"{role} must be an available {expected_kind.value} resource")
    if resource.family != "zimage" or resource.component != role:
        errors.append(f"{role} must declare family='zimage' and component={role!r}")
    if (
        resource.format
        != (ResourceFormat.DIRECTORY if role == "pipeline_support" else ResourceFormat.SAFETENSORS)
        or (expected["precision"] is not None and resource.precision != expected["precision"])
        or (
            expected["quantization"] is not None
            and resource.quantization != expected["quantization"]
        )
    ):
        errors.append(f"{role} stored format/precision/quantization metadata is incorrect")
    if (
        resource.metadata.get("quantization_contract") != expected["contract"]
        or resource.metadata.get("architecture") != expected["architecture"]
    ):
        errors.append(f"{role} immutable metadata contract is incorrect")
    if expected["base_model"] is not None and resource.base_model != expected["base_model"]:
        errors.append(f"{role} base_model does not match recipe base_model")
    if role == _STYLE_LORA_ROLE and (
        resource.default_strength != 1.0
        or resource.metadata.get("header_sha256")
        != "0f28d13bb8128539a02eebe1065757232969c3bf8bf09e66d510487198885778"
        or resource.metadata.get("schema_sha256")
        != "8b6aa274d5530c5b9d906c0855445f28d209b0eb7af9a31b198f0f4edf3c2088"
        or resource.metadata.get("target_count") != 240
        or resource.metadata.get("tensor_count") != 480
        or resource.metadata.get("qkv_row_slice_targets") != 90
        or resource.metadata.get("direct_targets") != 150
        or resource.metadata.get("rank") != 16
        or resource.metadata.get("license") != "not-declared-upstream"
    ):
        errors.append("style_lora immutable metadata contract is incorrect")
    size, revision, filename, sha256 = _IMMUTABLE_COMPONENTS[role]
    if resource.size_bytes != size:
        errors.append(f"{role} size does not match the pinned immutable artifact")
    if role == "pipeline_support":
        if len(resource.sources) != 1:
            errors.append("pipeline_support must have one pinned first-party source")
        elif (resource.sources[0].repo_id, resource.sources[0].revision) != (
            "Tongyi-MAI/Z-Image-Turbo",
            revision,
        ):
            errors.append("pipeline_support source does not match the pinned first-party support")
        return
    if len(resource.sources) != 1 or not resource.sources[0].is_exact():
        errors.append(f"{role} must have one exact immutable source")
    else:
        source = resource.sources[0]
        if (source.repo_id, source.revision, source.filename, source.sha256) != (
            "Kutches/ImageZV2" if role == _STYLE_LORA_ROLE else "Comfy-Org/z_image_turbo",
            revision,
            filename,
            sha256,
        ):
            errors.append(f"{role} source does not match the pinned immutable artifact")


def _plan_component(role: str, path: Path) -> Any:
    if role == "pipeline_support":
        return plan_z_image_pipeline_support(path)
    if role == "transformer":
        return _plan_transformer(path)
    if role == "text_encoder":
        from .runtime.z_image_qwen_checkpoint import plan_z_image_mixed_qwen

        return plan_z_image_mixed_qwen(path)
    if role == "vae":
        from .runtime.z_image_vae import plan_z_image_flux_ae

        return plan_z_image_flux_ae(path)
    if role == _STYLE_LORA_ROLE:
        from .runtime.z_image_stored_lora import plan_z_image_70s_horror_lora

        return plan_z_image_70s_horror_lora(path)
    raise ValueError(f"unsupported Z-Image component role: {role}")


def plan_z_image_pipeline_support(path: Path) -> ZImagePipelineSupportPlan:
    """Validate the tiny first-party support closure without accepting weights."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Z-Image pipeline support must be a directory")
    present = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != ".latentslate-model.toml" and ".cache" not in item.parts
    }
    expected = set(_Z_PIPELINE_SUPPORT_FILES)
    if present != expected:
        raise ValueError(
            "Z-Image pipeline support closure differs: "
            f"missing={sorted(expected - present)[:2]}, unexpected={sorted(present - expected)[:2]}"
        )
    files: dict[str, tuple[int, str]] = {}
    for relative, expected_identity in _Z_PIPELINE_SUPPORT_FILES.items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Z-Image support file escapes its root: {relative}")
        before = candidate.stat()
        digest = _sha256_file(candidate)
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Z-Image support file changed during validation: {relative}")
        if (after.st_size, digest) != expected_identity:
            raise ValueError(f"Z-Image support file identity mismatch: {relative}")
        files[relative] = expected_identity
    _validate_z_image_support_semantics(root)
    fingerprint = hashlib.sha256(
        json.dumps(
            sorted((name, *value) for name, value in files.items()), separators=(",", ":")
        ).encode()
    ).hexdigest()
    return ZImagePipelineSupportPlan(root, MappingProxyType(files), fingerprint)


def revalidate_z_image_pipeline_support(plan: ZImagePipelineSupportPlan) -> bool:
    try:
        return plan_z_image_pipeline_support(plan.root) == plan
    except (OSError, TypeError, ValueError):
        return False


def _validate_z_image_support_semantics(root: Path) -> None:
    config = json.loads((root / "text_encoder/config.json").read_text(encoding="utf-8"))
    expected = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 151936,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Z-Image first-party Qwen support config facts differ")
    if config.get("sliding_window") is not None or config.get("use_sliding_window") is not False:
        raise ValueError("Z-Image first-party Qwen support must disable sliding window")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    categories = Counter(_z_image_transformer_stored_category(key) for key in layers)
    if dict(categories) != dict(_Z_TRANSFORMER_STORED_CATEGORY_COUNTS):
        raise ValueError(
            "transformer ConvRot module category closure differs from the official mapping"
        )
    mapping = MappingProxyType({key: key for key in sorted(layers)})
    return ZImageTransformerPlan(
        probe.identity,
        probe.schema_sha256,
        mapping,
        MappingProxyType(dict(layers)),
    )


def _z_image_transformer_stored_category(source: str) -> str:
    """Classify the only six official stored NextDiT module kinds."""

    suffixes = {
        ".attention.qkv.weight": "attention.qkv",
        ".attention.out.weight": "attention.out",
        ".feed_forward.w1.weight": "feed_forward.w1",
        ".feed_forward.w2.weight": "feed_forward.w2",
        ".feed_forward.w3.weight": "feed_forward.w3",
        ".adaLN_modulation.0.weight": "adaLN_modulation",
    }
    for suffix, category in suffixes.items():
        if source.endswith(suffix):
            return category
    return "unknown"


def _plan_dense(role: str, path: Path) -> ZImageDensePlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors" or probe.tensor_count <= 0:
        raise ValueError(f"{role} is not a readable non-empty SafeTensors artifact")
    raw_header, header = _read_z_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    dtypes = {value.get("dtype") for value in header.values() if isinstance(value, dict)}
    if role == "text_encoder":
        from .runtime.z_image_qwen_checkpoint import QWEN_HEADER_SHA256

        # Comfy calls this file mixed FP8.  It must be structurally explicit, not
        # guessed from its filename: at least one F8 weight plus its sidecars.
        if hashlib.sha256(raw_header).hexdigest() != QWEN_HEADER_SHA256:
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
