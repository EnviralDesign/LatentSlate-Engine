"""Typed stored-component contract for Engine-native Wan 2.2 TI2V 5B.

Pinned workflow graphs and node implementations define topology and sampling
semantics. Execution remains entirely Engine-owned and uses direct Kitchen
kernels inside a disposable worker.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .artifacts import ArtifactIdentity, revalidate_artifact
from .resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)
from .runtime.kit import stable_fingerprint

Wan5KitchenOperation = Literal["wan5_t2v", "wan5_i2v"]

WAN5_BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN5_FPS = 24
WAN5_STEPS = 30
WAN5_GUIDANCE_SCALE = 5.0
WAN5_FLOW_SHIFT = 8.0
WAN5_MAX_SEQUENCE_LENGTH = 512
WAN5_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
_WORKFLOW_REVISION = "f9431bb000ce792094ff345446e22cac1ea6cef3"
_WORKFLOW_SHA256 = MappingProxyType(
    {
        "wan5_t2v": "6a4d79e1891ae1257654fa78d6716936aff9f8c7e578e4e716eb112f4e5a57c4",
        "wan5_i2v": "2b1784c9d6ecf03462651d6e8ded7b5cc5e18047e9eb5e2a885fb6c89c5ac515",
    }
)
_NODE_SEMANTICS_REVISION = "eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f"
_KITCHEN_REVISION = "78e6dd22fe4ebe7bde5062e050a045dc3a244ee4"
_ROLES = frozenset({"pipeline_support", "transformer", "text_encoder", "vae"})

_RESOURCE_IDS = MappingProxyType(
    {
        "pipeline_support": "model:wan22:ti2v-5b/support",
        "transformer": "model:wan22:ti2v-5b/wan2.2_ti2v_5b_fp16",
        "text_encoder": "model:wan22:ti2v-5b/umt5_xxl_fp8_e4m3fn_scaled",
        "vae": "model:wan22:ti2v-5b/wan2.2_vae",
    }
)
_SOURCE_SHA256 = MappingProxyType(
    {
        "transformer": "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e",
        "text_encoder": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        "vae": "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
    }
)
_SUPPORT_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    {
        "model_index.json": (
            499,
            "6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7",
        ),
        "scheduler/scheduler_config.json": (
            820,
            "571a3eed68bcd943a61b9fea8efa9e141472216ada59f928c8b5128ce24b32e0",
        ),
        "text_encoder/config.json": (
            855,
            "a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07",
        ),
        "tokenizer/special_tokens_map.json": (
            7_079,
            "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270",
        ),
        "tokenizer/spiece.model": (
            4_548_313,
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
        ),
        "tokenizer/tokenizer_config.json": (
            61_758,
            "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd",
        ),
        "tokenizer/tokenizer.json": (
            16_837_459,
            "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef",
        ),
        "transformer/config.json": (
            495,
            "dc00d9866e72cf77db6b531aaa33be4dc7148fef9338442bd0ae9181f7075e9b",
        ),
        "vae/config.json": (
            1_701,
            "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Wan5PipelineSupportPlan:
    root: Path
    files: Mapping[str, tuple[int, str]]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Wan5RecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class Wan5StoredRecipe:
    operation: Wan5KitchenOperation
    base_model: str
    components: Mapping[str, Wan5RecipeComponent]


@dataclass(frozen=True, slots=True)
class Wan5RecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, Wan5RecipeComponent]
    plans: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Wan5KitchenRuntimeRequest:
    schema_version: int
    family: str
    operation: Wan5KitchenOperation
    base_model: str
    components: Mapping[str, Mapping[str, str | int]]
    identities: Mapping[str, ArtifactIdentity] = field(repr=False)
    plans: Mapping[str, Any] = field(repr=False, compare=False)
    component_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        components = MappingProxyType(
            {role: MappingProxyType(dict(value)) for role, value in self.components.items()}
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        object.__setattr__(self, "plans", MappingProxyType(dict(self.plans)))
        base = {
            "schema_version": self.schema_version,
            "family": self.family,
            "base_model": self.base_model,
            "execution_contract": operation_execution_contract(self.operation),
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        object.__setattr__(
            self,
            "component_fingerprint",
            stable_fingerprint("wan5-kitchen-components", base),
        )
        object.__setattr__(
            self,
            "fingerprint",
            stable_fingerprint("wan5-kitchen-request", {**base, "operation": self.operation}),
        )

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {
            role: {key: value for key, value in component.items() if key != "path"}
            for role, component in self.components.items()
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "operation": self.operation,
            "base_model": self.base_model,
            "execution_contract": operation_execution_contract(self.operation),
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {
                role: dict(component) for role, component in sorted(self.components.items())
            },
        }


def operation_execution_contract(operation: Wan5KitchenOperation) -> dict[str, object]:
    if operation not in _WORKFLOW_SHA256:
        raise ValueError(f"unsupported Wan 5B operation {operation!r}")
    return {
        "workflow_revision": _WORKFLOW_REVISION,
        "workflow_sha256": _WORKFLOW_SHA256[operation],
        "node_semantics_revision": _NODE_SEMANTICS_REVISION,
        "kitchen_revision": _KITCHEN_REVISION,
        "steps": WAN5_STEPS,
        "guidance_scale": WAN5_GUIDANCE_SCALE,
        "sampler": "uni_pc/bh1",
        "scheduler": "simple",
        "scheduler_source_steps": WAN5_STEPS + 1,
        "discard_penultimate_sigma": True,
        "sampler_runtime": "diffusers/UniPCMultistepScheduler",
        "flow_shift": WAN5_FLOW_SHIFT,
        "fps": WAN5_FPS,
        "saved_workflow_default_frames": 41,
        "engine_product_default_frames": 121,
        "default_frame_deviation": (
            "publisher workflow notes identify 121 frames as optimal; saved 41-frame value "
            "is retained as source evidence rather than the Engine product default"
        ),
        "negative_prompt_sha256": hashlib.sha256(WAN5_NEGATIVE_PROMPT.encode()).hexdigest(),
    }


def plan_wan5_pipeline_support(path: Path) -> Wan5PipelineSupportPlan:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Wan 5B pipeline support must be a directory")
    present = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and ".cache" not in item.relative_to(root).parts
        and item.name != ".latentslate-model.toml"
    }
    if present != set(_SUPPORT_FILES):
        raise ValueError(
            "Wan 5B pipeline support differs from its exact nine-file closure: "
            f"missing={sorted(set(_SUPPORT_FILES) - present)[:3]}, "
            f"unexpected={sorted(present - set(_SUPPORT_FILES))[:3]}"
        )
    verified: dict[str, tuple[int, str]] = {}
    for relative, expected in _SUPPORT_FILES.items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Wan 5B support file escapes its root: {relative}")
        before = candidate.stat()
        actual = (before.st_size, _sha256_file(candidate))
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Wan 5B support file changed during validation: {relative}")
        if actual != expected:
            raise ValueError(f"Wan 5B support file identity mismatch: {relative}")
        verified[relative] = actual
    fingerprint = stable_fingerprint(
        "wan5-pipeline-support", [(name, *verified[name]) for name in sorted(verified)]
    )
    return Wan5PipelineSupportPlan(root, MappingProxyType(verified), fingerprint)


def validate_wan5_stored_recipe(
    recipe: Wan5StoredRecipe,
    inventory: ResourceInventory,
    *,
    include_plans: bool = True,
) -> Wan5RecipeValidation:
    errors: list[str] = []
    resolved: dict[str, Wan5RecipeComponent] = {}
    plans: dict[str, Any] = {}
    if recipe.operation not in _WORKFLOW_SHA256:
        errors.append("Wan 5B operation is unsupported")
    if recipe.base_model != WAN5_BASE_MODEL:
        errors.append(f"Wan 5B base_model must be {WAN5_BASE_MODEL!r}")
    if set(recipe.components) != _ROLES:
        errors.append("Wan 5B stored recipes require exactly support, transformer, text, and VAE")

    by_id = inventory.by_id()
    for role in sorted(_ROLES):
        requested = recipe.components.get(role)
        if requested is None:
            continue
        actual = by_id.get(requested.resource.id)
        if actual is None:
            errors.append(f"{role} resource is not owned by this inventory")
            continue
        if not actual.available:
            errors.append(
                f"{role} resource failed inventory integrity: "
                f"{actual.unavailable_reason or 'unavailable'}"
            )
            continue
        if not inventory.is_installed(actual.id):
            errors.append(f"{role} resource no longer matches its installed integrity proof")
            continue
        try:
            path = inventory.path_for(actual.id).resolve(strict=True)
        except (KeyError, OSError) as exc:
            errors.append(f"{role} inventory path is unavailable: {exc}")
            continue
        if actual != requested.resource or path != requested.path.resolve(strict=False):
            errors.append(f"{role} descriptor/path differs from inventory ownership")
            continue
        resolved[role] = Wan5RecipeComponent(actual, path)
        role_errors = _descriptor_errors(actual, role)
        errors.extend(role_errors)
        if role_errors or not include_plans:
            continue
        try:
            if role == "pipeline_support":
                plans[role] = plan_wan5_pipeline_support(path)
            elif role == "transformer":
                plans[role] = _plan_stored_artifact(path, "transformer")
                plans[role].require_available()
            elif role == "vae":
                plans[role] = _plan_stored_artifact(path, "vae")
                plans[role].require_available()
            else:
                plan = _plan_text_encoder(path)
                plan.require_available()
                plans[role] = plan
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{role} contract failed: {exc}")

    if set(resolved) != _ROLES:
        errors.append("Wan 5B recipe did not resolve every exact component")
    ids = [component.resource.id for component in resolved.values()]
    paths = [component.path.resolve(strict=False) for component in resolved.values()]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        errors.append("Wan 5B recipe roles must resolve to distinct resources and paths")
    return Wan5RecipeValidation(
        not errors,
        tuple(errors),
        MappingProxyType(resolved),
        MappingProxyType(plans),
    )


def build_wan5_kitchen_runtime_request(
    recipe: Wan5StoredRecipe, inventory: ResourceInventory
) -> Wan5KitchenRuntimeRequest:
    validation = validate_wan5_stored_recipe(recipe, inventory, include_plans=True)
    if not validation.available:
        raise ValueError("Wan 5B stored recipe is unavailable: " + "; ".join(validation.errors))
    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role, component in validation.resolved.items():
        plan = validation.plans[role]
        if role == "pipeline_support":
            components[role] = {
                "resource_id": component.resource.id,
                "path": str(component.path),
                "component": role,
                "support_fingerprint": plan.fingerprint,
                "file_count": len(plan.files),
            }
            continue
        identity = plan.identity
        identities[role] = identity
        components[role] = {
            "resource_id": component.resource.id,
            "path": str(component.path),
            "component": role,
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
            "source_sha256": _SOURCE_SHA256[role],
            "plan_fingerprint": _plan_fingerprint(role, plan),
        }
    return Wan5KitchenRuntimeRequest(
        1,
        "wan22",
        recipe.operation,
        recipe.base_model,
        components,
        identities,
        validation.plans,
    )


def revalidate_wan5_kitchen_runtime_request(request: Wan5KitchenRuntimeRequest) -> bool:
    if (
        request.schema_version != 1
        or request.family != "wan22"
        or request.operation not in _WORKFLOW_SHA256
        or request.base_model != WAN5_BASE_MODEL
        or set(request.components) != _ROLES
        or set(request.identities) != _ROLES - {"pipeline_support"}
        or set(request.plans) != _ROLES
    ):
        return False
    support = request.plans.get("pipeline_support")
    try:
        if not isinstance(support, Wan5PipelineSupportPlan):
            return False
        refreshed_support = plan_wan5_pipeline_support(support.root)
        if refreshed_support != support:
            return False
        for role in _ROLES - {"pipeline_support"}:
            identity = request.identities[role]
            plan = request.plans[role]
            component = request.components[role]
            if (
                getattr(plan, "identity", None) != identity
                or not getattr(plan, "available", False)
                or component.get("path") != str(identity.path)
                or component.get("size_bytes") != identity.size_bytes
                or component.get("mtime_ns") != identity.mtime_ns
                or component.get("header_sha256") != identity.header_sha256
                or component.get("source_sha256") != _SOURCE_SHA256[role]
                or component.get("plan_fingerprint") != _plan_fingerprint(role, plan)
                or not revalidate_artifact(identity)
                or not _refresh_plan_matches(role, plan)
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    canonical = Wan5KitchenRuntimeRequest(
        request.schema_version,
        request.family,
        request.operation,
        request.base_model,
        request.components,
        request.identities,
        request.plans,
    )
    return (
        request.component_fingerprint == canonical.component_fingerprint
        and request.fingerprint == canonical.fingerprint
        and request.to_json_dict() == canonical.to_json_dict()
    )


def rehydrate_wan5_kitchen_runtime_request(
    payload: Mapping[str, object],
) -> Wan5KitchenRuntimeRequest:
    """Rebuild and strictly compare one worker-safe request manifest."""

    expected_keys = {
        "schema_version",
        "family",
        "operation",
        "base_model",
        "execution_contract",
        "component_fingerprint",
        "fingerprint",
        "components",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("Wan 5B worker request has invalid top-level keys")
    schema_version = payload.get("schema_version")
    family = payload.get("family")
    base_model = payload.get("base_model")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("Wan 5B worker request schema_version must be an integer")
    if schema_version != 1:
        raise ValueError("Wan 5B worker request schema_version is unsupported")
    if not isinstance(family, str) or family != "wan22":
        raise ValueError("Wan 5B worker request family is invalid")
    if not isinstance(base_model, str) or base_model != WAN5_BASE_MODEL:
        raise ValueError("Wan 5B worker request base_model is invalid")
    operation = payload.get("operation")
    if operation not in _WORKFLOW_SHA256:
        raise ValueError("Wan 5B worker request has an unsupported operation")
    raw_components = payload.get("components")
    if not isinstance(raw_components, Mapping) or set(raw_components) != _ROLES:
        raise ValueError("Wan 5B worker request has an invalid component closure")
    components: dict[str, dict[str, str | int]] = {}
    plans: dict[str, Any] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role in sorted(_ROLES):
        raw = raw_components[role]
        if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
            raise TypeError(f"Wan 5B worker {role} component is invalid")
        component = dict(raw)
        path_value = component.get("path")
        if not isinstance(path_value, str):
            raise TypeError(f"Wan 5B worker {role} path is invalid")
        path = Path(path_value).resolve(strict=True)
        if role == "pipeline_support":
            plan = plan_wan5_pipeline_support(path)
        elif role == "transformer":
            plan = _plan_stored_artifact(path, "transformer")
            plan.require_available()
            identities[role] = plan.identity
        elif role == "vae":
            plan = _plan_stored_artifact(path, "vae")
            plan.require_available()
            identities[role] = plan.identity
        else:
            plan = _plan_text_encoder(path)
            plan.require_available()
            identities[role] = plan.identity
        plans[role] = plan
        components[role] = component
    request = Wan5KitchenRuntimeRequest(
        schema_version,
        family,
        operation,
        base_model,
        components,
        identities,
        plans,
    )
    if request.to_json_dict() != dict(payload) or not revalidate_wan5_kitchen_runtime_request(
        request
    ):
        raise ValueError("Wan 5B worker request does not match its canonical contract")
    return request


def _descriptor_errors(resource: ResourceDescriptor, role: str) -> list[str]:
    errors: list[str] = []
    expected_id = _RESOURCE_IDS[role]
    expected = {
        "pipeline_support": (ResourceFormat.DIRECTORY, ArtifactPrecision.UNKNOWN),
        "transformer": (ResourceFormat.SAFETENSORS, ArtifactPrecision.FP16),
        "text_encoder": (ResourceFormat.SAFETENSORS, ArtifactPrecision.FP8),
        "vae": (ResourceFormat.SAFETENSORS, ArtifactPrecision.FP16),
    }[role]
    if resource.id != expected_id:
        errors.append(f"{role} resource id must be {expected_id!r}")
    if resource.kind != ResourceKind.MODEL or resource.family != "wan22":
        errors.append(f"{role} must be a Wan model resource")
    if resource.component != role or resource.format != expected[0]:
        errors.append(f"{role} format/component contract differs")
    if resource.precision != expected[1]:
        errors.append(f"{role} precision differs")
    if role != "pipeline_support" and resource.quantization != ArtifactQuantization.NATIVE:
        errors.append(f"{role} must retain its declared stored representation")
    return errors


def _plan_fingerprint(role: str, plan: Any) -> str:
    from .runtime.umt5_stored_adapter import UMT5StoredAdapterPlan
    from .runtime.wan5_kitchen_contracts import Wan5StoredPlan

    if isinstance(plan, Wan5StoredPlan):
        return plan.fingerprint
    if isinstance(plan, UMT5StoredAdapterPlan):
        return stable_fingerprint(
            "wan5-umt5-plan",
            {
                "identity": plan.identity,
                "artifact_contract": plan.artifact_contract,
                "config": plan.config_fingerprint,
                "mapping": plan.mapping_fingerprint,
                "quant_sources": plan.quant_sources,
                "dense_sources": dict(plan.dense_source_dtypes),
                "auxiliary": plan.quant_auxiliary,
                "support": plan.support_auxiliary,
            },
        )
    raise TypeError(f"Wan 5B {role} plan has an unsupported type")


def _refresh_plan_matches(role: str, plan: Any) -> bool:
    if role == "transformer":
        return _plan_stored_artifact(plan.identity.path, "transformer") == plan
    if role == "vae":
        return _plan_stored_artifact(plan.identity.path, "vae") == plan
    if role == "text_encoder":
        return _plan_text_encoder(plan.identity.path) == plan
    return False


def _plan_stored_artifact(path: Path, role: Literal["transformer", "vae"]):
    from .runtime.wan5_kitchen_contracts import (
        WAN5_TRANSFORMER,
        WAN5_VAE,
        plan_wan5_stored_artifact,
    )

    contract = WAN5_TRANSFORMER if role == "transformer" else WAN5_VAE
    return plan_wan5_stored_artifact(path, contract)


def _plan_text_encoder(path: Path):
    from .runtime.umt5_stored_adapter import plan_stored_umt5_encoder

    return plan_stored_umt5_encoder(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
