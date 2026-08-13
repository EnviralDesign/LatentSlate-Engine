"""Typed Engine-native LTX 2.3 stored-component recipes.

Pinned workflow graphs define the operation topology, but execution is owned by
LatentSlate Engine.  These contracts bind exact SafeTensors artifacts and a
small Diffusers support shell to direct Comfy Kitchen materializers; they never
identify or invoke a workflow runtime.
"""

from __future__ import annotations

import hashlib
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
from .runtime.kit import stable_fingerprint
from .runtime.ltx23_kitchen_contracts import (
    LTX23_DEV_FP8,
    LTX23_DISTILLED_FP8,
    LTX23_GEMMA_MIXED,
    LTX23_MODEL_LORA,
    LTX23_SPATIAL_UPSCALER,
    LTX23_TEXT_LORA,
    LTX23ArtifactContract,
    plan_ltx23_stored_artifact,
)

LTX23KitchenOperation = Literal[
    "ltx23_dev_t2v",
    "ltx23_dev_i2v",
    "ltx23_distilled_flf",
]

# The Dev topology doubles a latent before its second denoise pass, so both
# public dimensions must be /64. The single-stage Distilled FLF graph only
# requires the native /32 spatial grid. Keeping this contract in the recipe
# layer lets authoring reject invalid geometry before a job reaches the worker.
_LTX23_KITCHEN_DIMENSION_ALIGNMENT: Mapping[str, int] = MappingProxyType(
    {
        "ltx23_dev_t2v": 64,
        "ltx23_dev_i2v": 64,
        "ltx23_distilled_flf": 32,
    }
)


def ltx23_kitchen_dimension_alignment(operation: LTX23KitchenOperation) -> int:
    """Return the exact public pixel-grid requirement for one Kitchen operation."""

    return _LTX23_KITCHEN_DIMENSION_ALIGNMENT[operation]


def validate_ltx23_kitchen_dimensions(
    operation: LTX23KitchenOperation,
    *,
    width: int,
    height: int,
) -> None:
    """Reject geometry that the pinned Engine-native topology cannot execute."""

    alignment = ltx23_kitchen_dimension_alignment(operation)
    if width <= 0 or height <= 0 or width % alignment or height % alignment:
        raise ValueError(
            f"LTX 2.3 Kitchen {operation} requires width and height divisible by "
            f"{alignment} pixels (received {width}x{height})"
        )

LTX23_BASE_MODEL = "Lightricks/LTX-2.3"
LTX23_FPS = 24
LTX23_GUIDANCE_SCALE = 1.0
LTX23_MODEL_LORA_STRENGTH = 0.5
LTX23_TEXT_LORA_STRENGTH = 1.0
LTX23_GUIDE_STRENGTH = 0.7
LTX23_MAIN_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
LTX23_REFINE_SIGMAS = (0.85, 0.725, 0.4219, 0.0)
_EXECUTION_CONTRACT = {
    "workflow_revision": "2b7f823136606344f0bccce249898d771b809aa1",
    "workflow_sha256": {
        "ltx23_dev_t2v": "75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f",
        "ltx23_dev_i2v": "91dd8e44926fd37f6d9307789484370fa333582b14e53ed771d63ed805379ee4",
        "ltx23_distilled_flf": "168bc2584ef117133e76341f04e001aab2641b72b75d81b66b5c0b66e56c24a5",
    },
    "node_semantics_revision": "725e6ec60621c6f001af04769173e7dbb3c53541",
    "kitchen_revision": "78e6dd22fe4ebe7bde5062e050a045dc3a244ee4",
    "pinned_workflow_default_width": 1280,
    "pinned_workflow_default_height": 720,
    "engine_acceptance_default_width": 768,
    "engine_acceptance_default_height": 512,
    "dimension_alignment": "dev=/64;distilled_flf=/32",
}

_DEV_ROLES = frozenset(
    {
        "pipeline_support",
        "checkpoint",
        "model_lora",
        "text_encoder",
        "text_lora",
        "latent_upscaler",
    }
)
_FLF_ROLES = frozenset({"pipeline_support", "checkpoint", "text_encoder"})

_SUPPORT_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    {
        "processor/added_tokens.json": (
            35,
            "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
        ),
        "processor/chat_template.jinja": (
            1_532,
            "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
        ),
        "processor/preprocessor_config.json": (
            570,
            "f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d",
        ),
        "processor/processor_config.json": (
            70,
            "3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53",
        ),
        "processor/special_tokens_map.json": (
            662,
            "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
        ),
        "processor/tokenizer.json": (
            33_384_568,
            "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        ),
        "processor/tokenizer.model": (
            4_689_074,
            "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
        "processor/tokenizer_config.json": (
            1_155_387,
            "983c80895e7b188911f5ccfbb4f780a4e73fb3131ed2080a3e847ef8623cfdca",
        ),
        "scheduler/scheduler_config.json": (
            489,
            "ad5ea953b2eceee813d67bd44f5e348a877f04810b9f3298acb869c5dac74771",
        ),
        "text_encoder/config.json": (
            2_960,
            "fb4508a46270e0af2778cd72c191f290984e6a5fa8f155895544507d3803dbe9",
        ),
        "text_encoder/generation_config.json": (
            168,
            "1b77c4610a27b5aa38b72be8a757da4e14849bc4e3dc641ccbe923a0e226ed9a",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class LTX23PipelineSupportPlan:
    root: Path
    files: Mapping[str, tuple[int, str]]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class LTX23StoredRecipeComponent:
    resource: ResourceDescriptor
    path: Path


@dataclass(frozen=True, slots=True)
class LTX23StoredRecipe:
    operation: LTX23KitchenOperation
    base_model: str
    components: Mapping[str, LTX23StoredRecipeComponent]


@dataclass(frozen=True, slots=True)
class LTX23StoredRecipeValidation:
    available: bool
    errors: tuple[str, ...]
    resolved: Mapping[str, LTX23StoredRecipeComponent]
    plans: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LTX23KitchenRuntimeRequest:
    schema_version: int
    family: str
    operation: LTX23KitchenOperation
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
            "execution_contract": _operation_execution_contract(self.operation),
            "components": {role: dict(value) for role, value in sorted(components.items())},
        }
        object.__setattr__(
            self,
            "component_fingerprint",
            stable_fingerprint("ltx23-kitchen-components", base),
        )
        object.__setattr__(
            self,
            "fingerprint",
            stable_fingerprint("ltx23-kitchen-request", {**base, "operation": self.operation}),
        )

    def public_component_manifest(self) -> dict[str, dict[str, str | int]]:
        return {
            role: {key: value for key, value in value.items() if key != "path"}
            for role, value in self.components.items()
        }

    def to_json_dict(self) -> dict[str, object]:
        """Return the exact JSON-safe manifest accepted by a disposable worker."""

        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "operation": self.operation,
            "base_model": self.base_model,
            "execution_contract": _operation_execution_contract(self.operation),
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {
                role: dict(component) for role, component in sorted(self.components.items())
            },
        }


def required_ltx23_roles(operation: LTX23KitchenOperation) -> frozenset[str]:
    if operation == "ltx23_distilled_flf":
        return _FLF_ROLES
    if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}:
        return _DEV_ROLES
    raise ValueError(f"unsupported LTX 2.3 operation {operation!r}")


def _operation_execution_contract(operation: LTX23KitchenOperation) -> dict[str, object]:
    return {
        key: value
        for key, value in _EXECUTION_CONTRACT.items()
        if key != "workflow_sha256"
    } | {"workflow_sha256": _EXECUTION_CONTRACT["workflow_sha256"][operation]}


def plan_ltx23_pipeline_support(path: Path) -> LTX23PipelineSupportPlan:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("LTX 2.3 pipeline support must be a directory")
    present = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and ".cache" not in item.relative_to(root).parts
        and item.name != ".latentslate-model.toml"
    }
    expected = set(_SUPPORT_FILES)
    if present != expected:
        raise ValueError(
            "LTX 2.3 pipeline support is not the exact bounded shell: "
            f"missing={sorted(expected - present)[:3]}, unexpected={sorted(present - expected)[:3]}"
        )
    verified: dict[str, tuple[int, str]] = {}
    for relative, expected_identity in _SUPPORT_FILES.items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"LTX 2.3 support file escapes its root: {relative}")
        before = candidate.stat()
        digest = _sha256_file(candidate)
        after = candidate.stat()
        actual = (after.st_size, digest)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"LTX 2.3 support file changed during validation: {relative}")
        if actual != expected_identity:
            raise ValueError(f"LTX 2.3 support file identity mismatch: {relative}")
        verified[relative] = actual
    fingerprint = stable_fingerprint(
        "ltx23-kitchen-support",
        [(name, *verified[name]) for name in sorted(verified)],
    )
    return LTX23PipelineSupportPlan(root, MappingProxyType(verified), fingerprint)


def revalidate_ltx23_pipeline_support(plan: LTX23PipelineSupportPlan) -> bool:
    try:
        refreshed = plan_ltx23_pipeline_support(plan.root)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed.files == plan.files and refreshed.fingerprint == plan.fingerprint


def validate_ltx23_stored_recipe(
    recipe: LTX23StoredRecipe,
    inventory: ResourceInventory,
    *,
    include_plans: bool = True,
) -> LTX23StoredRecipeValidation:
    errors: list[str] = []
    resolved: dict[str, LTX23StoredRecipeComponent] = {}
    plans: dict[str, Any] = {}
    roles = required_ltx23_roles(recipe.operation)
    if recipe.base_model != LTX23_BASE_MODEL:
        errors.append(f"LTX 2.3 base_model must be {LTX23_BASE_MODEL!r}")
    if set(recipe.components) != roles:
        errors.append(f"LTX 2.3 {recipe.operation} requires exactly {', '.join(sorted(roles))}")

    contracts = _role_contracts(recipe.operation)
    for role in sorted(roles):
        requested = recipe.components.get(role)
        if requested is None:
            continue
        actual = inventory.by_id().get(requested.resource.id)
        if actual is None:
            errors.append(f"{role} resource is not owned by this inventory")
            continue
        if not actual.available:
            errors.append(
                f"{role} resource failed inventory availability/integrity: "
                f"{actual.unavailable_reason or 'unavailable'}"
            )
            continue
        try:
            path = inventory.path_for(actual.id).resolve(strict=True)
        except (KeyError, OSError) as exc:
            errors.append(f"{role} inventory path is unavailable: {exc}")
            continue
        if actual != requested.resource or path != requested.path.resolve(strict=False):
            errors.append(f"{role} descriptor/path does not match the inventory mapping")
            continue
        resolved[role] = LTX23StoredRecipeComponent(actual, path)
        expected = contracts[role]
        errors.extend(_descriptor_errors(actual, role, expected))
        if errors and any(message.startswith(f"{role} ") for message in errors):
            continue
        try:
            if role == "pipeline_support":
                plans[role] = plan_ltx23_pipeline_support(path) if include_plans else None
            else:
                contract = expected[4]
                probe = probe_artifact(path)
                if probe.schema_sha256 != contract.schema_sha256:
                    raise ValueError("schema differs from the pinned stored artifact")
                if include_plans:
                    plan = plan_ltx23_stored_artifact(path, contract)
                    plan.require_available()
                    plans[role] = plan
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{role} contract failed: {exc}")

    if set(resolved) != roles:
        errors.append("LTX 2.3 recipe did not resolve every exact component")
    ids = [item.resource.id for item in resolved.values()]
    paths = [item.path.resolve(strict=False) for item in resolved.values()]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        errors.append("LTX 2.3 recipe roles must resolve to distinct resources and paths")
    return LTX23StoredRecipeValidation(
        not errors,
        tuple(errors),
        MappingProxyType(resolved),
        MappingProxyType(plans),
    )


def build_ltx23_kitchen_runtime_request(
    recipe: LTX23StoredRecipe,
    inventory: ResourceInventory,
) -> LTX23KitchenRuntimeRequest:
    validation = validate_ltx23_stored_recipe(recipe, inventory, include_plans=True)
    if not validation.available:
        raise ValueError("LTX 2.3 stored recipe is unavailable: " + "; ".join(validation.errors))
    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    for role, component in validation.resolved.items():
        if role == "pipeline_support":
            plan = validation.plans[role]
            components[role] = {
                "resource_id": component.resource.id,
                "path": str(component.path),
                "component": role,
                "support_fingerprint": plan.fingerprint,
                "file_count": len(plan.files),
            }
            continue
        plan = validation.plans[role]
        plan.require_available()
        # ``validate_ltx23_stored_recipe`` created this plan in this same
        # synchronous call.  Do not immediately reread every multi-gigabyte
        # artifact here: the managed runtime revalidates the completed request
        # immediately before process spawn, and the isolated worker validates
        # again at its materialization boundary.
        identities[role] = plan.identity
        components[role] = {
            "resource_id": component.resource.id,
            "path": str(component.path),
            "component": role,
            "size_bytes": plan.identity.size_bytes,
            "mtime_ns": plan.identity.mtime_ns,
            "header_sha256": plan.identity.header_sha256,
            "schema_sha256": plan.contract.schema_sha256,
            "source_sha256": plan.contract.source_sha256,
            "plan_fingerprint": plan.fingerprint,
        }
    return LTX23KitchenRuntimeRequest(
        1,
        "ltx23",
        recipe.operation,
        recipe.base_model,
        components,
        identities,
        validation.plans,
    )


def revalidate_ltx23_kitchen_runtime_request(request: LTX23KitchenRuntimeRequest) -> bool:
    try:
        roles = required_ltx23_roles(request.operation)
    except ValueError:
        return False
    if (
        request.schema_version != 1
        or request.family != "ltx23"
        or request.base_model != LTX23_BASE_MODEL
        or set(request.components) != roles
        or set(request.identities) != roles - {"pipeline_support"}
        or set(request.plans) != roles
    ):
        return False
    support = request.plans.get("pipeline_support")
    if not isinstance(support, LTX23PipelineSupportPlan) or not revalidate_ltx23_pipeline_support(
        support
    ):
        return False
    for role, identity in request.identities.items():
        component = request.components[role]
        plan = request.plans.get(role)
        contract = _role_contracts(request.operation)[role][4]
        if (
            contract is None
            or not hasattr(plan, "identity")
            or plan.identity != identity
            or plan.contract != contract
            or not plan.available
            or component.get("plan_fingerprint") != plan.fingerprint
            or component.get("source_sha256") != contract.source_sha256
            or component.get("path") != str(identity.path)
            or component.get("size_bytes") != identity.size_bytes
            or component.get("mtime_ns") != identity.mtime_ns
            or component.get("header_sha256") != identity.header_sha256
            or component.get("schema_sha256") != contract.schema_sha256
            or not revalidate_artifact(identity)
            or not plan.revalidate()
        ):
            return False
    canonical = LTX23KitchenRuntimeRequest(
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
    )


def rehydrate_ltx23_kitchen_runtime_request(
    payload: Mapping[str, Any],
) -> LTX23KitchenRuntimeRequest:
    """Rebuild and revalidate an exact worker request from canonical JSON."""

    expected_top = {
        "schema_version",
        "family",
        "operation",
        "base_model",
        "execution_contract",
        "component_fingerprint",
        "fingerprint",
        "components",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("LTX 2.3 Kitchen worker request fields are not canonical")
    operation = payload["operation"]
    if not isinstance(operation, str):
        raise TypeError("LTX 2.3 Kitchen worker operation is invalid")
    roles = required_ltx23_roles(operation)
    if (
        payload["schema_version"] != 1
        or payload["family"] != "ltx23"
        or payload["base_model"] != LTX23_BASE_MODEL
        or payload["execution_contract"] != _operation_execution_contract(operation)
        or not isinstance(payload["components"], Mapping)
        or set(payload["components"]) != roles
    ):
        raise ValueError("LTX 2.3 Kitchen worker request identity is invalid")

    components: dict[str, dict[str, str | int]] = {}
    identities: dict[str, ArtifactIdentity] = {}
    plans: dict[str, Any] = {}
    contracts = _role_contracts(operation)
    for role in sorted(roles):
        raw = payload["components"][role]
        if not isinstance(raw, Mapping):
            raise TypeError(f"LTX 2.3 Kitchen component {role!r} is invalid")
        if role == "pipeline_support":
            if set(raw) != {
                "resource_id",
                "path",
                "component",
                "support_fingerprint",
                "file_count",
            }:
                raise ValueError("LTX 2.3 Kitchen support fields are not canonical")
            if raw["component"] != role or not isinstance(raw["resource_id"], str):
                raise ValueError("LTX 2.3 Kitchen support contract is invalid")
            path_value = raw["path"]
            if not isinstance(path_value, str):
                raise TypeError("LTX 2.3 Kitchen support path is invalid")
            plan = plan_ltx23_pipeline_support(Path(path_value))
            if raw["support_fingerprint"] != plan.fingerprint or raw["file_count"] != len(
                plan.files
            ):
                raise ValueError("LTX 2.3 Kitchen support identity changed")
            plans[role] = plan
            components[role] = dict(raw)
            continue

        if set(raw) != {
            "resource_id",
            "path",
            "component",
            "size_bytes",
            "mtime_ns",
            "header_sha256",
            "schema_sha256",
            "source_sha256",
            "plan_fingerprint",
        }:
            raise ValueError(f"LTX 2.3 Kitchen component {role!r} fields are not canonical")
        path_value = raw["path"]
        if (
            not isinstance(path_value, str)
            or raw["component"] != role
            or not isinstance(raw["resource_id"], str)
            or isinstance(raw["size_bytes"], bool)
            or not isinstance(raw["size_bytes"], int)
            or isinstance(raw["mtime_ns"], bool)
            or not isinstance(raw["mtime_ns"], int)
            or not isinstance(raw["header_sha256"], str)
            or not isinstance(raw["schema_sha256"], str)
            or not isinstance(raw["source_sha256"], str)
            or not isinstance(raw["plan_fingerprint"], str)
        ):
            raise ValueError(f"LTX 2.3 Kitchen component {role!r} contract is invalid")
        path = Path(path_value).resolve(strict=True)
        identity = ArtifactIdentity(
            path,
            raw["size_bytes"],
            raw["mtime_ns"],
            raw["header_sha256"],
        )
        probe = probe_artifact(path)
        contract = contracts[role][4]
        if (
            contract is None
            or probe.identity != identity
            or raw["schema_sha256"] != contract.schema_sha256
            or probe.schema_sha256 != contract.schema_sha256
            or raw["source_sha256"] != contract.source_sha256
        ):
            raise ValueError(f"LTX 2.3 Kitchen component {role!r} identity changed")
        plan = plan_ltx23_stored_artifact(path, contract)
        plan.require_available()
        if raw["plan_fingerprint"] != plan.fingerprint:
            raise ValueError(f"LTX 2.3 Kitchen component {role!r} plan changed")
        components[role] = dict(raw)
        identities[role] = identity
        plans[role] = plan

    request = LTX23KitchenRuntimeRequest(
        1,
        "ltx23",
        operation,
        LTX23_BASE_MODEL,
        components,
        identities,
        plans,
    )
    if request.to_json_dict() != dict(payload) or not revalidate_ltx23_kitchen_runtime_request(
        request
    ):
        raise ValueError("LTX 2.3 Kitchen worker request fingerprint is not canonical")
    return request


def _role_contracts(
    operation: LTX23KitchenOperation,
) -> Mapping[
    str,
    tuple[
        ResourceKind, str, ResourceFormat, ArtifactPrecision | None, LTX23ArtifactContract | None
    ],
]:
    common = {
        "pipeline_support": (
            ResourceKind.MODEL,
            "pipeline_support",
            ResourceFormat.DIRECTORY,
            None,
            None,
        ),
        "text_encoder": (
            ResourceKind.MODEL,
            "text_encoder",
            ResourceFormat.SAFETENSORS,
            ArtifactPrecision.FP4,
            LTX23_GEMMA_MIXED,
        ),
    }
    if operation == "ltx23_distilled_flf":
        return MappingProxyType(
            {
                **common,
                "checkpoint": (
                    ResourceKind.MODEL,
                    "checkpoint",
                    ResourceFormat.SAFETENSORS,
                    ArtifactPrecision.FP8,
                    LTX23_DISTILLED_FP8,
                ),
            }
        )
    if operation not in {"ltx23_dev_t2v", "ltx23_dev_i2v"}:
        raise ValueError(f"unsupported LTX 2.3 operation {operation!r}")
    return MappingProxyType(
        {
            **common,
            "checkpoint": (
                ResourceKind.MODEL,
                "checkpoint",
                ResourceFormat.SAFETENSORS,
                ArtifactPrecision.FP8,
                LTX23_DEV_FP8,
            ),
            "model_lora": (
                ResourceKind.LORA,
                "model_lora",
                ResourceFormat.SAFETENSORS,
                ArtifactPrecision.BF16,
                LTX23_MODEL_LORA,
            ),
            "text_lora": (
                ResourceKind.LORA,
                "text_lora",
                ResourceFormat.SAFETENSORS,
                ArtifactPrecision.BF16,
                LTX23_TEXT_LORA,
            ),
            "latent_upscaler": (
                ResourceKind.MODEL,
                "latent_upscaler",
                ResourceFormat.SAFETENSORS,
                ArtifactPrecision.BF16,
                LTX23_SPATIAL_UPSCALER,
            ),
        }
    )


def _descriptor_errors(
    resource: ResourceDescriptor,
    role: str,
    expected: tuple[
        ResourceKind, str, ResourceFormat, ArtifactPrecision | None, LTX23ArtifactContract | None
    ],
) -> list[str]:
    kind, component, format_, precision, contract = expected
    errors: list[str] = []
    if resource.kind != kind or resource.family != "ltx23" or resource.component != component:
        errors.append(f"{role} descriptor family/kind/component differs from the recipe")
    if resource.format != format_ or (precision is not None and resource.precision != precision):
        errors.append(f"{role} descriptor format/precision differs from the recipe")
    if role != "pipeline_support" and resource.quantization != ArtifactQuantization.NATIVE:
        errors.append(f"{role} descriptor must not request runtime conversion")
    if role != "pipeline_support" and resource.base_model != LTX23_BASE_MODEL:
        errors.append(f"{role} descriptor base_model differs from the recipe")
    if contract is not None:
        declared = resource.metadata.get("header_schema_sha256")
        if declared != contract.schema_sha256:
            errors.append(f"{role} descriptor schema does not match the pinned artifact")
        source_hashes = {source.sha256 for source in resource.sources if source.sha256}
        if contract.source_sha256 not in source_hashes:
            errors.append(f"{role} descriptor source hash does not match the pinned artifact")
    return errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
