"""Exact fixed LoRA bypass for the stored Z-Image Turbo ConvRot transformer.

The Kitchen INT8 base weight remains immutable.  Each validated BF16 pair is an
Engine-owned additive branch; Q/K/V pairs write only their disjoint row slice
of the fused ``attention.qkv`` output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact

Z_IMAGE_70S_HORROR_LORA_SIZE = 85_094_800
Z_IMAGE_70S_HORROR_LORA_SHA256 = "c50285bd237c3b6f022aafd1b47ebed75a7137466c228ff516b061bede3c5236"
Z_IMAGE_70S_HORROR_LORA_HEADER_SHA256 = (
    "0f28d13bb8128539a02eebe1065757232969c3bf8bf09e66d510487198885778"
)
Z_IMAGE_70S_HORROR_LORA_SCHEMA_SHA256 = (
    "8b6aa274d5530c5b9d906c0855445f28d209b0eb7af9a31b198f0f4edf3c2088"
)
Z_IMAGE_70S_HORROR_LORA_RESOURCE_ID = "lora:zimage:kutches--imagezv2/70s-horror-movie-b"
Z_IMAGE_70S_HORROR_LORA_STRENGTH = 1.0
Z_IMAGE_70S_HORROR_LORA_RANK = 16
Z_IMAGE_70S_HORROR_LORA_TARGETS = 240


@dataclass(frozen=True, slots=True)
class ZImageFixedLoraTarget:
    target_id: str
    module_name: str
    down_key: str
    up_key: str
    row_start: int
    row_count: int
    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class ZImageFixedLoraPlan:
    identity: ArtifactIdentity
    file_sha256: str
    schema_sha256: str
    resource_id: str
    strength: float
    targets: tuple[ZImageFixedLoraTarget, ...]
    consumed_keys: frozenset[str]

    def require_exact_layout(self) -> None:
        if (
            self.identity.size_bytes != Z_IMAGE_70S_HORROR_LORA_SIZE
            or self.identity.header_sha256 != Z_IMAGE_70S_HORROR_LORA_HEADER_SHA256
            or self.file_sha256 != Z_IMAGE_70S_HORROR_LORA_SHA256
            or self.schema_sha256 != Z_IMAGE_70S_HORROR_LORA_SCHEMA_SHA256
            or self.resource_id != Z_IMAGE_70S_HORROR_LORA_RESOURCE_ID
            or self.strength != Z_IMAGE_70S_HORROR_LORA_STRENGTH
            or len(self.targets) != Z_IMAGE_70S_HORROR_LORA_TARGETS
            or len(self.consumed_keys) != 480
            or self.targets != _EXPECTED_TARGETS
            or self.consumed_keys != _EXPECTED_KEYS
        ):
            raise ValueError("Z-Image fixed LoRA identity/layout differs from the exact pin")
        qkv = tuple(
            target for target in self.targets if target.module_name.endswith(".attention.qkv")
        )
        direct = tuple(
            target for target in self.targets if not target.module_name.endswith(".attention.qkv")
        )
        if len(qkv) != 90 or len(direct) != 150:
            raise ValueError(
                "Z-Image fixed LoRA target category closure differs from the exact pin"
            )
        if len({target.target_id for target in self.targets}) != len(self.targets):
            raise ValueError("Z-Image fixed LoRA target IDs collide")
        claimed: dict[str, list[tuple[int, int]]] = {}
        for target in self.targets:
            claimed.setdefault(target.module_name, []).append(
                (target.row_start, target.row_start + target.row_count)
            )
        if len(claimed) != 180:
            raise ValueError("Z-Image fixed LoRA module closure differs from the exact pin")
        for module_name, spans in claimed.items():
            ordered = sorted(spans)
            if any(end > following for (_start, end), (following, _next) in pairwise(ordered)):
                raise ValueError(f"Z-Image fixed LoRA row slices collide: {module_name}")


def _expected_targets() -> tuple[ZImageFixedLoraTarget, ...]:
    targets: list[ZImageFixedLoraTarget] = []
    for layer in range(30):
        source_prefix = f"diffusion_model.layers.{layer}"
        module_prefix = f"layers.{layer}"
        for source, row_start in (("to_q", 0), ("to_k", 3840), ("to_v", 7680)):
            stem = f"{source_prefix}.attention.{source}"
            targets.append(
                ZImageFixedLoraTarget(
                    stem,
                    f"{module_prefix}.attention.qkv",
                    stem + ".lora_A.weight",
                    stem + ".lora_B.weight",
                    row_start,
                    3840,
                    3840,
                    3840,
                )
            )
        direct = (
            ("adaLN_modulation.0", "adaLN_modulation.0", 256, 15360),
            ("attention.to_out.0", "attention.out", 3840, 3840),
            ("feed_forward.w1", "feed_forward.w1", 3840, 10240),
            ("feed_forward.w2", "feed_forward.w2", 10240, 3840),
            ("feed_forward.w3", "feed_forward.w3", 3840, 10240),
        )
        for source_suffix, module_suffix, in_features, out_features in direct:
            stem = f"{source_prefix}.{source_suffix}"
            targets.append(
                ZImageFixedLoraTarget(
                    stem,
                    f"{module_prefix}.{module_suffix}",
                    stem + ".lora_A.weight",
                    stem + ".lora_B.weight",
                    0,
                    out_features,
                    in_features,
                    out_features,
                )
            )
    return tuple(sorted(targets, key=lambda target: target.target_id))


_EXPECTED_TARGETS = _expected_targets()
_EXPECTED_KEYS = frozenset(
    key for target in _EXPECTED_TARGETS for key in (target.down_key, target.up_key)
)


def plan_z_image_70s_horror_lora(path: Path) -> ZImageFixedLoraPlan:
    """Prove the exact Kutches ImageZV2 artifact and its closed 240-target map."""

    probe = probe_artifact(Path(path).resolve(strict=True))
    if (
        probe.format != "safetensors"
        or probe.identity.size_bytes != Z_IMAGE_70S_HORROR_LORA_SIZE
        or probe.identity.header_sha256 != Z_IMAGE_70S_HORROR_LORA_HEADER_SHA256
        or probe.schema_sha256 != Z_IMAGE_70S_HORROR_LORA_SCHEMA_SHA256
        or probe.tensor_count != 480
        or probe.tensor_dtypes != ("BF16",)
    ):
        raise ValueError("Z-Image fixed LoRA header/schema differs from the exact pin")
    digest = _sha256_file(probe.identity.path)
    if digest != Z_IMAGE_70S_HORROR_LORA_SHA256:
        raise ValueError("Z-Image fixed LoRA payload SHA-256 differs from the exact pin")
    with safe_open(str(probe.identity.path), framework="pt", device="cpu") as handle:
        keys = frozenset(handle.keys())
        if keys != _EXPECTED_KEYS:
            raise ValueError("Z-Image fixed LoRA key closure differs from the exact pin")
        for target in _EXPECTED_TARGETS:
            down_shape = tuple(handle.get_slice(target.down_key).get_shape())
            up_shape = tuple(handle.get_slice(target.up_key).get_shape())
            if down_shape != (Z_IMAGE_70S_HORROR_LORA_RANK, target.in_features) or up_shape != (
                target.out_features,
                Z_IMAGE_70S_HORROR_LORA_RANK,
            ):
                raise ValueError(
                    f"Z-Image fixed LoRA rank/geometry differs for {target.target_id!r}"
                )
    plan = ZImageFixedLoraPlan(
        probe.identity,
        digest,
        probe.schema_sha256,
        Z_IMAGE_70S_HORROR_LORA_RESOURCE_ID,
        Z_IMAGE_70S_HORROR_LORA_STRENGTH,
        _EXPECTED_TARGETS,
        keys,
    )
    plan.require_exact_layout()
    return plan


def revalidate_z_image_70s_horror_lora(plan: ZImageFixedLoraPlan) -> bool:
    try:
        plan.require_exact_layout()
        return (
            revalidate_artifact(plan.identity)
            and plan_z_image_70s_horror_lora(plan.identity.path) == plan
        )
    except (OSError, TypeError, ValueError):
        return False


class _ZImageFixedLoraBranch(nn.Module):
    def __init__(self, target: ZImageFixedLoraTarget, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        if (
            down.dtype is not torch.bfloat16
            or up.dtype is not torch.bfloat16
            or tuple(down.shape) != (Z_IMAGE_70S_HORROR_LORA_RANK, target.in_features)
            or tuple(up.shape) != (target.out_features, Z_IMAGE_70S_HORROR_LORA_RANK)
        ):
            raise ValueError("Z-Image fixed LoRA materialized tensor geometry differs")
        self.down = nn.Parameter(down.contiguous(), requires_grad=False)
        self.up = nn.Parameter(up.contiguous(), requires_grad=False)
        self.target_id = target.target_id
        self.row_start = target.row_start
        self.row_count = target.row_count
        self.strength = Z_IMAGE_70S_HORROR_LORA_STRENGTH
        self.dispatch_count = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype is not torch.bfloat16:
            raise TypeError("Z-Image fixed LoRA requires BF16 transformer activations")
        self.dispatch_count += 1
        return F.linear(F.linear(value, self.down), self.up) * self.strength


def add_z_image_fixed_lora_branch(
    module: nn.Module,
    name: str,
    target: ZImageFixedLoraTarget,
    down: torch.Tensor,
    up: torch.Tensor,
) -> None:
    from .z_image_stored_adapter import ZImageStoredConvRotLinear

    if not isinstance(module, ZImageStoredConvRotLinear):
        raise TypeError("Z-Image fixed LoRA target is not a stored ConvRot linear")
    base_rows = 11520 if target.module_name.endswith(".attention.qkv") else target.out_features
    if tuple(module.weight.shape) != (base_rows, target.in_features):
        raise ValueError("Z-Image fixed LoRA target geometry differs from the immutable base")
    if name in module._fixed_lora_branches:
        raise ValueError("Z-Image fixed LoRA branch is already installed")
    module._fixed_lora_branches[name] = _ZImageFixedLoraBranch(target, down, up)


def apply_z_image_fixed_lora(
    module: nn.Module, value: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    branches = getattr(module, "_fixed_lora_branches", None)
    if not isinstance(branches, nn.ModuleDict) or not branches:
        return output
    result = output.clone()
    for branch in branches.values():
        start = int(branch.row_start)
        end = start + int(branch.row_count)
        result[..., start:end] = result[..., start:end] + branch(value)
    return result


class ZImageFixedLoraLifecycle:
    """One immutable fixed-LoRA lifecycle; no selection, merging, or LRU."""

    def __init__(self) -> None:
        self._targets: MappingProxyType[str, tuple[str, str]] = MappingProxyType({})
        self._plan: ZImageFixedLoraPlan | None = None

    def install(
        self,
        transformer: nn.Module,
        plan: ZImageFixedLoraPlan,
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict[str, Any]:
        if self._plan is not None:
            raise RuntimeError("Z-Image fixed LoRA is already installed")
        if not revalidate_z_image_70s_horror_lora(plan):
            raise ValueError("Z-Image fixed LoRA changed after planning")
        installed: list[tuple[str, str]] = []
        target_map: dict[str, tuple[str, str]] = {}
        try:
            with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
                if frozenset(handle.keys()) != plan.consumed_keys:
                    raise ValueError("Z-Image fixed LoRA changed before installation")
                for index, target in enumerate(plan.targets):
                    if cancelled():
                        raise RuntimeError("Z-Image fixed LoRA installation canceled")
                    module = transformer.get_submodule(target.module_name)
                    name = f"fixed_{index:03d}"
                    add_z_image_fixed_lora_branch(
                        module,
                        name,
                        target,
                        handle.get_tensor(target.down_key).clone(),
                        handle.get_tensor(target.up_key).clone(),
                    )
                    installed.append((target.module_name, name))
                    target_map[target.target_id] = (target.module_name, name)
            if (
                not revalidate_artifact(plan.identity)
                or _sha256_file(plan.identity.path) != plan.file_sha256
            ):
                raise ValueError("Z-Image fixed LoRA changed during installation")
            if len(target_map) != Z_IMAGE_70S_HORROR_LORA_TARGETS:
                raise RuntimeError("Z-Image fixed LoRA installation is incomplete")
        except BaseException:
            for module_name, name in reversed(installed):
                module = transformer.get_submodule(module_name)
                if name in module._fixed_lora_branches:
                    module._fixed_lora_branches.pop(name)
            raise
        self._targets = MappingProxyType(target_map)
        self._plan = plan
        return self.status()

    def dispatch_snapshot(self, transformer: nn.Module) -> dict[str, int]:
        if self._plan is None or len(self._targets) != Z_IMAGE_70S_HORROR_LORA_TARGETS:
            raise RuntimeError("Z-Image fixed LoRA is not completely installed")
        snapshot: dict[str, int] = {}
        expected_targets = {target.target_id: target for target in self._plan.targets}
        for target_id, (module_name, name) in self._targets.items():
            branch = transformer.get_submodule(module_name)._fixed_lora_branches[name]
            target = expected_targets.get(target_id)
            if (
                target is None
                or branch.target_id != target_id
                or branch.row_start != target.row_start
                or branch.row_count != target.row_count
                or branch.strength != Z_IMAGE_70S_HORROR_LORA_STRENGTH
                or branch.down.dtype is not torch.bfloat16
                or branch.up.dtype is not torch.bfloat16
                or tuple(branch.down.shape) != (Z_IMAGE_70S_HORROR_LORA_RANK, target.in_features)
                or tuple(branch.up.shape) != (target.out_features, Z_IMAGE_70S_HORROR_LORA_RANK)
            ):
                raise RuntimeError("Z-Image fixed LoRA target binding changed")
            snapshot[target_id] = int(branch.dispatch_count)
        return snapshot

    def verify_dispatch(self, transformer: nn.Module, before: dict[str, int]) -> dict[str, Any]:
        after = self.dispatch_snapshot(transformer)
        if set(after) != set(before):
            raise RuntimeError("Z-Image fixed LoRA target set changed during generation")
        deltas = {target: after[target] - before[target] for target in after}
        missed = tuple(target for target, delta in deltas.items() if delta <= 0)
        if missed:
            raise RuntimeError(
                f"Z-Image fixed LoRA did not dispatch on {len(missed)} exact targets"
            )
        values = tuple(deltas.values())
        return {
            "status": "proven",
            "backend": "engine-native/bf16-additive-bypass",
            "resource_id": Z_IMAGE_70S_HORROR_LORA_RESOURCE_ID,
            "strength": Z_IMAGE_70S_HORROR_LORA_STRENGTH,
            "rank": Z_IMAGE_70S_HORROR_LORA_RANK,
            "target_count": len(values),
            "qkv_row_slice_targets": 90,
            "direct_targets": 150,
            "total_dispatch_delta": sum(values),
            "min_target_dispatch_delta": min(values),
            "max_target_dispatch_delta": max(values),
            "complete": True,
            "base_merged_or_dequantized": False,
        }

    def clear(self, transformer: nn.Module) -> None:
        for module_name, name in reversed(tuple(self._targets.values())):
            branches = transformer.get_submodule(module_name)._fixed_lora_branches
            if name in branches:
                branches.pop(name)
        self._targets = MappingProxyType({})
        self._plan = None

    def status(self) -> dict[str, Any]:
        return {
            "backend": "engine-native/bf16-additive-bypass",
            "resource_id": self._plan.resource_id if self._plan is not None else None,
            "strength": self._plan.strength if self._plan is not None else None,
            "target_count": len(self._targets),
            "loaded": self._plan is not None,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
