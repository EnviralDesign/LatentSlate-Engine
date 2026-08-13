"""Stage-aware, Comfy-compatible additive LoRAs for native Wan transformers.

The native Wan materializer keeps the original stored FP8/INT8 linear intact.
Like Comfy's LoRA loader, this module adds ``up(down(x)) * alpha/rank`` beside
that base operation; it never converts or rewrites a checkpoint's base weights.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from ..lora import active_loras
from .wan22_stored_adapter import NativeStoredLinear, map_comfy_wan_parameter_key

if TYPE_CHECKING:
    from ..tools.base import LoraExecution

_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_LORA_SUFFIXES = ((".lora_down.weight", "down"), (".lora_up.weight", "up"))


@dataclass(frozen=True, slots=True)
class _Target:
    module_name: str
    down_key: str
    up_key: str
    alpha_key: str | None


@dataclass(frozen=True, slots=True)
class WanStoredLoraPlan:
    """A complete header-validated mapping for one Wan-stage adapter."""

    path: Path
    targets: tuple[_Target, ...]
    consumed_keys: frozenset[str]


class _Adapter(nn.Module):
    def __init__(self, down: torch.Tensor, up: torch.Tensor, *, alpha: float | None) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError("Wan LoRA up/down geometry is invalid")
        if not down.dtype.is_floating_point or not up.dtype.is_floating_point:
            raise ValueError("Wan LoRA tensors must be floating-point")
        rank = int(down.shape[0])
        if rank <= 0:
            raise ValueError("Wan LoRA rank must be positive")
        self.down = nn.Parameter(down.contiguous(), requires_grad=False)
        self.up = nn.Parameter(up.contiguous(), requires_grad=False)
        self.scale = 1.0 if alpha is None else float(alpha) / rank
        self.strength = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(input, self.down.to(dtype=input.dtype)), self.up.to(dtype=input.dtype)) * (
            self.scale * self.strength
        )


class WanStoredLoraLinear(nn.Module):
    """One base Wan linear plus an ordered collection of additive adapters."""

    def __init__(self, base: NativeStoredLinear | nn.Linear) -> None:
        super().__init__()
        if type(base) is not nn.Linear and not isinstance(base, NativeStoredLinear):
            raise TypeError("Wan LoRA target must be a native stored or dense linear")
        self.base = base
        self.adapters = nn.ModuleDict()
        self.lora_dispatch_count = 0

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.base(input)
        active = False
        for adapter in self.adapters.values():
            if adapter.strength:
                output = output + adapter(input)
                active = True
        if active:
            self.lora_dispatch_count += 1
        return output

    def add_adapter(self, name: str, down: torch.Tensor, up: torch.Tensor, *, alpha: float | None) -> None:
        if not _ADAPTER_NAME.fullmatch(name) or name in self.adapters:
            raise ValueError("Wan LoRA adapter name is invalid or already installed")
        if down.shape[1] != self.weight.shape[1] or up.shape[0] != self.weight.shape[0]:
            raise ValueError("Wan LoRA geometry differs from target linear")
        self.adapters[name] = _Adapter(down, up, alpha=alpha)

    def set_strength(self, name: str, strength: float) -> None:
        if name not in self.adapters or not math.isfinite(strength):
            raise ValueError("Wan LoRA strength is invalid")
        self.adapters[name].strength = float(strength)

    def remove_adapter(self, name: str) -> None:
        self.adapters.pop(name, None)


def _adapter_name(resource_id: str) -> str:
    return "ls_" + hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]


def _split_role(key: str) -> tuple[str, str] | None:
    for suffix, role in _LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], role
    return None


def _module_name(transformer: nn.Module, stem: str) -> str:
    normalized = stem.removeprefix("diffusion_model.")
    mapped = map_comfy_wan_parameter_key(normalized + ".weight")
    if not mapped:
        raise ValueError(f"Wan LoRA target is unsupported: {stem!r}")
    module_name = mapped.removesuffix(".weight")
    module = transformer.get_submodule(module_name)
    if not isinstance(module, (NativeStoredLinear, WanStoredLoraLinear)) and type(module) is not nn.Linear:
        raise TypeError(f"Wan LoRA target {module_name!r} is not a linear module")
    return module_name


def plan_wan_stored_lora(transformer: nn.Module, path: Path) -> WanStoredLoraPlan:
    """Prove every tensor in one LoRA maps to one native Wan linear."""

    resolved = Path(path).resolve(strict=True)
    pairs: dict[str, dict[str, str]] = {}
    alpha_keys: set[str] = set()
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            split = _split_role(key)
            if split is not None:
                stem, role = split
                if role in pairs.setdefault(stem, {}):
                    raise ValueError(f"Wan LoRA has duplicate {role} tensor for {stem!r}")
                pairs[stem][role] = key
            elif key.endswith(".alpha"):
                alpha_keys.add(key)
            else:
                raise ValueError(f"Wan LoRA tensor is unsupported: {key!r}")
        targets: list[_Target] = []
        consumed: set[str] = set()
        claimed: set[str] = set()
        for stem, pair in sorted(pairs.items()):
            if set(pair) != {"down", "up"}:
                raise ValueError(f"Wan LoRA pair is incomplete for {stem!r}")
            down_key, up_key = pair["down"], pair["up"]
            down_shape = tuple(handle.get_slice(down_key).get_shape())
            up_shape = tuple(handle.get_slice(up_key).get_shape())
            if len(down_shape) != 2 or len(up_shape) != 2 or up_shape[1] != down_shape[0]:
                raise ValueError(f"Wan LoRA pair geometry is invalid for {stem!r}")
            module_name = _module_name(transformer, stem)
            if module_name in claimed:
                raise ValueError(f"Wan LoRA maps more than once to {module_name!r}")
            module = transformer.get_submodule(module_name)
            if down_shape[1] != module.weight.shape[1] or up_shape[0] != module.weight.shape[0]:
                raise ValueError(f"Wan LoRA output/input shape differs for {stem!r}")
            alpha_key = stem + ".alpha" if stem + ".alpha" in keys else None
            targets.append(_Target(module_name, down_key, up_key, alpha_key))
            consumed.update((down_key, up_key))
            if alpha_key:
                consumed.add(alpha_key)
            claimed.add(module_name)
        if not targets:
            raise ValueError("Wan LoRA contains no supported transformer adapters")
        if consumed != keys or alpha_keys - consumed:
            raise ValueError("Wan LoRA contains unconsumed tensors")
    return WanStoredLoraPlan(resolved, tuple(targets), frozenset(consumed))


def install_wan_stored_lora(transformer: nn.Module, plan: WanStoredLoraPlan, adapter_name: str) -> tuple[str, ...]:
    """Install a planned adapter transactionally, preserving all base linears."""

    if not _ADAPTER_NAME.fullmatch(adapter_name):
        raise ValueError("Wan LoRA adapter name is unsafe")
    installed: list[str] = []
    promoted: list[tuple[str, nn.Linear]] = []
    try:
        with safe_open(str(plan.path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(plan.consumed_keys):
                raise ValueError("Wan LoRA changed after planning")
            for target in plan.targets:
                module = transformer.get_submodule(target.module_name)
                if type(module) is nn.Linear or isinstance(module, NativeStoredLinear):
                    original = module
                    module = WanStoredLoraLinear(original)
                    parent_path, _, leaf = target.module_name.rpartition(".")
                    setattr(transformer.get_submodule(parent_path) if parent_path else transformer, leaf, module)
                    if type(original) is nn.Linear:
                        promoted.append((target.module_name, original))
                if not isinstance(module, WanStoredLoraLinear):
                    raise TypeError("Wan LoRA target changed after planning")
                alpha = None
                if target.alpha_key is not None:
                    alpha_tensor = handle.get_tensor(target.alpha_key)
                    if alpha_tensor.numel() != 1 or alpha_tensor.dtype is torch.bool:
                        raise ValueError("Wan LoRA alpha must be one numeric scalar")
                    if not (alpha_tensor.dtype.is_floating_point or alpha_tensor.dtype.is_signed):
                        raise ValueError("Wan LoRA alpha must be one numeric scalar")
                    alpha = float(alpha_tensor.item())
                    if not math.isfinite(alpha):
                        raise ValueError("Wan LoRA alpha must be finite")
                module.add_adapter(adapter_name, handle.get_tensor(target.down_key), handle.get_tensor(target.up_key), alpha=alpha)
                installed.append(target.module_name)
        return tuple(installed)
    except BaseException:
        for name in reversed(installed):
            module = transformer.get_submodule(name)
            if isinstance(module, WanStoredLoraLinear):
                module.remove_adapter(adapter_name)
        for name, original in reversed(promoted):
            parent_path, _, leaf = name.rpartition(".")
            setattr(transformer.get_submodule(parent_path) if parent_path else transformer, leaf, original)
        raise


def apply_wan_stage_loras(transformer: nn.Module, loras: tuple[LoraExecution, ...]) -> dict[str, object]:
    """Install ordered nonzero adapters for one high or low Wan transformer stage."""

    active = active_loras(loras)
    if len({item.resource_id for item in active}) != len(active):
        raise ValueError("the same Wan LoRA resource cannot be selected twice in one stage")
    modules: list[str] = []
    for item in active:
        name = _adapter_name(item.resource_id)
        plan = plan_wan_stored_lora(transformer, Path(item.path))
        modules.extend(install_wan_stored_lora(transformer, plan, name))
        for module_name in plan.targets:
            target = transformer.get_submodule(module_name.module_name)
            assert isinstance(target, WanStoredLoraLinear)
            target.set_strength(name, float(item.strength))
    return {
        "backend": "comfy-compatible/additive-bypass",
        "active": [item.resource_id for item in active],
        "weights": [float(item.strength) for item in active],
        "target_module_count": len(set(modules)),
    }
