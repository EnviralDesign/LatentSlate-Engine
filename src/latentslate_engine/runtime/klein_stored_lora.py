"""LoRA lifecycle for Comfy/Kitchen-backed Klein transformers.

Stored FP8 and NVFP4 linears cannot be handed to PEFT without replacing the
native quantized matmul.  Comfy's bypass contract is algebraically equivalent:
keep the base linear intact and add ``up(down(x)) * alpha/rank * strength``.
This module validates and installs that branch without dequantizing base weights.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from .kit import path_signature
from .klein_stored_adapter import (
    KleinStoredDenseLoraLinear,
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    map_comfy_flux2_parameter,
)

if TYPE_CHECKING:
    from ..tools.base import LoraExecution

_STORED_LINEAR_TYPES = (
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    KleinStoredDenseLoraLinear,
)
_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_LORA_SUFFIXES = (
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    ("_lora.down.weight", "down"),
    ("_lora.up.weight", "up"),
    (".lora.down.weight", "down"),
    (".lora.up.weight", "up"),
)


@dataclass(frozen=True, slots=True)
class _Target:
    module_name: str
    down_key: str
    up_key: str
    alpha_key: str | None
    row_start: int
    row_count: int


@dataclass(frozen=True, slots=True)
class KleinStoredLoraPlan:
    path: Path
    targets: tuple[_Target, ...]
    consumed_keys: frozenset[str]


@dataclass(slots=True)
class _Loaded:
    resource_id: str
    adapter_name: str
    signature: str
    modules: tuple[str, ...]


def _adapter_name(resource_id: str) -> str:
    return "ls_" + hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]


def _split_role(key: str) -> tuple[str, str] | None:
    for suffix, role in _LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], role
    return None


def _module_targets(transformer: Any, stem: str) -> tuple[str, ...]:
    if stem.startswith("diffusion_model."):
        source = stem.removeprefix("diffusion_model.") + ".weight"
        mapped = map_comfy_flux2_parameter(source)
        if mapped:
            return tuple(target.removesuffix(".weight") for target in mapped)

    candidates = [stem]
    for prefix in (
        "transformer.",
        "base_model.model.",
        "base_model.model.transformer.",
        "unet.",
    ):
        if stem.startswith(prefix):
            candidates.append(stem.removeprefix(prefix))
    candidates.extend(candidate.replace(".processor.", ".") for candidate in tuple(candidates))
    for candidate in candidates:
        try:
            module = transformer.get_submodule(candidate)
        except AttributeError:
            continue
        if isinstance(module, (*_STORED_LINEAR_TYPES, torch.nn.Linear)):
            return (candidate,)
    return ()


def plan_klein_stored_lora(transformer: Any, path: Path) -> KleinStoredLoraPlan:
    """Prove one LoRA maps completely onto stored Klein linear modules."""

    resolved = Path(path).resolve(strict=True)
    pairs: dict[str, dict[str, str]] = {}
    alpha_keys: set[str] = set()
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            role = _split_role(key)
            if role is not None:
                stem, kind = role
                pair = pairs.setdefault(stem, {})
                if kind in pair:
                    raise ValueError(f"Klein LoRA has duplicate {kind} tensor for {stem!r}")
                pair[kind] = key
            elif key.endswith(".alpha"):
                alpha_keys.add(key)
            else:
                raise ValueError(f"Klein stored LoRA tensor is unsupported: {key!r}")

        targets: list[_Target] = []
        consumed: set[str] = set()
        claimed_modules: set[str] = set()
        for stem, pair in sorted(pairs.items()):
            if set(pair) != {"down", "up"}:
                raise ValueError(f"Klein LoRA pair is incomplete for {stem!r}")
            down_key = pair["down"]
            up_key = pair["up"]
            down_shape = tuple(handle.get_slice(down_key).get_shape())
            up_shape = tuple(handle.get_slice(up_key).get_shape())
            if len(down_shape) != 2 or len(up_shape) != 2 or up_shape[1] != down_shape[0]:
                raise ValueError(f"Klein LoRA pair geometry is invalid for {stem!r}")
            module_names = _module_targets(transformer, stem)
            if not module_names:
                raise ValueError(f"Klein LoRA target is unsupported: {stem!r}")
            row_counts: list[int] = []
            for module_name in module_names:
                module = transformer.get_submodule(module_name)
                if not isinstance(module, (*_STORED_LINEAR_TYPES, torch.nn.Linear)):
                    raise TypeError(f"Klein LoRA target {module_name!r} is not a linear module")
                if module_name in claimed_modules:
                    raise ValueError(f"Klein LoRA maps more than once to {module_name!r}")
                if down_shape[1] != module.weight.shape[1]:
                    raise ValueError(f"Klein LoRA input shape differs for {module_name!r}")
                row_counts.append(int(module.weight.shape[0]))
                claimed_modules.add(module_name)
            if sum(row_counts) != up_shape[0]:
                raise ValueError(f"Klein LoRA output shape differs for {stem!r}")
            alpha_key = stem + ".alpha"
            if alpha_key not in keys:
                alpha_key = None
            offset = 0
            for module_name, row_count in zip(module_names, row_counts, strict=True):
                targets.append(
                    _Target(
                        module_name=module_name,
                        down_key=down_key,
                        up_key=up_key,
                        alpha_key=alpha_key,
                        row_start=offset,
                        row_count=row_count,
                    )
                )
                offset += row_count
            consumed.update((down_key, up_key))
            if alpha_key is not None:
                consumed.add(alpha_key)

        if not targets:
            raise ValueError("Klein LoRA contains no supported transformer adapters")
        if consumed != keys:
            extra = ", ".join(sorted(keys - consumed)[:5])
            raise ValueError(f"Klein LoRA contains unconsumed tensors: {extra}")
        if alpha_keys - consumed:
            raise ValueError("Klein LoRA contains alpha values without matching adapters")
    return KleinStoredLoraPlan(resolved, tuple(targets), frozenset(consumed))


def install_klein_stored_lora(
    transformer: Any,
    plan: KleinStoredLoraPlan,
    adapter_name: str,
) -> tuple[str, ...]:
    """Install one fully validated LoRA transactionally on CPU."""

    if not _ADAPTER_NAME.fullmatch(adapter_name):
        raise ValueError("Klein stored LoRA adapter name is unsafe")
    installed: list[str] = []
    promoted: list[tuple[str, torch.nn.Linear]] = []
    tensors: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(plan.path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(plan.consumed_keys):
                raise ValueError("Klein LoRA changed after planning")
            for target in plan.targets:
                if target.down_key not in tensors:
                    tensors[target.down_key] = handle.get_tensor(target.down_key)
                if target.up_key not in tensors:
                    tensors[target.up_key] = handle.get_tensor(target.up_key)
                down = tensors[target.down_key]
                up = tensors[target.up_key]
                alpha = None
                if target.alpha_key is not None:
                    if target.alpha_key not in tensors:
                        tensors[target.alpha_key] = handle.get_tensor(target.alpha_key)
                    alpha_tensor = tensors[target.alpha_key]
                    if alpha_tensor.numel() != 1 or not alpha_tensor.dtype.is_floating_point:
                        raise ValueError("Klein LoRA alpha must be one floating-point scalar")
                    alpha = float(alpha_tensor.item())
                    if not math.isfinite(alpha):
                        raise ValueError("Klein LoRA alpha must be finite")
                module = transformer.get_submodule(target.module_name)
                if type(module) is torch.nn.Linear:
                    original = module
                    module = KleinStoredDenseLoraLinear(original)
                    parent_path, _, leaf = target.module_name.rpartition(".")
                    parent = transformer.get_submodule(parent_path) if parent_path else transformer
                    setattr(parent, leaf, module)
                    promoted.append((target.module_name, original))
                if not isinstance(module, _STORED_LINEAR_TYPES):
                    raise TypeError("Klein LoRA target changed after planning")
                up_part = up.narrow(0, target.row_start, target.row_count)
                module.add_lora_adapter(adapter_name, down.clone(), up_part.clone(), alpha=alpha)
                installed.append(target.module_name)
        return tuple(installed)
    except BaseException:
        for module_name in reversed(installed):
            module = transformer.get_submodule(module_name)
            if isinstance(module, _STORED_LINEAR_TYPES):
                module.delete_lora_adapter(adapter_name)
        for module_name, original in reversed(promoted):
            parent_path, _, leaf = module_name.rpartition(".")
            parent = transformer.get_submodule(parent_path) if parent_path else transformer
            setattr(parent, leaf, original)
        raise


class KleinStoredLoraLifecycle:
    """Bounded warm-switching lifecycle for stored quantized Klein LoRAs."""

    def __init__(self, *, max_loaded: int = 8) -> None:
        self.max_loaded = max(1, int(max_loaded))
        self._loaded: OrderedDict[str, _Loaded] = OrderedDict()
        self._active_modules: tuple[str, ...] = ()

    def apply(
        self,
        transformer: Any,
        loras: tuple[LoraExecution, ...],
    ) -> dict[str, Any]:
        resource_ids = [lora.resource_id for lora in loras]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("The same LoRA resource cannot be selected more than once")
        if len(loras) > self.max_loaded:
            raise ValueError(f"This runtime allows at most {self.max_loaded} active LoRAs")

        desired = [
            (
                lora,
                _adapter_name(lora.resource_id),
                path_signature(Path(lora.path)),
            )
            for lora in loras
        ]
        desired_names = {name for _lora, name, _signature in desired}
        for _lora, name, signature in desired:
            existing = self._loaded.get(name)
            if existing is not None and existing.signature != signature:
                self._delete(transformer, name)
        while len(self._loaded) + sum(name not in self._loaded for _, name, _ in desired) > self.max_loaded:
            victim = next((name for name in self._loaded if name not in desired_names), None)
            if victim is None:
                raise RuntimeError("No inactive stored LoRA can be evicted")
            self._delete(transformer, victim)

        loaded_now = 0
        reused = 0
        for lora, name, signature in desired:
            if name not in self._loaded:
                plan = plan_klein_stored_lora(transformer, Path(lora.path))
                modules = install_klein_stored_lora(transformer, plan, name)
                self._loaded[name] = _Loaded(lora.resource_id, name, signature, modules)
                loaded_now += 1
            else:
                self._loaded.move_to_end(name)
                reused += 1

        for loaded in self._loaded.values():
            for module_name in loaded.modules:
                module = transformer.get_submodule(module_name)
                module.set_lora_strength(loaded.adapter_name, 0.0)
        active_modules: set[str] = set()
        for lora, name, _signature in desired:
            loaded = self._loaded[name]
            for module_name in loaded.modules:
                module = transformer.get_submodule(module_name)
                module.set_lora_strength(name, float(lora.strength))
                active_modules.add(module_name)
        self._active_modules = tuple(sorted(active_modules))
        return {
            "backend": "comfy-compatible/additive-bypass",
            "active": resource_ids,
            "weights": [float(lora.strength) for lora in loras],
            "loaded": [entry.resource_id for entry in self._loaded.values()],
            "loaded_now": loaded_now,
            "reused": reused,
            "target_module_count": len(self._active_modules),
        }

    def dispatch_snapshot(self, transformer: Any) -> dict[str, int]:
        return {
            name: int(transformer.get_submodule(name).lora_dispatch_count)
            for name in self._active_modules
        }

    def verify_dispatch(self, transformer: Any, before: dict[str, int]) -> dict[str, Any]:
        if set(before) != set(self._active_modules):
            raise RuntimeError("Klein stored LoRA active module set changed during generation")
        deltas = {
            name: int(transformer.get_submodule(name).lora_dispatch_count) - count
            for name, count in before.items()
        }
        missed = sorted(name for name, delta in deltas.items() if delta <= 0)
        if missed:
            raise RuntimeError(
                f"Klein stored LoRA did not execute on {len(missed)} selected modules"
            )
        return {
            "status": "proven",
            "backend": "comfy-compatible/additive-bypass",
            "module_count": len(deltas),
            "total_dispatch_delta": sum(deltas.values()),
            "min_dispatch_delta": min(deltas.values(), default=0),
            "max_dispatch_delta": max(deltas.values(), default=0),
        }

    def clear(self, transformer: Any | None = None) -> None:
        if transformer is None and self._loaded:
            raise RuntimeError("A live stored LoRA lifecycle requires its transformer to clear")
        if transformer is not None:
            for name in list(self._loaded):
                self._delete(transformer, name)
        self._active_modules = ()

    def status(self) -> dict[str, Any]:
        return {
            "loaded": [entry.resource_id for entry in self._loaded.values()],
            "max_loaded": self.max_loaded,
            "backend": "comfy-compatible/additive-bypass",
        }

    def _delete(self, transformer: Any, adapter_name: str) -> None:
        loaded = self._loaded.pop(adapter_name)
        for module_name in loaded.modules:
            module = transformer.get_submodule(module_name)
            module.delete_lora_adapter(adapter_name)
