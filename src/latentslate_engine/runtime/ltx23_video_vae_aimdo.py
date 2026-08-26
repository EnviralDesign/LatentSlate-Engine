"""Operation-local AIMDO residency for the pinned LTX 2.3 video VAE.

This owner adapts ComfyUI v0.34.0 commit
``12d5279438bfefc058a269eae805ceab6047777f`` ``ModelPatcherDynamic`` and
``comfy.ops`` at the narrow VAE seam: direct-state leaves larger than 16 KiB
are faulted, bound, and unpinned around the operation that consumes them;
tiny and cross-leaf alias state remains force-loaded. There is no VAE
prefetch, graph executor, or global model manager.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn

from .framework.residency.aimdo import AimdoDynamicResidency
from .framework.residency.dynamic import DynamicResidencyLease, DynamicResidencyPoisoned
from .ltx23_av_stored_adapter import (
    LTX23LeafSchedule,
    LTX23LeafStorage,
    LTX23ModuleBinding,
    capture_ltx23_leaf_storages,
)


@dataclass(slots=True)
class _LeafState:
    leaf: LTX23LeafStorage
    users: int = 0
    lease: DynamicResidencyLease | None = None
    binding: LTX23ModuleBinding | None = None


class LTX23VideoVAEAimdoState:
    """One VAE-local owner for per-leaf AIMDO residency and method scopes."""

    def __init__(self, vae: nn.Module, device: torch.device | str) -> None:
        self.vae = vae
        self.device = torch.device(device)
        captured = capture_ltx23_leaf_storages(
            vae,
            schedule_resolver=lambda path, _slots, _sources: LTX23LeafSchedule(
                path or "<root>"
            ),
        )
        self._leaves = tuple(_LeafState(leaf) for leaf in captured)
        self._by_path = {state.leaf.path: state for state in self._leaves}
        if len(self._by_path) != len(self._leaves):
            raise RuntimeError("LTX video VAE leaf paths are not unique")
        self._root_leaves = tuple(
            state
            for state in self._leaves
            if any(slot.module is vae for slot in state.leaf.storage.slots)
        )
        self.stored_bytes = sum(state.leaf.storage.physical_bytes for state in self._leaves)
        if self.stored_bytes <= 0:
            raise ValueError("LTX video VAE AIMDO state has no stored bytes")

        self._handles: list[Any] = []
        self._force_bindings: list[LTX23ModuleBinding] = []
        self._owner_thread: int | None = None
        self._closed = False
        self._scope: str | None = None
        self._poison_reason: str | None = None
        self._backend: AimdoDynamicResidency | None = None
        self._original_encode = vae.encode
        self._original_decode = vae.decode
        self._had_instance_encode = "encode" in vars(vae)
        self._had_instance_decode = "decode" in vars(vae)
        self._leaf_bind_calls = 0
        self._operation_calls = {"encode": 0, "decode": 0}
        self._operation_seconds = {"encode": 0.0, "decode": 0.0}
        self._whole_module_move_calls = 0

        try:
            self._initialize_backend()
            self._attach()
        except BaseException:
            self._close_after_failed_init()
            raise

    @property
    def policy(self) -> dict[str, Any]:
        return {
            "mode": "comfy_direct_video_vae_leaf_vbar",
            "stored_bytes": self.stored_bytes,
            "leaf_allocation_count": len(self._leaves),
            "force_resident_leaf_count": len(self._force_bindings),
            "prefetch": False,
            "full_module_moves": False,
            "dynamic_vram": self.diagnostics(),
        }

    def terminal_poison_reason(self) -> str | None:
        if self._poison_reason is not None:
            return self._poison_reason
        return None if self._backend is None else self._backend.terminal_poison_reason()

    def diagnostics(self) -> dict[str, Any]:
        dynamic = (
            {
                "backend": "cpu-test-double",
                "version": None,
                "mode": "dynamic_vbar",
                "allocation_count": 0,
                "loaded_bytes": 0,
                "faults": 0,
                "signature_hits": 0,
                "signature_misses": 0,
                "fault_none_temporaries": 0,
                "pinned_copy_bytes": 0,
                "pageable_copy_bytes": 0,
                "unpin_calls": 0,
            }
            if self._backend is None
            else self._backend.diagnostics()
        )
        return {
            **dynamic,
            "mode": "ltx23_video_vae_direct",
            "stored_bytes": self.stored_bytes,
            "leaf_bind_calls": self._leaf_bind_calls,
            "operation_calls": dict(self._operation_calls),
            "operation_seconds": dict(self._operation_seconds),
            "whole_module_move_calls": self._whole_module_move_calls,
            "active_scope": self._scope,
            "poison_reason": self.terminal_poison_reason(),
        }

    def diagnostics_delta(self, before: dict[str, Any]) -> dict[str, Any]:
        """Return the bounded per-generation delta from one prior snapshot."""

        after = self.diagnostics()
        scalar_fields = (
            "faults",
            "signature_hits",
            "signature_misses",
            "fault_none_temporaries",
            "pinned_copy_bytes",
            "pageable_copy_bytes",
            "unpin_calls",
            "leaf_bind_calls",
            "whole_module_move_calls",
        )
        if any(
            not isinstance(before.get(field), int)
            or isinstance(before[field], bool)
            or before[field] > after[field]
            for field in scalar_fields
        ):
            raise ValueError("LTX video VAE diagnostic snapshot is not canonical")
        delta: dict[str, Any] = {
            field: after[field] - before[field] for field in scalar_fields
        }
        for field in ("operation_calls", "operation_seconds"):
            prior = before.get(field)
            current = after[field]
            if (
                not isinstance(prior, dict)
                or set(prior) != {"encode", "decode"}
                or any(prior[name] > current[name] for name in prior)
            ):
                raise ValueError("LTX video VAE operation snapshot is not canonical")
            delta[field] = {
                name: current[name] - prior[name] for name in ("encode", "decode")
            }
        delta["h2d_bytes"] = delta["pinned_copy_bytes"] + delta["pageable_copy_bytes"]
        delta["full_module_moves"] = False
        return delta

    def close(self) -> None:
        if self._closed:
            return
        self._require_owner()
        if self._scope is not None:
            raise RuntimeError("cannot close LTX video VAE residency during an operation")
        self._release_all_operation_state()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._restore_methods()
        if self._backend is not None:
            try:
                self._backend.close()
            except DynamicResidencyPoisoned as poison:
                self._poison_reason = poison.reason
                self.vae._latentslate_ltx23_video_vae_residency_poisoned = poison.reason
                raise
            self._backend = None
        # Dynamic close is the device-quiescence proof for tiny force-loaded
        # values too. Never drop their CUDA owners before that barrier.
        for binding in reversed(self._force_bindings):
            binding.restore_cpu()
        self._force_bindings.clear()
        self._closed = True

    def _initialize_backend(self) -> None:
        dynamic = tuple(state for state in self._leaves if not state.leaf.force_resident)
        if self.device.type == "cuda":
            group_values = tuple(
                tuple(slot.cpu_value for slot in state.leaf.storage.slots) for state in dynamic
            )
            group_bytes = tuple(AimdoDynamicResidency.group_bytes(values) for values in group_values)
            backend = AimdoDynamicResidency(
                self.device,
                virtual_bytes=sum(size + 512 for size in group_bytes),
                # Current media materialization owns ordinary immutable CPU
                # tensors, not authenticated file spans. Do not invent file
                # DMA metadata at this boundary.
                gathered_host_transfer=False,
            )
            self._backend = backend
            self.device = backend.device
            for state, values in zip(dynamic, group_values, strict=True):
                backend.allocate_group(state.leaf.path, values)
            backend.prioritize()
        for state in self._leaves:
            if not state.leaf.force_resident:
                continue
            binding = state.leaf.storage.copy_to(self.device)
            binding.activate()
            self._force_bindings.append(binding)

    def _attach(self) -> None:
        for state in self._leaves:
            if state.leaf.force_resident:
                continue
            modules = tuple(dict.fromkeys(slot.module for slot in state.leaf.storage.slots))
            for module in modules:
                if module is self.vae:
                    continue
                self._handles.append(module.register_forward_pre_hook(self._leaf_pre(state)))
                self._handles.append(
                    module.register_forward_hook(self._leaf_post(state), always_call=True)
                )

        owner = self

        def encode_proxy(_vae: nn.Module, *args: Any, **kwargs: Any) -> Any:
            with owner._operation_scope("encode"):
                return owner._original_encode(*args, **kwargs)

        def decode_proxy(_vae: nn.Module, *args: Any, **kwargs: Any) -> Any:
            with owner._operation_scope("decode"):
                return owner._original_decode(*args, **kwargs)

        self.vae.encode = MethodType(encode_proxy, self.vae)
        self.vae.decode = MethodType(decode_proxy, self.vae)

    @contextmanager
    def _operation_scope(self, name: str):
        self._require_owner()
        if self._closed or self._scope is not None:
            raise RuntimeError("LTX video VAE residency operation scope is unavailable")
        if name not in self._operation_calls:
            raise ValueError("LTX video VAE residency operation is not canonical")
        if self._backend is not None:
            self._backend.prioritize()
        self._scope = name
        started = time.perf_counter()
        try:
            for state in self._root_leaves:
                self._enter_leaf(state)
            yield
        finally:
            try:
                self._release_all_operation_state()
            finally:
                self._operation_calls[name] += 1
                self._operation_seconds[name] += time.perf_counter() - started
                self._scope = None

    def _leaf_pre(self, state: _LeafState):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            if self._scope is None:
                raise RuntimeError("LTX video VAE leaf executed outside encode/decode scope")
            self._enter_leaf(state)

        return hook

    def _leaf_post(self, state: _LeafState):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            self._leave_leaf(state)
            return output

        return hook

    def _enter_leaf(self, state: _LeafState) -> None:
        if state.leaf.force_resident:
            return
        if state.users == 0:
            if self._backend is None:
                values = tuple(slot.cpu_value for slot in state.leaf.storage.slots)
                lease = DynamicResidencyLease(values, object())
            else:
                lease = self._backend.acquire(state.leaf.path)
            binding = LTX23ModuleBinding(state.leaf.storage, lease.values, self.device)
            binding.activate()
            state.lease = lease
            state.binding = binding
            self._leaf_bind_calls += 1
        state.users += 1

    def _leave_leaf(self, state: _LeafState) -> None:
        if state.leaf.force_resident or state.users <= 0:
            return
        state.users -= 1
        if state.users:
            return
        binding, lease = state.binding, state.lease
        state.binding = None
        state.lease = None
        if binding is not None:
            binding.restore_cpu()
        if self._backend is not None and lease is not None:
            try:
                self._backend.release(lease)
            except DynamicResidencyPoisoned as poison:
                self._poison_reason = poison.reason
                self.vae._latentslate_ltx23_video_vae_residency_poisoned = poison.reason
                raise

    def _release_all_operation_state(self) -> None:
        for state in self._leaves:
            while state.users:
                self._leave_leaf(state)

    def _restore_methods(self) -> None:
        if self._had_instance_encode:
            self.vae.encode = self._original_encode
        else:
            vars(self.vae).pop("encode", None)
        if self._had_instance_decode:
            self.vae.decode = self._original_decode
        else:
            vars(self.vae).pop("decode", None)

    def _require_owner(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("LTX video VAE residency crossed execution threads")

    def _close_after_failed_init(self) -> None:
        try:
            self.close()
        except DynamicResidencyPoisoned:
            raise
        except BaseException as exc:
            self._poison_reason = "ltx23_av_dynamic_initialization_cleanup_failed"
            self.vae._latentslate_ltx23_video_vae_residency_poisoned = self._poison_reason
            raise DynamicResidencyPoisoned(self._poison_reason) from exc
