"""Wan's per-operation residency, adapted from ComfyUI 36da3ff (GPL-3.0).

comfy/ops.py owns the reference fault/refill/consume/unpin sequence. This keeps
that sequence local to one Wan identity, using the existing mapped reader and
the same AIMDO primitives as Engine's Qwen path. Raw host sources stay immutable.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path

import torch
from comfy_aimdo import control

from ..mapped_checkpoint import MappedCheckpoint


def _aligned(size):
    return (size + 1023) & -1024


def _module(name):
    module = importlib.import_module(f"comfy_aimdo.{name}")
    return importlib.reload(module) if getattr(module, "lib", True) is None else module


class _Source:
    def __init__(self, checkpoint, keys, *, patch=False):
        self.checkpoint = checkpoint
        self.patch = patch
        self.values = {key: checkpoint.tensor(key) for key in keys}
        self.offsets = {}
        self.size = 0
        for key, value in self.values.items():
            self.offsets[key] = self.size
            self.size += _aligned(value.nbytes)
        self.host_offset = None
        self.host_pin = None
        self.cached = False

    def transfer(self, owner, destination, stream):
        host_cache = owner.patch_cache if self.patch else owner.host_cache
        if self.host_offset is None:
            self.host_offset = host_cache.size
            host_cache.extend(self.size, register=False)
            pointer = host_cache.get_raw_address() + self.host_offset
            if torch.cuda.cudart().cudaHostRegister(pointer, self.size, 1) == 0:
                self.host_pin = pointer
            else:
                # An unregistered range remains usable if driver pinning fails.
                try:
                    torch.ones(1, device=owner.device) + 1
                except RuntimeError:
                    pass
        if not self.cached:
            for key, offset in self.offsets.items():
                if self.patch:
                    self.checkpoint.copy_tensor_to_host(
                        key, host_cache, self.host_offset + offset
                    )
                else:
                    self.checkpoint.copy_tensor_to_device(
                        key,
                        destination,
                        offset,
                        owner.device.index or 0,
                        stream=stream,
                        host_buffer=host_cache,
                        host_offset=self.host_offset + offset,
                    )
            self.cached = True
            if not self.patch:
                return self.views(destination)
        # LowVramPatch first gathers both adapter tensors in host memory, then
        # copies the packed bundle once. Base first fills use file->host+device.
        host = owner.aimdo_torch.hostbuf_to_tensor(host_cache)
        destination.copy_(
            host[self.host_offset : self.host_offset + self.size], non_blocking=True
        )
        return self.views(destination)

    def release_host(self):
        if self.host_pin is not None:
            torch.cuda.cudart().cudaHostUnregister(self.host_pin)
        self.host_pin = self.host_offset = None
        self.cached = False

    def views(self, destination):
        return {
            key: destination[self.offsets[key] : self.offsets[key] + value.nbytes]
            .view(value.dtype)
            .view(value.shape)
            for key, value in self.values.items()
        }


class _Binding:
    def __init__(self, owner, prefix, keys):
        self.prefix = prefix
        self.source = _Source(owner.base, keys)
        self.patches = []
        for store, checkpoint in owner.adapters:
            parts = owner.weights._lora_parts(prefix, store)
            if parts is not None:
                self.patches.append((store, _Source(checkpoint, parts[:2], patch=True)))
        self.allocation = None
        self.signature = None
        self.resident = False
        self.patched = None
        self.stream = None

    def materialize(self, owner):
        signature = owner.model_vbar.vbar_fault(self.allocation)
        self.resident = owner.model_vbar.vbar_signature_compare(
            signature, self.signature
        )
        self.signature = signature
        self.stream = None
        if self.resident:
            destination = owner.aimdo_torch.aimdo_to_tensor(
                self.allocation, owner.device
            )
            return self.source.views(destination), {}
        self.patched = None
        stream = owner.transfer_stream()
        required = self.source.size if signature is None else 0
        # ops.ensure_offload_stream rotates again for the largest transient
        # weight so its scratch is not grown on both alternating streams.
        if required and owner.largest_binding is self:
            stream = owner.transfer_stream()
        if required > owner.largest_size:
            owner.largest_binding, owner.largest_size = self, required
        self.stream = stream
        # The prior consumer has already fenced this stream in release().
        with torch.cuda.stream(stream):
            offset = 0
            if signature is not None:
                allocation = self.allocation
            else:
                allocation = owner.buffers[stream].get(self.source.size)
                offset = self.source.size
            destination = owner.aimdo_torch.aimdo_to_tensor(allocation, owner.device)
            values = self.source.transfer(owner, destination, stream)
            updates = {}
            for store, source in self.patches:
                allocation = owner.buffers[stream].get(source.size, offset)
                offset += source.size
                destination = owner.aimdo_torch.aimdo_to_tensor(
                    allocation, owner.device
                )
                updates[id(store)] = source.transfer(owner, destination, stream)
        torch.cuda.current_stream(owner.device).wait_stream(stream)
        return values, updates

    def release(self, owner):
        if self.signature is not None:
            owner.model_vbar.vbar_unpin(self.allocation)
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(owner.device))


class WanResidency:
    """Storage owned by one high- or low-noise checkpoint/adapter identity."""

    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        torch.cuda.init()
        if not control.init(nvml_pressure=True):
            raise RuntimeError("unable to initialize comfy-aimdo")
        if not control.devctxs and not control.init_device(device.index or 0):
            raise RuntimeError("unable to initialize Wan CUDA device")
        self.model_vbar = _module("model_vbar")
        self.aimdo_torch = _module("torch")
        self.base = MappedCheckpoint(
            Path(weights.base.identity.path), weights.base._key_prefix
        )
        self.adapters = [
            (store, MappedCheckpoint(Path(store.identity.path)))
            for store in (weights.lora, weights.secondary_lora)
            if store is not None
        ]
        prefixes = {
            key.removesuffix(".weight")
            for key in weights.base.keys
            if key.endswith(".weight")
            and not key.endswith((".norm_q.weight", ".norm_k.weight"))
        }
        self.bindings = {}
        for prefix in prefixes:
            keys = [
                f"{prefix}.{suffix}"
                for suffix in (
                    "weight",
                    "scale_weight",
                    "weight_scale",
                    "weight_scale_2",
                    "bias",
                )
                if f"{prefix}.{suffix}" in weights.base.keys
            ]
            self.bindings[prefix] = _Binding(self, prefix, keys)

        # Comfy's dynamic load order: tiny first, then descending offload cost.
        # Wan's FP16 patch estimate is twice the logical FP16 weight size.
        def load_order(binding):
            value = binding.source.values[f"{binding.prefix}.weight"]
            elements = value.numel() * (
                2 if f"{binding.prefix}.weight_scale_2" in weights.base.keys else 1
            )
            size = sum(t.nbytes for t in binding.source.values.values())
            extra = (
                elements * 4
                if binding.patches
                else (elements * 2 if value.dtype != torch.float16 else 0)
            )
            bias = binding.source.values.get(f"{binding.prefix}.bias")
            if bias is not None and bias.dtype != torch.float16:
                extra += bias.numel() * 2
            cost = size + extra
            return cost >= 64 * 1024, -cost, size, binding.prefix

        ordered = sorted(self.bindings.values(), key=load_order)
        total = sum(b.source.size for b in ordered)
        self.vbar = self.model_vbar.ModelVBAR(10 * total, device.index or 0)
        self.streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        self.copy_index = None
        largest = max(
            b.source.size + sum(s.size for _, s in b.patches) for b in ordered
        )
        self.buffer_capacity = (largest + 64 * 1024**2 - 1) & -(64 * 1024**2)
        self.buffers = {}
        self.largest_binding, self.largest_size = None, 0
        self.patch_size = sum(s.size for b in ordered for _, s in b.patches)
        self.host_cache = _module("host_buffer").HostBuffer(0, 64 * 1024**2, total)
        self.patch_cache = None
        for binding in ordered:
            binding.allocation = self.vbar.alloc(binding.source.size)
        streamed = {key for b in ordered for key in b.source.values}
        self.direct_keys = {
            key
            for key in weights.base.keys - streamed
            if key.endswith(
                (
                    ".modulation",
                    ".norm_q.weight",
                    ".norm_k.weight",
                    ".scale_input",
                    ".input_scale",
                )
            )
        }
        self.direct = {}

    def transfer_stream(self):
        # Match model_management.get_offload_stream: fence the previous stream
        # against all intervening consumer work before rotating to the next one.
        if self.copy_index is None:
            self.copy_index = 0
        else:
            self.streams[self.copy_index].wait_stream(
                torch.cuda.current_stream(self.device)
            )
            self.copy_index = (self.copy_index + 1) % len(self.streams)
        return self.streams[self.copy_index]

    def activate(self):
        self.vbar.prioritize()
        self.buffers = {
            stream: _module("vram_buffer").VRAMBuffer(
                self.buffer_capacity, self.device.index or 0
            )
            for stream in self.streams
        }
        self.patch_cache = (
            _module("host_buffer").HostBuffer(0, 8 * 1024**2, self.patch_size)
            if self.patch_size
            else None
        )
        self.largest_binding, self.largest_size = None, 0
        self.direct = {
            key: self.base.tensor(key).to(self.device) for key in self.direct_keys
        }

    def deactivate(self):
        torch.cuda.current_stream(self.device).synchronize()
        for stream in self.streams:
            stream.synchronize()
        self.direct.clear()
        # The reference executor resets cast buffers and adapter pins after each
        # sampler node. Preserve base host/VBAR state across the phase boundary.
        self.buffers.clear()
        for binding in self.bindings.values():
            for _, source in binding.patches:
                source.release_host()
        if self.patch_cache is not None:
            self.patch_cache.truncate(0, do_unregister=False)
            self.patch_cache = None
        self.vbar.deprioritize()

    @contextmanager
    def use(self, prefix):
        binding = self.bindings[prefix]
        complete = False
        try:
            values, updates = binding.materialize(self)
            self.weights._current_values = values
            self.weights._current_updates = updates
            self.weights._current_binding = binding
            yield
            complete = True
        finally:
            self.weights._current_values = {}
            self.weights._current_updates = {}
            self.weights._current_binding = None
            binding.release(self)
            if not complete:
                binding.signature = binding.patched = None

    def close(self):
        self.deactivate()
        for binding in self.bindings.values():
            binding.patched = None
            binding.allocation = None
            binding.signature = None
            for source in (binding.source, *(s for _, s in binding.patches)):
                source.release_host()
        self.bindings.clear()
        self.buffers.clear()
        self.streams.clear()
        self.vbar = None
        self.host_cache.truncate(0, do_unregister=False)
        self.host_cache = None
        self.weights = None
