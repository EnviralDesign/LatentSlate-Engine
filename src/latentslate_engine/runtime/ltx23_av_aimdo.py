"""Direct AIMDO residency for the LTX 2.3 audio/video transformer.

Adapted from ComfyUI v0.34.0 commit 12d5279438bfefc058a269eae805ceab6047777f:
``ModelPatcherDynamic``, ``comfy.ops``, ``model_prefetch``, and gathered-layout
helpers. Engine binds those invariants directly to the authenticated LTX artifact
and Diffusers forward: one VBAR, per-leaf signatures, four model HostBuffers,
stream-local fallback, operation-local binding/unpin, and block-queue prefetch.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from importlib import import_module
from typing import Any, Never

import torch
from torch import nn

from .ltx23_av_stored_adapter import (
    LTX23AVFileBackedValue,
    LTX23AVFileSpan,
    LTX23LeafStorage,
    LTX23ModuleBinding,
    authenticate_ltx23_av_open_handle,
    capture_ltx23_leaf_storages,
    inspect_ltx23_av_artifact,
)

_ALIGNMENT = 1024
_PIN_HEADROOM = 2 * 1024**3
_INIT_LOCK = threading.Lock()


class LTX23AVAimdoPoisoned(RuntimeError):
    """Native ownership could not be proven safe for Python finalization."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"LTX AV AIMDO context poisoned: {reason}")


def _align(size: int) -> int:
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True, slots=True)
class _Physical:
    tensor: torch.Tensor | None
    span: LTX23AVFileSpan | None
    offset: int
    size: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Layout:
    template: Any
    fields: tuple[str, ...] | None
    context: object | None
    physical: tuple[_Physical, ...]


@dataclass(frozen=True, slots=True)
class _PinRegion:
    subset: str
    offset: int
    size: int
    physical: tuple[_Physical, ...]


@dataclass(slots=True)
class _Leaf:
    storage: LTX23LeafStorage
    layouts: tuple[_Layout, ...]
    value_indices: tuple[int, ...]
    size: int
    companion_layouts: tuple[_Layout, ...] = ()
    companion_value_indices: tuple[int, ...] = ()
    companion_size: int = 0
    allocation: object | None = None
    signature: object | None = None
    cached_values: tuple[Any, ...] | None = None
    prefetched_values: tuple[Any, ...] | None = None
    prefetched_signature: object | None = None
    transfer_stream: Any | None = None
    temporary_raw: torch.Tensor | None = None
    binding: LTX23ModuleBinding | None = None
    companion_prefetched_values: tuple[Any, ...] | None = None
    companion_temporary_raw: torch.Tensor | None = None
    companion_binding: LTX23ModuleBinding | None = None
    users: int = 0
    block_scoped: bool = False
    pins: dict[tuple[str, int, int], torch.Tensor] = dataclass_field(
        default_factory=dict
    )
    companion_pins: dict[tuple[str, int, int], torch.Tensor] = dataclass_field(
        default_factory=dict
    )


def _layout(value: Any, next_offset: list[int], offsets: dict[tuple[Any, ...], int]) -> _Layout:
    descriptor = value if isinstance(value, LTX23AVFileBackedValue) else None
    template = descriptor.template if descriptor is not None else value
    flatten = getattr(template, "__tensor_flatten__", None)
    if callable(flatten):
        names, context = flatten()
        fields = tuple(names)
        tensors = tuple(getattr(template, name) for name in fields)
    else:
        fields, context, tensors = None, None, (template,)
    spans = None if descriptor is None else descriptor.spans
    if spans is not None and len(spans) != len(tensors):
        raise ValueError("LTX AV file span count differs from Kitchen physical layout")
    physical: list[_Physical] = []
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("LTX AV Kitchen layout contains a non-tensor field")
        span = None if spans is None else spans[index]
        if span is None:
            if tensor.is_meta or tensor.device.type != "cpu":
                raise ValueError("LTX AV direct source must be CPU or authenticated file-backed")
            size = tensor.numel() * tensor.element_size()
            key = ("cpu", tensor.untyped_storage().data_ptr(), tensor.storage_offset(), size)
            dtype, shape = tensor.dtype, tuple(tensor.shape)
        else:
            size = span.size
            key = ("file", span.source_id, span.offset, size)
            dtype, shape = span.dtype, span.shape
        offset = offsets.get(key)
        if offset is None:
            offset = next_offset[0]
            offsets[key] = offset
            next_offset[0] += _align(size)
        physical.append(_Physical(None if span else tensor, span, offset, size, dtype, shape))
    return _Layout(template, fields, context, tuple(physical))


def _storage_layout(
    storage: Any,
    sources: Mapping[int, Any],
) -> tuple[tuple[_Layout, ...], tuple[int, ...], int]:
    values = tuple(sources.get(id(slot.cpu_value), slot.cpu_value) for slot in storage.slots)
    unique: list[Any] = []
    positions: dict[int, int] = {}
    indices: list[int] = []
    for value in values:
        index = positions.setdefault(id(value), len(unique))
        if index == len(unique):
            unique.append(value)
        indices.append(index)
    offsets: dict[tuple[Any, ...], int] = {}
    next_offset = [0]
    layouts = tuple(_layout(value, next_offset, offsets) for value in unique)
    return layouts, tuple(indices), next_offset[0]


def _leaf(storage: LTX23LeafStorage, sources: Mapping[int, Any]) -> _Leaf:
    layouts, indices, size = _storage_layout(storage.storage, sources)
    if storage.companion_storage is None:
        return _Leaf(storage, layouts, indices, size)
    companion_layouts, companion_indices, companion_size = _storage_layout(
        storage.companion_storage, sources
    )
    return _Leaf(
        storage,
        layouts,
        indices,
        size,
        companion_layouts,
        companion_indices,
        companion_size,
    )


def _raw_values(
    leaf: _Leaf,
    raw: torch.Tensor,
    *,
    companion: bool = False,
) -> tuple[Any, ...]:
    layouts = leaf.companion_layouts if companion else leaf.layouts
    value_indices = (
        leaf.companion_value_indices if companion else leaf.value_indices
    )
    unique: list[Any] = []
    for layout in layouts:
        actuals: dict[str, torch.Tensor] = {}
        fields = layout.fields or ("data",)
        for field, item in zip(fields, layout.physical, strict=True):
            actuals[field] = raw[item.offset : item.offset + item.size].view(item.dtype).view(item.shape)
        if layout.fields is None:
            unique.append(actuals["data"])
        else:
            unique.append(
                type(layout.template).__tensor_unflatten__(actuals, layout.context, 0, 0)
            )
    return tuple(unique[index] for index in value_indices)


def _physical_items(
    leaf: _Leaf,
    *,
    companion: bool = False,
) -> tuple[_Physical, ...]:
    layouts = leaf.companion_layouts if companion else leaf.layouts
    return tuple(
        {item.offset: item for layout in layouts for item in layout.physical}.values()
    )


def _pin_regions(
    leaf: _Leaf,
    *,
    companion: bool = False,
) -> tuple[_PinRegion, ...]:
    physical = sorted(
        _physical_items(leaf, companion=companion), key=lambda item: item.offset
    )
    regions: list[_PinRegion] = []
    run: list[_Physical] = []
    run_subset: str | None = None
    for item in physical:
        subset = "weights" if item.span is not None else "patches"
        if run and subset != run_subset:
            start = run[0].offset
            end = max(member.offset + member.size for member in run)
            regions.append(_PinRegion(run_subset, start, end - start, tuple(run)))
            run = []
        run_subset = subset
        run.append(item)
    if run:
        start = run[0].offset
        end = max(member.offset + member.size for member in run)
        regions.append(_PinRegion(run_subset, start, end - start, tuple(run)))
    return tuple(regions)


class LTX23AVAimdoState:
    """The single model-local owner of the pinned Comfy AV state machine."""

    def __init__(self, transformer: nn.Module, device: torch.device) -> None:
        self.transformer = transformer
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # The Engine's default device spelling is ``cuda``.  AIMDO needs a
            # concrete logical ordinal, and resolving that ordinal must not
            # create a CUDA context before control.init().  The unqualified
            # CUDA device is logical device zero by definition for this
            # single-device runtime.
            self.device = torch.device("cuda", 0)
        self.blocks = tuple(transformer.transformer_blocks)
        if len(self.blocks) != 48:
            raise RuntimeError("LTX 2.3 transformer block topology changed")
        self._sources = dict(getattr(transformer, "_latentslate_ltx23_av_source_descriptors", {}))
        self._plan = getattr(transformer, "_latentslate_ltx23_av_source_plan", None)
        if self.device.type == "cuda" and (not self._sources or self._plan is None):
            raise RuntimeError("LTX CUDA transformer requires authenticated file-backed sources")
        captured = capture_ltx23_leaf_storages(transformer, source_values=self._sources)
        self._leaves = tuple(_leaf(item, self._sources) for item in captured)
        self._by_group = {
            group: tuple(leaf for leaf in self._leaves if group in leaf.storage.schedule_groups)
            for group in ("root", *(f"transformer_blocks.{i}" for i in range(48)))
        }
        self.base_stored_bytes = sum(
            item.storage.storage.physical_bytes for item in self._leaves
        )
        self.companion_stored_bytes = sum(
            0
            if item.storage.companion_storage is None
            else item.storage.companion_storage.physical_bytes
            for item in self._leaves
        )
        self.stored_bytes = self.base_stored_bytes + self.companion_stored_bytes
        self._handles: list[Any] = []
        self._owner_thread: int | None = None
        self._before_first: Callable[[], None] | None = None
        self._scope_started = False
        self._executing = False
        self._active_block: str | None = None
        self._closed = False
        self._poison_reason: str | None = None
        self._file = None
        self._vbar = None
        self._model_vbar = None
        self._aimdo_torch = None
        self._host_buffer = None
        self._hostbufs: dict[str, Any] = {}
        self._streams: tuple[Any, ...] = ()
        self._vrambufs: tuple[Any, ...] = ()
        self._patch_vrambufs: tuple[Any, ...] = ()
        self._stream_index = 0
        self._pins: list[tuple[torch.Tensor, bool]] = []
        self._faults = self._signature_hits = self._signature_misses = 0
        self._fault_none = self._h2d_bytes = self._source_read_bytes = 0
        self._source_read_calls = self._prefetch_calls = self._unpin_calls = 0
        self._forward_stream_waits = self._reverse_stream_waits = 0
        self._cleanup_calls = 0
        try:
            if self.device.type == "cuda":
                self._initialize_cuda()
            self._attach()
        except BaseException:
            self._close_after_failed_init()
            raise

    @property
    def handles(self) -> list[Any]:
        return self._handles

    @property
    def active(self) -> str | None:
        return self._active_block

    @property
    def policy(self) -> dict[str, Any]:
        dynamic = self._diagnostics()
        return {
            "mode": "comfy_direct_leaf_vbar",
            "stored_bytes": self.stored_bytes,
            "base_stored_bytes": self.base_stored_bytes,
            "companion_stored_bytes": self.companion_stored_bytes,
            "leaf_allocation_count": len(self._leaves),
            "force_resident_leaf_count": sum(item.storage.force_resident for item in self._leaves),
            "block_count": 48,
            "prefetch": True,
            "base_file_backed": self.device.type == "cuda",
            "base_file_handle_live": self._file is not None,
            "dynamic_vram": dynamic,
        }

    def failure_diagnostics(self) -> dict[str, Any]:
        return {"dynamic_vram": self._diagnostics()} if self._vbar is not None else {}

    def terminal_poison_reason(self) -> str | None:
        return self._poison_reason

    @contextmanager
    def forward_scope(self, before_first: Callable[[], None]):
        self._require_owner()
        if self._poison_reason is not None:
            raise LTX23AVAimdoPoisoned(self._poison_reason)
        if self._closed or self._before_first is not None:
            raise RuntimeError("LTX transformer residency forward scope is unavailable")
        self._before_first = before_first
        self._scope_started = False
        try:
            yield self
        finally:
            if self._poison_reason is None:
                self._release_all_operation_state()
            self._before_first = None
            self._scope_started = False
            self._executing = False
            self._active_block = None

    def close(self) -> None:
        if self._poison_reason is not None:
            raise LTX23AVAimdoPoisoned(self._poison_reason)
        if self._closed:
            return
        self._require_owner()
        if self._executing:
            raise RuntimeError("cannot close LTX transformer residency during a forward")
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as exc:
                self._poison_reason = "device_quiescence_failed"
                self.transformer._latentslate_ltx23_residency_poisoned = self._poison_reason
                raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
        if self.device.type == "cuda":
            cudart = torch.cuda.cudart()
            for index in reversed(range(len(self._pins))):
                pin, registered = self._pins[index]
                if registered:
                    try:
                        result = cudart.cudaHostUnregister(pin.data_ptr())
                    except BaseException as exc:
                        self._poison_reason = "retirement_cleanup_failed"
                        self.transformer._latentslate_ltx23_residency_poisoned = (
                            self._poison_reason
                        )
                        raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                    if result != 0:
                        self._poison_reason = "retirement_cleanup_failed"
                        self.transformer._latentslate_ltx23_residency_poisoned = (
                            self._poison_reason
                        )
                        try:
                            self._clear_cuda_error(cudart)
                        except BaseException as exc:
                            raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                        raise LTX23AVAimdoPoisoned(self._poison_reason)
                    self._pins[index] = (pin, False)
        self._release_all_operation_state()
        for leaf in self._leaves:
            if leaf.storage.force_resident and leaf.cached_values is not None:
                leaf.storage.storage.restore_cpu()
                leaf.cached_values = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for leaf in self._leaves:
            leaf.pins.clear()
            leaf.companion_pins.clear()
        self._pins.clear()
        if self.device.type == "cuda":
            self._vrambufs = ()
            self._patch_vrambufs = ()
            self._streams = ()
            self._hostbufs.clear()
            ptr = getattr(self._vbar, "_ptr", None)
            if ptr:
                self._model_vbar.lib.vbar_free(self._vbar._devctx, ptr)
                self._vbar._ptr = None
            self._vbar = None
            self._cleanup_calls += 1
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True

    def _initialize_cuda(self) -> None:
        if self.device.index is None:
            raise RuntimeError("LTX AIMDO requires an explicit CUDA ordinal")
        rebound = inspect_ltx23_av_artifact(
            self._plan.contract.path, expected_variant=self._plan.contract.variant
        )
        if rebound != self._plan.contract:
            raise RuntimeError("LTX AV source changed before AIMDO initialization")
        self._file = self._plan.contract.path.open("rb")
        authenticate_ltx23_av_open_handle(self._file, self._plan.contract)
        with torch.cuda.device(self.device), _INIT_LOCK:
            control = import_module("comfy_aimdo.control")
            if control.init(simple_vram_headroom=None, nvml_pressure=True) is not True:
                raise RuntimeError("comfy-aimdo initialization failed")
            try:
                control.get_devctx(self.device.index)
            except RuntimeError:
                if control.init_devices([(self.device.index, 0)]) is not True:
                    raise RuntimeError("comfy-aimdo device initialization failed")
            self._model_vbar = import_module("comfy_aimdo.model_vbar")
            self._aimdo_torch = import_module("comfy_aimdo.torch")
            self._host_buffer = import_module("comfy_aimdo.host_buffer")
            vram_buffer = import_module("comfy_aimdo.vram_buffer")
            self._streams = (
                torch.cuda.Stream(device=self.device),
                torch.cuda.Stream(device=self.device),
            )
            largest_block = max(
                (sum(_align(leaf.size) for leaf in leaves) for leaves in self._by_group.values()),
                default=0,
            )
            largest_patch_block = max(
                (
                    sum(
                        _align(leaf.companion_size)
                        for leaf in leaves
                        if leaf.companion_size
                    )
                    for leaves in self._by_group.values()
                ),
                default=0,
            )
            self._vrambufs = tuple(
                vram_buffer.VRAMBuffer(max(largest_block, 16 * 1024**2), self.device.index)
                for _ in self._streams
            )
            self._patch_vrambufs = tuple(
                vram_buffer.VRAMBuffer(
                    max(largest_patch_block, 8 * 1024**2), self.device.index
                )
                for _ in self._streams
            )
            max_host = max(self.stored_bytes, 64 * 1024**2)
            self._hostbufs = {
                "weights": self._host_buffer.HostBuffer(0, 64 * 1024**2, max_host),
                "patches": self._host_buffer.HostBuffer(0, 8 * 1024**2, max_host),
                "weights-loaded": self._host_buffer.HostBuffer(0, 64 * 1024**2, max_host),
                "patches-loaded": self._host_buffer.HostBuffer(0, 8 * 1024**2, max_host),
            }
            self._vbar = self._model_vbar.ModelVBAR(
                self.base_stored_bytes * 10, self.device.index
            )
            for leaf in self._leaves:
                if leaf.storage.force_resident:
                    raw = torch.empty((leaf.size,), dtype=torch.uint8, device=self.device)
                    self._fill(leaf, raw, None)
                    leaf.cached_values = _raw_values(leaf, raw)
                    LTX23ModuleBinding(
                        leaf.storage.storage, leaf.cached_values, self.device
                    ).activate()
                else:
                    leaf.allocation = self._vbar.alloc(leaf.size)
            self._vbar.prioritize()

    def _attach(self) -> None:
        self._handles.append(self.transformer.register_forward_pre_hook(self._root_pre))
        self._handles.append(
            self.transformer.register_forward_hook(self._root_post, always_call=True)
        )
        for index, block in enumerate(self.blocks):
            name = f"transformer_blocks.{index}"
            self._handles.append(block.register_forward_pre_hook(self._block_pre(name)))
            self._handles.append(block.register_forward_hook(self._block_post(name), always_call=True))
        for leaf in self._leaves:
            if leaf.storage.force_resident and leaf.storage.companion_storage is None:
                continue
            for module in leaf.storage.operation_modules:
                if module is self.transformer:
                    continue
                self._handles.append(module.register_forward_pre_hook(self._leaf_pre(leaf)))
                self._handles.append(module.register_forward_hook(self._leaf_post(leaf), always_call=True))

    def _root_pre(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._require_owner()
        if self._poison_reason is not None:
            raise LTX23AVAimdoPoisoned(self._poison_reason)
        if self._executing or self._before_first is None:
            raise RuntimeError("LTX transformer forward escaped its direct AIMDO scope")
        if not self._scope_started:
            self._scope_started = True
            if self.device.type == "cuda":
                # I2V conditioning may run the video VAE after scope entry.
                # Match Comfy's load-before-forward transition by making the
                # transformer VBAR current only when its first forward begins.
                self._vbar.prioritize()
            self._before_first()
        self._executing = True
        for leaf in self._by_group["root"]:
            if any(slot.module is self.transformer for slot in leaf.storage.storage.slots):
                self._enter_leaf(leaf)

    def _root_post(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self._poison_reason is not None:
            return output
        try:
            self._release_prefetched_group("transformer_blocks.47")
            for leaf in self._by_group["root"]:
                if any(slot.module is self.transformer for slot in leaf.storage.storage.slots):
                    self._leave_leaf(leaf)
        finally:
            self._executing = False
            self._active_block = None
        return output

    def _block_pre(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            if self._poison_reason is not None:
                raise LTX23AVAimdoPoisoned(self._poison_reason)
            if not self._executing:
                raise RuntimeError("LTX block forward escaped transformer scope")
            index = int(name.rsplit(".", 1)[1])
            if index:
                self._release_prefetched_group(f"transformer_blocks.{index - 1}")
            self._active_block = name
            self._prefetch_block(index)

        return hook

    def _block_post(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            try:
                if name == "transformer_blocks.47":
                    self._release_prefetched_group(name)
            finally:
                if self._active_block == name:
                    self._active_block = None
            return output

        return hook

    def _leaf_pre(self, leaf: _Leaf):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._enter_leaf(leaf)

        return hook

    def _leaf_post(self, leaf: _Leaf):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            self._leave_leaf(leaf)
            return output

        return hook

    def _prefetch_block(self, index: int) -> None:
        stream_index = index % max(1, len(self._streams))
        temporary_offset = 0
        companion_offset = 0
        staged = False
        for leaf in self._by_group[f"transformer_blocks.{index}"]:
            needs_base = not leaf.storage.force_resident and leaf.prefetched_values is None
            needs_companion = (
                leaf.storage.companion_storage is not None
                and leaf.companion_prefetched_values is None
            )
            if leaf.users == 0 and (needs_base or needs_companion):
                temporary_offset, companion_offset = self._prefetch(
                    leaf,
                    stream_index=stream_index,
                    temporary_offset=temporary_offset,
                    companion_offset=companion_offset,
                )
                if self.device.type == "cuda":
                    leaf.transfer_stream = self._streams[stream_index]
                leaf.block_scoped = True
                staged = True
        if staged and self.device.type == "cuda":
            stream = self._streams[stream_index]
            try:
                torch.cuda.current_stream(self.device).wait_stream(stream)
            except BaseException as exc:
                self._poison_reason = "device_quiescence_failed"
                self.transformer._latentslate_ltx23_residency_poisoned = (
                    self._poison_reason
                )
                raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
            self._forward_stream_waits += 1

    def _prefetch(
        self,
        leaf: _Leaf,
        *,
        stream_index: int | None = None,
        temporary_offset: int = 0,
        companion_offset: int = 0,
    ) -> tuple[int, int]:
        if self._poison_reason is not None:
            raise LTX23AVAimdoPoisoned(self._poison_reason)
        self._prefetch_calls += 1
        if self.device.type != "cuda":
            if not leaf.storage.force_resident:
                leaf.prefetched_values = tuple(
                    slot.cpu_value for slot in leaf.storage.storage.slots
                )
            if leaf.storage.companion_storage is not None:
                leaf.companion_prefetched_values = tuple(
                    slot.cpu_value for slot in leaf.storage.companion_storage.slots
                )
            return temporary_offset, companion_offset
        signature: object | None = None
        stream: Any | None = None
        try:
            if stream_index is None:
                stream_index = self._stream_index % len(self._streams)
                self._stream_index += 1
            base_raw: torch.Tensor | None = None
            base_cache_update = False
            if not leaf.storage.force_resident:
                signature = self._model_vbar.vbar_fault(leaf.allocation)
                self._faults += 1
                resident = self._model_vbar.vbar_signature_compare(
                    signature, leaf.signature
                )
                if resident:
                    if leaf.cached_values is None:
                        raise RuntimeError("LTX AV resident signature has no cached view")
                    self._signature_hits += 1
                    leaf.prefetched_values = leaf.cached_values
                    leaf.prefetched_signature = signature
                else:
                    self._signature_misses += 1
                    stream = self._streams[stream_index]
                    if signature is None:
                        self._fault_none += 1
                        allocation = self._vrambufs[stream_index].get(
                            leaf.size, temporary_offset
                        )
                        base_raw = self._aimdo_torch.aimdo_to_tensor(
                            allocation, self.device
                        )
                        leaf.temporary_raw = base_raw
                        temporary_offset += _align(leaf.size)
                    else:
                        base_raw = self._aimdo_torch.aimdo_to_tensor(
                            leaf.allocation, self.device
                        )
                        base_cache_update = True
                    leaf.prefetched_values = _raw_values(leaf, base_raw)
                    leaf.prefetched_signature = signature

            companion_raw: torch.Tensor | None = None
            if leaf.storage.companion_storage is not None:
                stream = self._streams[stream_index]
                allocation = self._patch_vrambufs[stream_index].get(
                    leaf.companion_size, companion_offset
                )
                companion_raw = self._aimdo_torch.aimdo_to_tensor(
                    allocation, self.device
                )
                leaf.companion_temporary_raw = companion_raw
                companion_offset += _align(leaf.companion_size)
                leaf.companion_prefetched_values = _raw_values(
                    leaf, companion_raw, companion=True
                )

            leaf.transfer_stream = stream
            if stream is not None:
                with torch.cuda.stream(stream):
                    if base_raw is not None:
                        self._fill(leaf, base_raw, stream)
                    if companion_raw is not None:
                        self._fill(leaf, companion_raw, stream, companion=True)
            if base_cache_update:
                leaf.signature = signature
                leaf.cached_values = leaf.prefetched_values
            return temporary_offset, companion_offset
        except BaseException as exc:  # noqa: BLE001 - native lease failures are terminal
            self._abort_failed_prefetch(leaf, signature, stream, exc)

    def _abort_failed_prefetch(
        self,
        leaf: _Leaf,
        signature: object | None,
        stream: Any | None,
        failure: BaseException,
    ) -> Never:
        cleanup_failure: BaseException | None = None
        if stream is not None:
            try:
                stream.synchronize()
            except BaseException as exc:  # noqa: BLE001 - cleanup must contain all failures
                cleanup_failure = exc
        if cleanup_failure is not None:
            reason = "retirement_cleanup_failed"
            self._poison_reason = reason
            self.transformer._latentslate_ltx23_residency_poisoned = reason
            try:
                cleanup_failure.add_note(
                    f"original prefetch failure: {type(failure).__name__}: {failure}"
                )
            except AttributeError:
                pass
            raise LTX23AVAimdoPoisoned(reason) from cleanup_failure
        if signature is not None:
            try:
                self._model_vbar.vbar_unpin(leaf.allocation)
                self._unpin_calls += 1
            except BaseException as exc:
                reason = "retirement_cleanup_failed"
                self._poison_reason = reason
                self.transformer._latentslate_ltx23_residency_poisoned = reason
                try:
                    exc.add_note(
                        f"original prefetch failure: {type(failure).__name__}: {failure}"
                    )
                except AttributeError:
                    pass
                raise LTX23AVAimdoPoisoned(reason) from exc
        leaf.prefetched_values = None
        leaf.prefetched_signature = None
        leaf.transfer_stream = None
        leaf.temporary_raw = None
        leaf.binding = None
        leaf.companion_prefetched_values = None
        leaf.companion_temporary_raw = None
        leaf.companion_binding = None
        leaf.users = 0
        leaf.signature = None
        leaf.cached_values = None
        reason = "stage_prepare_failed"
        self._poison_reason = reason
        self.transformer._latentslate_ltx23_residency_poisoned = reason
        raise LTX23AVAimdoPoisoned(reason) from failure

    def _enter_leaf(self, leaf: _Leaf) -> None:
        if self._poison_reason is not None:
            raise LTX23AVAimdoPoisoned(self._poison_reason)
        if leaf.users == 0:
            needs_base = not leaf.storage.force_resident and leaf.prefetched_values is None
            needs_companion = (
                leaf.storage.companion_storage is not None
                and leaf.companion_prefetched_values is None
            )
            if needs_base or needs_companion:
                self._prefetch(leaf)
            if leaf.transfer_stream is not None and not leaf.block_scoped:
                try:
                    torch.cuda.current_stream(self.device).wait_stream(
                        leaf.transfer_stream
                    )
                except BaseException as exc:
                    self._poison_reason = "device_quiescence_failed"
                    self.transformer._latentslate_ltx23_residency_poisoned = (
                        self._poison_reason
                    )
                    raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                self._forward_stream_waits += 1
            try:
                if not leaf.storage.force_resident:
                    binding = LTX23ModuleBinding(
                        leaf.storage.storage, leaf.prefetched_values, self.device
                    )
                    binding.activate()
                    leaf.binding = binding
                if leaf.storage.companion_storage is not None:
                    companion_binding = LTX23ModuleBinding(
                        leaf.storage.companion_storage,
                        leaf.companion_prefetched_values,
                        self.device,
                    )
                    companion_binding.activate()
                    leaf.companion_binding = companion_binding
            except BaseException as exc:
                if leaf.companion_binding is not None:
                    leaf.companion_binding.restore_cpu()
                    leaf.companion_binding = None
                if leaf.binding is not None:
                    leaf.binding.restore_cpu()
                    leaf.binding = None
                if self.device.type == "cuda":
                    self._abort_failed_prefetch(
                        leaf, leaf.prefetched_signature, leaf.transfer_stream, exc
                    )
                raise
        leaf.users += 1

    def _leave_leaf(self, leaf: _Leaf) -> None:
        if self._poison_reason is not None:
            return
        if leaf.users <= 0:
            return
        leaf.users -= 1
        if leaf.users:
            return
        if leaf.companion_binding is not None:
            leaf.companion_binding.restore_cpu()
            leaf.companion_binding = None
        if leaf.binding is not None:
            leaf.binding.restore_cpu()
            leaf.binding = None
        if leaf.block_scoped:
            return
        if self.device.type == "cuda":
            stream = leaf.transfer_stream
            if stream is not None:
                try:
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                except BaseException as exc:
                    self._poison_reason = "device_quiescence_failed"
                    self.transformer._latentslate_ltx23_residency_poisoned = (
                        self._poison_reason
                    )
                    raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                self._reverse_stream_waits += 1
            if leaf.prefetched_signature is not None:
                try:
                    self._model_vbar.vbar_unpin(leaf.allocation)
                except BaseException as exc:
                    self._poison_reason = "retirement_release_failed"
                    self.transformer._latentslate_ltx23_residency_poisoned = (
                        self._poison_reason
                    )
                    raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                self._unpin_calls += 1
        leaf.prefetched_values = None
        leaf.prefetched_signature = None
        leaf.transfer_stream = None
        leaf.temporary_raw = None
        leaf.companion_prefetched_values = None
        leaf.companion_temporary_raw = None

    def _fill(
        self,
        leaf: _Leaf,
        raw: torch.Tensor,
        stream: Any | None,
        *,
        companion: bool = False,
    ) -> None:
        pins = leaf.companion_pins if companion else leaf.pins
        for region in _pin_regions(leaf, companion=companion):
            pin_key = (region.subset, region.offset, region.size)
            pin = pins.get(pin_key)
            if pin is None and self._can_pin(region.size):
                pin = self._make_pin(region)
            if pin is not None:
                device_region = raw[region.offset : region.offset + region.size]
                if pin_key not in pins:
                    self._h2d_bytes += self._fill_pin(
                        region, pin, raw, stream
                    )
                    pins[pin_key] = pin
                else:
                    device_region.copy_(pin, non_blocking=stream is not None)
                    self._h2d_bytes += region.size
                continue
            for item in region.physical:
                destination = raw[item.offset : item.offset + item.size]
                if item.span is not None:
                    self._host_buffer.read_file_to_device(
                        self._file,
                        item.span.offset,
                        item.size,
                        0 if stream is None else stream.cuda_stream,
                        destination.data_ptr(),
                        self.device.index,
                        mark_cold=False,
                    )
                    self._source_read_calls += 1
                    self._source_read_bytes += item.size
                else:
                    destination.view(item.dtype).view(item.shape).copy_(
                        item.tensor, non_blocking=stream is not None
                    )
                self._h2d_bytes += item.size

    def _make_pin(self, region: _PinRegion) -> torch.Tensor | None:
        hostbuf = self._hostbufs[region.subset]
        offset = hostbuf.size
        pin: torch.Tensor | None = None
        extended = False
        try:
            hostbuf.extend(region.size, register=False)
            extended = True
            pin = self._aimdo_torch.hostbuf_to_tensor(hostbuf)[
                offset : offset + region.size
            ]
            pin.untyped_storage()._comfy_hostbuf = hostbuf
        except RuntimeError:
            pin = None
            if extended and hostbuf.size > offset:
                hostbuf.truncate(offset, do_unregister=False)
            return None
        cudart = torch.cuda.cudart()
        if cudart.cudaHostRegister(pin.data_ptr(), region.size, 1) != 0:
            self._clear_cuda_error(cudart)
            pin = None
            hostbuf.truncate(offset, do_unregister=False)
            return None
        self._pins.append((pin, True))
        return pin

    @staticmethod
    def _clear_cuda_error(cudart: Any) -> None:
        get_last_error = getattr(cudart, "cudaGetLastError", None)
        if callable(get_last_error):
            try:
                get_last_error()
            except (RuntimeError, TypeError):
                pass

    def _fill_pin(
        self,
        region: _PinRegion,
        pin: torch.Tensor,
        raw: torch.Tensor,
        stream: Any | None,
    ) -> int:
        transferred = 0
        for item in region.physical:
            destination = pin[
                item.offset - region.offset : item.offset - region.offset + item.size
            ]
            device_destination = raw[item.offset : item.offset + item.size]
            if item.span is not None:
                hostbuf = self._hostbufs[region.subset]
                hostbuf.read_file_slice(
                    self._file,
                    item.span.offset,
                    item.size,
                    offset=destination.data_ptr() - hostbuf.get_raw_address(),
                    stream=0 if stream is None else stream.cuda_stream,
                    device_ptr=device_destination.data_ptr(),
                    device=self.device.index,
                )
                self._source_read_calls += 1
                self._source_read_bytes += item.size
            else:
                destination.view(item.dtype).view(item.shape).copy_(item.tensor)
                device_destination.copy_(destination, non_blocking=stream is not None)
            transferred += item.size
        return transferred

    def _can_pin(self, size: int) -> bool:
        try:
            import psutil

            return int(psutil.virtual_memory().available) >= size + _PIN_HEADROOM
        except (ImportError, OSError):
            return True

    def _release_all_operation_state(self) -> None:
        if self._poison_reason is not None:
            return
        for leaf in self._leaves:
            while leaf.users:
                self._leave_leaf(leaf)
        for index in range(48):
            name = f"transformer_blocks.{index}"
            if any(leaf.block_scoped for leaf in self._by_group[name]):
                self._release_prefetched_group(name)
        for leaf in self._leaves:
            if (
                leaf.prefetched_values is not None
                or leaf.companion_prefetched_values is not None
            ):
                leaf.users = 1
                self._leave_leaf(leaf)

    def _release_prefetched_group(self, name: str) -> None:
        if self._poison_reason is not None:
            return
        leaves = tuple(
            leaf
            for leaf in self._by_group[name]
            if leaf.block_scoped
            or leaf.prefetched_values is not None
            or leaf.companion_prefetched_values is not None
        )
        for leaf in leaves:
            if leaf.users:
                raise RuntimeError(f"LTX AV block retained active leaf state: {name}")
        if not leaves:
            return
        if self.device.type == "cuda":
            streams = tuple(
                dict.fromkeys(
                    leaf.transfer_stream
                    for leaf in leaves
                    if leaf.transfer_stream is not None
                )
            )
            if len(streams) > 1:
                raise RuntimeError(f"LTX AV block used multiple transfer streams: {name}")
            if streams:
                try:
                    streams[0].wait_stream(torch.cuda.current_stream(self.device))
                except BaseException as exc:
                    self._poison_reason = "device_quiescence_failed"
                    self.transformer._latentslate_ltx23_residency_poisoned = (
                        self._poison_reason
                    )
                    raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
                self._reverse_stream_waits += 1
            try:
                for leaf in leaves:
                    if leaf.prefetched_signature is not None:
                        self._model_vbar.vbar_unpin(leaf.allocation)
                        self._unpin_calls += 1
            except BaseException as exc:
                self._poison_reason = "retirement_release_failed"
                self.transformer._latentslate_ltx23_residency_poisoned = (
                    self._poison_reason
                )
                raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
        for leaf in leaves:
            leaf.prefetched_values = None
            leaf.prefetched_signature = None
            leaf.transfer_stream = None
            leaf.temporary_raw = None
            leaf.companion_prefetched_values = None
            leaf.companion_temporary_raw = None
            leaf.block_scoped = False

    def _diagnostics(self) -> dict[str, Any]:
        loaded = 0 if self._vbar is None else int(self._vbar.loaded_size())
        return {
            "backend": "comfy-aimdo",
            "version": "0.4.15",
            "mode": "ltx23_av_direct",
            "allocation_count": sum(not leaf.storage.force_resident for leaf in self._leaves),
            "loaded_bytes": loaded,
            "faults": self._faults,
            "signature_hits": self._signature_hits,
            "signature_misses": self._signature_misses,
            "fault_none_temporaries": self._fault_none,
            "gathered_h2d_bytes": self._h2d_bytes,
            "base_file_read_calls": self._source_read_calls,
            "base_file_read_bytes": self._source_read_bytes,
            "prefetch_calls": self._prefetch_calls,
            "unpin_calls": self._unpin_calls,
            "cleanup_calls": self._cleanup_calls,
            "forward_stream_waits": self._forward_stream_waits,
            "reverse_stream_waits": self._reverse_stream_waits,
            "poison_reason": self._poison_reason,
        }

    def _require_owner(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("LTX transformer residency crossed execution threads")

    def _close_after_failed_init(self) -> None:
        try:
            self.close()
        except LTX23AVAimdoPoisoned:
            raise
        except BaseException as exc:
            self._poison_reason = "ltx23_av_dynamic_initialization_cleanup_failed"
            self.transformer._latentslate_ltx23_residency_poisoned = self._poison_reason
            raise LTX23AVAimdoPoisoned(self._poison_reason) from exc
