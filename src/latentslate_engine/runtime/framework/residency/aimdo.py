"""Lazy standalone comfy-aimdo 0.4.15 DynamicVRAM adapter.

This module has no import-time dependency on AIMDO. The backend is initialized
only by a GPU child that explicitly constructs :class:`AimdoDynamicResidency`.
Frontend orchestration, VRAMBuffer, and allocator plugins are intentionally
outside this boundary. Optional gathered HostBuffer
source lanes retain immutable file-backed slices, including fault(None), and
own mutable patch temporaries until asynchronous transfers are proven quiescent.
Physical-RAM pressure may use AIMDO's narrow file-to-device reader rather than
mutating another HostBuffer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from typing import Any

import torch

from .dynamic import (
    DynamicResidencyDeviceError,
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
    DynamicResidencyUnavailable,
)
from .host_registration import (
    BestEffortHostRegistrationLedger,
    available_physical_memory_bytes,
    default_host_registration_budget_bytes,
)
from .host_source_pool import (
    AimdoHostSourcePool,
    HostSourceClass,
    HostSourceDirectTransferRequired,
    HostSourceLease,
    HostSourceLifetime,
    HostSourcePoolSetupError,
    HostSourcePoolSetupFallback,
    HostSourcePoolStructuralError,
    HostSourceWarmUnavailable,
)

_HOST_SOURCE_POOL_POISON_REASON = "host_source_pool_structural_failure"
_HOST_SOURCE_POOL_SETUP_POISON_REASON = "host_source_pool_setup_cleanup_failed"

_SUBALIGNMENT = 1024
_AIMDO_INIT_LOCK = threading.Lock()
_AIMDO_INITIALIZED_DEVICES: set[int] = set()


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def canonical_aimdo_cuda_device(device: torch.device | str) -> torch.device:
    """Resolve one exact CUDA identity before AIMDO import or allocation."""

    candidate = torch.device(device)
    if candidate.type != "cuda":
        raise DynamicResidencyUnavailable("AIMDO requires a CUDA device")
    if not torch.cuda.is_available():
        raise DynamicResidencyUnavailable("AIMDO requires available CUDA")
    try:
        device_count = int(torch.cuda.device_count())
        index = int(torch.cuda.current_device()) if candidate.index is None else candidate.index
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DynamicResidencyUnavailable(
            f"AIMDO could not resolve the active CUDA device: {exc}"
        ) from exc
    if index < 0 or index >= device_count:
        raise DynamicResidencyDeviceError(
            f"AIMDO CUDA index {index} is outside the available 0..{device_count - 1} range"
        )
    canonical = torch.device("cuda", index)
    try:
        with torch.cuda.device(canonical):
            selected = int(torch.cuda.current_device())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DynamicResidencyDeviceError(
            f"AIMDO could not establish CUDA device context {canonical}: {exc}"
        ) from exc
    if selected != index:
        raise DynamicResidencyDeviceError(
            f"AIMDO CUDA context selected index {selected}, expected {index}"
        )
    return canonical


@dataclass(frozen=True, slots=True)
class _PhysicalSource:
    tensor: torch.Tensor | None
    file_span: AimdoFileSpan | None
    offset: int
    size: int

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype if self.tensor is not None else self.file_span.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape) if self.tensor is not None else self.file_span.shape


@dataclass(frozen=True, slots=True)
class AimdoFileSpan:
    """One authenticated absolute file extent for a physical tensor."""

    source_id: str
    key: str
    offset: int
    size: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AimdoFileBackedValue:
    """Logical meta template whose flattened fields come from file spans."""

    template: Any
    spans: tuple[AimdoFileSpan, ...]


@dataclass(frozen=True, slots=True)
class _LogicalLayout:
    source: Any
    physical: tuple[_PhysicalSource, ...]
    names: tuple[str, ...] | None
    context: object | None
    total_bytes: int


@dataclass(slots=True)
class _Group:
    allocation: object
    layouts: tuple[_LogicalLayout, ...]
    value_indices: tuple[int, ...]
    staged_bytes: int
    signature: object | None = None
    resident_values: tuple[Any, ...] | None = None


@dataclass(slots=True)
class _LeaseToken:
    group: _Group
    temporary: torch.Tensor | None
    raw: torch.Tensor | None = None
    synchronized: bool = False
    waited: bool = False
    pending_events: list[Any] = field(default_factory=list)
    transfer_streams: list[Any] = field(default_factory=list)
    source_lease: HostSourceLease | None = None


@dataclass(slots=True)
class _RetirementBatch:
    """Ownership retained until a stream-ordered group is host-observable complete."""

    tokens: tuple[_LeaseToken, ...]
    completion_events: tuple[Any, ...]
    unpin_on_complete: bool = False


def _flatten_value(value: Any) -> _LogicalLayout:
    file_spans: tuple[AimdoFileSpan, ...] | None = None
    if isinstance(value, AimdoFileBackedValue):
        file_spans = value.spans
        value = value.template
    flatten = getattr(value, "__tensor_flatten__", None)
    if callable(flatten):
        names, context = flatten()
        if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
            raise TypeError("quantized tensor flatten contract returned invalid field names")
        tensors = tuple(getattr(value, name) for name in names)
        if not all(isinstance(item, torch.Tensor) for item in tensors):
            raise TypeError("quantized tensor flatten contract returned a non-tensor field")
        field_names: tuple[str, ...] | None = tuple(names)
    elif isinstance(value, torch.Tensor):
        tensors = (value,)
        context = None
        field_names = None
    else:
        raise TypeError(f"dynamic residency cannot flatten {type(value).__name__}")

    offset = 0
    physical: list[_PhysicalSource] = []
    if file_spans is not None and len(file_spans) != len(tensors):
        raise ValueError("dynamic residency file span count differs from flattened template")
    for index, tensor in enumerate(tensors):
        span = None if file_spans is None else file_spans[index]
        if span is None and (
            tensor.device.type != "cpu" or tensor.is_meta or not tensor.is_contiguous()
        ):
            raise ValueError("dynamic residency requires contiguous authoritative CPU tensors")
        if span is not None and (
            not tensor.is_meta
            or tensor.dtype is not span.dtype
            or tuple(tensor.shape) != span.shape
            or span.offset < 0
            or span.size <= 0
        ):
            raise ValueError("dynamic residency file span differs from its meta template")
        size = span.size if span is not None else tensor.numel() * tensor.element_size()
        if size <= 0:
            raise ValueError("dynamic residency does not accept empty physical tensors")
        physical.append(_PhysicalSource(None if span is not None else tensor, span, offset, size))
        offset += _round_up(size, _SUBALIGNMENT)
    return _LogicalLayout(value, tuple(physical), field_names, context, offset)


def _rebuild_value(layout: _LogicalLayout, raw: torch.Tensor) -> Any:
    rebuilt: dict[str, torch.Tensor] = {}
    values: list[torch.Tensor] = []
    for index, physical in enumerate(layout.physical):
        view = raw[physical.offset : physical.offset + physical.size]
        typed = view.view(dtype=physical.dtype).view(physical.shape)
        values.append(typed)
        if layout.names is not None:
            rebuilt[layout.names[index]] = typed
    if layout.names is None:
        result: Any = values[0]
    else:
        unflatten = getattr(type(layout.source), "__tensor_unflatten__", None)
        if not callable(unflatten):
            raise TypeError("quantized tensor type has no unflatten contract")
        result = unflatten(rebuilt, layout.context, 0, 0)
    if isinstance(layout.source, torch.nn.Parameter):
        if layout.names is None:
            # Ordinary tensors use PyTorch's normal Parameter construction.
            result = torch.nn.Parameter(result, requires_grad=layout.source.requires_grad)
        else:
            # Custom tensor subclasses implement Parameter identity through
            # ``_is_param``. Calling ``Parameter(result)`` invokes detach on
            # Kitchen QuantizedTensor and can clone its scale sidecars before
            # the VBAR destination has been filled. Mark the already-rebuilt
            # object in-place so qdata and every sidecar remain raw VBAR views.
            result._is_param = True
            if result.requires_grad != layout.source.requires_grad:
                result.requires_grad_(layout.source.requires_grad)
    return result


def _mark_signature_cacheable(values: tuple[Any, ...], *, cacheable: bool) -> None:
    """Expose whether reconstructed values belong to a retained VBAR signature.

    Model-owned execution may update a retained VBAR value in place, but must
    never mistake a signature-none temporary for resident storage merely
    because its Python object remains bound for more than one operation.
    """

    for value in values:
        value._latentslate_aimdo_signature_cacheable = cacheable


class AimdoDynamicResidency:
    """One VBAR with operation-faulted groups and exact transfer lifetimes."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        virtual_bytes: int,
        diagnostic: Callable[[str, Mapping[str, object]], None] | None = None,
        gathered_host_transfer: bool = True,
    ) -> None:
        self.device = canonical_aimdo_cuda_device(device)
        if virtual_bytes <= 0:
            raise ValueError("AIMDO virtual byte capacity must be positive")
        self._allocation_started = False
        self._closed = False
        self._poisoned = False
        self._close_failed = False
        self._poison_reason: str | None = None
        self._groups: dict[object, _Group] = {}
        self._streams: tuple[Any, Any] = ()
        self._copy_stream_count = 0
        self._gathered_stream_index = 0
        self._active: dict[int, _LeaseToken] = {}
        self._retirements: list[_RetirementBatch] = []
        self._faulted: set[int] = set()
        self._physical_bytes = 0
        self._staged_bytes = 0
        self._faults = 0
        self._signature_hits = 0
        self._signature_misses = 0
        self._fault_none_temporaries = 0
        self._pinned_copy_bytes = 0
        self._pageable_copy_bytes = 0
        self._transfer_events = 0
        self._transfer_waits = 0
        self._reverse_stream_waits = 0
        self._retirement_batches = 0
        self._retirement_polls = 0
        self._retirement_completions = 0
        self._stage_prepare_calls = 0
        self._stage_prepare_requested_bytes = 0
        self._stage_prepare_pending_before = 0
        self._stage_prepare_pending_after = 0
        self._stage_prepare_loaded_before = 0
        self._stage_prepare_loaded_after = 0
        self._stage_prepare_trim_requested = 0
        self._stage_prepare_trim_freed = 0
        self._stage_prepare_cuda_allocated_before = 0
        self._stage_prepare_cuda_reserved_before = 0
        self._stage_prepare_cuda_free_before = 0
        self._stage_prepare_cuda_allocated_after = 0
        self._stage_prepare_cuda_reserved_after = 0
        self._stage_prepare_cuda_free_after = 0
        self._prioritize_calls = 0
        self._unpin_calls = 0
        self._free_calls = 0
        self._allocation_count = 0
        self._diagnostic = diagnostic
        self._first_acquire_pending = True
        self._dirty_epoch = 0
        self._lora_invalidations = 0
        self._base_restores = 0
        self._gathered_host_buffer_requested = gathered_host_transfer
        self._copy_strategy = "per_physical"
        self._copy_fallback_reason: str | None = None
        self._host_buffer_module: Any | None = None
        self._host_source_pool: AimdoHostSourcePool | None = None
        self._host_source_pool_diagnostics: dict[str, int | bool] | None = None
        self._host_buffer: Any | None = None
        self._host_tensor: torch.Tensor | None = None
        self._host_buffer_pending_event: Any | None = None
        self._host_buffer_capacity_bytes = 0
        self._host_buffer_allocations = 0
        self._host_buffer_unregistrations = 0
        self._host_buffer_frees = 0
        self._gathered_misses = 0
        self._per_physical_misses = 0
        self._packed_source_bytes = 0
        self._gathered_h2d_bytes = 0
        self._host_buffer_reuse_barriers = 0
        self._pressure_direct_transfers = 0
        self._pressure_direct_bytes = 0
        self._prefetch_calls = 0
        self._host_registration_budget_bytes = default_host_registration_budget_bytes()
        self._ledger = BestEffortHostRegistrationLedger(
            self._host_registration_budget_bytes
        )
        self._file_sources: dict[str, Any] = {}
        self._base_file_backed = False
        self._base_file_read_calls = 0
        self._base_file_read_bytes = 0

        # Import order is contractual: model_vbar and comfy_aimdo.torch capture
        # control.lib at module import time.
        try:
            with torch.cuda.device(self.device):
                with _AIMDO_INIT_LOCK:
                    control = _import_module("comfy_aimdo.control")
                    if (
                        control.init(
                            simple_vram_headroom=None,
                            nvml_pressure=True,
                        )
                        is not True
                    ):
                        raise DynamicResidencyUnavailable(
                            "comfy-aimdo control.init() returned false"
                        )
                    get_devctx = getattr(control, "get_devctx", None)
                    device_initialized = self.device.index in _AIMDO_INITIALIZED_DEVICES
                    if callable(get_devctx):
                        try:
                            get_devctx(self.device.index)
                            device_initialized = True
                        except RuntimeError:
                            device_initialized = False
                    if not device_initialized:
                        if control.init_devices([(self.device.index, 0)]) is not True:
                            raise DynamicResidencyUnavailable(
                                "comfy-aimdo control.init_devices() returned false"
                            )
                    _AIMDO_INITIALIZED_DEVICES.add(self.device.index)
                    model_vbar = _import_module("comfy_aimdo.model_vbar")
                    aimdo_torch = _import_module("comfy_aimdo.torch")
                    if gathered_host_transfer:
                        try:
                            host_buffer = _import_module("comfy_aimdo.host_buffer")
                            if not callable(
                                getattr(host_buffer, "HostBuffer", None)
                            ) or not callable(getattr(aimdo_torch, "hostbuf_to_tensor", None)):
                                raise TypeError("comfy-aimdo HostBuffer tensor API is unavailable")
                        except BaseException as exc:  # noqa: BLE001 - explicit fallback
                            host_buffer = None
                            self._copy_fallback_reason = (
                                f"host_buffer_capability_unavailable: {type(exc).__name__}: {exc}"
                            )[:512]
                    else:
                        host_buffer = None
                    cudart = torch.cuda.cudart() if host_buffer is not None else None
                streams = (
                    torch.cuda.Stream(device=self.device),
                    torch.cuda.Stream(device=self.device),
                )
        except DynamicResidencyUnavailable:
            raise
        except BaseException as exc:
            raise DynamicResidencyUnavailable(
                f"comfy-aimdo initialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._control = control
        self._model_vbar_module = model_vbar
        self._aimdo_torch = aimdo_torch
        self._host_buffer_module = host_buffer
        self._cudart = cudart
        self._streams = streams
        self._copy_stream_count = len(streams)
        try:
            with torch.cuda.device(self.device):
                self._vbar = model_vbar.ModelVBAR(virtual_bytes, self.device.index)
                self._allocation_started = True
        except BaseException:
            self._streams = ()
            raise
        self._virtual_bytes = virtual_bytes

    @property
    def allocation_started(self) -> bool:
        return self._allocation_started

    @property
    def required_virtual_bytes(self) -> int:
        return self._virtual_bytes

    @staticmethod
    def group_bytes(values: tuple[Any, ...]) -> int:
        layouts = tuple(_flatten_value(value) for value in values)
        physical = {
            _physical_identity(item): item.size for layout in layouts for item in layout.physical
        }
        return sum(_round_up(size, _SUBALIGNMENT) for size in physical.values())

    def allocate_group(self, key: object, values: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("AIMDO residency is closed")
        if key in self._groups:
            raise ValueError("AIMDO residency group key is duplicated")
        unique_values: list[Any] = []
        value_positions: dict[int, int] = {}
        value_indices: list[int] = []
        for value in values:
            identity = id(value)
            if identity not in value_positions:
                value_positions[identity] = len(unique_values)
                unique_values.append(value)
            value_indices.append(value_positions[identity])
        layouts = tuple(_flatten_value(value) for value in unique_values)
        physical_offsets: dict[tuple[int, int], int] = {}
        next_offset = 0
        shifted: list[_LogicalLayout] = []
        for layout in layouts:
            physical: list[_PhysicalSource] = []
            for item in layout.physical:
                physical_key = _physical_identity(item)
                offset = physical_offsets.get(physical_key)
                if offset is None:
                    offset = next_offset
                    physical_offsets[physical_key] = offset
                    next_offset += _round_up(item.size, _SUBALIGNMENT)
                    self._physical_bytes += item.size
                physical.append(_PhysicalSource(item.tensor, item.file_span, offset, item.size))
            shifted.append(
                _LogicalLayout(
                    layout.source,
                    tuple(physical),
                    layout.names,
                    layout.context,
                    layout.total_bytes,
                )
            )
        size = next_offset
        projected_staged_bytes = self._staged_bytes + size
        if projected_staged_bytes > self._virtual_bytes:
            raise RuntimeError(
                "AIMDO cumulative group allocation exceeds requested VBAR capacity: "
                f"projected={projected_staged_bytes}, virtual={self._virtual_bytes}"
            )
        allocation = self._vbar.alloc(size)
        self._groups[key] = _Group(allocation, tuple(shifted), tuple(value_indices), size)
        self._allocation_count += 1
        self._staged_bytes += size

    def prioritize(self) -> None:
        if self._host_buffer_module is not None:
            self._initialize_gathered_host_buffer()
        self._vbar.prioritize()
        self._prioritize_calls += 1

    def bind_file_source(self, source_id: str, handle: Any) -> None:
        if self._closed or self._active or not source_id or source_id in self._file_sources:
            raise RuntimeError("AIMDO file source binding is invalid")
        if self._copy_strategy != "gathered_host_buffer":
            raise RuntimeError("AIMDO file source requires gathered HostBuffer transfer")
        if not hasattr(handle, "read") or not hasattr(handle, "fileno"):
            raise TypeError("AIMDO file source must be an open binary handle")
        self._file_sources[source_id] = handle
        self._base_file_backed = True

    def _initialize_gathered_host_buffer(self) -> None:
        """Reserve fixed-address source lanes before any Torch view is retained."""

        if self._host_buffer is not None or self._copy_strategy == "gathered_host_buffer":
            raise RuntimeError("AIMDO gathered HostBuffer was initialized twice")
        if not self._groups:
            raise RuntimeError("AIMDO cannot size HostBuffer before group allocation")
        base_bytes = sum(
            group.staged_bytes
            for group in self._groups.values()
            if _group_source_class(group) is HostSourceClass.BASE
        )
        patch_bytes = sum(
            group.staged_bytes
            for group in self._groups.values()
            if _group_source_class(group) is HostSourceClass.PATCH
        )
        capacities = {
            (HostSourceClass.BASE, HostSourceLifetime.WARM): base_bytes,
            (HostSourceClass.BASE, HostSourceLifetime.PREFETCH_TEMPORARY): base_bytes,
            # Wholly CPU adapter leaves are immutable Parameters for one
            # runtime identity. Patch invalidation purges this lane before a
            # changed epoch can reuse its bytes; mixed leaves remain temporary.
            (HostSourceClass.PATCH, HostSourceLifetime.WARM): patch_bytes,
            (HostSourceClass.PATCH, HostSourceLifetime.PREFETCH_TEMPORARY): patch_bytes,
        }
        try:
            pool = AimdoHostSourcePool(
                capacities,
                host_buffer_factory=self._host_buffer_module.HostBuffer,
                hostbuf_to_tensor=self._aimdo_torch.hostbuf_to_tensor,
                registration_budget_bytes=self._host_registration_budget_bytes,
                temporary_reserve_bytes=max(
                    (group.staged_bytes for group in self._groups.values()),
                    default=0,
                ),
                available_memory_bytes=available_physical_memory_bytes,
                cudart=self._cudart,
            )
            self._host_source_pool = pool
            self._host_buffer_allocations = pool.allocations
            self._host_buffer_capacity_bytes = pool.max_lane_capacity_bytes
            self._host_buffer = pool.owners[0] if pool.owners else None
        except HostSourcePoolSetupError as primary:
            # The exception owns the partially constructed pool so its native
            # HostBuffers cannot disappear into Python finalization. Preserve
            # that graph until the poisoned GPU child hard-exits.
            pool = primary.pool
            self._host_source_pool = pool
            self._host_buffer_allocations = pool.allocations
            self._host_buffer_unregistrations = pool.unregistrations
            self._host_buffer_frees = pool.frees
            self._host_buffer_capacity_bytes = pool.max_lane_capacity_bytes
            self._host_buffer = pool.owners[0] if pool.owners else None
            self._mark_poisoned(_HOST_SOURCE_POOL_SETUP_POISON_REASON)
            raise DynamicResidencyPoisoned(
                _HOST_SOURCE_POOL_SETUP_POISON_REASON
            ) from primary
        except HostSourcePoolSetupFallback as safe_failure:
            pool = safe_failure.pool
            self._host_source_pool_diagnostics = pool.diagnostics()
            self._host_buffer_allocations = pool.allocations
            self._host_buffer_unregistrations = pool.unregistrations
            self._host_buffer_frees = pool.frees
            self._host_buffer_capacity_bytes = pool.max_lane_capacity_bytes
            primary = safe_failure.primary
            self._copy_fallback_reason = (
                f"host_buffer_setup_failed: {type(primary).__name__}: {primary}"
            )[:512]
            self._host_buffer_module = None
            return
        except BaseException as primary:  # noqa: BLE001 - bounded fallback before acquire
            if self._host_source_pool is not None:
                try:
                    self._dispose_host_buffer()
                except BaseException as cleanup_error:  # noqa: BLE001
                    primary.add_note(
                        "AIMDO HostBuffer setup cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    raise RuntimeError("AIMDO HostBuffer setup cleanup was incomplete") from primary
            self._copy_fallback_reason = (
                f"host_buffer_setup_failed: {type(primary).__name__}: {primary}"
            )[:512]
            self._host_buffer_module = None
            return
        self._copy_strategy = "gathered_host_buffer"
        self._copy_fallback_reason = None

    def _dispose_host_buffer(self) -> None:
        """Drop all source views, unregister every lane, then free its owner."""

        pool = self._host_source_pool
        if pool is None:
            return
        # Legacy single-buffer code exposed this alias. The pooled path never
        # publishes one, but clear it before pool teardown defensively so no
        # backend reference can outlive native HostBuffer registration.
        self._host_tensor = None
        try:
            pool.close(quiesced=True)
        finally:
            # Diagnostics are purely Python counters and remain safe even
            # when a later lane's native teardown fails terminally.
            self._host_source_pool_diagnostics = pool.diagnostics()
            self._host_buffer_unregistrations = pool.unregistrations
            self._host_buffer_frees = pool.frees
        self._host_buffer = None
        self._host_source_pool = None

    def acquire(self, key: object) -> DynamicResidencyLease:
        lease = self._acquire(key, prefetched=False)
        self.wait(lease)
        return lease

    def prefetch(self, key: object) -> DynamicResidencyLease:
        """Fault/fill one group without placing a wait on the compute stream."""

        self._prefetch_calls += 1
        return self._acquire(key, prefetched=True)

    def _acquire(self, key: object, *, prefetched: bool) -> DynamicResidencyLease:
        if self._poisoned:
            raise DynamicResidencyPoisoned(self._poison_reason or "unknown")
        if self._closed:
            raise RuntimeError("AIMDO residency is closed")
        self._poll_retirements()
        group = self._groups[key]
        if id(group) in self._active:
            raise RuntimeError("AIMDO residency group is already acquired")
        first_acquire = self._first_acquire_pending
        if first_acquire:
            self._emit_diagnostic(
                "first_acquire_begin",
                group_bytes=group.staged_bytes,
                group_count=len(self._groups),
                staged_bytes=self._staged_bytes,
                virtual_bytes=self._virtual_bytes,
            )
        self._faults += 1
        signature = self._model_vbar_module.vbar_fault(group.allocation)
        if signature is not None:
            self._faulted.add(id(group))
        # Native fault ownership begins immediately when ``vbar_fault``
        # returns. No diagnostic, signature comparison, allocation, or copy is
        # allowed to run before a token can drive quiescence and exact unpin.
        token = _LeaseToken(group, None)
        self._active[id(group)] = token
        try:
            if first_acquire:
                self._emit_diagnostic(
                    "first_acquire_after_fault",
                    signature_none=signature is None,
                )
            hit = self._model_vbar_module.vbar_signature_compare(signature, group.signature)
            if hit:
                if group.resident_values is None:
                    raise RuntimeError("AIMDO signature hit has no reconstructed resident views")
                unique_values = group.resident_values
                self._signature_hits += 1
                if first_acquire:
                    self._emit_diagnostic(
                        "first_acquire_copy_ready",
                        signature_hit=True,
                        destination_bytes=group.staged_bytes,
                        transfer_events=0,
                    )
            else:
                self._signature_misses += 1
                if signature is None:
                    raw = torch.empty((group.allocation[2],), dtype=torch.uint8, device=self.device)
                    token.temporary = raw
                    self._fault_none_temporaries += 1
                else:
                    raw = self._aimdo_torch.aimdo_to_tensor(group.allocation, self.device)
                # Retain the raw destination before reconstruction and before
                # copy/event/current-stream operations can throw.
                token.raw = raw
                if first_acquire:
                    self._emit_diagnostic(
                        "first_acquire_raw_ready",
                        signature_hit=False,
                        destination_bytes=int(raw.numel() * raw.element_size()),
                        temporary=token.temporary is not None,
                    )
                unique_values = tuple(_rebuild_value(layout, raw) for layout in group.layouts)
                _mark_signature_cacheable(
                    unique_values,
                    cacheable=signature is not None,
                )
                if self._copy_strategy == "gathered_host_buffer":
                    self._copy_sources_gathered(group, raw, token)
                else:
                    self._copy_sources(group, raw, token)
                if first_acquire:
                    self._emit_diagnostic(
                        "first_acquire_copy_ready",
                        signature_hit=False,
                        destination_bytes=int(raw.numel() * raw.element_size()),
                        transfer_events=len(token.pending_events),
                    )
                if signature is not None:
                    group.signature = signature
                    group.resident_values = unique_values
            values = tuple(unique_values[index] for index in group.value_indices)
        except HostSourcePoolStructuralError as primary:
            self._abort_failed_acquire(token, primary)
            self._mark_poisoned(_HOST_SOURCE_POOL_POISON_REASON)
            raise DynamicResidencyPoisoned(_HOST_SOURCE_POOL_POISON_REASON) from primary
        except BaseException as primary:
            self._abort_failed_acquire(token, primary)
            raise
        if first_acquire:
            self._first_acquire_pending = False
        return DynamicResidencyLease(values, token)

    def wait(self, lease: DynamicResidencyLease) -> None:
        """Defer the transfer dependency until the owning execution group runs."""

        token = lease.token
        if not isinstance(token, _LeaseToken) or id(token.group) not in self._active:
            raise RuntimeError("AIMDO residency lease is invalid or already released")
        if token.waited:
            return
        try:
            current = torch.cuda.current_stream(self.device)
            for event in token.pending_events:
                current.wait_event(event)
                self._transfer_waits += 1
            token.waited = True
        except BaseException as primary:
            self._abort_failed_acquire(token, primary)
            raise

    def _emit_diagnostic(self, phase: str, **details: object) -> None:
        callback = self._diagnostic
        if callback is None:
            return
        callback(
            phase,
            {
                "device": str(self.device),
                "current_device": int(torch.cuda.current_device()),
                **details,
            },
        )

    def synchronize(self, lease: DynamicResidencyLease) -> None:
        token = lease.token
        if not isinstance(token, _LeaseToken) or id(token.group) not in self._active:
            raise RuntimeError("AIMDO residency lease is invalid or already released")
        if token.synchronized:
            return
        self.wait(lease)
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
        except BaseException as primary:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as quiescence_error:  # noqa: BLE001 - poison boundary
                primary.add_note(
                    "AIMDO lease device quiescence also failed: "
                    f"{type(quiescence_error).__name__}: {quiescence_error}"
                )
                self._mark_poisoned("device_quiescence_failed")
                raise DynamicResidencyPoisoned("device_quiescence_failed") from primary
            # A whole-device barrier proves the lease safe to clean. Preserve
            # the original event failure while allowing the caller to unbind
            # and unpin through the idempotent release path.
            token.synchronized = True
            raise
        token.synchronized = True

    def release(self, lease: DynamicResidencyLease) -> None:
        self.release_group((lease,))

    def release_group(self, leases: tuple[DynamicResidencyLease, ...]) -> None:
        """Retire one scheduling group without a successful-path host barrier.

        Transfers run on Engine-owned copy streams. ``wait`` establishes their
        forward dependency on the compute stream. At group exit, every transfer
        stream used by the group waits once on the current compute stream before
        it can be reused. VBAR pages are then unpinned just like the low-level
        AIMDO operation path, while fault(None) destinations and HostBuffer lease
        metadata remain owned until the completion event is host-observable.
        """

        if not leases:
            raise ValueError("AIMDO residency release group cannot be empty")
        tokens: list[_LeaseToken] = []
        seen_groups: set[int] = set()
        for lease in leases:
            token = lease.token
            group_id = id(token.group) if isinstance(token, _LeaseToken) else -1
            if (
                not isinstance(token, _LeaseToken)
                or group_id not in self._active
                or group_id in seen_groups
            ):
                raise RuntimeError("AIMDO residency release group contains an invalid lease")
            seen_groups.add(group_id)
            tokens.append(token)

        try:
            current = torch.cuda.current_stream(self.device)
            waited_tokens = tuple(token for token in tokens if token.waited)
            unwaited_tokens = tuple(token for token in tokens if not token.waited)
            compute_done: Any | None = None
            if waited_tokens:
                # One event represents all compute enqueued for this scheduling
                # group. Every Engine transfer stream consumes it before it may
                # fault or stage later VBAR work, including a stream that did
                # not participate in this group's inbound copies.
                compute_done = torch.cuda.Event()
                compute_done.record(current)
                for stream in self._streams:
                    stream.wait_event(compute_done)
                    self._reverse_stream_waits += 1

            for token in waited_tokens:
                if token.temporary is None:
                    self._model_vbar_module.vbar_unpin(token.group.allocation)
                    self._unpin_calls += 1
                    self._faulted.discard(id(token.group))
                self._active.pop(id(token.group), None)

            waited_deferred_ids = {
                id(token)
                for token in waited_tokens
                if token.temporary is not None or token.source_lease is not None
            }
            waited_deferred = tuple(
                token for token in waited_tokens if id(token) in waited_deferred_ids
            )
            for token in waited_tokens:
                if id(token) not in waited_deferred_ids:
                    self._clear_retired_token(token)
            if waited_deferred:
                self._retirements.append(
                    _RetirementBatch(waited_deferred, (compute_done,))
                )
                self._retirement_batches += 1

            # An abandoned or partially consumed prefetch has no compute-stream
            # dependency. Keep both its VBAR pin and allocation-busy identity
            # until its own inbound transfer events are host-observable complete.
            for token in unwaited_tokens:
                self._retirements.append(
                    _RetirementBatch(
                        (token,), tuple(token.pending_events), unpin_on_complete=True
                    )
                )
                self._retirement_batches += 1
        except BaseException as primary:  # noqa: BLE001 - native ownership boundary
            self._release_group_after_failure(tuple(tokens), primary)
        # Querying is non-blocking. Keep it outside the ordered-retirement
        # recovery boundary: a structural source-pool completion failure must
        # retain the already-retired batch exactly for hard child exit.
        self._poll_retirements()

    def invalidate(self, *, reason: str) -> None:
        self.invalidate_groups(tuple(self._groups), reason=reason)

    def invalidate_groups(self, keys: tuple[object, ...], *, reason: str) -> None:
        retirement_held = {
            id(token.group)
            for batch in self._retirements
            if batch.unpin_on_complete
            for token in batch.tokens
        }
        if any(group_id not in retirement_held for group_id in self._active):
            raise RuntimeError("AIMDO residency cannot invalidate active groups")
        self._drain_retirements()
        if self._active:
            raise RuntimeError("AIMDO retirement drain retained active groups")
        if not keys or len(set(keys)) != len(keys) or any(key not in self._groups for key in keys):
            raise ValueError("AIMDO residency invalidation groups are not canonical")
        self._dirty_epoch += 1
        groups = tuple(self._groups[key] for key in keys)
        for group in groups:
            group.signature = None
            group.resident_values = None
        if self._host_source_pool is not None and any(
            _group_source_class(group) is HostSourceClass.PATCH for group in groups
        ):
            try:
                self._host_source_pool.invalidate_patch_sources()
            except HostSourcePoolStructuralError as exc:
                self._mark_poisoned(_HOST_SOURCE_POOL_POISON_REASON)
                raise DynamicResidencyPoisoned(
                    _HOST_SOURCE_POOL_POISON_REASON
                ) from exc
        if reason == "lora_to_base":
            self._lora_invalidations += 1
            self._base_restores += 1

    def prepare_stage(self, required_free_bytes: int) -> None:
        """Drain one component boundary and trim only its measured shortfall."""

        if (
            not isinstance(required_free_bytes, int)
            or isinstance(required_free_bytes, bool)
            or required_free_bytes < 0
        ):
            raise ValueError("AIMDO stage requirement must be a non-negative integer")
        if self._poisoned:
            raise DynamicResidencyPoisoned(self._poison_reason or "unknown")
        if self._closed:
            raise RuntimeError("AIMDO residency is closed")

        self._stage_prepare_calls += 1
        self._stage_prepare_requested_bytes += required_free_bytes
        self._stage_prepare_pending_before = len(self._retirements)
        try:
            self._stage_prepare_loaded_before = int(self._vbar.loaded_size())
            before = self._cuda_memory_snapshot()
            self._stage_prepare_cuda_allocated_before = before["allocated"]
            self._stage_prepare_cuda_reserved_before = before["reserved"]
            self._stage_prepare_cuda_free_before = before["free"]
        except BaseException as exc:
            self._mark_poisoned("stage_prepare_failed")
            raise DynamicResidencyPoisoned("stage_prepare_failed") from exc
        try:
            # This is the one explicit stage-boundary host barrier. Hot leaf
            # release remains entirely stream ordered.
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            self._mark_poisoned("device_quiescence_failed")
            raise DynamicResidencyPoisoned("device_quiescence_failed") from exc

        batches = tuple(self._retirements)
        try:
            for index, batch in enumerate(batches):
                try:
                    self._complete_retirement_batch(batch)
                except BaseException:
                    self._retirements = list(batches[index:])
                    raise
            self._retirements.clear()
            self._retirement_completions += len(batches)
            self._stage_prepare_pending_after = 0

            # Cached Torch blocks are useful within a denoise stage but compete
            # with the next whole-module load. Release them only here.
            torch.cuda.empty_cache()
            after_cache = self._cuda_memory_snapshot()
            reusable = max(0, after_cache["reserved"] - after_cache["allocated"])
            allocatable = min(after_cache["total"], after_cache["free"] + reusable)
            shortfall = max(0, required_free_bytes - allocatable)
            loaded_after_drain = int(self._vbar.loaded_size())
            unpinned_resident = self._resident_unpinned_bytes(loaded_after_drain)
            trim_requested = min(shortfall, unpinned_resident)
            trim_freed = 0
            if trim_requested > 0:
                native_freed = self._vbar.free_memory(trim_requested)
                if (
                    not isinstance(native_freed, int)
                    or isinstance(native_freed, bool)
                    or native_freed < 0
                    or native_freed > unpinned_resident
                ):
                    raise RuntimeError("AIMDO VBAR trim returned invalid byte accounting")
                trim_freed = native_freed
            self._stage_prepare_trim_requested = trim_requested
            self._stage_prepare_trim_freed = trim_freed
            self._stage_prepare_loaded_after = int(self._vbar.loaded_size())
            after = self._cuda_memory_snapshot()
            self._stage_prepare_cuda_allocated_after = after["allocated"]
            self._stage_prepare_cuda_reserved_after = after["reserved"]
            self._stage_prepare_cuda_free_after = after["free"]
        except BaseException as exc:
            reason = (
                _HOST_SOURCE_POOL_POISON_REASON
                if isinstance(exc, HostSourcePoolStructuralError)
                else "stage_prepare_failed"
            )
            self._mark_poisoned(reason)
            raise DynamicResidencyPoisoned(reason) from exc

    def diagnostics(self) -> dict[str, Any]:
        # A failed device-quiescence barrier makes even a native loaded-size
        # query unsuitable as proof. ``None`` is deliberately unpublishable by
        # the managed validator while retained live ownership remains explicit.
        loaded = (
            0 if self._closed else None if self._close_failed else int(self._vbar.loaded_size())
        )
        pool_proof = (
            {
                "generation": 0,
                "lane_count": 0,
                "capacity_bytes": 0,
                "retained_slices": 0,
                "retained_bytes": 0,
                "temporary_slices": 0,
                "temporary_bytes": 0,
                "source_hits": 0,
                "source_misses": 0,
                "stale_rejections": 0,
                "warm_ram_pressure_bypasses": 0,
                "warm_zero_delta_extend_refusals": 0,
                "warm_registration_refusals": 0,
                "temporary_ram_pressure_bypasses": 0,
                "temporary_zero_delta_extend_refusals": 0,
                "temporary_registration_refusals": 0,
                "live": False,
                "poisoned": False,
                "poison_reason": None,
                "transfer_pending": False,
                "registration_budget_bytes": self._host_registration_budget_bytes,
                "registration_attempts": 0,
                "registration_attempt_bytes": 0,
                "registration_successes": 0,
                "registration_failures": 0,
                "registration_failure_bytes": 0,
                "registration_registered_bytes": 0,
                "registration_unregistered_bytes": 0,
                "registration_live_bytes": 0,
                "registration_peak_bytes": 0,
                "registration_state_proven": True,
            }
            if self._host_source_pool is None and self._host_source_pool_diagnostics is None
            else (
                self._host_source_pool_diagnostics
                if self._host_source_pool is None
                else self._host_source_pool.diagnostics()
            )
        )
        return {
            "backend": "comfy-aimdo",
            "version": "0.4.15",
            "mode": "dynamic_vbar",
            "physical_bytes": self._physical_bytes,
            "staged_bytes": self._staged_bytes,
            "virtual_bytes": self._virtual_bytes,
            "allocation_count": self._allocation_count,
            "live_allocations": 0 if self._closed else len(self._groups),
            "live_bytes": 0 if self._closed else self._virtual_bytes,
            "loaded_bytes": loaded,
            "faults": self._faults,
            "signature_hits": self._signature_hits,
            "signature_misses": self._signature_misses,
            "fault_none_temporaries": self._fault_none_temporaries,
            "pinned_copy_bytes": self._pinned_copy_bytes,
            "pageable_copy_bytes": self._pageable_copy_bytes,
            "transfer_events": self._transfer_events,
            "transfer_waits": self._transfer_waits,
            "reverse_stream_waits": self._reverse_stream_waits,
            "retirement_batches": self._retirement_batches,
            "retirement_polls": self._retirement_polls,
            "retirement_completions": self._retirement_completions,
            "pending_retirement_batches": len(self._retirements),
            "stage_prepare_calls": self._stage_prepare_calls,
            "stage_prepare_requested_bytes": self._stage_prepare_requested_bytes,
            "stage_prepare_pending_before": self._stage_prepare_pending_before,
            "stage_prepare_pending_after": self._stage_prepare_pending_after,
            "stage_prepare_loaded_before": self._stage_prepare_loaded_before,
            "stage_prepare_loaded_after": self._stage_prepare_loaded_after,
            "stage_prepare_trim_requested": self._stage_prepare_trim_requested,
            "stage_prepare_trim_freed": self._stage_prepare_trim_freed,
            "stage_prepare_cuda_allocated_before": self._stage_prepare_cuda_allocated_before,
            "stage_prepare_cuda_reserved_before": self._stage_prepare_cuda_reserved_before,
            "stage_prepare_cuda_free_before": self._stage_prepare_cuda_free_before,
            "stage_prepare_cuda_allocated_after": self._stage_prepare_cuda_allocated_after,
            "stage_prepare_cuda_reserved_after": self._stage_prepare_cuda_reserved_after,
            "stage_prepare_cuda_free_after": self._stage_prepare_cuda_free_after,
            "prioritize_calls": self._prioritize_calls,
            "unpin_calls": self._unpin_calls,
            "free_calls": self._free_calls,
            "dirty_epoch": self._dirty_epoch,
            "lora_invalidations": self._lora_invalidations,
            "base_restores": self._base_restores,
            "copy_stream_count": self._copy_stream_count,
            "copy_strategy": self._copy_strategy,
            "copy_fallback_reason": self._copy_fallback_reason,
            "gathered_host_buffer_requested": self._gathered_host_buffer_requested,
            "host_buffer_capacity_bytes": self._host_buffer_capacity_bytes,
            "host_buffer_allocations": self._host_buffer_allocations,
            "host_buffer_unregistrations": self._host_buffer_unregistrations,
            "host_buffer_frees": self._host_buffer_frees,
            "host_buffer_live": pool_proof["live"],
            "host_tensor_view_live": bool(
                pool_proof["retained_slices"] or pool_proof["temporary_slices"]
            ),
            "host_buffer_transfer_pending": pool_proof["transfer_pending"],
            "gathered_misses": self._gathered_misses,
            "per_physical_misses": self._per_physical_misses,
            "packed_source_bytes": self._packed_source_bytes,
            "gathered_h2d_bytes": self._gathered_h2d_bytes,
            "host_buffer_reuse_barriers": self._host_buffer_reuse_barriers,
            "pressure_direct_transfers": self._pressure_direct_transfers,
            "pressure_direct_bytes": self._pressure_direct_bytes,
            "host_source_pool_generation": pool_proof["generation"],
            "host_source_pool_lane_count": pool_proof["lane_count"],
            "host_source_pool_capacity_bytes": pool_proof["capacity_bytes"],
            "host_source_pool_retained_slices": pool_proof["retained_slices"],
            "host_source_pool_retained_bytes": pool_proof["retained_bytes"],
            "host_source_pool_temporary_slices": pool_proof["temporary_slices"],
            "host_source_pool_temporary_bytes": pool_proof["temporary_bytes"],
            "host_source_pool_hits": pool_proof["source_hits"],
            "host_source_pool_misses": pool_proof["source_misses"],
            "host_source_pool_stale_rejections": pool_proof["stale_rejections"],
            "host_source_pool_warm_ram_pressure_bypasses": pool_proof[
                "warm_ram_pressure_bypasses"
            ],
            "host_source_pool_warm_zero_delta_extend_refusals": pool_proof[
                "warm_zero_delta_extend_refusals"
            ],
            "host_source_pool_warm_registration_refusals": pool_proof[
                "warm_registration_refusals"
            ],
            "host_source_pool_temporary_ram_pressure_bypasses": pool_proof[
                "temporary_ram_pressure_bypasses"
            ],
            "host_source_pool_temporary_zero_delta_extend_refusals": pool_proof[
                "temporary_zero_delta_extend_refusals"
            ],
            "host_source_pool_temporary_registration_refusals": pool_proof[
                "temporary_registration_refusals"
            ],
            "host_source_pool_poisoned": pool_proof["poisoned"],
            "host_source_pool_poison_reason": pool_proof["poison_reason"],
            "host_source_registration": {
                "policy": "aimdo_hostbuffer_registered_append",
                "budget_bytes": pool_proof["registration_budget_bytes"],
                "attempts": pool_proof["registration_attempts"],
                "attempt_bytes": pool_proof["registration_attempt_bytes"],
                "successes": pool_proof["registration_successes"],
                "failures": pool_proof["registration_failures"],
                "failure_bytes": pool_proof["registration_failure_bytes"],
                "registered_bytes": pool_proof["registration_registered_bytes"],
                "unregistered_bytes": pool_proof["registration_unregistered_bytes"],
                "live_bytes": pool_proof["registration_live_bytes"],
                "peak_bytes": pool_proof["registration_peak_bytes"],
                "state_proven": pool_proof["registration_state_proven"],
            },
            "prefetch": self._prefetch_calls > 0,
            "prefetch_calls": self._prefetch_calls,
            "allocator_plugin": False,
            "poisoned": self._poisoned,
            "close_failed": self._close_failed,
            "poison_reason": self._poison_reason,
            "host_registration": self._ledger.provenance(),
            "base_file_backed": self._base_file_backed,
            "base_file_source_live": bool(self._file_sources),
            "base_file_read_calls": self._base_file_read_calls,
            "base_file_read_bytes": self._base_file_read_bytes,
        }

    def terminal_poison_reason(self) -> str | None:
        return self._poison_reason

    def close(self) -> None:
        if self._closed:
            return
        if self._close_failed:
            raise DynamicResidencyPoisoned(self._poison_reason or "unknown")
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            # Never unmap an address that may still be referenced by queued GPU
            # work. Preserve every token, reconstructed raw view, registration,
            # stream, group, and VBAR allocation for OS process teardown.
            self._mark_poisoned("device_quiescence_failed")
            raise DynamicResidencyPoisoned("device_quiescence_failed") from exc
        primary: BaseException | None = None
        for token in tuple(self._active.values()):
            token.source_lease = None
            token.temporary = None
            token.raw = None
            token.pending_events.clear()
        self._active.clear()
        for batch in self._retirements:
            for token in batch.tokens:
                token.source_lease = None
                token.temporary = None
                token.raw = None
                token.pending_events.clear()
                token.transfer_streams.clear()
        self._retirements.clear()
        # The whole-device barrier proves the reusable host buffer is no
        # longer referenced by any transfer stream.
        self._host_buffer_pending_event = None
        for group in self._groups.values():
            if id(group) not in self._faulted:
                continue
            try:
                self._model_vbar_module.vbar_unpin(group.allocation)
                self._unpin_calls += 1
            except BaseException as exc:  # noqa: BLE001 - continue mandatory cleanup
                if primary is None:
                    primary = exc
                else:
                    primary.add_note(f"AIMDO VBAR unpin also failed: {exc}")
        self._faulted.clear()
        errors = self._ledger.unregister_owned()
        if errors:
            if primary is None:
                primary = errors[0]
            else:
                primary.add_note(f"AIMDO host unregistration also failed: {errors[0]}")
        try:
            self._dispose_host_buffer()
        except HostSourcePoolStructuralError as exc:
            # The HostBuffer graph owns native registrations and may still
            # contain a partially mutated lane. Do not free VBAR, close file
            # sources, or discard any ownership after this terminal failure.
            self._mark_poisoned(_HOST_SOURCE_POOL_POISON_REASON)
            raise DynamicResidencyPoisoned(_HOST_SOURCE_POOL_POISON_REASON) from exc
        except BaseException as exc:  # noqa: BLE001 - continue VBAR cleanup after sync
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"AIMDO gathered HostBuffer cleanup also failed: {exc}")
        try:
            ptr = getattr(self._vbar, "_ptr", None)
            if ptr:
                self._model_vbar_module.lib.vbar_free(
                    self._vbar._devctx,
                    ptr,
                )
                self._vbar._ptr = None
                self._free_calls += 1
        except BaseException as exc:  # noqa: BLE001 - cleanup preserves the primary failure
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"AIMDO VBAR free also failed: {exc}")
        self._groups.clear()
        self._file_sources.clear()
        self._streams = ()
        self._closed = True
        if primary is not None:
            raise RuntimeError("AIMDO residency cleanup was incomplete") from primary

    def _abort_failed_acquire(
        self,
        token: _LeaseToken,
        primary: BaseException,
    ) -> None:
        """Release a failed fill only after a proven whole-device barrier."""

        try:
            torch.cuda.synchronize(self.device)
        except BaseException as quiescence_error:  # noqa: BLE001 - poison boundary
            primary.add_note(
                "AIMDO failed-fill device quiescence also failed: "
                f"{type(quiescence_error).__name__}: {quiescence_error}"
            )
            self._mark_poisoned("failed_fill_quiescence_failed")
            raise DynamicResidencyPoisoned("failed_fill_quiescence_failed") from primary
        if token.temporary is None and id(token.group) in self._faulted:
            try:
                self._model_vbar_module.vbar_unpin(token.group.allocation)
                self._unpin_calls += 1
                self._faulted.discard(id(token.group))
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary
                primary.add_note(
                    "AIMDO failed-fill unpin also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        # The successful whole-device barrier also proves any gathered copy
        # completed. Release only this lease's source slice; retained immutable
        # source data stays pooled for later nonresident signature misses.
        self._host_buffer_pending_event = None
        if token.source_lease is not None and self._host_source_pool is not None:
            try:
                self._host_source_pool.complete_transfer(
                    token.source_lease,
                    quiesced=True,
                )
                token.source_lease = None
            except HostSourcePoolStructuralError as cleanup_error:
                primary.add_note(
                    "AIMDO host source structural cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                self._mark_poisoned(_HOST_SOURCE_POOL_POISON_REASON)
                raise DynamicResidencyPoisoned(
                    _HOST_SOURCE_POOL_POISON_REASON
                ) from primary
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary
                primary.add_note(
                    "AIMDO host source cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        token.temporary = None
        token.raw = None
        token.pending_events.clear()
        token.transfer_streams.clear()
        self._active.pop(id(token.group), None)

    def _release_group_after_failure(
        self,
        tokens: tuple[_LeaseToken, ...],
        primary: BaseException,
    ) -> None:
        """Use one explicit barrier to recover a failed ordered retirement."""

        try:
            torch.cuda.synchronize(self.device)
        except BaseException as quiescence_error:  # noqa: BLE001 - poison boundary
            primary.add_note(
                "AIMDO retirement device quiescence also failed: "
                f"{type(quiescence_error).__name__}: {quiescence_error}"
            )
            self._mark_poisoned("retirement_quiescence_failed")
            raise DynamicResidencyPoisoned("retirement_quiescence_failed") from primary
        for token in tokens:
            try:
                if token.temporary is None and id(token.group) in self._faulted:
                    self._model_vbar_module.vbar_unpin(token.group.allocation)
                    self._unpin_calls += 1
                    self._faulted.discard(id(token.group))
                self._complete_source_lease(token)
                self._clear_retired_token(token)
                self._active.pop(id(token.group), None)
            except BaseException as cleanup_error:  # noqa: BLE001 - ownership boundary
                primary.add_note(
                    "AIMDO retirement recovery cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                reason = (
                    _HOST_SOURCE_POOL_POISON_REASON
                    if isinstance(cleanup_error, HostSourcePoolStructuralError)
                    else "retirement_cleanup_failed"
                )
                self._mark_poisoned(reason)
                raise DynamicResidencyPoisoned(reason) from primary
        # The scheduler cannot distinguish an exception raised before ownership
        # changed from one raised after this recovery consumed the leases. Make
        # the seam terminal so stale scheduler entries can never be retried.
        self._mark_poisoned("retirement_release_failed")
        raise DynamicResidencyPoisoned("retirement_release_failed") from primary

    def _poll_retirements(self) -> None:
        if not self._retirements:
            return
        self._retirement_polls += 1
        pending: list[_RetirementBatch] = []
        batches = tuple(self._retirements)
        for index, batch in enumerate(batches):
            try:
                complete = all(bool(event.query()) for event in batch.completion_events)
            except BaseException as exc:
                self._mark_poisoned("retirement_query_failed")
                raise DynamicResidencyPoisoned("retirement_query_failed") from exc
            if not complete:
                pending.append(batch)
                continue
            try:
                self._complete_retirement_batch(batch)
            except BaseException as exc:
                # Keep the exact batch and its ownership for hard child exit.
                self._retirements = pending + list(batches[index:])
                reason = (
                    _HOST_SOURCE_POOL_POISON_REASON
                    if isinstance(exc, HostSourcePoolStructuralError)
                    else "retirement_cleanup_failed"
                )
                self._mark_poisoned(reason)
                raise DynamicResidencyPoisoned(reason) from exc
            self._retirement_completions += 1
        self._retirements = pending

    def _drain_retirements(self) -> None:
        if not self._retirements:
            return
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            self._mark_poisoned("retirement_quiescence_failed")
            raise DynamicResidencyPoisoned("retirement_quiescence_failed") from exc
        batches = tuple(self._retirements)
        try:
            for batch in batches:
                self._complete_retirement_batch(batch)
        except BaseException as exc:
            reason = (
                _HOST_SOURCE_POOL_POISON_REASON
                if isinstance(exc, HostSourcePoolStructuralError)
                else "retirement_cleanup_failed"
            )
            self._mark_poisoned(reason)
            raise DynamicResidencyPoisoned(reason) from exc
        self._retirements.clear()
        self._retirement_completions += len(batches)

    def _complete_source_lease(self, token: _LeaseToken) -> None:
        if token.source_lease is None:
            return
        if self._host_source_pool is None:
            raise RuntimeError("AIMDO host source pool disappeared during retirement")
        self._host_source_pool.complete_transfer(token.source_lease, quiesced=True)
        token.source_lease = None

    def _complete_retirement_batch(self, batch: _RetirementBatch) -> None:
        for token in batch.tokens:
            if batch.unpin_on_complete:
                if token.temporary is None and id(token.group) in self._faulted:
                    self._model_vbar_module.vbar_unpin(token.group.allocation)
                    self._unpin_calls += 1
                    self._faulted.discard(id(token.group))
                self._active.pop(id(token.group), None)
            self._complete_source_lease(token)
            self._clear_retired_token(token)

    def _resident_unpinned_bytes(self, loaded_bytes: int) -> int:
        residency = self._vbar.get_residency()
        if not isinstance(residency, list) or any(
            not isinstance(flags, int) or isinstance(flags, bool) or flags < 0
            for flags in residency
        ):
            raise RuntimeError("AIMDO VBAR residency map is not canonical")
        page_bytes = 32 * 1024**2
        unpinned_pages = sum(
            1 for flags in residency if flags & 1 and not flags & 2
        )
        return min(loaded_bytes, unpinned_pages * page_bytes)

    def _cuda_memory_snapshot(self) -> dict[str, int]:
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        values = (free, total, allocated, reserved)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise RuntimeError("CUDA memory snapshot is not canonical")
        return {
            "free": int(free),
            "total": int(total),
            "allocated": int(allocated),
            "reserved": int(reserved),
        }

    @staticmethod
    def _clear_retired_token(token: _LeaseToken) -> None:
        token.temporary = None
        token.raw = None
        token.pending_events.clear()
        token.transfer_streams.clear()

    def _mark_poisoned(self, reason: str) -> None:
        self._poisoned = True
        self._close_failed = True
        self._poison_reason = reason

    def _copy_sources(
        self,
        group: _Group,
        raw: torch.Tensor,
        token: _LeaseToken,
    ) -> None:
        self._per_physical_misses += 1
        stream_index = 0
        copied: set[tuple[object, ...]] = set()
        for layout in group.layouts:
            for physical in layout.physical:
                if physical.tensor is None:
                    raise RuntimeError("AIMDO per-physical fallback cannot read file spans")
                physical_key = _physical_identity(physical)
                if physical_key in copied:
                    continue
                copied.add(physical_key)
                pinned = self._ledger.consider(physical.tensor)
                stream = self._streams[stream_index]
                if all(existing is not stream for existing in token.transfer_streams):
                    token.transfer_streams.append(stream)
                stream_index = (stream_index + 1) % 2
                destination = raw[physical.offset : physical.offset + physical.size]
                source = physical.tensor.reshape(-1).view(torch.uint8)
                with torch.cuda.stream(stream):
                    destination.copy_(source, non_blocking=True)
                    event = torch.cuda.Event()
                    token.pending_events.append(event)
                    event.record(stream)
                self._transfer_events += 1
                if pinned:
                    self._pinned_copy_bytes += physical.size
                else:
                    self._pageable_copy_bytes += physical.size

    def _copy_sources_gathered(
        self,
        group: _Group,
        raw: torch.Tensor,
        token: _LeaseToken,
    ) -> None:
        pool = self._host_source_pool
        if pool is None:
            raise RuntimeError("AIMDO gathered HostBuffer ownership is invalid")
        source_class = _group_source_class(group)
        lifetime = _group_source_lifetime(
            group,
            signature_none=token.temporary is not None,
        )
        cache_key = (
            _source_cache_key(group, source_class=source_class, patch_epoch=self._dirty_epoch)
            if lifetime is HostSourceLifetime.WARM
            else None
        )
        try:
            source_lease = pool.acquire(
                source_class,
                lifetime,
                size=group.staged_bytes,
                cache_key=cache_key,
            )
        except (HostSourceWarmUnavailable, HostSourceDirectTransferRequired) as unavailable:
            if unavailable.reason in {
                "physical_ram_pressure",
                "native_extend_refused_without_delta",
                "cuda_host_registration_refused",
                "host_buffer_appended_view_refused",
            }:
                # Do not ask another HostBuffer to grow under proven physical
                # pressure. File spans use AIMDO's narrow direct reader and
                # CPU physicals use ordinary pageable Torch copies, all on one
                # transfer stream with one completion event.
                self._copy_sources_pressure_direct(group, raw, token)
                return
            # Configured lane/registration admission happens before native
            # mutation. Keep the destination unchanged and stage this one
            # transfer through the reclaimable lane instead.
            source_lease = pool.acquire(
                source_class,
                HostSourceLifetime.PREFETCH_TEMPORARY,
                size=group.staged_bytes,
            )
        # Publish pool ownership before any fill, copy, event, or wait can
        # fail. Failed quiescence retains this exact lease and every view.
        token.source_lease = source_lease
        packed = source_lease.tensor
        copied: set[tuple[object, ...]] = set()
        source_bytes = 0
        if source_lease.needs_fill:
            packed.zero_()
            for layout in group.layouts:
                for physical in layout.physical:
                    physical_key = _physical_identity(physical)
                    if physical_key in copied:
                        continue
                    copied.add(physical_key)
                    if physical.file_span is not None:
                        span = physical.file_span
                        try:
                            handle = self._file_sources[span.source_id]
                        except KeyError as exc:
                            raise RuntimeError("AIMDO file source was not bound") from exc
                        pool.read_file_slice(
                            source_lease,
                            handle,
                            span.offset,
                            span.size,
                            slice_offset=physical.offset,
                        )
                        self._base_file_read_calls += 1
                        self._base_file_read_bytes += span.size
                    else:
                        source = physical.tensor.reshape(-1).view(torch.uint8)
                        packed[physical.offset : physical.offset + physical.size].copy_(source)
                    source_bytes += physical.size
        else:
            source_bytes = _group_physical_source_bytes(group)
        stream = self._streams[self._gathered_stream_index]
        token.transfer_streams.append(stream)
        self._gathered_stream_index = (self._gathered_stream_index + 1) % 2
        with torch.cuda.stream(stream):
            raw.copy_(packed, non_blocking=True)
            event = torch.cuda.Event()
            token.pending_events.append(event)
            event.record(stream)
            pool.add_fence(source_lease, event)
        self._gathered_misses += 1
        self._packed_source_bytes += source_bytes
        self._gathered_h2d_bytes += group.staged_bytes
        self._pinned_copy_bytes += group.staged_bytes
        self._transfer_events += 1

    def _copy_sources_pressure_direct(
        self,
        group: _Group,
        raw: torch.Tensor,
        token: _LeaseToken,
    ) -> None:
        """Bypass HostBuffer growth after an authenticated pressure refusal."""

        reader = getattr(self._host_buffer_module, "read_file_to_device", None)
        has_file_source = any(
            physical.file_span is not None
            for layout in group.layouts
            for physical in layout.physical
        )
        if has_file_source and not callable(reader):
            raise RuntimeError(
                "AIMDO direct file reader is unavailable for host-memory pressure bypass"
            )
        stream = self._streams[self._gathered_stream_index]
        token.transfer_streams.append(stream)
        self._gathered_stream_index = (self._gathered_stream_index + 1) % 2
        stream_handle = getattr(stream, "cuda_stream", None)
        if (
            not isinstance(stream_handle, int)
            or isinstance(stream_handle, bool)
            or stream_handle <= 0
        ):
            raise RuntimeError("AIMDO transfer stream has no canonical native handle")
        copied: set[tuple[object, ...]] = set()
        source_bytes = 0
        pageable_bytes = 0
        with torch.cuda.stream(stream):
            for layout in group.layouts:
                for physical in layout.physical:
                    physical_key = _physical_identity(physical)
                    if physical_key in copied:
                        continue
                    copied.add(physical_key)
                    destination = raw[physical.offset : physical.offset + physical.size]
                    if physical.file_span is not None:
                        span = physical.file_span
                        try:
                            handle = self._file_sources[span.source_id]
                        except KeyError as exc:
                            raise RuntimeError("AIMDO file source was not bound") from exc
                        reader(
                            handle,
                            span.offset,
                            span.size,
                            stream_handle,
                            int(destination.data_ptr()),
                            self.device.index,
                            mark_cold=True,
                        )
                        self._base_file_read_calls += 1
                        self._base_file_read_bytes += span.size
                    else:
                        if physical.tensor is None:
                            raise RuntimeError(
                                "AIMDO direct pressure bypass has no CPU source"
                            )
                        source = physical.tensor.reshape(-1).view(torch.uint8)
                        destination.copy_(source, non_blocking=True)
                        pageable_bytes += physical.size
                    source_bytes += physical.size
            event = torch.cuda.Event()
            token.pending_events.append(event)
            event.record(stream)
        self._gathered_misses += 1
        self._packed_source_bytes += source_bytes
        self._gathered_h2d_bytes += group.staged_bytes
        self._pageable_copy_bytes += pageable_bytes
        self._transfer_events += 1
        self._pressure_direct_transfers += 1
        self._pressure_direct_bytes += group.staged_bytes


def _physical_identity(source: _PhysicalSource) -> tuple[object, ...]:
    if source.file_span is not None:
        span = source.file_span
        return ("file", span.source_id, span.offset, span.size)
    if source.tensor is None:
        raise RuntimeError("dynamic residency physical source has no owner")
    return ("cpu", int(source.tensor.data_ptr()), source.size)


def _group_source_class(group: _Group) -> HostSourceClass:
    """Classify only wholly file-backed leaves as immutable base sources."""

    return (
        HostSourceClass.BASE
        if all(
            physical.file_span is not None
            for layout in group.layouts
            for physical in layout.physical
        )
        else HostSourceClass.PATCH
    )


def _group_source_lifetime(
    group: _Group,
    *,
    signature_none: bool,
) -> HostSourceLifetime:
    if signature_none:
        if _group_source_class(group) is HostSourceClass.BASE:
            return HostSourceLifetime.WARM
        return HostSourceLifetime.PREFETCH_TEMPORARY
    physical = [item for layout in group.layouts for item in layout.physical]
    # A non-null native signature binds ordinary CPU patch bytes to the current
    # dirty epoch. File-backed base is immutable for the runtime identity; any
    # future mixed leaf remains temporary by construction.
    if all(item.file_span is not None for item in physical) or all(
        item.tensor is not None for item in physical
    ):
        return HostSourceLifetime.WARM
    return HostSourceLifetime.PREFETCH_TEMPORARY
def _source_cache_key(
    group: _Group,
    *,
    source_class: HostSourceClass,
    patch_epoch: int,
) -> tuple[object, ...]:
    """Bind retained bytes to immutable source identity and exact geometry."""

    physical: dict[tuple[object, ...], _PhysicalSource] = {}
    for layout in group.layouts:
        for item in layout.physical:
            physical.setdefault(_physical_identity(item), item)
    geometry: list[tuple[object, ...]] = []
    for item in physical.values():
        if item.file_span is not None:
            span = item.file_span
            geometry.append(
                (
                    "file",
                    span.source_id,
                    span.key,
                    span.offset,
                    span.size,
                    str(span.dtype),
                    span.shape,
                )
            )
        elif item.tensor is not None:
            geometry.append(
                (
                    "cpu",
                    int(item.tensor.data_ptr()),
                    item.size,
                    str(item.tensor.dtype),
                    tuple(item.tensor.shape),
                )
            )
        else:
            raise RuntimeError("AIMDO retained source has no authoritative owner")
    return (
        "aimdo_leaf_source_v1",
        source_class.value,
        0 if source_class is HostSourceClass.BASE else patch_epoch,
        group.staged_bytes,
        tuple(geometry),
    )


def _group_physical_source_bytes(group: _Group) -> int:
    return sum(
        physical.size
        for physical in {
            _physical_identity(item): item for layout in group.layouts for item in layout.physical
        }.values()
    )
