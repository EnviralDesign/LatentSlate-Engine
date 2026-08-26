from __future__ import annotations

import ctypes
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from latentslate_engine.runtime.framework.residency import (
    aimdo,
    host_registration,
    host_source_pool,
)


class _Ledger:
    def __init__(self, _budget: int) -> None:
        self.owned = {}
        self.calls = 0
        self.unregister_calls = 0

    def consider(self, _value: torch.Tensor) -> bool:
        self.calls += 1
        return self.calls % 2 == 1

    def unregister_owned(self):
        self.unregister_calls += 1
        return []

    def provenance(self):
        return {"fixture": True, "calls": self.calls}


class _Stream:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.cuda_stream = 100 + sum(ord(character) for character in name)

    def wait_event(self, _event) -> None:
        self.events.append(f"wait:{self.name}")


class _Event:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.recorded_stream: str | None = None

    def record(self, stream: _Stream) -> None:
        self.recorded_stream = stream.name
        self.events.append(f"record:{stream.name}")

    def synchronize(self) -> None:
        self.events.append(f"event-sync:{self.recorded_stream or 'unrecorded'}")

    def query(self) -> bool:
        self.events.append(f"event-query:{self.recorded_stream or 'unrecorded'}")
        return True


class _Cudart:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.register_results: list[int] = []
        self.unregister_results: list[int] = []
        self.registrations: list[tuple[int, int, int]] = []
        self.unregistrations: list[int] = []

    def cudaHostRegister(self, address: int, size: int, flags: int) -> int:
        result = self.register_results.pop(0) if self.register_results else 0
        if self.events is not None:
            self.events.append(f"cuda-register:{size}:{result}")
        if result == 0:
            self.registrations.append((address, size, flags))
        return result

    def cudaHostUnregister(self, address: int) -> int:
        result = self.unregister_results.pop(0) if self.unregister_results else 0
        if self.events is not None:
            self.events.append(f"cuda-unregister:{result}")
        if result == 0:
            self.unregistrations.append(address)
        return result

    def cudaGetLastError(self) -> int:
        if self.events is not None:
            self.events.append("cuda-get-last-error")
        return 0


class _VBAR:
    def __init__(self, size: int, device: int, state: SimpleNamespace) -> None:
        state.events.append(f"vbar:{size}:{device}")
        self.state = state
        self.max_size = size
        self.offset = 0
        self._ptr = 99
        self._devctx = 77
        self.buffers: dict[int, torch.Tensor] = {}

    def alloc(self, size: int):
        self.offset = (self.offset + 511) & ~511
        ptr = 10_000 + self.offset
        self.buffers[ptr] = torch.empty(size, dtype=torch.uint8)
        result = (self, ptr, size)
        self.state.allocations.append(result)
        self.offset += size
        return result

    def prioritize(self) -> None:
        self.state.events.append("prioritize")

    def loaded_size(self) -> int:
        if self.state.loaded_override is not None:
            return self.state.loaded_override
        return sum(buffer.numel() for buffer in self.buffers.values())

    def get_residency(self) -> list[int]:
        if self.state.residency is not None:
            return list(self.state.residency)
        return [1] if self.loaded_size() else []

    def free_memory(self, size: int) -> int:
        self.state.trim_requests.append(size)
        freed = min(size, self.loaded_size())
        self.state.loaded_override = self.loaded_size() - freed
        self.state.cuda_free = min(self.state.cuda_total, self.state.cuda_free + freed)
        return freed


def _backend_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    faults=None,
    current_index: int = 0,
    device_count: int = 1,
    host_buffer: bool = False,
):
    state = SimpleNamespace(
        events=[],
        allocations=[],
        faults=list(faults or []),
        unpins=[],
        frees=[],
        streams=[],
        current_stream_devices=[],
        host_buffers=[],
        host_buffer_args=[],
        direct_file_reads=[],
        cudart=_Cudart(),
        loaded_override=None,
        residency=None,
        trim_requests=[],
        cuda_free=8 * 1024**3,
        cuda_total=24 * 1024**3,
        cuda_allocated=0,
        cuda_reserved=0,
    )

    class _Control:
        @staticmethod
        def init(*, simple_vram_headroom, nvml_pressure):
            state.events.append(f"init:headroom={simple_vram_headroom}:nvml={nvml_pressure}")
            return True

        @staticmethod
        def init_devices(devices):
            state.events.append(f"init_devices:{devices}")
            return True

    class _Lib:
        @staticmethod
        def vbar_free(devctx, ptr):
            state.events.append("free")
            state.frees.append((devctx, ptr))

    class _ModelVBARModule:
        lib = _Lib()

        @staticmethod
        def ModelVBAR(size, device):
            return _VBAR(size, device, state)

        @staticmethod
        def vbar_fault(allocation):
            state.events.append("fault")
            if state.faults:
                return state.faults.pop(0)
            return (1, 2)

        @staticmethod
        def vbar_signature_compare(left, right):
            return left is not None and left == right

        @staticmethod
        def vbar_unpin(allocation):
            state.events.append("unpin")
            state.unpins.append(allocation)

    class _AimdoTorch:
        @staticmethod
        def aimdo_to_tensor(allocation, _device):
            vbar, ptr, _size = allocation
            return vbar.buffers[ptr]

        @staticmethod
        def hostbuf_to_tensor(value):
            return value.backing

    class _HostBuffer:
        def __init__(self, size, *, prewarm, max_grow_size, mark_cold):
            assert mark_cold is True
            state.host_buffer_args.append((size, prewarm, max_grow_size))
            self.size = size
            self.capacity = max(size, max_grow_size)
            self.storage = torch.empty(self.capacity, dtype=torch.uint8)
            self.backing = self.storage[:size]
            self._ptr = 1234 + len(state.host_buffers)
            state.host_buffers.append(self)
            state.events.append(f"host-allocate:{size}:{self.capacity}")

        def get_raw_address(self):
            return self.storage.data_ptr() if self.size > 0 else 0

        def extend(self, size, *, reallocate, register):
            assert reallocate is False
            assert register is False
            target = self.size + size
            if target > self.capacity:
                raise RuntimeError("fixture HostBuffer capacity exceeded")
            self.size = target
            self.backing = self.storage[:target]
            state.events.append(f"host-extend:{size}:{target}")
            return self.storage.data_ptr() + target - size

        def truncate(self, size, *, do_unregister):
            assert do_unregister is False
            self.size = size
            self.backing = self.storage[:size]
            state.events.append(f"host-unregister:{size}")

        def read_file_slice(
            self,
            file_obj,
            file_offset,
            size,
            *,
            offset,
            stream,
            device_ptr,
            device,
        ):
            assert stream == 0 and device_ptr == 0 and device == -1
            file_obj.seek(file_offset)
            payload = file_obj.read(size)
            if len(payload) != size:
                raise RuntimeError("fixture file slice is truncated")
            self.backing[offset : offset + size].copy_(
                torch.tensor(list(payload), dtype=torch.uint8)
            )
            state.events.append(f"host-read:{file_offset}:{size}:{offset}")

        def __del__(self):
            if getattr(self, "_ptr", None):
                state.events.append("host-free")
                self._ptr = None

    class _HostBufferModule:
        HostBuffer = _HostBuffer

        @staticmethod
        def read_file_to_device(
            file_obj,
            file_offset,
            size,
            stream,
            device_ptr,
            device,
            *,
            mark_cold,
        ):
            assert stream > 0 and device == current_index and mark_cold is True
            file_obj.seek(file_offset)
            payload = file_obj.read(size)
            if len(payload) != size:
                raise RuntimeError("fixture direct file slice is truncated")
            ctypes.memmove(device_ptr, payload, size)
            state.direct_file_reads.append(
                (file_offset, size, stream, device_ptr, device)
            )
            state.events.append(f"direct-file-read:{file_offset}:{size}")

    modules = {
        "comfy_aimdo.control": _Control,
        "comfy_aimdo.model_vbar": _ModelVBARModule,
        "comfy_aimdo.torch": _AimdoTorch,
    }
    if host_buffer:
        modules["comfy_aimdo.host_buffer"] = _HostBufferModule

    def imported(name: str):
        state.events.append(f"import:{name}")
        try:
            return modules[name]
        except KeyError as exc:
            raise ImportError(f"fixture has no {name}") from exc

    stream_counter = iter(range(2))

    def stream(*, device=None):
        index = next(stream_counter)
        value = _Stream(state.events, f"copy-{index}")
        state.streams.append((device, value))
        return value

    current = _Stream(state.events, "current")
    monkeypatch.setattr(aimdo, "_import_module", imported)
    monkeypatch.setattr(aimdo, "BestEffortHostRegistrationLedger", _Ledger)
    monkeypatch.setattr(aimdo, "default_host_registration_budget_bytes", lambda: 1_000_000)
    monkeypatch.setattr(
        aimdo,
        "available_physical_memory_bytes",
        lambda: 1 << 50,
    )
    monkeypatch.setattr(aimdo.torch.cuda, "Stream", stream)
    monkeypatch.setattr(aimdo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(aimdo.torch.cuda, "cudart", lambda: state.cudart)
    monkeypatch.setattr(aimdo.torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(aimdo.torch.cuda, "current_device", lambda: current_index)
    monkeypatch.setattr(aimdo.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(aimdo.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(aimdo.torch.cuda, "Event", lambda: _Event(state.events))
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "current_stream",
        lambda device: state.current_stream_devices.append(device) or current,
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "synchronize",
        lambda _device=None: state.events.append("device-sync"),
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "mem_get_info",
        lambda _device=None: (state.cuda_free, state.cuda_total),
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "memory_allocated",
        lambda _device=None: state.cuda_allocated,
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "memory_reserved",
        lambda _device=None: state.cuda_reserved,
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "empty_cache",
        lambda: state.events.append("empty-cache"),
    )
    monkeypatch.setattr(aimdo, "_AIMDO_INITIALIZED_DEVICES", set())
    original_empty = torch.empty
    monkeypatch.setattr(
        aimdo.torch,
        "empty",
        lambda shape, *, dtype, device=None: original_empty(shape, dtype=dtype),
    )
    return state


def _quantized_values():
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout,
        TensorCoreNVFP4Layout,
    )

    fp8_qdata = torch.arange(16, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(4, 4)
    fp8 = QuantizedTensor(
        fp8_qdata,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=torch.tensor(0.5), orig_dtype=torch.bfloat16, orig_shape=(4, 4)
        ),
    )
    nvfp4 = QuantizedTensor(
        torch.arange(128, dtype=torch.uint8).reshape(16, 8),
        "TensorCoreNVFP4Layout",
        TensorCoreNVFP4Layout.Params(
            scale=torch.tensor(0.25),
            orig_dtype=torch.bfloat16,
            orig_shape=(16, 16),
            block_scale=torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        ),
    )
    return fp8, nvfp4


def _host_source_pool_order_fixture(
    lifetime: host_source_pool.HostSourceLifetime,
):
    events: list[str] = []

    class _Owner:
        def __init__(self, size, *, prewarm, max_grow_size, mark_cold):
            assert size == 0 and mark_cold is True
            assert prewarm == min(16, 8 * 1024 * 1024)
            self.storage = torch.empty(max_grow_size, dtype=torch.uint8)
            self.size = 0
            self._ptr = 1

        def get_raw_address(self):
            return self.storage.data_ptr() if self.size > 0 else 0

        def extend(self, size, *, reallocate, register):
            assert reallocate is False and register is False
            self.size += size
            return self.storage.data_ptr() + self.size - size

        def truncate(self, size, *, do_unregister):
            assert do_unregister is False
            events.append(f"truncate:{size}")
            self.size = size

        def __del__(self):
            if self._ptr:
                events.append("free")
                self._ptr = None

    pool = host_source_pool.AimdoHostSourcePool(
        {(host_source_pool.HostSourceClass.PATCH, lifetime): 16},
        host_buffer_factory=_Owner,
        hostbuf_to_tensor=lambda owner: owner.storage[: owner.size],
        registration_budget_bytes=16,
        available_memory_bytes=lambda: 1 << 50,
        cudart=_Cudart(),
    )
    return pool, events


def test_available_physical_memory_uses_exact_available_not_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host_registration,
        "_windows_memory_status",
        lambda: (128 * 1024**3, 19 * 1024**3),
    )

    assert host_registration.system_memory_bytes() == 128 * 1024**3
    assert host_registration.available_physical_memory_bytes() == 19 * 1024**3


def test_host_source_pool_close_drops_active_lease_views_before_native_cleanup() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", 1),
    )
    view = lease.tensor
    weakref.finalize(view, events.append, "view-drop")
    del view
    pool.add_fence(lease, object())

    pool.close(quiesced=True)

    assert lease.complete is True
    assert lease.tensor.numel() == 0
    assert events == ["view-drop", "truncate:0", "free"]


def test_host_source_pool_temporary_reclaim_drops_view_before_truncate() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
        size=8,
    )
    view = lease.tensor
    weakref.finalize(view, events.append, "view-drop")
    del view

    pool.complete_transfer(lease, quiesced=True)

    assert lease.tensor.numel() == 0
    assert events == ["view-drop", "truncate:0"]
    pool.close(quiesced=True)


def test_host_source_pool_patch_invalidation_drops_cached_view_before_truncate() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", 1),
    )
    view = lease.tensor
    weakref.finalize(view, events.append, "view-drop")
    del view
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)

    pool.invalidate_patch_sources()

    assert lease.tensor.numel() == 0
    assert events == ["view-drop", "truncate:0"]
    pool.close(quiesced=True)


def test_host_source_pool_partial_close_retains_owner_after_view_safe_truncate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", 1),
    )
    view = lease.tensor
    weakref.finalize(view, events.append, "view-drop")
    del view
    owner = pool.owners[0]
    def fail_truncate(_size, *, do_unregister):
        assert do_unregister is False
        events.append("truncate-failed")
        raise RuntimeError("truncate failed")

    monkeypatch.setattr(owner, "truncate", fail_truncate)
    with pytest.raises(
        host_source_pool.HostSourcePoolStructuralError,
        match="host_buffer_close_truncate_failed",
    ):
        pool.close(quiesced=True)

    assert lease.tensor.numel() == 0
    assert events == ["view-drop", "truncate-failed"]
    assert owner._ptr == 1
    assert pool.live is True
    assert pool.diagnostics()["registration_state_proven"] is False
    with pytest.raises(
        host_source_pool.HostSourcePoolStructuralError,
        match="host_buffer_close_truncate_failed",
    ):
        pool.close(quiesced=True)
    assert events == ["view-drop", "truncate-failed"]


def test_host_source_pool_registration_budget_and_exact_lifecycle_accounting() -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", 1),
    )
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)

    live = pool.diagnostics()
    assert live["registration_budget_bytes"] == 16
    assert live["registration_attempts"] == live["registration_successes"] == 1
    assert live["registration_attempt_bytes"] == live["registration_registered_bytes"] == 8
    assert live["registration_live_bytes"] == live["registration_peak_bytes"] == 8
    assert live["registration_unregistered_bytes"] == 0
    assert live["registration_state_proven"] is True

    pool.close(quiesced=True)
    closed = pool.diagnostics()
    assert closed["registration_live_bytes"] == 0
    assert closed["registration_unregistered_bytes"] == 8
    assert closed["registration_registered_bytes"] == 8


def test_host_source_pool_registration_refusal_rolls_back_before_direct_fallback() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    cudart = pool._cudart
    cudart.events = events
    cudart.register_results.append(2)

    with pytest.raises(
        host_source_pool.HostSourceDirectTransferRequired,
        match="cuda_host_registration_refused",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            size=8,
        )

    lane = next(iter(pool._lanes.values()))
    proof = pool.diagnostics()
    assert events == ["cuda-register:8:2", "cuda-get-last-error", "truncate:0"]
    assert lane.owner.size == lane.used == lane.registered_bytes == 0
    assert lane.registrations == []
    assert proof["temporary_registration_refusals"] == 1
    assert proof["registration_attempts"] == proof["registration_failures"] == 1
    assert proof["registration_successes"] == proof["registration_live_bytes"] == 0
    assert proof["registration_state_proven"] is True
    assert proof["poisoned"] is False
    pool.close(quiesced=True)


def test_host_source_pool_success_reclaims_unregister_before_truncate() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    cudart = pool._cudart
    cudart.events = events
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
        size=8,
    )
    view = lease.tensor
    weakref.finalize(view, events.append, "view-drop")
    del view

    pool.complete_transfer(lease, quiesced=True)

    assert events == [
        "cuda-register:8:0",
        "view-drop",
        "cuda-unregister:0",
        "truncate:0",
    ]
    assert pool.diagnostics()["registration_live_bytes"] == 0
    pool.close(quiesced=True)


def test_host_source_pool_success_close_unregisters_before_truncate_and_free() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    cudart = pool._cudart
    cudart.events = events
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", "close"),
    )
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)

    pool.close(quiesced=True)

    assert events == [
        "cuda-register:8:0",
        "cuda-unregister:0",
        "truncate:0",
        "free",
    ]


def test_host_source_pool_unregister_failure_poison_preserves_owner() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    cudart = pool._cudart
    cudart.events = events
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
        size=8,
    )
    cudart.unregister_results.append(3)

    with pytest.raises(
        host_source_pool.HostSourcePoolStructuralError,
        match="host_buffer_reclaim_truncate_failed",
    ):
        pool.complete_transfer(lease, quiesced=True)

    proof = pool.diagnostics()
    assert events == [
        "cuda-register:8:0",
        "cuda-unregister:3",
        "cuda-get-last-error",
    ]
    assert proof["poisoned"] is True
    assert proof["poison_reason"] == "cuda_host_unregister_failed"
    assert proof["registration_state_proven"] is False
    assert proof["registration_live_bytes"] == 8
    assert pool.owners[0]._ptr == 1


def test_host_source_pool_active_fence_blocks_reclaim_and_invalidation() -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    cudart = pool._cudart
    cudart.events = events
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", "active"),
    )
    pool.add_fence(lease, object())

    with pytest.raises(RuntimeError, match="not proven quiescent"):
        pool.complete_transfer(lease, quiesced=False)
    with pytest.raises(RuntimeError, match="active transfers"):
        pool.invalidate_patch_sources()

    assert events == ["cuda-register:8:0"]
    assert pool.diagnostics()["transfer_pending"] is True
    pool.complete_transfer(lease, quiesced=True)
    pool.invalidate_patch_sources()
    assert events[-2:] == ["cuda-unregister:0", "truncate:0"]
    pool.close(quiesced=True)


def test_host_source_pool_uses_comfy_lane_prewarm_arguments_without_logical_growth() -> None:
    calls: list[tuple[int, int, int]] = []

    class _Owner:
        _ptr = 1

        def __init__(self, size, *, prewarm, max_grow_size, mark_cold):
            assert mark_cold is True
            calls.append((size, prewarm, max_grow_size))

        def truncate(self, _size, *, do_unregister):
            assert do_unregister is True

        def __del__(self):
            self._ptr = None

    mib = 1024 * 1024
    pool = host_source_pool.AimdoHostSourcePool(
        {
            (
                host_source_pool.HostSourceClass.BASE,
                host_source_pool.HostSourceLifetime.WARM,
            ): 128 * mib,
            (
                host_source_pool.HostSourceClass.BASE,
                host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            ): 4 * mib,
            (
                host_source_pool.HostSourceClass.PATCH,
                host_source_pool.HostSourceLifetime.WARM,
            ): 16 * mib,
            (
                host_source_pool.HostSourceClass.PATCH,
                host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            ): 2 * mib,
        },
        host_buffer_factory=_Owner,
        hostbuf_to_tensor=lambda _owner: torch.empty(0, dtype=torch.uint8),
        registration_budget_bytes=256 * mib,
    )

    assert calls == [
        (0, 64 * mib, 128 * mib),
        (0, 4 * mib, 4 * mib),
        (0, 8 * mib, 16 * mib),
        (0, 2 * mib, 2 * mib),
    ]
    assert pool.diagnostics()["registration_live_bytes"] == 0
    pool.close(quiesced=True)


def test_host_source_pool_setup_cleanup_skips_native_empty_truncate() -> None:
    events: list[str] = []
    created: list[object] = []

    class _Owner:
        def __init__(self, _size, *, prewarm, max_grow_size, mark_cold):
            assert mark_cold is True
            self.size = 0
            self._ptr = 1
            created.append(self)

        def truncate(self, size, *, do_unregister):
            assert do_unregister is True
            if size == self.size == 0:
                raise RuntimeError("native empty truncate is invalid")
            self.size = size
            events.append(f"truncate:{size}")

        def __del__(self):
            events.append("free")
            self._ptr = None

    calls = 0

    def factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second lane failed")
        return _Owner(*args, **kwargs)

    with pytest.raises(host_source_pool.HostSourcePoolSetupFallback) as raised:
        host_source_pool.AimdoHostSourcePool(
            {
                (host_source_pool.HostSourceClass.BASE, host_source_pool.HostSourceLifetime.WARM): 8,
                (host_source_pool.HostSourceClass.PATCH, host_source_pool.HostSourceLifetime.WARM): 8,
            },
            host_buffer_factory=factory,
            hostbuf_to_tensor=lambda _owner: torch.empty(0, dtype=torch.uint8),
            registration_budget_bytes=16,
        )

    pool = raised.value.pool
    assert len(created) == pool.allocations == pool.frees == 1
    assert pool.unregistrations == 0
    assert events == ["free"]


def test_host_source_pool_patch_invalidation_skips_native_empty_truncate() -> None:
    events: list[str] = []

    class _Owner:
        def __init__(self, _size, *, prewarm, max_grow_size, mark_cold):
            assert mark_cold is True
            self.size = 0
            self._ptr = 1

        def truncate(self, size, *, do_unregister):
            assert do_unregister is True
            if size == self.size == 0:
                raise RuntimeError("native empty truncate is invalid")
            self.size = size
            events.append(f"truncate:{size}")

        def __del__(self):
            events.append("free")
            self._ptr = None

    pool = host_source_pool.AimdoHostSourcePool(
        {
            (host_source_pool.HostSourceClass.PATCH, host_source_pool.HostSourceLifetime.WARM): 8,
            (
                host_source_pool.HostSourceClass.PATCH,
                host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            ): 8,
        },
        host_buffer_factory=_Owner,
        hostbuf_to_tensor=lambda _owner: torch.empty(0, dtype=torch.uint8),
        registration_budget_bytes=16,
    )

    pool.invalidate_patch_sources()

    assert events == []
    assert pool.unregistrations == 0
    pool.close(quiesced=True)
    assert events == ["free", "free"]
    assert pool.frees == 2


def test_host_source_pool_clean_close_skips_native_empty_truncate_and_frees_owner() -> None:
    events: list[str] = []

    class _Owner:
        def __init__(self, _size, *, prewarm, max_grow_size, mark_cold):
            assert mark_cold is True
            self.size = 0
            self._ptr = 1

        def truncate(self, size, *, do_unregister):
            assert do_unregister is True
            if size == self.size == 0:
                raise RuntimeError("native empty truncate is invalid")
            self.size = size
            events.append(f"truncate:{size}")

        def __del__(self):
            events.append("free")
            self._ptr = None

    pool = host_source_pool.AimdoHostSourcePool(
        {(host_source_pool.HostSourceClass.BASE, host_source_pool.HostSourceLifetime.WARM): 8},
        host_buffer_factory=_Owner,
        hostbuf_to_tensor=lambda _owner: torch.empty(0, dtype=torch.uint8),
        registration_budget_bytes=8,
    )

    pool.close(quiesced=True)

    assert events == ["free"]
    assert pool.unregistrations == 0
    assert pool.frees == 1


def test_host_source_pool_accepts_empty_native_region_without_tensor_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lane = next(iter(pool._lanes.values()))
    original_to_tensor = pool._to_tensor
    tensor_calls = 0

    def count_tensor(owner):
        nonlocal tensor_calls
        tensor_calls += 1
        return original_to_tensor(owner)

    monkeypatch.setattr(pool, "_to_tensor", count_tensor)

    assert pool._validated_owner_region(lane) == (0, 0)
    assert tensor_calls == 0

    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", "first-append"),
    )
    assert lease.tensor.numel() == 8
    assert tensor_calls == 1
    pool.complete_transfer(lease, quiesced=True)
    pool.close(quiesced=True)


def test_host_source_pool_rejects_published_address_for_empty_native_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    lane = next(iter(pool._lanes.values()))
    monkeypatch.setattr(lane.owner, "get_raw_address", lambda: 1)

    with pytest.raises(RuntimeError, match="empty region has an invalid raw address"):
        pool._validated_owner_region(lane)

    pool.close(quiesced=True)


def test_host_source_pool_warm_budget_bypass_is_nonterminal_before_extend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    pool.warm_registration_budget_bytes = 4
    owner = pool.owners[0]
    calls = 0
    original_extend = owner.extend

    def count_extend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_extend(*args, **kwargs)

    monkeypatch.setattr(owner, "extend", count_extend)
    with pytest.raises(
        host_source_pool.HostSourceWarmUnavailable,
        match="registration_budget_exhausted",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=8,
            cache_key=("patch", 1),
        )
    assert calls == 0
    proof = pool.diagnostics()
    assert proof["registration_attempts"] == proof["registration_failures"] == 1
    assert proof["registration_failure_bytes"] == 8
    assert proof["registration_live_bytes"] == 0
    assert proof["registration_state_proven"] is True
    assert proof["poisoned"] is False
    assert proof["patch_warm_misses"] == proof["patch_warm_bypasses"] == 1
    pool.warm_registration_budget_bytes = 16
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", 2),
    )
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)
    assert calls == 1
    assert events == []
    pool.close(quiesced=True)


def test_host_source_pool_ram_pressure_preflight_has_exact_two_gib_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    owner = pool.owners[0]
    calls = 0
    original_extend = owner.extend

    def count_extend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_extend(*args, **kwargs)

    monkeypatch.setattr(owner, "extend", count_extend)
    pool._available_memory_bytes = lambda: 2 * 1024**3 + 7
    with pytest.raises(
        host_source_pool.HostSourceWarmUnavailable,
        match="physical_ram_pressure",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=8,
            cache_key=("patch", "pressure"),
        )
    proof = pool.diagnostics()
    assert calls == 0
    assert proof["warm_ram_pressure_bypasses"] == 1
    assert proof["warm_zero_delta_extend_refusals"] == 0
    assert proof["registration_attempts"] == proof["registration_failures"] == 0
    assert proof["poisoned"] is False

    pool._available_memory_bytes = lambda: 2 * 1024**3 + 8
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", "boundary"),
    )
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)
    assert calls == 1
    pool.close(quiesced=True)


def test_host_source_pool_temporary_ram_pressure_requires_direct_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    owner = pool.owners[0]
    extend_calls = 0
    original_extend = owner.extend

    def count_extend(*args, **kwargs):
        nonlocal extend_calls
        extend_calls += 1
        return original_extend(*args, **kwargs)

    monkeypatch.setattr(owner, "extend", count_extend)
    pool._available_memory_bytes = lambda: 2 * 1024**3 + 7

    with pytest.raises(
        host_source_pool.HostSourceDirectTransferRequired,
        match="physical_ram_pressure",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            size=8,
        )

    proof = pool.diagnostics()
    assert extend_calls == 0
    assert proof["temporary_ram_pressure_bypasses"] == 1
    assert proof["warm_ram_pressure_bypasses"] == 0
    assert proof["registration_attempts"] == proof["registration_failures"] == 0
    assert proof["poisoned"] is False


def test_host_source_pool_zero_delta_warm_extend_refusal_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    owner = pool.owners[0]
    original_extend = owner.extend
    monkeypatch.setattr(
        owner,
        "extend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native false")),
    )

    with pytest.raises(
        host_source_pool.HostSourceWarmUnavailable,
        match="native_extend_refused_without_delta",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=8,
            cache_key=("patch", "refused"),
        )
    proof = pool.diagnostics()
    assert proof["warm_zero_delta_extend_refusals"] == 1
    assert proof["registration_attempts"] == proof["registration_failures"] == 1
    assert proof["registration_state_proven"] is True
    assert proof["poisoned"] is False
    assert proof["retained_slices"] == proof["temporary_slices"] == 0

    monkeypatch.setattr(owner, "extend", original_extend)
    lease = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=8,
        cache_key=("patch", "retry"),
    )
    pool.add_fence(lease, object())
    pool.complete_transfer(lease, quiesced=True)
    pool.close(quiesced=True)


@pytest.mark.parametrize(
    "mutation",
    (
        "owner_size",
        "raw_address",
        "lane_used",
        "registered_bytes",
        "registration_live",
        "slice",
        "cache",
        "lease",
        "existing_view",
    ),
)
def test_host_source_pool_warm_refusal_requires_every_zero_delta_proof_gate(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    owner = pool.owners[0]
    lane = next(iter(pool._lanes.values()))
    seed = pool.acquire(
        host_source_pool.HostSourceClass.PATCH,
        host_source_pool.HostSourceLifetime.WARM,
        size=4,
        cache_key=("patch", "seed"),
    )
    pool.add_fence(seed, object())
    pool.complete_transfer(seed, quiesced=True)
    original_address = owner.get_raw_address
    original_to_tensor = pool._to_tensor
    refused = False

    def raw_address():
        return original_address() + (1 if refused and mutation == "raw_address" else 0)

    def existing_view(value):
        if refused and mutation == "existing_view":
            return torch.ones(1, dtype=torch.uint8)
        return original_to_tensor(value)

    def fail_extend(*_args, **_kwargs):
        nonlocal refused
        refused = True
        if mutation == "owner_size":
            owner.size += 1
        elif mutation == "lane_used":
            lane.used += 1
        elif mutation == "registered_bytes":
            lane.registered_bytes += 1
        elif mutation == "registration_live":
            pool.registration_live_bytes += 1
        elif mutation == "slice":
            lane.slices.append(None)
        elif mutation == "cache":
            lane.cache[("unexpected",)] = None
        elif mutation == "lease":
            pool._leases[123] = SimpleNamespace(fences=[])
        raise RuntimeError("native false with changed proof")

    monkeypatch.setattr(owner, "get_raw_address", raw_address)
    monkeypatch.setattr(pool, "_to_tensor", existing_view)
    monkeypatch.setattr(owner, "extend", fail_extend)
    with pytest.raises(
        host_source_pool.HostSourcePoolStructuralError,
        match="host_buffer_extend_failed",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=8,
            cache_key=("patch", mutation),
        )
    assert pool._poisoned is True
    assert pool.registration_state_proven is False


def test_host_source_pool_temporary_extend_refusal_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    owner = pool.owners[0]

    def mutate_then_refuse(*_args, **_kwargs):
        owner.size += 1
        raise RuntimeError("native false")

    monkeypatch.setattr(owner, "extend", mutate_then_refuse)

    with pytest.raises(
        host_source_pool.HostSourcePoolStructuralError,
        match="host_buffer_extend_failed",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            size=8,
        )
    proof = pool.diagnostics()
    assert proof["poisoned"] is True
    assert proof["registration_state_proven"] is False


def test_host_source_pool_temporary_zero_delta_extend_refusal_requires_direct_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY
    )
    owner = pool.owners[0]
    monkeypatch.setattr(
        owner,
        "extend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native false")),
    )

    with pytest.raises(
        host_source_pool.HostSourceDirectTransferRequired,
        match="native_extend_refused_without_delta",
    ):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
            size=8,
        )

    proof = pool.diagnostics()
    assert proof["temporary_zero_delta_extend_refusals"] == 1
    assert proof["registration_state_proven"] is True
    assert proof["poisoned"] is False


@pytest.mark.parametrize("rollback_fails", [False, True], ids=["extend", "rollback"])
def test_host_source_pool_structural_failure_rejects_session_reuse(
    monkeypatch: pytest.MonkeyPatch,
    rollback_fails: bool,
) -> None:
    pool, events = _host_source_pool_order_fixture(
        host_source_pool.HostSourceLifetime.WARM
    )
    owner = pool.owners[0]
    extend_calls = 0

    if rollback_fails:
        original_to_tensor = pool._to_tensor
        view_calls = 0

        def fail_appended_view(value):
            nonlocal view_calls
            view_calls += 1
            if view_calls == 1:
                raise RuntimeError("view failure")
            return original_to_tensor(value)

        monkeypatch.setattr(
            pool,
            "_to_tensor",
            fail_appended_view,
        )

        def fail_rollback(_size, *, do_unregister):
            assert do_unregister is False
            events.append("rollback-failed")
            raise RuntimeError("rollback failure")

        monkeypatch.setattr(owner, "truncate", fail_rollback)
    else:

        def fail_extend(*_args, **_kwargs):
            nonlocal extend_calls
            extend_calls += 1
            owner.size += 1
            raise RuntimeError("extend failure")

        monkeypatch.setattr(owner, "extend", fail_extend)

    expected = (
        "host_buffer_registration_rollback_failed"
        if rollback_fails
        else "host_buffer_extend_failed"
    )
    with pytest.raises(host_source_pool.HostSourcePoolStructuralError, match=expected):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=8,
            cache_key=("patch", 1),
        )
    proof = pool.diagnostics()
    assert proof["poisoned"] is True
    assert proof["registration_state_proven"] is False
    with pytest.raises(host_source_pool.HostSourcePoolStructuralError, match=expected):
        pool.acquire(
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.WARM,
            size=1,
            cache_key=("patch", 2),
        )
    assert extend_calls <= 1


@pytest.mark.parametrize("quantized_index", [0, 1], ids=["fp8", "nvfp4"])
def test_aimdo_rebuilds_kitchen_parameter_sidecars_as_raw_views_before_fill(
    quantized_index: int,
) -> None:
    source = torch.nn.Parameter(_quantized_values()[quantized_index], requires_grad=False)
    layout = aimdo._flatten_value(source)
    raw = torch.full((layout.total_bytes,), 0xA5, dtype=torch.uint8)

    rebuilt = aimdo._rebuild_value(layout, raw)

    assert isinstance(rebuilt, torch.nn.Parameter)
    assert type(rebuilt) is type(source)
    assert rebuilt.requires_grad is False
    rebuilt_names, _rebuilt_context = rebuilt.__tensor_flatten__()
    assert tuple(rebuilt_names) == layout.names
    assert layout.names == (
        ("_qdata", "_param_scale")
        if quantized_index == 0
        else ("_qdata", "_param_scale", "_param_block_scale")
    )
    for name, physical in zip(layout.names, layout.physical, strict=True):
        field = getattr(rebuilt, name)
        pointer = field.data_ptr()
        assert pointer == raw.data_ptr() + physical.offset
        assert torch.all(field.reshape(-1).view(torch.uint8) == 0xA5)

        source_bytes = physical.tensor.reshape(-1).view(torch.uint8)
        raw[physical.offset : physical.offset + physical.size].copy_(source_bytes)
        assert field.data_ptr() == pointer
        assert torch.equal(field.reshape(-1).view(torch.uint8), source_bytes)


def test_aimdo_keeps_ordinary_tensor_parameter_construction() -> None:
    source = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    layout = aimdo._flatten_value(source)
    raw = torch.empty(layout.total_bytes, dtype=torch.uint8)
    raw[: layout.physical[0].size].copy_(source.reshape(-1).view(torch.uint8))

    rebuilt = aimdo._rebuild_value(layout, raw)

    assert type(rebuilt) is torch.nn.Parameter
    assert rebuilt.requires_grad is False
    assert rebuilt.data_ptr() == raw.data_ptr()
    assert torch.equal(rebuilt, source)


def test_aimdo_import_init_order_and_flattened_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _backend_fixture(monkeypatch)
    dense = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    fp8, nvfp4 = _quantized_values()
    values = (dense, dense, fp8, nvfp4)
    required = aimdo.AimdoDynamicResidency.group_bytes(values)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=required + 1024)
    backend.allocate_group("text", values)
    backend.prioritize()

    lease = backend.acquire("text")
    rebuilt_dense, rebuilt_alias, rebuilt_fp8, rebuilt_nvfp4 = lease.values
    assert rebuilt_dense is rebuilt_alias
    assert torch.equal(rebuilt_dense, dense)
    assert rebuilt_fp8._layout_cls == fp8._layout_cls
    assert torch.equal(rebuilt_fp8._qdata.view(torch.uint8), fp8._qdata.view(torch.uint8))
    assert torch.equal(rebuilt_fp8.params.scale, fp8.params.scale)
    assert rebuilt_nvfp4._layout_cls == nvfp4._layout_cls
    assert torch.equal(rebuilt_nvfp4._qdata, nvfp4._qdata)
    assert torch.equal(rebuilt_nvfp4.params.block_scale, nvfp4.params.block_scale)
    assert state.events[:6] == [
        "import:comfy_aimdo.control",
        "init:headroom=None:nvml=True",
        "init_devices:[(0, 0)]",
        "import:comfy_aimdo.model_vbar",
        "import:comfy_aimdo.torch",
        "import:comfy_aimdo.host_buffer",
    ]
    assert all(source.offset % 1024 == 0 for source in backend._groups["text"].layouts[0].physical)

    backend.release(lease)
    backend.close()
    proof = backend.diagnostics()
    assert proof["allocation_count"] == 1
    assert proof["live_allocations"] == proof["live_bytes"] == proof["loaded_bytes"] == 0
    assert proof["free_calls"] == 1
    assert proof["copy_strategy"] == "per_physical"
    assert proof["copy_fallback_reason"].startswith(
        "host_buffer_capability_unavailable: ImportError"
    )
    assert proof["per_physical_misses"] == proof["signature_misses"] == 1
    assert proof["gathered_misses"] == proof["gathered_h2d_bytes"] == 0


def test_aimdo_gathered_host_buffer_packs_real_kitchen_layout_once_per_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, host_buffer=True)
    fp8, nvfp4 = (torch.nn.Parameter(value, requires_grad=False) for value in _quantized_values())
    values = (fp8, fp8, nvfp4)
    small = (torch.arange(8, dtype=torch.uint8),)
    required = aimdo.AimdoDynamicResidency.group_bytes(
        values
    ) + aimdo.AimdoDynamicResidency.group_bytes(small)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=required)
    backend.allocate_group("quantized", values)
    backend.allocate_group("small", small)
    backend.prioritize()

    # Wholly CPU leaves use a retained patch lane plus a fault(None)
    # temporary lane. Both reserve a stable maximum but append lazily.
    assert len(state.host_buffers) == 2
    assert all(item.backing.numel() == 0 for item in state.host_buffers)
    lease = backend.acquire("quantized")
    backend.synchronize(lease)
    rebuilt_fp8, rebuilt_alias, rebuilt_nvfp4 = lease.values
    assert rebuilt_fp8 is rebuilt_alias
    for source, rebuilt in ((fp8, rebuilt_fp8), (nvfp4, rebuilt_nvfp4)):
        source_names, _ = source.__tensor_flatten__()
        rebuilt_names, _ = rebuilt.__tensor_flatten__()
        assert rebuilt_names == source_names
        for name in source_names:
            assert torch.equal(
                getattr(rebuilt, name).reshape(-1).view(torch.uint8),
                getattr(source, name).reshape(-1).view(torch.uint8),
            )
    backend.release(lease)
    after_miss = backend.diagnostics()
    assert after_miss["copy_strategy"] == "gathered_host_buffer"
    assert after_miss["copy_fallback_reason"] is None
    assert after_miss["host_buffer_capacity_bytes"] == required
    assert after_miss["host_buffer_allocations"] == 2
    assert after_miss["host_buffer_live"] is True
    assert after_miss["host_tensor_view_live"] is True
    assert after_miss["host_buffer_transfer_pending"] is False
    assert after_miss["gathered_misses"] == 1
    assert after_miss["per_physical_misses"] == 0
    assert after_miss["transfer_events"] == after_miss["transfer_waits"] == 1
    assert after_miss["gathered_h2d_bytes"] == backend._groups["quantized"].staged_bytes
    assert after_miss["pinned_copy_bytes"] == after_miss["gathered_h2d_bytes"]
    assert after_miss["pageable_copy_bytes"] == 0
    assert after_miss["host_source_pool_retained_slices"] == 1
    assert after_miss["host_source_pool_misses"] == 1

    hit = backend.acquire("quantized")
    backend.release(hit)
    after_hit = backend.diagnostics()
    for field in (
        "transfer_events",
        "transfer_waits",
        "gathered_misses",
        "packed_source_bytes",
        "gathered_h2d_bytes",
    ):
        assert after_hit[field] == after_miss[field]
    # VBAR signature hits do not acquire or touch a host source slice.
    assert after_hit["host_source_pool_hits"] == after_miss["host_source_pool_hits"]

    backend.invalidate(reason="diagnostic_force_miss")
    forced = backend.acquire("quantized")
    backend.release(forced)
    forced_proof = backend.diagnostics()
    assert forced_proof["gathered_misses"] == 2
    assert forced_proof["host_source_pool_misses"] == 2
    assert forced_proof["host_source_pool_hits"] == 0
    assert len(state.host_buffers) == 2
    backend.close()
    closed = backend.diagnostics()
    assert closed["host_buffer_live"] is False
    assert closed["host_tensor_view_live"] is False
    assert closed["host_buffer_transfer_pending"] is False
    assert closed["host_buffer_unregistrations"] == 1
    assert closed["host_buffer_frees"] == 2
    close_sync = max(index for index, event in enumerate(state.events) if event == "device-sync")
    close_unregister = next(
        index
        for index, event in enumerate(state.events[close_sync + 1 :], close_sync + 1)
        if event.startswith("host-unregister:")
    )
    assert close_sync < close_unregister < state.events.index("host-free")
    assert state.events.index("host-free") < state.events.index("free")


def test_aimdo_host_buffer_view_failure_rolls_back_then_uses_direct_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(16, dtype=torch.uint8)
    _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("layer", (source,))
    backend._aimdo_torch.hostbuf_to_tensor = lambda _value: (_ for _ in ()).throw(
        RuntimeError("view failed")
    )
    backend.prioritize()

    lease = backend.acquire("layer")
    assert torch.equal(lease.values[0], source)
    proof = backend.diagnostics()
    assert proof["copy_strategy"] == "gathered_host_buffer"
    assert proof["copy_fallback_reason"] is None
    assert proof["host_buffer_allocations"] == 2
    assert proof["host_buffer_live"] is True
    assert proof["host_tensor_view_live"] is False
    assert proof["per_physical_misses"] == 0
    assert proof["host_source_pool_poisoned"] is False
    assert proof["host_source_pool_poison_reason"] is None
    assert proof["host_source_pool_warm_registration_refusals"] == 1
    assert proof["pressure_direct_transfers"] == 1
    assert proof["pageable_copy_bytes"] == 16
    assert proof["poison_reason"] is None
    backend.release(lease)
    backend.close()


@pytest.mark.parametrize("failure_site", ("truncate", "free"))
def test_aimdo_pool_close_failure_retains_vbar_file_and_remaining_lanes_for_hard_exit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    path = tmp_path / "base-close-poison.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("base", (descriptor,))
    backend.allocate_group("patch", (torch.arange(4, dtype=torch.uint8),))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    pool = backend._host_source_pool
    owners = pool.owners
    assert len(owners) == 4

    if failure_site == "truncate":
        owners[1].size = 1

        def fail_second_lane(_size, *, do_unregister):
            assert do_unregister is True
            raise RuntimeError("second lane truncate failed")

        monkeypatch.setattr(owners[1], "truncate", fail_second_lane)
    else:

        def fail_second_free():
            raise RuntimeError("second lane free failed")

        monkeypatch.setattr(owners[1], "__del__", fail_second_free)
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_structural_failure",
    ):
        backend.close()

    proof = backend.diagnostics()
    assert proof["poisoned"] is True
    assert proof["close_failed"] is True
    assert proof["host_source_pool_poisoned"] is True
    assert proof["host_buffer_unregistrations"] == 0
    assert proof["host_buffer_frees"] == 1
    assert backend._host_source_pool is pool
    assert backend._file_sources == {"base": handle}
    assert backend._vbar._ptr == 99
    assert state.frees == []
    events_before_retry = list(state.events)
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_structural_failure",
    ):
        backend.close()
    assert state.events == events_before_retry
    assert handle.closed is False
    handle.close()


@pytest.mark.parametrize(
    ("failed_lane", "cleanup_site"),
    ((2, "truncate"), (3, "free")),
)
def test_aimdo_partial_pool_setup_cleanup_failure_is_terminal_and_retained(
    monkeypatch: pytest.MonkeyPatch,
    failed_lane: int,
    cleanup_site: str,
) -> None:
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("base", (descriptor,))
    backend.allocate_group("patch", (torch.arange(4, dtype=torch.uint8),))
    original_factory = backend._host_buffer_module.HostBuffer
    calls = 0
    retained: list[object] = []

    def factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_lane:
            raise RuntimeError("later lane construction failed")
        owner = original_factory(*args, **kwargs)
        retained.append(owner)
        if len(retained) == 1:
            if cleanup_site == "truncate":
                owner.size = 1

                def fail_truncate(_size, *, do_unregister):
                    assert do_unregister is True
                    raise RuntimeError("setup truncate failed")

                owner.truncate = fail_truncate
            else:

                def fail_free():
                    raise RuntimeError("setup free failed")

                owner.__del__ = fail_free
        return owner

    monkeypatch.setattr(backend._host_buffer_module, "HostBuffer", factory)
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_setup_cleanup_failed",
    ):
        backend.prioritize()

    pool = backend._host_source_pool
    assert pool is not None
    assert pool.owners[0] is retained[0]
    proof = backend.diagnostics()
    assert proof["copy_strategy"] == "per_physical"
    assert proof["copy_fallback_reason"] is None
    assert proof["host_source_pool_poisoned"] is True
    assert proof["host_source_pool_poison_reason"] == "host_buffer_setup_cleanup_failed"
    assert proof["poison_reason"] == "host_source_pool_setup_cleanup_failed"
    assert backend._vbar._ptr == 99
    assert state.frees == []
    events_before_close = list(state.events)
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_setup_cleanup_failed",
    ):
        backend.close()
    assert state.events == events_before_close


def test_aimdo_safe_partial_pool_setup_fallback_preserves_exact_cleanup_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("base", (descriptor,))
    backend.allocate_group("patch", (torch.arange(4, dtype=torch.uint8),))
    original_factory = backend._host_buffer_module.HostBuffer
    calls = 0

    def factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("third lane unavailable")
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(backend._host_buffer_module, "HostBuffer", factory)
    backend.prioritize()

    proof = backend.diagnostics()
    assert proof["copy_strategy"] == "per_physical"
    assert proof["copy_fallback_reason"].startswith(
        "host_buffer_setup_failed: RuntimeError: third lane unavailable"
    )
    assert proof["host_buffer_allocations"] == 2
    assert proof["host_buffer_unregistrations"] == 0
    assert proof["host_buffer_frees"] == 2
    assert proof["host_buffer_live"] is False
    assert proof["host_source_pool_generation"] == 2
    assert proof["host_source_pool_lane_count"] == 2
    assert proof["host_source_pool_poisoned"] is False
    assert proof["host_source_registration"] == {
        "policy": "aimdo_hostbuffer_registered_append",
        "budget_bytes": 1_000_000,
        "attempts": 0,
        "attempt_bytes": 0,
        "successes": 0,
        "failures": 0,
        "failure_bytes": 0,
        "registered_bytes": 0,
        "unregistered_bytes": 0,
        "live_bytes": 0,
        "peak_bytes": 0,
        "state_proven": True,
    }
    assert backend._host_source_pool is None
    backend.close()


def test_aimdo_reclaim_failure_poison_retains_active_lease_and_rejects_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("temporary", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()
    lease = backend.acquire("temporary")
    pool = backend._host_source_pool
    owner = lease.token.source_lease.source
    lane_owner = next(
        item
        for item in pool.owners
        if item.get_raw_address() == owner.view.data_ptr() - owner.offset
    )

    def fail_truncate(_size, *, do_unregister):
        assert do_unregister is True
        raise RuntimeError("temporary reclaim truncate failed")

    monkeypatch.setattr(lane_owner, "truncate", fail_truncate)
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_structural_failure",
    ):
        backend.release(lease)
    assert backend.terminal_poison_reason() == "host_source_pool_structural_failure"
    assert backend._active == {}
    assert backend._retirements[0].tokens == (lease.token,)
    assert lease.token.source_lease is not None
    with pytest.raises(
        aimdo.DynamicResidencyPoisoned,
        match="host_source_pool_structural_failure",
    ):
        backend.acquire("temporary")


def test_aimdo_source_pool_queues_overlapping_group_transfers_without_repack_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, host_buffer=True)
    first_source = torch.arange(16, dtype=torch.uint8)
    second_source = torch.arange(32, dtype=torch.uint8) + 64
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("root", (first_source,))
    backend.allocate_group("layer", (second_source,))
    backend.prioritize()

    root = backend.acquire("root")
    assert backend.diagnostics()["host_buffer_transfer_pending"] is True
    first_record = state.events.index("record:copy-0")
    layer = backend.acquire("layer")
    second_record = state.events.index("record:copy-1", first_record + 1)

    assert first_record < second_record
    assert "event-sync:copy-0" not in state.events[first_record + 1 : second_record]
    assert torch.equal(layer.values[0], second_source)
    proof = backend.diagnostics()
    assert proof["gathered_misses"] == 2
    assert proof["host_buffer_reuse_barriers"] == 0
    assert proof["host_buffer_transfer_pending"] is True

    backend.release(layer)
    assert backend.diagnostics()["host_buffer_transfer_pending"] is True
    backend.release(root)
    assert backend.diagnostics()["host_buffer_transfer_pending"] is False
    backend.close()
    assert backend.diagnostics()["host_buffer_reuse_barriers"] == 0


def test_aimdo_gathered_host_buffer_mixes_file_spans_cpu_values_and_aliases(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(64))
    path = tmp_path / "base.bin"
    path.write_bytes(payload)
    dense_template = torch.empty((4,), dtype=torch.uint8, device="meta")
    descriptor = aimdo.AimdoFileBackedValue(
        dense_template,
        (aimdo.AimdoFileSpan("base", "dense", 8, 4, torch.uint8, (4,)),),
    )
    cpu = torch.tensor([91, 92], dtype=torch.uint8)
    state = _backend_fixture(monkeypatch, host_buffer=True)
    values = (descriptor, descriptor, cpu)
    required = aimdo.AimdoDynamicResidency.group_bytes(values)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=required)
    backend.allocate_group("mixed", values)
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    lease = backend.acquire("mixed")
    backend.synchronize(lease)
    assert lease.values[0] is lease.values[1]
    assert torch.equal(lease.values[0], torch.tensor([8, 9, 10, 11], dtype=torch.uint8))
    assert torch.equal(lease.values[2], cpu)
    backend.release(lease)
    proof = backend.diagnostics()
    assert proof["base_file_backed"] is True
    assert proof["base_file_read_calls"] == 1
    assert proof["base_file_read_bytes"] == 4
    assert proof["transfer_events"] == proof["transfer_waits"] == 1
    assert proof["host_source_pool_retained_slices"] == 0
    assert proof["host_source_pool_temporary_slices"] == 0
    backend.close()
    closed_proof = backend.diagnostics()
    assert closed_proof["base_file_backed"] is True
    assert closed_proof["base_file_source_live"] is False
    assert closed_proof["live_allocations"] == closed_proof["live_bytes"] == 0
    handle.close()
    assert state.events.count("host-read:8:4:0") == 1


def test_aimdo_file_base_and_cpu_patch_topology_owns_four_logical_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("base", (descriptor,))
    backend.allocate_group("patch", (torch.arange(4, dtype=torch.uint8),))
    backend.prioritize()

    proof = backend.diagnostics()
    assert proof["host_source_pool_lane_count"] == 4
    assert proof["host_buffer_allocations"] == 4
    assert proof["host_source_pool_capacity_bytes"] == 4096
    backend.close()


def test_aimdo_retained_file_source_is_read_once_across_nonresident_misses(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(32))
    path = tmp_path / "base-retained.bin"
    path.write_bytes(payload)
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((8,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 4, 8, torch.uint8, (8,)),),
    )
    state = _backend_fixture(
        monkeypatch,
        host_buffer=True,
        faults=[(1,), (2,)],
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    first = backend.acquire("file")
    backend.release(first)
    second = backend.acquire("file")
    assert torch.equal(second.values[0], torch.tensor(list(payload[4:12]), dtype=torch.uint8))
    backend.release(second)

    proof = backend.diagnostics()
    assert proof["signature_misses"] == proof["gathered_misses"] == 2
    assert proof["base_file_read_calls"] == 1
    assert proof["base_file_read_bytes"] == 8
    assert proof["pressure_direct_transfers"] == 0
    assert proof["pressure_direct_bytes"] == 0
    assert proof["host_source_pool_misses"] == 1
    assert proof["host_source_pool_hits"] == 1
    assert proof["host_source_pool_retained_slices"] == 1
    assert state.events.count("host-read:4:8:0") == 1
    backend.close()
    handle.close()


def test_aimdo_fault_none_reuses_explicitly_immutable_file_source(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(32))
    path = tmp_path / "base-fault-none.bin"
    path.write_bytes(payload)
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((8,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 4, 8, torch.uint8, (8,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None, None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    first = backend.acquire("file")
    assert first.token.temporary is not None
    backend.release(first)
    second = backend.acquire("file")
    assert second.token.temporary is not None
    assert torch.equal(second.values[0], torch.tensor(list(payload[4:12]), dtype=torch.uint8))
    backend.release(second)

    proof = backend.diagnostics()
    pool_proof = backend._host_source_pool.diagnostics()
    assert proof["fault_none_temporaries"] == 2
    assert proof["host_source_pool_misses"] == proof["host_source_pool_hits"] == 1
    assert proof["host_source_pool_retained_slices"] == 1
    assert proof["host_source_pool_temporary_slices"] == 0
    assert pool_proof["warm_source_misses"] == pool_proof["base_warm_misses"] == 1
    assert pool_proof["warm_source_hits"] == pool_proof["base_warm_hits"] == 1
    assert pool_proof["warm_source_bypasses"] == 0
    assert state.events.count("host-read:4:8:0") == 1
    backend.close()
    handle.close()


def test_aimdo_fault_none_bare_patch_never_enters_warm_source_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None, None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("bare", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    first = backend.acquire("bare")
    backend.release(first)
    second = backend.acquire("bare")
    backend.release(second)
    proof = backend._host_source_pool.diagnostics()
    assert proof["patch_warm_misses"] == proof["patch_warm_hits"] == 0
    assert proof["patch_warm_bypasses"] == 0
    assert proof["source_misses"] == 2
    assert proof["retained_slices"] == proof["temporary_slices"] == 0
    assert backend.diagnostics()["fault_none_temporaries"] == 2
    backend.close()


def test_aimdo_fault_none_warm_capacity_bypasses_to_temporary_without_poison(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-capacity-bypass.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    pool = backend._host_source_pool
    warm_lane = pool._lanes[
        (host_source_pool.HostSourceClass.BASE, host_source_pool.HostSourceLifetime.WARM)
    ]
    warm_lane.capacity = 0

    lease = backend.acquire("file")
    during = pool.diagnostics()
    assert during["base_warm_misses"] == during["base_warm_bypasses"] == 1
    assert during["temporary_slices"] == 1
    assert during["poisoned"] is False
    backend.release(lease)
    assert pool.diagnostics()["temporary_slices"] == 0
    assert backend.terminal_poison_reason() is None
    backend.close()
    handle.close()


def test_aimdo_fault_none_warm_budget_bypasses_with_reserved_temporary_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-budget-bypass.bin"
    path.write_bytes(bytes(range(16)))
    first_value = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "first", 0, 4, torch.uint8, (4,)),),
    )
    second_value = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "second", 4, 4, torch.uint8, (4,)),),
    )
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None, None])
    monkeypatch.setattr(aimdo, "default_host_registration_budget_bytes", lambda: 2048)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("first", (first_value,))
    backend.allocate_group("second", (second_value,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    first = backend.acquire("first")
    backend.release(first)
    second = backend.acquire("second")
    pool = backend._host_source_pool
    during = pool.diagnostics()
    assert during["temporary_reserve_bytes"] == 1024
    assert during["warm_registration_budget_bytes"] == 1024
    assert during["base_warm_misses"] == 2
    assert during["base_warm_bypasses"] == 1
    assert during["registration_failures"] == 1
    assert during["registration_successes"] == 2
    assert during["registration_live_bytes"] == 2048
    assert during["poisoned"] is False
    backend.release(second)
    after = pool.diagnostics()
    assert after["registration_live_bytes"] == 1024
    assert after["retained_slices"] == 1
    assert after["temporary_slices"] == 0
    assert backend.terminal_poison_reason() is None
    backend.close()
    handle.close()


def test_aimdo_fault_none_ram_pressure_bypasses_hostbuffer_via_direct_file_reader(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(32))
    path = tmp_path / "base-pressure-direct.bin"
    path.write_bytes(payload)
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((8,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 4, 8, torch.uint8, (8,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    monkeypatch.setattr(
        aimdo,
        "available_physical_memory_bytes",
        lambda: 2 * 1024**3 + 1023,
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    lease = backend.acquire("file")
    assert torch.equal(lease.values[0], torch.tensor(list(payload[4:12]), dtype=torch.uint8))
    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert pool_proof["warm_ram_pressure_bypasses"] == 1
    assert pool_proof["warm_zero_delta_extend_refusals"] == 0
    assert pool_proof["retained_slices"] == pool_proof["temporary_slices"] == 0
    assert pool_proof["registration_attempts"] == 0
    assert len(state.direct_file_reads) == 1
    assert not any(event.startswith("host-extend:") for event in state.events)
    assert proof["base_file_read_calls"] == 1
    assert proof["base_file_read_bytes"] == 8
    assert proof["pressure_direct_transfers"] == 1
    assert proof["pressure_direct_bytes"] == 1024
    assert proof["transfer_events"] == 1
    assert len(lease.token.pending_events) == 1
    backend.release(lease)
    assert backend.terminal_poison_reason() is None
    backend.close()
    handle.close()


def test_aimdo_zero_delta_warm_refusal_uses_direct_file_reader_not_temp_lane(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-zero-delta.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    pool = backend._host_source_pool
    warm_owner = pool._lanes[
        (host_source_pool.HostSourceClass.BASE, host_source_pool.HostSourceLifetime.WARM)
    ].owner
    monkeypatch.setattr(
        warm_owner,
        "extend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native false")),
    )

    lease = backend.acquire("file")
    during = pool.diagnostics()
    assert during["warm_zero_delta_extend_refusals"] == 1
    assert during["temporary_slices"] == 0
    assert during["registration_state_proven"] is True
    assert during["poisoned"] is False
    assert len(state.direct_file_reads) == 1
    assert not any(event.startswith("host-extend:") for event in state.events)
    backend.release(lease)
    backend.close()
    handle.close()


def test_aimdo_cuda_registration_refusal_rolls_back_then_uses_direct_reader(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-registration-refused.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    state.cudart.register_results.append(2)

    lease = backend.acquire("file")

    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert torch.equal(lease.values[0], torch.tensor([0, 1, 2, 3], dtype=torch.uint8))
    assert pool_proof["warm_registration_refusals"] == 1
    assert pool_proof["temporary_registration_refusals"] == 0
    assert pool_proof["registration_failures"] == 1
    assert pool_proof["registration_live_bytes"] == 0
    assert pool_proof["retained_slices"] == pool_proof["temporary_slices"] == 0
    assert pool_proof["registration_state_proven"] is True
    assert proof["pressure_direct_transfers"] == 1
    assert proof["host_source_pool_warm_registration_refusals"] == 1
    assert len(state.direct_file_reads) == 1
    backend.release(lease)
    backend.close()
    handle.close()


def test_aimdo_ram_pressure_uses_direct_pageable_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(16, dtype=torch.uint8)
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[(1,)])
    monkeypatch.setattr(
        aimdo,
        "available_physical_memory_bytes",
        lambda: 2 * 1024**3,
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (source,))
    backend.prioritize()

    lease = backend.acquire("patch")
    assert torch.equal(lease.values[0], source)
    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert pool_proof["warm_ram_pressure_bypasses"] == 1
    assert pool_proof["temporary_slices"] == 0
    assert proof["pageable_copy_bytes"] == 16
    assert proof["base_file_read_calls"] == 0
    assert proof["pressure_direct_transfers"] == 1
    assert proof["pressure_direct_bytes"] == 1024
    assert state.direct_file_reads == []
    assert len(lease.token.pending_events) == 1
    backend.release(lease)
    backend.close()


def test_aimdo_temporary_ram_pressure_uses_direct_pageable_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(16, dtype=torch.uint8)
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    monkeypatch.setattr(aimdo, "available_physical_memory_bytes", lambda: 2 * 1024**3)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (source,))
    backend.prioritize()

    lease = backend.acquire("patch")
    assert torch.equal(lease.values[0], source)
    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert pool_proof["temporary_ram_pressure_bypasses"] == 1
    assert pool_proof["warm_ram_pressure_bypasses"] == 0
    assert pool_proof["temporary_slices"] == 0
    assert pool_proof["registration_attempts"] == 0
    assert proof["pageable_copy_bytes"] == 16
    assert proof["pressure_direct_transfers"] == 1
    assert proof["pressure_direct_bytes"] == 1024
    assert proof["host_source_pool_temporary_ram_pressure_bypasses"] == 1
    assert state.direct_file_reads == []
    assert not any(event.startswith("host-extend:") for event in state.events)
    backend.release(lease)
    assert backend.terminal_poison_reason() is None
    backend.close()


def test_aimdo_temporary_ram_pressure_direct_copies_mixed_file_and_cpu_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(32))
    path = tmp_path / "mixed-pressure-direct.bin"
    path.write_bytes(payload)
    file_value = aimdo.AimdoFileBackedValue(
        torch.empty((8,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 4, 8, torch.uint8, (8,)),),
    )
    cpu_value = torch.arange(4, dtype=torch.uint8)
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    monkeypatch.setattr(aimdo, "available_physical_memory_bytes", lambda: 2 * 1024**3)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("mixed", (file_value, cpu_value))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)

    lease = backend.acquire("mixed")
    assert torch.equal(
        lease.values[0], torch.tensor(list(payload[4:12]), dtype=torch.uint8)
    )
    assert torch.equal(lease.values[1], cpu_value)
    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert pool_proof["temporary_ram_pressure_bypasses"] == 1
    assert pool_proof["registration_attempts"] == 0
    assert len(state.direct_file_reads) == 1
    assert proof["base_file_read_calls"] == 1
    assert proof["pageable_copy_bytes"] == 4
    assert proof["pressure_direct_transfers"] == 1
    assert proof["host_source_pool_temporary_ram_pressure_bypasses"] == 1
    assert not any(event.startswith("host-extend:") for event in state.events)
    backend.release(lease)
    assert backend.terminal_poison_reason() is None
    backend.close()
    handle.close()


def test_aimdo_temporary_zero_delta_refusal_direct_copies_mixed_file_and_cpu_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(32))
    path = tmp_path / "mixed-zero-delta-direct.bin"
    path.write_bytes(payload)
    file_value = aimdo.AimdoFileBackedValue(
        torch.empty((8,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 4, 8, torch.uint8, (8,)),),
    )
    cpu_value = torch.arange(4, dtype=torch.uint8)
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("mixed", (file_value, cpu_value))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    temporary_owner = backend._host_source_pool._lanes[
        (
            host_source_pool.HostSourceClass.PATCH,
            host_source_pool.HostSourceLifetime.PREFETCH_TEMPORARY,
        )
    ].owner
    monkeypatch.setattr(
        temporary_owner,
        "extend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native false")),
    )

    lease = backend.acquire("mixed")
    assert torch.equal(
        lease.values[0], torch.tensor(list(payload[4:12]), dtype=torch.uint8)
    )
    assert torch.equal(lease.values[1], cpu_value)
    pool_proof = backend._host_source_pool.diagnostics()
    proof = backend.diagnostics()
    assert pool_proof["temporary_zero_delta_extend_refusals"] == 1
    assert pool_proof["poisoned"] is False
    assert len(state.direct_file_reads) == 1
    assert proof["base_file_read_calls"] == 1
    assert proof["pageable_copy_bytes"] == 4
    assert proof["pressure_direct_transfers"] == 1
    assert proof["host_source_pool_temporary_zero_delta_extend_refusals"] == 1
    backend.release(lease)
    assert backend.terminal_poison_reason() is None
    backend.close()
    handle.close()


def test_aimdo_pressure_bypass_without_direct_file_capability_fails_nonpoisoned(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-pressure-no-reader.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 0, 4, torch.uint8, (4,)),),
    )
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    monkeypatch.setattr(
        aimdo,
        "available_physical_memory_bytes",
        lambda: 2 * 1024**3,
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    monkeypatch.delattr(backend._host_buffer_module, "read_file_to_device")

    with pytest.raises(RuntimeError, match="direct file reader is unavailable"):
        backend.acquire("file")
    assert backend.terminal_poison_reason() is None
    assert backend._host_source_pool.diagnostics()["poisoned"] is False
    backend.close()
    handle.close()


def test_aimdo_failed_file_fill_is_not_published_and_retry_rereads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base-provisional.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "weight", 2, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    owner = state.host_buffers[0]
    original_read = owner.read_file_slice
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("provisional read failed")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(owner, "read_file_slice", fail_once)
    with pytest.raises(RuntimeError, match="provisional read failed"):
        backend.acquire("file")

    failed = backend.diagnostics()
    assert failed["host_source_pool_retained_slices"] == 0
    assert failed["host_source_pool_temporary_slices"] == 0
    assert failed["host_source_pool_hits"] == 0
    assert failed["host_source_pool_misses"] == 1

    retry = backend.acquire("file")
    assert torch.equal(retry.values[0], torch.tensor([2, 3, 4, 5], dtype=torch.uint8))
    backend.release(retry)
    retried = backend.diagnostics()
    assert retried["host_source_pool_hits"] == 0
    assert retried["host_source_pool_misses"] == 2
    assert retried["base_file_read_calls"] == 1
    backend.close()
    handle.close()


@pytest.mark.parametrize("failure_site", ("copy", "event", "record", "add_fence"))
def test_aimdo_failed_warm_submission_is_not_published_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    if failure_site == "copy":
        original = torch.Tensor.copy_
        calls = 0

        def fail_once(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provisional copy failed")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "copy_", fail_once)
    elif failure_site == "event":
        original = aimdo.torch.cuda.Event
        calls = 0

        def fail_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provisional event failed")
            return original()

        monkeypatch.setattr(aimdo.torch.cuda, "Event", fail_once)
    elif failure_site == "record":
        original = _Event.record
        calls = 0

        def fail_once(self, stream):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provisional record failed")
            return original(self, stream)

        monkeypatch.setattr(_Event, "record", fail_once)
    else:
        pool = backend._host_source_pool
        original = pool.add_fence
        calls = 0

        def fail_once(lease, event):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provisional add_fence failed")
            return original(lease, event)

        monkeypatch.setattr(pool, "add_fence", fail_once)

    with pytest.raises(RuntimeError, match=f"provisional {failure_site} failed"):
        backend.acquire("patch")
    failed = backend.diagnostics()
    assert failed["host_source_pool_retained_slices"] == 0
    assert failed["host_source_pool_temporary_slices"] == 0
    assert failed["host_source_pool_hits"] == 0
    assert failed["host_source_pool_misses"] == 1

    retry = backend.acquire("patch")
    backend.release(retry)
    retried = backend.diagnostics()
    assert retried["host_source_pool_hits"] == 0
    assert retried["host_source_pool_misses"] == 2
    assert retried["host_source_pool_retained_slices"] == 1
    backend.close()


def test_aimdo_fault_none_uses_reclaimable_temporary_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("temporary", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    lease = backend.acquire("temporary")
    during = backend.diagnostics()
    assert during["fault_none_temporaries"] == 1
    assert during["host_source_pool_temporary_slices"] == 1
    assert during["host_source_pool_retained_slices"] == 0
    backend.release(lease)

    after = backend.diagnostics()
    assert after["host_source_pool_temporary_slices"] == 0
    assert after["host_source_pool_temporary_bytes"] == 0
    backend.close()


def test_aimdo_patch_invalidation_rejects_stale_source_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True, faults=[(1,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    lease = backend.acquire("patch")
    stale_source = lease.token.source_lease.source
    backend.release(lease)
    backend.invalidate(reason="patch_changed")
    stale = host_source_pool.HostSourceLease(stale_source, needs_fill=False)

    with pytest.raises(RuntimeError, match="generation is stale"):
        backend._host_source_pool.add_fence(stale, object())
    assert backend.diagnostics()["host_source_pool_stale_rejections"] == 1
    backend.close()


def test_aimdo_signature_bound_cpu_patch_source_reuses_only_within_patch_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, host_buffer=True, faults=[(1,), (2,), (3,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    first = backend.acquire("patch")
    backend.release(first)
    nonresident = backend.acquire("patch")
    backend.release(nonresident)
    warm = backend.diagnostics()
    assert warm["signature_misses"] == 2
    assert warm["host_source_pool_misses"] == 1
    assert warm["host_source_pool_hits"] == 1
    warm_pool = backend._host_source_pool.diagnostics()
    assert warm_pool["patch_warm_misses"] == 1
    assert warm_pool["patch_warm_hits"] == 1
    assert warm_pool["patch_warm_bypasses"] == 0

    backend.invalidate(reason="patch_changed")
    changed = backend.acquire("patch")
    backend.release(changed)
    after = backend.diagnostics()
    assert after["host_source_pool_misses"] == 2
    assert after["host_source_pool_hits"] == 1
    after_pool = backend._host_source_pool.diagnostics()
    assert after_pool["patch_warm_misses"] == 2
    assert after_pool["patch_warm_hits"] == 1
    assert after_pool["patch_warm_bypasses"] == 0
    backend.close()


def test_aimdo_file_read_failure_with_failed_quiescence_retains_handle_and_buffer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "base.bin"
    path.write_bytes(bytes(range(16)))
    descriptor = aimdo.AimdoFileBackedValue(
        torch.empty((4,), dtype=torch.uint8, device="meta"),
        (aimdo.AimdoFileSpan("base", "dense", 0, 4, torch.uint8, (4,)),),
    )
    state = _backend_fixture(monkeypatch, host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("file", (descriptor,))
    backend.prioritize()
    handle = path.open("rb")
    backend.bind_file_source("base", handle)
    monkeypatch.setattr(
        state.host_buffers[0],
        "read_file_slice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("read failed")),
    )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="failed_fill_quiescence_failed"):
        backend.acquire("file")
    proof = backend.diagnostics()
    assert proof["host_buffer_live"] is proof["host_tensor_view_live"] is True
    assert backend._file_sources["base"] is handle
    assert not handle.closed
    assert "host-unregister" not in state.events and "host-free" not in state.events
    assert state.frees == []
    handle.close()


def test_aimdo_first_acquire_emits_bounded_copy_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch)
    records: list[tuple[str, dict[str, object]]] = []
    value = torch.arange(16, dtype=torch.uint8)
    backend = aimdo.AimdoDynamicResidency(
        "cuda:0",
        virtual_bytes=4096,
        diagnostic=lambda phase, details: records.append((phase, dict(details))),
    )
    backend.allocate_group("layer", (value,))

    first = backend.acquire("layer")
    backend.release(first)
    second = backend.acquire("layer")
    backend.release(second)
    backend.close()

    assert [phase for phase, _details in records] == [
        "first_acquire_begin",
        "first_acquire_after_fault",
        "first_acquire_raw_ready",
        "first_acquire_copy_ready",
    ]
    assert all(details["device"] == "cuda:0" for _phase, details in records)
    assert all(details["current_device"] == 0 for _phase, details in records)
    assert records[-1][1]["destination_bytes"] == 1024


def test_aimdo_rejects_cumulative_group_bytes_beyond_virtual_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("first", (torch.arange(8, dtype=torch.uint8),))

    with pytest.raises(RuntimeError, match="exceeds requested VBAR capacity"):
        backend.allocate_group("second", (torch.arange(16, dtype=torch.uint8),))

    assert len(state.allocations) == 1
    assert backend.diagnostics()["staged_bytes"] == 1024
    backend.close()


@pytest.mark.parametrize("current_index", [0, 2])
def test_aimdo_unindexed_cuda_uses_one_canonical_current_device(
    monkeypatch: pytest.MonkeyPatch,
    current_index: int,
) -> None:
    state = _backend_fixture(
        monkeypatch, current_index=current_index, device_count=current_index + 1
    )
    backend = aimdo.AimdoDynamicResidency("cuda", virtual_bytes=4096)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.acquire("layer")
    backend.release(lease)
    backend.close()

    expected = torch.device("cuda", current_index)
    assert backend.device == expected
    assert f"init_devices:[({current_index}, 0)]" in state.events
    assert f"vbar:4096:{current_index}" in state.events
    assert [device for device, _stream in state.streams] == [expected, expected]
    assert state.current_stream_devices and set(state.current_stream_devices) == {expected}


@pytest.mark.parametrize(
    ("requested", "current_index", "device_count", "message"),
    [
        ("cuda:2", 0, 1, "outside the available"),
        ("cuda:0", 1, 2, "selected index 1, expected 0"),
    ],
)
def test_aimdo_rejects_invalid_or_mismatched_index_before_import_or_allocation(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    current_index: int,
    device_count: int,
    message: str,
) -> None:
    state = _backend_fixture(monkeypatch, current_index=current_index, device_count=device_count)
    with pytest.raises(aimdo.DynamicResidencyDeviceError, match=message):
        aimdo.AimdoDynamicResidency(requested, virtual_bytes=4096)
    assert state.events == []
    assert state.allocations == []


def test_aimdo_fault_hit_miss_none_and_patch_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    signature = (7, 8)
    state = _backend_fixture(monkeypatch, faults=[signature, signature, signature, None])
    value = torch.arange(16, dtype=torch.uint8)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=8192)
    backend.allocate_group("layer", (value,))

    first = backend.acquire("layer")
    backend.release(first)
    second = backend.acquire("layer")
    assert second.values[0] is first.values[0]
    backend.release(second)
    backend.invalidate(reason="lora_to_base")
    restored = backend.acquire("layer")
    assert restored.values[0] is not first.values[0]
    backend.release(restored)
    temporary = backend.acquire("layer")
    assert temporary.token.temporary is not None
    backend.release(temporary)
    backend.close()

    proof = backend.diagnostics()
    assert proof["faults"] == 4
    assert proof["signature_hits"] == 1
    assert proof["signature_misses"] == 3
    assert proof["fault_none_temporaries"] == 1
    assert proof["lora_invalidations"] == proof["base_restores"] == 1
    assert proof["dirty_epoch"] == 1
    assert len(state.unpins) == 3


def test_aimdo_selective_patch_invalidation_preserves_base_group_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(
        monkeypatch,
        faults=[("base",), ("patch",), ("base",), ("patch-new",)],
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=8192)
    backend.allocate_group("base", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("patch", (torch.arange(32, dtype=torch.uint8),))
    base_first = backend.acquire("base")
    patch_first = backend.acquire("patch")
    backend.release(patch_first)
    backend.release(base_first)

    backend.invalidate_groups(("patch",), reason="lora_to_base")
    base_hit = backend.acquire("base")
    patch_refill = backend.acquire("patch")

    assert base_hit.values[0] is base_first.values[0]
    assert patch_refill.values[0] is not patch_first.values[0]
    backend.release(patch_refill)
    backend.release(base_hit)
    proof = backend.diagnostics()
    assert proof["signature_hits"] == 1
    assert proof["signature_misses"] == 3
    assert proof["lora_invalidations"] == proof["base_restores"] == 1
    backend.close()
    assert state.frees


def test_aimdo_failed_fill_is_unpinned_and_freed(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("bad", (torch.arange(32, dtype=torch.uint8),))
    allocation = backend._groups["bad"].allocation
    allocation[0].buffers[allocation[1]] = torch.empty(1, dtype=torch.uint8)

    with pytest.raises(RuntimeError):
        backend.acquire("bad")
    backend.close()

    assert state.unpins == [allocation]
    assert state.frees == [(77, 99)]


@pytest.mark.parametrize("failure_site", ["callback", "current_device"])
def test_aimdo_post_fault_diagnostic_failure_quiesces_and_unpins_before_free(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])

    def diagnostic(phase: str, _details) -> None:
        if failure_site == "callback" and phase == "first_acquire_after_fault":
            raise RuntimeError("after-fault callback failed")

    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096, diagnostic=diagnostic)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    if failure_site == "current_device":
        current_device_calls = 0

        def current_device() -> int:
            nonlocal current_device_calls
            current_device_calls += 1
            if current_device_calls == 2:
                raise RuntimeError("after-fault current-device lookup failed")
            return 0

        monkeypatch.setattr(aimdo.torch.cuda, "current_device", current_device)

    with pytest.raises(RuntimeError, match="after-fault"):
        backend.acquire("layer")

    allocation = backend._groups["layer"].allocation
    assert state.unpins == [allocation]
    assert backend._active == {}
    assert backend._faulted == set()
    assert state.frees == []
    backend.close()

    fault_index = state.events.index("fault")
    quiescence_index = state.events.index("device-sync", fault_index)
    unpin_index = state.events.index("unpin", quiescence_index)
    close_sync_index = state.events.index("device-sync", unpin_index)
    free_index = state.events.index("free", close_sync_index)
    assert fault_index < quiescence_index < unpin_index < close_sync_index < free_index


def test_aimdo_fault_error_propagates_without_temporary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("bad", (torch.arange(16, dtype=torch.uint8),))

    def failed_fault(_allocation):
        raise RuntimeError("fault failed")

    backend._model_vbar_module.vbar_fault = failed_fault
    with pytest.raises(RuntimeError, match="fault failed"):
        backend.acquire("bad")
    backend.close()
    assert state.frees == [(77, 99)]


def test_aimdo_reverse_wait_failure_drains_then_terminally_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.acquire("layer")

    original_wait = _Stream.wait_event

    def failed_reverse_wait(self, event):
        if self.name.startswith("copy") and event.recorded_stream == "current":
            raise RuntimeError("reverse wait failed")
        return original_wait(self, event)

    monkeypatch.setattr(_Stream, "wait_event", failed_reverse_wait)
    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="retirement_release_failed"):
        backend.release(lease)
    assert state.events.count("device-sync") == 1
    assert backend._active == {}
    assert backend.terminal_poison_reason() == "retirement_release_failed"
    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="retirement_release_failed"):
        backend.close()
    assert len(state.unpins) == 1
    assert state.frees == []


def test_aimdo_failed_device_quiescence_retains_every_owned_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=8192)
    backend.allocate_group("faulted", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("temporary", (torch.arange(8, dtype=torch.uint8),))
    faulted = backend.acquire("faulted")
    temporary = backend.acquire("temporary")
    assert temporary.token.temporary is not None
    groups_before = dict(backend._groups)
    streams_before = backend._streams
    sync_calls = 0

    def failed_sync(_device=None) -> None:
        nonlocal sync_calls
        sync_calls += 1
        raise RuntimeError("device quiescence failed")

    monkeypatch.setattr(aimdo.torch.cuda, "synchronize", failed_sync)
    with pytest.raises(RuntimeError, match="device_quiescence_failed") as failure:
        backend.close()
    assert isinstance(failure.value.__cause__, RuntimeError)

    proof = backend.diagnostics()
    assert proof["poisoned"] is proof["close_failed"] is True
    assert proof["poison_reason"] == "device_quiescence_failed"
    assert proof["live_allocations"] == 2
    assert proof["live_bytes"] == 8192
    assert proof["loaded_bytes"] is None
    assert backend._groups == groups_before
    assert backend._streams is streams_before
    assert backend._active[id(faulted.token.group)] is faulted.token
    assert backend._active[id(temporary.token.group)] is temporary.token
    assert temporary.token.temporary is not None
    assert id(faulted.token.group) in backend._faulted
    assert backend._ledger.unregister_calls == 0
    assert state.unpins == []
    assert state.frees == []
    assert backend._vbar._ptr == 99

    # A poisoned backend cannot be retried into publishing a false clean proof,
    # even if a later synchronization call would happen to succeed.
    monkeypatch.setattr(aimdo.torch.cuda, "synchronize", lambda _device=None: None)
    with pytest.raises(RuntimeError, match="poisoned"):
        backend.close()
    assert sync_calls == 1
    assert backend._ledger.unregister_calls == 0
    assert state.unpins == []
    assert state.frees == []


@pytest.mark.parametrize("failure_site", ["copy", "event", "record", "wait"])
@pytest.mark.parametrize("quiescence_fails", [False, True])
def test_aimdo_failed_fill_owns_fault_none_temporary_until_quiescence(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    quiescence_fails: bool,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[None])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    captured = []

    def assert_pending() -> None:
        token = next(iter(backend._active.values()))
        assert token.temporary is not None
        assert token.raw is token.temporary
        captured.append(token)

    original_event = aimdo.torch.cuda.Event
    original_copy = torch.Tensor.copy_
    original_record = _Event.record
    original_wait = _Stream.wait_event
    if failure_site == "copy":

        def failed_copy(_self, *_args, **_kwargs):
            assert_pending()
            raise RuntimeError("copy failed")

        monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    elif failure_site == "event":

        def failed_event():
            assert_pending()
            raise RuntimeError("event construction failed")

        monkeypatch.setattr(aimdo.torch.cuda, "Event", failed_event)
    elif failure_site == "record":

        def failed_record(_self, _stream):
            assert_pending()
            raise RuntimeError("event record failed")

        monkeypatch.setattr(_Event, "record", failed_record)
    else:

        def failed_wait(_self, _event):
            assert_pending()
            raise RuntimeError("stream wait failed")

        monkeypatch.setattr(_Stream, "wait_event", failed_wait)

    if quiescence_fails:
        monkeypatch.setattr(
            aimdo.torch.cuda,
            "synchronize",
            lambda _device=None: (_ for _ in ()).throw(
                RuntimeError("failed-fill quiescence failed")
            ),
        )
        with pytest.raises(aimdo.DynamicResidencyPoisoned, match="failed_fill_quiescence_failed"):
            backend.acquire("layer")
        token = captured[0]
        assert backend._active[id(token.group)] is token
        assert token.temporary is not None
        assert token.raw is token.temporary
        if failure_site in {"record", "wait"}:
            assert token.pending_events
        assert backend.terminal_poison_reason() == "failed_fill_quiescence_failed"
        assert backend._ledger.unregister_calls == 0
        assert state.unpins == []
        assert state.frees == []
        return

    with pytest.raises(RuntimeError, match="failed"):
        backend.acquire("layer")
    token = captured[0]
    assert backend._active == {}
    assert token.temporary is None
    assert token.raw is None
    assert token.pending_events == []
    assert backend.terminal_poison_reason() is None

    # Restore the injected operation before proving ordinary synchronized close.
    monkeypatch.setattr(aimdo.torch.cuda, "Event", original_event)
    monkeypatch.setattr(torch.Tensor, "copy_", original_copy)
    monkeypatch.setattr(_Event, "record", original_record)
    monkeypatch.setattr(_Stream, "wait_event", original_wait)
    backend.close()
    assert backend._ledger.unregister_calls == 1
    assert state.frees == [(77, 99)]


@pytest.mark.parametrize("failure_site", ["copy", "event", "record", "wait"])
def test_aimdo_failed_gathered_fill_retains_host_buffer_when_quiescence_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)], host_buffer=True)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    if failure_site == "copy":
        monkeypatch.setattr(
            torch.Tensor,
            "copy_",
            lambda _self, *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("gathered copy failed")
            ),
        )
    elif failure_site == "event":
        monkeypatch.setattr(
            aimdo.torch.cuda,
            "Event",
            lambda: (_ for _ in ()).throw(RuntimeError("gathered event failed")),
        )
    elif failure_site == "record":
        monkeypatch.setattr(
            _Event,
            "record",
            lambda _self, _stream: (_ for _ in ()).throw(RuntimeError("gathered record failed")),
        )
    else:
        monkeypatch.setattr(
            _Stream,
            "wait_event",
            lambda _self, _event: (_ for _ in ()).throw(RuntimeError("gathered wait failed")),
        )
    monkeypatch.setattr(
        aimdo.torch.cuda,
        "synchronize",
        lambda _device=None: (_ for _ in ()).throw(RuntimeError("gathered quiescence failed")),
    )

    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="failed_fill_quiescence_failed"):
        backend.acquire("layer")

    proof = backend.diagnostics()
    assert proof["host_buffer_live"] is True
    assert proof["host_tensor_view_live"] is True
    assert proof["host_buffer_allocations"] == 2
    assert proof["host_buffer_unregistrations"] == proof["host_buffer_frees"] == 0
    assert proof["host_buffer_transfer_pending"] is (failure_site == "wait")
    assert proof["poisoned"] is proof["close_failed"] is True
    assert backend._host_buffer is state.host_buffers[0]
    assert backend._host_tensor is None
    retained_token = next(iter(backend._active.values()))
    assert retained_token.source_lease is not None
    assert retained_token.source_lease.tensor.data_ptr() == state.host_buffers[0].backing.data_ptr()
    assert not any(event.startswith("host-unregister:") for event in state.events)
    assert "host-free" not in state.events
    assert state.unpins == []
    assert state.frees == []


def test_aimdo_prefetch_defers_compute_wait_and_acquire_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), (2,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("first", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("second", (torch.arange(16, dtype=torch.uint8),))

    lease = backend.prefetch("first")
    assert "wait:current" not in state.events
    assert lease.token.pending_events
    backend.wait(lease)
    assert state.events.count("wait:current") == 1
    backend.wait(lease)
    assert state.events.count("wait:current") == 1
    backend.release(lease)

    compatible = backend.acquire("second")
    assert state.events.count("wait:current") == 2
    backend.release(compatible)
    proof = backend.diagnostics()
    assert proof["prefetch"] is True
    assert proof["prefetch_calls"] == 1
    assert proof["transfer_events"] == proof["transfer_waits"] == 2
    backend.close()


def test_aimdo_successful_group_release_is_stream_ordered_without_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), (2,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=4096)
    backend.allocate_group("left", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("right", (torch.arange(16, dtype=torch.uint8),))

    left = backend.acquire("left")
    right = backend.acquire("right")
    release_start = len(state.events)
    backend.release_group((left, right))
    released = state.events[release_start:]

    # Both Engine transfer streams are gated before either VBAR allocation can
    # be reused, including the stream that did not stage this group's leaves.
    assert released.count("record:current") == 1
    assert released.count("wait:copy-0") == 1
    assert released.count("wait:copy-1") == 1
    assert released.count("unpin") == 2
    assert released.index("wait:copy-0") < released.index("unpin")
    assert released.index("wait:copy-1") < released.index("unpin")
    assert not any(event.startswith("event-sync:") for event in released)
    assert "device-sync" not in released
    proof = backend.diagnostics()
    assert proof["transfer_waits"] == 2
    assert proof["reverse_stream_waits"] == 2
    assert proof["pending_retirement_batches"] == 0
    backend.close()


def test_aimdo_fault_none_and_host_source_ownership_retire_only_after_event_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, host_buffer=True, faults=[None])
    created: list[_Event] = []

    class _DeferredEvent(_Event):
        ready = False

        def query(self) -> bool:
            self.events.append(f"event-query:{self.recorded_stream or 'unrecorded'}")
            return self.ready

    def event_factory():
        event = _DeferredEvent(state.events)
        created.append(event)
        return event

    monkeypatch.setattr(aimdo.torch.cuda, "Event", event_factory)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("temporary", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    lease = backend.acquire("temporary")
    temporary = lease.token.temporary
    source_lease = lease.token.source_lease
    backend.release(lease)

    assert backend._active == {}
    assert backend._retirements[0].tokens == (lease.token,)
    assert lease.token.temporary is temporary
    assert lease.token.source_lease is source_lease
    assert source_lease.complete is False
    assert backend.diagnostics()["host_source_pool_temporary_slices"] == 1
    assert "device-sync" not in state.events

    for event in created:
        event.ready = True
    backend._poll_retirements()
    assert backend._retirements == []
    assert lease.token.temporary is None
    assert lease.token.source_lease is None
    assert source_lease.complete is True
    assert backend.diagnostics()["host_source_pool_temporary_slices"] == 0
    assert "device-sync" not in state.events
    backend.close()


def test_aimdo_close_drains_pending_pageable_temporary_before_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[None])

    class _PendingEvent(_Event):
        def query(self) -> bool:
            return False

    monkeypatch.setattr(
        aimdo.torch.cuda,
        "Event",
        lambda: _PendingEvent(state.events),
    )
    backend = aimdo.AimdoDynamicResidency(
        "cuda:0", virtual_bytes=1024, gathered_host_transfer=False
    )
    backend.allocate_group("temporary", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.acquire("temporary")
    owned_temporary = lease.token.temporary
    backend.release(lease)
    assert lease.token.temporary is owned_temporary
    assert backend.diagnostics()["pending_retirement_batches"] == 1

    backend.close()
    sync_index = state.events.index("device-sync")
    free_index = state.events.index("free")
    assert sync_index < free_index
    assert lease.token.temporary is None
    assert backend.diagnostics()["pending_retirement_batches"] == 0


def test_aimdo_stage_boundary_drains_once_and_preserves_warm_sources_and_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), None], host_buffer=True)

    class _PendingEvent(_Event):
        def query(self) -> bool:
            self.events.append(f"event-query:{self.recorded_stream or 'unrecorded'}")
            return False

    monkeypatch.setattr(
        aimdo.torch.cuda,
        "Event",
        lambda: _PendingEvent(state.events),
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("warm", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("temporary", (torch.arange(16, dtype=torch.uint8),))
    backend.prioritize()

    warm = backend.acquire("warm")
    warm_signature = warm.token.group.signature
    backend.release(warm)
    temporary = backend.acquire("temporary")
    temporary_owner = temporary.token.temporary
    backend.release(temporary)
    before = backend.diagnostics()
    assert before["pending_retirement_batches"] >= 1
    assert "device-sync" not in state.events

    backend.prepare_stage(0)

    proof = backend.diagnostics()
    assert state.events.count("device-sync") == 1
    assert state.events.count("empty-cache") == 1
    assert state.events.index("device-sync") < state.events.index("empty-cache")
    assert proof["stage_prepare_calls"] == 1
    assert proof["stage_prepare_pending_before"] == before["pending_retirement_batches"]
    assert proof["stage_prepare_pending_after"] == 0
    assert proof["pending_retirement_batches"] == 0
    assert proof["host_source_pool_retained_slices"] == 1
    assert proof["host_source_pool_temporary_slices"] == 0
    assert warm.token.group.signature is warm_signature
    assert temporary.token.temporary is None
    assert temporary_owner is not None
    assert state.trim_requests == []
    backend.close()


def test_aimdo_stage_boundary_trims_only_post_cache_capacity_shortfall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    state.loaded_override = 64
    state.residency = [1, 1]
    state.cuda_free = 32
    state.cuda_total = 256
    state.cuda_allocated = 16
    state.cuda_reserved = 64

    # Driver free plus reusable Torch reservation already meets the request.
    backend.prepare_stage(80)
    assert state.trim_requests == []

    state.cuda_reserved = state.cuda_allocated
    backend.prepare_stage(80)

    assert state.trim_requests == [48]
    proof = backend.diagnostics()
    assert proof["stage_prepare_calls"] == 2
    assert proof["stage_prepare_requested_bytes"] == 160
    assert proof["stage_prepare_trim_requested"] == 48
    assert proof["stage_prepare_trim_freed"] == 48
    assert proof["stage_prepare_loaded_before"] == 64
    assert proof["stage_prepare_loaded_after"] == 16
    assert proof["stage_prepare_cuda_free_before"] == 32
    assert proof["stage_prepare_cuda_free_after"] == 80
    assert state.events.count("device-sync") == 2
    assert state.events.count("empty-cache") == 2
    backend.close()


def test_aimdo_stage_boundary_trim_failure_terminally_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    state.loaded_override = 64
    state.residency = [1]
    state.cuda_free = 0
    state.cuda_total = 256
    backend._vbar.free_memory = lambda _size: (_ for _ in ()).throw(
        RuntimeError("native trim failed")
    )

    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="stage_prepare_failed"):
        backend.prepare_stage(32)

    assert backend.terminal_poison_reason() == "stage_prepare_failed"
    assert state.events.count("device-sync") == 1
    assert state.events.count("empty-cache") == 1
    assert state.frees == []


def test_aimdo_abandoned_prefetch_stays_pinned_and_busy_until_transfer_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), (2,)])
    created: list[_Event] = []

    class _DeferredEvent(_Event):
        ready = False

        def query(self) -> bool:
            return self.ready

    def event_factory():
        event = _DeferredEvent(state.events)
        created.append(event)
        return event

    monkeypatch.setattr(aimdo.torch.cuda, "Event", event_factory)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("prefetch", (torch.arange(16, dtype=torch.uint8),))

    abandoned = backend.prefetch("prefetch")
    backend.release(abandoned)
    group_id = id(abandoned.token.group)
    assert group_id in backend._active
    assert group_id in backend._faulted
    assert state.unpins == []
    with pytest.raises(RuntimeError, match="already acquired"):
        backend.prefetch("prefetch")

    for event in created:
        event.ready = True
    backend._poll_retirements()
    assert group_id not in backend._active
    assert group_id not in backend._faulted
    assert state.unpins == [abandoned.token.group.allocation]
    reacquired = backend.prefetch("prefetch")
    backend.wait(reacquired)
    backend.release(reacquired)
    backend.close()


def test_aimdo_mixed_consumed_and_unconsumed_group_uses_distinct_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,), (2,)])
    created: list[_Event] = []

    class _DeferredEvent(_Event):
        ready = False

        def query(self) -> bool:
            return self.ready

    def event_factory():
        event = _DeferredEvent(state.events)
        created.append(event)
        return event

    monkeypatch.setattr(aimdo.torch.cuda, "Event", event_factory)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=2048)
    backend.allocate_group("consumed", (torch.arange(16, dtype=torch.uint8),))
    backend.allocate_group("unconsumed", (torch.arange(16, dtype=torch.uint8),))

    consumed = backend.acquire("consumed")
    unconsumed = backend.prefetch("unconsumed")
    backend.release_group((consumed, unconsumed))

    consumed_id = id(consumed.token.group)
    unconsumed_id = id(unconsumed.token.group)
    assert consumed_id not in backend._active
    assert consumed_id not in backend._faulted
    assert unconsumed_id in backend._active
    assert unconsumed_id in backend._faulted
    assert state.unpins == [consumed.token.group.allocation]
    proof = backend.diagnostics()
    assert proof["reverse_stream_waits"] == 2
    assert proof["pending_retirement_batches"] == 1

    for event in created:
        event.ready = True
    backend._poll_retirements()
    assert backend._active == {}
    assert backend._faulted == set()
    assert state.unpins == [
        consumed.token.group.allocation,
        unconsumed.token.group.allocation,
    ]
    backend.close()


def test_aimdo_recovery_unpin_failure_retains_ownership_and_terminally_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("layer", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.acquire("layer")

    original_wait = _Stream.wait_event

    def failed_reverse_wait(self, event):
        if self.name.startswith("copy") and event.recorded_stream == "current":
            raise RuntimeError("reverse wait failed")
        return original_wait(self, event)

    monkeypatch.setattr(_Stream, "wait_event", failed_reverse_wait)
    monkeypatch.setattr(
        backend._model_vbar_module,
        "vbar_unpin",
        lambda _allocation: (_ for _ in ()).throw(RuntimeError("recovery unpin failed")),
    )

    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="retirement_cleanup_failed"):
        backend.release(lease)
    group_id = id(lease.token.group)
    assert state.events.count("device-sync") == 1
    assert group_id in backend._active
    assert group_id in backend._faulted
    assert backend.terminal_poison_reason() == "retirement_cleanup_failed"
    assert state.frees == []


def test_aimdo_deferred_unpin_failure_retains_batch_and_terminally_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])
    created: list[_Event] = []

    class _DeferredEvent(_Event):
        ready = False

        def query(self) -> bool:
            return self.ready

    def event_factory():
        event = _DeferredEvent(state.events)
        created.append(event)
        return event

    monkeypatch.setattr(aimdo.torch.cuda, "Event", event_factory)
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("prefetch", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.prefetch("prefetch")
    backend.release(lease)
    assert backend._retirements[0].tokens == (lease.token,)

    monkeypatch.setattr(
        backend._model_vbar_module,
        "vbar_unpin",
        lambda _allocation: (_ for _ in ()).throw(RuntimeError("deferred unpin failed")),
    )
    for event in created:
        event.ready = True
    with pytest.raises(aimdo.DynamicResidencyPoisoned, match="retirement_cleanup_failed"):
        backend._poll_retirements()

    group_id = id(lease.token.group)
    assert backend._retirements[0].tokens == (lease.token,)
    assert group_id in backend._active
    assert group_id in backend._faulted
    assert backend.terminal_poison_reason() == "retirement_cleanup_failed"


def test_aimdo_invalidation_drains_retirement_held_prefetch_before_epoch_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _backend_fixture(monkeypatch, faults=[(1,)])

    class _PendingEvent(_Event):
        def query(self) -> bool:
            return False

    monkeypatch.setattr(
        aimdo.torch.cuda,
        "Event",
        lambda: _PendingEvent(state.events),
    )
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.prefetch("patch")
    backend.release(lease)
    assert id(lease.token.group) in backend._active

    backend.invalidate(reason="patch_changed")
    assert backend._active == {}
    assert backend._retirements == []
    assert state.events.count("device-sync") == 1
    assert state.unpins == [lease.token.group.allocation]
    assert backend.diagnostics()["dirty_epoch"] == 1
    backend.close()


def test_aimdo_invalidation_still_rejects_genuine_active_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend_fixture(monkeypatch, faults=[(1,)])
    backend = aimdo.AimdoDynamicResidency("cuda:0", virtual_bytes=1024)
    backend.allocate_group("patch", (torch.arange(16, dtype=torch.uint8),))
    lease = backend.acquire("patch")

    with pytest.raises(RuntimeError, match="cannot invalidate active groups"):
        backend.invalidate(reason="patch_changed")

    backend.release(lease)
    backend.close()
