"""Fence-aware AIMDO HostBuffer source slices for dynamic residency.

The pool is model-neutral.  It owns four logical lanes (base/patch crossed
with retained/temporary lifetime), but allocates only lanes with non-zero
capacity.  Every lane reserves its maximum address range up front and grows
in place; a HostBuffer is never reallocated after a Torch view exists.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from .host_registration import available_physical_memory_bytes

_BASE_PREWARM_BYTES = 64 * 1024 * 1024
_PATCH_PREWARM_BYTES = 8 * 1024 * 1024
_MIN_AVAILABLE_RAM_HEADROOM_BYTES = 2 * 1024**3


class _HostRegistrationRefused(RuntimeError):
    """CUDA declined a new append registration without publishing it."""


class _HostAppendViewRefused(RuntimeError):
    """The appended logical region could not be authenticated for registration."""


class HostSourceClass(str, Enum):
    BASE = "base"
    PATCH = "patch"


class HostSourceLifetime(str, Enum):
    WARM = "warm_source"
    PREFETCH_TEMPORARY = "prefetch_temporary"


class HostSourcePoolStructuralError(RuntimeError):
    """A native HostBuffer mutation left the pool terminally non-reusable."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"AIMDO HostBuffer source pool is poisoned: {reason}")


class HostSourceWarmUnavailable(RuntimeError):
    """A retained source could not be admitted before any native mutation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"AIMDO warm host source is unavailable: {reason}")


class HostSourceDirectTransferRequired(RuntimeError):
    """A preflight proved HostBuffer growth unsafe; use the direct copy path."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"AIMDO direct host-source transfer is required: {reason}")


class HostSourcePoolSetupError(HostSourcePoolStructuralError):
    """Pool construction failed after an earlier native owner could not close."""

    def __init__(self, pool: AimdoHostSourcePool) -> None:
        self.pool = pool
        super().__init__("host_buffer_setup_cleanup_failed")


class HostSourcePoolSetupFallback(RuntimeError):
    """Construction failed, but every previously created owner closed safely."""

    def __init__(self, pool: AimdoHostSourcePool, primary: BaseException) -> None:
        self.pool = pool
        self.primary = primary
        super().__init__(
            f"AIMDO HostBuffer pool setup failed safely: {type(primary).__name__}: {primary}"
        )


@dataclass(slots=True)
class _Slice:
    lane: tuple[HostSourceClass, HostSourceLifetime]
    generation: int
    offset: int
    size: int
    view: torch.Tensor
    cache_key: Hashable | None
    released: bool = False


@dataclass(slots=True)
class HostSourceLease:
    """One transfer's ownership of a stable host source slice."""

    source: _Slice
    needs_fill: bool
    fences: list[Any] = field(default_factory=list)
    complete: bool = False
    published: bool = False

    @property
    def tensor(self) -> torch.Tensor:
        return self.source.view


@dataclass(slots=True)
class _Lane:
    owner: Any
    capacity: int
    generation: int = 1
    used: int = 0
    slices: list[_Slice] = field(default_factory=list)
    cache: dict[Hashable, _Slice] = field(default_factory=dict)
    registered_bytes: int = 0
    registrations: list[_Registration] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Registration:
    offset: int
    size: int
    address: int


class AimdoHostSourcePool:
    """Append-only retained sources plus LIFO temporary HostBuffer slices."""

    def __init__(
        self,
        capacities: Mapping[tuple[HostSourceClass, HostSourceLifetime], int],
        *,
        host_buffer_factory: Callable[..., Any],
        hostbuf_to_tensor: Callable[[Any], torch.Tensor],
        registration_budget_bytes: int,
        temporary_reserve_bytes: int = 0,
        available_memory_bytes: Callable[[], int | None] = available_physical_memory_bytes,
        cudart: Any | None = None,
    ) -> None:
        if (
            not isinstance(registration_budget_bytes, int)
            or isinstance(registration_budget_bytes, bool)
            or registration_budget_bytes < 0
        ):
            raise ValueError("AIMDO HostBuffer registration budget must be non-negative")
        if (
            not isinstance(temporary_reserve_bytes, int)
            or isinstance(temporary_reserve_bytes, bool)
            or temporary_reserve_bytes < 0
        ):
            raise ValueError("AIMDO temporary registration reserve must be non-negative")
        if not callable(available_memory_bytes):
            raise TypeError("AIMDO available-memory provider must be callable")
        self._to_tensor = hostbuf_to_tensor
        self._available_memory_bytes = available_memory_bytes
        self._cudart = cudart
        self._lanes: dict[tuple[HostSourceClass, HostSourceLifetime], _Lane] = {}
        self._closed = False
        self._poisoned = False
        self._poison_reason: str | None = None
        self._generation = 1
        self._leases: dict[int, HostSourceLease] = {}
        self.allocations = 0
        self.unregistrations = 0
        self.frees = 0
        self.source_hits = 0
        self.source_misses = 0
        self.warm_source_hits = 0
        self.warm_source_misses = 0
        self.warm_source_bypasses = 0
        self.base_warm_hits = 0
        self.base_warm_misses = 0
        self.base_warm_bypasses = 0
        self.patch_warm_hits = 0
        self.patch_warm_misses = 0
        self.patch_warm_bypasses = 0
        self.warm_ram_pressure_bypasses = 0
        self.warm_zero_delta_extend_refusals = 0
        self.warm_registration_refusals = 0
        self.temporary_ram_pressure_bypasses = 0
        self.temporary_zero_delta_extend_refusals = 0
        self.temporary_registration_refusals = 0
        self.stale_rejections = 0
        self.registration_budget_bytes = registration_budget_bytes
        self.temporary_reserve_bytes = min(
            temporary_reserve_bytes,
            registration_budget_bytes,
        )
        self.warm_registration_budget_bytes = (
            registration_budget_bytes - self.temporary_reserve_bytes
        )
        self.registration_attempts = 0
        self.registration_attempt_bytes = 0
        self.registration_successes = 0
        self.registration_failures = 0
        self.registration_failure_bytes = 0
        self.registration_registered_bytes = 0
        self.registration_unregistered_bytes = 0
        self.registration_live_bytes = 0
        self.registration_peak_bytes = 0
        self.registration_state_proven = True
        try:
            for key, capacity in capacities.items():
                if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
                    raise ValueError(
                        "AIMDO HostBuffer lane capacity must be a non-negative integer"
                    )
                if capacity == 0:
                    continue
                prewarm = min(
                    capacity,
                    _BASE_PREWARM_BYTES
                    if key[0] is HostSourceClass.BASE
                    else _PATCH_PREWARM_BYTES,
                )
                owner = host_buffer_factory(
                    0,
                    prewarm=prewarm,
                    max_grow_size=capacity,
                    mark_cold=True,
                )
                self._lanes[key] = _Lane(owner=owner, capacity=capacity)
                self.allocations += 1
        except BaseException as primary:
            cleanup_failed = False
            for lane in self._lanes.values():
                try:
                    if self._shrink_lane(lane, 0):
                        self.unregistrations += 1
                    lane.owner.__del__()
                    if getattr(lane.owner, "_ptr", None):
                        raise RuntimeError("AIMDO HostBuffer setup free retained its pointer")
                    self.frees += 1
                except BaseException as cleanup_error:  # noqa: BLE001
                    cleanup_failed = True
                    self.registration_state_proven = False
                    primary.add_note(
                        "AIMDO HostBuffer pool setup cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if cleanup_failed:
                self._poison("host_buffer_setup_cleanup_failed")
                raise HostSourcePoolSetupError(self) from primary
            self._generation += 1
            self._closed = True
            raise HostSourcePoolSetupFallback(self, primary) from primary

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def capacity_bytes(self) -> int:
        return sum(lane.capacity for lane in self._lanes.values())

    @property
    def max_lane_capacity_bytes(self) -> int:
        return max((lane.capacity for lane in self._lanes.values()), default=0)

    @property
    def live(self) -> bool:
        return not self._closed and bool(self._lanes)

    @property
    def owners(self) -> tuple[Any, ...]:
        return tuple(lane.owner for lane in self._lanes.values())

    def acquire(
        self,
        source_class: HostSourceClass,
        lifetime: HostSourceLifetime,
        *,
        size: int,
        cache_key: Hashable | None = None,
    ) -> HostSourceLease:
        self._require_open()
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("AIMDO host source slice size must be a positive integer")
        key = (source_class, lifetime)
        try:
            lane = self._lanes[key]
        except KeyError as exc:
            raise RuntimeError(f"AIMDO host source lane is unavailable: {key}") from exc
        if lifetime is HostSourceLifetime.WARM:
            if cache_key is None:
                raise ValueError("retained AIMDO host sources require a cache key")
            cached = lane.cache.get(cache_key)
            if cached is not None:
                self._validate_slice(cached)
                if cached.size != size:
                    raise RuntimeError("AIMDO retained host source size changed")
                self.source_hits += 1
                self._record_warm(source_class, "hit")
                lease = HostSourceLease(cached, needs_fill=False, published=True)
                self._leases[id(lease)] = lease
                return lease
            self._record_warm(source_class, "miss")
        elif cache_key is not None:
            raise ValueError("temporary AIMDO host sources cannot have a cache key")

        end = lane.used + size
        if end > lane.capacity:
            if lifetime is HostSourceLifetime.WARM:
                self._record_warm(source_class, "bypass")
                raise HostSourceWarmUnavailable("lane_capacity_exhausted")
            raise RuntimeError("AIMDO host source lane exhausted its fixed capacity")
        available = self._available_memory_bytes()
        if available is not None:
            if (
                not isinstance(available, int)
                or isinstance(available, bool)
                or available < 0
            ):
                raise RuntimeError("AIMDO available physical memory query is not canonical")
            if size + _MIN_AVAILABLE_RAM_HEADROOM_BYTES > available:
                if lifetime is HostSourceLifetime.WARM:
                    self._record_warm(source_class, "bypass")
                    self.warm_ram_pressure_bypasses += 1
                    raise HostSourceWarmUnavailable("physical_ram_pressure")
                self.temporary_ram_pressure_bypasses += 1
                raise HostSourceDirectTransferRequired("physical_ram_pressure")
        start = lane.used
        self.registration_attempts += 1
        self.registration_attempt_bytes += size
        registration_limit = (
            self.warm_registration_budget_bytes
            if lifetime is HostSourceLifetime.WARM
            else self.registration_budget_bytes
        )
        if self.registration_live_bytes + size > registration_limit:
            self.registration_failures += 1
            self.registration_failure_bytes += size
            if lifetime is HostSourceLifetime.WARM:
                self._record_warm(source_class, "bypass")
                raise HostSourceWarmUnavailable("registration_budget_exhausted")
            raise RuntimeError("AIMDO temporary host source exceeds registration budget")
        try:
            prior_owner_size, prior_raw_address = self._validated_owner_region(lane)
        except BaseException as primary:
            self.registration_failures += 1
            self.registration_failure_bytes += size
            self.registration_state_proven = False
            self._poison("host_buffer_view_validation_failed")
            raise HostSourcePoolStructuralError(
                "host_buffer_view_validation_failed"
            ) from primary
        prior_registered_bytes = lane.registered_bytes
        prior_registration_layout = tuple(lane.registrations)
        prior_registration_live = self.registration_live_bytes
        prior_slice_count = len(lane.slices)
        prior_cache_count = len(lane.cache)
        prior_lease_fences = tuple(
            sorted((lease_id, len(lease.fences)) for lease_id, lease in self._leases.items())
        )
        extended = False
        registered = False
        full: torch.Tensor | None = None
        appended: torch.Tensor | None = None
        try:
            # Match Comfy's narrow native seam: grow the HostBuffer without
            # registration, authenticate the exact appended view, then make
            # the CUDA registration an independently recoverable operation.
            result = lane.owner.extend(size, reallocate=False, register=False)
            if result is False:
                raise RuntimeError("AIMDO HostBuffer refused a non-reallocating append")
            extended = True
            try:
                full = self._to_tensor(lane.owner)
                if (
                    not isinstance(full, torch.Tensor)
                    or full.device.type != "cpu"
                    or full.dtype is not torch.uint8
                    or not full.is_contiguous()
                    or full.numel() != end
                    or full.data_ptr() != int(lane.owner.get_raw_address())
                ):
                    raise RuntimeError("AIMDO HostBuffer appended Torch view is not exact")
                appended = full[start:end]
                if appended.numel() != size or appended.data_ptr() != full.data_ptr() + start:
                    raise RuntimeError("AIMDO HostBuffer appended slice is not exact")
            except BaseException as view_error:
                raise _HostAppendViewRefused from view_error
            register_result = self._cuda_runtime().cudaHostRegister(
                appended.data_ptr(), size, 1
            )
            if register_result != 0:
                self._discard_cuda_registration_error()
                raise _HostRegistrationRefused
            registered = True
            lane.registrations.append(
                _Registration(offset=start, size=size, address=appended.data_ptr())
            )
            lane.registered_bytes += size
            self.registration_successes += 1
            self.registration_registered_bytes += size
            self.registration_live_bytes += size
            self.registration_peak_bytes = max(
                self.registration_peak_bytes,
                self.registration_live_bytes,
            )
        except (_HostRegistrationRefused, _HostAppendViewRefused) as primary:
            appended = None
            full = None
            self.registration_failures += 1
            self.registration_failure_bytes += size
            try:
                self._rollback_unregistered_append(
                    lane,
                    start=start,
                    owner_size=prior_owner_size,
                    raw_address=prior_raw_address,
                    registered_bytes=prior_registered_bytes,
                    registration_layout=prior_registration_layout,
                    registration_live=prior_registration_live,
                    slice_count=prior_slice_count,
                    cache_count=prior_cache_count,
                    lease_fences=prior_lease_fences,
                )
            except BaseException as rollback_error:  # noqa: BLE001 - native rollback boundary
                self.registration_state_proven = False
                self._poison("host_buffer_registration_rollback_failed")
                primary.add_note(
                    "AIMDO HostBuffer registration rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
                raise HostSourcePoolStructuralError(
                    "host_buffer_registration_rollback_failed"
                ) from primary
            if lifetime is HostSourceLifetime.WARM:
                self._record_warm(source_class, "bypass")
                self.warm_registration_refusals += 1
                reason = (
                    "cuda_host_registration_refused"
                    if isinstance(primary, _HostRegistrationRefused)
                    else "host_buffer_appended_view_refused"
                )
                raise HostSourceWarmUnavailable(reason) from primary
            self.temporary_registration_refusals += 1
            reason = (
                "cuda_host_registration_refused"
                if isinstance(primary, _HostRegistrationRefused)
                else "host_buffer_appended_view_refused"
            )
            raise HostSourceDirectTransferRequired(reason) from primary
        except BaseException as primary:
            if not extended:
                self.registration_failures += 1
                self.registration_failure_bytes += size
                if self._zero_delta_refusal_proven(
                    lane,
                    owner_size=prior_owner_size,
                    raw_address=prior_raw_address,
                    registered_bytes=prior_registered_bytes,
                    registration_layout=prior_registration_layout,
                    registration_live=prior_registration_live,
                    slice_count=prior_slice_count,
                    cache_count=prior_cache_count,
                    lease_fences=prior_lease_fences,
                ):
                    if lifetime is HostSourceLifetime.WARM:
                        self._record_warm(source_class, "bypass")
                        self.warm_zero_delta_extend_refusals += 1
                        raise HostSourceWarmUnavailable(
                            "native_extend_refused_without_delta"
                        ) from primary
                    self.temporary_zero_delta_extend_refusals += 1
                    raise HostSourceDirectTransferRequired(
                        "native_extend_refused_without_delta"
                    ) from primary
                self.registration_state_proven = False
                self._poison("host_buffer_extend_failed")
                raise HostSourcePoolStructuralError("host_buffer_extend_failed") from primary
            # A Torch view may already name the unregistered append. Drop it
            # before attempting the exact logical rollback.
            appended = None
            full = None
            try:
                if registered:
                    self._unregister_tail(lane, start)
                self._truncate_unregistered_lane(lane, start)
            except BaseException as rollback_error:  # noqa: BLE001 - native rollback boundary
                self.registration_state_proven = False
                self._poison("host_buffer_rollback_failed")
                primary.add_note(
                    "AIMDO HostBuffer append rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
                raise HostSourcePoolStructuralError("host_buffer_rollback_failed") from primary
            self._poison("host_buffer_view_validation_failed")
            raise HostSourcePoolStructuralError("host_buffer_view_validation_failed") from primary
        source = _Slice(
            lane=key,
            generation=lane.generation,
            offset=lane.used,
            size=size,
            view=full[lane.used : end],
            cache_key=cache_key,
        )
        lane.used = end
        lane.slices.append(source)
        self.source_misses += 1
        lease = HostSourceLease(source, needs_fill=True)
        self._leases[id(lease)] = lease
        return lease

    def add_fence(self, lease: HostSourceLease, event: Any) -> None:
        self._validate_lease(lease)
        lease.fences.append(event)
        source = lease.source
        if (
            source.lane[1] is HostSourceLifetime.WARM
            and lease.needs_fill
            and not lease.published
        ):
            if source.cache_key is None:
                raise RuntimeError("AIMDO retained source has no publication key")
            lane = self._lanes[source.lane]
            if source.cache_key in lane.cache:
                raise RuntimeError("AIMDO retained source was concurrently published")
            lane.cache[source.cache_key] = source
            lease.published = True

    def read_file_slice(
        self,
        lease: HostSourceLease,
        file_obj: Any,
        file_offset: int,
        size: int,
        *,
        slice_offset: int,
    ) -> None:
        """Fill a bounded extent of a newly acquired slice through AIMDO."""

        self._validate_lease(lease)
        if not lease.needs_fill or slice_offset < 0 or slice_offset + size > lease.source.size:
            raise RuntimeError("AIMDO host source file fill is outside its new slice")
        lane = self._lanes[lease.source.lane]
        lane.owner.read_file_slice(
            file_obj,
            file_offset,
            size,
            offset=lease.source.offset + slice_offset,
            stream=0,
            device_ptr=0,
            device=-1,
        )

    def complete_transfer(self, lease: HostSourceLease, *, quiesced: bool) -> None:
        self._validate_lease(lease)
        if not quiesced:
            raise RuntimeError("AIMDO host source transfer was not proven quiescent")
        lease.fences.clear()
        lease.complete = True
        self._leases.pop(id(lease), None)
        source = lease.source
        if source.lane[1] is HostSourceLifetime.PREFETCH_TEMPORARY or (
            source.lane[1] is HostSourceLifetime.WARM and not lease.published
        ):
            source.released = True
            self._reclaim_released_tail(self._lanes[source.lane])

    def invalidate_patch_sources(self) -> None:
        """Invalidate mutable patch lanes without disturbing immutable base slices."""

        self._require_open()
        for key, lane in self._lanes.items():
            if key[0] is not HostSourceClass.PATCH:
                continue
            if any(not item.complete for item in self._active_leases_for_lane(lane)):
                raise RuntimeError("AIMDO patch source invalidation has active transfers")
            self._drop_lane_views(lane)
            try:
                self._shrink_lane(lane, 0)
            except BaseException as exc:
                self.registration_state_proven = False
                self._poison("host_buffer_patch_truncate_failed")
                raise HostSourcePoolStructuralError(
                    "host_buffer_patch_truncate_failed"
                ) from exc
            lane.cache.clear()
            lane.slices.clear()
            lane.used = 0
            lane.generation += 1
        self._generation += 1

    def close(self, *, quiesced: bool) -> None:
        if self._closed:
            return
        if self._poisoned:
            raise HostSourcePoolStructuralError(self._poison_reason or "unknown")
        if not quiesced:
            raise RuntimeError("AIMDO host source pool cannot close before quiescence")
        # The caller's device barrier proves native consumers quiescent. Drop
        # every Torch object that can still name HostBuffer storage before the
        # first native unregister/truncate/free operation. External lease
        # objects remain safe because their shared slice is replaced in place.
        for lane in self._lanes.values():
            self._drop_lane_views(lane)
        for lease in self._leases.values():
            lease.fences.clear()
            lease.complete = True
        self._leases.clear()
        for lane in self._lanes.values():
            try:
                truncated = self._shrink_lane(lane, 0)
            except BaseException as exc:
                self.registration_state_proven = False
                self._poison("host_buffer_close_truncate_failed")
                raise HostSourcePoolStructuralError(
                    "host_buffer_close_truncate_failed"
                ) from exc
            lane.cache.clear()
            lane.slices.clear()
            if truncated:
                self.unregistrations += 1
            try:
                lane.owner.__del__()
            except BaseException as exc:
                self._poison("host_buffer_close_free_failed")
                raise HostSourcePoolStructuralError("host_buffer_close_free_failed") from exc
            if getattr(lane.owner, "_ptr", None):
                self._poison("host_buffer_close_free_failed")
                raise HostSourcePoolStructuralError("host_buffer_close_free_failed")
            self.frees += 1
            lane.used = 0
            lane.generation += 1
        self._generation += 1
        self._closed = True

    def diagnostics(self) -> dict[str, Any]:
        retained = [item for lane in self._lanes.values() for item in lane.cache.values()]
        retained_ids = {id(item) for item in retained}
        temporary = [
            item
            for lane in self._lanes.values()
            for item in lane.slices
            if not item.released and id(item) not in retained_ids
        ]
        return {
            "generation": self._generation,
            "lane_count": len(self._lanes),
            "capacity_bytes": self.capacity_bytes,
            "retained_slices": len(retained),
            "retained_bytes": sum(item.size for item in retained),
            "temporary_slices": len(temporary),
            "temporary_bytes": sum(item.size for item in temporary),
            "source_hits": self.source_hits,
            "source_misses": self.source_misses,
            "warm_source_hits": self.warm_source_hits,
            "warm_source_misses": self.warm_source_misses,
            "warm_source_bypasses": self.warm_source_bypasses,
            "base_warm_hits": self.base_warm_hits,
            "base_warm_misses": self.base_warm_misses,
            "base_warm_bypasses": self.base_warm_bypasses,
            "patch_warm_hits": self.patch_warm_hits,
            "patch_warm_misses": self.patch_warm_misses,
            "patch_warm_bypasses": self.patch_warm_bypasses,
            "warm_ram_pressure_bypasses": self.warm_ram_pressure_bypasses,
            "warm_zero_delta_extend_refusals": self.warm_zero_delta_extend_refusals,
            "warm_registration_refusals": self.warm_registration_refusals,
            "temporary_ram_pressure_bypasses": self.temporary_ram_pressure_bypasses,
            "temporary_zero_delta_extend_refusals": self.temporary_zero_delta_extend_refusals,
            "temporary_registration_refusals": self.temporary_registration_refusals,
            "stale_rejections": self.stale_rejections,
            "live": self.live,
            "poisoned": self._poisoned,
            "poison_reason": self._poison_reason,
            "transfer_pending": any(
                lease.fences
                for lane in self._lanes.values()
                for lease in self._active_leases_for_lane(lane)
            ),
            "registration_budget_bytes": self.registration_budget_bytes,
            "temporary_reserve_bytes": self.temporary_reserve_bytes,
            "warm_registration_budget_bytes": self.warm_registration_budget_bytes,
            "registration_attempts": self.registration_attempts,
            "registration_attempt_bytes": self.registration_attempt_bytes,
            "registration_successes": self.registration_successes,
            "registration_failures": self.registration_failures,
            "registration_failure_bytes": self.registration_failure_bytes,
            "registration_registered_bytes": self.registration_registered_bytes,
            "registration_unregistered_bytes": self.registration_unregistered_bytes,
            "registration_live_bytes": self.registration_live_bytes,
            "registration_peak_bytes": self.registration_peak_bytes,
            "registration_state_proven": self.registration_state_proven,
        }

    def _active_leases_for_lane(self, lane: _Lane) -> tuple[HostSourceLease, ...]:
        return tuple(
            lease for lease in self._leases.values() if self._lanes.get(lease.source.lane) is lane
        )

    def _reclaim_released_tail(self, lane: _Lane) -> None:
        new_end = lane.used
        while lane.slices and lane.slices[-1].released:
            item = lane.slices.pop()
            item.view = item.view.new_empty((0,))
            new_end = item.offset
        if new_end != lane.used:
            try:
                if not self._shrink_lane(lane, new_end):
                    raise RuntimeError("AIMDO HostBuffer reclaim made no progress")
            except BaseException as exc:
                self.registration_state_proven = False
                self._poison("host_buffer_reclaim_truncate_failed")
                raise HostSourcePoolStructuralError(
                    "host_buffer_reclaim_truncate_failed"
                ) from exc
            lane.used = new_end

    @staticmethod
    def _owner_size(lane: _Lane) -> int:
        """Return the native logical length, with fixture-safe lane fallback."""

        size = getattr(lane.owner, "size", lane.used)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("AIMDO HostBuffer reports an invalid logical size")
        return size

    def _validated_owner_region(self, lane: _Lane) -> tuple[int, int]:
        """Authenticate the existing logical region without retaining its Torch view."""

        owner_size = self._owner_size(lane)
        if owner_size != lane.used:
            raise RuntimeError("AIMDO HostBuffer logical size differs from lane ownership")
        raw_address = lane.owner.get_raw_address()
        if owner_size == 0:
            # AIMDO HostBuffer starts with no logical region even when its
            # native backing has been prewarmed.  In that state the wrapper
            # reports address 0, and hostbuf_to_tensor(0) is invalid in the
            # current native binding.  Treat exactly (size=0, address=0) as
            # the authenticated empty region; any published address without
            # logical bytes is still a state disagreement.
            if (
                not isinstance(raw_address, int)
                or isinstance(raw_address, bool)
                or raw_address != 0
            ):
                raise RuntimeError("AIMDO HostBuffer empty region has an invalid raw address")
            return owner_size, raw_address
        if (
            not isinstance(raw_address, int)
            or isinstance(raw_address, bool)
            or raw_address <= 0
        ):
            raise RuntimeError("AIMDO HostBuffer reports an invalid raw address")
        existing = self._to_tensor(lane.owner)
        try:
            if (
                not isinstance(existing, torch.Tensor)
                or existing.device.type != "cpu"
                or existing.dtype is not torch.uint8
                or not existing.is_contiguous()
                or existing.numel() != owner_size
                or owner_size > 0
                and existing.data_ptr() != raw_address
            ):
                raise RuntimeError("AIMDO HostBuffer existing Torch view is not exact")
        finally:
            existing = None
        return owner_size, raw_address

    def _zero_delta_refusal_proven(
        self,
        lane: _Lane,
        *,
        owner_size: int,
        raw_address: int,
        registered_bytes: int,
        registration_layout: tuple[_Registration, ...],
        registration_live: int,
        slice_count: int,
        cache_count: int,
        lease_fences: tuple[tuple[int, int], ...],
    ) -> bool:
        """Prove one failed ``reallocate=False`` call changed no owned state."""

        try:
            current_owner_size, current_raw_address = self._validated_owner_region(lane)
        except BaseException:  # noqa: BLE001 - proof failure is a structural failure
            return False
        return bool(
            current_owner_size == owner_size
            and current_raw_address == raw_address
            and lane.used == owner_size
            and lane.registered_bytes == registered_bytes
            and tuple(lane.registrations) == registration_layout
            and self.registration_live_bytes == registration_live
            and len(lane.slices) == slice_count
            and len(lane.cache) == cache_count
            and tuple(
                sorted(
                    (lease_id, len(lease.fences))
                    for lease_id, lease in self._leases.items()
                )
            )
            == lease_fences
        )

    def _shrink_lane(self, lane: _Lane, target: int) -> bool:
        """Unregister an exact LIFO suffix, then shrink its logical storage.

        Native AIMDO treats an empty ``truncate(0)`` as a false/no-op result on
        supported Windows builds. Avoid that call entirely. Registrations are
        owned per append, so CUDA unregistration happens explicitly before the
        HostBuffer is truncated with ``do_unregister=False``.
        """

        if target < 0:
            raise ValueError("AIMDO HostBuffer truncate target cannot be negative")
        owner_size = self._owner_size(lane)
        if target > owner_size:
            raise RuntimeError("AIMDO HostBuffer truncate target exceeds logical size")
        registrations_before = len(lane.registrations)
        self._unregister_tail(lane, target)
        if (
            owner_size == target
            and lane.used == target
            and lane.registered_bytes == 0
        ):
            return registrations_before != len(lane.registrations)
        lane.owner.truncate(target, do_unregister=False)
        return True

    def _truncate_unregistered_lane(self, lane: _Lane, target: int) -> None:
        """Rollback one unpublished append without asking native to unregister."""

        if target < 0 or target > self._owner_size(lane):
            raise RuntimeError("AIMDO HostBuffer rollback target is invalid")
        lane.owner.truncate(target, do_unregister=False)

    def _unregister_tail(self, lane: _Lane, target: int) -> None:
        while lane.registrations and lane.registrations[-1].offset >= target:
            registration = lane.registrations[-1]
            result = self._cuda_runtime().cudaHostUnregister(registration.address)
            if result != 0:
                self._discard_cuda_registration_error()
                self.registration_state_proven = False
                self._poison("cuda_host_unregister_failed")
                raise HostSourcePoolStructuralError("cuda_host_unregister_failed")
            lane.registrations.pop()
            self._record_unregister(lane, registration.size)
        if lane.registrations:
            tail = lane.registrations[-1]
            if tail.offset + tail.size > target:
                self.registration_state_proven = False
                self._poison("host_registration_suffix_mismatch")
                raise HostSourcePoolStructuralError("host_registration_suffix_mismatch")

    def _rollback_unregistered_append(
        self,
        lane: _Lane,
        *,
        start: int,
        owner_size: int,
        raw_address: int,
        registered_bytes: int,
        registration_layout: tuple[_Registration, ...],
        registration_live: int,
        slice_count: int,
        cache_count: int,
        lease_fences: tuple[tuple[int, int], ...],
    ) -> None:
        self._truncate_unregistered_lane(lane, start)
        if not self._zero_delta_refusal_proven(
            lane,
            owner_size=owner_size,
            raw_address=raw_address,
            registered_bytes=registered_bytes,
            registration_layout=registration_layout,
            registration_live=registration_live,
            slice_count=slice_count,
            cache_count=cache_count,
            lease_fences=lease_fences,
        ):
            raise RuntimeError("AIMDO HostBuffer registration rollback was not exact")

    def _discard_cuda_registration_error(self) -> None:
        get_last_error = getattr(self._cuda_runtime(), "cudaGetLastError", None)
        if callable(get_last_error):
            get_last_error()

    def _cuda_runtime(self) -> Any:
        if self._cudart is None:
            self._cudart = torch.cuda.cudart()
        return self._cudart

    def _drop_lane_views(self, lane: _Lane) -> None:
        unique: dict[int, _Slice] = {id(item): item for item in lane.slices}
        unique.update((id(item), item) for item in lane.cache.values())
        unique.update(
            (id(lease.source), lease.source)
            for lease in self._active_leases_for_lane(lane)
        )
        for source in unique.values():
            source.view = source.view.new_empty((0,))
            source.released = True

    def _validate_lease(self, lease: HostSourceLease) -> None:
        self._require_open()
        if lease.complete:
            raise RuntimeError("AIMDO host source lease is already complete")
        self._validate_slice(lease.source)

    def _validate_slice(self, source: _Slice) -> None:
        try:
            lane = self._lanes[source.lane]
        except KeyError as exc:
            self.stale_rejections += 1
            raise RuntimeError("AIMDO host source lease belongs to a stale lane") from exc
        if source.generation != lane.generation or source.released:
            self.stale_rejections += 1
            raise RuntimeError("AIMDO host source lease generation is stale")

    def _require_open(self) -> None:
        if self._poisoned:
            raise HostSourcePoolStructuralError(self._poison_reason or "unknown")
        if self._closed:
            raise RuntimeError("AIMDO host source pool is closed")

    def _record_unregister(self, lane: _Lane, size: int) -> None:
        if size < 0 or size > lane.registered_bytes:
            self.registration_state_proven = False
            self._poison("host_registration_accounting_failed")
            raise HostSourcePoolStructuralError("host_registration_accounting_failed")
        lane.registered_bytes -= size
        self.registration_live_bytes -= size
        self.registration_unregistered_bytes += size

    def _poison(self, reason: str) -> None:
        self._poisoned = True
        if self._poison_reason is None:
            self._poison_reason = reason

    def _record_warm(self, source_class: HostSourceClass, outcome: str) -> None:
        suffix = {"hit": "hits", "miss": "misses", "bypass": "bypasses"}.get(outcome)
        if suffix is None:
            raise ValueError("AIMDO warm source outcome is not canonical")
        total = f"warm_source_{suffix}"
        setattr(self, total, getattr(self, total) + 1)
        prefix = "base" if source_class is HostSourceClass.BASE else "patch"
        field = f"{prefix}_warm_{suffix}"
        setattr(self, field, getattr(self, field) + 1)
