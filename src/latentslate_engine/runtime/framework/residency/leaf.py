"""Model-neutral path-addressed scheduling over dynamic residency backends.

The backend owns VBAR allocations and transfer lifetimes.  This module owns
only deterministic leaf ordering, scheduling-group prefetch, binding, and
release.  Model adapters provide the logical values and binding callbacks.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .dynamic import (
    DynamicResidencyBackend,
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
)


@dataclass(frozen=True, slots=True)
class LeafResidencyDescriptor:
    """One stable allocation and every scheduling group that consumes it."""

    path: str
    schedule_groups: tuple[str, ...]
    values: tuple[Any, ...]
    physical_bytes: int
    force_resident: bool = False


@dataclass(slots=True)
class _LeafLease:
    descriptor: LeafResidencyDescriptor
    lease: DynamicResidencyLease
    binding: Any | None = None


class LeafResidencyScheduler:
    """Schedule path-addressed leaves with one-group-ahead transfer prefetch."""

    def __init__(
        self,
        backend: DynamicResidencyBackend,
        descriptors: Iterable[LeafResidencyDescriptor],
        *,
        schedule_order: tuple[str, ...],
        activate: Callable[[LeafResidencyDescriptor, tuple[Any, ...]], Any],
        restore: Callable[[LeafResidencyDescriptor, Any], None],
    ) -> None:
        ordered = tuple(descriptors)
        if not ordered:
            raise ValueError("leaf residency requires at least one descriptor")
        if not schedule_order or len(set(schedule_order)) != len(schedule_order):
            raise ValueError("leaf residency schedule order must be unique")
        allowed = set(schedule_order)
        paths: set[str] = set()
        groups: dict[str, list[str]] = {name: [] for name in schedule_order}
        for descriptor in ordered:
            if (
                not descriptor.path
                or descriptor.path in paths
                or not descriptor.schedule_groups
                or any(group not in allowed for group in descriptor.schedule_groups)
                or descriptor.physical_bytes <= 0
            ):
                raise ValueError("leaf residency descriptor is not canonical")
            paths.add(descriptor.path)
            for group in descriptor.schedule_groups:
                groups[group].append(descriptor.path)
            backend.allocate_group(descriptor.path, descriptor.values)
        self._backend = backend
        self._descriptors = OrderedDict((item.path, item) for item in ordered)
        self._groups = {key: tuple(value) for key, value in groups.items()}
        self._schedule_order = schedule_order
        self._activate = activate
        self._restore = restore
        self._leases: dict[str, _LeafLease] = {}
        self._active_groups: set[str] = set()
        self._prefetched_group: str | None = None
        self._closed = False
        self._terminal_poison_reason: str | None = None
        self.prefetch_groups = 0
        self.prefetch_leaves = 0
        self.deferred_waits = 0
        self.force_resident_waits = 0
        self.consumed_groups = 0
        self.force_resident_leaves = sum(item.force_resident for item in ordered)

    @property
    def descriptors(self) -> tuple[LeafResidencyDescriptor, ...]:
        return tuple(self._descriptors.values())

    @property
    def allocation_count(self) -> int:
        return len(self._descriptors)

    def terminal_poison_reason(self) -> str | None:
        """Return the captured backend poison without touching native state."""

        return self._terminal_poison_reason

    def mark_terminal_poison(self, reason: str) -> None:
        """Freeze exact ownership after an outer device barrier fails."""

        if not reason or self._closed:
            raise RuntimeError("leaf residency terminal poison is not canonical")
        if self._terminal_poison_reason is None:
            self._terminal_poison_reason = reason
        elif self._terminal_poison_reason != reason:
            raise RuntimeError("leaf residency terminal poison reason changed")

    def onload(self) -> None:
        """Bind tiny/cross-group leaves for the stage lifetime."""

        for descriptor in self._descriptors.values():
            if descriptor.force_resident:
                self._acquire(descriptor, prefetched=False, activate=True)

    def prefetch(self, group: str) -> None:
        self._require_open()
        if group not in self._groups:
            raise KeyError(group)
        if self._prefetched_group not in {None, group}:
            raise RuntimeError("leaf residency already owns another prefetched group")
        started = 0
        try:
            for path in self._groups[group]:
                descriptor = self._descriptors[path]
                if descriptor.force_resident or path in self._leases:
                    continue
                self._acquire(descriptor, prefetched=True, activate=False)
                started += 1
        except DynamicResidencyPoisoned as poison:
            self._mark_poisoned(poison)
            raise
        except BaseException as primary:
            try:
                self._discard_unbound(group)
            except DynamicResidencyPoisoned as poison:
                primary.add_note(f"leaf prefetch cleanup poisoned: {poison.reason}")
                raise poison from primary
            raise
        self._prefetched_group = group
        self.prefetch_groups += 1
        self.prefetch_leaves += started

    def enter(self, group: str) -> None:
        self._require_open()
        if group not in self._groups or group in self._active_groups:
            raise RuntimeError("leaf residency scheduling group cannot be entered")
        if self._prefetched_group not in {None, group}:
            raise RuntimeError("leaf residency entered ahead of its prefetched group")
        activated: list[str] = []
        try:
            for path in self._groups[group]:
                descriptor = self._descriptors[path]
                entry = self._leases.get(path)
                if entry is None:
                    entry = self._acquire(descriptor, prefetched=False, activate=False)
                if entry.binding is None:
                    self._backend_call("wait", entry.lease)
                    self.deferred_waits += 1
                    entry.binding = self._activate(descriptor, entry.lease.values)
                    activated.append(path)
            self._active_groups.add(group)
            if self._prefetched_group == group:
                self._prefetched_group = None
            self.consumed_groups += 1
        except DynamicResidencyPoisoned as poison:
            self._mark_poisoned(poison)
            raise
        except BaseException as primary:
            for path in reversed(activated):
                entry = self._leases[path]
                self._restore(entry.descriptor, entry.binding)
                entry.binding = None
            try:
                self._discard_unbound(group)
            except DynamicResidencyPoisoned as poison:
                primary.add_note(f"leaf enter cleanup poisoned: {poison.reason}")
                raise poison from primary
            raise

    def leave(self, group: str) -> None:
        self._require_open()
        if group not in self._active_groups:
            raise RuntimeError("leaf residency scheduling group is not active")
        release_paths: list[str] = []
        for path in reversed(self._groups[group]):
            descriptor = self._descriptors[path]
            if descriptor.force_resident:
                continue
            # Cross-group aliases remain active until their final active group.
            if any(
                other != group and other in self._active_groups
                for other in descriptor.schedule_groups
            ):
                continue
            release_paths.append(path)
        self._release_group(tuple(release_paths))
        self._active_groups.remove(group)

    def clear_stage(self, *, release_force_resident: bool = False) -> None:
        """Clear temporary scheduling state while preserving warm signatures."""

        self._require_open()
        self._drain(release_force_resident=release_force_resident)

    def prepare_stage(self, required_free_bytes: int) -> None:
        """Establish one explicit model-neutral component stage boundary."""

        self._require_open()
        if (
            not isinstance(required_free_bytes, int)
            or isinstance(required_free_bytes, bool)
            or required_free_bytes < 0
        ):
            raise ValueError("leaf residency stage requirement must be a non-negative integer")
        self._drain(release_force_resident=False)
        self._backend_call("prepare_stage", required_free_bytes)

    def _drain(self, *, release_force_resident: bool) -> None:
        for group in reversed(self._schedule_order):
            if group in self._active_groups:
                self.leave(group)
        if self._prefetched_group is not None:
            self._discard_unbound(self._prefetched_group)
            self._prefetched_group = None
        remaining = tuple(
            path
            for path in self._leases
            if release_force_resident or not self._descriptors[path].force_resident
        )
        self._release_group(remaining)

    def invalidate(
        self, *, reason: str, paths: tuple[str, ...] | None = None
    ) -> None:
        self._require_open()
        self._drain(release_force_resident=True)
        if paths is None:
            self._backend_call("invalidate", reason=reason)
        else:
            if not paths or len(set(paths)) != len(paths):
                raise ValueError("leaf residency invalidation paths must be unique")
            if any(path not in self._descriptors for path in paths):
                raise KeyError("leaf residency invalidation path is unknown")
            self._backend_call("invalidate_groups", paths, reason=reason)
        self.onload()

    def close(self) -> None:
        if self._closed:
            return
        self._require_open()
        self._drain(release_force_resident=True)
        self._backend_call("close")
        self._closed = True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "leaf_allocation_count": self.allocation_count,
            "force_resident_leaf_count": self.force_resident_leaves,
            "schedule_group_count": len(self._schedule_order),
            "prefetch_groups": self.prefetch_groups,
            "prefetch_leaves": self.prefetch_leaves,
            "deferred_waits": self.deferred_waits,
            "force_resident_waits": self.force_resident_waits,
            "consumed_groups": self.consumed_groups,
            "active_groups": len(self._active_groups),
            "pending_prefetch": self._prefetched_group is not None,
        }

    def _acquire(
        self,
        descriptor: LeafResidencyDescriptor,
        *,
        prefetched: bool,
        activate: bool,
    ) -> _LeafLease:
        lease = (
            self._backend_call("prefetch", descriptor.path)
            if prefetched
            else self._backend_call("acquire", descriptor.path)
        )
        entry = _LeafLease(descriptor, lease)
        self._leases[descriptor.path] = entry
        if activate:
            self._backend_call("wait", lease)
            if not descriptor.force_resident:
                raise RuntimeError("eager leaf activation requires force residency")
            self.force_resident_waits += 1
            entry.binding = self._activate(descriptor, lease.values)
        return entry

    def _release(self, path: str) -> None:
        entry = self._leases[path]
        primary: BaseException | None = None
        try:
            self._backend_call("synchronize", entry.lease)
        except BaseException as exc:  # backend poison must retain exact ownership
            if isinstance(exc, DynamicResidencyPoisoned):
                raise
            primary = exc
        try:
            if entry.binding is not None:
                self._restore(entry.descriptor, entry.binding)
                entry.binding = None
            self._backend_call("release", entry.lease)
            self._leases.pop(path)
        except DynamicResidencyPoisoned as poison:
            if primary is not None:
                primary.add_note(f"leaf release cleanup poisoned: {poison.reason}")
                raise poison from primary
            raise
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "leaf residency cleanup after a proven fallback barrier also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if primary is not None:
            raise primary

    def _release_group(self, paths: tuple[str, ...]) -> None:
        if not paths:
            return
        entries = tuple(self._leases[path] for path in paths)
        release_group = getattr(self._backend, "release_group", None)
        if not callable(release_group):
            for path in paths:
                self._release(path)
            return
        for entry in entries:
            if entry.binding is not None:
                self._restore(entry.descriptor, entry.binding)
                entry.binding = None
        self._backend_call("release_group", tuple(entry.lease for entry in entries))
        for path in paths:
            self._leases.pop(path)

    def _discard_unbound(self, group: str) -> None:
        paths = tuple(
            path
            for path in reversed(self._groups[group])
            if (entry := self._leases.get(path)) is not None and entry.binding is None
        )
        self._release_group(paths)

    def _require_open(self) -> None:
        if self._terminal_poison_reason is not None:
            raise DynamicResidencyPoisoned(self._terminal_poison_reason)
        if self._closed:
            raise RuntimeError("leaf residency scheduler is closed")

    def _backend_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._require_open()
        try:
            return getattr(self._backend, method)(*args, **kwargs)
        except DynamicResidencyPoisoned as poison:
            self._mark_poisoned(poison)
            raise

    def _mark_poisoned(self, poison: DynamicResidencyPoisoned) -> None:
        if self._terminal_poison_reason is None:
            self._terminal_poison_reason = poison.reason
