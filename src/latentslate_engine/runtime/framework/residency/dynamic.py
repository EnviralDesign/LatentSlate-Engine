"""Model-neutral contracts for operation-scoped dynamic weight residency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class DynamicResidencyUnavailable(RuntimeError):
    """A dynamic backend was unavailable before it allocated runtime state."""


class DynamicResidencyPoisoned(RuntimeError):
    """GPU residency could not quiesce and requires immediate hard child exit."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"dynamic residency poisoned: {reason}")


class DynamicResidencyDeviceError(RuntimeError):
    """Requested CUDA identity is invalid and must not silently fall back."""


@dataclass(frozen=True, slots=True)
class DynamicResidencyLease:
    """One acquired logical-value group and its backend-owned lifetime token."""

    values: tuple[Any, ...]
    token: object


@runtime_checkable
class DynamicResidencyBackend(Protocol):
    """Small backend boundary used by model-owned module binding code.

    Implementations own virtual allocations and transfer lifetimes. Model
    adapters continue to own module grouping, slot assignment, patch epochs,
    forward hooks, and execution policy.
    """

    @property
    def allocation_started(self) -> bool: ...

    @property
    def required_virtual_bytes(self) -> int: ...

    def allocate_group(self, key: object, values: tuple[Any, ...]) -> None: ...

    def prioritize(self) -> None: ...

    def acquire(self, key: object) -> DynamicResidencyLease: ...

    def prefetch(self, key: object) -> DynamicResidencyLease: ...

    def wait(self, lease: DynamicResidencyLease) -> None: ...

    def synchronize(self, lease: DynamicResidencyLease) -> None: ...

    def release(self, lease: DynamicResidencyLease) -> None: ...

    def release_group(self, leases: tuple[DynamicResidencyLease, ...]) -> None: ...

    def prepare_stage(self, required_free_bytes: int) -> None: ...

    def invalidate(self, *, reason: str) -> None: ...

    def invalidate_groups(self, keys: tuple[object, ...], *, reason: str) -> None: ...

    def diagnostics(self) -> dict[str, Any]: ...

    def terminal_poison_reason(self) -> str | None: ...

    def close(self) -> None: ...
