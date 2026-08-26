"""Model-neutral residency coordination and dynamic-weight primitives."""

from .dynamic import (
    DynamicResidencyBackend,
    DynamicResidencyDeviceError,
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
    DynamicResidencyUnavailable,
)
from .leaf import LeafResidencyDescriptor, LeafResidencyScheduler
from .session import GLOBAL_STORED_RESIDENCY_LEASE, ExclusiveResidencyLease, canonical_device

__all__ = (
    "GLOBAL_STORED_RESIDENCY_LEASE",
    "DynamicResidencyBackend",
    "DynamicResidencyDeviceError",
    "DynamicResidencyLease",
    "DynamicResidencyPoisoned",
    "DynamicResidencyUnavailable",
    "ExclusiveResidencyLease",
    "LeafResidencyDescriptor",
    "LeafResidencyScheduler",
    "canonical_device",
)
