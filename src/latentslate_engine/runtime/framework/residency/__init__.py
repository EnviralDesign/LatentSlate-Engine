"""Model-neutral residency coordination primitives."""

from .session import GLOBAL_STORED_RESIDENCY_LEASE, ExclusiveResidencyLease, canonical_device

__all__ = (
    "GLOBAL_STORED_RESIDENCY_LEASE",
    "ExclusiveResidencyLease",
    "canonical_device",
)
