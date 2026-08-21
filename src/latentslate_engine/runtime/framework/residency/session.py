"""Small process-local coordination seams for tensor residency sessions."""

from __future__ import annotations

import threading
from typing import Any


class ExclusiveResidencyLease:
    """Permit one active owner across cooperating residency strategies."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: object | None = None

    def claim(self, owner: object) -> None:
        if owner is None:
            raise TypeError("residency lease owner cannot be None")
        with self._lock:
            if self._owner is not None:
                raise RuntimeError("another residency session is already active process-wide")
            self._owner = owner

    def release(self, owner: object) -> bool:
        """Release only the matching owner; return whether the lease changed."""

        with self._lock:
            if self._owner is not owner:
                return False
            self._owner = None
            return True

    @property
    def active(self) -> bool:
        with self._lock:
            return self._owner is not None


GLOBAL_STORED_RESIDENCY_LEASE = ExclusiveResidencyLease()


def canonical_device(device: Any):
    """Resolve an index-free CUDA request once for exact device comparisons."""

    import torch

    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return target
