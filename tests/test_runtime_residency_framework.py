from __future__ import annotations

import pytest
import torch

from latentslate_engine.runtime import (
    klein_stored_adapter,
    umt5_stored_adapter,
    wan21_vae_adapter,
    wan22_stored_adapter,
)
from latentslate_engine.runtime.framework.residency import (
    GLOBAL_STORED_RESIDENCY_LEASE,
    ExclusiveResidencyLease,
    canonical_device,
)


def test_exclusive_residency_lease_requires_matching_owner():
    lease = ExclusiveResidencyLease()
    first = object()
    second = object()

    lease.claim(first)
    assert lease.active
    with pytest.raises(RuntimeError, match="already active process-wide"):
        lease.claim(second)
    assert lease.release(second) is False
    assert lease.active
    assert lease.release(first) is True
    assert not lease.active


def test_exclusive_residency_lease_rejects_none_owner():
    lease = ExclusiveResidencyLease()
    with pytest.raises(TypeError, match="cannot be None"):
        lease.claim(None)


def test_canonical_device_preserves_non_cuda_and_explicit_cuda(monkeypatch):
    assert canonical_device("cpu") == torch.device("cpu")
    assert canonical_device("cuda:3") == torch.device("cuda:3")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert canonical_device("cuda") == torch.device("cuda:2")


def test_all_stored_residency_families_bind_the_same_process_wide_lease():
    for module in (
        klein_stored_adapter,
        umt5_stored_adapter,
        wan21_vae_adapter,
        wan22_stored_adapter,
    ):
        assert module.GLOBAL_STORED_RESIDENCY_LEASE is GLOBAL_STORED_RESIDENCY_LEASE


def test_klein_owner_blocks_wan_owner_through_shared_lease():
    klein_owner = object()
    wan_owner = object()
    try:
        klein_stored_adapter.KleinTransformerResidencySession._claim_global(klein_owner)
        with pytest.raises(RuntimeError, match="Wan residency session is already active"):
            wan22_stored_adapter.WanTransformerResidencySession._claim_transformer(wan_owner)
    finally:
        klein_stored_adapter.KleinTransformerResidencySession._release_global(klein_owner)
    assert not GLOBAL_STORED_RESIDENCY_LEASE.active
