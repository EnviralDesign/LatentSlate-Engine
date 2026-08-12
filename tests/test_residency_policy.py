import pytest

from latentslate_engine.runtime.residency_policy import (
    choose_cuda_residency,
    require_grouped_residency,
)


def test_residency_policy_retains_activation_headroom_before_full_onload():
    gib = 1024**3
    decision = choose_cuda_residency(
        free_bytes=12 * gib,
        total_bytes=16 * gib,
        stored_bytes=4 * gib,
        largest_group_bytes=gib // 2,
    )
    assert decision.mode == "grouped"
    assert decision.reserved_headroom_bytes == int(16 * gib * 0.60)
    assert decision.stream_buffer_bytes == gib // 2
    assert decision.resident_weight_budget_bytes == (
        12 * gib - int(16 * gib * 0.60) - gib // 2
    )


def test_residency_policy_allows_full_onload_when_headroom_remains():
    gib = 1024**3
    decision = choose_cuda_residency(
        free_bytes=70 * gib,
        total_bytes=80 * gib,
        stored_bytes=16 * gib,
    )
    assert decision.mode == "full"
    assert decision.stream_buffer_bytes == 0
    assert decision.resident_weight_budget_bytes == 16 * gib
    assert "retains activation headroom" in decision.reason


def test_clean_5080_base_i2i_cannot_select_full_residency():
    total = int(15.9 * 1024**3)
    stored = 4_089_498_488
    largest_block = 256 * 1024**2
    adaptive = choose_cuda_residency(
        free_bytes=total,
        total_bytes=total,
        stored_bytes=stored,
        largest_group_bytes=largest_block,
    )
    assert adaptive.mode == "full"  # Demonstrates why free-memory inference is unsafe.

    i2i = require_grouped_residency(
        adaptive,
        largest_group_bytes=largest_block,
        reason="stored-FP8 I2I requires partial residency",
    )
    assert i2i.mode == "grouped"
    assert i2i.stream_buffer_bytes == largest_block
    assert i2i.resident_weight_budget_bytes <= stored - largest_block


@pytest.mark.parametrize(
    "kwargs",
    [
        {"free_bytes": -1, "total_bytes": 1, "stored_bytes": 1},
        {"free_bytes": 2, "total_bytes": 1, "stored_bytes": 1},
    ],
)
def test_residency_policy_rejects_invalid_capacity(kwargs):
    with pytest.raises(ValueError, match="invalid CUDA residency capacity"):
        choose_cuda_residency(**kwargs)
