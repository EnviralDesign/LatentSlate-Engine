from __future__ import annotations

from latentslate_engine.wan22_recipe import Wan22RuntimeRequest, wan22_i2v_operation


def _request(configured_loras):
    return Wan22RuntimeRequest(
        4,
        "wan22",
        "wan22_14b_36ch_40block_out16",
        "wan22-14b-i2v",
        {},
        {},
        operation="wan22_i2v_base",
        configured_loras=configured_loras,
    )


def test_disabled_configured_stack_is_part_of_native_wan_request_fingerprint() -> None:
    base = _request(())
    disabled = _request(
        (
            {
                "slot": "high_noise",
                "stage": "high",
                "resource_reference": "lora:wan22:not-installed",
                "strength": 0.0,
                "active": False,
            },
        )
    )

    assert base.fingerprint != disabled.fingerprint


def test_neutral_wan_operation_identity_preserves_accepted_execution_contracts() -> None:
    """Schema revision 2 renamed provenance identities, not accepted runtime math."""

    expected = {
        "wan22_i2v_base": (20, 3.5, 3.5, 5.0),
        "wan22_i2v_lightx2v_4step": (4, 1.0, 1.0, 5.0),
        "wan22_flf_base": (20, 4.0, 4.0, 8.0),
        "wan22_flf_lightx2v_4step": (4, 1.0, 1.0, 5.0),
        "wan22_t2v_base": (20, 3.5, 3.5, 5.0),
        "wan22_t2v_lightx2v_4step": (4, 1.0, 1.0, 5.0),
    }
    for operation, contract in expected.items():
        resolved = wan22_i2v_operation(operation)
        assert resolved["stage_policy"] == "expert_split"
        assert resolved["sampler"] == "euler"
        assert resolved["scheduler"] == "simple"
        assert (
            resolved["steps"],
            resolved["high_guidance"],
            resolved["low_guidance"],
            resolved["shift"],
        ) == contract
