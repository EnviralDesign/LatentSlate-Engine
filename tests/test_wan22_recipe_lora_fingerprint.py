from __future__ import annotations

from latentslate_engine.wan22_recipe import Wan22RuntimeRequest


def _request(configured_loras):
    return Wan22RuntimeRequest(
        4,
        "wan22",
        "wan22_14b_36ch_40block_out16",
        "wan22-14b-i2v",
        {},
        {},
        operation="comfy_i2v_base",
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
