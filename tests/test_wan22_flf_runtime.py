from __future__ import annotations

import pytest

from latentslate_engine.runtime.wan22_flf_runtime import (
    WanFLFRequest,
    validate_wan_flf_request,
)


def test_flf_lightx_request_keeps_the_pinned_four_step_contract() -> None:
    request = WanFLFRequest(
        start_image=None,
        end_image=None,
        prompt="transition",
        steps=4,
        high_guidance=1.0,
        low_guidance=1.0,
        operation="comfy_i2v_flf_lightx2v_4step",
    )

    validate_wan_flf_request(request)

    with pytest.raises(ValueError, match="low_guidance=1.0"):
        validate_wan_flf_request(
            WanFLFRequest(
                start_image=None,
                end_image=None,
                prompt="transition",
                steps=4,
                high_guidance=1.0,
                low_guidance=4.0,
                operation="comfy_i2v_flf_lightx2v_4step",
            )
        )
