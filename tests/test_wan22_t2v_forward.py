from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers.hooks import HookRegistry

from latentslate_engine.runtime.wan22_t2v_forward import WanT2VForward
from latentslate_engine.runtime.wan22_t2v_runtime import WanT2VRequest, validate_wan_t2v_request


class _Transformer(torch.nn.Module):
    def forward(self, **kwargs):
        self.call = kwargs
        hidden = kwargs["hidden_states"]
        return (torch.zeros_like(hidden, dtype=torch.float16),)


def test_t2v_forward_never_adds_i2v_condition_channels() -> None:
    model = _Transformer()
    registry = HookRegistry.check_if_exists_or_initialize(model)
    registry._set_context = lambda _identity: None
    session = SimpleNamespace(transformer=model, active=True, onload_device=torch.device("cpu"))
    latents = torch.zeros((1, 16, 2, 4, 4), dtype=torch.float32)
    result = WanT2VForward()(model, session, latents, 1, torch.zeros((1, 512, 4096), dtype=torch.float16), "cond")
    assert result.shape == latents.shape
    assert model.call["hidden_states"].shape == (1, 16, 2, 4, 4)


def test_t2v_request_validation_fails_before_materialization() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_wan_t2v_request(WanT2VRequest(prompt=" ", num_frames=5, width=64, height=64))
    with pytest.raises(ValueError, match=r"4k\+1"):
        validate_wan_t2v_request(WanT2VRequest(prompt="move", num_frames=6, width=64, height=64))


@pytest.mark.parametrize("steps", [True, 1.5, 1, 1001])
def test_t2v_request_rejects_noncanonical_step_budgets(steps: object) -> None:
    with pytest.raises((TypeError, ValueError), match="steps"):
        validate_wan_t2v_request(
            WanT2VRequest(prompt="move", num_frames=5, width=64, height=64, steps=steps)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["high_guidance", "low_guidance"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_t2v_request_rejects_nonfinite_or_boolean_guidance(field: str, value: object) -> None:
    request = WanT2VRequest(prompt="move", num_frames=5, width=64, height=64)
    with pytest.raises((TypeError, ValueError), match="guidance"):
        validate_wan_t2v_request(replace(request, **{field: value}))  # type: ignore[arg-type]
