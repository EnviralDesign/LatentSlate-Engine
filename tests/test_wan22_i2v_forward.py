from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers.hooks import HookRegistry

from latentslate_engine.runtime.wan22_i2v_forward import WanI2VForward


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call = None
        self._latentslate_compute_dtype = torch.float16

    def forward(self, **kwargs):
        self.call = kwargs
        hidden = kwargs["hidden_states"]
        return (
            torch.full((1, 16, *hidden.shape[2:]), 0.5, dtype=hidden.dtype, device=hidden.device),
        )


def _session(model):
    return SimpleNamespace(
        transformer=model,
        active=True,
        onload_device=torch.device("cpu"),
    )


def test_forward_builds_36_channels_and_uses_cache_identity():
    model = FakeTransformer()
    cache_events = []
    registry = HookRegistry.check_if_exists_or_initialize(model)
    registry._set_context = lambda identity: cache_events.append(identity)
    forward = WanI2VForward(torch.zeros((1, 20, 2, 4, 4), dtype=torch.float32))
    latents = torch.ones((1, 16, 2, 4, 4), dtype=torch.float32)
    prompt = torch.ones((1, 512, 4096), dtype=torch.float16)

    result = forward(model, _session(model), latents, torch.tensor(999), prompt, "cond")

    assert result.shape == latents.shape
    assert result.dtype == torch.float16
    assert model.call["hidden_states"].shape == (1, 36, 2, 4, 4)
    assert model.call["hidden_states"].dtype == torch.float16
    assert model.call["timestep"].shape == (1,)
    assert model.call["encoder_hidden_states"] is prompt
    assert model.call["encoder_hidden_states_image"] is None
    assert model.call["attention_kwargs"] is None
    assert model.call["return_dict"] is False
    assert cache_events == ["cond", None]


def test_forward_rejects_wrong_session_dtype_and_conditioning():
    model = FakeTransformer()
    forward = WanI2VForward(torch.zeros((1, 20, 2, 4, 4), dtype=torch.float32))
    latents = torch.zeros((1, 16, 2, 4, 4), dtype=torch.float32)
    prompt = torch.zeros((1, 512, 4096), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="active matching"):
        forward(model, _session(object()), latents, 1, prompt, "cond")
    model._latentslate_compute_dtype = torch.bfloat16
    with pytest.raises(RuntimeError, match="dtype/device"):
        forward(model, _session(model), latents, 1, prompt, "cond")
    model._latentslate_compute_dtype = torch.float16
    with pytest.raises(ValueError, match="cache identity"):
        forward(model, _session(model), latents, 1, prompt, "other")
    with pytest.raises(ValueError, match="prompt conditioning"):
        forward(model, _session(model), latents, 1, torch.zeros((1, 1, 1)), "cond")


def test_forward_rejects_bad_timestep_and_output():
    model = FakeTransformer()
    forward = WanI2VForward(torch.zeros((1, 20, 2, 4, 4), dtype=torch.float32))
    latents = torch.zeros((1, 16, 2, 4, 4), dtype=torch.float32)
    prompt = torch.zeros((1, 512, 4096), dtype=torch.float16)

    with pytest.raises(ValueError, match="finite scalar"):
        forward(model, _session(model), latents, torch.ones(2), prompt, "cond")

    model.forward = lambda **_kwargs: (torch.zeros((1, 1)),)
    with pytest.raises(RuntimeError, match="incompatible"):
        forward(model, _session(model), latents, 1, prompt, "cond")


def test_cache_context_is_cleared_when_model_raises_baseexception():
    model = FakeTransformer()
    forward = WanI2VForward(torch.zeros((1, 20, 2, 4, 4), dtype=torch.float32))
    events = []
    registry = HookRegistry.check_if_exists_or_initialize(model)
    registry._set_context = lambda identity: events.append(identity)

    def fail(**_kwargs):
        raise MemoryError("simulated OOM")

    model.forward = fail
    with pytest.raises(MemoryError, match="simulated OOM"):
        forward(
            model,
            _session(model),
            torch.zeros((1, 16, 2, 4, 4), dtype=torch.float32),
            1,
            torch.zeros((1, 512, 4096), dtype=torch.float16),
            "uncond",
        )
    assert events == ["uncond", None]
