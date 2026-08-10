from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
import torch
from diffusers import UniPCMultistepScheduler

from latentslate_engine.runtime.wan22_i2v_orchestration import (
    StagePolicy,
    coordinate_denoise,
)


class AdditiveScheduler:
    def step(self, prediction, _timestep, latents, *, return_dict):
        assert return_dict is False
        return (latents + prediction,)


def test_comfy_and_diffusers_stage_policies_match_reference_counts():
    assert StagePolicy("comfy_split").assignments(tuple(range(20, 0, -1))).count("high") == 10
    assert StagePolicy("comfy_split").assignments((5, 4, 3, 2, 1)) == (
        "high",
        "high",
        "high",
        "low",
        "low",
    )

    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    scheduler.set_timesteps(20)
    assignments = StagePolicy("diffusers_boundary", boundary_ratio=0.9).assignments(
        tuple(scheduler.timesteps)
    )
    assert assignments.count("high") == 6
    assert assignments.count("low") == 14


def test_coordinator_owns_one_contiguous_session_and_applies_cfg():
    events = []
    active = []
    cache_identities = []

    @contextmanager
    def session_factory(model, stage):
        assert not active
        active.append(model)
        events.append(("enter", stage))
        try:
            yield object()
        finally:
            events.append(("exit", stage))
            active.clear()

    def forward(model, _session, latents, timestep, conditioning, cache_identity):
        assert active == [model]
        cache_identities.append((model, cache_identity))
        value = conditioning[model]
        return torch.full_like(latents, value)

    latents = torch.zeros((1, 1))
    result = coordinate_denoise(
        latents=latents,
        timesteps=(4, 3, 2, 1),
        policy=StagePolicy("comfy_split"),
        high_model="high",
        low_model="low",
        session_factory=session_factory,
        forward=forward,
        scheduler=AdditiveScheduler(),
        conditional={"high": 3.0, "low": 5.0},
        unconditional={"high": 1.0, "low": 2.0},
        high_guidance=2.0,
        low_guidance=3.0,
    )

    # Two high predictions are 1 + 2*(3-1) = 5. Two low are 2 + 3*(5-2) = 11.
    assert torch.equal(result, torch.tensor([[32.0]]))
    assert events == [("enter", "high"), ("exit", "high"), ("enter", "low"), ("exit", "low")]
    assert cache_identities == [
        ("high", "cond"),
        ("high", "uncond"),
        ("high", "cond"),
        ("high", "uncond"),
        ("low", "cond"),
        ("low", "uncond"),
        ("low", "cond"),
        ("low", "uncond"),
    ]
    assert not active


def test_guidance_one_skips_unconditional_forward():
    identities = []

    @contextmanager
    def session_factory(_model, _stage):
        yield object()

    def forward(_model, _session, latents, _timestep, _conditioning, cache_identity):
        identities.append(cache_identity)
        return torch.ones_like(latents)

    coordinate_denoise(
        latents=torch.zeros((1,)),
        timesteps=(2, 1),
        policy=StagePolicy("comfy_split"),
        high_model="high",
        low_model="low",
        session_factory=session_factory,
        forward=forward,
        scheduler=AdditiveScheduler(),
        conditional=object(),
        high_guidance=1.0,
        low_guidance=1.0,
    )
    assert identities == ["cond", "cond"]


def test_pinned_unipc_accepts_fp16_predictions_with_fp32_latents():
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    scheduler.set_timesteps(2)

    @contextmanager
    def session_factory(_model, _stage):
        yield object()

    result = coordinate_denoise(
        latents=torch.zeros((1, 2), dtype=torch.float32),
        timesteps=tuple(scheduler.timesteps),
        policy=StagePolicy("comfy_split"),
        high_model="high",
        low_model="low",
        session_factory=session_factory,
        forward=lambda *_: torch.zeros((1, 2), dtype=torch.float16),
        scheduler=scheduler,
        conditional=object(),
    )
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("failure", ["cancel", "forward", "scheduler"])
def test_cancellation_and_failures_close_the_active_session(failure):
    active = []
    closed = []

    @contextmanager
    def session_factory(model, _stage):
        active.append(model)
        try:
            yield object()
        finally:
            active.clear()
            closed.append(model)

    def forward(_model, _session, latents, _timestep, _conditioning, _cache_identity):
        if failure == "forward":
            raise MemoryError("simulated OOM")
        return torch.ones_like(latents)

    class Scheduler(AdditiveScheduler):
        def step(self, *args, **kwargs):
            if failure == "scheduler":
                raise RuntimeError("scheduler failed")
            return super().step(*args, **kwargs)

    kwargs = {
        "latents": torch.zeros((1,)),
        "timesteps": (2, 1),
        "policy": StagePolicy("comfy_split"),
        "high_model": "high",
        "low_model": "low",
        "session_factory": session_factory,
        "forward": forward,
        "scheduler": Scheduler(),
        "conditional": object(),
        "cancelled": (lambda: True) if failure == "cancel" else None,
    }
    error = (
        asyncio.CancelledError
        if failure == "cancel"
        else (MemoryError if failure == "forward" else RuntimeError)
    )
    with pytest.raises(error):
        coordinate_denoise(**kwargs)
    assert not active
    assert closed == ["high"]


@pytest.mark.parametrize(
    "policy,timesteps",
    [
        (StagePolicy("unknown"), (2, 1)),
        (StagePolicy("diffusers_boundary", boundary_ratio=float("nan")), (2, 1)),
        (StagePolicy("comfy_split"), (1, 2)),
        (StagePolicy("comfy_split"), (torch.ones(2),)),
    ],
)
def test_stage_policy_rejects_malformed_inputs(policy, timesteps):
    with pytest.raises((TypeError, ValueError)):
        policy.assignments(timesteps)


def test_coordinator_rejects_missing_unconditional_and_shape_mismatch():
    @contextmanager
    def session_factory(_model, _stage):
        yield object()

    common = {
        "latents": torch.zeros((1,)),
        "timesteps": (2, 1),
        "policy": StagePolicy("comfy_split"),
        "high_model": "high",
        "low_model": "low",
        "session_factory": session_factory,
        "scheduler": AdditiveScheduler(),
        "conditional": object(),
    }
    with pytest.raises(ValueError, match="unconditional"):
        coordinate_denoise(
            **common,
            forward=lambda *_: torch.ones((1,)),
            high_guidance=3.5,
        )
    with pytest.raises(ValueError, match="prediction shape"):
        coordinate_denoise(
            **common,
            forward=lambda *_: torch.ones((2,)),
        )
