from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
import torch

from latentslate_engine.runtime.wan22_i2v_orchestration import (
    ComfyWanEulerScheduler,
    StagePolicy,
    coordinate_denoise,
)


class AdditiveScheduler:
    def step(self, prediction, _timestep, latents, *, return_dict):
        assert return_dict is False
        return (latents + prediction,)


def test_comfy_stage_policy_matches_the_pinned_two_sampler_graph():
    assert StagePolicy("comfy_split").assignments(tuple(range(20, 0, -1))).count("high") == 10
    assert StagePolicy("comfy_split").assignments((5, 4, 3, 2, 1)) == (
        "high",
        "high",
        "high",
        "low",
        "low",
    )

    scheduler = ComfyWanEulerScheduler()
    scheduler.set_timesteps(20, device=torch.device("cpu"))
    assert (
        StagePolicy("comfy_split").assignments(tuple(scheduler.timesteps))
        == ("high",) * 10 + ("low",) * 10
    )


def test_comfy_wan_euler_scheduler_matches_discreteflow_simple_source_math():
    scheduler = ComfyWanEulerScheduler()
    scheduler.set_timesteps(20, device=torch.device("cpu"))

    # Active I2V ModelSamplingDiscreteFlow.shift=5 applied to the 1..1000 grid, then
    # simple_scheduler chooses index 999, 949, ... and appends zero.
    assert scheduler.sigmas.shape == (21,)
    assert scheduler.sigmas[0].item() == pytest.approx(1.0)
    assert scheduler.sigmas[1].item() == pytest.approx(5 * 0.95 / (1 + 4 * 0.95))
    assert scheduler.sigmas[-2].item() == pytest.approx(5 * 0.05 / (1 + 4 * 0.05))
    assert scheduler.sigmas[-1].item() == 0.0
    assert scheduler.timesteps[0].item() == pytest.approx(1000.0)
    assert scheduler.timesteps[-1].item() == pytest.approx(scheduler.sigmas[-2].item() * 1000)


def test_comfy_wan_euler_scheduler_converts_flow_velocity_through_const_denoised_value():
    scheduler = ComfyWanEulerScheduler()
    scheduler.set_timesteps(2, device=torch.device("cpu"))
    sample = torch.full((1, 2), 2.0)
    velocity = torch.full_like(sample, 0.25)

    stepped = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]

    expected = sample + velocity * (scheduler.sigmas[1].float() - scheduler.sigmas[0].float())
    assert torch.allclose(stepped, expected)
    with pytest.raises(ValueError, match="timestep"):
        scheduler.step(velocity, 1.0, stepped, return_dict=False)


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


def test_pinned_comfy_euler_accepts_fp16_predictions_with_fp32_latents():
    scheduler = ComfyWanEulerScheduler()
    scheduler.set_timesteps(2, device=torch.device("cpu"))

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


def test_comfy_cfg_combines_per_branch_fp32_denoised_values_not_raw_fp16_velocity():
    scheduler = ComfyWanEulerScheduler()
    scheduler.set_timesteps(2, device=torch.device("cpu"))

    @contextmanager
    def session_factory(_model, _stage):
        yield object()

    calls = iter(
        (
            torch.tensor([0.3333], dtype=torch.float16),
            torch.tensor([0.1111], dtype=torch.float16),
            torch.tensor([0.0], dtype=torch.float16),
        )
    )
    sample = torch.tensor([0.12345679], dtype=torch.float32)
    result = coordinate_denoise(
        latents=sample,
        timesteps=tuple(scheduler.timesteps),
        policy=StagePolicy("comfy_split"),
        high_model="high",
        low_model="low",
        session_factory=session_factory,
        forward=lambda *_: next(calls),
        scheduler=scheduler,
        conditional=object(),
        unconditional=object(),
        high_guidance=3.5,
        low_guidance=1.0,
    )

    # The first step takes `velocity.float()` into CONST independently for both
    # branches, then CFGs the FP32 denoised values. Combining the original F16
    # velocities first would lose a different rounding path.
    sigma = 1.0
    cond = sample - sigma * torch.tensor([0.3333], dtype=torch.float16).float()
    uncond = sample - sigma * torch.tensor([0.1111], dtype=torch.float16).float()
    combined = uncond + 3.5 * (cond - uncond)
    expected_first = sample + (sample - combined) * (scheduler.sigmas[1] - 1.0)
    # The second low step has CFG=1 and a zero velocity; it therefore only takes
    # its final Euler transition from the first result.
    assert torch.allclose(result, expected_first)


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
