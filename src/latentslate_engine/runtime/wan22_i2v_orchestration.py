"""Engine-owned, dependency-injected Wan 2.2 high/low denoise orchestration."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import torch

# The pinned active Wan 2.2 I2V template uses ModelSamplingSD3 shift 5.
# (The older Comfy example used shift 8; FLF is a distinct operation.)
WAN_FLOW_SHIFT = 5.0
WAN_FLOW_MULTIPLIER = 1000


class WanEulerScheduler:
    """Exact minimal scheduler contract from the pinned Comfy Wan 14B I2V graph.

    The official graph uses ``ModelSamplingSD3`` with shift 5 / multiplier
    1000, ``simple_scheduler``, and non-ancestral Euler. ``ModelSamplingSD3``
    combines DiscreteFlow with CONST; the stored Diffusers transformer reports
    flow velocity, which the graph converts to FP32 before its Comfy CONST
    denoised value is formed. CFG combines those *denoised* branch values before
    k-diffusion Euler computes the derivative. This small adapter states that
    source-derived behavior without importing Comfy implementation code.
    """

    def __init__(
        self,
        *,
        shift: float = WAN_FLOW_SHIFT,
        multiplier: int = WAN_FLOW_MULTIPLIER,
    ) -> None:
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Wan flow shift must be finite and positive")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
            raise ValueError("Wan timestep multiplier must be a positive integer")
        self.shift = float(shift)
        self.multiplier = multiplier
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self._step_index = 0

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("Wan Euler steps must be an integer of at least two")
        # Comfy `ModelSamplingDiscreteFlow.set_parameters` followed by
        # `simple_scheduler`: shift discrete 1..1000 time, select evenly from
        # high to low, then append zero for the final Euler transition.
        grid = torch.arange(1, self.multiplier + 1, dtype=torch.float32, device=device)
        grid = grid / self.multiplier
        flow_sigmas = self.shift * grid / (1 + (self.shift - 1) * grid)
        simple = torch.stack(
            [flow_sigmas[-(1 + int(index * len(flow_sigmas) / steps))] for index in range(steps)]
        )
        self.sigmas = torch.cat((simple, torch.zeros(1, dtype=simple.dtype, device=device)))
        self.timesteps = self.sigmas[:-1] * self.multiplier
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: object,
        sample: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        if return_dict is not False:
            raise TypeError("Wan Euler scheduler supports return_dict=False only")
        if self._step_index >= len(self.timesteps):
            raise RuntimeError("Wan Euler scheduler received too many steps")
        denoised = self.denoised(model_output, timestep, sample)
        return self.step_denoised(denoised, timestep, sample, return_dict=return_dict)

    def denoised(
        self,
        model_output: torch.Tensor,
        timestep: object,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Comfy's per-branch ``velocity.float()`` CONST conversion."""

        sigma = self._validate_step_timestep(timestep, sample)
        if not isinstance(model_output, torch.Tensor) or model_output.shape != sample.shape:
            raise ValueError("Wan Euler prediction shape does not match its sample")
        return sample - sigma * model_output.float()

    def step_denoised(
        self,
        denoised: torch.Tensor,
        timestep: object,
        sample: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        """Take the Euler transition after CFG has combined denoised branches."""

        if return_dict is not False:
            raise TypeError("Wan Euler scheduler supports return_dict=False only")
        sigma = self._validate_step_timestep(timestep, sample)
        next_sigma = self.sigmas[self._step_index + 1].to(
            device=sample.device,
            dtype=sample.dtype,
        )
        if not isinstance(denoised, torch.Tensor) or denoised.shape != sample.shape:
            raise ValueError("Wan Euler denoised value shape does not match its sample")
        derivative = (sample - denoised) / sigma
        result = sample + derivative * (next_sigma - sigma)
        self._step_index += 1
        return (result,)

    def _validate_step_timestep(self, timestep: object, sample: torch.Tensor) -> torch.Tensor:
        if self._step_index >= len(self.timesteps):
            raise RuntimeError("Wan Euler scheduler received too many steps")
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=sample.dtype)
        expected_timestep = self.timesteps[self._step_index]
        actual_timestep = _scalar_timestep(timestep, device=sample.device)
        if not torch.isclose(
            actual_timestep,
            expected_timestep.to(device=sample.device, dtype=actual_timestep.dtype),
            rtol=0,
            atol=1e-4,
        ):
            raise ValueError("Wan Euler timestep does not match the simple schedule")
        return sigma


def _scalar_timestep(value: object, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError("Wan Euler timestep must be a finite scalar")
        return value.to(device=device, dtype=torch.float64)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError("Wan Euler timestep must be a finite numeric scalar")
    return torch.tensor(float(value), device=device, dtype=torch.float64)


class DenoiseSession(Protocol):
    """Marker protocol for a residency session yielded by the session factory."""


SessionFactory = Callable[[Any, str], AbstractContextManager[DenoiseSession]]
ForwardCallback = Callable[[Any, DenoiseSession, torch.Tensor, object, Any, str], torch.Tensor]
ProgressCallback = Callable[[int, int, str], None]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """Deterministically divide descending scheduler timesteps into two stages."""

    name: str
    boundary_ratio: float = 0.9
    num_train_timesteps: int = 1000

    def assignments(self, timesteps: Sequence[object]) -> tuple[str, ...]:
        values = _validated_timestep_values(timesteps)
        if self.name == "expert_split":
            high_count = (len(values) + 1) // 2
        elif self.name == "diffusers_boundary":
            if (
                not math.isfinite(self.boundary_ratio)
                or not 0.0 < self.boundary_ratio < 1.0
                or isinstance(self.num_train_timesteps, bool)
                or self.num_train_timesteps <= 0
            ):
                raise ValueError("Wan Diffusers boundary settings are invalid")
            boundary = self.boundary_ratio * self.num_train_timesteps
            high_count = sum(value >= boundary for value in values)
            if high_count == 0 or high_count == len(values):
                raise ValueError("Wan Diffusers boundary must select both stages")
        else:
            raise ValueError(f"unknown Wan stage policy: {self.name!r}")
        return ("high",) * high_count + ("low",) * (len(values) - high_count)

    def split(self, timesteps: Sequence[object]) -> tuple[tuple[object, ...], tuple[object, ...]]:
        assignments = self.assignments(timesteps)
        high_count = assignments.count("high")
        return tuple(timesteps[:high_count]), tuple(timesteps[high_count:])


def coordinate_denoise(
    *,
    latents: torch.Tensor,
    timesteps: Sequence[object],
    policy: StagePolicy,
    high_model: Any,
    low_model: Any,
    session_factory: SessionFactory,
    forward: ForwardCallback,
    scheduler: Any,
    conditional: Any,
    unconditional: Any | None = None,
    high_guidance: float = 1.0,
    low_guidance: float = 1.0,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCheck | None = None,
) -> torch.Tensor:
    """Run contiguous high/low stages while keeping only one model resident.

    The session factory must return the Engine-owned transformer residency
    context. A session remains open across every timestep in its contiguous
    stage and is closed before the other model is entered. Context-manager
    cleanup therefore owns cancellation, OOM, and arbitrary ``BaseException``
    teardown without involving Diffusers or Accelerate offload hooks.
    """

    if not isinstance(latents, torch.Tensor) or latents.ndim < 1:
        raise TypeError("Wan denoise latents must be a tensor with at least one dimension")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Wan denoise latents must be finite")
    guidance = {
        "high": _validated_guidance(high_guidance, "high"),
        "low": _validated_guidance(low_guidance, "low"),
    }
    assignments = policy.assignments(timesteps)
    if any(scale > 1.0 for scale in guidance.values()) and unconditional is None:
        raise ValueError("Wan classifier-free guidance requires unconditional conditioning")

    result = latents
    completed = 0
    stages = (
        (
            "high",
            high_model,
            tuple(t for t, stage in zip(timesteps, assignments) if stage == "high"),
        ),
        ("low", low_model, tuple(t for t, stage in zip(timesteps, assignments) if stage == "low")),
    )
    for stage, model, stage_timesteps in stages:
        if not stage_timesteps:
            continue
        with session_factory(model, stage) as session:
            for timestep in stage_timesteps:
                if cancelled is not None and cancelled():
                    raise asyncio.CancelledError
                conditional_prediction = _validated_prediction(
                    forward(model, session, result, timestep, conditional, "cond"),
                    result,
                )
                scale = guidance[stage]
                if _supports_denoised_cfg(scheduler):
                    # The active Comfy graph casts every velocity to FP32 and
                    # constructs CONST denoised samples before CFG. Combining
                    # raw FP16 velocities first is observably different.
                    conditional_denoised = scheduler.denoised(
                        conditional_prediction,
                        timestep,
                        result,
                    )
                    if scale > 1.0:
                        unconditional_prediction = _validated_prediction(
                            forward(model, session, result, timestep, unconditional, "uncond"),
                            result,
                        )
                        unconditional_denoised = scheduler.denoised(
                            unconditional_prediction,
                            timestep,
                            result,
                        )
                        denoised = unconditional_denoised + scale * (
                            conditional_denoised - unconditional_denoised
                        )
                    else:
                        denoised = conditional_denoised
                    result = scheduler.step_denoised(
                        denoised,
                        timestep,
                        result,
                        return_dict=False,
                    )[0]
                elif scale > 1.0:
                    unconditional_prediction = _validated_prediction(
                        forward(model, session, result, timestep, unconditional, "uncond"),
                        result,
                    )
                    if unconditional_prediction.dtype != conditional_prediction.dtype:
                        raise ValueError(
                            "Wan conditional and unconditional prediction dtypes differ"
                        )
                    prediction = unconditional_prediction + scale * (
                        conditional_prediction - unconditional_prediction
                    )
                else:
                    prediction = conditional_prediction
                if not _supports_denoised_cfg(scheduler):
                    result = _scheduler_step(scheduler, prediction, timestep, result)
                completed += 1
                if progress is not None:
                    progress(completed, len(timesteps), stage)
    return result


def _supports_denoised_cfg(scheduler: Any) -> bool:
    return callable(getattr(scheduler, "denoised", None)) and callable(
        getattr(scheduler, "step_denoised", None)
    )


def _validated_timestep_values(timesteps: Sequence[object]) -> tuple[float, ...]:
    if not isinstance(timesteps, Sequence) or isinstance(timesteps, (str, bytes)) or not timesteps:
        raise ValueError("Wan stage policy requires nonempty scheduler timesteps")
    values: list[float] = []
    for timestep in timesteps:
        if isinstance(timestep, torch.Tensor):
            if timestep.ndim != 0:
                raise ValueError("Wan scheduler timesteps must be scalars")
            value = float(timestep.item())
        elif isinstance(timestep, bool) or not isinstance(timestep, (int, float)):
            raise TypeError("Wan scheduler timesteps must be numeric scalars")
        else:
            value = float(timestep)
        if not math.isfinite(value):
            raise ValueError("Wan scheduler timesteps must be finite")
        values.append(value)
    if any(first <= second for first, second in pairwise(values)):
        raise ValueError("Wan scheduler timesteps must be strictly descending")
    return tuple(values)


def _validated_guidance(value: float, stage: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Wan {stage} guidance must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Wan {stage} guidance must be finite and nonnegative")
    return result


def _validated_prediction(prediction: object, latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor) or prediction.shape != latents.shape:
        raise ValueError("Wan transformer prediction shape does not match latents")
    if prediction.device != latents.device or not prediction.is_floating_point():
        raise ValueError("Wan transformer prediction must be floating point on the latent device")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("Wan transformer prediction must be finite")
    return prediction


def _scheduler_step(
    scheduler: Any,
    prediction: torch.Tensor,
    timestep: object,
    latents: torch.Tensor,
) -> torch.Tensor:
    try:
        stepped = scheduler.step(prediction, timestep, latents, return_dict=False)
    except TypeError as exc:
        raise TypeError("Wan scheduler.step must support return_dict=False") from exc
    if isinstance(stepped, torch.Tensor):
        result = stepped
    elif isinstance(stepped, (tuple, list)) and stepped:
        result = stepped[0]
    else:
        raise TypeError("Wan scheduler.step must return a tensor or nonempty tuple")
    if not isinstance(result, torch.Tensor) or result.shape != latents.shape:
        raise ValueError("Wan scheduler output shape does not match latents")
    if result.device != latents.device or result.dtype != latents.dtype:
        raise ValueError("Wan scheduler output device/dtype does not match latents")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Wan scheduler output must be finite")
    return result
