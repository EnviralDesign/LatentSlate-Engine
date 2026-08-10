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
        if self.name == "comfy_split":
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
                if scale > 1.0:
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
                result = _scheduler_step(scheduler, prediction, timestep, result)
                completed += 1
                if progress is not None:
                    progress(completed, len(timesteps), stage)
    return result


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
