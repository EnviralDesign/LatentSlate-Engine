"""Saved Z-Image Turbo simple/AuraFlow schedule and RES multistep solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Z_IMAGE_AURAFLOW_SHIFT = 3.0
Z_IMAGE_STEPS = 8
Z_IMAGE_SCHEDULE_TIMESTEPS = 1_000


def _indexed_aura_flow_shift_3_sigmas() -> torch.Tensor:
    """Reproduce the saved simple-scheduler table lookup in F32."""

    base = torch.arange(
        1, Z_IMAGE_SCHEDULE_TIMESTEPS + 1, dtype=torch.float32
    ) / Z_IMAGE_SCHEDULE_TIMESTEPS
    shifted = (Z_IMAGE_AURAFLOW_SHIFT * base) / (
        1.0 + (Z_IMAGE_AURAFLOW_SHIFT - 1.0) * base
    )
    stride = len(shifted) / Z_IMAGE_STEPS
    indices = torch.tensor(
        [len(shifted) - (1 + int(index * stride)) for index in range(Z_IMAGE_STEPS)],
        dtype=torch.int64,
    )
    return torch.cat((shifted[indices], torch.zeros(1, dtype=torch.float32)))


Z_IMAGE_AURAFLOW_SHIFT_3_SIGMAS = tuple(
    float(value) for value in _indexed_aura_flow_shift_3_sigmas()
)


@dataclass(frozen=True, slots=True)
class ZImageSamplerStep:
    index: int
    sigma: float
    sigma_next: float
    method: str


class ZImageSamplingCancelled(RuntimeError):
    pass


def z_image_auraflow_shift_3_sigmas(*, steps: int = Z_IMAGE_STEPS) -> tuple[float, ...]:
    if steps != Z_IMAGE_STEPS:
        raise ValueError("Z-Image Turbo requires exactly 8 AuraFlow simple steps")
    return Z_IMAGE_AURAFLOW_SHIFT_3_SIGMAS


def z_image_auraflow_shift_3_sigma_tensor(
    *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return the exact saved F32 scheduler values on the requested device."""

    return _indexed_aura_flow_shift_3_sigmas().to(device=device)


@torch.no_grad()
def z_image_res_multistep(
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    *,
    sigmas: tuple[float, ...] = Z_IMAGE_AURAFLOW_SHIFT_3_SIGMAS,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[ZImageSamplerStep], None] | None = None,
) -> torch.Tensor:
    """Run the deterministic, non-ancestral two-step RES update.

    The update is independently expressed from the second-order exponential
    multistep formula.  It performs eight denoiser evaluations, introduces no
    ancestral noise, and has no classifier-free or negative-conditioning branch.
    """

    supplied_sigmas = torch.tensor(sigmas, dtype=torch.float32)
    expected_sigmas = _indexed_aura_flow_shift_3_sigmas()
    if not torch.equal(supplied_sigmas.view(torch.int32), expected_sigmas.view(torch.int32)):
        raise ValueError("Z-Image Turbo sigma schedule differs from the saved path")
    if not latents.is_floating_point() or not torch.isfinite(latents).all():
        raise ValueError("Z-Image initial latents must be finite floating point values")

    current = latents
    previous_denoised: torch.Tensor | None = None
    previous_sigma: torch.Tensor | None = None
    sigma_values = supplied_sigmas.to(device=latents.device)
    batch_scale = torch.ones((latents.shape[0],), dtype=torch.float32, device=latents.device)

    for index in range(len(sigma_values) - 1):
        if cancelled():
            raise ZImageSamplingCancelled(f"Z-Image sampling canceled before step {index + 1}")
        sigma = sigma_values[index]
        sigma_next = sigma_values[index + 1]
        denoised = denoiser(current, sigma * batch_scale)
        if denoised.shape != current.shape or not torch.isfinite(denoised).all():
            raise ValueError(f"Z-Image denoiser returned invalid values at step {index + 1}")

        if previous_denoised is None or float(sigma_next) == 0.0:
            derivative = (current - denoised) / sigma
            current = current + derivative * (sigma_next - sigma)
            method = "euler"
        else:
            assert previous_sigma is not None
            time = -torch.log(sigma)
            previous_time = -torch.log(previous_sigma)
            next_time = -torch.log(sigma_next)
            schedule_previous_time = -torch.log(sigma_values[index - 1])
            step = next_time - time
            ratio = (schedule_previous_time - previous_time) / step
            phi1 = torch.expm1(-step) / -step
            phi2 = (phi1 - 1.0) / -step
            coefficient_current = torch.nan_to_num(phi1 - phi2 / ratio, nan=0.0)
            coefficient_previous = torch.nan_to_num(phi2 / ratio, nan=0.0)
            current = torch.exp(-step) * current + step * (
                coefficient_current * denoised + coefficient_previous * previous_denoised
            )
            method = "res_second_order"
        if not torch.isfinite(current).all():
            raise ValueError(f"Z-Image sampler produced non-finite latents at step {index + 1}")
        if progress is not None:
            progress(ZImageSamplerStep(index, float(sigma), float(sigma_next), method))
        previous_denoised = denoised
        previous_sigma = sigma_next
    return current


def z_image_sampler_contract() -> dict[str, object]:
    return {
        "executable": True,
        "kind": "deterministic_res_multistep",
        "sampling": "auraflow_shift_3",
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": Z_IMAGE_STEPS,
        "guider": "basic",
        "ancestral_eta": 0.0,
        "denoiser_evaluations": Z_IMAGE_STEPS,
        "sigmas": z_image_auraflow_shift_3_sigmas(),
    }
