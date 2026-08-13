"""Noise-only latent initialization for the native Wan 2.2 T2V operation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .wan22_i2v_conditioning import _validate_dimensions
from .wan22_stored_adapter import _canonicalize_residency_device


@dataclass(frozen=True, slots=True)
class WanT2VLatents:
    noise_latents: torch.Tensor
    num_frames: int
    height: int
    width: int
    seed: int


def prepare_wan_t2v_latents(
    *,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device | str,
) -> WanT2VLatents:
    """Create the exact 16-channel noise latent required by official T2V experts."""

    _validate_dimensions(height=height, width=width, num_frames=num_frames)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Wan T2V seed must be an integer in [0, 2^63)")
    target = _canonicalize_residency_device(torch.device(device))
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Wan T2V supports CPU or CUDA devices")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, 16, ((num_frames - 1) // 4) + 1, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(target)
    if not bool(torch.isfinite(latents).all()):
        raise RuntimeError("Wan T2V latent initialization produced non-finite values")
    return WanT2VLatents(latents, num_frames, height, width, seed)
