"""Source-conformant sampling primitives for the canonical LTX 2.3 T2V fixture."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .transformer_context import Ltx23TransformerContext


CANONICAL_FIRST_PASS_VIDEO_SHAPE = (1, 128, 19, 8, 8)
CANONICAL_AUDIO_SHAPE = (1, 8, 126, 16)


def canonical_empty_latents(device: torch.device | str = "cuda") -> list[torch.Tensor]:
    """Create the fixture's 256px first-pass AV lattice."""
    return [
        torch.zeros(CANONICAL_FIRST_PASS_VIDEO_SHAPE, device=device, dtype=torch.float32),
        torch.zeros(CANONICAL_AUDIO_SHAPE, device=device, dtype=torch.float32),
    ]


def canonical_noise(seed: int, device: torch.device | str = "cuda") -> list[torch.Tensor]:
    """Match Comfy's sequential CPU noise draws for the nested AV latent."""
    return nested_noise(seed, canonical_empty_latents(device))


def nested_noise(seed: int, latents: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Match Comfy's sequential CPU noise draws for a packed AV latent."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randn(sample.shape, generator=generator, dtype=torch.float32).to(sample.device)
        for sample in latents
    ]


@torch.inference_mode()
def euler_sample(
    model: Ltx23TransformerContext,
    condition: torch.Tensor,
    latents: Sequence[torch.Tensor],
    noise: Sequence[torch.Tensor],
    sigmas: Sequence[float],
    *,
    frame_rate: int,
) -> list[torch.Tensor]:
    """Run the fixture's CFG=1 Euler flow loop without a graph runtime."""
    if len(latents) != 2 or len(noise) != 2:
        raise ValueError("LTX 2.3 AV sampling requires one video and one audio latent")
    if len(sigmas) < 2:
        raise ValueError("Euler sampling needs at least two sigmas")

    # LTXAV's Comfy model wrapper preprocesses the raw dual Gemma projection
    # through the model-owned video and audio embedding connectors before the
    # first denoising call.  It is conditioning work, not part of Euler.
    condition = model.model.preprocess_text_embeds(
        condition.to(dtype=torch.bfloat16), unprocessed=True
    )

    sigma0 = float(sigmas[0])
    x = [latent + sample * sigma0 for latent, sample in zip(latents, noise, strict=True)]
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:], strict=True):
        timestep = torch.full((x[0].shape[0],), float(sigma), device=x[0].device)
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in x],
            timestep,
            condition,
            frame_rate=frame_rate,
        )
        delta = float(sigma_next) - float(sigma)
        x = [stream + predicted.float() * delta for stream, predicted in zip(x, flow, strict=True)]
    return x
