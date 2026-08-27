"""Source-conformant sampling primitives for the canonical LTX 2.3 T2V fixture."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .transformer_context import Ltx23TransformerContext


CANONICAL_AUDIO_SHAPE = (1, 8, 126, 16)


def canonical_empty_latents(
    device: torch.device | str = "cuda", resolution: int = 512
) -> list[torch.Tensor]:
    """Create the 512px or 768px fixture's half-resolution first-pass AV lattice."""
    if resolution not in (512, 768):
        raise ValueError("canonical LTX 2.3 T2V resolution must be 512 or 768")
    latent_side = resolution // 64
    return [
        torch.zeros(
            (1, 128, 19, latent_side, latent_side), device=device, dtype=torch.float32
        ),
        torch.zeros(CANONICAL_AUDIO_SHAPE, device=device, dtype=torch.float32),
    ]


def canonical_noise(
    seed: int, device: torch.device | str = "cuda", resolution: int = 512
) -> list[torch.Tensor]:
    """Match Comfy's sequential CPU noise draws for the nested AV latent."""
    return nested_noise(seed, canonical_empty_latents(device, resolution))


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


@torch.inference_mode()
def euler_sample_masked(
    model: Ltx23TransformerContext,
    condition: torch.Tensor,
    latents: Sequence[torch.Tensor],
    noise: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    sigmas: Sequence[float],
    *,
    frame_rate: int,
) -> list[torch.Tensor]:
    """Run the canonical I2V Euler loop with Comfy's latent-mask semantics."""
    if len(latents) != 2 or len(noise) != 2 or len(masks) != 2:
        raise ValueError("LTX 2.3 masked AV sampling requires two streams")
    if len(sigmas) < 2:
        raise ValueError("Euler sampling needs at least two sigmas")
    for latent, sample, mask in zip(latents, noise, masks, strict=True):
        if latent.shape != sample.shape or mask.shape != latent.shape:
            raise ValueError("masked AV latent, noise, and mask shapes must match")

    condition = model.model.preprocess_text_embeds(
        condition.to(dtype=torch.bfloat16), unprocessed=True
    )
    sigma0 = float(sigmas[0])
    x = [
        sample * sigma0 + latent * (1.0 - sigma0)
        for latent, sample in zip(latents, noise, strict=True)
    ]
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:], strict=True):
        sigma_value = float(sigma)
        model_input = [
            stream * mask + latent * (1.0 - mask)
            for stream, mask, latent in zip(x, masks, latents, strict=True)
        ]
        video_timestep = model.model.patchifier.patchify(
            masks[0][:, :1] * sigma_value
        )[0]
        audio_timestep = model.model.a_patchifier.patchify(
            masks[1][:, :1, :, :1] * sigma_value
        )[0]
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in model_input],
            [video_timestep, audio_timestep],
            condition,
            frame_rate=frame_rate,
            denoise_mask=masks[0],
        )
        denoised = [
            (source - predicted.float() * sigma_value) * mask
            + latent * (1.0 - mask)
            for source, predicted, mask, latent in zip(
                model_input, flow, masks, latents, strict=True
            )
        ]
        delta = float(sigma_next) - sigma_value
        x = [
            stream + ((stream - clean) / sigma_value) * delta
            for stream, clean in zip(x, denoised, strict=True)
        ]
    return x
