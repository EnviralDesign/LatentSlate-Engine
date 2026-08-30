"""Source-conformant sampling primitives for the canonical LTX 2.3 T2V fixture."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

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
        torch.randn(sample.shape, generator=generator, dtype=torch.float32).to(
            sample.device
        )
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

    sigma_values = torch.as_tensor(
        sigmas, device=latents[0].device, dtype=torch.float32
    )
    x = [
        sample * sigma_values[0] + latent * (1.0 - sigma_values[0])
        for latent, sample in zip(latents, noise, strict=True)
    ]
    for sigma, sigma_next in pairwise(sigma_values):
        timestep = sigma.expand(x[0].shape[0])
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in x],
            timestep,
            condition,
            frame_rate=frame_rate,
        )
        delta = sigma_next - sigma
        shapes = [stream.shape for stream in x]
        sizes = [stream[0].numel() for stream in x]
        packed_x = torch.cat(
            [stream.reshape(stream.shape[0], 1, -1) for stream in x], dim=-1
        )
        packed_flow = torch.cat(
            [
                predicted.float().reshape(predicted.shape[0], 1, -1)
                for predicted in flow
            ],
            dim=-1,
        )
        broadcast_sigma = sigma.reshape(1, 1, 1)
        denoised = packed_x - packed_flow * broadcast_sigma
        derivative = (packed_x - denoised) / broadcast_sigma
        packed_x = packed_x + derivative * delta
        x = [
            stream.view(shape)
            for stream, shape in zip(packed_x.split(sizes, dim=-1), shapes, strict=True)
        ]
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
    sigma_values = torch.as_tensor(
        sigmas, device=latents[0].device, dtype=torch.float32
    )
    x = [
        sample * sigma_values[0] + latent * (1.0 - sigma_values[0])
        for latent, sample in zip(latents, noise, strict=True)
    ]
    for sigma_value, sigma_next in pairwise(sigma_values):
        model_input = [
            stream * mask + latent * (1.0 - mask)
            for stream, mask, latent in zip(x, masks, latents, strict=True)
        ]
        video_timestep = model.model.patchifier.patchify(
            masks[0][:, :1].to(torch.bfloat16).float() * sigma_value
        )[0]
        audio_timestep = model.model.a_patchifier.patchify(
            masks[1][:, :1, :, :1].to(torch.bfloat16).float() * sigma_value
        )[0]
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in model_input],
            [video_timestep, audio_timestep],
            condition,
            frame_rate=frame_rate,
            denoise_mask=masks[0],
        )
        denoised = [
            (source - predicted.float() * sigma_value) * mask + latent * (1.0 - mask)
            for source, predicted, mask, latent in zip(
                model_input, flow, masks, latents, strict=True
            )
        ]
        delta = sigma_next - sigma_value
        shapes = [stream.shape for stream in x]
        sizes = [stream[0].numel() for stream in x]
        packed_x = torch.cat(
            [stream.reshape(stream.shape[0], 1, -1) for stream in x], dim=-1
        )
        packed_denoised = torch.cat(
            [stream.reshape(stream.shape[0], 1, -1) for stream in denoised], dim=-1
        )
        broadcast_sigma = sigma_value.reshape(1, 1, 1)
        derivative = (packed_x - packed_denoised) / broadcast_sigma
        packed_x = packed_x + derivative * delta
        x = [
            stream.view(shape)
            for stream, shape in zip(packed_x.split(sizes, dim=-1), shapes, strict=True)
        ]
    return x
