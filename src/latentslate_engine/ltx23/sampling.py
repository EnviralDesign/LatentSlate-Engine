"""Source-conformant sampling primitives for the concrete LTX 2.3 operations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch

from .transformer_context import Ltx23TransformerContext

FRAME_RATE = 30
MIN_SIDE = 64
MAX_PIXELS = 942_080
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 10.0
MAX_SEED = 0xFFFF_FFFF_FFFF_FFFF


def validate_ltx_request(
    width: int,
    height: int,
    duration_seconds: float,
    seed: int,
    *,
    alignment: int,
) -> None:
    """Validate the recovered LTX product domain without changing inputs."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise TypeError("LTX width and height must be integers")
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ValueError(f"LTX width and height must each be at least {MIN_SIDE}")
    if width % alignment or height % alignment:
        raise ValueError(f"LTX width and height must each be divisible by {alignment}")
    if width * height > MAX_PIXELS:
        raise ValueError(f"LTX width * height must not exceed {MAX_PIXELS}")

    if isinstance(duration_seconds, bool):
        raise TypeError("LTX duration_seconds must be numeric")
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("LTX duration_seconds must be numeric") from error
    if not math.isfinite(duration):
        raise ValueError("LTX duration_seconds must be finite")
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"LTX duration_seconds must be between {MIN_DURATION_SECONDS} and "
            f"{MAX_DURATION_SECONDS}"
        )
    if not math.isclose(duration * 2.0, round(duration * 2.0), abs_tol=1e-9):
        raise ValueError("LTX duration_seconds must use 0.5-second increments")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("LTX seed must be an integer")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"LTX seed must be between 0 and {MAX_SEED}")


def ltx_temporal_shapes(duration_seconds: float) -> tuple[int, int, int, int]:
    """Return requested frames, video latents, decoded frames, and audio latents."""
    requested_frames = round(float(duration_seconds) * FRAME_RATE) + 1
    video_latent_frames = ((requested_frames - 1) // 8) + 1
    decoded_video_frames = video_latent_frames * 8 - 7
    audio_latent_frames = round((float(requested_frames) / FRAME_RATE) * 25.0)
    return (
        requested_frames,
        video_latent_frames,
        decoded_video_frames,
        audio_latent_frames,
    )


def empty_av_latents(
    width: int,
    height: int,
    duration_seconds: float,
    *,
    spatial_divisor: int,
    device: torch.device | str = "cuda",
) -> list[torch.Tensor]:
    """Create the pinned Comfy video/audio latent shapes for one LTX request."""
    if width % spatial_divisor or height % spatial_divisor:
        raise ValueError(
            f"LTX dimensions must be divisible by latent divisor {spatial_divisor}"
        )
    _, video_frames, _, audio_frames = ltx_temporal_shapes(duration_seconds)
    return [
        torch.zeros(
            (
                1,
                128,
                video_frames,
                height // spatial_divisor,
                width // spatial_divisor,
            ),
            device=device,
            dtype=torch.float32,
        ),
        torch.zeros((1, 8, audio_frames, 16), device=device, dtype=torch.float32),
    ]


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
    """Run the pinned CFG=1 Euler flow loop without a graph runtime."""
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
    """Run the pinned I2V Euler loop with Comfy's latent-mask semantics."""
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
