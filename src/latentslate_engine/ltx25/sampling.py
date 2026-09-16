"""LTX 2.5's curated rectified-flow Euler ancestral sampling.

Adapted from ComfyUI 1a14b82e ``sample_euler_ancestral_RF`` and CONST
noise scaling. AV streams are packed before arithmetic and noise draws.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from latentslate_engine.ltx23.sampling import nested_noise
from latentslate_engine.ltx23.transformer_context import Ltx23TransformerContext

FIRST_PASS_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
SECOND_PASS_SIGMAS = (0.85, 0.725, 0.4219, 0.0)


def _pack(streams: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([s.reshape(s.shape[0], 1, -1) for s in streams], dim=-1)


@torch.inference_mode()
def ancestral_sample(
    model: Ltx23TransformerContext,
    condition: torch.Tensor,
    latents: Sequence[torch.Tensor],
    sigmas: Sequence[float],
    *,
    seed: int,
    frame_rate: int,
    eta: float = 1.0,
    keyframe_idxs: torch.Tensor | None = None,
    guide_attention_entries: list[dict[str, object]] | None = None,
    masks: Sequence[torch.Tensor] | None = None,
    step_callback: Callable[[int, int], None] | None = None,
) -> list[torch.Tensor]:
    """Sample one AV stage with the reference CPU initial/GPU ancestral RNGs."""
    shapes = [s.shape for s in latents]
    sizes = [s[0].numel() for s in latents]

    def unpack(value):
        return [
            v.reshape(shape)
            for v, shape in zip(value.split(sizes, dim=-1), shapes, strict=True)
        ]

    device = latents[0].device
    schedule = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
    noise = _pack(nested_noise(seed, latents))
    x = schedule[0] * noise + (1.0 - schedule[0]) * _pack(latents)
    condition = model.model.preprocess_text_embeds(
        condition.to(device=device, dtype=torch.bfloat16), unprocessed=True
    )
    generator = torch.Generator(device=device).manual_seed(
        seed + (device.type == "cpu")
    )
    for index in range(len(schedule) - 1):
        sigma, next_sigma = schedule[index], schedule[index + 1]
        if masks is None:
            flow = model.model(
                [s.to(torch.bfloat16) for s in unpack(x)],
                sigma.expand(x.shape[0]),
                condition,
                frame_rate=frame_rate,
            )
            denoised = x - _pack([s.float() for s in flow]) * sigma
        else:
            model_input = [
                stream * mask + latent * (1.0 - mask)
                for stream, mask, latent in zip(unpack(x), masks, latents, strict=True)
            ]
            video_timestep = model.model.patchifier.patchify(
                masks[0][:, :1].to(torch.bfloat16).float() * sigma
            )[0]
            audio_timestep = model.model.a_patchifier.patchify(
                masks[1][:, :1, :, :1].to(torch.bfloat16).float() * sigma
            )[0]
            flow = model.model(
                [s.to(torch.bfloat16) for s in model_input],
                [video_timestep, audio_timestep],
                condition,
                frame_rate=frame_rate,
                denoise_mask=masks[0],
                keyframe_idxs=keyframe_idxs,
                guide_attention_entries=guide_attention_entries,
            )
            denoised = _pack(
                [
                    (source - predicted.float() * sigma) * mask + latent * (1.0 - mask)
                    for source, predicted, mask, latent in zip(
                        model_input, flow, masks, latents, strict=True
                    )
                ]
            )
        if next_sigma == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (next_sigma / sigma - 1) * eta
            sigma_down = next_sigma * downstep_ratio
            alpha_next, alpha_down = 1 - next_sigma, 1 - sigma_down
            renoise = (
                next_sigma**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
            ) ** 0.5
            ratio = sigma_down / sigma
            x = ratio * x + (1 - ratio) * denoised
            if eta > 0:
                noise = torch.randn(
                    x.shape, dtype=x.dtype, device=device, generator=generator
                )
                x = (alpha_next / alpha_down) * x + noise * renoise
        if step_callback is not None:
            step_callback(index + 1, len(schedule) - 1)
    return unpack(x)
