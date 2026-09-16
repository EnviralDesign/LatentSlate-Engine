"""H3 joint AV noise and the curated simple/res_multistep solver.

Adapted from ComfyUI 1a14b82e, sample.py, model_sampling.py, samplers.py
and k_diffusion/sampling.py (GPL-3.0). The model callable returns denoised
packed latents; H3's audio carry conversion belongs at that model boundary.
"""

import math

import torch


def temporal_shape(length: int) -> tuple[int, int, int]:
    """Return actual frames, video latent time and audio latent time at 24 fps."""
    frames = max(5, length)
    frames += (5 - frames % 17) % 17
    video_time = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return frames, video_time, round(frames / 24 * 40)


def pack(streams):
    """Pack video followed by audio without changing element order."""
    return torch.cat([x.reshape(x.shape[0], 1, -1) for x in streams], dim=-1)


def unpack(latent, shapes):
    """Recover views of the request's video and stereo audio tensors."""
    streams = []
    for shape in shapes:
        size = math.prod(shape[1:])
        streams.append(latent[:, :, :size].reshape(latent.shape[0], *shape[1:]))
        latent = latent[:, :, size:]
    return streams


def noise(shapes, seed: int):
    """Consume one CPU RNG stream, video first and audio second."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return pack(
        [
            torch.randn(shape, generator=generator, dtype=torch.float32)
            for shape in shapes
        ]
    )


def simple_schedule(steps: int, shift: float = 12.0):
    """Sample Comfy's discrete thousand-point flow schedule, including zero."""
    timesteps = (torch.arange(1, 1001) / 1000) * 1000
    base = timesteps / 1000
    sigmas = shift * base / (1 + (shift - 1) * base)
    stride = len(sigmas) / steps
    return torch.tensor(
        [float(sigmas[-(1 + int(i * stride))]) for i in range(steps)] + [0.0],
        dtype=torch.float32,
    )


@torch.inference_mode()
def res_multistep(model, latent, sigmas, progress=None):
    """Run the reference's deterministic eta=0, non-CFG++ solver."""
    previous_sigma = previous_denoised = None
    batch = latent.new_ones([latent.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        denoised = model(latent, sigma * batch)
        if next_sigma == 0 or previous_denoised is None:
            derivative = (latent - denoised) / sigma
            latent = latent + derivative * (next_sigma - sigma)
        else:
            t = -sigma.log()
            old_t = -previous_sigma.log()
            next_t = -next_sigma.log()
            prev_t = -sigmas[index - 1].log()
            h = next_t - t
            c2 = (prev_t - old_t) / h
            phi1 = torch.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            latent = (-h).exp() * latent + h * (b1 * denoised + b2 * previous_denoised)
        previous_denoised = denoised
        previous_sigma = next_sigma
        if progress is not None:
            progress(index + 1, len(sigmas) - 1)
    return latent
