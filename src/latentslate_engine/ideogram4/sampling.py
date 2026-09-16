"""Ideogram v4's official Comfy template schedule and dual-model Euler path.

Adapted from ComfyUI 1a14b82e nodes_ideogram4.py, nodes_custom_sampler.py,
model_sampling.py and k_diffusion/sampling.py (GPL-3.0).
"""

import math

import torch


def sigmas(width: int, height: int, steps: int = 20, mu=0.0, std=1.75):
    mean = mu + 0.5 * math.log(width * height / (512 * 512))
    u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u))
    t = t.clamp(1.0 / (1.0 + math.exp(9.0)), 1.0 / (1.0 + math.exp(-7.5)))
    schedule = (1.0 - t).flip(0)
    schedule[-1] = 0.0
    return schedule.float()


@torch.inference_mode()
def sample(
    model, negative_model, conditioning, seed, width, height, device, progress=None
):
    context = conditioning.to(device=device, dtype=torch.bfloat16)
    negative = None if negative_model is None else torch.zeros_like(context)
    schedule = sigmas(width, height).to(device)
    noise = torch.randn(
        (1, 128, height // 16, width // 16),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device)
    x = schedule[0] * noise
    del noise
    batch = x.new_ones([x.shape[0]])
    for index in range(len(schedule) - 1):
        sigma = schedule[index]
        timestep = sigma * batch
        positive_output = model(x.to(torch.bfloat16), timestep, context).float()
        cond = x - positive_output * sigma
        del positive_output
        if negative_model is None:
            denoised = cond
        else:
            negative_output = negative_model(
                x.to(torch.bfloat16), timestep, negative
            ).float()
            uncond = x - negative_output * sigma
            del negative_output
            cfg = 3.0 if float(sigma) <= 1.0 - 0.7 else 7.0
            denoised = uncond + (cond - uncond) * cfg
        derivative = (x - denoised) / sigma
        x = x + derivative * (schedule[index + 1] - sigma)
        if progress is not None:
            progress(index + 1, len(schedule) - 1)
    return x.cpu()
