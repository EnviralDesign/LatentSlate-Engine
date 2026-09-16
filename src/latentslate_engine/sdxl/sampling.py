"""Ordinary SDXL epsilon sampling from ComfyUI 1a14b82e (GPL-3.0).

Sources: model_sampling.py, model_base.SDXL, samplers.py and
k_diffusion/sampling.py's Euler and DPM-Solver++(2M) paths.
"""

import torch

from .model import timestep_embedding


class DiscreteEPS(torch.nn.Module):
    """Reference schedule buffers follow the loaded model's CPU-to-device lifetime."""

    def __init__(self):
        super().__init__()
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
        alpha = torch.cumprod(1 - betas, dim=0)
        training = ((1 - alpha) / alpha) ** 0.5
        self.register_buffer("sigmas", training.float())
        self.register_buffer("log_sigmas", training.log().float())

    def schedule(self, steps, scheduler):
        if scheduler == "karras":
            # Comfy passes Python floats; tensor endpoints round differently.
            low = float(self.sigmas[0]) ** (1 / 7)
            high = float(self.sigmas[-1]) ** (1 / 7)
            sigmas = (high + torch.linspace(0, 1, steps) * (low - high)) ** 7
        elif scheduler == "normal":
            values = []
            for timestep in torch.linspace(999, 0, steps):
                t = timestep.to(self.log_sigmas.device)
                low, high, fraction = t.floor().long(), t.ceil().long(), t.frac()
                values.append(
                    float(
                        (
                            (1 - fraction) * self.log_sigmas[low]
                            + fraction * self.log_sigmas[high]
                        ).exp()
                    )
                )
            sigmas = torch.tensor(values)
        else:
            raise ValueError("Unsupported SDXL scheduler")
        return torch.cat((sigmas, torch.zeros(1)))


def schedule(steps, scheduler):
    """Build a cold CPU schedule for standalone callers."""
    sampling = DiscreteEPS()
    return sampling.schedule(steps, scheduler), sampling.log_sigmas


def conditioning(values, width, height, device):
    context, pooled = values
    dimensions = (height, width, 0, 0, height, width)
    sizes = (
        torch.cat([timestep_embedding(torch.tensor([x]), 256) for x in dimensions])
        .flatten()
        .unsqueeze(0)
    )
    adm = torch.cat((pooled, sizes), dim=1)
    return context.to(device=device, dtype=torch.float16), adm.to(
        device=device, dtype=torch.float16
    )


@torch.inference_mode()
def sample(
    model,
    positive,
    negative,
    seed,
    width,
    height,
    device,
    *,
    steps=25,
    cfg=7.0,
    sampler="dpmpp_2m",
    scheduler="karras",
    progress=None,
    model_sampling=None,
):
    model_sampling = model_sampling if model_sampling is not None else DiscreteEPS()
    sigmas = model_sampling.schedule(steps, scheduler).to(device)
    # Comfy moves model buffers after the first schedule has been constructed.
    model_sampling.to(device)
    logs = model_sampling.log_sigmas
    pos_context, pos_adm = conditioning(positive, width, height, device)
    neg_context, neg_adm = conditioning(negative, width, height, device)
    # Comfy repeats shorter CLIP chunks to a shared sequence length before CFG batching.
    import math

    length = math.lcm(pos_context.shape[1], neg_context.shape[1])
    contexts = torch.cat(
        (
            neg_context.repeat(1, length // neg_context.shape[1], 1),
            pos_context.repeat(1, length // pos_context.shape[1], 1),
        )
    )
    adms = torch.cat((neg_adm, pos_adm))
    batch_size = 2
    if cfg == 1.0:
        contexts, adms = pos_context, pos_adm
        batch_size = 1
    noise = torch.randn(
        (1, 4, height // 8, width // 8),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device)
    x = noise * torch.sqrt(1 + sigmas[0] ** 2)
    old_denoised = None
    ancestral_rng = torch.Generator(device=device).manual_seed(
        seed + (device.type == "cpu")
    )
    for index in range(steps):
        sigma, next_sigma = sigmas[index], sigmas[index + 1]
        timestep = (sigma.log() - logs).abs().argmin().float().repeat(batch_size)
        model_input = (
            (x / (sigma**2 + 1) ** 0.5).to(torch.float16).repeat(batch_size, 1, 1, 1)
        )
        output = model(model_input, timestep, contexts, adms).float()
        denoised = x - output * sigma
        if batch_size == 2:
            neg, pos = denoised.chunk(2)
            denoised = neg + (pos - neg) * cfg
        if sampler == "euler":
            x = x + ((x - denoised) / sigma) * (next_sigma - sigma)
        elif sampler == "euler_ancestral":
            up = min(
                next_sigma,
                (next_sigma**2 * (sigma**2 - next_sigma**2) / sigma**2) ** 0.5,
            )
            down = (next_sigma**2 - up**2) ** 0.5
            if down == 0:
                x = denoised
            else:
                noise = torch.randn(
                    x.shape, dtype=x.dtype, device=device, generator=ancestral_rng
                )
                x = x + ((x - denoised) / sigma) * (down - sigma) + noise * up
        elif sampler == "dpmpp_2m":
            t, next_t = -sigma.log(), -next_sigma.log()
            h = next_t - t
            if old_denoised is None or next_sigma == 0:
                x = (next_t.neg().exp() / t.neg().exp()) * x - (-h).expm1() * denoised
            else:
                previous_h = t - (-sigmas[index - 1].log())
                ratio = previous_h / h
                combined = (1 + 1 / (2 * ratio)) * denoised - (
                    1 / (2 * ratio)
                ) * old_denoised
                x = (next_t.neg().exp() / t.neg().exp()) * x - (-h).expm1() * combined
            old_denoised = denoised
        else:
            raise ValueError("Unsupported SDXL sampler")
        if progress:
            progress(index + 1, steps)
    return x
