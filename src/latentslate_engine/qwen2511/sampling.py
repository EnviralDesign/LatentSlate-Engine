"""Curated 40-step Euler sampling, CFG four and post-CFG norm attenuation.

Follows pinned ComfyUI model_sampling.py, samplers.py, nodes_cfg.py and
Wan21 latent normalization (GPL-3.0).
"""

import torch

LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def sigmas() -> torch.Tensor:
    """Select forty entries from the shifted 1000-point reference schedule."""
    grid = torch.arange(1, 1001) / 1000
    grid = 3.1 * grid / (1 + (3.1 - 1) * grid)
    return torch.tensor([float(grid[-(1 + index * 25)]) for index in range(40)] + [0.0])


def noise(seed: int, width: int, height: int) -> torch.Tensor:
    """Generate the oracle's FP32 CPU noise without changing global RNG state."""
    return torch.randn(
        (1, 16, 1, height // 8, width // 8),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
    )


def normalize(latent):
    """Convert the raw VAE latent to the transformer's reference domain."""
    mean = torch.tensor(LATENT_MEAN, device=latent.device, dtype=latent.dtype).view(1, 16, 1, 1, 1)
    std = torch.tensor(LATENT_STD, device=latent.device, dtype=latent.dtype).view(1, 16, 1, 1, 1)
    return (latent - mean) / std


def denormalize(latent):
    """Convert the final sampler state back to the VAE domain."""
    mean = torch.tensor(LATENT_MEAN, device=latent.device, dtype=latent.dtype).view(1, 16, 1, 1, 1)
    std = torch.tensor(LATENT_STD, device=latent.device, dtype=latent.dtype).view(1, 16, 1, 1, 1)
    return latent * std + mean


@torch.inference_mode()
def sample(model, positive, negative, references, seed, width, height, device, progress=None):
    """Return the denormalized CPU latent for the Qwen Image VAE."""
    x = noise(seed, width, height).to(device)
    schedule = sigmas().to(device)
    positive = positive.to(device=device, dtype=torch.bfloat16)
    negative = negative.to(device=device, dtype=torch.bfloat16)
    references = [normalize(latent).to(device=device, dtype=torch.bfloat16) for latent in references]
    for index in range(40):
        sigma = schedule[index]
        inputs = x.to(torch.bfloat16)
        cond = x - model(inputs, sigma.expand(1), positive, references).float() * sigma
        uncond = x - model(inputs, sigma.expand(1), negative, references).float() * sigma
        denoised = uncond + (cond - uncond) * 4.0
        scale = (torch.norm(cond, dim=1, keepdim=True) / (torch.norm(denoised, dim=1, keepdim=True) + 1e-8)).clamp(0.0, 1.0)
        denoised = denoised * scale
        derivative = (x - denoised) / sigma
        x = x + derivative * (schedule[index + 1] - sigma)
        if progress is not None:
            progress(index + 1, 40)
    return denormalize(x.cpu())
