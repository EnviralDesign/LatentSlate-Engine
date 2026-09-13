"""Certified Krea Turbo: eight simple-schedule Euler steps, CFG one."""

import math

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
    """Select the same eight entries from the oracle's 10,000-step grid."""
    grid = torch.arange(1, 10001) / 10000
    grid = math.exp(1.15) / (math.exp(1.15) + (1 / grid - 1))
    return torch.tensor([float(grid[-(1 + i * 1250)]) for i in range(8)] + [0.0])


def noise(seed: int, width: int, height: int) -> torch.Tensor:
    """Generate the oracle's FP32 CPU noise without changing global RNG state."""
    return torch.randn(
        (1, 16, 1, height // 8, width // 8),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
    )


@torch.inference_mode()
def sample(model, conditioning, seed, width, height, device, progress=None):
    """Return the denormalized CPU latent consumed by the Qwen Image VAE."""
    x = noise(seed, width, height).to(device)
    schedule = sigmas().to(device)
    context = conditioning.to(device=device, dtype=torch.bfloat16)
    for index in range(8):
        sigma = schedule[index]
        velocity = model(x.to(torch.bfloat16), sigma.expand(1), context).float()
        denoised = x - velocity * sigma
        derivative = (x - denoised) / sigma
        x = x + derivative * (schedule[index + 1] - sigma)
        if progress is not None:
            progress(index + 1, 8)
    x = x.cpu()
    mean = torch.tensor(LATENT_MEAN).view(1, 16, 1, 1, 1)
    std = torch.tensor(LATENT_STD).view(1, 16, 1, 1, 1)
    return x * std + mean
