"""Z-Image's simple flow schedule and deterministic RES multistep solver.

Narrowly adapted from Comfy-Org/ComfyUI (GPL-3.0), commit
1a14b82e7339176357d627c41a262543a6a1356b: model_sampling.py,
samplers.py and k_diffusion/sampling.py. The official workflow uses
sample_res_multistep with eta=0 and CFG=1.
"""

import torch


def sigmas(steps: int = 8, shift: float = 3.0) -> torch.Tensor:
    """Select the reference simple schedule from its 1,000-entry flow grid."""
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if shift <= 0:
        raise ValueError("shift must be positive")
    grid = torch.arange(1, 1001) / 1000
    grid = shift * grid / (1 + (shift - 1) * grid)
    return torch.tensor(
        [float(grid[-(1 + int(i * (1000 / steps)))]) for i in range(steps)] + [0.0],
        dtype=torch.float32,
    )


def noise(seed: int, width: int, height: int) -> torch.Tensor:
    """Generate the reference CPU noise without changing global RNG state."""
    return torch.randn(
        (1, 16, height // 8, width // 8),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
    )


@torch.inference_mode()
def res_multistep(denoise, x, schedule, progress=None):
    """Integrate a denoised-prediction callable using the reference eta=0 path."""
    old_sigma_down = None
    old_denoised = None
    batch = x.new_ones([x.shape[0]])
    for index in range(len(schedule) - 1):
        sigma = schedule[index]
        sigma_down = schedule[index + 1]
        denoised = denoise(x, sigma * batch)
        if sigma_down == 0 or old_denoised is None:
            derivative = (x - denoised) / sigma
            x = x + derivative * (sigma_down - sigma)
        else:
            t = -sigma.log()
            t_old = -old_sigma_down.log()
            t_next = -sigma_down.log()
            t_prev = -schedule[index - 1].log()
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1 = torch.expm1(-h) / (-h)
            phi2 = (phi1 - 1.0) / (-h)
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            x = (-h).exp() * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
        old_sigma_down = sigma_down
        if progress is not None:
            progress(index + 1, len(schedule) - 1)
    return x


@torch.inference_mode()
def sample(model, conditioning, seed, width, height, device, progress=None):
    """Finish the sampling phase before releasing its temporary device tensors."""
    context = conditioning.to(device=device, dtype=torch.bfloat16)

    def denoise(x, sigma):
        prediction = model(x.to(torch.bfloat16), sigma, context).float()
        return x - prediction * sigma.reshape(-1, 1, 1, 1)

    latent = res_multistep(
        denoise, noise(seed, width, height).to(device), sigmas().to(device), progress
    )
    return (latent / 0.3611 + 0.1159).cpu()
