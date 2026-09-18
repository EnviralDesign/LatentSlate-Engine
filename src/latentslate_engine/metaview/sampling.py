"""MetaView FlowMatch schedule and Euler integration (Apache-2.0/GPL-3.0)."""

import math
import torch
from latentslate_engine.qwen2511.sampling import noise, normalize, denormalize


def sigmas(width, height, steps=8):
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = (width // 16) * (height // 16) * slope + 0.5 - slope * 256
    sigma = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    sigma = math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0))
    remainder = 1.0 - sigma
    sigma = 1.0 - remainder / (remainder[-1] / 0.98)
    return torch.cat((sigma, sigma.new_zeros(1)))


@torch.inference_mode()
def sample(model, positive, reference, geometry, seed, width, height, device, progress=None):
    x = noise(seed, width, height).to(device)
    schedule = sigmas(width, height).to(device)
    reference = normalize(reference).to(device=device, dtype=torch.bfloat16)
    positive = positive.to(device=device, dtype=torch.bfloat16)
    for index in range(8):
        sigma = schedule[index]
        prediction = model(x.to(torch.bfloat16), sigma.expand(1), positive,
                           ref_latents=[reference], metaview=geometry).float()
        denoised = x - prediction * sigma
        x = x + (x - denoised) / sigma * (schedule[index + 1] - sigma)
        if progress is not None:
            progress(index + 1, 8)
    return denormalize(x.cpu())
