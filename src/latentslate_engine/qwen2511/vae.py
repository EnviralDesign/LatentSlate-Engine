"""Qwen 2511 single-frame reference encoder and shared Qwen Image decoder.

Encoder adapted from ComfyUI 12d5279438bfefc058a269eae805ceab6047777f,
comfy/ldm/wan/vae.py (GPL-3.0), original copyright Alibaba Wan Team 2024–2025.
The image path does not execute temporal resampling or maintain temporal caches.
"""

from itertools import pairwise
from pathlib import Path

import torch
from torch import nn

from latentslate_engine.mapped_checkpoint import MappedCheckpoint
from latentslate_engine.qwen_image_vae import (
    AttentionBlock,
    CausalConv3d,
    QwenImageDecoder,
    RMSNorm,
    ResidualBlock,
)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2)
        )

    def forward(self, x):
        return self.resample(x[:, :, 0]).unsqueeze(2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv3d(3, 96, 3, padding=1)
        layers = []
        for stage, (in_dim, out_dim) in enumerate(pairwise((96, 96, 192, 384, 384))):
            for _ in range(2):
                layers.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if stage != 3:
                layers.append(Downsample(out_dim))
        self.downsamples = nn.Sequential(*layers)
        self.middle = nn.Sequential(
            ResidualBlock(384, 384), AttentionBlock(384), ResidualBlock(384, 384)
        )
        self.head = nn.Sequential(
            RMSNorm(384), nn.SiLU(), CausalConv3d(384, 32, 3, padding=1)
        )

    def forward(self, x):
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class QwenImageAutoencoder(QwenImageDecoder):
    """Encode and decode one RGB frame with the Qwen Image VAE checkpoint."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.conv1 = CausalConv3d(32, 32, 1)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode CPU BHWC values in [0, 1] to the raw, unnormalized VAE latent."""
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] != 3:
            raise ValueError("Qwen Image encode requires one RGB image")
        device = self.conv1.weight.device
        # The reference treats the image batch as the one-frame temporal axis.
        # Keep this view order: it determines the encoder convolution layout.
        x = pixels.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        x = (x * 2.0 - 1.0).to(torch.bfloat16).to(device)
        return self.conv1(self.encoder(x)).chunk(2, dim=1)[0].float().cpu()


def load_vae(path: Path, device: torch.device) -> QwenImageAutoencoder:
    """Load only the checkpoint weights exercised by single-frame editing."""
    source = MappedCheckpoint(path)
    with torch.device("meta"):
        model = QwenImageAutoencoder()
    model.load_state_dict(
        {name: source.tensor(name) for name in model.state_dict()}, assign=True
    )
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
