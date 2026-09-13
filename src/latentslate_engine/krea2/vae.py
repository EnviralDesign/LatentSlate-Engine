"""Single-frame Qwen Image VAE decode from pinned ComfyUI's Wan 2.1 VAE.

ComfyUI 12d5279438bfefc058a269eae805ceab6047777f, GPL-3.0; original Wan
implementation copyright Alibaba Wan Team. The Krea image path has no temporal
cache. Contiguous attention inputs reproduce this oracle's kernel dispatch;
the existing Wan family's frozen decoder uses a different layout.
"""

from itertools import pairwise
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .weights import KreaCheckpoint
from .attention import attention


class CausalConv3d(nn.Conv3d):
    def forward(self, x):
        return F.conv3d(
            x,
            self.weight[:, :, -1:],
            self.bias,
            self.stride,
            (0, self.padding[1], self.padding[2]),
            self.dilation,
            self.groups,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, images=False):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(
            torch.empty((dim, 1, 1) if images else (dim, 1, 1, 1))
        )

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, _, h, w = x.shape
        x = x[:, :, 0]
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [
            value.view(b, 1, c, -1).transpose(2, 3).contiguous() for value in (q, k, v)
        ]
        x = attention(q, k, v).transpose(2, 3).reshape(b, c, h, w)
        return self.proj(x).unsqueeze(2) + identity


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
        )

    def forward(self, x):
        return self.resample(x[:, :, 0]).unsqueeze(2)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv3d(16, 384, 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(384, 384), AttentionBlock(384), ResidualBlock(384, 384)
        )
        layers = []
        for stage, (in_dim, out_dim) in enumerate(pairwise((384, 384, 384, 192, 96))):
            if stage:
                in_dim //= 2
            for _ in range(3):
                layers.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if stage != 3:
                layers.append(Upsample(out_dim))
        self.upsamples = nn.Sequential(*layers)
        self.head = nn.Sequential(
            RMSNorm(96), nn.SiLU(), CausalConv3d(96, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.head(self.upsamples(self.middle(self.conv1(x))))


class QwenImageDecoder(nn.Module):
    """Decode only one 16-channel frame; temporal generation is unsupported."""

    def __init__(self):
        super().__init__()
        self.conv2 = CausalConv3d(16, 16, 1)
        self.decoder = Decoder()

    def decode(self, latent):
        """Return RGB values in the checkpoint's [-1, 1] range."""
        if latent.ndim != 5 or tuple(latent.shape[:3]) != (1, 16, 1):
            raise ValueError("Krea Qwen Image decode requires one 16-channel frame")
        return self.decoder(self.conv2(latent))


def load_vae(path: Path, device: torch.device) -> QwenImageDecoder:
    """Load only the weights exercised by a single-frame decode."""
    source = KreaCheckpoint(path)
    with torch.device("meta"):
        model = QwenImageDecoder()
    model.load_state_dict(
        {name: source.tensor(name) for name in model.state_dict()}, assign=True
    )
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
