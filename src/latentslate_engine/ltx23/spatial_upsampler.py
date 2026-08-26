"""Standalone LTX 2.3 spatial latent upsampler for the canonical T2V fixture."""

from __future__ import annotations

import json

import torch
from torch import nn

from .checkpoint import Ltx23Checkpoint


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class _LatentUpsampler(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, blocks: int) -> None:
        super().__init__()
        self.initial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()
        self.res_blocks = nn.ModuleList(_ResBlock(mid_channels) for _ in range(blocks))
        self.upsampler = nn.Sequential(
            nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)
        )
        self.post_upsample_res_blocks = nn.ModuleList(_ResBlock(mid_channels) for _ in range(blocks))
        self.final_conv = nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _, _ = latent.shape
        x = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            x = block(x)

        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, x.shape[1], x.shape[3], x.shape[4])
        x = self.upsampler(x)
        x = x.reshape(batch * frames, -1, 2, 2, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(batch * frames, -1, x.shape[4] * 2, x.shape[5] * 2)
        x = x.reshape(batch, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)

        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)


class Ltx23SpatialUpsampler:
    """Own the fixture's x2 latent upscaler and required VAE channel statistics."""

    def __init__(self, upsampler_path: str, checkpoint_path: str, device: str = "cuda") -> None:
        upsampler_checkpoint = Ltx23Checkpoint(upsampler_path)
        config = json.loads(upsampler_checkpoint.metadata["config"])
        if config != {
            "_class_name": "LatentUpsampler",
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "spatial_scale": 2.0,
            "rational_resampler": False,
        }:
            raise ValueError("unexpected LTX 2.3 spatial upsampler configuration")

        with torch.device("meta"):
            self.model = _LatentUpsampler(128, 1024, 4)
        self.model.load_state_dict(
            {name: upsampler_checkpoint.tensor(name) for name in upsampler_checkpoint.tensor_names},
            assign=True,
        )
        self.model.to(device=device, dtype=torch.bfloat16).eval()

        vae_checkpoint = Ltx23Checkpoint(checkpoint_path)
        self._mean = vae_checkpoint.tensor("vae.per_channel_statistics.mean-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, -1, 1, 1, 1)
        self._std = vae_checkpoint.tensor("vae.per_channel_statistics.std-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, -1, 1, 1, 1)

    @torch.inference_mode()
    def upsample(self, latents: torch.Tensor) -> torch.Tensor:
        input_dtype = latents.dtype
        x = latents.to(dtype=torch.bfloat16)
        x = x * self._std + self._mean
        x = self.model(x)
        return ((x - self._mean) / self._std).to(dtype=input_dtype)

    def close(self) -> None:
        self.model = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()
