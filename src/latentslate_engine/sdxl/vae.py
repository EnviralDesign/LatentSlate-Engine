"""SDXL latent decoder with the checkpoint's post-quantization convolution.

Narrow adaptation of ComfyUI 1a14b82e (GPL-3.0),
comfy/ldm/modules/diffusionmodules/model.py's 2D decoder path.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from latentslate_engine.torch_attention import attention


def norm(channels):
    return nn.GroupNorm(32, channels, eps=1e-6)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = norm(512)
        self.q = nn.Conv2d(512, 512, 1)
        self.k = nn.Conv2d(512, 512, 1)
        self.v = nn.Conv2d(512, 512, 1)
        self.proj_out = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        h = self.norm(x)
        batch, channels, height, width = h.shape
        q, k, v = (
            projection(h).view(batch, 1, channels, -1).transpose(2, 3).contiguous()
            for projection in (self.q, self.k, self.v)
        )
        h = attention(q, k, v).transpose(2, 3).reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(512, 512)
        self.mid.attn_1 = AttnBlock()
        self.mid.block_2 = ResnetBlock(512, 512)
        self.up = nn.ModuleList()
        channels = 512
        for level in reversed(range(4)):
            output_channels = (128, 256, 512, 512)[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(3):
                up.block.append(ResnetBlock(channels, output_channels))
                channels = output_channels
            if level != 0:
                up.upsample = Upsample(channels)
            self.up.insert(0, up)
        self.norm_out = norm(128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, latent):
        x = self.conv_in(self.post_quant_conv(latent))
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))
        for level in reversed(range(4)):
            for block in self.up[level].block:
                x = block(x)
            if level != 0:
                x = self.up[level].upsample(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def load_decoder(path: Path, device: torch.device, *, embedded=False):
    """Load only the base decoder and post-quantization convolution."""
    with torch.device("meta"):
        model = Decoder()
    prefix = "first_stage_model." if embedded else ""
    with safe_open(path, framework="pt") as source:
        tensor_names = source.keys()
        state = {}
        for key in tensor_names:
            if key.startswith(prefix + "decoder."):
                target = key.removeprefix(prefix + "decoder.")
            elif key.startswith(prefix + "post_quant_conv."):
                target = key.removeprefix(prefix)
            else:
                continue
            state[target] = source.get_tensor(key).to(
                device=device, dtype=torch.bfloat16
            )
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


@torch.inference_mode()
def decode(model, latent, device):
    """Decode SDXL sampler latents with its 0.13025 latent scale."""
    pixels = model((latent / 0.13025).to(device=device, dtype=torch.bfloat16)).float()
    return pixels.add_(1.0).div_(2.0).clamp_(0.0, 1.0).movedim(1, -1).cpu()
