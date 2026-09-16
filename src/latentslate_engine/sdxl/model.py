"""SDXL base UNet, adapted from ComfyUI 1a14b82e (GPL-3.0).

Sources: ldm/modules/diffusionmodules/openaimodel.py and modules/attention.py.
Only the ordinary two-dimensional base model path is retained.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from latentslate_engine.torch_attention import attention


def timestep_embedding(timesteps, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, incoming),
            nn.SiLU(),
            nn.Conv2d(incoming, outgoing, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(1280, outgoing))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, outgoing),
            nn.SiLU(),
            nn.Dropout(0),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
        )
        self.skip_connection = (
            nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb).to(h.dtype)[:, :, None, None]
        return self.skip_connection(x) + self.out_layers(h)


class CrossAttention(nn.Module):
    def __init__(self, channels, context):
        super().__init__()
        self.heads = channels // 64
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context, channels, bias=False)
        self.to_v = nn.Linear(context, channels, bias=False)
        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0))

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = (
            p(t).reshape(t.shape[0], -1, self.heads, 64).transpose(1, 2)
            for p, t in ((self.to_q, x), (self.to_k, context), (self.to_v, context))
        )
        out = attention(q, k, v).transpose(1, 2).reshape(x.shape)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, 8 * channels)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn1 = CrossAttention(channels, channels)
        self.attn2 = CrossAttention(channels, 2048)
        self.ff = nn.Module()
        self.ff.net = nn.Sequential(
            GEGLU(channels), nn.Dropout(0), nn.Linear(4 * channels, channels)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return x + self.ff.net(self.norm3(x))


class SpatialTransformer(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(channels) for _ in range(depth)
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x, context):
        batch, channels, height, width = x.shape
        h = self.norm(x).movedim(1, 3).flatten(1, 2).contiguous()
        h = self.proj_in(h)
        for block in self.transformer_blocks:
            h = block(h, context)
        h = (
            self.proj_out(h)
            .reshape(batch, height, width, channels)
            .movedim(3, 1)
            .contiguous()
        )
        return h + x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, output_shape):
        return self.conv(F.interpolate(x, size=output_shape[-2:], mode="nearest"))


def run_block(block, x, emb, context, output_shape=None):
    for layer in block:
        if isinstance(layer, ResBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context)
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape)
        else:
            x = layer(x)
    return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280)
        )
        self.label_emb = nn.Sequential(
            nn.Sequential(nn.Linear(2816, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        )
        self.input_blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(4, 320, 3, padding=1))]
        )
        channels = 320
        skips = [channels]
        for level, (out, depth) in enumerate(((320, 0), (640, 2), (1280, 10))):
            for _ in range(2):
                block = [ResBlock(channels, out)]
                channels = out
                if depth:
                    block.append(SpatialTransformer(channels, depth))
                self.input_blocks.append(nn.Sequential(*block))
                skips.append(channels)
            if level < 2:
                self.input_blocks.append(nn.Sequential(Downsample(channels)))
                skips.append(channels)
        self.middle_block = nn.Sequential(
            ResBlock(1280, 1280), SpatialTransformer(1280, 10), ResBlock(1280, 1280)
        )
        self.output_blocks = nn.ModuleList()
        for level, (out, depth) in reversed(
            list(enumerate(((320, 0), (640, 2), (1280, 10))))
        ):
            for index in range(3):
                block = [ResBlock(channels + skips.pop(), out)]
                channels = out
                if depth:
                    block.append(SpatialTransformer(channels, depth))
                if level and index == 2:
                    block.append(Upsample(channels))
                self.output_blocks.append(nn.Sequential(*block))
        self.out = nn.Sequential(
            nn.GroupNorm(32, 320), nn.SiLU(), nn.Conv2d(320, 4, 3, padding=1)
        )

    def forward(self, x, timestep, context, adm):
        emb = self.time_embed(timestep_embedding(timestep, 320).to(x.dtype))
        emb = emb + self.label_emb(adm)
        skips = []
        for block in self.input_blocks:
            x = run_block(block, x, emb, context)
            skips.append(x)
        x = run_block(self.middle_block, x, emb, context)
        for block in self.output_blocks:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            del skip
            x = run_block(block, x, emb, context, skips[-1].shape if skips else None)
        return self.out(x)
