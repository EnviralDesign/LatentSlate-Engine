"""Z-Image Turbo NextDiT, adapted from ComfyUI 1a14b82e (GPL-3.0).

Source: comfy/ldm/lumina/model.py and its Flux RoPE/MM-DiT timestep helpers.
This implements the exercised text-to-image path, without graph patches,
reference-image conditioning or the separate pixel-space architecture.
"""

import math

import comfy_kitchen as ck
import torch
from torch import nn
from torch.nn import functional as F

from latentslate_engine.torch_attention import attention

from .weights import Linear, RMSNorm


def rotary(ids):
    embeddings = []
    for index, dim in enumerate((32, 48, 48)):
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=ids.device
        )
        omega = 1.0 / (256.0**scale)
        out = torch.einsum("...n,d->...nd", ids[..., index].float(), omega)
        out = torch.stack((out.cos(), -out.sin(), out.sin(), out.cos()), dim=-1)
        embeddings.append(out.reshape(*out.shape[:-1], 2, 2).float())
    return torch.cat(embeddings, dim=-3).unsqueeze(2)


def pad_tokens(x, token):
    extra = (-x.shape[1]) % 32
    return torch.cat(
        (x, token.to(x, copy=True).unsqueeze(0).repeat(x.shape[0], extra, 1)), dim=1
    ), extra


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(Linear(256, 1024), nn.SiLU(), Linear(1024, 256))

    def forward(self, t, dtype):
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(128, dtype=torch.float32, device=t.device)
            / 128
        )
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat((args.cos(), args.sin()), dim=-1).to(dtype))


class JointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = Linear(3840, 11520, bias=False)
        self.out = Linear(3840, 3840, bias=False)
        self.q_norm = RMSNorm(128, device="meta")
        self.k_norm = RMSNorm(128, device="meta")

    def forward(self, x, freqs):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).split(3840, dim=-1)
        q, k, v = (item.view(batch, length, 30, 128) for item in (q, k, v))
        eps = (
            self.q_norm.eps
            if self.q_norm.eps is not None
            else torch.finfo(torch.float32).eps
        )
        q, k = ck.rms_rope(
            q,
            k,
            freqs,
            self.q_norm.weight.to(x.dtype),
            self.k_norm.weight.to(x.dtype),
            eps,
        )
        result = attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2))
        return self.out(result.transpose(1, 2).reshape(batch, length, 3840))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Linear(3840, 10240, bias=False)
        self.w2 = Linear(10240, 3840, bias=False)
        self.w3 = Linear(3840, 10240, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointTransformerBlock(nn.Module):
    def __init__(self, modulation=True):
        super().__init__()
        self.attention = JointAttention()
        self.feed_forward = FeedForward()
        self.attention_norm1 = RMSNorm(3840, eps=1e-5, device="meta")
        self.attention_norm2 = RMSNorm(3840, eps=1e-5, device="meta")
        self.ffn_norm1 = RMSNorm(3840, eps=1e-5, device="meta")
        self.ffn_norm2 = RMSNorm(3840, eps=1e-5, device="meta")
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(Linear(256, 15360))

    def forward(self, x, freqs, t=None):
        if self.modulation:
            scale_attn, gate_attn, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(
                4, dim=1
            )
            normalized = self.attention_norm1(x) * (1 + scale_attn.unsqueeze(1))
            x = x + gate_attn.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(normalized, freqs)
            )
            normalized = self.ffn_norm1(x) * (1 + scale_mlp.unsqueeze(1))
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(normalized)
            )
        else:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), freqs))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = nn.LayerNorm(3840, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(3840, 64)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), Linear(256, 3840))

    def forward(self, x, t):
        return self.linear(
            self.norm_final(x) * (1 + self.adaLN_modulation(t).unsqueeze(1))
        )


class NextDiT(nn.Module):
    """The official 30-layer, BF16-compute text-to-image topology."""

    def __init__(self):
        super().__init__()
        self.x_embedder = Linear(64, 3840)
        self.noise_refiner = nn.ModuleList([JointTransformerBlock() for _ in range(2)])
        self.context_refiner = nn.ModuleList(
            [JointTransformerBlock(False) for _ in range(2)]
        )
        self.t_embedder = TimestepEmbedder()
        self.cap_embedder = nn.Sequential(
            RMSNorm(2560, eps=1e-5, device="meta"), Linear(2560, 3840)
        )
        self.layers = nn.ModuleList([JointTransformerBlock() for _ in range(30)])
        self.final_layer = FinalLayer()
        self.x_pad_token = nn.Parameter(torch.empty(1, 3840, device="meta"))
        self.cap_pad_token = nn.Parameter(torch.empty(1, 3840, device="meta"))

    def forward(self, x, timesteps, context):
        batch, channels, original_h, original_w = x.shape
        x = F.pad(x, (0, (-original_w) % 2, 0, (-original_h) % 2), mode="circular")
        height, width = x.shape[-2:]
        t = self.t_embedder((1.0 - timesteps) * 1000.0, x.dtype)
        cap, _ = pad_tokens(self.cap_embedder(context), self.cap_pad_token)
        cap_ids = torch.zeros(1, cap.shape[1], 3, device=x.device)
        cap_ids[:, :, 0] = (
            torch.arange(cap.shape[1], device=x.device, dtype=torch.float32) + 1.0
        )
        cap_freqs = rotary(cap_ids)
        pixels = (
            x.view(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )
        img = self.x_embedder(pixels)
        img_ids = torch.zeros(1, (height // 2) * (width // 2), 3, device=x.device)
        img_ids[:, :, 0] = cap.shape[1] + 1
        img_ids[:, :, 1] = (
            torch.arange(height // 2, device=x.device, dtype=torch.float32)
            .view(-1, 1)
            .repeat(1, width // 2)
            .flatten()
        )
        img_ids[:, :, 2] = (
            torch.arange(width // 2, device=x.device, dtype=torch.float32)
            .view(1, -1)
            .repeat(height // 2, 1)
            .flatten()
        )
        img, extra = pad_tokens(img, self.x_pad_token)
        img_freqs = rotary(F.pad(img_ids, (0, 0, 0, extra)))
        for layer in self.context_refiner:
            cap = layer(cap, cap_freqs)
        for layer in self.noise_refiner:
            img = layer(img, img_freqs, t)
        cap_length = cap.shape[1]
        img = torch.cat((cap, img), dim=1)
        freqs = torch.cat((cap_freqs, img_freqs), dim=1)
        for layer in self.layers:
            img = layer(img, freqs, t)
        img = self.final_layer(img, t)
        img = img[:, cap_length : cap_length + (height // 2) * (width // 2)]
        img = (
            img.reshape(batch, height // 2, width // 2, 2, 2, 16)
            .permute(0, 5, 1, 3, 2, 4)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        return -img[:, :, :original_h, :original_w]
