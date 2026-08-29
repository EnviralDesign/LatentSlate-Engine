from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from comfy_kitchen import apply_rope1

from .attention import scaled_dot_product_attention
from .weights import WanWeights

DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS
NUM_LAYERS = 40
PATCH_SIZE = (1, 2, 2)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.to(torch.float32)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000, -torch.arange(half, device=position.device).to(position).div(half)
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def _rope(pos: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    scale = torch.linspace(
        0,
        (dim - 2) / dim,
        steps=dim // 2,
        dtype=torch.float64,
        device=pos.device,
    )
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    return out.reshape(*out.shape[:-1], 2, 2).to(torch.float32)


def rope_frequencies(t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros((t, h, w, 3), device=device, dtype=torch.float16)
    ids[..., 0] += torch.linspace(
        0, t - 1, steps=t, device=device, dtype=torch.float16
    ).reshape(-1, 1, 1)
    ids[..., 1] += torch.linspace(
        0, h - 1, steps=h, device=device, dtype=torch.float16
    ).reshape(1, -1, 1)
    ids[..., 2] += torch.linspace(
        0, w - 1, steps=w, device=device, dtype=torch.float16
    ).reshape(1, 1, -1)
    ids = ids.reshape(1, -1, 3)
    axes = [HEAD_DIM - 4 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6)]
    emb = torch.cat([_rope(ids[..., axis], axes[axis]) for axis in range(3)], dim=-3)
    return emb.unsqueeze(1).movedim(1, 2)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)


class WanT2VTransformer:
    def __init__(self, weights: WanWeights):
        self.weights = weights

    def _self_attention(
        self, x: torch.Tensor, freqs: torch.Tensor, block: int
    ) -> torch.Tensor:
        prefix = f"blocks.{block}.self_attn"
        q = self.weights.linear(x, f"{prefix}.q")
        q = _rms_norm(
            q, self.weights.affine(f"{prefix}.norm_q.weight", x.device, x.dtype)
        )
        q = apply_rope1(q.view(x.shape[0], x.shape[1], NUM_HEADS, HEAD_DIM), freqs)
        k = self.weights.linear(x, f"{prefix}.k")
        k = _rms_norm(
            k, self.weights.affine(f"{prefix}.norm_k.weight", x.device, x.dtype)
        )
        k = apply_rope1(k.view(x.shape[0], x.shape[1], NUM_HEADS, HEAD_DIM), freqs)
        v = self.weights.linear(x, f"{prefix}.v").view(
            x.shape[0], x.shape[1], NUM_HEADS, HEAD_DIM
        )
        return self.weights.linear(_attention(q, k, v), f"{prefix}.o")

    def _cross_attention(
        self, x: torch.Tensor, context: torch.Tensor, block: int
    ) -> torch.Tensor:
        prefix = f"blocks.{block}.cross_attn"
        q = self.weights.linear(x, f"{prefix}.q")
        q = _rms_norm(
            q, self.weights.affine(f"{prefix}.norm_q.weight", x.device, x.dtype)
        )
        k = self.weights.linear(context, f"{prefix}.k")
        k = _rms_norm(
            k, self.weights.affine(f"{prefix}.norm_k.weight", x.device, x.dtype)
        )
        v = self.weights.linear(context, f"{prefix}.v")
        q = q.view(q.shape[0], q.shape[1], NUM_HEADS, HEAD_DIM)
        k = k.view(k.shape[0], k.shape[1], NUM_HEADS, HEAD_DIM)
        v = v.view(v.shape[0], v.shape[1], NUM_HEADS, HEAD_DIM)
        return self.weights.linear(_attention(q, k, v), f"{prefix}.o")

    def _block(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        context: torch.Tensor,
        freqs: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        modulation = self.weights.affine(
            f"blocks.{index}.modulation", x.device, x.dtype
        )
        e = (modulation + e0).unbind(2)
        y = self._self_attention(
            torch.addcmul(e[0], F.layer_norm(x, (DIM,), eps=1e-6), 1 + e[1]),
            freqs,
            index,
        )
        x = torch.addcmul(x, y, e[2])
        norm3 = self.weights.affine(f"blocks.{index}.norm3.weight", x.device, x.dtype)
        x = x + self._cross_attention(
            F.layer_norm(x, (DIM,), norm3, eps=1e-6), context, index
        )
        y = torch.addcmul(e[3], F.layer_norm(x, (DIM,), eps=1e-6), 1 + e[4])
        y = F.gelu(self.weights.linear(y, f"blocks.{index}.ffn.0"), approximate="tanh")
        y = self.weights.linear(y, f"blocks.{index}.ffn.2")
        return torch.addcmul(x, y, e[5])

    @torch.inference_mode()
    def __call__(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        original_shape = latent.shape
        patched = self.weights.conv3d(
            latent.float(), "patch_embedding", stride=PATCH_SIZE
        )
        if trace is not None:
            trace["patch_embedding"] = patched.detach().float().cpu().contiguous()
        x = patched.to(latent.dtype)
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)

        e = sinusoidal_embedding_1d(256, timestep.flatten()).to(x.dtype)
        e = self.weights.linear(e, "time_embedding.0")
        e = F.silu(e)
        e = self.weights.linear(e, "time_embedding.2")
        if trace is not None:
            trace["time_embedding"] = e.detach().float().cpu().contiguous()
        e = e.reshape(timestep.shape[0], -1, DIM)
        projected = self.weights.linear(F.silu(e), "time_projection.1")
        if trace is not None:
            trace["time_projection"] = projected.detach().float().cpu().contiguous()
        e0 = projected.unflatten(2, (6, DIM))

        context = self.weights.linear(context, "text_embedding.0")
        context = F.gelu(context, approximate="tanh")
        context = self.weights.linear(context, "text_embedding.2")
        if trace is not None:
            trace["text_embedding"] = context.detach().float().cpu().contiguous()
        freqs = rope_frequencies(*grid, latent.device)

        for index in range(NUM_LAYERS):
            x = self._block(x, e0, context, freqs, index)
            if trace is not None and index == 0:
                trace["block_0"] = x.detach().float().cpu().contiguous()

        head_modulation = self.weights.affine("head.modulation", x.device, x.dtype)
        head_e = (head_modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        x = torch.addcmul(head_e[0], F.layer_norm(x, (DIM,), eps=1e-6), 1 + head_e[1])
        x = self.weights.linear(x, "head.head")
        b, c = x.shape[0], 16
        x = x[:, : math.prod(grid)].view(b, *grid, *PATCH_SIZE, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(b, c, *[a * b_ for a, b_ in zip(grid, PATCH_SIZE)])[
            :, :, : original_shape[2], : original_shape[3], : original_shape[4]
        ]
