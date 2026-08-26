"""Small operation set required by the pinned LTX 2.3 transformer source."""

from __future__ import annotations

import torch
from torch import nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def optimized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    batch, tokens, width = q.shape
    head_width = width // heads
    q = q.view(batch, tokens, heads, head_width).transpose(1, 2)
    k = k.view(batch, -1, heads, head_width).transpose(1, 2)
    v = v.view(batch, -1, heads, head_width).transpose(1, 2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    return output.transpose(1, 2).reshape(batch, tokens, width)


def linear_input_act(layer: nn.Linear, x: torch.Tensor, activation: str) -> torch.Tensor:
    output = torch.nn.functional.linear(x, layer.weight, layer.bias)
    if activation == "gelu_tanh":
        return torch.nn.functional.gelu(output, approximate="tanh")
    raise ValueError(f"unsupported activation: {activation}")


class _Operations:
    Linear = nn.Linear
    LayerNorm = nn.LayerNorm
    RMSNorm = nn.RMSNorm


operations = _Operations()
