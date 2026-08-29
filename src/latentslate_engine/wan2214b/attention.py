from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_PRIORITY = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.is_cuda and q.numel() >= 1024 * 128:
        with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
