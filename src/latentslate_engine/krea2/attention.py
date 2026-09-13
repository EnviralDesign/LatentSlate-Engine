"""Krea's pinned Comfy Torch attention dispatch, without runtime dependencies."""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention(q, k, v, mask=None, *, enable_gqa=False):
    """Preserve small-input dispatch and the masked-GQA availability boundary."""
    if q.numel() < 1024 * 128:
        return F.scaled_dot_product_attention(q, k, v, mask, enable_gqa=enable_gqa)
    with sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        set_priority=True,
    ):
        if enable_gqa and mask is not None and q.shape[-3] != k.shape[-3]:
            params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, True)
            available = (
                torch.backends.cuda.can_use_flash_attention(params)
                or torch.backends.cuda.can_use_cudnn_attention(params)
                or torch.backends.cuda.can_use_efficient_attention(params)
            )
            if not available:
                repeats = q.shape[-3] // k.shape[-3]
                k = k.repeat_interleave(repeats, dim=-3)
                v = v.repeat_interleave(repeats, dim=-3)
                enable_gqa = False
        return F.scaled_dot_product_attention(q, k, v, mask, enable_gqa=enable_gqa)
