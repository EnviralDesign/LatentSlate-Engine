"""The small subset of reshape patterns reached by LTX 2.3 T2V."""

from __future__ import annotations

import torch


def rearrange(tensor: torch.Tensor, pattern: str, **sizes: int) -> torch.Tensor:
    normalized = " ".join(pattern.split())
    if normalized == "b c f h w -> b c (f h w)":
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    if normalized == "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)":
        b, c, fp1, hp2, wp3 = tensor.shape
        p1, p2, p3 = sizes["p1"], sizes["p2"], sizes["p3"]
        return tensor.view(b, c, fp1 // p1, p1, hp2 // p2, p2, wp3 // p3, p3).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, c * p1 * p2 * p3)
    if normalized == "b (f h w) (c p q) -> b c f (h p) (w q)":
        b, _, _ = tensor.shape
        f, h, w, p, q = (sizes[name] for name in ("f", "h", "w", "p", "q"))
        c = tensor.shape[-1] // (p * q)
        return tensor.view(b, f, h, w, c, p, q).permute(0, 4, 1, 2, 5, 3, 6).reshape(b, c, f, h * p, w * q)
    if normalized == "b c t f -> b t (c f)":
        return tensor.permute(0, 2, 1, 3).reshape(tensor.shape[0], tensor.shape[2], -1)
    if normalized == "b t (c f) -> b c t f":
        c, f = sizes["c"], sizes["f"]
        return tensor.view(tensor.shape[0], tensor.shape[1], c, f).permute(0, 2, 1, 3)
    raise NotImplementedError(f"LTX T2V does not use rearrange pattern: {pattern}")
