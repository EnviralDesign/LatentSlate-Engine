"""Small operation set required by the pinned LTX 2.3 transformer source."""

from __future__ import annotations

import torch
from torch import nn

from .fp8_linear import Ltx23Fp8Linear, Ltx23PlainLinear


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
    output = layer(x) if isinstance(layer, Ltx23Linear) else torch.nn.functional.linear(x, layer.weight, layer.bias)
    if activation == "gelu_tanh":
        return torch.nn.functional.gelu(output, approximate="tanh")
    raise ValueError(f"unsupported activation: {activation}")


class Ltx23Linear(nn.Linear):
    """Pinned-model linear shell whose weights are bound after meta construction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._latentslate_weight = None
        self._latentslate_device_index = None

    def bind_ltx23_weight(self, checkpoint, prefix: str, vbar, device_index: int) -> None:
        if f"{prefix}.weight_scale" in checkpoint.tensor_names:
            weight = Ltx23Fp8Linear(checkpoint, prefix)
        else:
            weight = Ltx23PlainLinear(checkpoint, prefix)
        weight.allocate(vbar)
        self._latentslate_weight = weight
        self._latentslate_device_index = device_index

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._latentslate_weight is None:
            return super().forward(input)
        grouped = getattr(self, "_latentslate_grouped", False)
        prepared = getattr(self, "_latentslate_prepared", None)
        weight, bias = prepared if prepared is not None else self._latentslate_weight.materialize(self._latentslate_device_index)
        try:
            return torch.nn.functional.linear(input, weight, bias)
        finally:
            if not grouped:
                self._latentslate_weight.unpin(self._latentslate_device_index)

    def unpin_ltx23_weight(self) -> None:
        if self._latentslate_weight is not None:
            self._latentslate_weight.unpin(self._latentslate_device_index)


class _Operations:
    Linear = Ltx23Linear
    LayerNorm = nn.LayerNorm
    RMSNorm = nn.RMSNorm


operations = _Operations()
