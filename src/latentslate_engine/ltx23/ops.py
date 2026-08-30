"""Small operation set required by the pinned LTX 2.3 transformer source."""

from __future__ import annotations

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .fp8_linear import Ltx23Fp8Linear, Ltx23PlainLinear

_SDPA_BACKEND_PRIORITY = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _string_to_seed(value: str) -> int:
    crc = 0xFFFFFFFF
    for character in value:
        crc ^= ord(character)
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def _requantize_patched_fp8(weight: torch.Tensor, prefix: str) -> QuantizedTensor:
    """Match pinned Comfy's seeded FP8 requantization after a LoRA patch."""
    scale = (
        torch.amax(weight.abs()).to(dtype=torch.float32)
        / torch.finfo(torch.float8_e4m3fn).max
    )
    scaled = weight * (1.0 / scale).to(weight.dtype)
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(_string_to_seed(prefix.removeprefix("model.")))
    random = torch.randint(
        0,
        256,
        scaled.size(),
        dtype=torch.uint8,
        layout=scaled.layout,
        device=scaled.device,
        generator=generator,
    )
    data = ck.stochastic_rounding_fp8(scaled, random, torch.float8_e4m3fn)
    return QuantizedTensor(
        data,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=scale.float(),
            orig_dtype=weight.dtype,
            orig_shape=tuple(weight.shape),
        ),
    )


def _quantize_fp8_input(input: torch.Tensor, scale: torch.Tensor) -> QuantizedTensor:
    """Match pinned Comfy's deterministic FP8 activation quantization."""
    data = ck.quantize_per_tensor_fp8(input, scale, torch.float8_e4m3fn)
    return QuantizedTensor(
        data,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=scale.float(),
            orig_dtype=input.dtype,
            orig_shape=tuple(input.shape),
        ),
    )


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
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
    if q.nelement() < 1024 * 128:
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
    else:
        with sdpa_kernel(_SDPA_BACKEND_PRIORITY, set_priority=True):
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0
            )
    return output.transpose(1, 2).reshape(batch, tokens, width)


def linear_input_act(
    layer: nn.Linear, x: torch.Tensor, activation: str
) -> torch.Tensor:
    if activation == "gelu_tanh":
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return (
            layer(x)
            if isinstance(layer, Ltx23Linear)
            else torch.nn.functional.linear(x, layer.weight, layer.bias)
        )
    raise ValueError(f"unsupported activation: {activation}")


class Ltx23Linear(nn.Linear):
    """Pinned-model linear shell whose weights are bound after meta construction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._latentslate_weight = None
        self._latentslate_device_index = None

    def bind_ltx23_weight(
        self, checkpoint, prefix: str, vbar, device_index: int
    ) -> None:
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
        weight, bias, input_scale = (
            prepared
            if prepared is not None
            else self._latentslate_weight.materialize(self._latentslate_device_index)
        )
        try:
            quantized_weight = isinstance(weight, QuantizedTensor)
            lora = getattr(self, "_latentslate_lora", None)
            quantized_input = quantized_weight
            if lora is not None:
                if quantized_weight:
                    weight = weight.to(dtype=input.dtype).dequantize()
                weight = lora.apply(
                    self._latentslate_weight.prefix,
                    weight,
                    getattr(self, "_latentslate_lora_prepared", None),
                    disposable_weight=quantized_weight,
                )
                if quantized_weight:
                    weight = _requantize_patched_fp8(
                        weight,
                        self._latentslate_weight.prefix,
                    )
            if quantized_input:
                input_shape = input.shape
                reshaped = (
                    input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input
                )
                output = torch.nn.functional.linear(
                    _quantize_fp8_input(reshaped, input_scale),
                    weight,
                    bias,
                )
                return (
                    output.reshape((*input_shape[:-1], weight.shape[0]))
                    if input.ndim >= 3
                    else output
                )
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
