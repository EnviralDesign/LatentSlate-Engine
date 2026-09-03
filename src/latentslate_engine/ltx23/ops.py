"""Small operation set required by the pinned LTX 2.3 transformer source."""

from __future__ import annotations

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .fp8_linear import (
    Ltx23Fp8Linear,
    Ltx23Int8Linear,
    Ltx23Nvfp4Linear,
    Ltx23PlainLinear,
)

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


def _requantize_patched_nvfp4(
    weight: torch.Tensor, prefix: str
) -> QuantizedTensor:
    """Match Comfy's NVFP4 LoRA re-quantization from the logical layer shape."""
    scale = (torch.amax(weight.abs()) / (448.0 * 6.0)).to(dtype=torch.float32)
    qdata, block_scale = _stochastic_quantize_nvfp4(
        weight, scale, _string_to_seed(prefix.removeprefix("model."))
    )
    return QuantizedTensor(
        qdata,
        "TensorCoreNVFP4Layout",
        TensorCoreNVFP4Layout.Params(
            scale=scale,
            block_scale=block_scale,
            orig_dtype=weight.dtype,
            orig_shape=tuple(weight.shape),
        ),
    )


def _stochastic_quantize_nvfp4(
    weight: torch.Tensor, scale: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ComfyUI's seeded NVFP4 rounding and blocked-scale layout.

    This is a narrow GPLv3 adaptation of the pinned
    ``comfy.float.stochastic_round_quantize_nvfp4_by_block`` path.
    """
    rows, columns = weight.shape
    padded_rows = (rows + 15) // 16 * 16
    padded_columns = (columns + 15) // 16 * 16
    if (padded_rows, padded_columns) != (rows, columns):
        weight = torch.nn.functional.pad(
            weight, (0, padded_columns - columns, 0, padded_rows - rows)
        )
    output_fp4 = torch.empty(
        (weight.shape[0], weight.shape[1] // 2),
        dtype=torch.uint8,
        device=weight.device,
    )
    output_block = torch.empty(
        (weight.shape[0], weight.shape[1] // 16),
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    num_slices = max(1, weight.numel() / (4096 * 4096))
    slice_size = max(1, round(weight.shape[0] / num_slices))
    for start in range(0, weight.shape[0], slice_size):
        fp4, block_scale = _stochastic_quantize_nvfp4_block(
            weight[start : start + slice_size], scale, generator
        )
        output_fp4[start : start + slice_size].copy_(fp4)
        output_block[start : start + slice_size].copy_(block_scale)
    row_blocks = (output_block.shape[0] + 127) // 128
    column_blocks = (output_block.shape[1] + 3) // 4
    padded_block_scale = torch.zeros(
        (row_blocks * 128, column_blocks * 4),
        device=output_block.device,
        dtype=output_block.dtype,
    )
    padded_block_scale[: output_block.shape[0], : output_block.shape[1]] = output_block
    blocked = (
        padded_block_scale.view(row_blocks, 128, column_blocks, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(row_blocks * 128, column_blocks * 4)
    )
    return output_fp4, blocked


def _stochastic_quantize_nvfp4_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one source-compatible NVFP4 random-rounding slice."""
    blocks = weight.reshape(weight.shape[0], -1, 16)
    block_scale = torch.clamp(
        torch.amax(blocks.abs(), dim=-1) / 6.0 / scale.to(weight.dtype), max=448.0
    ).to(torch.float8_e4m3fn)
    normalized = (
        blocks
        / (scale.to(weight.dtype) * block_scale.to(weight.dtype).unsqueeze(-1))
    ).reshape(weight.shape).nan_to_num()
    sign = torch.signbit(normalized).to(torch.uint8)
    exponent = torch.floor(torch.log2(normalized.abs()) + 1.0).clamp(0, 3)
    normalized.add_(
        (torch.rand(
            normalized.size(),
            dtype=normalized.dtype,
            layout=normalized.layout,
            device=normalized.device,
            generator=generator,
        ) - 0.5)
        * (2 ** (exponent - 2.0))
        * 1.25
    )
    normalized = normalized.abs()
    exponent = torch.floor(torch.log2(normalized) + 1.1925).clamp(0, 3)
    mantissa = torch.where(
        exponent > 0,
        (normalized / (2.0 ** (exponent - 1)) - 1.0) * 2.0,
        normalized * 2.0,
    ).round().to(torch.uint8)
    fp4 = (sign << 3) | (exponent.to(torch.uint8) << 1) | mantissa
    packed = (fp4.view(-1)[0::2] << 4) | fp4.view(-1)[1::2]
    return packed.reshape(weight.shape[0], weight.shape[1] // 2), block_scale


def _quantize_fp8_input(
    input: torch.Tensor, scale: torch.Tensor | None
) -> QuantizedTensor:
    """Match pinned Comfy's deterministic FP8 activation quantization."""
    if scale is None:
        # Pinned Comfy treats an omitted TensorCoreFP8 activation scale as one.
        # Kitchen's convenience conversion instead recalculates a scale, which
        # changes the model path for weight-only FP8 checkpoints.
        scale = torch.ones((), device=input.device, dtype=torch.float32)
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


def _quantize_input(
    input: torch.Tensor, weight: QuantizedTensor, scale: torch.Tensor | None
) -> QuantizedTensor:
    if weight.layout_cls is TensorCoreFP8Layout:
        return _quantize_fp8_input(input, scale)
    return QuantizedTensor.from_float(input, weight._layout_cls, scale=scale)


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
        weight_tensor = checkpoint.tensor(f"{prefix}.weight")
        if weight_tensor.dtype is torch.uint8:
            weight = Ltx23Nvfp4Linear(checkpoint, prefix, tuple(self.weight.shape))
        elif weight_tensor.dtype is torch.int8:
            weight = Ltx23Int8Linear(checkpoint, prefix)
        elif f"{prefix}.weight_scale" in checkpoint.tensor_names:
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
            quantized_input = (
                quantized_weight and weight.layout_cls is not TensorWiseINT8Layout
            )
            if lora is not None:
                if quantized_weight and weight.layout_cls is TensorWiseINT8Layout:
                    raise ValueError("LTX INT8 transformer adapters are not implemented")
                if quantized_weight:
                    weight = weight.to(dtype=input.dtype).dequantize()
                weight = lora.apply(
                    self._latentslate_weight.prefix,
                    weight,
                    getattr(self, "_latentslate_lora_prepared", None),
                    disposable_weight=quantized_weight,
                )
                if quantized_weight:
                    if isinstance(self._latentslate_weight, Ltx23Nvfp4Linear):
                        weight = _requantize_patched_nvfp4(
                            weight,
                            self._latentslate_weight.prefix,
                        )
                    else:
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
                    _quantize_input(reshaped, weight, input_scale),
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
