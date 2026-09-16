"""Seeded patch rounding adapted from ComfyUI 1a14b82e comfy/float.py (GPL-3.0)."""

import comfy_kitchen as ck
import torch
import torch.nn.functional as F
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
)


def _nvfp4_blocked_scales(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix
        input_matrix = padded
    row_blocks = padded_rows // 128
    col_blocks = padded_cols // 4
    blocks = input_matrix.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(padded_rows, padded_cols)
    )


def _stochastic_fp4_e2m1(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shape = x.shape
    sign = torch.signbit(x).to(torch.uint8)
    exponent = torch.floor(torch.log2(x.abs()) + 1.0).clamp(0, 3)
    x = (
        x
        + (
            torch.rand(
                x.size(),
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
                generator=generator,
            )
            - 0.5
        )
        * (2 ** (exponent - 2.0))
        * 1.25
    )
    x = x.abs()
    exponent = torch.floor(torch.log2(x) + 1.1925).clamp(0, 3)
    mantissa = (
        torch.where(
            exponent > 0,
            (x / (2.0 ** (exponent - 1)) - 1.0) * 2.0,
            x * 2.0,
            out=x,
        )
        .round()
        .to(torch.uint8)
    )
    fp4 = (sign << 3) | (exponent.to(torch.uint8) << 1) | mantissa
    flat = fp4.view(-1)
    return ((flat[0::2] << 4) | flat[1::2]).reshape(*shape[:-1], -1)


def _stochastic_quantize_nvfp4(
    value: torch.Tensor, scale: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = value.shape
    padded_rows = ((rows + 15) // 16) * 16
    padded_cols = ((cols + 15) // 16) * 16
    if (rows, cols) != (padded_rows, padded_cols):
        value = F.pad(value, (0, padded_cols - cols, 0, padded_rows - rows))
    shape = value.shape
    qdata = torch.empty(
        (shape[0], shape[1] // 2), dtype=torch.uint8, device=value.device
    )
    block_scales = torch.empty(
        (shape[0], shape[1] // 16),
        dtype=torch.float8_e4m3fn,
        device=value.device,
    )
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    slice_count = max(1, value.numel() / (4096 * 4096))
    slice_size = max(1, round(shape[0] / slice_count))
    for start in range(0, shape[0], slice_size):
        current = value[start : start + slice_size]
        current_shape = current.shape
        blocks = current.reshape(current_shape[0], -1, 16)
        current_scales = torch.clamp(
            (torch.amax(torch.abs(blocks), dim=-1) / 6.0) / scale.to(current.dtype),
            max=448.0,
        ).to(torch.float8_e4m3fn)
        blocks = blocks / (
            scale.to(current.dtype) * current_scales.to(current.dtype)
        ).unsqueeze(-1)
        qdata[start : start + slice_size].copy_(
            _stochastic_fp4_e2m1(blocks.view(current_shape).nan_to_num(), generator)
        )
        block_scales[start : start + slice_size].copy_(current_scales)
    return qdata, _nvfp4_blocked_scales(block_scales)


def requantize(weight, representation, seed):
    if representation == "nvfp4":
        scale = (weight.abs().amax() / (448.0 * 6.0)).float()
        data, blocks = _stochastic_quantize_nvfp4(weight, scale, seed)
        return QuantizedTensor(
            data,
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=scale,
                block_scale=blocks,
                orig_dtype=weight.dtype,
                orig_shape=tuple(weight.shape),
            ),
        )
    scale = weight.abs().amax().float() / 448.0
    scaled = weight * (1.0 / scale).to(weight.dtype)
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    random = torch.randint(
        0,
        256,
        weight.shape,
        dtype=torch.uint8,
        device=weight.device,
        generator=generator,
    )
    data = ck.stochastic_rounding_fp8(scaled, random, torch.float8_e4m3fn)
    return QuantizedTensor(
        data,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=scale, orig_dtype=weight.dtype, orig_shape=tuple(weight.shape)
        ),
    )
