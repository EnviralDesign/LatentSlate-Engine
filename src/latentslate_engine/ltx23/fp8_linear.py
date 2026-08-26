"""LTX 2.3 FP8 linear weights backed by AIMDO virtual VRAM.

This is deliberately an LTX-local building block. It preserves the pinned
checkpoint's float8 weights and Kitchen layout rather than expanding weights
to BF16 before inference.
"""

from __future__ import annotations

import importlib

import torch
from comfy_aimdo import control as aimdo_control
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

from .checkpoint import Ltx23Checkpoint


def _aimdo_modules(device_index: int):
    torch.cuda.init()
    if not aimdo_control.init():
        raise RuntimeError(f"unable to initialize comfy-aimdo for CUDA device {device_index}")
    if not aimdo_control.devctxs and not aimdo_control.init_device(device_index):
        raise RuntimeError(f"unable to initialize comfy-aimdo for CUDA device {device_index}")

    model_vbar = importlib.import_module("comfy_aimdo.model_vbar")
    if model_vbar.lib is None:
        model_vbar = importlib.reload(model_vbar)
    aimdo_torch = importlib.import_module("comfy_aimdo.torch")
    return model_vbar, aimdo_torch


def _aligned(offset: int, alignment: int = 16) -> int:
    return (offset + alignment - 1) & -alignment


class Ltx23Fp8Linear:
    """One mapped LTX FP8 linear layer materialized only while it is in use."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._scale = checkpoint.tensor(f"{prefix}.weight_scale")
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        if self._weight.dtype is not torch.float8_e4m3fn:
            raise ValueError(f"{prefix} is not an E4M3 FP8 linear layer")
        if self._scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} does not have a float32 FP8 scale")

        self._offsets: dict[str, int] = {}
        offset = 0
        for name, value in (("weight", self._weight), ("scale", self._scale), ("bias", self._bias)):
            offset = _aligned(offset)
            self._offsets[name] = offset
            offset += value.nbytes
        self._allocation_size = _aligned(offset)
        self._allocation = None
        self._signature = None

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def materialize(self, device_index: int) -> tuple[QuantizedTensor, torch.Tensor]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")

        signature = model_vbar.vbar_fault(self._allocation)
        if signature is None:
            raise MemoryError(f"AIMDO could not fault {self.prefix} into virtual VRAM")

        device = torch.device("cuda", device_index)
        destination = aimdo_torch.aimdo_to_tensor(self._allocation, device)
        resident = model_vbar.vbar_signature_compare(signature, self._signature)
        self._signature = signature
        if not resident:
            for name, source in (("weight", self._weight), ("scale", self._scale), ("bias", self._bias)):
                offset = self._offsets[name]
                destination[offset : offset + source.nbytes].copy_(
                    source.reshape(-1).view(torch.uint8), non_blocking=True
                )

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return destination[offset : offset + source.nbytes].view(source.dtype).view(source.shape)

        scale = view("scale", self._scale)
        weight = QuantizedTensor(
            view("weight", self._weight),
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=scale,
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(self._weight.shape),
            ),
        )
        return weight, view("bias", self._bias)

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)


class Ltx23PlainLinear:
    """One non-quantized LTX linear layer faulted into AIMDO virtual VRAM."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        self._offsets = {"weight": 0, "bias": _aligned(self._weight.numel() * 2)}
        self._allocation_size = _aligned(self._offsets["bias"] + self._bias.numel() * 2)
        self._allocation = None
        self._signature = None

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def materialize(self, device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")
        signature = model_vbar.vbar_fault(self._allocation)
        if signature is None:
            raise MemoryError(f"AIMDO could not fault {self.prefix} into virtual VRAM")

        destination = aimdo_torch.aimdo_to_tensor(self._allocation, torch.device("cuda", device_index))
        resident = model_vbar.vbar_signature_compare(signature, self._signature)
        self._signature = signature
        if not resident:
            for name, source in (("weight", self._weight), ("bias", self._bias)):
                encoded = source.to(dtype=torch.bfloat16).reshape(-1).view(torch.uint8)
                offset = self._offsets[name]
                destination[offset : offset + encoded.numel()].copy_(encoded, non_blocking=True)

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return destination[offset : offset + source.numel() * 2].view(torch.bfloat16).view(source.shape)

        return view("weight", self._weight), view("bias", self._bias)

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)
