"""LTX 2.3 FP8 linear weights backed by AIMDO virtual VRAM.

This is deliberately an LTX-local building block. It preserves the pinned
checkpoint's float8 weights and Kitchen layout rather than expanding weights
to BF16 before inference.
"""

from __future__ import annotations

import importlib
import json
from contextlib import nullcontext

import torch
from comfy_aimdo import control as aimdo_control
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)

from .checkpoint import Ltx23Checkpoint


def _aimdo_modules(device_index: int):
    torch.cuda.init()
    if not aimdo_control.init(nvml_pressure=True):
        raise RuntimeError(
            f"unable to initialize comfy-aimdo for CUDA device {device_index}"
        )
    if not aimdo_control.devctxs and not aimdo_control.init_device(device_index):
        raise RuntimeError(
            f"unable to initialize comfy-aimdo for CUDA device {device_index}"
        )

    model_vbar = importlib.import_module("comfy_aimdo.model_vbar")
    if model_vbar.lib is None:
        model_vbar = importlib.reload(model_vbar)
    aimdo_torch = importlib.import_module("comfy_aimdo.torch")
    return model_vbar, aimdo_torch


def _aligned(offset: int, alignment: int = 1024) -> int:
    return (offset + alignment - 1) & -alignment


class Ltx23Fp8Linear:
    """One mapped LTX FP8 linear layer materialized only while it is in use."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._checkpoint = checkpoint
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._scale = checkpoint.tensor(f"{prefix}.weight_scale")
        input_scale_name = f"{prefix}.input_scale"
        self._input_scale = (
            checkpoint.tensor(input_scale_name)
            if input_scale_name in checkpoint.tensor_names
            else None
        )
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        if self._weight.dtype is not torch.float8_e4m3fn:
            raise ValueError(f"{prefix} is not an E4M3 FP8 linear layer")
        if self._scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} does not have a float32 FP8 scale")

        self._offsets: dict[str, int] = {}
        offset = 0
        values = [("weight", self._weight), ("scale", self._scale)]
        if self._input_scale is not None:
            values.append(("input_scale", self._input_scale))
        values.append(("bias", self._bias))
        for name, value in values:
            offset = _aligned(offset)
            self._offsets[name] = offset
            offset += value.nbytes
        self._allocation_size = _aligned(offset)
        self._allocation = None
        self._signature = None
        self._host_cache = None
        self._host_cache_offset = 0
        self._host_cache_loaded = False
        self._host_cache_aligned = False

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def source_size(self) -> int:
        return (
            self._weight.nbytes
            + self._scale.nbytes
            + (0 if self._input_scale is None else self._input_scale.nbytes)
            + self._bias.nbytes
        )

    @property
    def offload_size(self) -> int:
        """Pinned-Comfy dynamic-loader cost used to order VBAR allocations."""
        return self.source_size + self._weight.numel() * torch.bfloat16.itemsize

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def enable_host_cache(
        self, host_buffer, offset: int, aligned: bool = False
    ) -> None:
        self._host_cache = host_buffer
        self._host_cache_offset = offset
        self._host_cache_aligned = aligned

    def materialize(
        self,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> tuple[QuantizedTensor, torch.Tensor, torch.Tensor]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")

        signature = model_vbar.vbar_fault(self._allocation)
        device = torch.device("cuda", device_index)
        destination = (
            aimdo_torch.aimdo_to_tensor(self._allocation, device)
            if signature is not None
            else torch.empty((self._allocation_size,), dtype=torch.uint8, device=device)
        )
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self._signature
        )
        self._signature = signature
        if not resident:
            source_tensors = [
                ("weight", "weight", self._weight),
                ("scale", "weight_scale", self._scale),
            ]
            if self._input_scale is not None:
                source_tensors.append(
                    ("input_scale", "input_scale", self._input_scale)
                )
            source_tensors.append(("bias", "bias", self._bias))
            if self._host_cache_loaded:
                cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)
                cursor = self._host_cache_offset
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    for name, _, source in source_tensors:
                        source_offset = (
                            self._host_cache_offset + self._offsets[name]
                            if self._host_cache_aligned
                            else cursor
                        )
                        destination_offset = self._offsets[name]
                        destination[
                            destination_offset : destination_offset + source.nbytes
                        ].copy_(
                            cache[source_offset : source_offset + source.nbytes],
                            non_blocking=True,
                        )
                        if not self._host_cache_aligned:
                            cursor += source.nbytes
            else:
                cache = (
                    self._host_cache if self._host_cache is not None else host_buffer
                )
                cursor = (
                    self._host_cache_offset
                    if self._host_cache is not None
                    else host_offset
                )
                for name, checkpoint_suffix, source in source_tensors:
                    source_offset = (
                        self._host_cache_offset + self._offsets[name]
                        if self._host_cache is not None and self._host_cache_aligned
                        else cursor
                    )
                    offset = self._offsets[name]
                    self._checkpoint.copy_tensor_to_device(
                        f"{self.prefix}.{checkpoint_suffix}",
                        destination,
                        offset,
                        device_index,
                        stream,
                        cache,
                        source_offset,
                    )
                    if self._host_cache is None or not self._host_cache_aligned:
                        cursor += source.nbytes
                self._host_cache_loaded = self._host_cache is not None

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return (
                destination[offset : offset + source.nbytes]
                .view(source.dtype)
                .view(source.shape)
            )

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
        input_scale = (
            view("input_scale", self._input_scale)
            if self._input_scale is not None
            else None
        )
        return weight, view("bias", self._bias), input_scale

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)


class Ltx23Nvfp4Linear:
    """One mapped LTX NVFP4 linear layer materialized only while it is in use."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._checkpoint = checkpoint
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._block_scale = checkpoint.tensor(f"{prefix}.weight_scale")
        self._tensor_scale = checkpoint.tensor(f"{prefix}.weight_scale_2")
        input_scale_name = f"{prefix}.input_scale"
        self._input_scale = (
            checkpoint.tensor(input_scale_name)
            if input_scale_name in checkpoint.tensor_names
            else None
        )
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        if self._weight.dtype is not torch.uint8:
            raise ValueError(f"{prefix} is not an NVFP4 linear layer")
        if self._block_scale.dtype is not torch.float8_e4m3fn:
            raise ValueError(f"{prefix} does not have an E4M3 NVFP4 block scale")
        if self._tensor_scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} does not have a float32 NVFP4 tensor scale")

        self._offsets: dict[str, int] = {}
        offset = 0
        values = [
            ("weight", self._weight),
            ("block_scale", self._block_scale),
            ("tensor_scale", self._tensor_scale),
        ]
        if self._input_scale is not None:
            values.append(("input_scale", self._input_scale))
        values.append(("bias", self._bias))
        for name, value in values:
            offset = _aligned(offset)
            self._offsets[name] = offset
            offset += value.nbytes
        self._allocation_size = _aligned(offset)
        self._allocation = None
        self._signature = None
        self._host_cache = None
        self._host_cache_offset = 0
        self._host_cache_loaded = False
        self._host_cache_aligned = False

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def source_size(self) -> int:
        return (
            self._weight.nbytes
            + self._block_scale.nbytes
            + self._tensor_scale.nbytes
            + (0 if self._input_scale is None else self._input_scale.nbytes)
            + self._bias.nbytes
        )

    @property
    def offload_size(self) -> int:
        """Pinned-Comfy dynamic-loader cost used to order VBAR allocations."""
        return self.source_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def enable_host_cache(
        self, host_buffer, offset: int, aligned: bool = False
    ) -> None:
        self._host_cache = host_buffer
        self._host_cache_offset = offset
        self._host_cache_aligned = aligned

    def materialize(
        self,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> tuple[QuantizedTensor, torch.Tensor, torch.Tensor | None]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")

        signature = model_vbar.vbar_fault(self._allocation)
        device = torch.device("cuda", device_index)
        destination = (
            aimdo_torch.aimdo_to_tensor(self._allocation, device)
            if signature is not None
            else torch.empty((self._allocation_size,), dtype=torch.uint8, device=device)
        )
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self._signature
        )
        self._signature = signature
        if not resident:
            source_tensors = [
                ("weight", "weight", self._weight),
                ("block_scale", "weight_scale", self._block_scale),
                ("tensor_scale", "weight_scale_2", self._tensor_scale),
            ]
            if self._input_scale is not None:
                source_tensors.append(
                    ("input_scale", "input_scale", self._input_scale)
                )
            source_tensors.append(("bias", "bias", self._bias))
            if self._host_cache_loaded:
                cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)
                cursor = self._host_cache_offset
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    for name, _, source in source_tensors:
                        source_offset = (
                            self._host_cache_offset + self._offsets[name]
                            if self._host_cache_aligned
                            else cursor
                        )
                        destination_offset = self._offsets[name]
                        destination[
                            destination_offset : destination_offset + source.nbytes
                        ].copy_(
                            cache[source_offset : source_offset + source.nbytes],
                            non_blocking=True,
                        )
                        if not self._host_cache_aligned:
                            cursor += source.nbytes
            else:
                cache = (
                    self._host_cache if self._host_cache is not None else host_buffer
                )
                cursor = (
                    self._host_cache_offset
                    if self._host_cache is not None
                    else host_offset
                )
                for name, checkpoint_suffix, source in source_tensors:
                    source_offset = (
                        self._host_cache_offset + self._offsets[name]
                        if self._host_cache is not None and self._host_cache_aligned
                        else cursor
                    )
                    offset = self._offsets[name]
                    self._checkpoint.copy_tensor_to_device(
                        f"{self.prefix}.{checkpoint_suffix}",
                        destination,
                        offset,
                        device_index,
                        stream,
                        cache,
                        source_offset,
                    )
                    if self._host_cache is None or not self._host_cache_aligned:
                        cursor += source.nbytes
                self._host_cache_loaded = self._host_cache is not None

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return (
                destination[offset : offset + source.nbytes]
                .view(source.dtype)
                .view(source.shape)
            )

        weight = QuantizedTensor(
            view("weight", self._weight),
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=view("tensor_scale", self._tensor_scale),
                block_scale=view("block_scale", self._block_scale),
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(self._weight.shape),
            ),
        )
        input_scale = (
            view("input_scale", self._input_scale)
            if self._input_scale is not None
            else None
        )
        return weight, view("bias", self._bias), input_scale

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)


class Ltx23Int8Linear:
    """One mapped LTX tensorwise-INT8 linear layer faulted into virtual VRAM."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._checkpoint = checkpoint
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._scale = checkpoint.tensor(f"{prefix}.weight_scale")
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        config_name = f"{prefix}.comfy_quant"
        if config_name not in checkpoint.tensor_names:
            raise ValueError(f"missing INT8 metadata for {prefix}")
        try:
            config = json.loads(checkpoint.tensor(config_name).numpy().tobytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid INT8 metadata for {prefix}") from error
        if self._weight.dtype is not torch.int8:
            raise ValueError(f"{prefix} is not an INT8 linear layer")
        if self._scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} does not have a float32 INT8 scale")
        if config.get("format") != "int8_tensorwise":
            raise ValueError(f"unsupported INT8 quantization for {prefix}")
        self._convrot = bool(config.get("convrot", False))
        self._convrot_groupsize = int(config.get("convrot_groupsize", 256))
        if self._convrot_groupsize <= 0:
            raise ValueError(f"invalid INT8 rotation group size for {prefix}")

        self._offsets: dict[str, int] = {}
        offset = 0
        for name, value in (
            ("weight", self._weight),
            ("scale", self._scale),
            ("bias", self._bias),
        ):
            offset = _aligned(offset)
            self._offsets[name] = offset
            offset += value.nbytes
        self._allocation_size = _aligned(offset)
        self._allocation = None
        self._signature = None
        self._host_cache = None
        self._host_cache_offset = 0
        self._host_cache_loaded = False
        self._host_cache_aligned = False

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def source_size(self) -> int:
        return self._weight.nbytes + self._scale.nbytes + self._bias.nbytes

    @property
    def offload_size(self) -> int:
        """Pinned-Comfy dynamic-loader cost used to order VBAR allocations."""
        return self.source_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def enable_host_cache(
        self, host_buffer, offset: int, aligned: bool = False
    ) -> None:
        self._host_cache = host_buffer
        self._host_cache_offset = offset
        self._host_cache_aligned = aligned

    def materialize(
        self,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> tuple[QuantizedTensor, torch.Tensor, None]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")

        signature = model_vbar.vbar_fault(self._allocation)
        device = torch.device("cuda", device_index)
        destination = (
            aimdo_torch.aimdo_to_tensor(self._allocation, device)
            if signature is not None
            else torch.empty((self._allocation_size,), dtype=torch.uint8, device=device)
        )
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self._signature
        )
        self._signature = signature
        if not resident:
            source_tensors = (
                ("weight", "weight", self._weight),
                ("scale", "weight_scale", self._scale),
                ("bias", "bias", self._bias),
            )
            if self._host_cache_loaded:
                cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)
                cursor = self._host_cache_offset
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    for name, _, source in source_tensors:
                        source_offset = (
                            self._host_cache_offset + self._offsets[name]
                            if self._host_cache_aligned
                            else cursor
                        )
                        destination_offset = self._offsets[name]
                        destination[
                            destination_offset : destination_offset + source.nbytes
                        ].copy_(
                            cache[source_offset : source_offset + source.nbytes],
                            non_blocking=True,
                        )
                        if not self._host_cache_aligned:
                            cursor += source.nbytes
            else:
                cache = (
                    self._host_cache if self._host_cache is not None else host_buffer
                )
                cursor = (
                    self._host_cache_offset
                    if self._host_cache is not None
                    else host_offset
                )
                for name, checkpoint_suffix, source in source_tensors:
                    source_offset = (
                        self._host_cache_offset + self._offsets[name]
                        if self._host_cache is not None and self._host_cache_aligned
                        else cursor
                    )
                    offset = self._offsets[name]
                    self._checkpoint.copy_tensor_to_device(
                        f"{self.prefix}.{checkpoint_suffix}",
                        destination,
                        offset,
                        device_index,
                        stream,
                        cache,
                        source_offset,
                    )
                    if self._host_cache is None or not self._host_cache_aligned:
                        cursor += source.nbytes
                self._host_cache_loaded = self._host_cache is not None

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return (
                destination[offset : offset + source.nbytes]
                .view(source.dtype)
                .view(source.shape)
            )

        weight = QuantizedTensor(
            view("weight", self._weight),
            "TensorWiseINT8Layout",
            TensorWiseINT8Layout.Params(
                scale=view("scale", self._scale),
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(self._weight.shape),
                convrot=self._convrot,
                convrot_groupsize=self._convrot_groupsize,
            ),
        )
        return weight, view("bias", self._bias), None

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)


class Ltx23PlainLinear:
    """One non-quantized LTX linear layer faulted into AIMDO virtual VRAM."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str) -> None:
        self.prefix = prefix
        self._checkpoint = checkpoint
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._bias = checkpoint.tensor(f"{prefix}.bias")
        self._offsets = {"weight": 0, "bias": _aligned(self._weight.numel() * 2)}
        self._allocation_size = _aligned(self._offsets["bias"] + self._bias.numel() * 2)
        self._allocation = None
        self._signature = None
        self._host_cache = None
        self._host_cache_offset = 0
        self._host_cache_loaded = False
        self._host_cache_aligned = False

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def source_size(self) -> int:
        return self._weight.nbytes + self._bias.nbytes

    @property
    def offload_size(self) -> int:
        """Pinned-Comfy dynamic-loader cost used to order VBAR allocations."""
        return self.source_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def enable_host_cache(
        self, host_buffer, offset: int, aligned: bool = False
    ) -> None:
        self._host_cache = host_buffer
        self._host_cache_offset = offset
        self._host_cache_aligned = aligned

    def materialize(
        self,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")
        signature = model_vbar.vbar_fault(self._allocation)
        device = torch.device("cuda", device_index)
        destination = (
            aimdo_torch.aimdo_to_tensor(self._allocation, device)
            if signature is not None
            else torch.empty((self._allocation_size,), dtype=torch.uint8, device=device)
        )
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self._signature
        )
        self._signature = signature
        if not resident:
            source_tensors = (("weight", self._weight), ("bias", self._bias))
            if self._host_cache_loaded:
                cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)
                cursor = self._host_cache_offset
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    for name, source in source_tensors:
                        source_offset = (
                            self._host_cache_offset + self._offsets[name]
                            if self._host_cache_aligned
                            else cursor
                        )
                        destination_offset = self._offsets[name]
                        destination[
                            destination_offset : destination_offset + source.nbytes
                        ].copy_(
                            cache[source_offset : source_offset + source.nbytes],
                            non_blocking=True,
                        )
                        if not self._host_cache_aligned:
                            cursor += source.nbytes
            else:
                cache = (
                    self._host_cache if self._host_cache is not None else host_buffer
                )
                cursor = (
                    self._host_cache_offset
                    if self._host_cache is not None
                    else host_offset
                )
                for name, source in source_tensors:
                    source_offset = (
                        self._host_cache_offset + self._offsets[name]
                        if self._host_cache is not None and self._host_cache_aligned
                        else cursor
                    )
                    offset = self._offsets[name]
                    self._checkpoint.copy_tensor_to_device(
                        f"{self.prefix}.{name}",
                        destination,
                        offset,
                        device_index,
                        stream,
                        cache,
                        source_offset,
                    )
                    if self._host_cache is None or not self._host_cache_aligned:
                        cursor += source.nbytes
                self._host_cache_loaded = self._host_cache is not None

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return (
                destination[offset : offset + source.numel() * 2]
                .view(torch.bfloat16)
                .view(source.shape)
            )

        return view("weight", self._weight), view("bias", self._bias), None

    def unpin(self, device_index: int) -> None:
        if self._allocation is None:
            return
        model_vbar, _ = _aimdo_modules(device_index)
        model_vbar.vbar_unpin(self._allocation)
