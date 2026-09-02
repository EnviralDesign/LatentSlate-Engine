"""Klein transformer weights staged through AIMDO virtual VRAM.

This is intentionally local to the Klein runtime.  It holds checkpoint bytes on
the host and faults only the active linear layer into VRAM, matching the
lifetime required by the concrete oversized Klein checkpoints.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import struct
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from comfy_aimdo import control as aimdo_control
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)

if TYPE_CHECKING:
    from .model import Linear


_MAX_HEADER_BYTES = 100_000_000
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3FN": torch.float8_e4m3fn,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


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


def _discard_cuda_async_error(device_index: int) -> None:
    try:
        device = torch.device("cuda", device_index)
        torch.ones(1, dtype=torch.uint8, device=device) + torch.ones(
            1, dtype=torch.uint8, device=device
        )
    except RuntimeError:
        pass


class KleinCheckpoint:
    """Keep a Klein safetensors checkpoint mapped while its weights are staged."""

    def __init__(self, path: Path, key_prefix: str = "") -> None:
        self.path = path
        self._key_prefix = key_prefix
        file_size = os.path.getsize(path)
        if file_size < 8:
            raise ValueError(f"incomplete safetensors file: {path}")

        torch.cuda.init()
        if not aimdo_control.init(nvml_pressure=True):
            raise RuntimeError("unable to initialize comfy-aimdo")
        model_mmap = importlib.import_module("comfy_aimdo.model_mmap")
        if model_mmap.lib is None:
            model_mmap = importlib.reload(model_mmap)
        self._mapping = model_mmap.ModelMMAP(str(path))
        self._file_handle = self._mapping.get_file_handle()
        self._file_lock = threading.Lock()
        self._raw_buffer = (ctypes.c_uint8 * file_size).from_address(
            self._mapping.get()
        )
        raw_view = memoryview(self._raw_buffer)

        header_size = struct.unpack("<Q", raw_view[:8])[0]
        if header_size > _MAX_HEADER_BYTES or 8 + header_size > file_size:
            raise ValueError(f"invalid safetensors header: {path}")
        try:
            self._header: dict[str, Any] = json.loads(
                raw_view[8 : 8 + header_size].tobytes().decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors header: {path}") from error
        if not isinstance(self._header, dict):
            raise TypeError(f"invalid safetensors header: {path}")
        self._data_base_offset = 8 + header_size
        self._data = raw_view[self._data_base_offset :]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._header if name != "__metadata__")

    def tensor(self, name: str) -> torch.Tensor:
        source_name = f"{self._key_prefix}{name}"
        try:
            descriptor = self._header[source_name]
            start, end = descriptor["data_offsets"]
            dtype = _SAFETENSORS_DTYPES[descriptor["dtype"]]
            shape = descriptor["shape"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor descriptor for {name!r}") from error
        if start < 0 or end < start or end > len(self._data):
            raise ValueError(f"tensor {name!r} extends past the checkpoint data")
        if (
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            != end - start
        ):
            raise ValueError(f"tensor {name!r} does not match its declared shape")
        return torch.frombuffer(self._data[start:end], dtype=dtype).view(shape)

    def quantization_config(self, name: str) -> dict[str, Any]:
        value = self.tensor(name)
        if value.dtype is not torch.uint8:
            raise ValueError(f"invalid quantization metadata for {name}")
        try:
            config = json.loads(bytes(value.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid quantization metadata for {name}") from error
        if not isinstance(config, dict):
            raise ValueError(f"invalid quantization metadata for {name}")
        return config

    def copy_tensor_to_device(
        self,
        name: str,
        destination: torch.Tensor,
        destination_offset: int,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> None:
        source_name = f"{self._key_prefix}{name}"
        try:
            descriptor = self._header[source_name]
            start, end = descriptor["data_offsets"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor descriptor for {name!r}") from error
        size = end - start
        if (
            start < 0
            or end < start
            or end > len(self._data)
            or destination.device.type != "cuda"
            or destination_offset < 0
            or destination_offset + size > destination.nbytes
        ):
            raise ValueError(f"invalid direct transfer for tensor {name!r}")
        stream_ptr = 0 if stream is None else stream.cuda_stream
        with self._file_lock:
            if host_buffer is None:
                host_buffer_module = importlib.import_module("comfy_aimdo.host_buffer")
                if host_buffer_module.lib is None:
                    host_buffer_module = importlib.reload(host_buffer_module)
                host_buffer_module.read_file_to_device(
                    self._file_handle,
                    self._data_base_offset + start,
                    size,
                    stream_ptr,
                    destination.data_ptr() + destination_offset,
                    device_index,
                    mark_cold=False,
                )
            else:
                host_buffer.read_file_slice(
                    self._file_handle,
                    self._data_base_offset + start,
                    size,
                    offset=host_offset,
                    stream=stream_ptr,
                    device_ptr=destination.data_ptr() + destination_offset,
                    device=device_index,
                )


class KleinDynamicWeight:
    """One Klein linear's mapped parameter bundle."""

    def __init__(
        self, checkpoint: KleinCheckpoint, prefix: str, linear: Linear
    ) -> None:
        self.prefix = prefix
        self._checkpoint = checkpoint
        self._linear = linear
        self._weight = checkpoint.tensor(f"{prefix}.weight")
        self._nvfp4 = self._weight.dtype is torch.uint8
        self._int8_tensorwise = self._weight.dtype is torch.int8
        self._tensors: tuple[tuple[str, torch.Tensor], ...]
        if self._nvfp4:
            self._weight_scale = checkpoint.tensor(f"{prefix}.weight_scale")
            self._weight_scale_2 = checkpoint.tensor(f"{prefix}.weight_scale_2")
            if self._weight_scale.dtype is not torch.float8_e4m3fn:
                raise ValueError(f"invalid NVFP4 block scale for {prefix}")
            if self._weight_scale_2.dtype is not torch.float32:
                raise ValueError(f"invalid NVFP4 tensor scale for {prefix}")
            self._tensors = (
                ("weight", self._weight),
                ("weight_scale", self._weight_scale),
                ("weight_scale_2", self._weight_scale_2),
            )
        elif self._int8_tensorwise:
            self._weight_scale = checkpoint.tensor(f"{prefix}.weight_scale")
            expected_shape = (linear.out_features, linear.in_features)
            if tuple(self._weight.shape) != expected_shape:
                raise ValueError(
                    f"unexpected Klein linear shape for {prefix}: {tuple(self._weight.shape)}"
                )
            if self._weight_scale.dtype is not torch.float32 or tuple(
                self._weight_scale.shape
            ) != (linear.out_features, 1):
                raise ValueError(f"invalid INT8 weight scale for {prefix}")
            config = checkpoint.quantization_config(f"{prefix}.comfy_quant")
            if config.get("format") != "int8_tensorwise" or not config.get(
                "convrot", False
            ):
                raise ValueError(f"unsupported INT8 quantization for {prefix}")
            self._convrot_groupsize = int(config.get("convrot_groupsize", 256))
            if self._convrot_groupsize <= 0:
                raise ValueError(f"invalid ConvRot group size for {prefix}")
            self._tensors = (
                ("weight", self._weight),
                ("weight_scale", self._weight_scale),
            )
        else:
            expected_shape = (linear.out_features, linear.in_features)
            if tuple(self._weight.shape) != expected_shape:
                raise ValueError(
                    f"unexpected Klein linear shape for {prefix}: {tuple(self._weight.shape)}"
                )
            self._tensors = (("weight", self._weight),)

        self._offsets: dict[str, int] = {}
        offset = 0
        for name, value in self._tensors:
            offset = _aligned(offset)
            self._offsets[name] = offset
            offset += value.nbytes
        self._allocation_size = _aligned(offset)
        self._source_size = sum(value.nbytes for _, value in self._tensors)
        self._allocation = None
        self._signature = None
        self._host_cache = None
        self._host_cache_offset = 0
        self._host_cache_loaded = False
        self._host_pin = None
        self._host_pin_registered = False
        self._device_index: int | None = None

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def source_size(self) -> int:
        return self._source_size

    def allocate(self, vbar) -> None:
        if self._allocation is None:
            self._allocation = vbar.alloc(self._allocation_size)

    def enable_host_cache(self, host_buffer, offset: int) -> None:
        self._host_cache = host_buffer
        self._host_cache_offset = offset

    def materialize(self, device_index: int, stream: torch.cuda.Stream | None = None):
        if self._allocation is None:
            raise RuntimeError(f"{self.prefix} has not been assigned VBAR space")
        self._device_index = device_index
        model_vbar, aimdo_torch = _aimdo_modules(device_index)
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
            if self._host_cache_loaded:
                cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    for name, source in self._tensors:
                        source_offset = self._host_cache_offset + self._offsets[name]
                        destination_offset = self._offsets[name]
                        destination[
                            destination_offset : destination_offset + source.nbytes
                        ].copy_(
                            cache[source_offset : source_offset + source.nbytes],
                            non_blocking=True,
                        )
            else:
                for name, source in self._tensors:
                    self._checkpoint.copy_tensor_to_device(
                        f"{self.prefix}.{name}",
                        destination,
                        self._offsets[name],
                        device_index,
                        stream,
                        self._host_cache,
                        self._host_cache_offset + self._offsets[name],
                    )
                self._host_cache_loaded = self._host_cache is not None
                self._pin_host_cache(device_index)

        def view(name: str, source: torch.Tensor) -> torch.Tensor:
            offset = self._offsets[name]
            return (
                destination[offset : offset + source.nbytes]
                .view(source.dtype)
                .view(source.shape)
            )

        weight = view("weight", self._weight)
        if self._int8_tensorwise:
            return QuantizedTensor(
                weight,
                "TensorWiseINT8Layout",
                TensorWiseINT8Layout.Params(
                    scale=view("weight_scale", self._weight_scale),
                    orig_dtype=torch.bfloat16,
                    orig_shape=(self._linear.out_features, self._linear.in_features),
                    convrot=True,
                    convrot_groupsize=self._convrot_groupsize,
                ),
            )
        if not self._nvfp4:
            return weight
        return QuantizedTensor(
            weight,
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=view("weight_scale_2", self._weight_scale_2),
                block_scale=view("weight_scale", self._weight_scale),
                orig_dtype=torch.bfloat16,
                orig_shape=(self._linear.out_features, self._linear.in_features),
            ),
        )

    def unpin(self, device_index: int) -> None:
        if self._allocation is not None:
            model_vbar, _ = _aimdo_modules(device_index)
            model_vbar.vbar_unpin(self._allocation)

    def _pin_host_cache(self, device_index: int) -> None:
        if self._host_cache is None or self._host_pin is not None:
            return
        _, aimdo_torch = _aimdo_modules(device_index)
        self._host_pin = aimdo_torch.hostbuf_to_tensor(self._host_cache)[
            self._host_cache_offset : self._host_cache_offset + self._allocation_size
        ]
        self._host_pin.untyped_storage()._klein_host_cache = self._host_cache
        if (
            torch.cuda.cudart().cudaHostRegister(
                self._host_pin.data_ptr(), self._allocation_size, 1
            )
            != 0
        ):
            _discard_cuda_async_error(device_index)
            self._host_pin = None
            return
        self._host_pin_registered = True

    def close(self) -> None:
        if (
            self._host_pin_registered
            and torch.cuda.cudart().cudaHostUnregister(self._host_pin.data_ptr()) != 0
        ):
            _discard_cuda_async_error(self._device_index or 0)
        self._allocation = None
        self._signature = None
        self._host_pin = None
        self._host_pin_registered = False
        self._device_index = None
        self._host_cache = None
        self._checkpoint = None


class KleinDynamicWeights:
    """Own the staged linear weights for one warm Klein transformer."""

    def __init__(
        self, checkpoint_path: Path, transformer, device_index: int, key_prefix: str = ""
    ) -> None:
        from .model import Linear

        self._checkpoint = KleinCheckpoint(checkpoint_path, key_prefix)
        self._device_index = device_index
        bindings: list[tuple[Linear, KleinDynamicWeight]] = []
        for name, linear in transformer.named_modules():
            if not isinstance(linear, Linear):
                continue
            prefix = name
            binding = KleinDynamicWeight(self._checkpoint, prefix, linear)
            bindings.append((linear, binding))

        model_vbar, _ = _aimdo_modules(device_index)
        self._vbar = model_vbar.ModelVBAR(
            10 * sum(binding.source_size for _, binding in bindings), device_index
        )
        for linear, binding in bindings:
            binding.allocate(self._vbar)
            linear.bind_dynamic_weight(binding, device_index)

        host_buffer_module = importlib.import_module("comfy_aimdo.host_buffer")
        if host_buffer_module.lib is None:
            host_buffer_module = importlib.reload(host_buffer_module)
        self._bindings = bindings
        self._host_cache = host_buffer_module.HostBuffer(
            0,
            64 * 1024 * 1024,
            sum(binding.allocation_size for _, binding in bindings),
        )
        for _, binding in bindings:
            offset = self._host_cache.size
            self._host_cache.extend(binding.allocation_size, register=False)
            binding.enable_host_cache(self._host_cache, offset)

    def close(self) -> None:
        for linear, binding in getattr(self, "_bindings", ()):
            linear.clear_dynamic_weight()
            binding.close()
        self._bindings = ()
        self._host_cache = None
        self._vbar = None
        self._checkpoint = None
