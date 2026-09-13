"""Krea-family mapped checkpoints; AIMDO owns the virtual device allocation.

The narrow mapping/transfer code follows the existing measured Klein path and
pinned Comfy loader. This family owns its quantization and layer lifetimes.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import struct
import threading
from pathlib import Path
from typing import Any

import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from torch import nn
from torch.nn import functional as F
from comfy_aimdo import control as aimdo_control

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


def _discard_cuda_async_error(device: torch.device) -> None:
    try:
        torch.ones(1, dtype=torch.uint8, device=device) + torch.ones(
            1, dtype=torch.uint8, device=device
        )
    except RuntimeError:
        pass


class KreaCheckpoint:
    """Keep a Krea safetensors checkpoint mapped while its weights are staged."""

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


class Linear(nn.Linear):
    """A Krea linear with its checkpoint-owned mixed-precision dispatch."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device="meta", dtype=dtype)
        self.binding = None

    def forward(self, x):
        if self.binding is None:
            raise RuntimeError("Krea linear has no loaded checkpoint")
        binding = self.binding
        values = binding.materialize()
        try:
            weight = values["weight"]
            bias = values.get("bias")
            if bias is not None:
                bias = bias.to(x.dtype)
            if weight.dtype != torch.float8_e4m3fn:
                return F.linear(x, weight.to(x.dtype), bias)
            weight = QuantizedTensor(
                weight,
                "TensorCoreFP8Layout",
                TensorCoreFP8Layout.Params(
                    scale=values["weight_scale"],
                    orig_dtype=x.dtype,
                    orig_shape=(self.out_features, self.in_features),
                ),
            )
            if binding.full_precision:
                return F.linear(x, weight.dequantize().to(x.dtype), bias)
            original_shape = x.shape
            quantized = QuantizedTensor.from_float(
                x.reshape(-1, original_shape[-1]),
                "TensorCoreFP8Layout",
                scale=values.get("input_scale", 1.0),
            )
            return F.linear(quantized, weight, bias).reshape(
                *original_shape[:-1], self.out_features
            )
        finally:
            binding.unpin()


class KreaWeight:
    """One linear's raw weight/scales/bias faulted as a single bundle."""

    def __init__(self, owner, name, module, config):
        self.owner = owner
        self.name = name
        self.full_precision = config.get("full_precision_matrix_mult", False)
        self.tensors = {}
        self.offsets = {}
        size = 0
        for key in ("weight", "bias", "weight_scale", "input_scale"):
            if f"{name}.{key}" not in owner.checkpoint.tensor_names:
                continue
            value = owner.checkpoint.tensor(f"{name}.{key}")
            self.tensors[key] = value
            self.offsets[key] = size
            size += _aligned(value.nbytes)
        weight = self.tensors["weight"]
        if tuple(weight.shape) != (module.out_features, module.in_features):
            raise ValueError(f"Unsupported Krea linear shape: {name}")
        if weight.dtype == torch.float8_e4m3fn:
            if (
                config.get("format") != "float8_e4m3fn"
                or "weight_scale" not in self.tensors
            ):
                raise ValueError(f"Missing Krea FP8 metadata: {name}")
        elif weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported Krea weight representation: {name}")
        self.size = size
        self.allocation = None
        self.signature = None
        self.cached = False
        self.host_offset = 0
        self.host_pin = None

    def materialize(self):
        owner = self.owner
        model_vbar, aimdo_torch = _aimdo_modules(owner.device_index)
        signature = model_vbar.vbar_fault(self.allocation)
        destination = (
            aimdo_torch.aimdo_to_tensor(self.allocation, owner.device)
            if signature is not None
            else torch.empty(self.size, dtype=torch.uint8, device=owner.device)
        )
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self.signature
        )
        self.signature = signature
        if not resident:
            if self.cached:
                host = aimdo_torch.hostbuf_to_tensor(owner.host_cache)
                for key, value in self.tensors.items():
                    offset = self.offsets[key]
                    start = self.host_offset + offset
                    destination[offset : offset + value.nbytes].copy_(
                        host[start : start + value.nbytes], non_blocking=True
                    )
            else:
                for key in self.tensors:
                    offset = self.offsets[key]
                    owner.checkpoint.copy_tensor_to_device(
                        f"{self.name}.{key}",
                        destination,
                        offset,
                        owner.device_index,
                        host_buffer=owner.host_cache,
                        host_offset=self.host_offset + offset,
                    )
                self.cached = True
                pointer = owner.host_cache.get_raw_address() + self.host_offset
                if torch.cuda.cudart().cudaHostRegister(pointer, self.size, 1) == 0:
                    self.host_pin = pointer
                else:
                    _discard_cuda_async_error(owner.device)
        return {
            key: destination[self.offsets[key] : self.offsets[key] + value.nbytes]
            .view(value.dtype)
            .view(value.shape)
            for key, value in self.tensors.items()
        }

    def unpin(self):
        model_vbar, _ = _aimdo_modules(self.owner.device_index)
        model_vbar.vbar_unpin(self.allocation)


class KreaWeights:
    """Own one transformer's mapped source, host cache, and virtual VRAM."""

    def __init__(self, path: Path, model: nn.Module, device: torch.device):
        self.device = device
        self.device_index = device.index or 0
        self.checkpoint = KreaCheckpoint(path)
        metadata = self.checkpoint._header.get("__metadata__", {})
        config = json.loads(metadata.get("_quantization_metadata", "{}")).get(
            "layers", {}
        )
        self.bindings = []
        self.modules = []
        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                binding = KreaWeight(self, name, module, config.get(name, {}))
                self.bindings.append(binding)
                self.modules.append(module)
                linear_names.add(name)
                module.binding = binding
        model_vbar, _ = _aimdo_modules(self.device_index)
        self.vbar = model_vbar.ModelVBAR(
            10 * sum(b.size for b in self.bindings), self.device_index
        )
        host_module = importlib.import_module("comfy_aimdo.host_buffer")
        if host_module.lib is None:
            host_module = importlib.reload(host_module)
        self.host_cache = host_module.HostBuffer(
            0, 64 * 1024 * 1024, sum(b.size for b in self.bindings)
        )
        for binding in self.bindings:
            binding.allocation = self.vbar.alloc(binding.size)
            binding.host_offset = self.host_cache.size
            self.host_cache.extend(binding.size, register=False)
        for name, parameter in list(model.named_parameters()):
            parent, _, key = name.rpartition(".")
            if parent in linear_names:
                continue
            value = self.checkpoint.tensor(name)
            if value.shape != parameter.shape:
                raise ValueError(f"Unsupported Krea parameter shape: {name}")
            setattr(
                model.get_submodule(parent),
                key,
                nn.Parameter(value.to(device), requires_grad=False),
            )

    def close(self):
        """Release the model bindings before releasing their storage owners."""
        torch.cuda.synchronize(self.device)
        for module in self.modules:
            module.binding = None
        self.modules.clear()
        for binding in self.bindings:
            if binding.host_pin is not None:
                if torch.cuda.cudart().cudaHostUnregister(binding.host_pin) != 0:
                    _discard_cuda_async_error(self.device)
                binding.host_pin = None
            binding.allocation = None
            binding.signature = None
            binding.owner = None
        self.bindings.clear()
        self.vbar = None
        self.host_cache = None
        self.checkpoint = None
