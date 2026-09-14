"""Krea-family mapped checkpoints; AIMDO owns the virtual device allocation.

The narrow mapping/transfer code follows the existing measured Klein path and
pinned Comfy loader. This family owns its quantization and layer lifetimes.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from comfy_kitchen.tensor import (
    AsymW4A8Int8Layout,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorCoreMXFP8Layout,
    TensorWiseINT8Layout,
)
from torch import nn
from torch.nn import functional as F
from comfy_aimdo import control as aimdo_control

from latentslate_engine.mapped_checkpoint import MappedCheckpoint as KreaCheckpoint

from .adapters import load_updates, patch_weight, requantize_fp8

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


class Linear(nn.Linear):
    """A Krea linear with its checkpoint-owned mixed-precision dispatch."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device="meta", dtype=dtype)
        self.binding = None
        self.updates = ()

    def forward(self, x):
        if self.binding is None:
            raise RuntimeError("Krea linear has no loaded checkpoint")
        binding = self.binding
        stream = binding.transfer_stream
        if stream is not None:
            current = torch.cuda.current_stream(x.device)
            with torch.cuda.stream(stream):
                values = binding.materialize(stream)
            current.wait_stream(stream)
        else:
            values = binding.materialize()
        try:
            weight = values["weight"]
            bias = values.get("bias")
            if bias is not None:
                bias = bias.to(x.dtype)
            if binding.format in ("nvfp4", "mxfp8"):
                if binding.format == "nvfp4":
                    layout = "TensorCoreNVFP4Layout"
                    params = TensorCoreNVFP4Layout.Params(
                        scale=values["weight_scale_2"],
                        block_scale=values["weight_scale"].view(torch.float8_e4m3fn),
                        orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                    )
                else:
                    layout = "TensorCoreMXFP8Layout"
                    params = TensorCoreMXFP8Layout.Params(
                        scale=values["weight_scale"].view(torch.float8_e8m0fnu),
                        orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                    )
                weight = QuantizedTensor(weight, layout, params)
                if binding.full_precision:
                    return F.linear(x, weight.dequantize(), bias)
                original_shape = x.shape
                quantized = QuantizedTensor.from_float(
                    x.reshape(-1, original_shape[-1]), layout,
                    scale=values.get("input_scale"),
                )
                return F.linear(quantized, weight, bias).reshape(
                    *original_shape[:-1], self.out_features
                )
            if binding.format == "asym_w4a8_int8":
                weight = QuantizedTensor(
                    weight, "AsymW4A8Int8Layout",
                    AsymW4A8Int8Layout.Params(
                        scale=values["weight_s_rel"].view(torch.float8_e4m3fn),
                        s_channel=values["weight_s_channel"],
                        codebook=values.get("weight_codebook"),
                        orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                        group_size=16, convrot_groupsize=256,
                    ),
                )
                if binding.full_precision:
                    weight = weight.dequantize()
                return F.linear(x, weight, bias)
            if weight.dtype == torch.int8:
                weight = QuantizedTensor(
                    weight, "TensorWiseINT8Layout",
                    TensorWiseINT8Layout.Params(
                        scale=values["weight_scale"], orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                        convrot=True, convrot_groupsize=256,
                    ),
                )
                if binding.full_precision:
                    weight = weight.dequantize()
                return F.linear(x, weight, bias)
            if weight.dtype != torch.float8_e4m3fn:
                if self.updates:
                    # The reference eagerly patches this tiny projection in FP16.
                    patch_dtype = (
                        torch.float16 if binding.name == "txtfusion.projector" else x.dtype
                    )
                    weight = patch_weight(weight.to(patch_dtype), self.updates)
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
            if self.updates:
                patched = patch_weight(weight.dequantize().to(x.dtype), self.updates)
                if binding.full_precision:
                    return F.linear(x, patched, bias)
                weight = requantize_fp8(patched, binding.name)
                binding.cache_patched_fp8(weight, values)
                self.updates = ()
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
            if stream is not None:
                stream.wait_stream(current)


class KreaWeight:
    """One linear's raw weight/scales/bias faulted as a single bundle."""

    def __init__(self, owner, name, module, config):
        self.owner = owner
        self.name = name
        self.format = config.get("format")
        self.full_precision = config.get("full_precision_matrix_mult", False)
        self.tensors = {}
        self.offsets = {}
        size = 0
        for key in (
            "weight", "bias", "weight_scale", "weight_scale_2", "input_scale",
            "weight_s_rel", "weight_s_channel", "weight_codebook",
        ):
            if f"{name}.{key}" not in owner.checkpoint.tensor_names:
                continue
            value = owner.checkpoint.tensor(f"{name}.{key}")
            self.tensors[key] = value
            self.offsets[key] = size
            size += _aligned(value.nbytes)
        weight = self.tensors["weight"]
        expected_shape = (module.out_features, module.in_features)
        if self.format == "nvfp4":
            expected_shape = TensorCoreNVFP4Layout.get_storage_shape(expected_shape)
        elif self.format == "mxfp8":
            expected_shape = TensorCoreMXFP8Layout.get_storage_shape(expected_shape)
        elif self.format == "asym_w4a8_int8":
            expected_shape = (module.out_features, module.in_features // 2)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"Unsupported Krea linear shape: {name}")
        if self.format == "nvfp4":
            if (weight.dtype != torch.uint8
                or "weight_scale" not in self.tensors
                or "weight_scale_2" not in self.tensors):
                raise ValueError(f"Missing Krea NVFP4 metadata: {name}")
        elif self.format == "mxfp8":
            if weight.dtype != torch.float8_e4m3fn or "weight_scale" not in self.tensors:
                raise ValueError(f"Missing Krea MXFP8 metadata: {name}")
        elif weight.dtype == torch.float8_e4m3fn:
            if (
                config.get("format") != "float8_e4m3fn"
                or "weight_scale" not in self.tensors
            ):
                raise ValueError(f"Missing Krea FP8 metadata: {name}")
        elif self.format == "asym_w4a8_int8":
            if (
                weight.dtype != torch.int8
                or config.get("group_size") != 16
                or config.get("convrot_groupsize") != 256
                or tuple(config.get("orig_shape", ())) != (module.out_features, module.in_features)
                or "weight_s_rel" not in self.tensors
                or "weight_s_channel" not in self.tensors
            ):
                raise ValueError(f"Unsupported Krea W4A8 metadata: {name}")
        elif weight.dtype == torch.int8:
            scale = self.tensors.get("weight_scale")
            if (
                self.format != "int8_tensorwise"
                or config.get("convrot") is not True
                or config.get("convrot_groupsize") != 256
                or module.in_features % 256
                or scale is None
                or scale.dtype != torch.float32
                or tuple(scale.shape) != (module.out_features, 1)
            ):
                raise ValueError(f"Unsupported Krea INT8 ConvRot metadata: {name}")
        elif weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported Krea weight representation: {name}")
        self.size = size
        self.allocation = None
        self.signature = None
        self.cached = False
        self.host_offset = 0
        self.host_pin = None
        self.transfer_stream = None
        self.transfer_buffer = None

    def materialize(self, stream=None):
        owner = self.owner
        model_vbar, aimdo_torch = _aimdo_modules(owner.device_index)
        signature = model_vbar.vbar_fault(self.allocation)
        if signature is not None:
            destination = aimdo_torch.aimdo_to_tensor(self.allocation, owner.device)
        elif stream is not None and self.transfer_buffer is not None:
            destination = aimdo_torch.aimdo_to_tensor(
                self.transfer_buffer.get(self.size), owner.device
            )
        else:
            destination = torch.empty(self.size, dtype=torch.uint8, device=owner.device)
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
                        stream=stream,
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

    def cache_patched_fp8(self, patched, values):
        """Commit this identity's FP8 result to the existing reload storage."""
        _, aimdo_torch = _aimdo_modules(self.owner.device_index)
        host = aimdo_torch.hostbuf_to_tensor(self.owner.host_cache)
        for key, value in (("weight", patched._qdata), ("weight_scale", patched._params.scale)):
            offset = self.host_offset + self.offsets[key]
            host[offset : offset + value.nbytes].view(value.dtype).view(
                value.shape
            ).copy_(value)
            values[key].copy_(value)

    def unpin(self):
        # Bounded transfer storage does not acquire a VBAR residency pin.
        if self.signature is None and self.transfer_buffer is not None:
            return
        model_vbar, _ = _aimdo_modules(self.owner.device_index)
        model_vbar.vbar_unpin(self.allocation)


class KreaWeights:
    """Own one transformer's mapped source, host cache, and virtual VRAM."""

    def __init__(self, path: Path, model: nn.Module, device: torch.device, adapters=()):
        self.device = device
        self.device_index = device.index or 0
        self.checkpoint = KreaCheckpoint(path)
        metadata = self.checkpoint._header.get("__metadata__", {})
        config = json.loads(metadata.get("_quantization_metadata", "{}")).get(
            "layers", {}
        )
        for name in self.checkpoint.tensor_names:
            if name.endswith(".comfy_quant"):
                config[name.removesuffix(".comfy_quant")] = (
                    self.checkpoint.quantization_config(name)
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
        if adapters and any(
            b.format in ("int8_tensorwise", "nvfp4", "mxfp8", "asym_w4a8_int8")
            for b in self.bindings
        ):
            raise ValueError("Krea adapters require a validated BF16 or FP8 checkpoint")
        updates = load_updates(
            adapters, {name: model.get_submodule(name) for name in linear_names}, device
        )
        for name, values in updates.items():
            model.get_submodule(name).updates = values
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
        self.transfer_streams = ()
        self.transfer_buffers = ()
        if any(binding.format == "float8_e4m3fn" for binding in self.bindings):
            buffer_module = importlib.import_module("comfy_aimdo.vram_buffer")
            if buffer_module.lib is None:
                buffer_module = importlib.reload(buffer_module)
            maximum_size = max(binding.size for binding in self.bindings)
            self.transfer_buffers = tuple(
                buffer_module.VRAMBuffer(maximum_size, self.device_index) for _ in range(2)
            )
            self.transfer_streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
            for stream in self.transfer_streams:
                stream.wait_stream(torch.cuda.current_stream(device))
            for index, binding in enumerate(self.bindings):
                binding.transfer_stream = self.transfer_streams[index % 2]
                binding.transfer_buffer = self.transfer_buffers[index % 2]
        cast_parameters = not config or any(
            binding.format in ("int8_tensorwise", "asym_w4a8_int8")
            for binding in self.bindings
        )
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
                nn.Parameter(
                    value.to(
                        device=device,
                        dtype=parameter.dtype if cast_parameters else value.dtype,
                    ),
                    requires_grad=False,
                ),
            )

    def close(self):
        """Release the model bindings before releasing their storage owners."""
        torch.cuda.synchronize(self.device)
        for module in self.modules:
            module.binding = None
            module.updates = ()
        self.modules.clear()
        for binding in self.bindings:
            if binding.host_pin is not None:
                if torch.cuda.cudart().cudaHostUnregister(binding.host_pin) != 0:
                    _discard_cuda_async_error(self.device)
                binding.host_pin = None
            binding.allocation = None
            binding.signature = None
            binding.transfer_stream = None
            binding.transfer_buffer = None
            binding.owner = None
        self.bindings.clear()
        self.transfer_streams = ()
        self.transfer_buffers = ()
        self.vbar = None
        if self.host_cache is not None and self.host_cache.size:
            self.host_cache.truncate(0, do_unregister=False)
        self.host_cache = None
        self.checkpoint = None
