"""H3 mapped linear weights and AIMDO residency.

The transfer lifetime follows the measured Ideogram/Krea path. Quantized math
follows ComfyUI 1a14b82e comfy/ops.py (GPL-3.0); Kitchen owns each layout.
"""

import importlib
import json
from pathlib import Path

import comfy_kitchen as ck
import torch
from comfy_aimdo import control as aimdo_control
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from torch import nn
from torch.nn import functional as F

from latentslate_engine.mapped_checkpoint import MappedCheckpoint


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
    """A linear bound to one supported H3 checkpoint representation."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device="meta", dtype=dtype)
        self.binding = None

    def forward(self, x):
        binding = self.binding
        if binding is None:
            raise RuntimeError("H3 linear has no loaded checkpoint")
        values = binding.materialize()
        try:
            if "pre_quant_scale" in values:
                x = x * values["pre_quant_scale"].to(x.dtype)
            weight = values["weight"]
            bias = values.get("bias")
            if bias is not None:
                bias = bias.to(binding.owner.compute_dtype).to(x.dtype)
            if binding.format in ("nvfp4", "float8_e4m3fn"):
                layout = (
                    TensorCoreNVFP4Layout
                    if binding.format == "nvfp4"
                    else TensorCoreFP8Layout
                )
                scales = {"scale": values["weight_scale"]}
                if binding.format == "nvfp4":
                    scales = {
                        "scale": values["weight_scale_2"],
                        "block_scale": values["weight_scale"].view(torch.float8_e4m3fn),
                    }
                weight = QuantizedTensor(
                    weight,
                    layout.__name__,
                    layout.Params(
                        **scales,
                        orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                    ),
                )
                if binding.full_precision:
                    return F.linear(x, weight.dequantize(), bias)
                shape = x.shape
                quantized = QuantizedTensor.from_float(
                    x.reshape(-1, shape[-1]),
                    layout.__name__,
                    scale=values.get(
                        "input_scale",
                        1.0 if binding.format == "float8_e4m3fn" else None,
                    ),
                )
                return F.linear(quantized, weight, bias).reshape(
                    *shape[:-1], self.out_features
                )
            if weight.dtype == torch.int8:
                weight = QuantizedTensor(
                    weight,
                    "TensorWiseINT8Layout",
                    TensorWiseINT8Layout.Params(
                        scale=values["weight_scale"],
                        orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                        convrot=binding.convrot,
                        convrot_groupsize=256,
                    ),
                )
            elif weight.is_floating_point():
                # MixedPrecisionOps stores unquantized islands at model dtype,
                # even when their forward runs in FP32 (patch and adaLN heads).
                weight = weight.to(binding.owner.compute_dtype)
            return F.linear(
                x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
            )
        finally:
            binding.unpin()


class RMSNorm(nn.RMSNorm):
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


def swiglu_linear(linear, x):
    """Match H3's SwiGLU fused into the INT8 input quantizer."""
    binding = getattr(linear, "binding", None)
    if binding is not None and binding.format == "int8_tensorwise":
        values = binding.materialize()
        try:
            bias = values.get("bias")
            return ck.int8_linear(
                x,
                values["weight"],
                values["weight_scale"],
                None if bias is None else bias.to(x.dtype),
                x.dtype,
                convrot=binding.convrot,
                convrot_groupsize=256,
                input_act="swiglu",
            )
        finally:
            binding.unpin()
    gate, up = x.chunk(2, dim=-1)
    return linear(F.silu(gate).mul_(up))


class H3Weight:
    """One linear bundle, faulted and unpinned through AIMDO."""

    def __init__(self, owner, name, module, config):
        self.owner, self.name = owner, name
        self.format = config.get("format")
        self.full_precision = config.get("full_precision_matrix_mult", False)
        self.convrot = config.get("convrot", False)
        self.tensors, self.offsets = {}, {}
        size = 0
        for key in (
            "weight",
            "bias",
            "weight_scale",
            "weight_scale_2",
            "input_scale",
            "pre_quant_scale",
        ):
            if f"{name}.{key}" not in owner.checkpoint.tensor_names:
                continue
            value = owner.checkpoint.tensor(f"{name}.{key}")
            self.tensors[key] = value
            self.offsets[key] = size
            size += _aligned(value.nbytes)
        weight = self.tensors["weight"]
        shape = (module.out_features, module.in_features)
        if self.format == "nvfp4":
            shape = TensorCoreNVFP4Layout.get_storage_shape(shape)
        if tuple(weight.shape) != shape:
            raise ValueError(f"Unsupported H3 linear shape: {name}")
        if self.format == "nvfp4":
            if (
                weight.dtype != torch.uint8
                or not {"weight_scale", "weight_scale_2"} <= self.tensors.keys()
            ):
                raise ValueError(f"Missing H3 NVFP4 metadata: {name}")
        elif self.format == "float8_e4m3fn":
            if (
                weight.dtype != torch.float8_e4m3fn
                or "weight_scale" not in self.tensors
            ):
                raise ValueError(f"Missing H3 FP8 metadata: {name}")
        elif weight.dtype == torch.int8:
            if (
                config.get("format") != "int8_tensorwise"
                or (self.convrot and config.get("convrot_groupsize", 256) != 256)
                or config.get("full_precision_matrix_mult", False)
                or "weight_scale" not in self.tensors
            ):
                raise ValueError(f"Unsupported H3 INT8 ConvRot metadata: {name}")
        elif weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported H3 weight representation: {name}")
        self.size = size
        self.allocation = self.signature = None
        self.cached = False
        self.host_offset = 0
        self.host_pin = None

    def materialize(self):
        owner = self.owner
        model_vbar, aimdo_torch = _aimdo_modules(owner.device_index)
        signature = model_vbar.vbar_fault(self.allocation)
        resident = signature is not None and model_vbar.vbar_signature_compare(
            signature, self.signature
        )
        self.signature = signature
        self._copy_stream = None
        if resident:
            destination = aimdo_torch.aimdo_to_tensor(self.allocation, owner.device)
        else:
            stream = owner.copy_streams[owner.copy_index % len(owner.copy_streams)]
            owner.copy_index += 1
            self._copy_stream = stream
            with torch.cuda.stream(stream):
                allocation = (
                    self.allocation
                    if signature is not None
                    else owner.copy_buffers[stream].get(self.size)
                )
                destination = aimdo_torch.aimdo_to_tensor(allocation, owner.device)
                if self.cached:
                    host = aimdo_torch.hostbuf_to_tensor(owner.host_cache)
                    destination.copy_(
                        host[self.host_offset : self.host_offset + self.size],
                        non_blocking=True,
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
            current = torch.cuda.current_stream(owner.device)
            current.wait_stream(stream)
        return {
            key: destination[self.offsets[key] : self.offsets[key] + value.nbytes]
            .view(value.dtype)
            .view(value.shape)
            for key, value in self.tensors.items()
        }

    def unpin(self):
        model_vbar, _ = _aimdo_modules(self.owner.device_index)
        if self.signature is not None:
            model_vbar.vbar_unpin(self.allocation)
        if self._copy_stream is not None:
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.owner.device))


class H3Weights:
    """Own one transformer's mapped source, host cache, and virtual VRAM."""

    def __init__(self, path: Path, model: nn.Module, device: torch.device):
        self.device = device
        self.device_index = device.index or 0
        self.compute_dtype = getattr(model, "dtype", torch.bfloat16)
        self.checkpoint = MappedCheckpoint(path)
        metadata = self.checkpoint._header.get("__metadata__", {})
        config = json.loads(metadata.get("_quantization_metadata", "{}")).get(
            "layers", {}
        )
        for name in self.checkpoint.tensor_names:
            if name.endswith(".comfy_quant"):
                config[name.removesuffix(".comfy_quant")] = (
                    self.checkpoint.quantization_config(name)
                )
        self.copy_streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        self.copy_index = 0
        self.bindings = []
        self.modules = []
        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                binding = H3Weight(self, name, module, config.get(name, {}))
                self.bindings.append(binding)
                self.modules.append(module)
                linear_names.add(name)
                module.binding = binding
        model_vbar, _ = _aimdo_modules(self.device_index)
        # MappedCheckpoint initializes the native library before this owner.
        # Match Comfy main.py's callback verbosity after initialization.
        aimdo_control.set_log_info()
        buffer_module = importlib.import_module("comfy_aimdo.vram_buffer")
        if buffer_module.lib is None:
            buffer_module = importlib.reload(buffer_module)
        self.buffer_size = _aligned(
            max(b.size for b in self.bindings), 64 * 1024 * 1024
        )
        self.copy_buffers = {}
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
                raise ValueError(f"Unsupported H3 parameter shape: {name}")
            setattr(
                model.get_submodule(parent),
                key,
                nn.Parameter(
                    value.to(
                        device=device,
                        dtype=value.dtype,
                    ),
                    requires_grad=False,
                ),
            )

        for name, buffer in list(model.named_buffers()):
            if name in self.checkpoint.tensor_names:
                value = self.checkpoint.tensor(name)
                if value.shape != buffer.shape:
                    raise ValueError(f"Unsupported H3 buffer shape: {name}")
                parent, _, key = name.rpartition(".")
                setattr(model.get_submodule(parent), key, value.to(device))

    def activate(self):
        """Match Comfy's per-sample dynamic model activation."""
        self.vbar.prioritize()
        buffer_module = importlib.import_module("comfy_aimdo.vram_buffer")
        self.copy_buffers = {
            stream: buffer_module.VRAMBuffer(self.buffer_size, self.device_index)
            for stream in self.copy_streams
        }

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
        self.copy_buffers.clear()
        self.copy_streams.clear()
        self.vbar = None
        if self.host_cache is not None and self.host_cache.size:
            self.host_cache.truncate(0, do_unregister=False)
        self.host_cache = None
        self.checkpoint = None


class Operations:
    Linear = Linear
    RMSNorm = RMSNorm
