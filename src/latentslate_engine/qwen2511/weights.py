"""Qwen 2511 dense, mixed-FP8 and official INT8 ConvRot weights.

The proven mapped transfer and VBAR lifecycle follow the existing Krea path.
Kitchen owns quantized math; AIMDO owns mapped transfer and residency.
"""

import importlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from comfy_aimdo import control as aimdo_control
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorWiseINT8Layout
from latentslate_engine.mapped_checkpoint import MappedCheckpoint
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
    """A linear bound to one supported Qwen checkpoint representation."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device="meta", dtype=dtype)
        self.binding = None
        self.updates = ()

    def forward(self, x):
        binding = self.binding
        if binding is None:
            raise RuntimeError("Qwen linear has no loaded checkpoint")
        values = binding.materialize()
        try:
            weight = values["weight"]
            if weight.dtype == torch.float8_e4m3fn:
                weight = QuantizedTensor(
                    weight, "TensorCoreFP8Layout",
                    TensorCoreFP8Layout.Params(
                        scale=values["weight_scale"], orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                    ),
                ).dequantize()
            elif weight.dtype == torch.int8:
                weight = QuantizedTensor(
                    weight, "TensorWiseINT8Layout",
                    TensorWiseINT8Layout.Params(
                        scale=values["weight_scale"], orig_dtype=x.dtype,
                        orig_shape=(self.out_features, self.in_features),
                        convrot=True, convrot_groupsize=256,
                    ),
                )
            bias = values.get("bias")
            fp8 = values["weight"].dtype == torch.float8_e4m3fn
            if self.updates and not (fp8 and binding.resident):
                weight = patch_weight(weight.to(x.dtype), self.updates)
                if fp8 and binding.signature is not None:
                    # First use consumes the BF16 patch; later resident uses consume
                    # its rounded FP8 payload. Host/source bytes remain unpatched.
                    data, scale = requantize_fp8(weight, binding.name)
                    values["weight"].copy_(data)
                    values["weight_scale"].copy_(scale)
            return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))
        finally:
            binding.unpin()


class RMSNorm(nn.RMSNorm):
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class QwenWeight:
    """One linear bundle, faulted and unpinned through AIMDO."""

    def __init__(self, owner, name, module, config):
        self.owner, self.name = owner, name
        self.tensors, self.offsets = {}, {}
        size = 0
        for key in ("weight", "bias", "weight_scale"):
            if f"{name}.{key}" not in owner.checkpoint.tensor_names:
                continue
            value = owner.checkpoint.tensor(f"{name}.{key}")
            self.tensors[key] = value
            self.offsets[key] = size
            size += _aligned(value.nbytes)
        weight = self.tensors["weight"]
        if tuple(weight.shape) != (module.out_features, module.in_features):
            raise ValueError(f"Unsupported Qwen linear shape: {name}")
        if weight.dtype == torch.float8_e4m3fn:
            if (config.get("format") != "float8_e4m3fn"
                or config.get("full_precision_matrix_mult") is not True
                or "weight_scale" not in self.tensors):
                raise ValueError(f"Qwen curated FP8 metadata mismatch: {name}")
        elif weight.dtype == torch.int8:
            if (config.get("format") != "int8_tensorwise"
                or config.get("convrot") is not True
                or config.get("convrot_groupsize") != 256
                or config.get("full_precision_matrix_mult", False)
                or "weight_scale" not in self.tensors):
                raise ValueError(f"Unsupported Qwen INT8 ConvRot metadata: {name}")
        elif weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported Qwen weight representation: {name}")
        self.size = size
        self.allocation = self.signature = None
        self.cached = False
        self.host_offset = 0
        self.host_pin = None

    def materialize(self):
        owner = self.owner
        model_vbar, aimdo_torch = _aimdo_modules(owner.device_index)
        signature = model_vbar.vbar_fault(self.allocation)
        resident = signature is not None and model_vbar.vbar_signature_compare(signature, self.signature)
        self.resident = resident
        self.signature = signature
        self._copy_stream = None
        if resident:
            destination = aimdo_torch.aimdo_to_tensor(self.allocation, owner.device)
        else:
            stream = owner.copy_streams[owner.copy_index % len(owner.copy_streams)]
            owner.copy_index += 1
            self._copy_stream = stream
            with torch.cuda.stream(stream):
                allocation = self.allocation if signature is not None else owner.copy_buffers[stream].get(self.size)
                destination = aimdo_torch.aimdo_to_tensor(allocation, owner.device)
                if self.cached:
                    host = aimdo_torch.hostbuf_to_tensor(owner.host_cache)
                    destination.copy_(host[self.host_offset:self.host_offset + self.size], non_blocking=True)
                else:
                    for key in self.tensors:
                        offset = self.offsets[key]
                        owner.checkpoint.copy_tensor_to_device(f"{self.name}.{key}", destination, offset, owner.device_index, stream=stream, host_buffer=owner.host_cache, host_offset=self.host_offset + offset)
                    self.cached = True
                    pointer = owner.host_cache.get_raw_address() + self.host_offset
                    if torch.cuda.cudart().cudaHostRegister(pointer, self.size, 1) == 0:
                        self.host_pin = pointer
                    else:
                        _discard_cuda_async_error(owner.device)
            current = torch.cuda.current_stream(owner.device)
            current.wait_stream(stream)
        return {key: destination[self.offsets[key]:self.offsets[key] + value.nbytes].view(value.dtype).view(value.shape) for key, value in self.tensors.items()}

    def unpin(self):
        model_vbar, _ = _aimdo_modules(self.owner.device_index)
        if self.signature is not None:
            model_vbar.vbar_unpin(self.allocation)
        if self._copy_stream is not None:
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.owner.device))

class QwenWeights:
    """Own one transformer's mapped source, host cache, and virtual VRAM."""

    def __init__(self, path: Path, model: nn.Module, device: torch.device, adapters=()):
        self.device = device
        self.device_index = device.index or 0
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
                binding = QwenWeight(self, name, module, config.get(name, {}))
                self.bindings.append(binding)
                self.modules.append(module)
                linear_names.add(name)
                module.binding = binding
        if adapters and any(b.tensors["weight"].dtype == torch.int8 for b in self.bindings):
            raise ValueError("Qwen adapter execution is not certified for INT8 ConvRot")
        updates = load_updates(
            adapters, {name: model.get_submodule(name) for name in linear_names}, device
        )
        for name in linear_names:
            model.get_submodule(name).updates = updates.get(name, ())
        model_vbar, _ = _aimdo_modules(self.device_index)
        buffer_module = importlib.import_module("comfy_aimdo.vram_buffer")
        if buffer_module.lib is None:
            buffer_module = importlib.reload(buffer_module)
        buffer_size = _aligned(max(b.size for b in self.bindings), 64 * 1024 * 1024)
        self.copy_buffers = {
            stream: buffer_module.VRAMBuffer(buffer_size, self.device_index)
            for stream in self.copy_streams
        }
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
                raise ValueError(f"Unsupported Qwen parameter shape: {name}")
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
            binding.owner = None
        self.bindings.clear()
        self.copy_buffers.clear()
        self.copy_streams.clear()
        self.vbar = None
        if self.host_cache is not None and self.host_cache.size:
            self.host_cache.truncate(0, do_unregister=False)
        self.host_cache = None
        self.checkpoint = None
