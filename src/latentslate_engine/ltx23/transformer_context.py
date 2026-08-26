"""One warm, standalone LTX 2.3 AV transformer context for T2V."""

from __future__ import annotations

import json
import importlib

import torch

from .av_model import LTXAVModel
from .checkpoint import Ltx23Checkpoint
from .fp8_linear import Ltx23Fp8Linear, Ltx23PlainLinear, _aimdo_modules
from .ops import Ltx23Linear, operations


class Ltx23TransformerContext:
    """Own the concrete transformer state for one LTX 2.3 checkpoint identity."""

    def __init__(self, checkpoint_path: str, device_index: int = 0) -> None:
        self.device_index = device_index
        self.checkpoint = Ltx23Checkpoint(checkpoint_path)
        config = json.loads(self.checkpoint.metadata["config"])["transformer"]
        self.model = LTXAVModel(
            dtype=torch.bfloat16,
            device="meta",
            operations=operations,
            **config,
        )

        linear_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, Ltx23Linear)
        ]
        bindings = []
        for name, module in linear_modules:
            prefix = f"model.diffusion_model.{name}"
            binding = (
                Ltx23Fp8Linear(self.checkpoint, prefix)
                if f"{prefix}.weight_scale" in self.checkpoint.tensor_names
                else Ltx23PlainLinear(self.checkpoint, prefix)
            )
            bindings.append((module, binding))

        bindings.sort(
            key=lambda item: (
                item[1].offload_size >= 64 * 1024,
                -item[1].offload_size,
                item[1].source_size,
                item[1].prefix,
            )
        )

        model_vbar, _ = _aimdo_modules(device_index)
        source_model_bytes = sum(
            self.checkpoint.tensor(name).nbytes
            for name in self.checkpoint.tensor_names
            if name.startswith("model.diffusion_model.")
        )
        vbar_bytes = 10 * source_model_bytes
        self._vbar = model_vbar.ModelVBAR(vbar_bytes, device_index)
        for module, binding in bindings:
            binding.allocate(self._vbar)
            module._latentslate_weight = binding
            module._latentslate_device_index = device_index

        block_host_sizes = []
        for block in self.model.transformer_blocks:
            block_linears = [
                module for module in block.modules() if isinstance(module, Ltx23Linear)
            ]
            block_host_sizes.append(
                sum(module._latentslate_weight.source_size for module in block_linears)
            )
            for module in block_linears:
                module._latentslate_grouped = True

            def prepare(stream=None, host_buffer=None, linears=block_linears):
                host_offset = 0
                for module in linears:
                    module._latentslate_prepared = module._latentslate_weight.materialize(
                        device_index, stream, host_buffer, host_offset
                    )
                    host_offset += module._latentslate_weight.source_size

            def release(linears=block_linears):
                for module in linears:
                    module._latentslate_prepared = None
                    module._latentslate_weight.unpin(device_index)

            block._latentslate_prepare = prepare
            block._latentslate_release = release

        aimdo_host_buffer = importlib.import_module("comfy_aimdo.host_buffer")
        if aimdo_host_buffer.lib is None:
            aimdo_host_buffer = importlib.reload(aimdo_host_buffer)
        self._host_buffers = []
        host_buffer_size = max(block_host_sizes)
        for _ in range(2):
            host_buffer = aimdo_host_buffer.HostBuffer(0, 64 * 1024 * 1024, host_buffer_size)
            host_buffer.extend(host_buffer_size, register=False)
            if torch.cuda.cudart().cudaHostRegister(host_buffer.get_raw_address(), host_buffer_size, 1) != 0:
                raise RuntimeError("unable to register LTX transformer host buffer")
            self._host_buffers.append(host_buffer)
        self._host_buffers = tuple(self._host_buffers)
        for block in self.model.transformer_blocks:
            block._latentslate_host_buffers = self._host_buffers

        linear_parameter_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in linear_modules
            for parameter_name in module.state_dict()
        }
        device = torch.device("cuda", device_index)
        for name, parameter in list(self.model.named_parameters()):
            if name in linear_parameter_names:
                continue
            parent, attribute = self._resolve_parent(name)
            source = self.checkpoint.tensor(f"model.diffusion_model.{name}")
            setattr(
                parent,
                attribute,
                torch.nn.Parameter(source.to(device=device, dtype=torch.bfloat16), requires_grad=False),
            )

        self.model.eval()

    def _resolve_parent(self, parameter_name: str):
        parent = self.model
        parts = parameter_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def close(self) -> None:
        """Drop this exact model context and all of its warm state."""
        self._host_buffers = ()
        self.model = None
        self._vbar = None
        self.checkpoint = None
        torch.cuda.empty_cache()
