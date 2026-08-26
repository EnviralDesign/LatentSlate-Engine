"""One warm, standalone LTX 2.3 AV transformer context for T2V."""

from __future__ import annotations

import json

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

        model_vbar, _ = _aimdo_modules(device_index)
        vbar_bytes = sum(binding.allocation_size + 511 for _, binding in bindings)
        self._vbar = model_vbar.ModelVBAR(vbar_bytes, device_index)
        for module, binding in bindings:
            binding.allocate(self._vbar)
            module._latentslate_weight = binding
            module._latentslate_device_index = device_index

        for block in self.model.transformer_blocks:
            block_linears = [
                module for module in block.modules() if isinstance(module, Ltx23Linear)
            ]
            for module in block_linears:
                module._latentslate_grouped = True

            def prepare(linears=block_linears):
                for module in linears:
                    module._latentslate_prepared = module._latentslate_weight.materialize(device_index)

            def release(linears=block_linears):
                for module in linears:
                    module._latentslate_prepared = None
                    module._latentslate_weight.unpin(device_index)

            block._latentslate_prepare = prepare
            block._latentslate_release = release

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
        self.model = None
        self._vbar = None
        self.checkpoint = None
        torch.cuda.empty_cache()
