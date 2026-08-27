"""One warm, standalone LTX 2.3 AV transformer context for T2V."""

from __future__ import annotations

import importlib
import json

import torch

from .av_model import LTXAVModel
from .checkpoint import Ltx23Checkpoint
from .fp8_linear import Ltx23Fp8Linear, Ltx23PlainLinear, _aimdo_modules
from .lora import Ltx23TransformerLora
from .ops import Ltx23Linear, operations


class Ltx23TransformerContext:
    """Own the concrete transformer state for one LTX 2.3 checkpoint identity."""

    def __init__(
        self,
        checkpoint_path: str,
        device_index: int = 0,
        lora_path: str | None = None,
        lora_strength: float = 0.5,
        block_contiguous: bool = False,
    ) -> None:
        self.device_index = device_index
        self.checkpoint = Ltx23Checkpoint(checkpoint_path)
        self.lora = (
            Ltx23TransformerLora(lora_path, lora_strength) if lora_path is not None else None
        )
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

        if not block_contiguous:
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
            module._latentslate_lora = (
                self.lora
                if self.lora is not None and self.lora.has_weight(binding.prefix)
                else None
            )

        aimdo_host_buffer = importlib.import_module("comfy_aimdo.host_buffer")
        if aimdo_host_buffer.lib is None:
            aimdo_host_buffer = importlib.reload(aimdo_host_buffer)
        self._host_cache = aimdo_host_buffer.HostBuffer(
            0,
            64 * 1024 * 1024,
            sum(
                binding.allocation_size if block_contiguous else binding.source_size
                for _, binding in bindings
            ),
        )
        if block_contiguous:
            for _, binding in bindings:
                offset = self._host_cache.size
                self._host_cache.extend(binding.allocation_size, register=False)
                binding.enable_host_cache(self._host_cache, offset, aligned=True)

            block_binding_ids = set()
            registration_ranges = []
            for block in self.model.transformer_blocks:
                block_bindings = [
                    module._latentslate_weight
                    for module in block.modules()
                    if isinstance(module, Ltx23Linear)
                ]
                block_binding_ids.update(id(binding) for binding in block_bindings)
                start = block_bindings[0]._host_cache_offset
                size = sum(binding.allocation_size for binding in block_bindings)
                registration_ranges.append((start, size))
            registration_ranges.extend(
                (binding._host_cache_offset, binding.allocation_size)
                for _, binding in bindings
                if id(binding) not in block_binding_ids
            )
            for offset, size in registration_ranges:
                if (
                    torch.cuda.cudart().cudaHostRegister(
                        self._host_cache.get_raw_address() + offset,
                        size,
                        1,
                    )
                    != 0
                ):
                    raise RuntimeError("unable to register block-contiguous LTX host cache")
        else:
            host_cache_enabled = True
            for _, binding in bindings:
                if not host_cache_enabled:
                    continue
                offset = self._host_cache.size
                self._host_cache.extend(binding.source_size, register=False)
                if (
                    torch.cuda.cudart().cudaHostRegister(
                        self._host_cache.get_raw_address() + offset,
                        binding.source_size,
                        1,
                    )
                    != 0
                ):
                    self._host_cache.truncate(offset, do_unregister=False)
                    host_cache_enabled = False
                    continue
                binding.enable_host_cache(self._host_cache, offset)

        for block in self.model.transformer_blocks:
            block_linears = [
                module for module in block.modules() if isinstance(module, Ltx23Linear)
            ]
            lora_linears = [
                module for module in block_linears if module._latentslate_lora is not None
            ]
            lora_prefixes = [module._latentslate_weight.prefix for module in lora_linears]
            for module in block_linears:
                module._latentslate_grouped = True

            def prepare(
                stream=None,
                lora_buffer=None,
                linears=block_linears,
                stage_linears=lora_linears,
                stage_prefixes=lora_prefixes,
            ):
                host_offset = 0
                for module in linears:
                    module._latentslate_prepared = module._latentslate_weight.materialize(
                        device_index, stream, None, host_offset
                    )
                    host_offset += module._latentslate_weight.source_size
                if stage_linears:
                    staged = self.lora.stage_block(
                        stage_prefixes, lora_buffer, device_index, stream
                    )
                    for module in stage_linears:
                        module._latentslate_lora_prepared = staged[module._latentslate_weight.prefix]

            def release(
                linears=block_linears,
                stage_linears=lora_linears,
            ):
                for module in linears:
                    module._latentslate_prepared = None
                    module._latentslate_weight.unpin(device_index)
                for module in stage_linears:
                    module._latentslate_lora_prepared = None

            block._latentslate_prepare = prepare
            block._latentslate_release = release

        for block in self.model.transformer_blocks:
            block._latentslate_host_buffers = (None, None)

        linear_parameter_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in linear_modules
            for parameter_name in module.state_dict()
        }
        device = torch.device("cuda", device_index)
        if self.lora is None:
            self._lora_stage_buffers = (None, None)
            self._lora_stage_vram_buffers = ()
        else:
            max_lora_stage_size = max(
                self.lora.block_stage_size(
                    [
                        module._latentslate_weight.prefix
                        for module in block.modules()
                        if isinstance(module, Ltx23Linear)
                        and module._latentslate_lora is not None
                    ]
                )
                for block in self.model.transformer_blocks
            )
            _, aimdo_torch = _aimdo_modules(device_index)
            aimdo_vram_buffer = importlib.import_module("comfy_aimdo.vram_buffer")
            self._lora_stage_vram_buffers = tuple(
                aimdo_vram_buffer.VRAMBuffer(16 * 1024 ** 3, device_index) for _ in range(2)
            )
            self._lora_stage_buffers = tuple(
                aimdo_torch.aimdo_to_tensor(buffer.get(max_lora_stage_size), device)
                for buffer in self._lora_stage_vram_buffers
            )
        for block in self.model.transformer_blocks:
            block._latentslate_host_buffers = self._lora_stage_buffers
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
        host_cache = getattr(self, "_host_cache", None)
        if host_cache is not None:
            host_cache.truncate(0)
        self._host_cache = None
        self._lora_stage_buffers = ()
        self._lora_stage_vram_buffers = ()
        self.model = None
        self._vbar = None
        self.lora = None
        self.checkpoint = None
        torch.cuda.empty_cache()
