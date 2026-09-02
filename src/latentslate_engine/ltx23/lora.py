"""Canonical LTX 2.3 transformer LoRA application for the T2V fixture."""

from __future__ import annotations

import torch

from .checkpoint import Ltx23Checkpoint


def _aligned(offset: int, alignment: int = 1024) -> int:
    return (offset + alignment - 1) & -alignment


class Ltx23TransformerLora:
    """Mapped canonical LoRA, applied only while a matching layer is materialized."""

    def __init__(self, checkpoint_path: str, strength: float) -> None:
        self.checkpoint = Ltx23Checkpoint(checkpoint_path)
        self.strength = strength
        self._names = frozenset(self.checkpoint.tensor_names)

    @staticmethod
    def _lora_prefix(prefix: str) -> str:
        # Comfy maps this fixture's diffusion_model.* LoRA names onto the
        # checkpoint's model.diffusion_model.* state-dict names.
        return prefix.removeprefix("model.")

    def has_weight(self, prefix: str) -> bool:
        return f"{self._lora_prefix(prefix)}.lora_A.weight" in self._names

    def block_stage_size(self, prefixes: list[str]) -> int:
        offset = 0
        for prefix in prefixes:
            prefix = self._lora_prefix(prefix)
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                offset = _aligned(offset)
                offset += self.checkpoint.tensor(f"{prefix}{suffix}").nbytes
        return offset

    def stage_block(
        self,
        prefixes: list[str],
        destination: torch.Tensor,
        device_index: int,
        stream: torch.cuda.Stream,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        staged = {}
        offset = 0
        for prefix in prefixes:
            lora_prefix = self._lora_prefix(prefix)
            tensors = []
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                source = self.checkpoint.tensor(f"{lora_prefix}{suffix}")
                offset = _aligned(offset)
                self.checkpoint.copy_tensor_to_device(
                    f"{lora_prefix}{suffix}", destination, offset, device_index, stream
                )
                tensors.append(
                    destination[offset : offset + source.nbytes]
                    .view(source.dtype)
                    .view(source.shape)
                )
                offset += source.nbytes
            staged[prefix] = tuple(tensors)
        return staged

    def apply(
        self,
        prefix: str,
        weight: torch.Tensor,
        staged: tuple[torch.Tensor, torch.Tensor] | None = None,
        disposable_weight: bool = False,
    ) -> torch.Tensor:
        """Match Comfy's regular LoRA branch for this fixture's A/B pairs."""
        if not self.has_weight(prefix):
            return weight

        prefix = self._lora_prefix(prefix)
        if staged is None:
            down = self.checkpoint.tensor(f"{prefix}.lora_A.weight").to(
                device=weight.device, dtype=weight.dtype
            )
            up = self.checkpoint.tensor(f"{prefix}.lora_B.weight").to(
                device=weight.device, dtype=weight.dtype
            )
        else:
            down, up = (tensor.to(dtype=weight.dtype) for tensor in staged)
        alpha_name = f"{prefix}.alpha"
        # Pinned Comfy's regular-LoRA adapter divides by rank only when an
        # explicit per-layer alpha exists.  The canonical dynamic-rank LoRA
        # deliberately omits alpha for some layers, where its fallback is 1.
        if alpha_name in self._names:
            scale = self.strength * self.checkpoint.tensor(alpha_name).item() / down.shape[0]
        else:
            scale = self.strength
        difference = (
            scale
            * torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))
        ).reshape(weight.shape).to(weight.dtype)
        return weight.add_(difference) if disposable_weight else weight + difference


class Ltx23TransformerLoras:
    """Ordered LTX transformer LoRAs, staged together for one model block."""

    def __init__(self, lora_paths: tuple[tuple[str, float], ...]) -> None:
        if len(lora_paths) < 2:
            raise ValueError("LTX multi-LoRA staging requires at least two LoRAs")
        self.loras = tuple(
            Ltx23TransformerLora(path, strength) for path, strength in lora_paths
        )

    def has_weight(self, prefix: str) -> bool:
        return any(lora.has_weight(prefix) for lora in self.loras)

    def block_stage_size(self, prefixes: list[str]) -> int:
        return sum(
            lora.block_stage_size(
                [prefix for prefix in prefixes if lora.has_weight(prefix)]
            )
            for lora in self.loras
        )

    def stage_block(
        self,
        prefixes: list[str],
        destination: torch.Tensor,
        device_index: int,
        stream: torch.cuda.Stream,
    ) -> dict[str, tuple[tuple[Ltx23TransformerLora, tuple[torch.Tensor, torch.Tensor]], ...]]:
        staged: dict[
            str, list[tuple[Ltx23TransformerLora, tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        offset = 0
        for lora in self.loras:
            matching_prefixes = [prefix for prefix in prefixes if lora.has_weight(prefix)]
            if not matching_prefixes:
                continue
            size = lora.block_stage_size(matching_prefixes)
            lora_staged = lora.stage_block(
                matching_prefixes,
                destination.narrow(0, offset, size),
                device_index,
                stream,
            )
            for prefix, tensors in lora_staged.items():
                staged.setdefault(prefix, []).append((lora, tensors))
            offset += size
        return {prefix: tuple(entries) for prefix, entries in staged.items()}

    def apply(
        self,
        prefix: str,
        weight: torch.Tensor,
        staged: tuple[
            tuple[Ltx23TransformerLora, tuple[torch.Tensor, torch.Tensor]], ...
        ]
        | None = None,
        disposable_weight: bool = False,
    ) -> torch.Tensor:
        entries = (
            staged
            if staged is not None
            else tuple((lora, None) for lora in self.loras if lora.has_weight(prefix))
        )
        for lora, tensors in entries:
            weight = lora.apply(prefix, weight, tensors, disposable_weight)
        return weight
