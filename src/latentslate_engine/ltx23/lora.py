"""Canonical LTX 2.3 transformer LoRA application for the T2V fixture."""

from __future__ import annotations

import torch

from .checkpoint import Ltx23Checkpoint


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

    def apply(self, prefix: str, weight: torch.Tensor) -> torch.Tensor:
        """Match Comfy's regular LoRA branch for this fixture's A/B pairs."""
        if not self.has_weight(prefix):
            return weight

        prefix = self._lora_prefix(prefix)
        down = self.checkpoint.tensor(f"{prefix}.lora_A.weight").to(
            device=weight.device, dtype=weight.dtype
        )
        up = self.checkpoint.tensor(f"{prefix}.lora_B.weight").to(
            device=weight.device, dtype=weight.dtype
        )
        alpha_name = f"{prefix}.alpha"
        # Pinned Comfy's regular-LoRA adapter divides by rank only when an
        # explicit per-layer alpha exists.  The canonical dynamic-rank LoRA
        # deliberately omits alpha for some layers, where its fallback is 1.
        if alpha_name in self._names:
            scale = self.strength * self.checkpoint.tensor(alpha_name).item() / down.shape[0]
        else:
            scale = self.strength
        return weight + (scale * torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))).reshape(
            weight.shape
        ).to(weight.dtype)
