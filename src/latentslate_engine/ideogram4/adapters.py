"""Ordinary LoRA and full LoKR mapping for the Ideogram v4 transformer.

LoRA follows Comfy's A/B factor product. Full LoKR follows Comfy's
`LoKrAdapter.calculate_weight` for stored w1/w2 pairs: Kronecker product
scaled by strength only. Alpha is consumed and ignored unless factors are
rebuilt from decomposed tensors, which this loader does not accept.
"""

from dataclasses import dataclass

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class LoKrUpdate:
    """Direct Kronecker factors, retained without expanding a dense weight."""

    w1: torch.Tensor
    w2: torch.Tensor
    strength: float


def _linear(modules, name):
    module = modules.get(name)
    if module is None or not hasattr(module, "in_features") or not hasattr(module, "out_features"):
        return None
    return module


def load_updates(adapters, modules, device):
    """Validate complete factor pairs and retain ordered, unexpanded updates."""
    updates = {}
    for artifact, strength in adapters:
        tensors = load_file(artifact.path)
        consumed = set()
        targets = set()
        for key, w1 in tensors.items():
            if not key.endswith(".lokr_w1"):
                continue
            prefix = key.removesuffix(".lokr_w1")
            w2_key = prefix + ".lokr_w2"
            name = prefix.removeprefix("diffusion_model.")
            module = _linear(modules, name)
            if module is None or w2_key not in tensors or name in targets:
                raise ValueError(
                    "Unsupported Ideogram v4 LoKR target or incomplete pair"
                )
            w2 = tensors[w2_key]
            if (
                w1.ndim != 2
                or w2.ndim != 2
                or w1.shape[0] * w2.shape[0] != module.out_features
                or w1.shape[1] * w2.shape[1] != module.in_features
            ):
                raise ValueError("Unsupported Ideogram v4 LoKR factor dimensions")
            consumed.update((key, w2_key))
            alpha_key = prefix + ".alpha"
            if alpha_key in tensors:
                value = tensors[alpha_key]
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError("Invalid Ideogram v4 LoKR alpha")
                consumed.add(alpha_key)
            if strength:
                updates.setdefault(name, []).append(
                    LoKrUpdate(
                        w1.to(device=device, dtype=torch.bfloat16),
                        w2.to(device=device, dtype=torch.bfloat16),
                        strength,
                    )
                )
            targets.add(name)
        for key, down in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            up_key = prefix + ".lora_B.weight"
            name = prefix.removeprefix("diffusion_model.")
            module = _linear(modules, name)
            if module is None or up_key not in tensors or name in targets:
                raise ValueError(
                    "Unsupported Ideogram v4 LoRA target or incomplete pair"
                )
            up = tensors[up_key]
            if (
                down.ndim != 2
                or up.ndim != 2
                or down.shape[0] == 0
                or down.shape != (up.shape[1], module.in_features)
                or up.shape[0] != module.out_features
            ):
                raise ValueError("Unsupported Ideogram v4 LoRA factor dimensions")
            consumed.update((key, up_key))
            alpha = 1.0
            alpha_key = prefix + ".alpha"
            if alpha_key in tensors:
                value = tensors[alpha_key]
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError("Invalid Ideogram v4 LoRA alpha")
                alpha = value.item() / down.shape[0]
                consumed.add(alpha_key)
            if strength:
                updates.setdefault(name, []).append(
                    (
                        down.to(device=device, dtype=torch.bfloat16),
                        up.to(device=device, dtype=torch.bfloat16),
                        strength * alpha,
                    )
                )
            targets.add(name)
        if not targets or consumed != tensors.keys():
            raise ValueError(
                "Unsupported Ideogram v4 adapter format or unconsumed tensors"
            )
    return updates


def apply_updates(weight, updates):
    """Apply Comfy's ordered BF16 factor products to whole weights."""
    for update in updates:
        if isinstance(update, LoKrUpdate):
            delta = update.strength * torch.kron(update.w1, update.w2)
        else:
            down, up, strength = update
            delta = strength * torch.mm(up, down)
        weight.add_(delta.to(weight.dtype))
    return weight
