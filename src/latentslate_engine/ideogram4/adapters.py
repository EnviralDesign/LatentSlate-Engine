"""Ordinary LoRA mapping for the Ideogram v4 transformer, following Comfy's mapping."""

import torch
from safetensors.torch import load_file


def load_updates(adapters, modules, device):
    """Validate complete factor pairs and retain ordered, unexpanded updates."""
    updates = {}
    for artifact, strength in adapters:
        tensors = load_file(artifact.path)
        consumed = set()
        pairs = 0
        for key, down in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            up_key = prefix + ".lora_B.weight"
            name = prefix.removeprefix("diffusion_model.")
            module = modules.get(name)
            if module is None or up_key not in tensors:
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
            pairs += 1
        if not pairs or consumed != tensors.keys():
            raise ValueError(
                "Unsupported Ideogram v4 adapter format or unconsumed tensors"
            )
    return updates


def apply_updates(weight, updates):
    """Apply Comfy's ordered BF16 factor products to whole weights."""
    for down, up, strength in updates:
        weight.add_((strength * torch.mm(up, down)).to(weight.dtype))
    return weight
