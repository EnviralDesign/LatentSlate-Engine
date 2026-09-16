"""Ordinary H3 LoRA factors, following ComfyUI 1a14b82e (GPL-3.0)."""

import torch
from safetensors.torch import load_file


def load_updates(adapters, modules):
    """Retain ordered CPU factors after validating complete H3 linear targets."""
    updates = {}
    for path, strength in adapters:
        tensors = load_file(path)
        consumed = set()
        for key, down in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            up_key = prefix + ".lora_B.weight"
            name = prefix.removeprefix("diffusion_model.")
            module = modules.get(name)
            if module is None or up_key not in tensors:
                raise ValueError("Unsupported H3 LoRA target or incomplete pair")
            up = tensors[up_key]
            if (
                down.ndim != 2
                or up.ndim != 2
                or down.shape[0] == 0
                or down.shape != (up.shape[1], module.in_features)
                or up.shape[0] != module.out_features
            ):
                raise ValueError("Unsupported H3 LoRA factor dimensions")
            consumed.update((key, up_key))
            alpha = 1.0
            alpha_key = prefix + ".alpha"
            if alpha_key in tensors:
                value = tensors[alpha_key]
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError("Invalid H3 LoRA alpha")
                alpha = value.item() / down.shape[0]
                consumed.add(alpha_key)
            if strength:
                updates.setdefault(name, []).append((down, up, strength * alpha))
        if not consumed or consumed != tensors.keys():
            raise ValueError("Unsupported H3 adapter format or unconsumed tensors")
    return updates


def apply_updates(weight, updates):
    """Apply each factor product at the reference's current compute dtype."""
    for down, up, strength in updates:
        delta = torch.mm(
            up.to(device=weight.device, dtype=weight.dtype),
            down.to(device=weight.device, dtype=weight.dtype),
        )
        weight.add_((strength * delta).to(weight.dtype))
    return weight
