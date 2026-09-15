"""Ordinary LoRA mapping for the Z-Image transformer, following Comfy's mapping."""

import torch
from safetensors.torch import load_file


def target_name(name):
    name = name.removeprefix("diffusion_model.").removeprefix("transformer.")
    for index, component in enumerate(("to_q", "to_k", "to_v")):
        suffix = ".attention." + component
        if name.endswith(suffix):
            return name.removesuffix(suffix) + ".attention.qkv", index
    return name.replace(".attention.to_out.0", ".attention.out"), None


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
            name, part = target_name(prefix)
            module = modules.get(name)
            if module is None or up_key not in tensors:
                raise ValueError("Unsupported Z-Image LoRA target or incomplete pair")
            up = tensors[up_key]
            rows = module.out_features if part is None else module.out_features // 3
            if (
                down.ndim != 2
                or up.ndim != 2
                or down.shape[0] == 0
                or down.shape != (up.shape[1], module.in_features)
                or up.shape[0] != rows
            ):
                raise ValueError("Unsupported Z-Image LoRA factor dimensions")
            consumed.update((key, up_key))
            alpha = 1.0
            alpha_key = prefix + ".alpha"
            if alpha_key in tensors:
                value = tensors[alpha_key]
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError("Invalid Z-Image LoRA alpha")
                alpha = value.item() / down.shape[0]
                consumed.add(alpha_key)
            if strength:
                updates.setdefault(name, []).append(
                    (
                        down.to(device=device, dtype=torch.bfloat16),
                        up.to(device=device, dtype=torch.bfloat16),
                        strength * alpha,
                        0 if part is None else part * rows,
                        rows,
                    )
                )
            pairs += 1
        if not pairs or consumed != tensors.keys():
            raise ValueError("Unsupported Z-Image adapter format or unconsumed tensors")
    return updates


def apply_updates(weight, updates):
    """Apply Comfy's ordered BF16 factor products to whole weights or Q/K/V slices."""
    for down, up, strength, start, rows in updates:
        target = weight.narrow(0, start, rows)
        target.add_((strength * torch.mm(up, down)).to(weight.dtype))
    return weight
