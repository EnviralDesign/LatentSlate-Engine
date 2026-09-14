"""Qwen transformer LoRA mapping and pinned Comfy BF16 patch arithmetic."""

import zlib

import comfy_kitchen as ck
import torch
from safetensors.torch import load_file


def load_updates(adapters, modules, device):
    """Consume complete regular LoRA pairs, rejecting unknown keys and shapes."""
    updates = {}
    for artifact, strength in adapters:
        tensors = load_file(artifact.path)
        consumed = set()
        targets = set()
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            target = key.removesuffix(".lora_down.weight")
            up_key = target + ".lora_up.weight"
            if target not in modules or up_key not in tensors or target in targets:
                raise ValueError(f"Unsupported or duplicate Qwen LoRA target: {target}")
            up = tensors[up_key]
            module = modules[target]
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0 or (
                down.shape != (up.shape[1], module.in_features)
                or up.shape[0] != module.out_features
            ):
                raise ValueError(f"Unsupported Qwen LoRA shape: {target}")
            consumed.update((key, up_key))
            alpha_key = target + ".alpha"
            scale = strength
            if alpha_key in tensors:
                alpha = tensors[alpha_key]
                if alpha.numel() != 1 or not torch.isfinite(alpha).all():
                    raise ValueError(f"Invalid Qwen LoRA alpha: {target}")
                scale *= alpha.item() / down.shape[0]
                consumed.add(alpha_key)
            updates.setdefault(target, []).append((
                up.to(device=device, dtype=torch.bfloat16),
                down.to(device=device, dtype=torch.bfloat16), scale,
            ))
            targets.add(target)
        if not targets or consumed != tensors.keys():
            raise ValueError(f"Unsupported Qwen adapter tensors in {artifact.path.name}")
    return updates


def patch_weight(weight, updates):
    """Apply ordered deltas to a fresh compute weight, leaving the source intact."""
    for up, down, strength in updates:
        weight = weight + (strength * torch.mm(up, down)).to(weight.dtype)
    return weight


def requantize_fp8(weight, name):
    """Recalculate the resident scale with pinned module-seeded FP8 rounding."""
    scale = weight.abs().amax().float() / 448.0
    scaled = weight * (1.0 / scale).to(weight.dtype)
    generator = torch.Generator(device=weight.device).manual_seed(
        zlib.crc32(("diffusion_model." + name).encode())
    )
    random = torch.randint(0, 256, weight.shape, dtype=torch.uint8,
                           device=weight.device, generator=generator)
    return ck.stochastic_rounding_fp8(scaled, random, torch.float8_e4m3fn), scale
