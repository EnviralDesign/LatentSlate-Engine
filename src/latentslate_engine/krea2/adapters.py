"""Measured Krea transformer LoRA key mapping and BF16 patch arithmetic."""

import zlib
from dataclasses import dataclass

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from safetensors.torch import load_file


@dataclass(frozen=True)
class LoKrUpdate:
    """Direct Kronecker factors, retained without expanding a dense weight."""

    w1: torch.Tensor
    w2: torch.Tensor
    strength: float


def _target_name(name):
    name = name.removeprefix("transformer.").removeprefix("diffusion_model.")
    basic = {
        "img_in": "first",
        "time_embed.linear_1": "tmlp.0",
        "time_embed.linear_2": "tmlp.2",
        "time_mod_proj": "tproj.1",
        "txt_in.linear_1": "txtmlp.1",
        "txt_in.linear_2": "txtmlp.3",
        "text_fusion.projector": "txtfusion.projector",
        "final_layer.linear": "last.linear",
    }
    if name in basic:
        return basic[name]
    name = name.replace("transformer_blocks.", "blocks.").replace("text_fusion.", "txtfusion.")
    for source, target in {
        "attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv",
        "attn.to_gate": "attn.gate", "attn.to_out.0": "attn.wo", "attn.to_out": "attn.wo",
        "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
    }.items():
        if name.endswith("." + source):
            return name[:-len(source)] + target
    return name


def load_updates(adapters, modules, device):
    """Load LoRA pairs or direct LoKr factors in declared adapter order."""
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
            target = _target_name(prefix)
            if target not in modules or w2_key not in tensors or target in targets:
                raise ValueError(f"Unsupported or duplicate Krea LoKr target: {prefix}")
            w2 = tensors[w2_key]
            module = modules[target]
            if w1.ndim != 2 or w2.ndim != 2 or (
                w1.shape[0] * w2.shape[0] != module.out_features
                or w1.shape[1] * w2.shape[1] != module.in_features
            ):
                raise ValueError(f"Unsupported Krea LoKr shape: {prefix}")
            consumed.update((key, w2_key))
            alpha_key = prefix + ".alpha"
            if alpha_key in tensors:
                alpha = tensors[alpha_key]
                if alpha.numel() != 1 or not torch.isfinite(alpha).all():
                    raise ValueError(f"Invalid Krea LoKr alpha: {prefix}")
                consumed.add(alpha_key)
            # Comfy LoKrAdapter.calculate_weight applies alpha/rank only when
            # rebuilding decomposed factors; direct w1/w2 use strength alone.
            if strength:
                dtype = torch.float32 if target == "txtfusion.projector" else torch.bfloat16
                updates.setdefault(target, []).append(LoKrUpdate(
                    w1.to(device=device, dtype=dtype),
                    w2.to(device=device, dtype=dtype), strength,
                ))
            targets.add(target)
        for key, down in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key[:-len(".lora_A.weight")]
            up_key = prefix + ".lora_B.weight"
            target = _target_name(prefix)
            if target not in modules or up_key not in tensors or target in targets:
                raise ValueError(f"Unsupported or duplicate Krea LoRA target: {prefix}")
            up = tensors[up_key]
            module = modules[target]
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0 or (
                down.shape != (up.shape[1], module.in_features)
                or up.shape[0] != module.out_features
            ):
                raise ValueError(f"Unsupported Krea LoRA shape: {prefix}")
            consumed.update((key, up_key))
            alpha_key = prefix + ".alpha"
            scale = strength
            if alpha_key in tensors:
                alpha = tensors[alpha_key]
                if alpha.numel() != 1 or not torch.isfinite(alpha).all():
                    raise ValueError(f"Invalid Krea LoRA alpha: {prefix}")
                scale *= alpha.item() / down.shape[0]
                consumed.add(alpha_key)
            if strength:
                dtype = torch.float32 if target == "txtfusion.projector" else torch.bfloat16
                updates.setdefault(target, []).append((
                    up.to(device=device, dtype=dtype),
                    down.to(device=device, dtype=dtype), scale,
                ))
            targets.add(target)
        if not targets or consumed != tensors.keys():
            raise ValueError(f"Unsupported Krea adapter tensors in {artifact.path.name}")
    return updates


def patch_weight(weight, updates):
    """Apply each LoRA in compute dtype, preserving the reference's addition order."""
    for update in updates:
        if isinstance(update, LoKrUpdate):
            delta = update.strength * torch.kron(update.w1, update.w2)
        else:
            up, down, strength = update
            delta = strength * torch.mm(up, down)
        weight = weight + delta.to(weight.dtype)
    return weight


def requantize_fp8(weight, name):
    """Match the reference's recalculated scale and module-seeded FP8 rounding."""
    scale = weight.abs().amax().float() / 448.0
    scaled = weight * (1.0 / scale).to(weight.dtype)
    generator = torch.Generator(device=weight.device).manual_seed(
        zlib.crc32(("diffusion_model." + name).encode())
    )
    random = torch.randint(0, 256, weight.shape, dtype=torch.uint8,
                           device=weight.device, generator=generator)
    data = ck.stochastic_rounding_fp8(scaled, random, torch.float8_e4m3fn)
    return QuantizedTensor(data, "TensorCoreFP8Layout", TensorCoreFP8Layout.Params(
        scale=scale, orig_dtype=weight.dtype, orig_shape=tuple(weight.shape),
    ))
