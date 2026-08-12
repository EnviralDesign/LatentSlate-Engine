"""Lightweight immutable artifact/architecture constants for Klein runtimes."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

KLEIN4B_CONFIG: MappingProxyType[str, Any] = MappingProxyType(
    {
        "patch_size": 1,
        "in_channels": 128,
        "out_channels": None,
        "num_layers": 5,
        "num_single_layers": 20,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 7680,
        "axes_dims_rope": (32, 32, 32, 32),
        "rope_theta": 2000,
        "timestep_guidance_channels": 256,
        "guidance_embeds": False,
        "mlp_ratio": 3.0,
        "eps": 1e-6,
    }
)

KLEIN9B_CONFIG: MappingProxyType[str, Any] = MappingProxyType(
    {
        "patch_size": 1,
        "in_channels": 128,
        "out_channels": None,
        "num_layers": 8,
        "num_single_layers": 24,
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "joint_attention_dim": 12_288,
        "axes_dims_rope": (32, 32, 32, 32),
        "rope_theta": 2000,
        "timestep_guidance_channels": 256,
        "guidance_embeds": False,
        "mlp_ratio": 3.0,
        "eps": 1e-6,
    }
)

KLEIN9_QWEN_MIXED_SCHEMA_SHA256 = (
    "42333ea5d161147268b724ca269782a6be0b4db0e41c19216a4f739b869e0ff6"
)
KLEIN9_QWEN_MIXED_CONTRACT = "comfy_quant/mixed_fp8_nvfp4"
KLEIN9_QWEN_MIXED_ARCHITECTURE = "qwen3_8b_fp8mixed"
