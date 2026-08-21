"""Exact Z-Image Qwen support config and mixed-checkpoint planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from ..stored_quant import (
    read_safetensors_header_bytes,
    read_safetensors_u8_payload,
)
from .z_image_qwen_architecture import (
    QWEN_BLOCK_COUNT,
    QWEN_HEAD_DIM,
    QWEN_HIDDEN_SIZE,
    QWEN_INTERMEDIATE_SIZE,
    QWEN_KV_HEADS,
    QWEN_NORM_EPS,
    QWEN_QUERY_HEADS,
    QWEN_ROPE_THETA,
    QWEN_VOCAB_SIZE,
    QWEN_WEIGHT_COUNT,
)

QWEN_DTYPE_COUNTS = {"BF16": 209, "F8_E4M3": 177, "U8": 12}
QWEN_HEADER_SHA256 = "7537b0cd31f4fc963d334b4f997cedee6f51c62aa8518b7b7a852b182144aed9"
QWEN_FIRST_LINEAR_SOURCE = "model.layers.0.self_attn.q_proj.weight"
QWEN_FIRST_LINEAR_SHAPE = (4096, 2560)
QWEN_FIRST_LINEAR_FORMAT = "fp8"


class _ZImageQwenSupport(Protocol):
    root: Path


def expected_qwen_weight_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (QWEN_VOCAB_SIZE, QWEN_HIDDEN_SIZE),
        "model.norm.weight": (QWEN_HIDDEN_SIZE,),
    }
    for index in range(QWEN_BLOCK_COUNT):
        prefix = f"model.layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (QWEN_HIDDEN_SIZE,),
                f"{prefix}.post_attention_layernorm.weight": (QWEN_HIDDEN_SIZE,),
                f"{prefix}.self_attn.q_norm.weight": (QWEN_HEAD_DIM,),
                f"{prefix}.self_attn.k_norm.weight": (QWEN_HEAD_DIM,),
                f"{prefix}.self_attn.q_proj.weight": (
                    QWEN_QUERY_HEADS * QWEN_HEAD_DIM,
                    QWEN_HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    QWEN_KV_HEADS * QWEN_HEAD_DIM,
                    QWEN_HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    QWEN_KV_HEADS * QWEN_HEAD_DIM,
                    QWEN_HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    QWEN_HIDDEN_SIZE,
                    QWEN_QUERY_HEADS * QWEN_HEAD_DIM,
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    QWEN_INTERMEDIATE_SIZE,
                    QWEN_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    QWEN_INTERMEDIATE_SIZE,
                    QWEN_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    QWEN_HIDDEN_SIZE,
                    QWEN_INTERMEDIATE_SIZE,
                ),
            }
        )
    if len(shapes) != QWEN_WEIGHT_COUNT:
        raise RuntimeError("Z-Image Qwen expected-key construction is incomplete")
    return shapes


def validate_support_qwen_config(support: _ZImageQwenSupport) -> None:
    config = json.loads((support.root / "text_encoder" / "config.json").read_text("utf-8"))
    expected = {
        "hidden_size": QWEN_HIDDEN_SIZE,
        "intermediate_size": QWEN_INTERMEDIATE_SIZE,
        "num_hidden_layers": QWEN_BLOCK_COUNT,
        "num_attention_heads": QWEN_QUERY_HEADS,
        "num_key_value_heads": QWEN_KV_HEADS,
        "head_dim": QWEN_HEAD_DIM,
        "rms_norm_eps": QWEN_NORM_EPS,
        "rope_theta": int(QWEN_ROPE_THETA),
        "vocab_size": QWEN_VOCAB_SIZE,
        "attention_bias": False,
        "hidden_act": "silu",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Z-Image Qwen support config differs from the exact architecture")


@dataclass(frozen=True, slots=True)
class ZImageMixedQwenPlan:
    identity: ArtifactIdentity
    header_sha256: str
    schema_sha256: str
    source_to_target: Mapping[str, str]
    fp8_sources: tuple[str, ...]
    nvfp4_sources: tuple[str, ...]
    dense_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    first_linear_format: str
    fingerprint: str


def plan_z_image_mixed_qwen(path: Path) -> ZImageMixedQwenPlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors":
        raise ValueError("Z-Image mixed Qwen is not SafeTensors")
    raw, header = read_safetensors_header_bytes(
        probe.identity.path, probe.identity.size_bytes
    )
    header_sha256 = hashlib.sha256(raw).hexdigest()
    if header_sha256 != QWEN_HEADER_SHA256:
        raise ValueError("Z-Image mixed Qwen header differs from the exact official mapping")
    weights = {
        key: value
        for key, value in header.items()
        if key.endswith(".weight") and isinstance(value, dict)
    }
    counts = {
        dtype: sum(value.get("dtype") == dtype for value in weights.values())
        for dtype in QWEN_DTYPE_COUNTS
    }
    if counts != QWEN_DTYPE_COUNTS or len(weights) != QWEN_WEIGHT_COUNT:
        raise ValueError("Z-Image mixed Qwen BF16/FP8/NVFP4 closure changed")
    expected_shapes = expected_qwen_weight_shapes()
    if set(weights) != set(expected_shapes):
        raise ValueError("Z-Image mixed Qwen is not the exact raw model.* closure")
    for key, expected_shape in expected_shapes.items():
        stored_shape = tuple(weights[key].get("shape", ()))
        if weights[key].get("dtype") == "U8":
            expected_shape = (expected_shape[0], expected_shape[1] // 2)
        if stored_shape != expected_shape:
            raise ValueError(f"Z-Image mixed Qwen weight geometry changed: {key}")
    fp8 = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "F8_E4M3"))
    nvfp4 = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "U8"))
    dense = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "BF16"))
    first_linear_format = (
        "fp8"
        if QWEN_FIRST_LINEAR_SOURCE in fp8
        else "nvfp4"
        if QWEN_FIRST_LINEAR_SOURCE in nvfp4
        else "dense"
    )
    if first_linear_format != QWEN_FIRST_LINEAR_FORMAT:
        raise ValueError("Z-Image Qwen first-linear format differs from the exact header")
    auxiliary: set[str] = set()
    for source in fp8:
        stem = source.removesuffix(".weight")
        scale = header.get(stem + ".weight_scale")
        marker = header.get(stem + ".comfy_quant")
        if (
            not isinstance(scale, dict)
            or scale.get("dtype") != "F32"
            or scale.get("shape") != []
            or not isinstance(marker, dict)
            or marker.get("dtype") != "U8"
            or marker.get("shape") != [27]
        ):
            raise ValueError(f"Z-Image mixed Qwen FP8 scale is invalid: {stem}")
        if _marker_format(probe.identity.path, probe.identity.size_bytes, raw, marker) != "float8_e4m3fn":
            raise ValueError(f"Z-Image mixed Qwen FP8 marker is invalid: {stem}")
        auxiliary.update((stem + ".weight_scale", stem + ".comfy_quant"))
    for source in nvfp4:
        stem = source.removesuffix(".weight")
        weight, block_scale, tensor_scale, marker = (
            header[source],
            header.get(stem + ".weight_scale"),
            header.get(stem + ".weight_scale_2"),
            header.get(stem + ".comfy_quant"),
        )
        shape = weight.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[1] % 8
            or not isinstance(block_scale, dict)
            or block_scale.get("dtype") != "F8_E4M3"
            or block_scale.get("shape") != [shape[0], shape[1] // 8]
            or not isinstance(tensor_scale, dict)
            or tensor_scale.get("dtype") != "F32"
            or tensor_scale.get("shape") != []
            or not isinstance(marker, dict)
            or marker.get("dtype") != "U8"
            or marker.get("shape") != [19]
        ):
            raise ValueError(f"Z-Image mixed Qwen NVFP4 storage is invalid: {stem}")
        if _marker_format(probe.identity.path, probe.identity.size_bytes, raw, marker) != "nvfp4":
            raise ValueError(f"Z-Image mixed Qwen NVFP4 marker is invalid: {stem}")
        auxiliary.update(
            (stem + ".weight_scale", stem + ".weight_scale_2", stem + ".comfy_quant")
        )
    expected_keys = set(weights) | auxiliary
    if expected_keys != {key for key in header if key != "__metadata__"}:
        raise ValueError("Z-Image mixed Qwen source/sidecar closure is incomplete")
    mapping = MappingProxyType({key: key for key in sorted(weights)})
    fingerprint = hashlib.sha256(
        json.dumps(
            {"header": header_sha256, "mapping": sorted(mapping.items()), "counts": counts},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return ZImageMixedQwenPlan(
        probe.identity,
        header_sha256,
        probe.schema_sha256,
        mapping,
        fp8,
        nvfp4,
        dense,
        tuple(sorted(auxiliary)),
        first_linear_format,
        fingerprint,
    )


def revalidate_z_image_mixed_qwen(plan: ZImageMixedQwenPlan) -> bool:
    try:
        return plan_z_image_mixed_qwen(plan.identity.path) == plan and revalidate_artifact(
            plan.identity
        )
    except (OSError, TypeError, ValueError):
        return False


def _marker_format(
    path: Path,
    size_bytes: int,
    raw_header: bytes,
    marker: Mapping[str, Any],
) -> str | None:
    try:
        parsed = json.loads(
            read_safetensors_u8_payload(path, size_bytes, raw_header, marker).decode(
                "utf-8"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Z-Image mixed Qwen marker is malformed") from exc
    return parsed.get("format") if isinstance(parsed, dict) else None
