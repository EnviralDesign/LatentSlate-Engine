"""Exact stored-layout planner/materializer for Z-Image's Qwen 3 4B encoder.

The official component is neither a generic FP8 checkpoint nor a dense Qwen
fallback: it contains 209 BF16 weights, 177 scalar-scale FP8 weights, and 12
packed NVFP4 weights.  This module accepts exactly that physical closure and
reuses the already-proven Kitchen Linear wrappers for native CUDA dispatch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from ..z_image_turbo_recipe import (
    _Z_QWEN_HEADER_SHA256,
    _read_u8_payload,
    _read_z_safetensors_header,
)
from .klein_stored_adapter import (
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    _restore_global_fp8_tensor,
    _restore_nvfp4_tensor,
    move_klein_module_storage,
)

_COUNTS = {"BF16": 209, "F8_E4M3": 177, "U8": 12}
_TIED_HEAD_SOURCE = "model.embed_tokens.weight"
_TIED_HEAD_TARGET = "lm_head.weight"


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
    fingerprint: str


def plan_z_image_mixed_qwen(path: Path) -> ZImageMixedQwenPlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors":
        raise ValueError("Z-Image mixed Qwen is not SafeTensors")
    raw, header = _read_z_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    header_sha256 = hashlib.sha256(raw).hexdigest()
    if header_sha256 != _Z_QWEN_HEADER_SHA256:
        raise ValueError("Z-Image mixed Qwen header differs from the exact official mapping")
    weights = {
        key: value
        for key, value in header.items()
        if key.endswith(".weight") and isinstance(value, dict)
    }
    counts = {
        dtype: sum(value.get("dtype") == dtype for value in weights.values()) for dtype in _COUNTS
    }
    if counts != _COUNTS or len(weights) != sum(_COUNTS.values()):
        raise ValueError("Z-Image mixed Qwen BF16/FP8/NVFP4 closure changed")
    fp8 = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "F8_E4M3"))
    nvfp4 = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "U8"))
    dense = tuple(sorted(key for key, value in weights.items() if value.get("dtype") == "BF16"))
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
        if (
            _marker_format(probe.identity.path, probe.identity.size_bytes, raw, marker)
            != "float8_e4m3fn"
        ):
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
        auxiliary.update((stem + ".weight_scale", stem + ".weight_scale_2", stem + ".comfy_quant"))
    expected_keys = set(weights) | auxiliary
    actual_model_keys = {key for key in header if key != "__metadata__"}
    # The official file contains only model state and exact quant sidecars.
    if expected_keys != actual_model_keys:
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
        fingerprint,
    )


def revalidate_z_image_mixed_qwen(plan: ZImageMixedQwenPlan) -> bool:
    try:
        return plan_z_image_mixed_qwen(plan.identity.path) == plan and revalidate_artifact(
            plan.identity
        )
    except (OSError, TypeError, ValueError):
        return False


def materialize_z_image_mixed_qwen(plan: ZImageMixedQwenPlan, model: nn.Module) -> nn.Module:
    """Install stored FP8/NVFP4 Linear wrappers into a caller-owned exact shell.

    The caller supplies the Qwen architecture shell because its config/support
    package is a separate future native-runtime concern.  This function proves
    every stored source maps one-to-one to that shell; it never substitutes a
    dense conversion for either low-bit format.
    """

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    if not revalidate_z_image_mixed_qwen(plan):
        raise ValueError("Z-Image mixed Qwen changed after planning")
    target = model.state_dict()
    expected = set(plan.source_to_target.values())
    tied_head = _validate_shell_closure(model, target, expected)
    if set(target) - ({_TIED_HEAD_TARGET} if tied_head else set()) != expected:
        missing, extra = sorted(expected - set(target)), sorted(set(target) - expected)
        raise ValueError(
            f"Z-Image Qwen shell does not exactly match pinned mapping: missing={missing[:2]}, extra={extra[:2]}"
        )
    consumed: set[str] = set()
    native: dict[str, str] = {}
    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("Z-Image mixed Qwen changed before materialization")
        for source in plan.fp8_sources:
            _replace_quantized_linear(
                model,
                source,
                _restore_global_fp8_tensor(
                    handle.get_tensor(source),
                    handle.get_tensor(source.removesuffix(".weight") + ".weight_scale"),
                    torch.bfloat16,
                ),
                "fp8",
            )
            consumed.add(source)
            native[source.removesuffix(".weight")] = "fp8"
        for source in plan.nvfp4_sources:
            stem = source.removesuffix(".weight")
            qdata = handle.get_tensor(source)
            _replace_quantized_linear(
                model,
                source,
                _restore_nvfp4_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    handle.get_tensor(stem + ".weight_scale_2"),
                    (qdata.shape[0], qdata.shape[1] * 2),
                    torch.bfloat16,
                ),
                "nvfp4",
            )
            consumed.add(source)
            native[stem] = "nvfp4"
        for source in plan.dense_sources:
            value = handle.get_tensor(source)
            if tuple(value.shape) != tuple(target[source].shape):
                raise ValueError(f"Z-Image Qwen dense shape mismatch: {source}")
            set_module_tensor_to_device(model, source, "cpu", value=value, dtype=torch.bfloat16)
            consumed.add(source)
    if tied_head:
        _restore_exact_tied_head(model)
    if consumed != set(plan.source_to_target):
        raise ValueError("Z-Image mixed Qwen source materialization is incomplete")
    unresolved = [key for key, value in model.state_dict().items() if value.is_meta]
    if unresolved:
        raise ValueError(f"Z-Image Qwen retains meta parameters: {unresolved[:2]}")
    model._latentslate_z_image_quant_modules = MappingProxyType(native)
    model._latentslate_z_image_qwen_identity = plan.identity
    model.eval()
    return model


def _validate_shell_closure(
    model: nn.Module,
    target: Mapping[str, torch.Tensor],
    expected: set[str],
) -> bool:
    """Allow only the standard Qwen tied output head beyond checkpoint keys."""

    target_keys = set(target)
    extras = target_keys - expected
    if not extras:
        return False
    if extras != {_TIED_HEAD_TARGET} or _TIED_HEAD_SOURCE not in expected:
        missing = sorted(expected - target_keys)
        raise ValueError(
            "Z-Image Qwen shell does not exactly match pinned mapping: "
            f"missing={missing[:2]}, extra={sorted(extras)[:2]}"
        )
    config = getattr(model, "config", None)
    if getattr(config, "tie_word_embeddings", None) is not True:
        raise TypeError("Z-Image Qwen extra lm_head requires tie_word_embeddings=True")
    input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    output_embeddings = getattr(model, "get_output_embeddings", lambda: None)()
    if (
        not isinstance(input_embeddings, nn.Embedding)
        or not isinstance(output_embeddings, nn.Linear)
        or output_embeddings.bias is not None
        or output_embeddings.weight is not input_embeddings.weight
        or tuple(input_embeddings.weight.shape) != tuple(target[_TIED_HEAD_SOURCE].shape)
        or tuple(output_embeddings.weight.shape) != tuple(target[_TIED_HEAD_TARGET].shape)
    ):
        raise TypeError("Z-Image Qwen lm_head must be the exact tied input embedding parameter")
    return True


def _restore_exact_tied_head(model: nn.Module) -> None:
    """Re-establish the only permitted shell-only parameter after loading embeddings."""

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    output_embeddings.weight = input_embeddings.weight
    if output_embeddings.weight is not input_embeddings.weight:
        raise RuntimeError("Z-Image Qwen failed to restore the exact tied lm_head parameter")


def _replace_quantized_linear(model: nn.Module, source: str, weight: Any, kind: str) -> None:
    stem = source.removesuffix(".weight")
    module = model.get_submodule(stem)
    if (
        type(module) is not nn.Linear
        or module.bias is not None
        or tuple(module.weight.shape) != tuple(weight.shape)
    ):
        raise TypeError(f"Z-Image Qwen quantized target is not exact bias-free Linear: {stem}")
    parent_path, _, leaf = stem.rpartition(".")
    replacement: nn.Module = (
        KleinStoredLinear(weight, input_scale=None)
        if kind == "fp8"
        else KleinStoredNVFP4Linear(weight, input_scale=None)
    )
    setattr(model.get_submodule(parent_path), leaf, replacement)


def z_image_mixed_dispatch_snapshot(model: nn.Module) -> dict[str, int]:
    expected = getattr(model, "_latentslate_z_image_quant_modules", None)
    if not isinstance(expected, Mapping) or len(expected) != 189:
        raise ValueError("Z-Image mixed Qwen native module closure is incomplete")
    values = {
        name: int(getattr(model.get_submodule(name), "native_dispatch_count", -1))
        for name in expected
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("Z-Image mixed Qwen native dispatch counter is missing")
    return values


def verify_z_image_mixed_dispatch(
    model: nn.Module, before: Mapping[str, int]
) -> dict[str, int | str]:
    after = z_image_mixed_dispatch_snapshot(model)
    deltas = {name: after[name] - int(before[name]) for name in after}
    if set(before) != set(after) or any(value <= 0 for value in deltas.values()):
        raise RuntimeError("Z-Image mixed Qwen did not dispatch every FP8/NVFP4 layer natively")
    return {
        "backend": "comfy-kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": len(deltas),
        "total_dispatches": sum(deltas.values()),
        "fp8_modules": 177,
        "nvfp4_modules": 12,
    }


class ZImageMixedQwenStage:
    def __init__(self, model: nn.Module, execution_device: torch.device | str) -> None:
        self.model = model
        self.execution_device = execution_device
        self._before: dict[str, int] | None = None

    def onload(self) -> None:
        move_klein_module_storage(self.model, self.execution_device)
        self._before = z_image_mixed_dispatch_snapshot(self.model)

    def verify_dispatch(self) -> dict[str, int | str]:
        if self._before is None:
            raise RuntimeError("Z-Image mixed Qwen was not staged")
        return verify_z_image_mixed_dispatch(self.model, self._before)

    def offload(self) -> None:
        move_klein_module_storage(self.model, "cpu")
        self._before = None


def _marker_format(
    path: Path, size_bytes: int, raw_header: bytes, marker: Mapping[str, Any]
) -> str | None:
    try:
        parsed = json.loads(_read_u8_payload(path, size_bytes, raw_header, marker).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Z-Image mixed Qwen marker is malformed") from exc
    return parsed.get("format") if isinstance(parsed, dict) else None
