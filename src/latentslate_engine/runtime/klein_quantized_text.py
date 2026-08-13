"""Exact mixed-quantized Qwen3-8B component for Klein 9B.

The pinned official Klein 9B workflows select one standalone Qwen checkpoint
whose Linear weights deliberately mix tensor-wise FP8 and packed NVFP4.  This
module validates that immutable file, restores its physical Kitchen layouts
without a converted copy, and owns its explicit whole-component staging.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from .klein_contracts import (
    KLEIN9_QWEN_MIXED_ARCHITECTURE as _KLEIN9_QWEN_MIXED_ARCHITECTURE,
)
from .klein_contracts import (
    KLEIN9_QWEN_MIXED_CONTRACT as _KLEIN9_QWEN_MIXED_CONTRACT,
)
from .klein_contracts import (
    KLEIN9_QWEN_MIXED_SCHEMA_SHA256 as _KLEIN9_QWEN_MIXED_SCHEMA_SHA256,
)
from .klein_stored_adapter import (
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    _restore_global_fp8_tensor,
    _restore_nvfp4_tensor,
    move_klein_module_storage,
)

KLEIN9_QWEN_MIXED_ARCHITECTURE = _KLEIN9_QWEN_MIXED_ARCHITECTURE
KLEIN9_QWEN_MIXED_CONTRACT = _KLEIN9_QWEN_MIXED_CONTRACT
KLEIN9_QWEN_MIXED_SCHEMA_SHA256 = _KLEIN9_QWEN_MIXED_SCHEMA_SHA256

_SIZE_BYTES = 8_664_848_742
_TENSOR_COUNT = 935
_TENSOR_DTYPES = ("BF16", "F32", "F8_E4M3", "U8")
_AUXILIARY_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2")


@dataclass(frozen=True, slots=True)
class KleinMixedTextEncoderPlan:
    identity: ArtifactIdentity
    schema_sha256: str
    quantized_formats: Mapping[str, str]
    dense_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]


def plan_klein_mixed_text_encoder(path: Path) -> KleinMixedTextEncoderPlan:
    """Validate the exact public Qwen3-8B FP8/NVFP4 checkpoint."""

    from safetensors import safe_open

    probe = probe_artifact(Path(path))
    errors: list[str] = []
    if probe.format != "safetensors":
        errors.append("container is not SafeTensors")
    if probe.identity.size_bytes != _SIZE_BYTES:
        errors.append("file size differs from the pinned artifact")
    if probe.schema_sha256 != KLEIN9_QWEN_MIXED_SCHEMA_SHA256:
        errors.append("key/shape/dtype schema differs from the pinned artifact")
    if probe.tensor_count != _TENSOR_COUNT or probe.tensor_dtypes != _TENSOR_DTYPES:
        errors.append("tensor count or stored dtypes differ from the pinned artifact")
    if errors:
        raise ValueError("Klein 9B text encoder contract failed: " + "; ".join(errors))

    formats: dict[str, str] = {}
    dense: list[str] = []
    auxiliary: set[str] = set()
    with safe_open(str(probe.identity.path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        base = sorted(key for key in keys if not key.endswith(_AUXILIARY_SUFFIXES))
        for source in base:
            view = handle.get_slice(source)
            dtype = view.get_dtype()
            if not source.endswith(".weight") or dtype == "BF16":
                dense.append(source)
                continue
            stem = source.removesuffix(".weight")
            config_key = stem + ".comfy_quant"
            scale_key = stem + ".weight_scale"
            if config_key not in keys or scale_key not in keys:
                raise ValueError(f"Klein 9B text encoder sidecars are incomplete: {stem}")
            raw_config = handle.get_tensor(config_key)
            if raw_config.dtype is not torch.uint8 or raw_config.ndim != 1:
                raise ValueError(f"Klein 9B text encoder quant descriptor is invalid: {stem}")
            try:
                config = json.loads(raw_config.numpy().tobytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Klein 9B text encoder quant descriptor is malformed: {stem}"
                ) from exc
            expected_format = "float8_e4m3fn" if dtype == "F8_E4M3" else "nvfp4"
            if config != {"format": expected_format}:
                raise ValueError(f"Klein 9B text encoder quant format is invalid: {stem}")
            scale = handle.get_slice(scale_key)
            shape = tuple(view.get_shape())
            if expected_format == "float8_e4m3fn":
                if scale.get_dtype() != "F32" or scale.get_shape() != []:
                    raise ValueError(f"Klein 9B FP8 scale is invalid: {stem}")
            else:
                tensor_scale_key = stem + ".weight_scale_2"
                if dtype != "U8" or len(shape) != 2 or shape[1] % 8:
                    raise ValueError(f"Klein 9B NVFP4 storage is invalid: {stem}")
                if (
                    scale.get_dtype() != "F8_E4M3"
                    or tuple(scale.get_shape()) != (shape[0], shape[1] // 8)
                    or tensor_scale_key not in keys
                    or handle.get_slice(tensor_scale_key).get_dtype() != "F32"
                    or handle.get_slice(tensor_scale_key).get_shape() != []
                ):
                    raise ValueError(f"Klein 9B NVFP4 scales are invalid: {stem}")
                auxiliary.add(tensor_scale_key)
            formats[stem] = expected_format
            auxiliary.update((config_key, scale_key))

        if len(formats) != 226 or sum(value == "nvfp4" for value in formats.values()) != 85:
            raise ValueError("Klein 9B text encoder mixed quantization coverage changed")
        if sum(value == "float8_e4m3fn" for value in formats.values()) != 141:
            raise ValueError("Klein 9B text encoder FP8 coverage changed")
        if len(dense) != 172 or set(base) | auxiliary != keys:
            raise ValueError("Klein 9B text encoder source-role closure changed")

    return KleinMixedTextEncoderPlan(
        probe.identity,
        probe.schema_sha256,
        MappingProxyType(dict(sorted(formats.items()))),
        tuple(sorted(dense)),
        tuple(sorted(auxiliary)),
    )


def revalidate_klein_mixed_text_encoder(plan: KleinMixedTextEncoderPlan) -> bool:
    try:
        refreshed = plan_klein_mixed_text_encoder(plan.identity.path)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed == plan and revalidate_artifact(plan.identity)


def load_klein_mixed_text_encoder(
    plan: KleinMixedTextEncoderPlan,
    support_root: Path,
) -> Any:
    """Restore exact FP8/NVFP4 Qwen storage into a Transformers meta shell."""

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    if not revalidate_klein_mixed_text_encoder(plan):
        raise ValueError("Klein 9B text encoder changed after planning")
    config = Qwen3Config.from_pretrained(
        Path(support_root) / "text_encoder",
        local_files_only=True,
    )
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)
    model.tie_weights()
    target = model.state_dict()
    expected_sources = set(target) - {"lm_head.weight"}
    consumed: set[str] = set()
    quantized_modules: dict[str, str] = {}

    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("Klein 9B text encoder changed before materialization")
        base_sources = set(plan.dense_sources) | {
            stem + ".weight" for stem in plan.quantized_formats
        }
        if base_sources != expected_sources:
            missing = sorted(expected_sources - base_sources)
            extra = sorted(base_sources - expected_sources)
            raise RuntimeError(
                f"Klein 9B text encoder does not cover its Qwen shell: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

        for stem, quant_format in plan.quantized_formats.items():
            source = stem + ".weight"
            module = model.get_submodule(stem)
            if type(module) is not nn.Linear or module.bias is not None:
                raise TypeError(f"Klein 9B quantized target is not a bias-free Linear: {stem}")
            qdata = handle.get_tensor(source)
            expected_shape = tuple(module.weight.shape)
            if quant_format == "float8_e4m3fn":
                if tuple(qdata.shape) != expected_shape:
                    raise RuntimeError(f"Klein 9B FP8 shape mismatch: {stem}")
                weight = _restore_global_fp8_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    torch.bfloat16,
                )
                replacement: nn.Module = KleinStoredLinear(weight, input_scale=None)
            else:
                logical_shape = (qdata.shape[0], qdata.shape[1] * 2)
                if logical_shape != expected_shape:
                    raise RuntimeError(f"Klein 9B NVFP4 shape mismatch: {stem}")
                weight = _restore_nvfp4_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    handle.get_tensor(stem + ".weight_scale_2"),
                    logical_shape,
                    torch.bfloat16,
                )
                replacement = KleinStoredNVFP4Linear(weight, input_scale=None)
            parent_path, _, leaf = stem.rpartition(".")
            setattr(model.get_submodule(parent_path), leaf, replacement)
            consumed.add(source)
            quantized_modules[stem] = quant_format

        for source in plan.dense_sources:
            value = handle.get_tensor(source)
            if tuple(value.shape) != tuple(target[source].shape):
                raise RuntimeError(f"Klein 9B dense text-encoder shape mismatch: {source}")
            set_module_tensor_to_device(
                model,
                source,
                "cpu",
                value=value,
                dtype=torch.bfloat16,
            )
            consumed.add(source)

    if consumed != expected_sources:
        raise RuntimeError("Klein 9B text-encoder materialization is incomplete")
    model.lm_head.weight = model.model.embed_tokens.weight
    unresolved = [name for name, value in model.state_dict().items() if value.is_meta]
    if unresolved:
        raise RuntimeError(f"Klein 9B text encoder retained meta state: {unresolved[:3]}")
    model._latentslate_klein_mixed_quant_modules = MappingProxyType(quantized_modules)
    model._latentslate_klein_artifact_identity = plan.identity
    model.eval()
    return model


def mixed_dispatch_snapshot(model: Any) -> dict[str, int]:
    expected = getattr(model, "_latentslate_klein_mixed_quant_modules", None)
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("Klein 9B mixed text-encoder dispatch contract is missing")
    actual = {
        name: int(getattr(model.get_submodule(name), "native_dispatch_count", -1))
        for name in expected
    }
    if any(value < 0 for value in actual.values()):
        raise RuntimeError("Klein 9B mixed text-encoder native counters are missing")
    return actual


def verify_mixed_dispatch(model: Any, before: Mapping[str, int]) -> dict[str, int | str]:
    after = mixed_dispatch_snapshot(model)
    if set(after) != set(before):
        raise RuntimeError("Klein 9B mixed text-encoder module identity changed")
    deltas = {name: after[name] - int(before[name]) for name in after}
    if not deltas or any(value <= 0 for value in deltas.values()):
        raise RuntimeError("Klein 9B mixed text encoder did not use every native quantized layer")
    return {
        "backend": "comfy-kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": len(deltas),
        "total_dispatches": sum(deltas.values()),
        "minimum_module_dispatches": min(deltas.values()),
        "maximum_module_dispatches": max(deltas.values()),
    }


class KleinMixedTextEncoderStage:
    """Explicit whole-Qwen CPU/CUDA staging with measured native dispatch."""

    def __init__(self, model: Any, execution_device: torch.device | str) -> None:
        self.model = model
        self.execution_device = torch.device(execution_device)
        self._before: dict[str, int] | None = None
        self.last_dispatch: dict[str, int | str] | None = None

    def onload(self) -> None:
        move_klein_module_storage(self.model, self.execution_device)
        self._before = mixed_dispatch_snapshot(self.model)

    def verify_dispatch(self) -> dict[str, int | str]:
        if self._before is None:
            raise RuntimeError("Klein 9B text encoder was not staged before dispatch")
        self.last_dispatch = verify_mixed_dispatch(self.model, self._before)
        return dict(self.last_dispatch)

    def offload(self) -> None:
        move_klein_module_storage(self.model, "cpu")
        self._before = None
