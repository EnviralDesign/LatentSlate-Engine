"""Header-only planning for Comfy-native FLUX.2 Klein stored FP8 weights.

The official Klein FP8 artifact stores FP8 weight payloads and scalar scales in
the Comfy/Black Forest Labs topology.  This module proves that topology against
the pinned Diffusers ``Flux2Transformer2DModel`` shell without loading tensor
payloads and without constructing a quantization configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import (
    _MAX_HEADER_BYTES,
    ArtifactIdentity,
    probe_safetensors,
    revalidate_artifact,
)
from .klein_contracts import (
    KLEIN4B_CONFIG as _KLEIN4B_CONFIG,
)
from .klein_contracts import (
    KLEIN9B_CONFIG as _KLEIN9B_CONFIG,
)
from .residency_policy import (
    ResidencyDecision,
    choose_cuda_residency,
    require_grouped_residency,
)

KLEIN4B_CONFIG = _KLEIN4B_CONFIG
KLEIN9B_CONFIG = _KLEIN9B_CONFIG

KLEIN_STORED_FP8_CONTRACT = "comfy_quant/float8_e4m3fn_global"
KLEIN_STORED_NVFP4_CONTRACT = "comfy_quant/nvfp4_tensorcore"
_QUANT_SUFFIXES = (".weight_scale", ".input_scale")
_NVFP4_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# Whole-model Klein residency deliberately has its own guard.  A stored FP8
# transformer is reconstructed while moving its physical storage, so two
# sessions must never race to change a transformer's device residency.
_KLEIN_SESSION_GUARD_LOCK = threading.RLock()
_ACTIVE_KLEIN_SESSION: KleinTransformerResidencySession | None = None

_ROOT_MAP = {
    "img_in.weight": "x_embedder.weight",
    "txt_in.weight": "context_embedder.weight",
    "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
    "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.linear.weight": "proj_out.weight",
}

_DOUBLE_DIRECT_MAP = {
    "img_attn.norm.query_norm.scale": "attn.norm_q.weight",
    "img_attn.norm.key_norm.scale": "attn.norm_k.weight",
    "img_attn.proj.weight": "attn.to_out.0.weight",
    "img_mlp.0.weight": "ff.linear_in.weight",
    "img_mlp.2.weight": "ff.linear_out.weight",
    "txt_attn.norm.query_norm.scale": "attn.norm_added_q.weight",
    "txt_attn.norm.key_norm.scale": "attn.norm_added_k.weight",
    "txt_attn.proj.weight": "attn.to_add_out.weight",
    "txt_mlp.0.weight": "ff_context.linear_in.weight",
    "txt_mlp.2.weight": "ff_context.linear_out.weight",
}

_SINGLE_MAP = {
    "linear1.weight": "attn.to_qkv_mlp_proj.weight",
    "linear2.weight": "attn.to_out.weight",
    "norm.query_norm.scale": "attn.norm_q.weight",
    "norm.key_norm.scale": "attn.norm_k.weight",
}


@dataclass(frozen=True, slots=True)
class KleinShapeMismatch:
    source_key: str
    target_keys: tuple[str, ...]
    source_shape: tuple[int, ...]
    target_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class KleinStoredAdapterPlan:
    """Immutable proof that one file exactly fits the supported Klein shell."""

    identity: ArtifactIdentity
    artifact_contract: str | None
    config_fingerprint: str
    source_to_targets: Mapping[str, tuple[str, ...]]
    quantized_sources: tuple[str, ...]
    dense_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    mapping_fingerprint: str
    missing_targets: tuple[str, ...]
    duplicate_targets: tuple[str, ...]
    unexpected_sources: tuple[str, ...]
    shape_mismatches: tuple[KleinShapeMismatch, ...]
    contract_errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.artifact_contract is not None and not (
            self.missing_targets
            or self.duplicate_targets
            or self.unexpected_sources
            or self.shape_mismatches
            or self.contract_errors
        )

    @property
    def errors(self) -> tuple[str, ...]:
        errors: list[str] = list(self.contract_errors)
        if self.missing_targets:
            errors.append(f"missing Diffusers parameters: {len(self.missing_targets)}")
        if self.duplicate_targets:
            errors.append(f"duplicate mapped parameters: {len(self.duplicate_targets)}")
        if self.unexpected_sources:
            errors.append(f"unrecognized source tensors: {len(self.unexpected_sources)}")
        if self.shape_mismatches:
            errors.append(f"parameter shape mismatches: {len(self.shape_mismatches)}")
        return tuple(errors)

    def require_available(self) -> None:
        if not self.available:
            raise ValueError("Klein stored adapter unavailable: " + "; ".join(self.errors))


def build_klein_transformer_skeleton(
    config: Mapping[str, Any] = KLEIN4B_CONFIG,
):
    """Build the pinned Diffusers Klein shell with parameters on ``meta``."""

    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel

    with init_empty_weights():
        return Flux2Transformer2DModel(**dict(config))


def map_comfy_flux2_parameter(source_key: str) -> tuple[str, ...]:
    """Map one Comfy/BFL source parameter to its Diffusers shell target(s)."""

    root = _ROOT_MAP.get(source_key)
    if root is not None:
        return (root,)

    match = re.fullmatch(r"double_blocks\.(\d+)\.(.+)", source_key)
    if match:
        block, suffix = match.groups()
        prefix = f"transformer_blocks.{block}."
        if suffix == "img_attn.qkv.weight":
            return tuple(
                prefix + name
                for name in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")
            )
        if suffix == "txt_attn.qkv.weight":
            return tuple(
                prefix + name
                for name in (
                    "attn.add_q_proj.weight",
                    "attn.add_k_proj.weight",
                    "attn.add_v_proj.weight",
                )
            )
        mapped = _DOUBLE_DIRECT_MAP.get(suffix)
        return (prefix + mapped,) if mapped is not None else ()

    match = re.fullmatch(r"single_blocks\.(\d+)\.(.+)", source_key)
    if match:
        block, suffix = match.groups()
        mapped = _SINGLE_MAP.get(suffix)
        return (f"single_transformer_blocks.{block}.{mapped}",) if mapped is not None else ()
    return ()


def comfy_flux2_source_for_target(target_key: str) -> str | None:
    """Return the canonical source key for a Diffusers Klein state key."""

    for source, target in _ROOT_MAP.items():
        if target == target_key:
            return source

    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", target_key)
    if match:
        block, suffix = match.groups()
        if suffix in {"attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"}:
            return f"double_blocks.{block}.img_attn.qkv.weight"
        if suffix in {"attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"}:
            return f"double_blocks.{block}.txt_attn.qkv.weight"
        for source_suffix, target_suffix in _DOUBLE_DIRECT_MAP.items():
            if target_suffix == suffix:
                return f"double_blocks.{block}.{source_suffix}"

    match = re.fullmatch(r"single_transformer_blocks\.(\d+)\.(.+)", target_key)
    if match:
        block, suffix = match.groups()
        for source_suffix, target_suffix in _SINGLE_MAP.items():
            if target_suffix == suffix:
                return f"single_blocks.{block}.{source_suffix}"
    return None


def plan_comfy_klein_transformer(
    artifact_path: Path,
    config: Mapping[str, Any] = KLEIN4B_CONFIG,
) -> KleinStoredAdapterPlan:
    """Validate an official-style Klein FP8 file without loading tensor data."""

    probe = probe_safetensors(Path(artifact_path).resolve(strict=True))
    header = _read_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    metadata = header.pop("__metadata__", {})
    skeleton = build_klein_transformer_skeleton(config)
    target_shapes = {key: tuple(value.shape) for key, value in skeleton.state_dict().items()}

    base_sources = {
        key: value for key, value in header.items() if not key.endswith(_QUANT_SUFFIXES)
    }
    source_to_targets: dict[str, tuple[str, ...]] = {}
    unexpected: list[str] = []
    shape_mismatches: list[KleinShapeMismatch] = []
    target_sources: dict[str, list[str]] = defaultdict(list)

    for source, entry in sorted(base_sources.items()):
        targets = map_comfy_flux2_parameter(source)
        if not targets:
            unexpected.append(source)
            continue
        source_to_targets[source] = targets
        for target in targets:
            target_sources[target].append(source)
        source_shape = tuple(entry["shape"])
        expected_shapes = tuple(target_shapes.get(target, ()) for target in targets)
        if not _source_shape_matches(source_shape, expected_shapes):
            shape_mismatches.append(
                KleinShapeMismatch(source, targets, source_shape, expected_shapes)
            )

    quantized = tuple(
        sorted(source for source, entry in base_sources.items() if entry.get("dtype") == "F8_E4M3")
    )
    dense = tuple(sorted(set(base_sources) - set(quantized)))
    contract_errors, auxiliaries = _validate_fp8_contract(
        header=header,
        metadata=metadata,
        quantized=quantized,
        dense=dense,
        source_to_targets=source_to_targets,
        skeleton=skeleton,
    )
    expected_auxiliaries = set(auxiliaries)
    actual_auxiliaries = {key for key in header if key.endswith(_QUANT_SUFFIXES)}
    unexpected.extend(sorted(actual_auxiliaries - expected_auxiliaries))

    missing = tuple(sorted(set(target_shapes) - set(target_sources)))
    duplicates = tuple(
        sorted(target for target, sources in target_sources.items() if len(sources) != 1)
    )
    config_fingerprint = _fingerprint(dict(config))
    mapping_fingerprint = _fingerprint(
        {
            "config": config_fingerprint,
            "contract": KLEIN_STORED_FP8_CONTRACT,
            "mapping": source_to_targets,
            "quantized": quantized,
            "dense": {source: header[source]["dtype"] for source in dense},
            "auxiliary": auxiliaries,
        }
    )
    contract = KLEIN_STORED_FP8_CONTRACT if not contract_errors else None
    return KleinStoredAdapterPlan(
        identity=probe.identity,
        artifact_contract=contract,
        config_fingerprint=config_fingerprint,
        source_to_targets=MappingProxyType(source_to_targets),
        quantized_sources=quantized,
        dense_sources=dense,
        auxiliary_sources=auxiliaries,
        mapping_fingerprint=mapping_fingerprint,
        missing_targets=missing,
        duplicate_targets=duplicates,
        unexpected_sources=tuple(sorted(set(unexpected))),
        shape_mismatches=tuple(shape_mismatches),
        contract_errors=tuple(contract_errors),
    )


def plan_bfl_klein_nvfp4_transformer(
    artifact_path: Path,
    config: Mapping[str, Any] = KLEIN4B_CONFIG,
) -> KleinStoredAdapterPlan:
    """Validate the exact first-party BFL Distilled NVFP4 tensor layout."""

    probe = probe_safetensors(Path(artifact_path).resolve(strict=True))
    header = _read_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    metadata = header.pop("__metadata__", {})
    skeleton = build_klein_transformer_skeleton(config)
    target_shapes = {key: tuple(value.shape) for key, value in skeleton.state_dict().items()}
    base_sources = {
        key: value for key, value in header.items() if not key.endswith(_NVFP4_SUFFIXES)
    }
    source_to_targets: dict[str, tuple[str, ...]] = {}
    unexpected: list[str] = []
    mismatches: list[KleinShapeMismatch] = []
    target_sources: dict[str, list[str]] = defaultdict(list)
    for source, entry in sorted(base_sources.items()):
        targets = map_comfy_flux2_parameter(source)
        if not targets:
            unexpected.append(source)
            continue
        source_to_targets[source] = targets
        for target in targets:
            target_sources[target].append(source)
        stored_shape = tuple(entry["shape"])
        logical_shape = (
            (stored_shape[0], stored_shape[1] * 2)
            if entry.get("dtype") == "U8" and len(stored_shape) == 2
            else stored_shape
        )
        expected = tuple(target_shapes.get(target, ()) for target in targets)
        if not _source_shape_matches(logical_shape, expected):
            mismatches.append(KleinShapeMismatch(source, targets, logical_shape, expected))

    quantized = tuple(
        sorted(source for source, entry in base_sources.items() if entry.get("dtype") == "U8")
    )
    dense = tuple(sorted(set(base_sources) - set(quantized)))
    errors, auxiliaries = _validate_nvfp4_contract(
        header=header,
        metadata=metadata,
        quantized=quantized,
        dense=dense,
        source_to_targets=source_to_targets,
        skeleton=skeleton,
    )
    actual_auxiliaries = {key for key in header if key.endswith(_NVFP4_SUFFIXES)}
    unexpected.extend(sorted(actual_auxiliaries - set(auxiliaries)))
    missing = tuple(sorted(set(target_shapes) - set(target_sources)))
    duplicates = tuple(
        sorted(target for target, sources in target_sources.items() if len(sources) != 1)
    )
    config_fingerprint = _fingerprint(dict(config))
    mapping_fingerprint = _fingerprint(
        {
            "config": config_fingerprint,
            "contract": KLEIN_STORED_NVFP4_CONTRACT,
            "mapping": source_to_targets,
            "quantized": quantized,
            "dense": {source: header[source]["dtype"] for source in dense},
            "auxiliary": auxiliaries,
        }
    )
    return KleinStoredAdapterPlan(
        identity=probe.identity,
        artifact_contract=KLEIN_STORED_NVFP4_CONTRACT if not errors else None,
        config_fingerprint=config_fingerprint,
        source_to_targets=MappingProxyType(source_to_targets),
        quantized_sources=quantized,
        dense_sources=dense,
        auxiliary_sources=auxiliaries,
        mapping_fingerprint=mapping_fingerprint,
        missing_targets=missing,
        duplicate_targets=duplicates,
        unexpected_sources=tuple(sorted(set(unexpected))),
        shape_mismatches=tuple(mismatches),
        contract_errors=tuple(errors),
    )


def materialize_klein_transformer(
    plan: KleinStoredAdapterPlan,
    config: Mapping[str, Any] = KLEIN4B_CONFIG,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Restore one validated official Klein FP8 artifact into its Diffusers shell.

    The SafeTensors payload is opened once on CPU. Stored FP8 bytes and their
    scalar scales are wrapped directly in Comfy Kitchen ``QuantizedTensor``
    objects; this function never invokes a quantizer for model weights. Fused
    QKV payloads are row views of the stored tensor and therefore retain their
    exact FP8 dtype and scale.
    """

    from safetensors import safe_open

    plan.require_available()
    if plan.artifact_contract != KLEIN_STORED_FP8_CONTRACT:
        raise ValueError("Klein materializer: unsupported stored precision contract")
    if compute_dtype is not torch.bfloat16:
        raise ValueError("Klein materializer: official stored FP8 requires BF16 compute")
    if plan.config_fingerprint != _fingerprint(dict(config)):
        raise ValueError("Klein materializer: config does not match validated plan")
    _validate_materializer_plan(plan)

    transformer = build_klein_transformer_skeleton(config)
    expected_targets = set(transformer.state_dict())
    mapped_targets = [target for targets in plan.source_to_targets.values() for target in targets]
    if set(mapped_targets) != expected_targets or len(mapped_targets) != len(expected_targets):
        raise ValueError(
            "Klein materializer: plan targets do not exactly match this transformer shell"
        )

    consumed_sources: set[str] = set()
    consumed_targets: set[str] = set()
    consumed_auxiliary: set[str] = set()
    qdata = qdata_parts = qdata_part = None
    weight_scale = input_scale = quantized_weight = dense = None
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            # Revalidate only after the handle is bound. This rejects a directory
            # entry replacement even when the old file handle remains readable.
            if not revalidate_artifact(plan.identity):
                raise ValueError(
                    "Klein materializer: artifact identity changed before materialization"
                )
            _validate_bound_header(handle, plan)
            if not revalidate_artifact(plan.identity):
                raise ValueError(
                    "Klein materializer: artifact identity changed during descriptor discovery"
                )

            for source in plan.quantized_sources:
                targets = plan.source_to_targets[source]
                qdata = handle.get_tensor(source)
                stem = source.removesuffix(".weight")
                weight_key = stem + ".weight_scale"
                input_key = stem + ".input_scale"
                weight_scale = handle.get_tensor(weight_key)
                input_scale = handle.get_tensor(input_key)
                _validate_stored_fp8_payload(source, qdata, weight_scale, input_scale)

                row_counts = tuple(
                    int(transformer.get_submodule(target.rpartition(".")[0]).weight.shape[0])
                    for target in targets
                )
                qdata_parts = (
                    (qdata,) if len(targets) == 1 else torch.split(qdata, row_counts, dim=0)
                )
                if len(qdata_parts) != len(targets):
                    raise ValueError("Klein materializer: fused QKV split is incomplete")
                for target, qdata_part in zip(targets, qdata_parts, strict=True):
                    parent_path, _, leaf = target.rpartition(".")
                    module = transformer.get_submodule(parent_path)
                    if leaf != "weight" or type(module) is not nn.Linear or module.bias is not None:
                        raise TypeError(
                            f"Klein materializer: {target!r} is not a bias-free nn.Linear weight"
                        )
                    quantized_weight = _restore_global_fp8_tensor(
                        qdata_part, weight_scale, compute_dtype
                    )
                    _replace_linear(
                        transformer,
                        parent_path,
                        KleinStoredLinear(quantized_weight, input_scale=input_scale),
                    )
                    consumed_targets.add(target)
                consumed_sources.add(source)
                consumed_auxiliary.update((weight_key, input_key))

            for source in plan.dense_sources:
                targets = plan.source_to_targets[source]
                dense = handle.get_tensor(source)
                if source == "final_layer.adaLN_modulation.1.weight":
                    dense = _swap_adaln_scale_shift(dense)
                row_counts = tuple(
                    int(transformer.get_submodule(target.rpartition(".")[0]).weight.shape[0])
                    for target in targets
                )
                dense_parts = (
                    (dense,)
                    if len(targets) == 1
                    else torch.split(dense, row_counts, dim=0)
                )
                if len(dense_parts) != len(targets):
                    raise ValueError("Klein materializer: fused dense split is incomplete")
                for target, dense_part in zip(targets, dense_parts, strict=True):
                    _assign_dense_target(transformer, target, dense_part)
                    consumed_targets.add(target)
                consumed_sources.add(source)

        if consumed_sources != set(plan.source_to_targets):
            raise ValueError("Klein materializer: planned source consumption is incomplete")
        if consumed_targets != expected_targets:
            raise ValueError("Klein materializer: target materialization is incomplete")
        if consumed_auxiliary != set(plan.auxiliary_sources):
            raise ValueError("Klein materializer: quant auxiliary consumption is incomplete")
        _validate_materialized_transformer(transformer)
        transformer._latentslate_compute_dtype = compute_dtype
        transformer._latentslate_klein_config_fingerprint = plan.config_fingerprint
        transformer._latentslate_klein_mapping_fingerprint = plan.mapping_fingerprint
        transformer._latentslate_klein_artifact_identity = plan.identity
        return transformer
    except BaseException:
        _dematerialize(transformer)
        # Do not retain any restored payload through traceback locals after a
        # late failure in a multi-gigabyte real checkpoint.
        qdata = qdata_parts = qdata_part = None
        weight_scale = input_scale = quantized_weight = dense = None
        raise


def materialize_klein_nvfp4_transformer(
    plan: KleinStoredAdapterPlan,
    config: Mapping[str, Any] = KLEIN4B_CONFIG,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    execution_device: torch.device | str = "cuda",
) -> nn.Module:
    """Restore exact packed BFL NVFP4 storage without dequantizing weights."""

    from safetensors import safe_open

    plan.require_available()
    if plan.artifact_contract != KLEIN_STORED_NVFP4_CONTRACT:
        raise ValueError("Klein NVFP4 materializer: unsupported artifact contract")
    if compute_dtype is not torch.bfloat16:
        raise ValueError("Klein NVFP4 materializer requires BF16 compute")
    if plan.config_fingerprint != _fingerprint(dict(config)):
        raise ValueError("Klein NVFP4 materializer: config differs from validated plan")
    _require_nvfp4_cuda_backend(execution_device)
    _validate_nvfp4_materializer_plan(plan)
    transformer = build_klein_transformer_skeleton(config)
    expected_targets = set(transformer.state_dict())
    consumed_sources: set[str] = set()
    consumed_targets: set[str] = set()
    consumed_auxiliary: set[str] = set()
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            if not revalidate_artifact(plan.identity):
                raise ValueError("Klein NVFP4 artifact changed before materialization")
            _validate_bound_nvfp4_header(handle, plan)
            for source in plan.quantized_sources:
                targets = plan.source_to_targets[source]
                stem = source.removesuffix(".weight")
                qdata = handle.get_tensor(source)
                block_scale = handle.get_tensor(stem + ".weight_scale")
                tensor_scale = handle.get_tensor(stem + ".weight_scale_2")
                input_scale = handle.get_tensor(stem + ".input_scale")
                logical_shape = (qdata.shape[0], qdata.shape[1] * 2)
                _validate_stored_nvfp4_payload(
                    source, qdata, block_scale, tensor_scale, input_scale
                )
                row_counts = tuple(
                    int(transformer.get_submodule(target.rpartition(".")[0]).weight.shape[0])
                    for target in targets
                )
                qparts = (qdata,) if len(targets) == 1 else torch.split(qdata, row_counts, 0)
                sparts = (
                    (block_scale,)
                    if len(targets) == 1
                    else torch.split(block_scale, row_counts, 0)
                )
                if sum(row_counts) != logical_shape[0]:
                    raise ValueError("Klein NVFP4 fused row split is incomplete")
                for target, qpart, spart in zip(targets, qparts, sparts, strict=True):
                    parent_path, _, leaf = target.rpartition(".")
                    module = transformer.get_submodule(parent_path)
                    if leaf != "weight" or type(module) is not nn.Linear or module.bias is not None:
                        raise TypeError(f"Klein NVFP4 target {target!r} is not bias-free Linear")
                    weight = _restore_nvfp4_tensor(
                        qpart,
                        spart,
                        tensor_scale,
                        (qpart.shape[0], qpart.shape[1] * 2),
                        compute_dtype,
                    )
                    _replace_linear(
                        transformer,
                        parent_path,
                        KleinStoredNVFP4Linear(weight, input_scale=input_scale),
                    )
                    consumed_targets.add(target)
                consumed_sources.add(source)
                consumed_auxiliary.update(stem + suffix for suffix in _NVFP4_SUFFIXES)

            for source in plan.dense_sources:
                targets = plan.source_to_targets[source]
                dense = handle.get_tensor(source)
                if source == "final_layer.adaLN_modulation.1.weight":
                    dense = _swap_adaln_scale_shift(dense)
                row_counts = tuple(
                    int(transformer.get_submodule(target.rpartition(".")[0]).weight.shape[0])
                    for target in targets
                )
                dense_parts = (
                    (dense,)
                    if len(targets) == 1
                    else torch.split(dense, row_counts, dim=0)
                )
                if len(dense_parts) != len(targets):
                    raise ValueError("Klein NVFP4 fused dense split is incomplete")
                for target, dense_part in zip(targets, dense_parts, strict=True):
                    _assign_dense_target(transformer, target, dense_part)
                    consumed_targets.add(target)
                consumed_sources.add(source)
        if consumed_sources != set(plan.source_to_targets):
            raise ValueError("Klein NVFP4 source consumption is incomplete")
        if consumed_targets != expected_targets:
            raise ValueError("Klein NVFP4 target materialization is incomplete")
        if consumed_auxiliary != set(plan.auxiliary_sources):
            raise ValueError("Klein NVFP4 sidecar consumption is incomplete")
        _validate_materialized_transformer(transformer)
        transformer._latentslate_compute_dtype = compute_dtype
        transformer._latentslate_klein_config_fingerprint = plan.config_fingerprint
        transformer._latentslate_klein_mapping_fingerprint = plan.mapping_fingerprint
        transformer._latentslate_klein_artifact_identity = plan.identity
        transformer._latentslate_klein_quantization_contract = KLEIN_STORED_NVFP4_CONTRACT
        transformer._latentslate_klein_native_backend = "comfy-kitchen/cuda/scaled_mm_nvfp4"
        expected_module_count = sum(
            len(plan.source_to_targets[source]) for source in plan.quantized_sources
        )
        native_modules = tuple(
            name
            for name, module in transformer.named_modules()
            if isinstance(module, KleinStoredNVFP4Linear)
        )
        if len(native_modules) != expected_module_count:
            raise RuntimeError("Klein NVFP4 materialized module count differs from its plan")
        transformer._latentslate_klein_nvfp4_modules = native_modules
        return transformer
    except BaseException:
        _dematerialize(transformer)
        raise


class KleinStoredLinear(nn.Module):
    """Bias-free linear backed by official stored FP8 qdata and scalar scales.

    Transformer checkpoints carry a fixed activation scale. Comfy's mixed Qwen
    checkpoint intentionally omits it and quantizes activations dynamically;
    ``input_scale=None`` represents that exact second contract.
    """

    def __init__(self, weight, *, input_scale: torch.Tensor | None) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorCoreFP8Layout"
            or weight._qdata.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("KleinStoredLinear requires stored TensorCore FP8 weight data")
        if input_scale is not None:
            _validate_positive_scalar(input_scale, "input_scale")
        self.weight = nn.Parameter(weight, requires_grad=False)
        # Python storage prevents an ancestor dtype cast from corrupting the
        # authoritative F32 activation scale.
        self.input_scale = None if input_scale is None else float(input_scale.item())
        self.native_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("KleinStoredLinear input feature count does not match weight")
        original_shape = input.shape
        flat_input = input.reshape(-1, original_shape[-1])
        if self.input_scale is None:
            import comfy_kitchen as ck
            from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

            if flat_input.device.type != "cuda":
                raise RuntimeError("Klein dynamic FP8 dispatch requires CUDA input")
            scale = torch.amax(flat_input.abs()).to(dtype=torch.float32)
            scale = torch.clamp(scale / torch.finfo(torch.float8_e4m3fn).max, min=1e-12)
            with ck.use_backend("cuda"):
                quantize = ck.registry.get_implementation(
                    "quantize_per_tensor_fp8", backend="cuda"
                )
                qdata = quantize(flat_input, scale, torch.float8_e4m3fn)
                output = scaled_mm_v2(
                    qdata,
                    self.weight._qdata.t(),
                    scale,
                    self.weight.params.scale,
                    out_dtype=input.dtype,
                )
            self.native_dispatch_count += 1
        else:
            activation = _quantize_fp8_activation(flat_input, self.input_scale)
            output = F.linear(activation, self.weight)
        return output.reshape(*original_shape[:-1], self.weight.shape[0])

    def move_stored_storage(self, device: torch.device | str) -> None:
        """Move FP8 qdata and its F32 scale together without changing either."""

        from comfy_kitchen.tensor import QuantizedTensor

        target = _canonical_device(torch.device(device))
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("KleinStoredLinear weight is no longer a QuantizedTensor")
        if weight._qdata.dtype is not torch.float8_e4m3fn:
            raise RuntimeError("KleinStoredLinear qdata dtype changed")
        params = dataclass_replace(weight.params, scale=weight.params.scale.to(device=target))
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        self._parameters["weight"] = nn.Parameter(restored, requires_grad=False)


class KleinStoredNVFP4Linear(nn.Module):
    """Bias-free Linear that permits only Kitchen's native CUDA NVFP4 kernels."""

    def __init__(self, weight, *, input_scale: torch.Tensor | None) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorCoreNVFP4Layout"
            or weight._qdata.dtype is not torch.uint8
            or weight.params.block_scale.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("KleinStoredNVFP4Linear requires packed TensorCore NVFP4")
        if input_scale is not None:
            _validate_positive_scalar(input_scale, "input_scale")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.input_scale = None if input_scale is None else float(input_scale.item())
        self.native_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

        if input.device.type != "cuda":
            raise RuntimeError("Klein NVFP4 native dispatch requires CUDA input")
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("Klein NVFP4 input feature count differs from weight")
        original_shape = input.shape
        flat = input.reshape(-1, original_shape[-1])
        if self.input_scale is None:
            scale = torch.amax(flat.abs()).to(dtype=torch.float32)
            scale = torch.clamp(scale / (448.0 * 6.0), min=1e-12)
        else:
            scale = torch.tensor(self.input_scale, device=input.device, dtype=torch.float32)
        # Explicit backend pinning plus direct kernel invocation means Kitchen's
        # QuantizedTensor catch-and-dequantize fallback is never in this path.
        padded = TensorCoreNVFP4Layout.get_padded_shape(tuple(flat.shape)) != tuple(flat.shape)
        with ck.use_backend("cuda"):
            quantize = ck.registry.get_implementation("quantize_nvfp4", backend="cuda")
            native_mm = ck.registry.get_implementation("scaled_mm_nvfp4", backend="cuda")
            aqdata, block_scale_a = quantize(flat, scale, pad_16x=padded)
            if aqdata.dtype is not torch.uint8:
                raise RuntimeError("Klein NVFP4 activation did not remain packed U8")
            weight = self.weight
            if not isinstance(weight, QuantizedTensor):
                raise TypeError("Klein NVFP4 weight lost its QuantizedTensor wrapper")
            result = native_mm(
                aqdata,
                weight._qdata,
                tensor_scale_a=scale,
                tensor_scale_b=weight.params.scale,
                block_scale_a=block_scale_a,
                block_scale_b=weight.params.block_scale,
                out_dtype=input.dtype,
            )
        result = result[: flat.shape[0], : self.weight.shape[0]]
        self.native_dispatch_count += 1
        return result.reshape(*original_shape[:-1], self.weight.shape[0])


def move_klein_transformer_storage(
    transformer: nn.Module,
    device: torch.device | str,
) -> None:
    """Move a materialized Klein transformer and prove exact stored-state residency.

    Ancestor ``Module.to`` behavior is not sufficient evidence for a third-party
    tensor subclass: after the logical move, every stored linear is rebuilt from
    the same FP8 bytes and F32 scale on the target device and then checked.
    """

    poisoned = getattr(transformer, "_latentslate_klein_residency_poisoned", None)
    if poisoned:
        raise RuntimeError(f"Klein transformer residency is poisoned: {poisoned}")
    if not hasattr(transformer, "_latentslate_klein_artifact_identity"):
        raise ValueError("Klein transformer is not bound to a materialized artifact plan")
    target = _canonical_device(torch.device(device))
    before = _state_values(transformer)
    dtypes = {name: value.dtype for name, value in before.items()}
    try:
        move_klein_module_storage(transformer, target)
        after = _state_values(transformer)
        if set(after) != set(dtypes):
            raise RuntimeError("Klein transformer state changed during residency move")
        for name, value in after.items():
            if value.is_meta or value.dtype != dtypes[name] or value.device != target:
                raise RuntimeError(f"Klein transformer residency mismatch for {name!r}")
        for name, module in transformer.named_modules():
            if not isinstance(module, (KleinStoredLinear, KleinStoredNVFP4Linear)):
                continue
            _assert_stored_weight_device(module.weight, target, name)
    except BaseException as exc:
        transformer._latentslate_klein_residency_poisoned = str(exc)
        raise


def move_klein_module_storage(module: nn.Module, device: torch.device | str) -> None:
    """Move one Klein movement group without delegating QuantizedTensor to ``_apply``.

    PyTorch's wrapper-subclass swap path can reject a live Comfy Kitchen
    parameter during exception cleanup.  Detaching stored parameters before the
    ordinary module move and rebuilding them from their exact qdata/scales makes
    the third-party boundary explicit and deterministic.
    """

    target = _canonical_device(torch.device(device))
    stored: list[tuple[nn.Module, Any]] = []
    for nested in module.modules():
        if isinstance(nested, (KleinStoredLinear, KleinStoredNVFP4Linear)):
            weight = nested.weight
            stored.append((nested, weight))
            nested._parameters["weight"] = None
    try:
        module.to(device=target)
    except BaseException:
        for nested, weight in stored:
            nested._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
        raise
    for nested, weight in stored:
        from comfy_kitchen.tensor import QuantizedTensor

        params = weight.params.to_device(target)
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        nested._parameters["weight"] = nn.Parameter(restored, requires_grad=False)
        if (
            restored._qdata.device != target
            or restored.params.scale.device != target
            or restored._qdata.dtype is not weight._qdata.dtype
            or restored._layout_cls != weight._layout_cls
        ):
            raise RuntimeError("Klein grouped move changed stored quantized identity")
        _assert_stored_weight_device(restored, target, "grouped storage")


class KleinTransformerResidencySession:
    """One-shot, Engine-owned whole-transformer stored-FP8 residency.

    Klein's quantized linears own FP8 bytes outside normal ``Module.to``
    semantics.  This session is therefore the only whole-transformer boundary:
    it synchronously moves every state value and physical FP8 payload to one
    exact execution device, observes the outer transformer forward, then
    synchronizes and returns all state to CPU.  It intentionally installs no
    Accelerate or Diffusers hooks.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        onload_device: torch.device | str,
        offload_device: torch.device | str = "cpu",
        lazy_onload: bool = False,
        residency_mode: str = "adaptive",
        require_partial: bool = False,
    ) -> None:
        poisoned = getattr(transformer, "_latentslate_klein_residency_poisoned", None)
        if poisoned:
            raise RuntimeError(f"Klein transformer residency is poisoned: {poisoned}")
        if not hasattr(transformer, "_latentslate_klein_artifact_identity"):
            raise ValueError("Klein residency requires a materialized artifact-bound transformer")
        self.transformer = transformer
        self.onload_device = _canonical_device(torch.device(onload_device))
        self.offload_device = _canonical_device(torch.device(offload_device))
        self.lazy_onload = lazy_onload
        if residency_mode not in {"adaptive", "full", "grouped"}:
            raise ValueError("Klein residency mode must be adaptive, full, or grouped")
        self.residency_mode = residency_mode
        self.require_partial = require_partial
        self._decision: ResidencyDecision | None = None
        self._group_handles: list[Any] = []
        self._active_group: str | None = None
        self._resident_groups: tuple[str, ...] = ()
        self._streamed_groups: tuple[str, ...] = ()
        self._group_sizes: dict[str, int] = {}
        self._root_bytes = 0
        self._onloaded = False
        if self.offload_device.type != "cpu":
            raise ValueError("Klein transformer residency requires CPU as the offload device")
        self._dtype_snapshot = self._snapshot_state()
        self._assert_devices(self.offload_device)
        self._entered = False
        self._closed = False
        self._owner_thread_id: int | None = None
        self._execution_thread_id: int | None = None
        self._execution_depth = 0
        self._execution_handles: list[Any] = []
        self._lock = threading.RLock()

    @property
    def active(self) -> bool:
        """Whether the session currently owns transformer residency."""

        return self._entered and not self._closed

    @property
    def device(self) -> torch.device:
        """The canonical execution device selected for this one-shot session."""

        return self.onload_device

    @property
    def policy(self) -> dict[str, int | str]:
        """Effective residency policy, suitable for job provenance."""

        if self._decision is None:
            return {"mode": self.residency_mode, "reason": "not activated"}
        policy = self._decision.provenance()
        policy.update(
            {
                "root_bytes": self._root_bytes,
                "resident_block_count": len(self._resident_groups),
                "resident_block_bytes": sum(
                    self._group_sizes[name] for name in self._resident_groups
                ),
                "streamed_block_count": len(self._streamed_groups),
                "streamed_block_bytes": sum(
                    self._group_sizes[name] for name in self._streamed_groups
                ),
            }
        )
        return policy

    def __enter__(self) -> Self:
        with self._lock:
            if self._closed or self._entered:
                raise RuntimeError(
                    "Klein transformer residency session is one-shot and cannot re-enter"
                )
            self._claim_global()
            try:
                self._owner_thread_id = threading.get_ident()
                self._validate_state()
                self._assert_devices(self.offload_device)
                self._attach_execution_tracking()
                self._entered = True
                if not self.lazy_onload:
                    self._onload()
                return self
            except BaseException:
                self._teardown(suppress_errors=True, require_idle=False)
                raise

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        if self._closed:
            return False
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "Klein transformer residency context exited from a non-owning thread"
            )
        self._teardown(suppress_errors=False, require_idle=True)
        return False

    def close(self) -> None:
        """Synchronously return full transformer storage to CPU and release ownership."""

        with self._lock:
            if self._closed:
                return
            if threading.get_ident() != self._owner_thread_id:
                raise RuntimeError(
                    "Klein transformer residency close must run on the owning context thread"
                )
            if self._is_executing():
                raise RuntimeError(
                    "cannot close Klein transformer residency while a forward is active"
                )
            self._teardown(suppress_errors=False, require_idle=True)

    def _claim_global(self) -> None:
        global _ACTIVE_KLEIN_SESSION
        with _KLEIN_SESSION_GUARD_LOCK:
            if _ACTIVE_KLEIN_SESSION is not None:
                raise RuntimeError(
                    "a Klein transformer residency session is already active process-wide"
                )
            _ACTIVE_KLEIN_SESSION = self

    def _release_global(self) -> None:
        global _ACTIVE_KLEIN_SESSION
        with _KLEIN_SESSION_GUARD_LOCK:
            if _ACTIVE_KLEIN_SESSION is self:
                _ACTIVE_KLEIN_SESSION = None

    def _snapshot_state(self) -> dict[str, torch.dtype]:
        state = _state_values(self.transformer)
        if not state:
            raise ValueError("Klein transformer residency requires materialized state")
        if any(value.is_meta for value in state.values()):
            raise ValueError("Klein transformer residency cannot accept meta state")
        return {name: value.dtype for name, value in state.items()}

    def _validate_state(self) -> None:
        state = _state_values(self.transformer)
        if set(state) != set(self._dtype_snapshot):
            raise RuntimeError(
                "Klein transformer residency state changed after session construction"
            )
        for name, value in state.items():
            if value.is_meta or value.dtype != self._dtype_snapshot[name]:
                raise RuntimeError(f"Klein transformer residency state changed for {name!r}")

    def _assert_devices(self, requested: torch.device) -> None:
        state = _state_values(self.transformer)
        for name, value in state.items():
            if value.device != requested:
                raise RuntimeError(
                    f"Klein transformer residency state is on the wrong device: {name!r}"
                )
        for name, module in self.transformer.named_modules():
            if not isinstance(module, (KleinStoredLinear, KleinStoredNVFP4Linear)):
                continue
            _assert_stored_weight_device(module.weight, requested, name)

    def _attach_execution_tracking(self) -> None:
        if self._execution_handles:
            raise RuntimeError("Klein transformer residency execution tracking is already attached")

        def pre_hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            with self._lock:
                thread_id = threading.get_ident()
                if not self.active or thread_id != self._owner_thread_id:
                    raise RuntimeError(
                        "Klein transformer forward requires its active owning residency session"
                    )
                if self._execution_thread_id not in {None, thread_id}:
                    raise RuntimeError("Klein transformer forward crossed residency threads")
                if not self._onloaded:
                    self._onload()
                self._execution_thread_id = thread_id
                self._execution_depth += 1

        def post_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            with self._lock:
                if self._execution_depth <= 0:
                    # ``always_call=True`` also invokes this hook when our
                    # pre-hook rejects a non-owner thread before it records an
                    # execution. In that case there is nothing to unwind.
                    return output
                self._execution_depth -= 1
                if self._execution_depth == 0:
                    self._execution_thread_id = None
            return output

        self._execution_handles = [
            self.transformer.register_forward_pre_hook(pre_hook),
            self.transformer.register_forward_hook(post_hook, always_call=True),
        ]

    def _onload(self) -> None:
        self._decision = self._choose_policy()
        if self._decision.mode == "full":
            move_klein_transformer_storage(self.transformer, self.onload_device)
            self._assert_devices(self.onload_device)
        else:
            self._attach_grouped_residency()
        self._onloaded = True

    def _choose_policy(self) -> ResidencyDecision:
        groups = self._group_modules()
        self._group_sizes = {
            name: _physical_state_bytes(block) for name, block in groups.items()
        }
        stored_bytes = _physical_state_bytes(self.transformer)
        self._root_bytes = stored_bytes - sum(self._group_sizes.values())
        if self._root_bytes < 0:
            raise RuntimeError("Klein residency physical byte accounting overlaps blocks")
        largest_group = max(self._group_sizes.values())
        if self.onload_device.type != "cuda":
            mode = "full" if self.residency_mode == "adaptive" else self.residency_mode
            budget = stored_bytes if mode == "full" else self._root_bytes + largest_group
            return ResidencyDecision(
                mode=mode,
                free_bytes=0,
                total_bytes=0,
                stored_bytes=stored_bytes,
                reserved_headroom_bytes=0,
                stream_buffer_bytes=largest_group if mode == "grouped" else 0,
                resident_weight_budget_bytes=budget,
                reason="non-CUDA test residency",
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.onload_device)
        decision = choose_cuda_residency(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            stored_bytes=stored_bytes,
            largest_group_bytes=largest_group,
        )
        if self.residency_mode != "adaptive":
            explicit_grouped_budget = min(
                stored_bytes - largest_group,
                max(
                    0,
                    decision.free_bytes
                    - decision.reserved_headroom_bytes
                    - largest_group,
                ),
            )
            decision = ResidencyDecision(
                mode=self.residency_mode,
                free_bytes=decision.free_bytes,
                total_bytes=decision.total_bytes,
                stored_bytes=decision.stored_bytes,
                reserved_headroom_bytes=decision.reserved_headroom_bytes,
                stream_buffer_bytes=(
                    largest_group if self.residency_mode == "grouped" else 0
                ),
                resident_weight_budget_bytes=(
                    explicit_grouped_budget
                    if self.residency_mode == "grouped"
                    else stored_bytes
                ),
                reason="explicit Engine residency override",
            )
        if self.require_partial:
            decision = require_grouped_residency(
                decision,
                largest_group_bytes=largest_group,
                reason=(
                    "stored-FP8 image conditioning requires partial residency until "
                    "a workload-aware activation estimate is proven"
                ),
            )
        return decision

    def _group_modules(self) -> dict[str, nn.Module]:
        groups: dict[str, nn.Module] = {}
        for list_name in ("transformer_blocks", "single_transformer_blocks"):
            block_list = getattr(self.transformer, list_name, None)
            if not isinstance(block_list, nn.ModuleList) or not block_list:
                raise RuntimeError(f"Klein grouped residency lacks {list_name}")
            groups.update({f"{list_name}.{index}": block for index, block in enumerate(block_list)})
        return groups

    @staticmethod
    def _move_module(module: nn.Module, device: torch.device) -> None:
        move_klein_module_storage(module, device)

    def _attach_grouped_residency(self) -> None:
        groups = self._group_modules()
        if self._decision is None:
            raise RuntimeError("Klein grouped residency lacks a budget decision")
        budget = self._decision.resident_weight_budget_bytes
        if self._root_bytes > budget:
            import logging

            diagnostics = {
                **self._decision.provenance(),
                "root_bytes": self._root_bytes,
                "largest_group_bytes": max(self._group_sizes.values()),
                "torch_allocated_bytes": (
                    int(torch.cuda.memory_allocated(self.onload_device))
                    if self.onload_device.type == "cuda"
                    else 0
                ),
                "torch_reserved_bytes": (
                    int(torch.cuda.memory_reserved(self.onload_device))
                    if self.onload_device.type == "cuda"
                    else 0
                ),
            }
            logging.getLogger(__name__).error(
                "Klein residency root budget failure: %s", diagnostics
            )
            raise RuntimeError(
                "Klein residency budget cannot retain required root state: "
                + ", ".join(f"{key}={value}" for key, value in diagnostics.items())
            )
        resident_bytes = self._root_bytes
        resident: set[str] = set()
        # Every block executes once per transformer traversal. Prioritizing the
        # largest groups avoids the greatest repeated PCIe traffic for a fixed
        # resident byte budget, with the name providing a stable tie break.
        for name in sorted(groups, key=lambda item: (-self._group_sizes[item], item)):
            size = self._group_sizes[name]
            if resident_bytes + size <= budget:
                resident.add(name)
                resident_bytes += size
        self._resident_groups = tuple(name for name in groups if name in resident)
        self._streamed_groups = tuple(name for name in groups if name not in resident)
        if not self._streamed_groups:
            raise RuntimeError("Klein grouped residency selected no streamed blocks")
        group_ids = {id(module) for module in groups.values()}
        try:
            for _name, child in self.transformer.named_children():
                if isinstance(child, nn.ModuleList) and all(
                    id(item) in group_ids for item in child
                ):
                    continue
                self._move_module(child, self.onload_device)
            for name, block in groups.items():
                if name in resident:
                    self._move_module(block, self.onload_device)
                    continue
                self._move_module(block, self.offload_device)

                def pre_hook(
                    module: nn.Module, _args: tuple[Any, ...], group_name=name
                ) -> None:
                    if self._active_group is not None:
                        raise RuntimeError("Klein grouped residency is non-reentrant")
                    self._active_group = group_name
                    try:
                        self._move_module(module, self.onload_device)
                    except BaseException as exc:
                        self.transformer._latentslate_klein_residency_poisoned = (
                            f"Klein grouped residency onload failed for {group_name}: {exc}"
                        )
                        raise

                def post_hook(
                    module: nn.Module, _args: tuple[Any, ...], output: Any, group_name=name
                ) -> Any:
                    if self._active_group != group_name:
                        # ``always_call`` may observe a rejected pre-hook. Never
                        # disturb a different active group in that case.
                        return output
                    try:
                        self._move_module(module, self.offload_device)
                    except BaseException as exc:
                        self.transformer._latentslate_klein_residency_poisoned = (
                            f"Klein grouped residency offload failed for {group_name}: {exc}"
                        )
                        raise
                    finally:
                        self._active_group = None
                    return output

                self._group_handles.append(block.register_forward_pre_hook(pre_hook))
                self._group_handles.append(
                    block.register_forward_hook(post_hook, always_call=True)
                )
        except BaseException as setup_error:
            self._rollback_grouped_setup(setup_error)
            raise

    def _rollback_grouped_setup(self, setup_error: BaseException) -> None:
        """Synchronously undo a partially installed grouped-residency plan."""

        cleanup_errors: list[BaseException] = []
        handles, self._group_handles = self._group_handles, []
        for handle in handles:
            try:
                handle.remove()
            except BaseException as exc:  # noqa: BLE001 - setup must fail closed
                cleanup_errors.append(exc)
        barrier_succeeded = True
        if self.onload_device.type == "cuda":
            try:
                torch.cuda.synchronize(self.onload_device)
            except BaseException as exc:  # noqa: BLE001 - CUDA storage is now uncertain
                barrier_succeeded = False
                cleanup_errors.append(exc)
                self.transformer._latentslate_klein_residency_poisoned = (
                    f"Klein grouped setup rollback barrier failed after {setup_error}: {exc}"
                )
        if barrier_succeeded:
            try:
                for child in self.transformer.children():
                    self._move_module(child, self.offload_device)
                self._assert_devices(self.offload_device)
                self._validate_state()
            except BaseException as exc:  # noqa: BLE001 - partial CPU cleanup is unsafe
                cleanup_errors.append(exc)
                self.transformer._latentslate_klein_residency_poisoned = (
                    f"Klein grouped setup rollback failed after {setup_error}: {exc}"
                )
        self._active_group = None
        if cleanup_errors:
            if not getattr(
                self.transformer, "_latentslate_klein_residency_poisoned", None
            ):
                self.transformer._latentslate_klein_residency_poisoned = (
                    f"Klein grouped setup rollback was incomplete after {setup_error}: "
                    f"{cleanup_errors[0]}"
                )
            raise RuntimeError(
                f"Klein grouped residency setup failed and rollback was incomplete: "
                f"{cleanup_errors[0]}"
            ) from setup_error

    def _remove_grouped_residency(self, *, move_to_cpu: bool) -> None:
        handles, self._group_handles = self._group_handles, []
        for handle in handles:
            handle.remove()
        if move_to_cpu:
            for child in self.transformer.children():
                self._move_module(child, self.offload_device)
            self._assert_devices(self.offload_device)

    def _remove_execution_tracking(self) -> None:
        handles, self._execution_handles = self._execution_handles, []
        for handle in handles:
            handle.remove()

    def _is_executing(self) -> bool:
        with self._lock:
            return self._execution_depth != 0

    def _teardown(self, *, suppress_errors: bool, require_idle: bool) -> None:
        if require_idle and self._is_executing():
            # A rejected public close must leave the session fully usable; do
            # not release the global guard or alter device state in this case.
            raise RuntimeError(
                "cannot teardown Klein transformer residency while a forward is active"
            )
        error: BaseException | None = None
        barrier_succeeded = True
        try:
            if self._onloaded and self.onload_device.type == "cuda":
                try:
                    # qdata/scale wrappers are reconstructed on CPU below.  Do
                    # not replace their CUDA storage until every kernel is done.
                    torch.cuda.synchronize(self.onload_device)
                except BaseException as exc:  # noqa: BLE001 - fail closed on CUDA loss
                    barrier_succeeded = False
                    error = exc
                    self.transformer._latentslate_klein_residency_poisoned = (
                        f"Klein CUDA residency teardown barrier failed: {exc}"
                    )
            if barrier_succeeded:
                self._remove_execution_tracking()
                if self._onloaded:
                    if self._decision is not None and self._decision.mode == "grouped":
                        self._remove_grouped_residency(move_to_cpu=True)
                    else:
                        move_klein_transformer_storage(self.transformer, self.offload_device)
                self._assert_devices(self.offload_device)
                self._validate_state()
        except BaseException as exc:  # noqa: BLE001 - preserve original teardown fault
            error = error or exc
        finally:
            if not barrier_succeeded:
                # Keep all current CUDA allocations intact.  A failed barrier
                # means rebuilding CPU wrappers could race unfinished kernels.
                self._remove_execution_tracking()
                self._remove_grouped_residency(move_to_cpu=False)
            self._entered = False
            self._closed = True
            self._release_global()
        if error is not None and not suppress_errors:
            raise RuntimeError(f"Klein transformer residency teardown failed: {error}") from error


def _state_values(module: nn.Module) -> dict[str, torch.Tensor]:
    values = dict(module.named_parameters())
    values.update(module.named_buffers())
    return values


def _physical_state_bytes(module: nn.Module) -> int:
    """Count authoritative storage, including Kitchen qdata and F32 scales."""

    total = 0
    for value in _state_values(module).values():
        qdata = getattr(value, "_qdata", None)
        params = getattr(value, "params", None)
        scale = getattr(params, "scale", None)
        if isinstance(qdata, torch.Tensor) and isinstance(scale, torch.Tensor):
            total += qdata.numel() * qdata.element_size()
            for field in value.params._tensor_fields():
                sidecar = getattr(value.params, field)
                total += sidecar.numel() * sidecar.element_size()
        else:
            total += value.numel() * value.element_size()
    return total


def _assert_stored_weight_device(weight: Any, target: torch.device, name: str) -> None:
    if weight._qdata.device != target or weight.params.scale.device != target:
        raise RuntimeError(f"Klein physical quantized state is on the wrong device: {name!r}")
    if weight.params.scale.dtype is not torch.float32:
        raise RuntimeError(f"Klein tensor scale precision changed: {name!r}")
    if weight._layout_cls == "TensorCoreFP8Layout":
        if weight._qdata.dtype is not torch.float8_e4m3fn:
            raise RuntimeError(f"Klein FP8 storage precision changed: {name!r}")
        return
    if weight._layout_cls == "TensorCoreNVFP4Layout":
        if (
            weight._qdata.dtype is not torch.uint8
            or weight.params.block_scale.device != target
            or weight.params.block_scale.dtype is not torch.float8_e4m3fn
        ):
            raise RuntimeError(f"Klein NVFP4 physical storage changed: {name!r}")
        return
    raise RuntimeError(f"Klein stored layout is unsupported: {name!r}")


def _canonical_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _validate_materializer_plan(plan: KleinStoredAdapterPlan) -> None:
    sources = set(plan.source_to_targets)
    quantized = set(plan.quantized_sources)
    dense = set(plan.dense_sources)
    if quantized & dense or quantized | dense != sources:
        raise ValueError("Klein materializer: stored source roles differ from validated plan")
    expected_auxiliary = {
        source.removesuffix(".weight") + suffix
        for source in quantized
        for suffix in _QUANT_SUFFIXES
    }
    if set(plan.auxiliary_sources) != expected_auxiliary:
        raise ValueError("Klein materializer: quant auxiliary roles differ from validated plan")
    expected_fingerprint = _fingerprint(
        {
            "config": plan.config_fingerprint,
            "contract": KLEIN_STORED_FP8_CONTRACT,
            "mapping": dict(plan.source_to_targets),
            "quantized": tuple(sorted(quantized)),
            "dense": {source: "BF16" for source in dense},
            "auxiliary": tuple(sorted(expected_auxiliary)),
        }
    )
    if plan.mapping_fingerprint != expected_fingerprint:
        raise ValueError("Klein materializer: mapping fingerprint differs from validated plan")


def _validate_nvfp4_materializer_plan(plan: KleinStoredAdapterPlan) -> None:
    sources = set(plan.source_to_targets)
    quantized = set(plan.quantized_sources)
    dense = set(plan.dense_sources)
    expected_auxiliary = {
        source.removesuffix(".weight") + suffix
        for source in quantized
        for suffix in _NVFP4_SUFFIXES
    }
    if quantized & dense or quantized | dense != sources:
        raise ValueError("Klein NVFP4 source roles differ from the validated plan")
    if set(plan.auxiliary_sources) != expected_auxiliary:
        raise ValueError("Klein NVFP4 sidecar roles differ from the validated plan")
    expected = _fingerprint(
        {
            "config": plan.config_fingerprint,
            "contract": KLEIN_STORED_NVFP4_CONTRACT,
            "mapping": dict(plan.source_to_targets),
            "quantized": tuple(sorted(quantized)),
            "dense": {source: "BF16" for source in dense},
            "auxiliary": tuple(sorted(expected_auxiliary)),
        }
    )
    if plan.mapping_fingerprint != expected:
        raise ValueError("Klein NVFP4 mapping fingerprint differs from validated plan")


def _require_nvfp4_cuda_backend(device: torch.device | str) -> None:
    import comfy_kitchen as ck

    target = _canonical_device(torch.device(device))
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Klein NVFP4 requires a configured CUDA device")
    cuda_version = getattr(torch.version, "cuda", None)
    try:
        cuda_major = int(str(cuda_version).split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Klein NVFP4 requires a CUDA 13.x Torch build") from exc
    if cuda_major < 13:
        raise RuntimeError("Klein NVFP4 requires a CUDA 13.x Torch build")
    capability = tuple(torch.cuda.get_device_capability(target))
    if capability < (10, 0):
        raise RuntimeError(f"Klein NVFP4 requires SM >= 10.0, found {capability}")
    backend = ck.list_backends().get("cuda", {})
    if (
        not backend.get("available")
        or backend.get("disabled")
        or "quantize_nvfp4" not in backend.get("capabilities", ())
        or "scaled_mm_nvfp4" not in backend.get("capabilities", ())
    ):
        raise RuntimeError("Klein NVFP4 requires Kitchen CUDA quantize and matmul kernels")


def _validate_bound_header(handle, plan: KleinStoredAdapterPlan) -> None:
    expected = set(plan.source_to_targets) | set(plan.auxiliary_sources)
    if set(handle.keys()) != expected:
        raise ValueError("Klein materializer: bound artifact tensors differ from validated plan")
    for source in plan.quantized_sources:
        view = handle.get_slice(source)
        if view.get_dtype() != "F8_E4M3" or len(view.get_shape()) != 2:
            raise ValueError("Klein materializer: stored FP8 role or geometry changed")
    for source in plan.dense_sources:
        if handle.get_slice(source).get_dtype() != "BF16":
            raise ValueError("Klein materializer: stored dense precision changed")
    for auxiliary in plan.auxiliary_sources:
        view = handle.get_slice(auxiliary)
        if view.get_dtype() != "F32" or view.get_shape() != []:
            raise ValueError("Klein materializer: stored FP8 scalar sidecar changed")


def _validate_bound_nvfp4_header(handle, plan: KleinStoredAdapterPlan) -> None:
    expected = set(plan.source_to_targets) | set(plan.auxiliary_sources)
    if set(handle.keys()) != expected:
        raise ValueError("Klein NVFP4 bound artifact differs from validated plan")
    for source in plan.quantized_sources:
        view = handle.get_slice(source)
        if view.get_dtype() != "U8" or len(view.get_shape()) != 2:
            raise ValueError("Klein NVFP4 packed weight role changed")
    for source in plan.dense_sources:
        if handle.get_slice(source).get_dtype() != "BF16":
            raise ValueError("Klein NVFP4 dense precision changed")
    for auxiliary in plan.auxiliary_sources:
        view = handle.get_slice(auxiliary)
        if auxiliary.endswith(".weight_scale"):
            if view.get_dtype() != "F8_E4M3" or len(view.get_shape()) != 2:
                raise ValueError("Klein NVFP4 block-scale role changed")
        elif view.get_dtype() != "F32" or view.get_shape() != []:
            raise ValueError("Klein NVFP4 scalar sidecar role changed")


def _restore_global_fp8_tensor(
    qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    compute_dtype: torch.dtype,
):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    if qdata.dtype is not torch.float8_e4m3fn or qdata.ndim != 2:
        raise ValueError("Klein materializer: weight qdata is not 2D FP8 E4M3FN")
    _validate_positive_scalar(weight_scale, "weight_scale")
    params = TensorCoreFP8Layout.Params(
        scale=weight_scale,
        orig_dtype=compute_dtype,
        orig_shape=tuple(qdata.shape),
    )
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _restore_nvfp4_tensor(
    qdata: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    logical_shape: tuple[int, int],
    compute_dtype: torch.dtype,
):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

    params = TensorCoreNVFP4Layout.Params(
        scale=tensor_scale,
        orig_dtype=compute_dtype,
        orig_shape=logical_shape,
        block_scale=block_scale,
    )
    return QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)


def _validate_stored_fp8_payload(
    source: str,
    qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> None:
    if qdata.dtype is not torch.float8_e4m3fn or qdata.ndim != 2:
        raise ValueError(f"Klein materializer: invalid stored FP8 qdata for {source!r}")
    _validate_positive_scalar(weight_scale, "weight_scale")
    _validate_positive_scalar(input_scale, "input_scale")


def _validate_stored_nvfp4_payload(
    source: str,
    qdata: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> None:
    if qdata.dtype is not torch.uint8 or qdata.ndim != 2 or qdata.shape[1] % 8:
        raise ValueError(f"Klein NVFP4 invalid packed weight for {source!r}")
    if (
        block_scale.dtype is not torch.float8_e4m3fn
        or tuple(block_scale.shape) != (qdata.shape[0], qdata.shape[1] // 8)
    ):
        raise ValueError(f"Klein NVFP4 invalid block scale for {source!r}")
    _validate_positive_scalar(tensor_scale, "weight_scale_2")
    _validate_positive_scalar(input_scale, "input_scale")


def _validate_positive_scalar(value: torch.Tensor, name: str) -> None:
    if (
        value.dtype is not torch.float32
        or value.ndim != 0
        or not bool(torch.isfinite(value))
        or not bool(value > 0)
    ):
        raise ValueError(f"Klein materializer: {name} must be one positive finite F32 scalar")


def _quantize_fp8_activation(input: torch.Tensor, input_scale: float):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    qdata, params = TensorCoreFP8Layout.quantize(input, scale=scale, dtype=torch.float8_e4m3fn)
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _swap_adaln_scale_shift(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[0] % 2:
        raise ValueError("Klein materializer: final AdaLN weight cannot be row-permuted")
    shift, scale = tensor.chunk(2, dim=0)
    return torch.cat((scale, shift), dim=0)


def _replace_linear(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if type(getattr(parent, attribute, None)) is not nn.Linear:
        raise TypeError(f"Klein materializer: {path!r} is not an nn.Linear")
    setattr(parent, attribute, replacement)


def _assign_dense_target(root: nn.Module, target: str, tensor: torch.Tensor) -> None:
    parent_path, separator, attribute = target.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    current = getattr(parent, attribute, None)
    if not isinstance(current, torch.Tensor) or attribute not in parent._parameters:
        raise TypeError(f"Klein materializer: {target!r} is not a parameter")
    if tuple(tensor.shape) != tuple(current.shape) or tensor.dtype is not torch.bfloat16:
        raise ValueError(f"Klein materializer: invalid BF16 dense tensor for {target!r}")
    parent._parameters[attribute] = nn.Parameter(tensor, requires_grad=False)


def _validate_materialized_transformer(transformer: nn.Module) -> None:
    meta = [name for name, value in transformer.named_parameters() if value.is_meta]
    meta.extend(name for name, value in transformer.named_buffers() if value.is_meta)
    if meta:
        raise ValueError(f"Klein materializer: meta tensors remain after restore: {meta[:3]}")


def _dematerialize(transformer: nn.Module) -> None:
    """Release all partially restored storage after any materialization failure."""

    for module in transformer.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is not None:
                module._parameters[name] = nn.Parameter(
                    torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
                    requires_grad=False,
                )
        for name, buffer in tuple(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = torch.empty(
                    tuple(buffer.shape), dtype=buffer.dtype, device="meta"
                )


def _validate_fp8_contract(
    *,
    header: Mapping[str, Any],
    metadata: Any,
    quantized: tuple[str, ...],
    dense: tuple[str, ...],
    source_to_targets: Mapping[str, tuple[str, ...]],
    skeleton,
) -> tuple[list[str], tuple[str, ...]]:
    errors: list[str] = []
    raw = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid global FP8 metadata: {exc}")
        parsed = {}
    layers = parsed.get("layers") if isinstance(parsed, dict) else None
    if not isinstance(layers, dict):
        errors.append("missing global FP8 layer metadata")
        layers = {}

    quantized_stems = {source.removesuffix(".weight") for source in quantized}
    if set(layers) != quantized_stems:
        errors.append("global FP8 metadata does not exactly match quantized layers")
    if any(
        not isinstance(value, dict) or value != {"format": "float8_e4m3fn"}
        for value in layers.values()
    ):
        errors.append("global FP8 metadata contains an unsupported layer contract")

    auxiliaries: list[str] = []
    for source in quantized:
        entry = header[source]
        targets = source_to_targets.get(source, ())
        if entry.get("dtype") != "F8_E4M3" or len(entry.get("shape", ())) != 2:
            errors.append(f"quantized source has invalid FP8 geometry: {source}")
        if not targets or not all(_target_parent_is_linear(skeleton, target) for target in targets):
            errors.append(f"quantized source does not map only to Linear weights: {source}")
        stem = source.removesuffix(".weight")
        for suffix in _QUANT_SUFFIXES:
            auxiliary = stem + suffix
            auxiliaries.append(auxiliary)
            value = header.get(auxiliary)
            if (
                not isinstance(value, dict)
                or value.get("dtype") != "F32"
                or value.get("shape") != []
            ):
                errors.append(f"FP8 layer requires a scalar F32 {suffix}: {source}")

    if not quantized:
        errors.append("artifact contains no stored FP8 weights")
    for source in dense:
        if header[source].get("dtype") != "BF16":
            errors.append(f"dense source must remain BF16: {source}")
    return errors, tuple(sorted(auxiliaries))


def _validate_nvfp4_contract(
    *,
    header: Mapping[str, Any],
    metadata: Any,
    quantized: tuple[str, ...],
    dense: tuple[str, ...],
    source_to_targets: Mapping[str, tuple[str, ...]],
    skeleton,
) -> tuple[list[str], tuple[str, ...]]:
    errors: list[str] = []
    raw = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid global NVFP4 metadata: {exc}")
        parsed = {}
    if not isinstance(parsed, dict):
        errors.append("global NVFP4 metadata must be an object")
        parsed = {}
    layers = parsed.get("layers") if isinstance(parsed, dict) else None
    if parsed.get("format_version") != "1.0" or not isinstance(layers, dict):
        errors.append("missing NVFP4 format_version 1.0 layer metadata")
        layers = {}
    stems = {source.removesuffix(".weight") for source in quantized}
    if set(layers) != stems or any(value != {"format": "nvfp4"} for value in layers.values()):
        errors.append("NVFP4 metadata does not exactly match stored quantized layers")

    auxiliaries: list[str] = []
    for source in quantized:
        entry = header[source]
        targets = source_to_targets.get(source, ())
        shape = entry.get("shape", ())
        if entry.get("dtype") != "U8" or len(shape) != 2 or shape[1] <= 0:
            errors.append(f"quantized source has invalid packed NVFP4 geometry: {source}")
            continue
        if not targets or not all(_target_parent_is_linear(skeleton, target) for target in targets):
            errors.append(f"quantized source does not map only to Linear weights: {source}")
        stem = source.removesuffix(".weight")
        expected = {
            ".weight_scale": ("F8_E4M3", [shape[0], shape[1] // 8]),
            ".weight_scale_2": ("F32", []),
            ".input_scale": ("F32", []),
        }
        if shape[1] % 8:
            errors.append(f"NVFP4 packed width is not block-aligned: {source}")
        for suffix, (dtype, expected_shape) in expected.items():
            auxiliary = stem + suffix
            auxiliaries.append(auxiliary)
            value = header.get(auxiliary)
            if (
                not isinstance(value, dict)
                or value.get("dtype") != dtype
                or value.get("shape") != expected_shape
            ):
                errors.append(f"NVFP4 layer has invalid {suffix}: {source}")
        if stem + ".pre_quant_scale" in header:
            errors.append(f"unsupported NVFP4 pre_quant_scale present: {source}")
    if not quantized:
        errors.append("artifact contains no stored NVFP4 weights")
    for source in dense:
        if header[source].get("dtype") != "BF16":
            errors.append(f"dense source must remain BF16: {source}")
    return errors, tuple(sorted(auxiliaries))


def _target_parent_is_linear(skeleton, target: str) -> bool:
    from torch import nn

    parent_path, separator, leaf = target.rpartition(".")
    if leaf != "weight":
        return False
    parent = skeleton.get_submodule(parent_path) if separator else skeleton
    return type(parent) is nn.Linear


def _source_shape_matches(
    source_shape: tuple[int, ...],
    target_shapes: tuple[tuple[int, ...], ...],
) -> bool:
    if not target_shapes or any(not shape for shape in target_shapes):
        return False
    if len(target_shapes) == 1:
        return source_shape == target_shapes[0]
    return (
        len(source_shape) == 2
        and all(len(shape) == 2 and shape[1] == source_shape[1] for shape in target_shapes)
        and sum(shape[0] for shape in target_shapes) == source_shape[0]
    )


def _read_safetensors_header(path: Path, size_bytes: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("Klein stored adapter: SafeTensors header is truncated")
        length = struct.unpack("<Q", raw_length)[0]
        if length > _MAX_HEADER_BYTES or length > size_bytes - 8:
            raise ValueError("Klein stored adapter: SafeTensors header exceeds bounds")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("Klein stored adapter: SafeTensors header is truncated")
    parsed = json.loads(raw_header, object_pairs_hook=_unique_object)
    if not isinstance(parsed, dict):
        raise TypeError("Klein stored adapter: SafeTensors header must be an object")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
