"""CPU/meta-only adapter planning for stored-quantized Comfy Wan 2.2 weights.

The mapping is intentionally local to the pinned Diffusers revision in
``pyproject.toml``.  It maps Comfy's native Wan topology to the public
``WanTransformer3DModel`` module layout without loading a checkpoint.  A future
loader may restore one stored quantized layer at a time only after this plan has
validated complete parameter coverage.

``NativeStoredLinear`` never quantizes or converts weights.  Its FP8 path may
quantize a *runtime activation* to execute a stored FP8 weight; that transient
compute operation is distinct from model-weight conversion and is never saved.

Stored quant is deliberately limited to Diffusers block-level group offload.
The Engine owns those synchronous block moves directly: Diffusers/Accelerate
group hooks must never be used because their meta reconstruction does not move a
third-party ``QuantizedTensor``'s internal storage. The Engine-owned residency
primitive has a separate tiny CUDA proof; full-model stored-quant generation
remains unproven and unavailable.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import _MAX_HEADER_BYTES, ArtifactIdentity, probe_safetensors, revalidate_artifact
from ..stored_quant import (
    StoredQuantizedLayer,
    describe_stored_layers_from_handle,
    restore_stored_quantized_tensor,
)

# This schema is the official I2V 14B config staged with the resources.  The
# model implementation is pinned by the Diffusers Git revision in pyproject.toml.
WAN22_14B_I2V_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "in_channels": 36,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 13824,
        "num_layers": 40,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
    }
)

# The official T2V-A14B experts share the Wan 14B topology with I2V except for
# their 16-channel noise-only input.  Keep this distinct from I2V so a text-only
# operation can never accidentally accept a 36-channel conditioned checkpoint.
WAN22_14B_T2V_CONFIG: Mapping[str, Any] = MappingProxyType(
    {**WAN22_14B_I2V_CONFIG, "in_channels": 16}
)

_PREFIX = "model.diffusion_model."
_QUANT_SUFFIXES = (".comfy_quant", ".weight_scale", ".scale_weight", ".scale_input")
_FOREIGN_COMPONENT_PREFIXES = ("vae.",)
_LEGACY_QUANT_SENTINELS = ("scaled_fp8",)
_SUPPORTED_ARTIFACT_CONTRACTS = frozenset(
    {
        "comfy_quant/float8_e4m3fn",
        "comfy_legacy/scaled_fp8_e4m3fn",
        "comfy_quant/int8_tensorwise_convrot",
    }
)
SUPPORTED_STORED_QUANT_OFFLOAD_MODES = frozenset({"group_block"})
BlockModules: TypeAlias = Mapping[str, nn.Module]
_WAN_SESSION_GUARD_LOCK = threading.RLock()
_ACTIVE_WAN_SESSION: object | None = None


@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    """One source parameter whose shape differs from Diffusers' meta skeleton."""

    source_key: str
    target_key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WanStoredAdapterPlan:
    """A fail-closed header-only mapping plan for one Comfy transformer artifact."""

    identity: ArtifactIdentity
    artifact_contract: str
    config_fingerprint: str
    source_to_target: Mapping[str, str]
    dense_precision_contract: str | None
    dense_source_dtypes: Mapping[str, str]
    dense_precision_errors: tuple[str, ...]
    invalid_non_linear_quant_sources: tuple[str, ...]
    mapping_fingerprint: str
    quant_auxiliary: tuple[str, ...]
    foreign_component_extras: tuple[str, ...]
    unexpected_extras: tuple[str, ...]
    missing_targets: tuple[str, ...]
    duplicate_targets: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]

    @property
    def available(self) -> bool:
        """Whether every required Diffusers parameter maps exactly once."""

        return not (
            self.unexpected_extras
            or self.missing_targets
            or self.duplicate_targets
            or self.shape_mismatches
            or self.dense_precision_errors
            or self.invalid_non_linear_quant_sources
        )

    @property
    def errors(self) -> tuple[str, ...]:
        """Short actionable reasons that prevent a future layer loader."""

        errors: list[str] = []
        if self.missing_targets:
            errors.append(f"missing Diffusers parameters: {len(self.missing_targets)}")
        if self.duplicate_targets:
            errors.append(f"duplicate mapped parameters: {len(self.duplicate_targets)}")
        if self.shape_mismatches:
            errors.append(f"parameter shape mismatches: {len(self.shape_mismatches)}")
        if self.unexpected_extras:
            errors.append(f"unrecognized non-auxiliary source keys: {len(self.unexpected_extras)}")
        errors.extend(self.dense_precision_errors)
        if self.invalid_non_linear_quant_sources:
            errors.append(
                "stored quantization sidecars attached to non-linear weights: "
                f"{len(self.invalid_non_linear_quant_sources)}"
            )
        return tuple(errors)

    def require_available(self) -> None:
        """Raise before any future tensor materialization when the plan has gaps."""

        if not self.available:
            raise ValueError("Wan stored adapter unavailable: " + "; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class WanRootResidencyPlan:
    """Explicit root components and transformer blocks for future Engine residency."""

    root_components: tuple[str, ...]
    blocks: tuple[str, ...]
    root_state: tuple[str, ...]
    block_state: Mapping[str, tuple[str, ...]]


def build_wan_transformer_skeleton(
    config: Mapping[str, Any] = WAN22_14B_I2V_CONFIG,
) -> nn.Module:
    """Construct a Diffusers Wan skeleton with all parameter storage on ``meta``."""

    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        return WanTransformer3DModel(**dict(config))


def plan_comfy_wan_transformer(
    artifact_path: Path,
    config: Mapping[str, Any] = WAN22_14B_I2V_CONFIG,
) -> WanStoredAdapterPlan:
    """Map and validate a Comfy Wan checkpoint header without reading payloads."""

    source = Path(artifact_path).resolve(strict=True)
    probe = probe_safetensors(source)
    if probe.quantization_contract not in _SUPPORTED_ARTIFACT_CONTRACTS:
        raise ValueError(
            "Wan stored adapter: unsupported artifact contract "
            f"{probe.quantization_contract!r}; a supported stored-quant artifact is required"
        )
    header = _read_safetensors_header(source, probe.identity.size_bytes)
    skeleton = build_wan_transformer_skeleton(config)
    expected_shapes = {name: tuple(value.shape) for name, value in skeleton.state_dict().items()}

    source_to_target: dict[str, str] = {}
    quant_auxiliary: list[str] = []
    foreign_component_extras: list[str] = []
    unexpected_extras: list[str] = []
    target_sources: dict[str, list[str]] = {}
    shape_mismatches: list[ShapeMismatch] = []

    for raw_key, entry in header.items():
        if raw_key == "__metadata__":
            continue
        key = _normalize_comfy_key(raw_key)
        if key in _LEGACY_QUANT_SENTINELS:
            quant_auxiliary.append(raw_key)
            continue
        if _is_quant_auxiliary(key):
            if _quant_auxiliary_target(key) is None:
                unexpected_extras.append(raw_key)
            else:
                quant_auxiliary.append(raw_key)
            continue
        if key.startswith(_FOREIGN_COMPONENT_PREFIXES):
            foreign_component_extras.append(raw_key)
            continue
        target = map_comfy_wan_parameter_key(key)
        if target is None:
            unexpected_extras.append(raw_key)
            continue
        source_to_target[raw_key] = target
        target_sources.setdefault(target, []).append(raw_key)
        expected_shape = expected_shapes.get(target)
        source_shape = _entry_shape(entry, raw_key)
        if expected_shape is None:
            unexpected_extras.append(raw_key)
        elif source_shape != expected_shape:
            shape_mismatches.append(ShapeMismatch(raw_key, target, source_shape, expected_shape))

    duplicate_targets = tuple(
        sorted(target for target, sources in target_sources.items() if len(sources) != 1)
    )
    missing_targets = tuple(sorted(set(expected_shapes) - set(target_sources)))
    dense_precision_contract, dense_source_dtypes, dense_precision_errors = (
        _plan_dense_precision_contract(
            probe.quantization_contract,
            source_to_target,
            header,
        )
    )
    invalid_non_linear_quant_sources = _non_linear_quantized_sources(
        probe.quantization_contract,
        source_to_target,
        header,
        skeleton,
    )
    mapping_fingerprint = _mapping_fingerprint(
        probe.quantization_contract,
        source_to_target,
        dense_precision_contract,
        dense_source_dtypes,
    )
    return WanStoredAdapterPlan(
        identity=probe.identity,
        artifact_contract=probe.quantization_contract,
        config_fingerprint=_config_fingerprint(config),
        source_to_target=MappingProxyType(dict(sorted(source_to_target.items()))),
        dense_precision_contract=dense_precision_contract,
        dense_source_dtypes=MappingProxyType(dict(sorted(dense_source_dtypes.items()))),
        dense_precision_errors=dense_precision_errors,
        invalid_non_linear_quant_sources=invalid_non_linear_quant_sources,
        mapping_fingerprint=mapping_fingerprint,
        quant_auxiliary=tuple(sorted(quant_auxiliary)),
        foreign_component_extras=tuple(sorted(foreign_component_extras)),
        unexpected_extras=tuple(sorted(set(unexpected_extras))),
        missing_targets=missing_targets,
        duplicate_targets=duplicate_targets,
        shape_mismatches=tuple(sorted(shape_mismatches, key=lambda item: item.source_key)),
    )


def materialize_wan_transformer(
    plan: WanStoredAdapterPlan,
    config: Mapping[str, Any],
    *,
    compute_dtype: torch.dtype,
) -> nn.Module:
    """Restore one fully validated Comfy Wan artifact into a Diffusers skeleton.

    This is intentionally a CPU-only materializer. It opens SafeTensors once,
    binds the already-planned artifact identity, and installs stored-quant linear
    weights directly as ``NativeStoredLinear`` wrappers. No whole-model dequant,
    quantization, or GPU transfer occurs here.
    """

    from safetensors import safe_open

    plan.require_available()
    if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Wan materializer: unsupported compute dtype")
    if plan.config_fingerprint != _config_fingerprint(config):
        raise ValueError("Wan materializer: config does not match the validated plan")
    transformer = build_wan_transformer_skeleton(config)
    expected_targets = set(transformer.state_dict())
    if set(plan.source_to_target.values()) != expected_targets:
        raise ValueError("Wan materializer: plan targets do not exactly match this config skeleton")
    if plan.mapping_fingerprint != _mapping_fingerprint(
        plan.artifact_contract,
        plan.source_to_target,
        plan.dense_precision_contract,
        plan.dense_source_dtypes,
    ):
        raise ValueError(
            "Wan materializer: mapped stored precision roles do not match the validated plan"
        )

    consumed_targets: set[str] = set()
    consumed_sources: set[str] = set()
    consumed_auxiliary: set[str] = set()
    quant_layers: dict[str, StoredQuantizedLayer] = {}
    quantized_weight = None
    bias = None
    replacement = None
    tensor = None
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            # The handle binds to the opened file. Revalidating the directory entry now
            # rejects a replacement before any checkpoint tensor is read.
            if not revalidate_artifact(plan.identity):
                raise ValueError(
                    "Wan materializer: artifact identity changed before materialization"
                )
            available_keys = set(handle.keys())
            if not set(plan.source_to_target).issubset(available_keys):
                raise ValueError("Wan materializer: planned source parameters are absent")
            quant_layers = _describe_plan_quant_layers(handle, plan)
            if not revalidate_artifact(plan.identity):
                raise ValueError(
                    "Wan materializer: artifact identity changed during quant descriptor discovery"
                )
            if any(layer.identity != plan.identity for layer in quant_layers.values()):
                raise ValueError("Wan materializer: quant descriptor identity does not match plan")
            _validate_authoritative_compute_dtype(handle, plan, quant_layers, compute_dtype)
            for source_key, layer in quant_layers.items():
                target_weight = plan.source_to_target.get(source_key)
                if target_weight is None or not target_weight.endswith(".weight"):
                    raise ValueError(
                        f"Wan materializer: quantized source {source_key!r} has no linear target"
                    )
                target_parent, _ = target_weight.rsplit(".", 1)
                module = transformer.get_submodule(target_parent)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Wan materializer: quantized target {target_parent!r} is not nn.Linear"
                    )
                bias_source = source_key.removesuffix(".weight") + ".bias"
                target_bias = target_parent + ".bias"
                if plan.source_to_target.get(bias_source) != target_bias:
                    raise ValueError(
                        f"Wan materializer: missing exact bias mapping for {source_key!r}"
                    )
                quantized_weight = restore_stored_quantized_tensor(handle, layer, compute_dtype)
                bias = handle.get_tensor(bias_source)
                _validate_dense_tensor(bias, target_bias, module.bias)
                scale_input_key = source_key.removesuffix(".weight") + ".scale_input"
                if scale_input_key in available_keys:
                    if plan.artifact_contract != "comfy_legacy/scaled_fp8_e4m3fn":
                        raise ValueError(
                            "Wan materializer: scale_input is only supported by the legacy FP8 contract"
                        )
                    input_scale = handle.get_tensor(scale_input_key)
                    consumed_auxiliary.add(scale_input_key)
                else:
                    input_scale = None
                replacement = NativeStoredLinear(quantized_weight, bias, input_scale)
                _replace_module(transformer, target_parent, replacement)
                consumed_sources.update({source_key, bias_source})
                consumed_targets.update({target_weight, target_bias})
                consumed_auxiliary.add(layer.scale_key)
                if layer.marker_key:
                    consumed_auxiliary.add(layer.marker_key)

            for source_key, target_key in plan.source_to_target.items():
                if source_key in consumed_sources:
                    continue
                if target_key in consumed_targets:
                    raise ValueError(
                        f"Wan materializer: duplicate target consumption {target_key!r}"
                    )
                tensor = handle.get_tensor(source_key)
                _assign_dense_target(transformer, target_key, tensor)
                consumed_sources.add(source_key)
                consumed_targets.add(target_key)

            _validate_consumed_quant_auxiliaries(plan, consumed_auxiliary)
            if plan.dense_precision_contract == "current_fp8_patch_f32_rest_f16":
                _install_patch_embedding_precision_wrapper(transformer, compute_dtype)
        if consumed_sources != set(plan.source_to_target) or consumed_targets != expected_targets:
            raise ValueError("Wan materializer: missing or unconsumed planned parameters")
        _validate_materialized_transformer(transformer)
        transformer._latentslate_compute_dtype = compute_dtype
        transformer._latentslate_wan_config_fingerprint = plan.config_fingerprint
        transformer._latentslate_wan_mapping_fingerprint = plan.mapping_fingerprint
        transformer._latentslate_wan_artifact_identity = plan.identity
        return transformer
    except BaseException:
        _dematerialize_transformer(transformer)
        # ``raise`` keeps this frame in the traceback. Drop local references too,
        # so a late error cannot retain a partially restored 14B layer.
        quant_layers.clear()
        quantized_weight = bias = replacement = tensor = None
        raise


def plan_wan_root_residency(transformer: nn.Module) -> WanRootResidencyPlan:
    """Identify top-level Wan components separately from explicit transformer blocks."""

    root = tuple(
        name
        for name in ("rope", "patch_embedding", "condition_embedder", "norm_out", "proj_out")
        if hasattr(transformer, name)
    )
    blocks = tuple(f"blocks.{index}" for index, _ in enumerate(getattr(transformer, "blocks", ())))
    if not root or not blocks:
        raise ValueError("Wan residency plan requires a complete transformer root and blocks")
    all_state = set(dict(transformer.named_parameters())) | set(dict(transformer.named_buffers()))
    block_state: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for block in blocks:
        members = tuple(sorted(name for name in all_state if name.startswith(block + ".")))
        block_state[block] = members
        assigned.update(members)
    root_state = tuple(sorted(all_state - assigned))
    if not {"scale_shift_table", "rope.freqs_cos", "rope.freqs_sin"} <= set(root_state):
        raise ValueError("Wan residency plan lacks required root parameters or rope buffers")
    if assigned | set(root_state) != all_state or assigned & set(root_state):
        raise ValueError("Wan residency plan does not exhaustively classify transformer state")
    return WanRootResidencyPlan(
        root_components=root,
        blocks=blocks,
        root_state=root_state,
        block_state=MappingProxyType(block_state),
    )


def map_comfy_wan_parameter_key(source_key: str) -> str | None:
    """Map one normalized Comfy Wan parameter name to the pinned Diffusers layout."""

    key = _normalize_comfy_key(source_key)
    top_level = {
        "head.modulation": "scale_shift_table",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    }
    if key in top_level:
        return top_level[key]
    replacements = (
        (".modulation", ".scale_shift_table"),
        (".self_attn.q.", ".attn1.to_q."),
        (".self_attn.k.", ".attn1.to_k."),
        (".self_attn.v.", ".attn1.to_v."),
        (".self_attn.o.", ".attn1.to_out.0."),
        (".self_attn.norm_q.", ".attn1.norm_q."),
        (".self_attn.norm_k.", ".attn1.norm_k."),
        (".cross_attn.q.", ".attn2.to_q."),
        (".cross_attn.k.", ".attn2.to_k."),
        (".cross_attn.v.", ".attn2.to_v."),
        (".cross_attn.o.", ".attn2.to_out.0."),
        (".cross_attn.norm_q.", ".attn2.norm_q."),
        (".cross_attn.norm_k.", ".attn2.norm_k."),
        (".norm3.", ".norm2."),
        (".ffn.0.", ".ffn.net.0.proj."),
        (".ffn.2.", ".ffn.net.2."),
    )
    if not key.startswith("blocks."):
        return None
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    return None


def comfy_source_key_for_diffusers_parameter(target_key: str) -> str | None:
    """Return the canonical Comfy source key for one mapped Diffusers parameter."""

    top_level = {
        "scale_shift_table": "head.modulation",
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    }
    if target_key in top_level:
        return top_level[target_key]
    replacements = (
        (".scale_shift_table", ".modulation"),
        (".attn1.to_q.", ".self_attn.q."),
        (".attn1.to_k.", ".self_attn.k."),
        (".attn1.to_v.", ".self_attn.v."),
        (".attn1.to_out.0.", ".self_attn.o."),
        (".attn1.norm_q.", ".self_attn.norm_q."),
        (".attn1.norm_k.", ".self_attn.norm_k."),
        (".attn2.to_q.", ".cross_attn.q."),
        (".attn2.to_k.", ".cross_attn.k."),
        (".attn2.to_v.", ".cross_attn.v."),
        (".attn2.to_out.0.", ".cross_attn.o."),
        (".attn2.norm_q.", ".cross_attn.norm_q."),
        (".attn2.norm_k.", ".cross_attn.norm_k."),
        (".norm2.", ".norm3."),
        (".ffn.net.0.proj.", ".ffn.0."),
        (".ffn.net.2.", ".ffn.2."),
    )
    if not target_key.startswith("blocks."):
        return None
    for target, source in replacements:
        if target in target_key:
            return target_key.replace(target, source)
    return None


def _describe_plan_quant_layers(
    handle, plan: WanStoredAdapterPlan
) -> dict[str, StoredQuantizedLayer]:
    """Derive descriptors from the bound handle after identity revalidation."""

    auxiliaries = set(plan.quant_auxiliary)
    candidates: list[str] = []
    for source_key in plan.source_to_target:
        if not source_key.endswith(".weight"):
            continue
        stem = source_key.removesuffix(".weight")
        if plan.artifact_contract == "comfy_legacy/scaled_fp8_e4m3fn":
            required = {stem + ".scale_weight"}
        else:
            required = {stem + ".weight_scale", stem + ".comfy_quant"}
        if required <= auxiliaries:
            candidates.append(source_key)
    if not candidates:
        raise ValueError("Wan materializer: plan has no stored quantized linear layers")
    return describe_stored_layers_from_handle(
        handle,
        identity=plan.identity,
        keys=tuple(candidates),
        contract=plan.artifact_contract,
    )


def _quant_sidecars_for_contract(source_key: str, contract: str) -> set[str]:
    """Return the exact stored sidecars that prove one mapped weight is quantized."""

    stem = source_key.removesuffix(".weight")
    if contract == "comfy_legacy/scaled_fp8_e4m3fn":
        return {stem + ".scale_weight"}
    return {stem + ".weight_scale", stem + ".comfy_quant"}


def _plan_dense_precision_contract(
    artifact_contract: str,
    source_to_target: Mapping[str, str],
    header: Mapping[str, Any],
) -> tuple[str | None, dict[str, str], tuple[str, ...]]:
    """Bind supported Wan dense roles to their exact stored dtypes from the header.

    SmoothMix's current Comfy FP8 artifact is deliberately mixed: only the
    convolutional patch embedding is F32; every other unquantized tensor,
    including biases, is F16.  This is a stored artifact contract, not a
    conversion request.  Legacy FP8 and ConvRot are accepted only as uniform
    F16 dense state until another layout is independently proven.
    """

    header_keys = set(header)
    quantized_sources = {
        source_key
        for source_key in source_to_target
        if source_key.endswith(".weight")
        and _quant_sidecars_for_contract(source_key, artifact_contract) <= header_keys
    }
    dense_sources = set(source_to_target) - quantized_sources
    if not dense_sources:
        return None, {}, ("stored dense precision contract has no dense or bias tensors",)

    if artifact_contract == "comfy_quant/float8_e4m3fn":
        contract = "current_fp8_patch_f32_rest_f16"
        patch_targets = {"patch_embedding.weight", "patch_embedding.bias"}
        expected = {
            source_key: "F32" if target_key in patch_targets else "F16"
            for source_key, target_key in source_to_target.items()
            if source_key in dense_sources
        }
    elif artifact_contract in {
        "comfy_legacy/scaled_fp8_e4m3fn",
        "comfy_quant/int8_tensorwise_convrot",
    }:
        contract = "uniform_f16_dense"
        expected = {source_key: "F16" for source_key in dense_sources}
    else:
        return None, {}, (f"unsupported stored dense precision contract: {artifact_contract!r}",)

    errors: list[str] = []
    if artifact_contract == "comfy_quant/float8_e4m3fn":
        patch_sources = {
            target_key: source_key
            for source_key, target_key in source_to_target.items()
            if target_key in patch_targets
        }
        if set(patch_sources) != patch_targets or any(
            source not in dense_sources for source in patch_sources.values()
        ):
            errors.append(
                "current FP8 patch embedding weight and bias must be explicit unquantized dense sources"
            )
    mismatches = [
        source_key
        for source_key, expected_dtype in expected.items()
        if _entry_dtype(header.get(source_key), source_key) != expected_dtype
    ]
    if mismatches:
        errors.append(
            "stored dense precision contract mismatch: "
            f"{len(mismatches)} mapped tensor(s) violate {contract}"
        )
    return (contract if not errors else None), expected, tuple(errors)


def _non_linear_quantized_sources(
    artifact_contract: str,
    source_to_target: Mapping[str, str],
    header: Mapping[str, Any],
    skeleton: nn.Module,
) -> tuple[str, ...]:
    """Reject stored-quant sidecars unless their mapped target is exactly ``nn.Linear``."""

    header_keys = set(header)
    invalid: list[str] = []
    for source_key, target_key in source_to_target.items():
        if not source_key.endswith(".weight"):
            continue
        if not _quant_sidecars_for_contract(source_key, artifact_contract) <= header_keys:
            continue
        parent_path, separator, _ = target_key.rpartition(".")
        parent = skeleton.get_submodule(parent_path) if separator else skeleton
        if not isinstance(parent, nn.Linear):
            invalid.append(source_key)
    return tuple(sorted(invalid))


def _validate_consumed_quant_auxiliaries(plan: WanStoredAdapterPlan, consumed: set[str]) -> None:
    """Reject every planned sidecar that is not bound to a restored weight."""

    sentinels = {
        auxiliary
        for auxiliary in plan.quant_auxiliary
        if _normalize_comfy_key(auxiliary) in _LEGACY_QUANT_SENTINELS
    }
    if sentinels and plan.artifact_contract != "comfy_legacy/scaled_fp8_e4m3fn":
        raise ValueError(
            "Wan materializer: legacy quantization sentinel requires the legacy FP8 contract"
        )
    unconsumed = set(plan.quant_auxiliary) - consumed - sentinels
    if unconsumed:
        raise ValueError(
            f"Wan materializer: unconsumed quant auxiliaries: {sorted(unconsumed)[:3]}"
        )


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the exact canonical config because equal state shapes need not behave alike."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("Wan config contains a non-finite float")
            return value
        raise TypeError(f"Wan config has an unsupported value type: {type(value).__name__}")

    raw = json.dumps(
        normalize(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mapping_fingerprint(
    artifact_contract: str,
    source_to_target: Mapping[str, str],
    dense_precision_contract: str | None,
    dense_source_dtypes: Mapping[str, str],
) -> str:
    """Bind exact source roles and stored dtypes to the header-only adapter plan."""

    payload = {
        "artifact_contract": artifact_contract,
        "source_to_target": sorted(source_to_target.items()),
        "dense_precision_contract": dense_precision_contract,
        "dense_source_dtypes": sorted(dense_source_dtypes.items()),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if not isinstance(parent, nn.Module):
        raise TypeError(f"Wan materializer: invalid target parent {parent_path!r}")
    if not isinstance(getattr(parent, attribute, None), nn.Linear):
        raise TypeError(f"Wan materializer: {path!r} is not an nn.Linear target")
    setattr(parent, attribute, replacement)


def _assign_dense_target(root: nn.Module, target_key: str, tensor: torch.Tensor) -> None:
    parent_path, separator, attribute = target_key.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    current = getattr(parent, attribute, None)
    _validate_dense_tensor(tensor, target_key, current)
    if attribute in parent._parameters:
        parent._parameters[attribute] = nn.Parameter(tensor, requires_grad=False)
    elif attribute in parent._buffers:
        parent._buffers[attribute] = tensor
    else:
        raise ValueError(f"Wan materializer: {target_key!r} is not a parameter or buffer")


def _validate_dense_tensor(tensor: torch.Tensor, target_key: str, current: Any) -> None:
    if not isinstance(current, torch.Tensor):
        raise TypeError(f"Wan materializer: {target_key!r} is not tensor-backed")
    if tuple(tensor.shape) != tuple(current.shape):
        raise ValueError(f"Wan materializer: shape mismatch for {target_key!r}")
    if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(
            f"Wan materializer: non-quantized {target_key!r} has unsupported dtype {tensor.dtype}"
        )


def _validate_authoritative_compute_dtype(
    handle,
    plan: WanStoredAdapterPlan,
    quant_layers: Mapping[str, StoredQuantizedLayer],
    compute_dtype: torch.dtype,
) -> None:
    """Revalidate the plan's exact stored dense-role contract before payload reads."""

    if plan.dense_precision_contract is None:
        raise ValueError("Wan materializer: plan has no supported stored dense precision contract")
    dense_sources = set(plan.source_to_target) - set(quant_layers)
    expected = dict(plan.dense_source_dtypes)
    if set(expected) != dense_sources:
        raise ValueError(
            "Wan materializer: quantized/dense source roles differ from the validated plan"
        )
    observed = {
        source_key: handle.get_slice(source_key).get_dtype() for source_key in dense_sources
    }
    if observed != expected:
        raise ValueError(
            "Wan materializer: stored dense precision roles differ from the validated plan"
        )
    if plan.dense_precision_contract not in {"current_fp8_patch_f32_rest_f16", "uniform_f16_dense"}:
        raise ValueError("Wan materializer: unsupported stored dense precision contract")
    if compute_dtype != torch.float16:
        raise ValueError(
            "Wan materializer: compute dtype must exactly match the stored dense precision contract "
            "(torch.float16)"
        )


def _validate_materialized_transformer(transformer: nn.Module) -> None:
    meta = [name for name, value in transformer.named_parameters() if value.is_meta]
    meta.extend(name for name, value in transformer.named_buffers() if value.is_meta)
    if meta:
        raise ValueError(f"Wan materializer: meta tensors remain after restore: {meta[:3]}")


def _dematerialize_transformer(transformer: nn.Module) -> None:
    """Drop partially restored tensor storage after a failed materialization."""

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


class StoredPrecisionConv3d(nn.Module):
    """Keep a proven F32 patch embedding stored as-is while producing F16 activations.

    The wrapper performs only transient activation/output casts.  Its F32 weight
    and bias are registered directly and are never converted, copied to a new
    precision, or persisted under another name.
    """

    def __init__(self, conv: nn.Conv3d, *, output_dtype: torch.dtype) -> None:
        super().__init__()
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("StoredPrecisionConv3d requires an nn.Conv3d")
        if conv.weight.dtype != torch.float32 or (
            conv.bias is not None and conv.bias.dtype != torch.float32
        ):
            raise ValueError("StoredPrecisionConv3d requires stored F32 weight and bias")
        if output_dtype != torch.float16:
            raise ValueError("StoredPrecisionConv3d currently supports F16 output only")
        self.weight = nn.Parameter(conv.weight, requires_grad=False)
        self.bias = nn.Parameter(conv.bias, requires_grad=False) if conv.bias is not None else None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.output_dtype = output_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute in the stored F32 precision and return the planned F16 activation."""

        output = F.conv3d(
            input.to(dtype=self.weight.dtype),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output.to(dtype=self.output_dtype)


def _install_patch_embedding_precision_wrapper(
    transformer: nn.Module, compute_dtype: torch.dtype
) -> None:
    """Install the only accepted mixed-precision component without touching its weights."""

    patch_embedding = getattr(transformer, "patch_embedding", None)
    if not isinstance(patch_embedding, nn.Conv3d):
        raise TypeError("Wan materializer: patch_embedding is not an nn.Conv3d target")
    transformer.patch_embedding = StoredPrecisionConv3d(patch_embedding, output_dtype=compute_dtype)


class NativeStoredLinear(nn.Module):
    """A linear module backed by one already-restored Comfy ``QuantizedTensor``.

    Quantized weights and biases are frozen ``nn.Parameter`` objects so planned
    block-group residency accounting sees them. The optional activation scale is
    stored as an immutable Python float so a module dtype move cannot downcast it.
    """

    def __init__(
        self,
        weight,
        bias: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(weight, QuantizedTensor) or weight.ndim != 2:
            raise TypeError("NativeStoredLinear requires a 2D restored QuantizedTensor weight")
        if weight._layout_cls not in {"TensorCoreFP8Layout", "TensorWiseINT8Layout"}:
            raise ValueError(f"NativeStoredLinear does not support {weight._layout_cls!r}")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("NativeStoredLinear bias must match output features")
        if input_scale is not None and (
            input_scale.dtype != torch.float32
            or input_scale.ndim != 0
            or not bool(torch.isfinite(input_scale))
            or not bool(input_scale > 0)
        ):
            raise ValueError(
                "NativeStoredLinear input_scale must be one positive finite F32 scalar"
            )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.input_scale = float(input_scale.item()) if input_scale is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute with native kitchen layouts; only FP8 activations are transiently quantized."""

        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("NativeStoredLinear input feature count does not match stored weight")
        original_shape = input.shape
        flat_input = input.reshape(-1, original_shape[-1])
        if self.weight._layout_cls == "TensorWiseINT8Layout":
            output = F.linear(flat_input, self.weight, self.bias)
        else:
            activation = _quantize_fp8_activation(flat_input, self.input_scale)
            output = F.linear(activation, self.weight, self.bias)
        return output.reshape(*original_shape[:-1], self.weight.shape[0])

    def move_stored_storage(self, device: torch.device | str) -> None:
        """Physically move stored qdata/scale plus bias without weight conversion.

        ``nn.Module.to`` on an ancestor can replace this wrapper parameter's
        logical tensor while leaving third-party ``QuantizedTensor`` internals on
        the previous device. Rebuild the wrapper from its stored bytes so qdata
        and scale move together, retaining the exact layout and dtype.
        """

        from comfy_kitchen.tensor import QuantizedTensor

        target = torch.device(device)
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("NativeStoredLinear stored weight is no longer a QuantizedTensor")
        if weight._qdata.dtype != weight.storage_dtype:
            raise RuntimeError("NativeStoredLinear stored qdata dtype changed")
        params = dataclass_replace(weight.params, scale=weight.params.scale.to(device=target))
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        self._parameters["weight"] = nn.Parameter(restored, requires_grad=False)
        if self.bias is not None:
            self._parameters["bias"] = nn.Parameter(
                self.bias.to(device=target), requires_grad=False
            )


def _move_module_and_stored_linears(module: nn.Module, device: torch.device | str) -> None:
    """Move normal module state plus Kitchen storage hidden behind wrappers.

    A Wan LoRA wrapper makes ``NativeStoredLinear`` a nested child.  Plain
    ``Module.to`` updates its logical parameter but cannot relocate the
    third-party qdata/scale storage, so every residency transition must repair
    every nested native stored linear immediately afterwards.
    """

    module.to(device=device)
    for nested in module.modules():
        if isinstance(nested, NativeStoredLinear):
            nested.move_stored_storage(device)


class SynchronousBlockResidencyManager:
    """Engine-owned, non-reentrant residency for explicit transformer blocks.

    Every pre-hook moves the whole block to the execution device with
    ``module.to``. Every post-hook is registered with ``always_call=True`` and
    moves the whole block back to the offload device, including after a failed
    forward. Outputs are deliberately not moved during cleanup, so they stay on
    the execution device for the next block.

    This does not use Diffusers or Accelerate hooks. A tiny CUDA residency proof
    covers this primitive, but full-model stored-quant generation remains
    unproven and unavailable.
    """

    def __init__(
        self,
        blocks: BlockModules,
        *,
        onload_device: torch.device | str,
        offload_device: torch.device | str = "cpu",
    ) -> None:
        if not blocks:
            raise ValueError("stored-quant block residency requires explicit blocks")
        ordered = OrderedDict(blocks)
        if len({id(block) for block in ordered.values()}) != len(ordered):
            raise ValueError("stored-quant block residency does not permit duplicate block modules")
        if not all(
            isinstance(name, str) and name and isinstance(block, nn.Module)
            for name, block in ordered.items()
        ):
            raise TypeError("stored-quant block residency requires named nn.Module blocks")
        self._blocks = ordered
        self.onload_device = _canonicalize_residency_device(torch.device(onload_device))
        self.offload_device = _canonicalize_residency_device(torch.device(offload_device))
        self._handles: list[Any] = []
        self._active_name: str | None = None
        self._closed = False
        self._failed_reason: str | None = None
        self._transitioning = False
        self._lock = threading.RLock()

    @property
    def attached(self) -> bool:
        """Whether this manager currently owns hooks on its explicit blocks."""

        with self._lock:
            return bool(self._handles)

    @property
    def active_block(self) -> str | None:
        """The one block executing synchronously, if any."""

        with self._lock:
            return self._active_name

    def attach(self) -> None:
        """Attach paired pre/post hooks exactly once."""

        with self._lock:
            if self._closed:
                raise RuntimeError("stored-quant block residency manager is closed")
            if self._failed_reason:
                raise RuntimeError(
                    f"stored-quant block residency manager failed: {self._failed_reason}"
                )
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            if self._handles:
                raise RuntimeError("stored-quant block residency hooks are already attached")
            for name, block in self._blocks.items():
                self._handles.append(block.register_forward_pre_hook(self._make_pre_hook(name)))
                self._handles.append(
                    block.register_forward_hook(self._make_post_hook(name), always_call=True)
                )

    def force_offload(self) -> None:
        """Synchronously move every managed block to the configured offload device."""

        with self._lock:
            if self._active_name is not None:
                raise RuntimeError("cannot force offload while a stored-quant block is active")
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
        try:
            self._offload_all_blocks()
        finally:
            with self._lock:
                self._transitioning = False

    def _offload_all_blocks(self) -> None:
        errors: list[str] = []
        for name, block in self._blocks.items():
            try:
                _move_module_and_stored_linears(block, self.offload_device)
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure for fail-closed state
                errors.append(f"{name}: {exc}")
        if errors:
            reason = "force-offload failed: " + "; ".join(errors)
            with self._lock:
                self._failed_reason = reason
            raise RuntimeError(reason)

    def remove(self, *, force_offload: bool = True) -> None:
        """Remove hooks and, by default, synchronously return all blocks to CPU."""

        with self._lock:
            if self._active_name is not None:
                raise RuntimeError("cannot remove stored-quant residency while a block is active")
            if self._closed:
                return
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
            handles = tuple(self._handles)
        try:
            for handle in handles:
                handle.remove()
            if force_offload:
                self._offload_all_blocks()
        finally:
            with self._lock:
                self._handles.clear()
                self._closed = True
                self._transitioning = False

    def abort_and_force_offload(self) -> None:
        """Emergency teardown after a BaseException bypassed the post-forward hook.

        This is intentionally reserved for an owning session after control has
        returned from the block call. Normal concurrent ``remove`` remains
        fail-closed while a block is active.
        """

        with self._lock:
            if self._closed:
                return
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
            handles = tuple(self._handles)
            self._active_name = None
        try:
            for handle in handles:
                handle.remove()
            self._offload_all_blocks()
        finally:
            with self._lock:
                self._handles.clear()
                self._closed = True
                self._transitioning = False

    def poison_and_remove_hooks(self, reason: str) -> None:
        """Detach hooks without moving storage after a failed CUDA barrier."""

        if not isinstance(reason, str) or not reason:
            raise ValueError("stored-quant poison reason must be nonempty")
        with self._lock:
            if self._closed:
                return
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
            handles = tuple(self._handles)
        try:
            for handle in handles:
                handle.remove()
        finally:
            with self._lock:
                self._handles.clear()
                self._active_name = None
                self._failed_reason = reason
                self._closed = True
                self._transitioning = False

    def _make_pre_hook(self, name: str):
        def pre_hook(module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            with self._lock:
                if self._closed or self._failed_reason or self._transitioning:
                    raise RuntimeError("stored-quant block residency is unavailable")
                if self._active_name is not None:
                    self._failed_reason = (
                        f"non-reentrant block execution: {self._active_name} -> {name}"
                    )
                    raise RuntimeError("stored-quant block residency is non-reentrant")
                self._active_name = name
            try:
                _move_module_and_stored_linears(module, self.onload_device)
            except Exception as exc:
                reason = f"onload failed for {name}: {exc}"
                with self._lock:
                    self._active_name = None
                    self._failed_reason = reason
                raise RuntimeError(reason) from exc

        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            try:
                _move_module_and_stored_linears(module, self.offload_device)
            except Exception as exc:
                reason = f"offload failed for {name}: {exc}"
                with self._lock:
                    self._failed_reason = reason
                raise RuntimeError(reason) from exc
            finally:
                with self._lock:
                    self._active_name = None
            return output

        return post_hook


class WanTransformerResidencySession:
    """Engine-owned full-transformer residency with root/static and per-block moves.

    The session deliberately never calls ``transformer.to`` and never installs
    Diffusers/Accelerate hooks. Root modules and direct root state stay on the
    execution device; each transformer block is moved synchronously just before
    its forward and returned to CPU immediately afterward.
    """

    def __init__(
        self,
        transformer: nn.Module,
        plan: WanRootResidencyPlan,
        *,
        onload_device: torch.device | str,
        offload_device: torch.device | str = "cpu",
    ) -> None:
        poisoned = getattr(transformer, "_latentslate_residency_poisoned", None)
        if poisoned:
            raise RuntimeError(f"Wan transformer residency is poisoned: {poisoned}")
        self.transformer = transformer
        self.plan = plan
        self.onload_device = _canonicalize_residency_device(torch.device(onload_device))
        self.offload_device = _canonicalize_residency_device(torch.device(offload_device))
        if self.offload_device.type != "cpu":
            raise ValueError("Wan transformer residency requires CPU as the offload device")
        self._blocks = OrderedDict((name, transformer.get_submodule(name)) for name in plan.blocks)
        self._block_residency = SynchronousBlockResidencyManager(
            self._blocks,
            onload_device=self.onload_device,
            offload_device=self.offload_device,
        )
        self._dtype_snapshot = self._validate_plan_coverage()
        self._entered = False
        self._closed = False
        self._owner_thread_id: int | None = None
        self._execution_thread_id: int | None = None
        self._execution_lock = threading.RLock()
        self._execution_handles: list[Any] = []

    @property
    def active(self) -> bool:
        """Whether the session owns active root/block residency."""

        return self._entered and not self._closed

    def __enter__(self) -> Self:
        if self._closed or self._entered:
            raise RuntimeError(
                "Wan transformer residency session is one-shot and cannot be re-entered"
            )
        self._claim_transformer()
        try:
            self._owner_thread_id = threading.get_ident()
            self._validate_runtime_state()
            self._move_roots(self.onload_device)
            self._block_residency.force_offload()
            self._assert_devices(self.plan.root_state, self.onload_device)
            self._assert_devices(self._block_state_names(), self.offload_device)
            self._block_residency.attach()
            self._attach_execution_tracking()
            self._entered = True
            return self
        except BaseException:
            self._teardown(suppress_errors=True, allow_abort=False)
            raise

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("Wan transformer residency context exited from a non-owning thread")
        self._teardown(suppress_errors=False, allow_abort=True)
        return False

    def close(self) -> None:
        """Remove block hooks and return all planned state to CPU synchronously."""

        if self._closed:
            return
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "Wan transformer residency close must run on the owning context thread"
            )
        if self._is_executing():
            raise RuntimeError("cannot close Wan transformer residency while a forward is active")
        self._teardown(suppress_errors=False, allow_abort=False)

    def _claim_transformer(self) -> None:
        global _ACTIVE_WAN_SESSION
        with _WAN_SESSION_GUARD_LOCK:
            if _ACTIVE_WAN_SESSION is not None:
                raise RuntimeError("a Wan residency session is already active process-wide")
            _ACTIVE_WAN_SESSION = self

    def _release_transformer(self) -> None:
        global _ACTIVE_WAN_SESSION
        with _WAN_SESSION_GUARD_LOCK:
            if _ACTIVE_WAN_SESSION is self:
                _ACTIVE_WAN_SESSION = None

    def _teardown(self, *, suppress_errors: bool, allow_abort: bool) -> None:
        errors: list[BaseException] = []
        barrier_succeeded = True
        if self.onload_device.type == "cuda":
            try:
                # Stored QuantizedTensor parameters are reconstructed while
                # moving them back to CPU. Ensure every kernel using current
                # CUDA storage has completed before replacing that storage.
                torch.cuda.synchronize(self.onload_device)
            except BaseException as exc:  # noqa: BLE001 - poison unsafe CUDA state
                barrier_succeeded = False
                errors.append(exc)
                reason = f"CUDA stage teardown barrier failed: {exc}"
                self.transformer._latentslate_residency_poisoned = reason
        try:
            if self._is_executing() and (
                not allow_abort or threading.get_ident() != self._owner_thread_id
            ):
                raise RuntimeError(
                    "cannot teardown Wan transformer residency while a forward is active"
                )
            if not barrier_succeeded:
                self._block_residency.poison_and_remove_hooks(
                    self.transformer._latentslate_residency_poisoned
                )
            elif self._block_residency.active_block is None:
                self._block_residency.remove(force_offload=True)
            else:
                if not allow_abort:
                    raise RuntimeError(
                        "cannot abort stored-quant block residency outside owning context exit"
                    )
                self._block_residency.abort_and_force_offload()
        except BaseException as exc:  # noqa: BLE001 - teardown must attempt root cleanup too
            errors.append(exc)
        if barrier_succeeded:
            try:
                self._move_roots(self.offload_device)
                self._assert_devices(self._planned_state_names(), self.offload_device)
                self._validate_runtime_state()
            except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
                errors.append(exc)
        try:
            self._remove_execution_tracking()
        finally:
            self._entered = False
            self._closed = True
            self._release_transformer()
        if errors and not suppress_errors:
            raise RuntimeError(
                f"Wan transformer residency teardown failed: {errors[0]}"
            ) from errors[0]

    def _validate_plan_coverage(self) -> dict[str, torch.dtype]:
        actual = self._state_values()
        if len(set(self.plan.blocks)) != len(self.plan.blocks):
            raise ValueError("Wan transformer residency plan has duplicate blocks")
        if set(self.plan.block_state) != set(self.plan.blocks):
            raise ValueError(
                "Wan transformer residency plan block state keys do not exactly match blocks"
            )
        for block in self.plan.blocks:
            module = self.transformer.get_submodule(block)
            expected_block_state = {block + "." + name for name, _ in module.named_parameters()}
            expected_block_state.update(block + "." + name for name, _ in module.named_buffers())
            if set(self.plan.block_state[block]) != expected_block_state:
                raise ValueError(
                    f"Wan transformer residency block state is stale or incomplete: {block!r}"
                )
        planned = self._planned_state_names()
        if set(actual) != planned:
            raise ValueError(
                "Wan transformer residency plan does not exactly cover parameters and buffers"
            )
        if set(self.plan.root_state) & self._block_state_names():
            raise ValueError("Wan transformer residency plan overlaps root and block state")
        if set(self.plan.root_state) | self._block_state_names() != planned:
            raise ValueError("Wan transformer residency plan does not exhaustively classify state")
        for component in self.plan.root_components:
            self.transformer.get_submodule(component)
        for name in self.plan.root_state:
            if "." not in name:
                if (
                    name not in self.transformer._parameters
                    and name not in self.transformer._buffers
                ):
                    raise ValueError(
                        f"Wan transformer residency direct root state is absent: {name!r}"
                    )
            elif not any(
                name.startswith(component + ".") for component in self.plan.root_components
            ):
                raise ValueError(
                    f"Wan transformer residency root state lacks a root component: {name!r}"
                )
        if not {"scale_shift_table", "rope.freqs_cos", "rope.freqs_sin"} <= set(
            self.plan.root_state
        ):
            raise ValueError(
                "Wan transformer residency must cover scale_shift_table and rope buffers"
            )
        return {name: value.dtype for name, value in actual.items()}

    def _validate_runtime_state(self) -> None:
        actual = self._state_values()
        if set(actual) != set(self._dtype_snapshot):
            raise RuntimeError("Wan transformer residency state changed after session construction")
        for name, value in actual.items():
            if value.is_meta:
                raise RuntimeError(f"Wan transformer residency cannot move meta state: {name!r}")
            if value.dtype != self._dtype_snapshot[name]:
                raise RuntimeError(f"Wan transformer residency dtype changed for {name!r}")

    def _move_roots(self, device: torch.device) -> None:
        for component in self.plan.root_components:
            root_module = self.transformer.get_submodule(component)
            _move_module_and_stored_linears(root_module, device)
        for name in self.plan.root_state:
            if "." in name:
                continue
            if name in self.transformer._parameters:
                parameter = self.transformer._parameters[name]
                if parameter is not None:
                    self.transformer._parameters[name] = nn.Parameter(
                        parameter.to(device=device), requires_grad=parameter.requires_grad
                    )
            elif name in self.transformer._buffers:
                buffer = self.transformer._buffers[name]
                if buffer is not None:
                    self.transformer._buffers[name] = buffer.to(device=device)
            else:
                raise RuntimeError(
                    f"Wan transformer residency direct root state disappeared: {name!r}"
                )
        self._validate_runtime_state()

    def _state_values(self) -> dict[str, torch.Tensor]:
        values = dict(self.transformer.named_parameters())
        values.update(self.transformer.named_buffers())
        return values

    def _planned_state_names(self) -> set[str]:
        return set(self.plan.root_state) | self._block_state_names()

    def _block_state_names(self) -> set[str]:
        return {name for names in self.plan.block_state.values() for name in names}

    def _assert_devices(self, names: tuple[str, ...] | set[str], device: torch.device) -> None:
        actual = self._state_values()
        wrong = [
            name for name in names if not _matches_requested_device(actual[name].device, device)
        ]
        if wrong:
            raise RuntimeError(f"Wan transformer residency device coverage failed: {wrong[:3]}")
        names_set = set(names)
        physical_wrong: list[str] = []
        for module_name, module in self.transformer.named_modules():
            if not isinstance(module, NativeStoredLinear):
                continue
            prefix = module_name + "." if module_name else ""
            if prefix + "weight" not in names_set:
                continue
            if not _matches_requested_device(
                module.weight._qdata.device, device
            ) or not _matches_requested_device(module.weight.params.scale.device, device):
                physical_wrong.append(prefix + "weight")
            if module.bias is not None and not _matches_requested_device(
                module.bias.device, device
            ):
                physical_wrong.append(prefix + "bias")
        if physical_wrong:
            raise RuntimeError(
                f"Wan transformer residency physical storage coverage failed: {physical_wrong[:3]}"
            )

    def _attach_execution_tracking(self) -> None:
        self._execution_handles.append(
            self.transformer.register_forward_pre_hook(self._forward_pre_hook)
        )
        self._execution_handles.append(
            self.transformer.register_forward_hook(self._forward_post_hook, always_call=True)
        )

    def _remove_execution_tracking(self) -> None:
        for handle in self._execution_handles:
            handle.remove()
        self._execution_handles.clear()
        with self._execution_lock:
            self._execution_thread_id = None

    def _forward_pre_hook(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        thread_id = threading.get_ident()
        with self._execution_lock:
            if self._execution_thread_id is not None:
                raise RuntimeError(
                    "Wan transformer residency does not permit concurrent or reentrant forwards"
                )
            if thread_id != self._owner_thread_id:
                raise RuntimeError(
                    "Wan transformer residency forward must run on the owning context thread"
                )
            self._execution_thread_id = thread_id

    def _forward_post_hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        with self._execution_lock:
            self._execution_thread_id = None
        return output

    def _is_executing(self) -> bool:
        with self._execution_lock:
            return self._execution_thread_id is not None


def _matches_requested_device(actual: torch.device, requested: torch.device) -> bool:
    """Require the exact previously-canonicalized device, including CUDA ordinal."""

    return actual == requested


def _canonicalize_residency_device(device: torch.device) -> torch.device:
    """Resolve an index-unspecified CUDA request once for exact residency checks."""

    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def attach_native_stored_linear(
    parent: nn.Module,
    attribute: str,
    weight,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
) -> NativeStoredLinear:
    """Replace one ``nn.Linear`` child with a stored-quant wrapper, fail-closed."""

    if not isinstance(getattr(parent, attribute, None), nn.Linear):
        raise TypeError(f"{attribute!r} is not an nn.Linear child")
    replacement = NativeStoredLinear(weight, bias, input_scale)
    setattr(parent, attribute, replacement)
    return replacement


def validate_stored_quant_offload_mode(mode: str) -> str:
    """Require the sole planned stored-quant residency mode.

    ``group_block`` means Engine-owned synchronous whole-block moves only; it must
    never select Diffusers or Accelerate group hooks. This function does not imply
    that stored-quant CUDA generation is available today.
    """

    if mode not in SUPPORTED_STORED_QUANT_OFFLOAD_MODES:
        raise ValueError(
            "stored quant requires Engine-owned block-level group offload "
            "(offload='group_block'); sequential/meta, leaf-level, whole-model, and disk modes are unsupported"
        )
    return mode


def _quantize_fp8_activation(input: torch.Tensor, input_scale: float | None):
    """Make a temporary FP8 activation; this never mutates or converts weights."""

    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    if input_scale is None:
        scale = (
            input.detach().abs().amax().to(dtype=torch.float32)
            / torch.finfo(torch.float8_e4m3fn).max
        )
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    else:
        scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    qdata, params = TensorCoreFP8Layout.quantize(input, scale=scale, dtype=torch.float8_e4m3fn)
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _read_safetensors_header(path: Path, size_bytes: int) -> dict[str, Any]:
    """Read the already structurally-probed header, never any tensor payload."""

    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("Wan adapter: SafeTensors header is truncated")
        length = struct.unpack("<Q", raw_length)[0]
        if length > _MAX_HEADER_BYTES or length > size_bytes - 8:
            raise ValueError("Wan adapter: SafeTensors header exceeds bounds")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("Wan adapter: SafeTensors header is truncated")
    parsed = json.loads(raw_header)
    if not isinstance(parsed, dict):
        raise TypeError("Wan adapter: SafeTensors header must be an object")
    return parsed


def _normalize_comfy_key(key: str) -> str:
    return key.removeprefix(_PREFIX)


def _is_quant_auxiliary(key: str) -> bool:
    return key.endswith(_QUANT_SUFFIXES)


def _quant_auxiliary_target(key: str) -> str | None:
    for suffix in _QUANT_SUFFIXES:
        if key.endswith(suffix):
            target = map_comfy_wan_parameter_key(key.removesuffix(suffix) + ".weight")
            return target if target and target.endswith(".weight") else None
    return None


def _entry_shape(entry: Any, key: str) -> tuple[int, ...]:
    if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
        raise TypeError(f"Wan adapter: invalid source shape for {key!r}")
    shape = tuple(entry["shape"])
    if not all(isinstance(item, int) and item >= 0 for item in shape):
        raise ValueError(f"Wan adapter: invalid source shape for {key!r}")
    return shape


def _entry_dtype(entry: Any, key: str) -> str:
    """Return the validated SafeTensors dtype label from a header entry."""

    if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
        raise TypeError(f"Wan adapter: invalid source dtype for {key!r}")
    return entry["dtype"]
