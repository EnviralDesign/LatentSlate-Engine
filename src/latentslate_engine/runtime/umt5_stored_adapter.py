"""Header-bound native stored-quant UMT5 encoder materialization.

This is intentionally Comfy-first: it restores only the quantized bytes and
scales already stored in a SafeTensors artifact.  It never calls a quantizer,
``from_pretrained``, TorchAO, or ModelOpt.  The standard Transformers
``UMT5EncoderModel`` is used only after an exact header/schema plan succeeds.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_safetensors, revalidate_artifact
from ..stored_quant import (
    StoredQuantizedLayer,
    describe_stored_layers_from_handle,
    restore_stored_quantized_tensor,
)
from . import wan22_stored_adapter as wan_residency
from .wan22_stored_adapter import NativeStoredLinear, _read_safetensors_header

UMT5_XXL_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "vocab_size": 256384,
        "d_model": 4096,
        "d_kv": 64,
        "d_ff": 10240,
        "num_layers": 24,
        "num_heads": 64,
        "dropout_rate": 0.0,
        "feed_forward_proj": "gated-gelu",
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "layer_norm_epsilon": 1e-6,
        "tie_word_embeddings": True,
    }
)
_SUPPORTED = frozenset({"comfy_legacy/scaled_fp8_e4m3fn", "comfy_quant/int8_tensorwise_convrot"})
_AUX_SUFFIXES = (".scale_weight", ".weight_scale", ".input_scale", ".comfy_quant")
_SUPPORT_KEYS = frozenset({"spiece_model", "scaled_fp8"})
_MAX_SENTENCEPIECE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class UMT5StoredAdapterPlan:
    """Exact header-only source-to-UMT5EncoderModel mapping."""

    identity: ArtifactIdentity
    artifact_contract: str
    config_fingerprint: str
    mapping_fingerprint: str
    source_to_targets: Mapping[str, tuple[str, ...]]
    quant_sources: tuple[str, ...]
    dense_source_dtypes: Mapping[str, str]
    quant_auxiliary: tuple[str, ...]
    support_auxiliary: tuple[str, ...]
    missing_targets: tuple[str, ...]
    duplicate_targets: tuple[str, ...]
    unexpected_extras: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    precision_errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not (
            self.missing_targets
            or self.duplicate_targets
            or self.unexpected_extras
            or self.shape_mismatches
            or self.precision_errors
        )

    @property
    def errors(self) -> tuple[str, ...]:
        result: list[str] = []
        for label, values in (
            ("missing UMT5 parameters", self.missing_targets),
            ("duplicate mapped UMT5 parameters", self.duplicate_targets),
            ("UMT5 parameter shape mismatches", self.shape_mismatches),
            ("unrecognized UMT5 source keys", self.unexpected_extras),
        ):
            if values:
                result.append(f"{label}: {len(values)}")
        result.extend(self.precision_errors)
        return tuple(result)

    def require_available(self) -> None:
        if not self.available:
            raise ValueError("UMT5 stored adapter unavailable: " + "; ".join(self.errors))


def build_umt5_encoder_skeleton(config: Mapping[str, Any] = UMT5_XXL_CONFIG) -> nn.Module:
    """Create the pinned standard UMT5 encoder shell with meta storage only."""

    from accelerate import init_empty_weights
    from transformers import UMT5Config, UMT5EncoderModel

    with init_empty_weights():
        return UMT5EncoderModel(UMT5Config(**dict(config)))


def plan_comfy_umt5_encoder(
    artifact_path: Path, config: Mapping[str, Any] = UMT5_XXL_CONFIG
) -> UMT5StoredAdapterPlan:
    """Prove a staged Comfy UMT5 artifact exactly matches the standard encoder shell."""

    source = Path(artifact_path).resolve(strict=True)
    probe = probe_safetensors(source)
    if probe.quantization_contract not in _SUPPORTED:
        raise ValueError(
            f"UMT5 stored adapter: unsupported artifact contract {probe.quantization_contract!r}"
        )
    header = _read_safetensors_header(source, probe.identity.size_bytes)
    skeleton = build_umt5_encoder_skeleton(config)
    expected = {name: tuple(value.shape) for name, value in skeleton.state_dict().items()}
    source_to_targets: dict[str, tuple[str, ...]] = {}
    target_sources: dict[str, list[str]] = {}
    quant_auxiliary: list[str] = []
    support_auxiliary: list[str] = []
    unexpected: list[str] = []
    mismatches: list[str] = []

    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if key in _SUPPORT_KEYS:
            support_auxiliary.append(key)
            continue
        if key.endswith(_AUX_SUFFIXES):
            if _auxiliary_maps_to_source(key, probe.quantization_contract, set(header)):
                quant_auxiliary.append(key)
            else:
                unexpected.append(key)
            continue
        targets = _source_targets(key)
        if targets is None:
            unexpected.append(key)
            continue
        source_to_targets[key] = targets
        shape = _entry_shape(entry, key)
        for target in targets:
            target_sources.setdefault(target, []).append(key)
            if target not in expected or shape != expected[target]:
                mismatches.append(key)

    quant_sources, topology_errors = _plan_quant_roles(
        probe.quantization_contract, header, source_to_targets, skeleton
    )
    dense_sources = set(source_to_targets) - set(quant_sources)
    dense_dtypes = {key: _entry_dtype(header[key], key) for key in dense_sources}
    precision_errors = list(
        _validate_header_contract(
            probe.quantization_contract, header, source_to_targets, dense_dtypes
        )
    )
    precision_errors.extend(topology_errors)
    missing = tuple(sorted(set(expected) - set(target_sources)))
    duplicate = tuple(sorted(key for key, sources in target_sources.items() if len(sources) != 1))
    mapping_fingerprint = _mapping_fingerprint(
        probe.quantization_contract, source_to_targets, quant_sources, dense_dtypes, quant_auxiliary
    )
    return UMT5StoredAdapterPlan(
        identity=probe.identity,
        artifact_contract=probe.quantization_contract,
        config_fingerprint=_config_fingerprint(config),
        mapping_fingerprint=mapping_fingerprint,
        source_to_targets=MappingProxyType(dict(sorted(source_to_targets.items()))),
        quant_sources=quant_sources,
        dense_source_dtypes=MappingProxyType(dict(sorted(dense_dtypes.items()))),
        quant_auxiliary=tuple(sorted(quant_auxiliary)),
        support_auxiliary=tuple(sorted(support_auxiliary)),
        missing_targets=missing,
        duplicate_targets=duplicate,
        unexpected_extras=tuple(sorted(set(unexpected))),
        shape_mismatches=tuple(sorted(set(mismatches))),
        precision_errors=tuple(precision_errors),
    )


def materialize_umt5_encoder(
    plan: UMT5StoredAdapterPlan, config: Mapping[str, Any], *, compute_dtype: torch.dtype
) -> nn.Module:
    """Restore a completely validated UMT5 encoder using one identity-bound handle."""

    from safetensors import safe_open

    plan.require_available()
    if compute_dtype != torch.float16:
        raise ValueError("UMT5 materializer: only proven F16 dense compute is available")
    if plan.config_fingerprint != _config_fingerprint(config):
        raise ValueError("UMT5 materializer: config does not match validated plan")
    if plan.mapping_fingerprint != _mapping_fingerprint(
        plan.artifact_contract,
        plan.source_to_targets,
        plan.quant_sources,
        plan.dense_source_dtypes,
        plan.quant_auxiliary,
    ):
        raise ValueError("UMT5 materializer: mapped roles do not match validated plan")
    encoder = build_umt5_encoder_skeleton(config)
    expected_targets = set(encoder.state_dict())
    if {
        target for targets in plan.source_to_targets.values() for target in targets
    } != expected_targets:
        raise ValueError(
            "UMT5 materializer: plan targets do not exactly match this encoder skeleton"
        )
    consumed_sources: set[str] = set()
    consumed_auxiliary: set[str] = set()
    restored: list[object] = []
    quant_layers: dict[str, StoredQuantizedLayer] = {}
    quantized = None
    tensor = None
    value = None
    scale = None
    spiece = None
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            if not revalidate_artifact(plan.identity):
                raise ValueError(
                    "UMT5 materializer: artifact identity changed before materialization"
                )
            keys = set(handle.keys())
            if not set(plan.source_to_targets).issubset(keys):
                raise ValueError("UMT5 materializer: planned source tensors are absent")
            quant_layers = _describe_quant_layers(handle, plan)
            if not revalidate_artifact(plan.identity):
                raise ValueError("UMT5 materializer: artifact changed during descriptor discovery")
            if set(quant_layers) != set(plan.quant_sources):
                raise ValueError("UMT5 materializer: quantized source roles differ from plan")
            if set(plan.dense_source_dtypes) != set(plan.source_to_targets) - set(quant_layers):
                raise ValueError("UMT5 materializer: quantized/dense source roles differ from plan")
            if any(
                handle.get_slice(key).get_dtype() != dtype
                for key, dtype in plan.dense_source_dtypes.items()
            ):
                raise ValueError("UMT5 materializer: dense precision differs from validated plan")
            if "spiece_model" not in plan.support_auxiliary:
                raise ValueError("UMT5 materializer: embedded tokenizer is absent from the plan")
            spiece_slice = handle.get_slice("spiece_model")
            spiece_shape = tuple(spiece_slice.get_shape())
            if (
                spiece_slice.get_dtype() != "U8"
                or len(spiece_shape) != 1
                or not 0 < spiece_shape[0] <= _MAX_SENTENCEPIECE_BYTES
            ):
                raise ValueError("UMT5 materializer: embedded tokenizer header is invalid")
            spiece = handle.get_tensor("spiece_model")
            if spiece.dtype != torch.uint8 or tuple(spiece.shape) != spiece_shape:
                raise ValueError("UMT5 materializer: embedded tokenizer payload is invalid")
            tokenizer_sha256 = hashlib.sha256(spiece.contiguous().numpy().tobytes()).hexdigest()
            for source, layer in quant_layers.items():
                targets = plan.source_to_targets[source]
                if len(targets) != 1:
                    raise ValueError("UMT5 materializer: stored quant source cannot target aliases")
                target = targets[0]
                parent_path, _, attribute = target.rpartition(".")
                parent = encoder.get_submodule(parent_path)
                if attribute != "weight" or not isinstance(parent, nn.Linear):
                    raise TypeError(
                        "UMT5 materializer: stored quant is permitted only for nn.Linear weights"
                    )
                quantized = restore_stored_quantized_tensor(handle, layer, compute_dtype)
                restored.append(quantized)
                _replace_parameter_module(encoder, parent_path, NativeStoredLinear(quantized))
                consumed_sources.add(source)
                consumed_auxiliary.add(layer.scale_key)
                if layer.marker_key:
                    consumed_auxiliary.add(layer.marker_key)
                input_scale = source.removesuffix(".weight") + ".input_scale"
                if input_scale in keys:
                    # Official ConvRot UMT5 duplicates the per-row stored weight
                    # scale under input_scale.  Validate and consume it rather
                    # than silently applying an unproven extra transformation.
                    value = handle.get_tensor(input_scale)
                    scale = handle.get_tensor(layer.scale_key)
                    if (
                        value.dtype != torch.float32
                        or tuple(value.shape) != tuple(scale.shape)
                        or not torch.equal(value, scale)
                    ):
                        raise ValueError(
                            "UMT5 materializer: ConvRot input_scale is not the proven stored scale alias"
                        )
                    consumed_auxiliary.add(input_scale)
            for source, targets in plan.source_to_targets.items():
                if source in consumed_sources:
                    continue
                tensor = handle.get_tensor(source)
                if tensor.dtype != torch.float16:
                    raise ValueError("UMT5 materializer: unquantized tensor must be stored F16")
                _assign_alias_targets(encoder, targets, tensor)
                consumed_sources.add(source)
            allowed_support = set(plan.support_auxiliary)
            expected_aux = set(plan.quant_auxiliary) - allowed_support
            if consumed_auxiliary != expected_aux:
                raise ValueError(
                    "UMT5 materializer: unconsumed or unexpected quantization auxiliaries"
                )
        if consumed_sources != set(plan.source_to_targets):
            raise ValueError("UMT5 materializer: source consumption is incomplete")
        _validate_no_meta(encoder)
        encoder._latentslate_tokenizer_sha256 = tokenizer_sha256
        encoder._latentslate_umt5_config_fingerprint = plan.config_fingerprint
        encoder._latentslate_umt5_mapping_fingerprint = plan.mapping_fingerprint
        encoder._latentslate_umt5_artifact_identity = plan.identity
        return encoder
    except BaseException:
        _dematerialize(encoder)
        restored.clear()
        quant_layers.clear()
        quantized = tensor = value = scale = spiece = None
        raise


class UMT5EncoderResidencySession:
    """One-shot Engine-owned whole-UMT5 prompt residency.

    UMT5 is prompt-only, so this deliberately moves the full encoder for one
    explicit encode and returns all state to CPU before Wan transformer work.
    It never installs Accelerate/Diffusers hooks or changes stored weights.
    """

    def __init__(
        self, encoder: nn.Module, *, onload_device: torch.device | str, offload_device: str = "cpu"
    ) -> None:
        self.encoder = encoder
        self.onload_device = wan_residency._canonicalize_residency_device(
            torch.device(onload_device)
        )
        self.offload_device = torch.device(offload_device)
        if self.offload_device.type != "cpu":
            raise ValueError("UMT5 residency requires CPU as the offload device")
        self._snapshot = self._snapshot_state()
        self._entered = False
        self._closed = False
        self._encoding = False
        self._owner_thread_id: int | None = None
        self._lock = threading.RLock()

    @property
    def active(self) -> bool:
        return self._entered and not self._closed

    @property
    def tokenizer_sha256(self) -> str:
        value = getattr(self.encoder, "_latentslate_tokenizer_sha256", None)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("UMT5 encoder is missing its bound tokenizer identity")
        return value

    def __enter__(self) -> Self:
        with self._lock:
            if self._closed or self._entered:
                raise RuntimeError("UMT5 residency session is one-shot and cannot be re-entered")
            self._claim_global()
            try:
                self._owner_thread_id = threading.get_ident()
                self._validate_state()
                self._move_all(self.onload_device)
                self._assert_devices(self.onload_device)
                self._entered = True
                return self
            except BaseException:
                self._teardown(suppress_errors=True)
                raise

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if threading.get_ident() != self._owner_thread_id:
                raise RuntimeError("UMT5 residency close must run on the owning context thread")
            if self._encoding:
                raise RuntimeError("cannot close UMT5 residency while prompt encoding is active")
            self._teardown(suppress_errors=False)

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, *, sequence_length: int
    ) -> torch.Tensor:
        """Encode to explicit F16 `[B,S,H]`, zeroing all padded positions.

        Callers must choose ``sequence_length`` deliberately.  The planned
        Comfy-first policy is 512; tokenization itself is intentionally outside
        this loader boundary.
        """

        with self._lock:
            if not self.active or threading.get_ident() != self._owner_thread_id:
                raise RuntimeError(
                    "UMT5 prompt encode requires its active owning residency session"
                )
            if self._encoding:
                raise RuntimeError("UMT5 prompt encode is non-reentrant")
            self._encoding = True
        try:
            if (
                not isinstance(sequence_length, int)
                or isinstance(sequence_length, bool)
                or sequence_length <= 0
            ):
                raise ValueError("UMT5 prompt sequence_length must be a positive non-bool integer")
            if input_ids.ndim != 2 or input_ids.dtype not in {torch.int32, torch.int64}:
                raise ValueError("UMT5 prompt input_ids must be a 2D int32 or int64 tensor")
            if input_ids.shape[1] > sequence_length:
                raise ValueError("UMT5 prompt input_ids exceed the explicit sequence_length")
            vocab_size = int(getattr(self.encoder.config, "vocab_size", 0))
            if (
                vocab_size <= 0
                or bool((input_ids < 0).any())
                or bool((input_ids >= vocab_size).any())
            ):
                raise ValueError(
                    "UMT5 prompt input_ids must be nonnegative and within the encoder vocabulary"
                )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            if attention_mask.shape != input_ids.shape or attention_mask.dtype not in {
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError(
                    "UMT5 prompt attention_mask must match input_ids and use a boolean or integer dtype"
                )
            mask_bool = attention_mask.to(dtype=torch.bool)
            if not torch.equal(attention_mask, mask_bool.to(dtype=attention_mask.dtype)):
                raise ValueError(
                    "UMT5 prompt attention_mask must contain only binary 0 or 1 values"
                )
            pad = sequence_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0).to(self.onload_device)
            mask_bool = torch.nn.functional.pad(mask_bool, (0, pad), value=False).to(
                self.onload_device
            )
            output = self.encoder(input_ids=input_ids, attention_mask=mask_bool).last_hidden_state
            output = output.to(dtype=torch.float16)
            return output.masked_fill(~mask_bool.unsqueeze(-1), 0)
        except BaseException:
            self._teardown(suppress_errors=True)
            raise
        finally:
            with self._lock:
                self._encoding = False

    def _claim_global(self) -> None:
        with wan_residency._WAN_SESSION_GUARD_LOCK:
            if wan_residency._ACTIVE_WAN_SESSION is not None:
                raise RuntimeError("a Wan/UMT5 residency session is already active process-wide")
            wan_residency._ACTIVE_WAN_SESSION = self

    def _release_global(self) -> None:
        with wan_residency._WAN_SESSION_GUARD_LOCK:
            if wan_residency._ACTIVE_WAN_SESSION is self:
                wan_residency._ACTIVE_WAN_SESSION = None

    def _snapshot_state(self) -> dict[str, torch.dtype]:
        state = dict(self.encoder.named_parameters()) | dict(self.encoder.named_buffers())
        if not state:
            raise ValueError("UMT5 residency requires a materialized encoder state")
        if any(value.is_meta for value in state.values()):
            raise ValueError("UMT5 residency cannot accept meta state")
        return {name: value.dtype for name, value in state.items()}

    def _validate_state(self) -> None:
        state = dict(self.encoder.named_parameters()) | dict(self.encoder.named_buffers())
        if set(state) != set(self._snapshot) or any(
            value.is_meta or value.dtype != self._snapshot[name] for name, value in state.items()
        ):
            raise RuntimeError("UMT5 residency state changed after session construction")

    def _move_all(self, device: torch.device) -> None:
        self.encoder.to(device=device)
        for module in self.encoder.modules():
            if isinstance(module, NativeStoredLinear):
                module.move_stored_storage(device)

    def _assert_devices(self, requested: torch.device) -> None:
        for name, value in (
            dict(self.encoder.named_parameters()) | dict(self.encoder.named_buffers())
        ).items():
            if hasattr(value, "_qdata") and hasattr(value, "params"):
                continue
            if value.device != requested:
                raise RuntimeError(f"UMT5 residency state is on the wrong device: {name!r}")
        for module in self.encoder.modules():
            if isinstance(module, NativeStoredLinear):
                weight = module.weight
                if weight._qdata.device != requested or weight.params.scale.device != requested:
                    raise RuntimeError("UMT5 stored quant physical state is on the wrong device")
                if module.bias is not None and module.bias.device != requested:
                    raise RuntimeError("UMT5 stored quant bias is on the wrong device")

    def _teardown(self, *, suppress_errors: bool) -> None:
        error: BaseException | None = None
        try:
            self._move_all(self.offload_device)
            self._assert_devices(self.offload_device)
            self._validate_state()
        except BaseException as exc:  # noqa: BLE001
            error = exc
        finally:
            self._entered = False
            self._closed = True
            self._release_global()
        if error is not None and not suppress_errors:
            raise RuntimeError(f"UMT5 residency teardown failed: {error}") from error


def _source_targets(source: str) -> tuple[str, ...] | None:
    if source == "shared.weight":
        return ("shared.weight", "encoder.embed_tokens.weight")
    if source.startswith("encoder."):
        return (source,)
    return None


def _auxiliary_maps_to_source(key: str, contract: str, header_keys: set[str]) -> bool:
    for suffix in _AUX_SUFFIXES:
        if key.endswith(suffix):
            source = key.removesuffix(suffix) + ".weight"
            if source not in header_keys or _source_targets(source) is None:
                return False
            if contract == "comfy_legacy/scaled_fp8_e4m3fn":
                return suffix == ".scale_weight"
            return suffix in {".weight_scale", ".input_scale", ".comfy_quant"}
    return False


def _plan_quant_roles(
    contract: str,
    header: Mapping[str, Any],
    mapping: Mapping[str, tuple[str, ...]],
    skeleton: nn.Module,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Prove every stored quant role maps once to a concrete ``nn.Linear`` weight."""

    keys = set(header)
    quant_sources: list[str] = []
    errors: list[str] = []
    for source, targets in mapping.items():
        if not source.endswith(".weight"):
            continue
        stem = source.removesuffix(".weight")
        permitted = (
            {stem + ".scale_weight"}
            if contract == "comfy_legacy/scaled_fp8_e4m3fn"
            else {stem + ".weight_scale", stem + ".input_scale", stem + ".comfy_quant"}
        )
        present = {
            key for key in keys if key.startswith(stem + ".") and key.endswith(_AUX_SUFFIXES)
        }
        if not present:
            continue
        if present != permitted:
            errors.append("UMT5 quantization sidecar topology is partial or orphaned")
            continue
        if len(targets) != 1:
            errors.append("UMT5 stored quant cannot target tied/shared embedding aliases")
            continue
        target = targets[0]
        parent_path, _, attribute = target.rpartition(".")
        parent = skeleton.get_submodule(parent_path) if parent_path else skeleton
        if attribute != "weight" or not isinstance(parent, nn.Linear):
            errors.append("UMT5 stored quant is permitted only for exact nn.Linear weights")
            continue
        source_shape = _entry_shape(header[source], source)
        if len(source_shape) != 2:
            errors.append("UMT5 stored quant weight must be a 2D nn.Linear matrix")
            continue
        if contract == "comfy_quant/int8_tensorwise_convrot":
            marker = header[stem + ".comfy_quant"]
            scale = header[stem + ".weight_scale"]
            input_scale = header[stem + ".input_scale"]
            if (
                _entry_dtype(marker, stem + ".comfy_quant") != "U8"
                or len(_entry_shape(marker, stem + ".comfy_quant")) != 1
                or not 0 < _entry_shape(marker, stem + ".comfy_quant")[0] <= 64 * 1024
                or _entry_dtype(scale, stem + ".weight_scale") != "F32"
                or _entry_shape(scale, stem + ".weight_scale") != (source_shape[0], 1)
                or _entry_dtype(input_scale, stem + ".input_scale") != "F32"
                or _entry_shape(input_scale, stem + ".input_scale") != (source_shape[0], 1)
            ):
                errors.append(
                    "UMT5 ConvRot marker or scale header does not match the stored per-row contract"
                )
                continue
        else:
            scale = header[stem + ".scale_weight"]
            if (
                _entry_dtype(scale, stem + ".scale_weight") != "F32"
                or _entry_shape(scale, stem + ".scale_weight") != ()
            ):
                errors.append("UMT5 legacy FP8 scale header does not match the scalar F32 contract")
                continue
        quant_sources.append(source)
    if contract == "comfy_quant/int8_tensorwise_convrot":
        errors.extend(
            _validate_convrot_global_metadata(header.get("__metadata__"), quant_sources, header)
        )
    return tuple(sorted(quant_sources)), tuple(errors)


def _validate_convrot_global_metadata(
    metadata: Any, quant_sources: list[str], header: Mapping[str, Any]
) -> tuple[str, ...]:
    """Cross-check every ConvRot role against the artifact's header metadata map."""

    raw = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        return ("UMT5 ConvRot global metadata is invalid JSON",)
    layers = parsed.get("layers") if isinstance(parsed, dict) else None
    if not isinstance(layers, dict):
        return ("UMT5 ConvRot global metadata lacks a layer map",)
    errors: list[str] = []
    expected_stems = {source.removesuffix(".weight") for source in quant_sources}
    if set(layers) != expected_stems:
        errors.append("UMT5 ConvRot global metadata entries do not exactly match quantized roles")
    for stem in expected_stems:
        layer = layers.get(stem)
        if not isinstance(layer, dict):
            errors.append("UMT5 ConvRot global metadata has a missing layer entry")
            break
        group_size = layer.get("convrot_groupsize")
        shape = _entry_shape(header[stem + ".weight"], stem + ".weight")
        if (
            layer.get("format") != "int8_tensorwise"
            or layer.get("convrot") is not True
            or not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size <= 0
            or shape[1] % group_size
        ):
            errors.append("UMT5 ConvRot global metadata does not match the stored quantized weight")
            break
    return tuple(errors)


def _describe_quant_layers(handle, plan: UMT5StoredAdapterPlan) -> dict[str, StoredQuantizedLayer]:
    keys = plan.quant_sources
    if not keys:
        raise ValueError("UMT5 materializer: no stored quantized linear weights")
    return describe_stored_layers_from_handle(
        handle, identity=plan.identity, keys=keys, contract=plan.artifact_contract
    )


def _replace_parameter_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if not isinstance(getattr(parent, attribute, None), nn.Linear):
        raise TypeError(f"UMT5 materializer: {path!r} is not an nn.Linear target")
    setattr(parent, attribute, replacement)


def _assign_alias_targets(root: nn.Module, targets: tuple[str, ...], tensor: torch.Tensor) -> None:
    if not targets:
        raise ValueError("UMT5 materializer: source has no targets")
    parameter = nn.Parameter(tensor, requires_grad=False)
    for target in targets:
        parent_path, _, attribute = target.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        current = getattr(parent, attribute, None)
        if not isinstance(current, torch.Tensor) or tuple(current.shape) != tuple(tensor.shape):
            raise ValueError(f"UMT5 materializer: shape mismatch for {target!r}")
        if attribute not in parent._parameters:
            raise ValueError(f"UMT5 materializer: {target!r} is not a parameter")
        parent._parameters[attribute] = parameter


def _validate_no_meta(module: nn.Module) -> None:
    meta = [name for name, value in module.named_parameters() if value.is_meta]
    meta.extend(name for name, value in module.named_buffers() if value.is_meta)
    if meta:
        raise ValueError(f"UMT5 materializer: meta tensors remain: {meta[:3]}")


def _dematerialize(module: nn.Module) -> None:
    for child in module.modules():
        for name, parameter in tuple(child._parameters.items()):
            if parameter is not None:
                child._parameters[name] = nn.Parameter(
                    torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
                    requires_grad=False,
                )
        for name, buffer in tuple(child._buffers.items()):
            if buffer is not None:
                child._buffers[name] = torch.empty(
                    tuple(buffer.shape), dtype=buffer.dtype, device="meta"
                )


def _validate_header_contract(
    contract: str,
    header: Mapping[str, Any],
    mapping: Mapping[str, tuple[str, ...]],
    dense_dtypes: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate exact staged UMT5 storage roles before any tensor payload is read."""

    errors: list[str] = []
    dense_mismatches = sum(dtype != "F16" for dtype in dense_dtypes.values())
    if dense_mismatches:
        errors.append(
            f"UMT5 dense precision contract requires F16: {dense_mismatches} mismatch(es)"
        )
    expected_q_dtype = "F8_E4M3" if contract == "comfy_legacy/scaled_fp8_e4m3fn" else "I8"
    for source in set(mapping) - set(dense_dtypes):
        if _entry_dtype(header[source], source) != expected_q_dtype:
            errors.append("UMT5 quantized weight dtype differs from stored contract")
            break
        stem = source.removesuffix(".weight")
        required = (
            {stem + ".scale_weight"}
            if contract == "comfy_legacy/scaled_fp8_e4m3fn"
            else {stem + ".weight_scale", stem + ".input_scale", stem + ".comfy_quant"}
        )
        if not required <= set(header):
            errors.append("UMT5 quantized weight is missing required stored sidecars")
            break
    spiece = header.get("spiece_model")
    if (
        not isinstance(spiece, dict)
        or spiece.get("dtype") != "U8"
        or not isinstance(spiece.get("shape"), list)
        or len(spiece["shape"]) != 1
        or not isinstance(spiece["shape"][0], int)
        or isinstance(spiece["shape"][0], bool)
        or not 0 < spiece["shape"][0] <= _MAX_SENTENCEPIECE_BYTES
    ):
        errors.append("UMT5 artifact requires an embedded U8 spiece_model")
    sentinel = header.get("scaled_fp8")
    if contract == "comfy_legacy/scaled_fp8_e4m3fn":
        if (
            not isinstance(sentinel, dict)
            or sentinel.get("dtype") != "F8_E4M3"
            or sentinel.get("shape") != [0]
        ):
            errors.append("legacy UMT5 FP8 artifact requires the empty F8 scaled_fp8 sentinel")
    elif sentinel is not None:
        errors.append("ConvRot UMT5 artifact must not carry the legacy scaled_fp8 sentinel")
    return tuple(errors)


def _entry_shape(entry: Any, key: str) -> tuple[int, ...]:
    if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
        raise TypeError(f"UMT5 adapter: invalid source shape for {key!r}")
    return tuple(entry["shape"])


def _entry_dtype(entry: Any, key: str) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
        raise TypeError(f"UMT5 adapter: invalid source dtype for {key!r}")
    return entry["dtype"]


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(sorted(config.items())), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mapping_fingerprint(
    contract: str,
    mapping: Mapping[str, tuple[str, ...]],
    quant_sources: tuple[str, ...],
    dense: Mapping[str, str],
    quant_auxiliary: list[str] | tuple[str, ...],
) -> str:
    raw = json.dumps(
        {
            "contract": contract,
            "mapping": sorted(mapping.items()),
            "quant_sources": list(quant_sources),
            "dense": sorted(dense.items()),
            "quant_auxiliary": sorted(quant_auxiliary),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
