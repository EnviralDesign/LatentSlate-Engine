"""Engine-owned Kitchen materialization for the LTX 2.3 mixed Gemma text path.

The optimized LTX artifact is one multimodal SafeTensors file.  LTX prompt
conditioning uses only its 626 ``model.*`` language-model weights; the vision
tower, projector, and SentencePiece payload are deliberately classified but
never loaded by this component. Quantized language linears retain their stored
FP8/NVFP4 bytes, but strict Comfy parity dequantizes/casts each resident layer
and uses ordinary linear math (`full_precision_mm=True`).
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from ..stored_quant import (
    read_safetensors_header_bytes,
    restore_global_fp8_tensor,
    restore_nvfp4_tensor,
)
from .framework.residency.dynamic import (
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
    DynamicResidencyUnavailable,
)
from .framework.residency.leaf import LeafResidencyDescriptor, LeafResidencyScheduler
from .framework.stored_quant import StoredFP8Linear, StoredNVFP4Linear
from .ltx23_av_stored_adapter import (
    LTX23LeafSchedule,
    LTX23ModuleBinding,
    _assign_storage_slot,
    capture_ltx23_leaf_storages,
    capture_ltx23_module_storage,
)

LTX23_GEMMA_MIXED_SCHEMA_SHA256 = (
    "ddf523b18b1a724da6d4a3b0a97d4305ad3ad02a89ab7ada299663a9047040cd"
)
LTX23_GEMMA_TEXT_LORA_SCHEMA_SHA256 = (
    "601c8857a7d830f05f80792e044f97df6df8ff125079d5a305f3de5a2999d027"
)
LTX23_GEMMA_MIXED_CONTRACT = "comfy_quant/mixed_fp8_nvfp4"

_GEMMA_SIZE_BYTES = 9_447_702_218
_GEMMA_TENSOR_COUNT = 2_040
_GEMMA_DTYPES = ("BF16", "F32", "F8_E4M3", "U8")
_TEXT_BASE_COUNT = 626
_TEXT_NVFP4_COUNT = 302
_TEXT_FP8_COUNT = 34
_IGNORED_AUXILIARY_COUNT = 440
_TEXT_LORA_SIZE_BYTES = 628_203_616
_TEXT_LORA_TENSOR_COUNT = 1_000
_TEXT_LORA_TARGET_COUNT = 337
_VISION_LORA_TARGET_COUNT = 163
_TEXT_LORA_PREFIX = "text_encoders.transformer."
_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_STORED_LINEAR_TYPES = (StoredFP8Linear, StoredNVFP4Linear)
_HOST_REGISTER_ALLOWED_TYPES = frozenset({"Tensor", "Parameter", "QuantizedTensor"})
_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712


def _system_memory_bytes() -> int | None:
    if os.name == "nt":

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _host_registration_budget_bytes() -> int:
    total = _system_memory_bytes()
    return 4 * 1024**3 if total is None else max(0, int(total * 0.40))


class _LTX23HostRegistrationLedger:
    """Best-effort in-place CUDA host registrations owned by one text stage."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        self.seen: set[tuple[int, int]] = set()
        self.owned: dict[tuple[int, int], torch.Tensor] = {}
        self.counts = {
            "candidates": 0,
            "candidate_bytes": 0,
            "deduplicated_aliases": 0,
            "already_registered": 0,
            "already_registered_bytes": 0,
            "attempts": 0,
            "attempt_bytes": 0,
            "successes": 0,
            "registered_bytes": 0,
            "failures": 0,
            "failure_bytes": 0,
            "ineligible": 0,
            "ineligible_bytes": 0,
            "unregistered": 0,
            "unregistered_bytes": 0,
            "unregister_failures": 0,
            "unregister_failure_bytes": 0,
        }
        self.categories = {
            "unsupported_type": 0,
            "non_cpu": 0,
            "noncontiguous": 0,
            "zero_pointer": 0,
            "budget_exceeded": 0,
            "eligibility_error": 0,
            "register_error": 0,
            "unregister_error": 0,
        }

    def consider(self, value: torch.Tensor) -> None:
        size = int(getattr(value, "nbytes", 0))
        self.counts["candidates"] += 1
        self.counts["candidate_bytes"] += max(0, size)
        try:
            ptr = int(value.data_ptr())
        except (RuntimeError, TypeError, ValueError):
            self._ineligible("eligibility_error", max(0, size))
            return
        key = (ptr, max(0, size))
        if key in self.seen:
            self.counts["deduplicated_aliases"] += 1
            return
        self.seen.add(key)
        if type(value).__name__ not in _HOST_REGISTER_ALLOWED_TYPES:
            self._ineligible("unsupported_type", max(0, size))
            return
        if value.device.type != "cpu":
            self._ineligible("non_cpu", max(0, size))
            return
        if not value.is_contiguous():
            self._ineligible("noncontiguous", max(0, size))
            return
        if ptr == 0 or size <= 0:
            self._ineligible("zero_pointer", max(0, size))
            return
        if value.is_pinned():
            self.counts["already_registered"] += 1
            self.counts["already_registered_bytes"] += size
            return
        if self.counts["registered_bytes"] + size > self.budget_bytes:
            self._ineligible("budget_exceeded", size)
            return
        self.counts["attempts"] += 1
        self.counts["attempt_bytes"] += size
        try:
            result = torch.cuda.cudart().cudaHostRegister(ptr, size, 1)
        except (RuntimeError, TypeError, ValueError):
            result = -1
        if result == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            self._discard_cuda_registration_error()
            self.counts["attempts"] -= 1
            self.counts["attempt_bytes"] -= size
            self.counts["already_registered"] += 1
            self.counts["already_registered_bytes"] += size
            return
        if result != 0:
            self._discard_cuda_registration_error()
            self.counts["failures"] += 1
            self.counts["failure_bytes"] += size
            self.categories["register_error"] += 1
            return
        self.owned[key] = value
        self.counts["successes"] += 1
        self.counts["registered_bytes"] += size

    def unregister_owned(self) -> list[BaseException]:
        errors: list[BaseException] = []
        for (ptr, size), _value in tuple(self.owned.items()):
            try:
                result = torch.cuda.cudart().cudaHostUnregister(ptr)
            except (RuntimeError, TypeError, ValueError) as exc:
                result = -1
                errors.append(exc)
            if result != 0:
                self.counts["unregister_failures"] += 1
                self.counts["unregister_failure_bytes"] += size
                self.categories["unregister_error"] += 1
                if not errors or not isinstance(errors[-1], RuntimeError):
                    errors.append(RuntimeError("CUDA host unregistration failed"))
                continue
            self.owned.pop((ptr, size), None)
            self.counts["unregistered"] += 1
            self.counts["unregistered_bytes"] += size
        return errors

    def provenance(self) -> dict[str, Any]:
        return {
            "policy": "comfy_best_effort_in_place_cuda_host_register",
            "lifecycle": "text_stage_onload_through_synchronized_offload",
            "budget_bytes": self.budget_bytes,
            **self.counts,
            "owned_active": len(self.owned),
            "owned_active_bytes": sum(size for _, size in self.owned),
            "categories": dict(self.categories),
        }

    def _ineligible(self, category: str, size: int) -> None:
        self.counts["ineligible"] += 1
        self.counts["ineligible_bytes"] += size
        self.categories[category] += 1

    @staticmethod
    def _discard_cuda_registration_error() -> None:
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass


@dataclass(frozen=True, slots=True)
class LTX23GemmaMixedTextPlan:
    """Exact text-only subset of the LTX 2.3 mixed Gemma artifact."""

    identity: ArtifactIdentity
    schema_sha256: str
    quantized_formats: Mapping[str, str]
    dense_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    ignored_auxiliary_sources: tuple[str, ...]
    base_spans: Mapping[str, LTX23SafetensorSpan] = field(
        default_factory=lambda: MappingProxyType({})
    )
    header_size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class LTX23SafetensorSpan:
    """Immutable authenticated absolute payload extent for one base field."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class LTX23GemmaTextLoraPlan:
    """Structural contract for LTX's fixed Gemma prompt LoRA.

    The official adapter's embedding and linear targets are all executable via
    Engine-owned additive wrappers without merging or rematerializing the base.
    Vision targets remain explicitly ignored by this text-only component.
    """

    identity: ArtifactIdentity
    schema_sha256: str
    text_targets: tuple[str, ...]
    ignored_vision_targets: tuple[str, ...]
    embedding_target: str
    rank: int
    pairs: tuple[LTX23GemmaTextLoraTarget, ...]


@dataclass(frozen=True, slots=True)
class LTX23GemmaTextLoraTarget:
    """One exactly mapped additive LoRA pair, retained in stored orientation."""

    module_name: str
    down_key: str
    up_key: str
    kind: str


class LTX23GemmaEmbeddingLora(nn.Module):
    """Sparse additive LoRA for Gemma's tied token embedding matrix.

    The stored pair has ``down=[rank, hidden]`` and ``up=[vocab, rank]``.
    Looking up ``up`` first gives precisely the selected rows of ``up @ down``
    without materializing that 262,208-by-3,840 delta matrix.
    """

    def __init__(self, base: nn.Embedding) -> None:
        super().__init__()
        if not isinstance(base, nn.Embedding):
            raise TypeError("LTX Gemma embedding LoRA requires an nn.Embedding base")
        self.base = base
        self._lora_adapters = nn.ModuleDict()
        self.lora_dispatch_count = 0
        self.patch_seed_key = "gemma3_12b.transformer.model.embed_tokens"
        self.patched_resident_merge_misses = 0
        self.patched_resident_hits = 0
        self.signature_none_patch_rematerializations = 0
        self.patched_resident_writebacks = 0

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # Comfy patches the complete tied embedding before either consumer.
        weight = self._merged_weight()
        output = torch.nn.functional.embedding(
            input_ids,
            weight,
            self.base.padding_idx,
            self.base.max_norm,
            self.base.norm_type,
            self.base.scale_grad_by_freq,
            self.base.sparse,
        )
        embed_scale = getattr(self.base, "embed_scale", 1.0)
        if out_dtype is not None:
            output = output.to(dtype=out_dtype)
        output = output * embed_scale
        return output

    def tied_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply one tied linear from the same effective embedding weight."""

        return torch.nn.functional.linear(hidden_states, self._merged_weight())

    def _active_patch_fingerprint(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (name, *adapter.patch_identity, float(adapter.strength))
            for name, adapter in self._lora_adapters.items()
            if adapter.strength != 0.0
        )

    def _merged_weight(self) -> torch.Tensor:
        weight = self.base.weight
        fingerprint = self._active_patch_fingerprint()
        marker_name = "_latentslate_ltx23_patch_fingerprint"
        resident_fingerprint = getattr(weight, marker_name, None)
        if not fingerprint:
            if resident_fingerprint is not None:
                raise RuntimeError(
                    "LTX Gemma embedding retained patched bytes after its patch state changed"
                )
            return weight

        self.lora_dispatch_count += 1
        cacheable = bool(
            getattr(weight, "_latentslate_aimdo_signature_cacheable", False)
        )
        if resident_fingerprint is not None:
            if not cacheable or resident_fingerprint != fingerprint:
                raise RuntimeError(
                    "LTX Gemma embedding patch fingerprint changed on one resident value"
                )
            self.patched_resident_hits += 1
            return weight

        merged = weight
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            down = adapter.down.to(device=weight.device, dtype=weight.dtype)
            up = adapter.up.to(device=weight.device, dtype=weight.dtype)
            merged = merged + torch.mm(up, down) * adapter.strength
        if not cacheable:
            self.signature_none_patch_rematerializations += 1
            return merged

        with torch.no_grad():
            weight.copy_(merged)
        setattr(weight, marker_name, fingerprint)
        self.patched_resident_merge_misses += 1
        self.patched_resident_writebacks += 1
        return merged

    def add_lora_adapter(self, name: str, down: torch.Tensor, up: torch.Tensor) -> None:
        if not _ADAPTER_NAME.fullmatch(name):
            raise ValueError("LTX Gemma LoRA adapter name is unsafe")
        if name in self._lora_adapters:
            raise ValueError(f"LTX Gemma LoRA adapter {name!r} is already loaded")
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] != up.shape[1]
            or down.shape[1] != self.base.embedding_dim
            or up.shape[0] != self.base.num_embeddings
            or not down.dtype.is_floating_point
            or not up.dtype.is_floating_point
        ):
            raise ValueError("LTX Gemma embedding LoRA geometry differs from embed_tokens")
        adapter = _LTX23GemmaEmbeddingAdapter(down, up)
        self._lora_adapters[name] = adapter

    def set_lora_strength(self, name: str, strength: float) -> None:
        try:
            adapter = self._lora_adapters[name]
        except KeyError as exc:
            raise KeyError(f"LTX Gemma LoRA adapter {name!r} is not loaded") from exc
        value = float(strength)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("LTX Gemma LoRA strength must be finite")
        adapter.strength = value

    def delete_lora_adapter(self, name: str) -> None:
        if name in self._lora_adapters:
            self._lora_adapters.pop(name)


class _LTX23GemmaEmbeddingAdapter(nn.Module):
    def __init__(self, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        self.down = nn.Parameter(down.contiguous(), requires_grad=False)
        self.up = nn.Parameter(up.contiguous(), requires_grad=False)
        self.patch_identity = (id(self.down), id(self.up))
        self.strength = 1.0


@dataclass(slots=True)
class LTX23GemmaTextLoraApplication:
    """One installed, Engine-owned LTX Gemma adapter and its dispatch proof."""

    model: Any
    plan: LTX23GemmaTextLoraPlan
    adapter_name: str
    target_modules: tuple[str, ...]
    _replaced_embedding: nn.Embedding | None

    def set_strength(self, strength: float) -> None:
        for module_name in self.target_modules:
            _lora_module(self.model, module_name).set_lora_strength(self.adapter_name, strength)

    def dispatch_snapshot(self) -> dict[str, int]:
        return {
            name: int(_lora_module(self.model, name).lora_dispatch_count)
            for name in self.target_modules
        }

    def verify_dispatch(self, before: Mapping[str, int]) -> dict[str, int | str]:
        after = self.dispatch_snapshot()
        if set(after) != set(before):
            raise RuntimeError("LTX Gemma LoRA target set changed during generation")
        deltas = {name: after[name] - int(before[name]) for name in after}
        missed = sorted(name for name, value in deltas.items() if value <= 0)
        if missed:
            raise RuntimeError(
                f"LTX Gemma LoRA did not execute on {len(missed)} selected targets: {missed[:3]}"
            )
        return {
            "backend": "engine-native/additive-lora",
            "target_module_count": len(deltas),
            "linear_target_count": len(deltas) - 1,
            "embedding_target": self.plan.embedding_target,
            "ignored_vision_target_count": len(self.plan.ignored_vision_targets),
            "total_dispatches": sum(deltas.values()),
            "minimum_target_dispatches": min(deltas.values()),
            "maximum_target_dispatches": max(deltas.values()),
        }

    def provenance(self) -> dict[str, int | str]:
        return {
            "backend": "engine-native/additive-lora",
            "target_module_count": len(self.target_modules),
            "linear_target_count": len(self.target_modules) - 1,
            "embedding_target": self.plan.embedding_target,
            "ignored_vision_target_count": len(self.plan.ignored_vision_targets),
        }

    def remove(self) -> None:
        for module_name in self.target_modules:
            _lora_module(self.model, module_name).delete_lora_adapter(self.adapter_name)
        if self._replaced_embedding is not None:
            language = self.model.model.language_model
            if isinstance(language.embed_tokens, LTX23GemmaEmbeddingLora):
                language.embed_tokens = self._replaced_embedding
            self._replaced_embedding = None
        _retie_lm_head(self.model)


def ltx23_gemma_source_to_transformers(source: str) -> str:
    """Map one stored language weight into Gemma3's Transformers namespace."""

    if not source.startswith("model.") or not source.endswith(".weight"):
        raise ValueError(f"LTX Gemma source is not a language-model weight: {source!r}")
    return "model.language_model." + source.removeprefix("model.")


def ltx23_gemma_comfy_seed_key(module_name: str) -> str:
    """Map the Engine shell path to Comfy's pinned Gemma stochastic seed key."""

    prefix = "model.language_model."
    if not module_name.startswith(prefix) or module_name.endswith(".weight"):
        raise ValueError(f"LTX Gemma module path is not canonical: {module_name!r}")
    return "gemma3_12b.transformer.model." + module_name.removeprefix(prefix)


def plan_ltx23_gemma_mixed_text_encoder(path: Path) -> LTX23GemmaMixedTextPlan:
    """Validate and classify the immutable LTX 2.3 Gemma mixed artifact."""

    from safetensors import safe_open

    probe = probe_artifact(Path(path))
    errors: list[str] = []
    if probe.format != "safetensors":
        errors.append("container is not SafeTensors")
    if probe.identity.size_bytes != _GEMMA_SIZE_BYTES:
        errors.append("file size differs from the pinned artifact")
    if probe.schema_sha256 != LTX23_GEMMA_MIXED_SCHEMA_SHA256:
        errors.append("key/shape/dtype schema differs from the pinned artifact")
    if probe.tensor_count != _GEMMA_TENSOR_COUNT or probe.tensor_dtypes != _GEMMA_DTYPES:
        errors.append("tensor count or stored dtypes differ from the pinned artifact")
    if errors:
        raise ValueError("LTX 2.3 Gemma text contract failed: " + "; ".join(errors))

    formats: dict[str, str] = {}
    dense: list[str] = []
    auxiliary: set[str] = set()
    ignored: list[str] = []
    with safe_open(str(probe.identity.path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        text_keys = {key for key in keys if key.startswith("model.")}
        ignored = sorted(keys - text_keys)
        for source in sorted(text_keys):
            if source.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
                continue
            if not source.endswith(".weight"):
                raise ValueError(f"LTX Gemma text state has an unsupported tensor: {source}")
            view = handle.get_slice(source)
            dtype = view.get_dtype()
            if dtype == "BF16":
                dense.append(source)
                continue
            stem = source.removesuffix(".weight")
            quant_format, consumed_sidecars = _validate_quantized_weight(handle, keys, stem, dtype)
            formats[stem] = quant_format
            auxiliary.update(consumed_sidecars)

        text_base = set(dense) | {stem + ".weight" for stem in formats}
        expected_text_keys = text_base | auxiliary
        if expected_text_keys != text_keys:
            extra = sorted(text_keys - expected_text_keys)
            raise ValueError(f"LTX Gemma text source-role closure changed: {extra[:3]}")
        if (
            len(text_base) != _TEXT_BASE_COUNT
            or len(dense) != _TEXT_BASE_COUNT - _TEXT_NVFP4_COUNT - _TEXT_FP8_COUNT
            or sum(value == "nvfp4" for value in formats.values()) != _TEXT_NVFP4_COUNT
            or sum(value == "float8_e4m3fn" for value in formats.values()) != _TEXT_FP8_COUNT
            or len(ignored) != _IGNORED_AUXILIARY_COUNT
        ):
            raise ValueError("LTX Gemma text source coverage changed")
        _validate_ignored_auxiliaries(ignored)

    raw_header, header = read_safetensors_header_bytes(
        probe.identity.path, probe.identity.size_bytes
    )
    physical_keys = set(dense) | {
        stem + suffix
        for stem, quant_format in formats.items()
        for suffix in (
            (".weight", ".weight_scale")
            if quant_format == "float8_e4m3fn"
            else (".weight", ".weight_scale", ".weight_scale_2")
        )
    }
    spans = _authenticated_base_spans(
        header,
        physical_keys=physical_keys,
        payload_offset=8 + len(raw_header),
        file_size=probe.identity.size_bytes,
    )
    return LTX23GemmaMixedTextPlan(
        probe.identity,
        probe.schema_sha256,
        MappingProxyType(dict(sorted(formats.items()))),
        tuple(sorted(dense)),
        tuple(sorted(auxiliary)),
        tuple(ignored),
        MappingProxyType(spans),
        len(raw_header),
    )


def revalidate_ltx23_gemma_mixed_text_encoder(plan: LTX23GemmaMixedTextPlan) -> bool:
    """Re-plan the header and identity before materializing any payload bytes."""

    try:
        refreshed = plan_ltx23_gemma_mixed_text_encoder(plan.identity.path)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed == plan and revalidate_artifact(plan.identity)


_SPAN_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
_SPAN_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
}


def _authenticated_base_spans(
    header: Mapping[str, Any],
    *,
    physical_keys: set[str],
    payload_offset: int,
    file_size: int,
) -> dict[str, LTX23SafetensorSpan]:
    spans: dict[str, LTX23SafetensorSpan] = {}
    for key in sorted(physical_keys):
        entry = header.get(key)
        if not isinstance(entry, Mapping):
            raise TypeError(f"LTX Gemma base span is missing: {key}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            dtype not in _SPAN_DTYPE_BYTES
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"LTX Gemma base span metadata is invalid: {key}")
        start, end = offsets
        expected = int(torch.Size(shape).numel()) * _SPAN_DTYPE_BYTES[dtype]
        absolute = payload_offset + start
        if start < 0 or end - start != expected or absolute + expected > file_size:
            raise ValueError(f"LTX Gemma base span bounds are invalid: {key}")
        spans[key] = LTX23SafetensorSpan(
            key, dtype, tuple(shape), absolute, expected
        )
    return spans


def _build_ltx23_gemma_file_backed_shell(
    plan: LTX23GemmaMixedTextPlan,
    support_root: Path,
) -> Any:
    """Build a Gemma shell without reading or retaining base payload tensors."""

    from accelerate import init_empty_weights
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout,
        TensorCoreNVFP4Layout,
    )
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    from .framework.residency.aimdo import AimdoFileBackedValue, AimdoFileSpan
    from .ltx23_gemma_comfy import build_ltx23_comfy_gemma

    if not revalidate_ltx23_gemma_mixed_text_encoder(plan) or not plan.base_spans:
        raise ValueError("LTX 2.3 Gemma file-backed plan changed after planning")
    config = Gemma3Config.from_pretrained(Path(support_root), local_files_only=True)
    with init_empty_weights():
        if hasattr(config, "text_config"):
            model = build_ltx23_comfy_gemma(config)
        else:
            # Structural tests inject a deliberately incomplete tiny config.
            model = Gemma3ForConditionalGeneration(config)
    _materialize_ltx23_gemma_runtime_buffers(model)
    model.tie_weights()
    target_state = model.state_dict()
    source_to_target = {
        source: ltx23_gemma_source_to_transformers(source)
        for source in (*plan.dense_sources, *(stem + ".weight" for stem in plan.quantized_formats))
    }
    language_targets = {
        key for key in target_state if key.startswith("model.language_model.")
    }
    if set(source_to_target.values()) != language_targets:
        raise RuntimeError("LTX Gemma file-backed shell coverage changed")

    descriptors: dict[int, AimdoFileBackedValue] = {}
    quantized_modules: dict[str, str] = {}

    def file_span(key: str) -> AimdoFileSpan:
        span = plan.base_spans[key]
        return AimdoFileSpan(
            "ltx23_gemma_base",
            span.key,
            span.offset,
            span.size,
            _SPAN_TORCH_DTYPES[span.dtype],
            span.shape,
        )

    for stem, quant_format in plan.quantized_formats.items():
        source = stem + ".weight"
        module_name = source_to_target[source].removesuffix(".weight")
        module = model.get_submodule(module_name)
        if type(module) is not nn.Linear or module.bias is not None:
            raise TypeError(f"LTX Gemma quantized target is not a bias-free Linear: {module_name}")
        expected_shape = tuple(module.weight.shape)
        q_span = plan.base_spans[source]
        qdata = torch.empty(q_span.shape, dtype=_SPAN_TORCH_DTYPES[q_span.dtype], device="meta")
        if quant_format == "float8_e4m3fn":
            if q_span.shape != expected_shape:
                raise RuntimeError(f"LTX Gemma FP8 meta shape mismatch: {stem}")
            scale_span = plan.base_spans[stem + ".weight_scale"]
            scale = torch.empty(scale_span.shape, dtype=torch.float32, device="meta")
            params = TensorCoreFP8Layout.Params(
                scale=scale, orig_dtype=torch.bfloat16, orig_shape=expected_shape
            )
            weight = QuantizedTensor(qdata, "TensorCoreFP8Layout", params)
            replacement: nn.Module = StoredFP8Linear(weight, input_scale=None)
            spans = (file_span(source), file_span(stem + ".weight_scale"))
        else:
            logical_shape = (q_span.shape[0], q_span.shape[1] * 2)
            if logical_shape != expected_shape:
                raise RuntimeError(f"LTX Gemma NVFP4 meta shape mismatch: {stem}")
            block_span = plan.base_spans[stem + ".weight_scale"]
            tensor_span = plan.base_spans[stem + ".weight_scale_2"]
            params = TensorCoreNVFP4Layout.Params(
                scale=torch.empty(tensor_span.shape, dtype=torch.float32, device="meta"),
                orig_dtype=torch.bfloat16,
                orig_shape=logical_shape,
                block_scale=torch.empty(
                    block_span.shape, dtype=torch.float8_e4m3fn, device="meta"
                ),
            )
            weight = QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)
            replacement = StoredNVFP4Linear(weight, input_scale=None)
            spans = (
                file_span(source),
                file_span(stem + ".weight_scale_2"),
                file_span(stem + ".weight_scale"),
            )
        replacement.patch_seed_key = ltx23_gemma_comfy_seed_key(module_name)
        parent_path, _, leaf = module_name.rpartition(".")
        setattr(model.get_submodule(parent_path), leaf, replacement)
        descriptors[id(replacement.weight)] = AimdoFileBackedValue(
            replacement.weight, spans
        )
        quantized_modules[module_name] = quant_format

    for source in plan.dense_sources:
        target = source_to_target[source]
        value = model.get_parameter(target)
        span = plan.base_spans[source]
        if not value.is_meta or tuple(value.shape) != span.shape:
            raise RuntimeError(f"LTX Gemma dense meta shell mismatch: {source}")
        if value.dtype is not torch.bfloat16:
            parent_path, _, leaf = target.rpartition(".")
            parent = model.get_submodule(parent_path)
            value = nn.Parameter(
                torch.empty(span.shape, dtype=torch.bfloat16, device="meta"),
                requires_grad=value.requires_grad,
            )
            parent._parameters[leaf] = value
        descriptors[id(value)] = AimdoFileBackedValue(value, (file_span(source),))
    model.tie_weights()
    _retie_lm_head(model)
    model._latentslate_ltx23_gemma_text_only = True
    model._latentslate_ltx23_gemma_source_backed = True
    model._latentslate_ltx23_gemma_source_descriptors = descriptors
    model._latentslate_ltx23_gemma_plan = plan
    model._latentslate_ltx23_gemma_support_root = Path(support_root)
    model._latentslate_ltx23_gemma_artifact_identity = plan.identity
    model._latentslate_ltx23_gemma_quantization_contract = LTX23_GEMMA_MIXED_CONTRACT
    model._latentslate_ltx23_gemma_quant_modules = MappingProxyType(quantized_modules)
    model.eval()
    return model


def load_ltx23_gemma_mixed_text_encoder(
    plan: LTX23GemmaMixedTextPlan,
    support_root: Path,
    *,
    source_backed: bool = True,
) -> Any:
    """Build the authenticated file-backed shell, or explicit CPU fallback."""

    if not source_backed:
        return _load_ltx23_gemma_mixed_text_encoder_cpu(plan, support_root)
    return _build_ltx23_gemma_file_backed_shell(plan, support_root)


def _load_ltx23_gemma_mixed_text_encoder_cpu(
    plan: LTX23GemmaMixedTextPlan,
    support_root: Path,
) -> Any:
    """Restore just LTX's language model into a Gemma3 meta shell on CPU.

    ``support_root`` is the local LTX repository ``text_encoder`` support
    directory, not a model download location.  Vision/projector state remains
    meta by design and must not be used through this text-only component.
    """

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    from .ltx23_gemma_comfy import build_ltx23_comfy_gemma

    if not revalidate_ltx23_gemma_mixed_text_encoder(plan):
        raise ValueError("LTX 2.3 Gemma text artifact changed after planning")
    config = Gemma3Config.from_pretrained(Path(support_root), local_files_only=True)
    with init_empty_weights():
        if hasattr(config, "text_config"):
            model = build_ltx23_comfy_gemma(config)
        else:
            model = Gemma3ForConditionalGeneration(config)
    _materialize_ltx23_gemma_runtime_buffers(model)
    model.tie_weights()
    target_state = model.state_dict()
    source_to_target = {
        source: ltx23_gemma_source_to_transformers(source)
        for source in (*plan.dense_sources, *(stem + ".weight" for stem in plan.quantized_formats))
    }
    expected_targets = set(source_to_target.values())
    language_targets = {key for key in target_state if key.startswith("model.language_model.")}
    if expected_targets != language_targets:
        missing = sorted(language_targets - expected_targets)
        extra = sorted(expected_targets - language_targets)
        raise RuntimeError(
            "LTX Gemma artifact does not exactly cover the Transformers language shell: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    consumed: set[str] = set()
    quantized_modules: dict[str, str] = {}
    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("LTX 2.3 Gemma text artifact changed before materialization")
        for stem, quant_format in plan.quantized_formats.items():
            source = stem + ".weight"
            target = source_to_target[source]
            module_name = target.removesuffix(".weight")
            module = model.get_submodule(module_name)
            if type(module) is not nn.Linear or module.bias is not None:
                raise TypeError(f"LTX Gemma quantized target is not a bias-free Linear: {module_name}")
            qdata = handle.get_tensor(source)
            expected_shape = tuple(module.weight.shape)
            if quant_format == "float8_e4m3fn":
                if tuple(qdata.shape) != expected_shape:
                    raise RuntimeError(f"LTX Gemma FP8 shape mismatch: {stem}")
                weight = restore_global_fp8_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    torch.bfloat16,
                )
                replacement: nn.Module = StoredFP8Linear(weight, input_scale=None)
            else:
                logical_shape = (qdata.shape[0], qdata.shape[1] * 2)
                if logical_shape != expected_shape:
                    raise RuntimeError(f"LTX Gemma NVFP4 shape mismatch: {stem}")
                weight = restore_nvfp4_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    handle.get_tensor(stem + ".weight_scale_2"),
                    logical_shape,
                    torch.bfloat16,
                )
                replacement = StoredNVFP4Linear(weight, input_scale=None)
            replacement.patch_seed_key = ltx23_gemma_comfy_seed_key(module_name)
            parent_path, _, leaf = module_name.rpartition(".")
            setattr(model.get_submodule(parent_path), leaf, replacement)
            consumed.add(source)
            quantized_modules[module_name] = quant_format

        for source in plan.dense_sources:
            target = source_to_target[source]
            value = handle.get_tensor(source)
            if tuple(value.shape) != tuple(target_state[target].shape):
                raise RuntimeError(f"LTX Gemma dense text shape mismatch: {source}")
            set_module_tensor_to_device(model, target, "cpu", value=value, dtype=torch.bfloat16)
            consumed.add(source)

    expected_sources = set(source_to_target)
    if consumed != expected_sources:
        raise RuntimeError("LTX Gemma text materialization is incomplete")
    model.tie_weights()
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise RuntimeError("LTX Gemma tied language head no longer aliases embed_tokens")
    unresolved = [name for name in language_targets if model.state_dict()[name].is_meta]
    if unresolved:
        raise RuntimeError(f"LTX Gemma text encoder retained meta state: {unresolved[:3]}")
    model._latentslate_ltx23_gemma_text_only = True
    model._latentslate_ltx23_gemma_artifact_identity = plan.identity
    model._latentslate_ltx23_gemma_quantization_contract = LTX23_GEMMA_MIXED_CONTRACT
    model._latentslate_ltx23_gemma_quant_modules = MappingProxyType(quantized_modules)
    model.eval()
    return model


def _materialize_ltx23_gemma_runtime_buffers(model: Any) -> None:
    """Restore Gemma's nonpersistent language buffers outside empty-init mode.

    Transformers' Gemma3 rotary module registers RoPE frequencies as
    ``persistent=False`` buffers.  Accelerate therefore creates them on meta
    while they are absent from the checkpoint and from ``state_dict()``.  The
    LTX text artifact deliberately materializes only checkpoint-backed language
    weights, so rebuild the small deterministic rotary helper on CPU before any
    residency move.  This is not a model-weight conversion or a vision load.
    """

    try:
        language_model = model.model.language_model
    except AttributeError as exc:
        raise TypeError("LTX Gemma text shell lacks its language model") from exc
    if not any(buffer.is_meta for _, buffer in language_model.named_buffers()):
        return
    try:
        rotary = language_model.rotary_emb
    except AttributeError as exc:
        raise TypeError("LTX Gemma text shell lacks its language rotary module") from exc
    if any(buffer.is_meta for _, buffer in language_model.named_buffers()):
        language_model.rotary_emb = type(rotary)(rotary.config)
    unresolved = [name for name, buffer in language_model.named_buffers() if buffer.is_meta]
    if unresolved:
        raise RuntimeError(
            "LTX Gemma text shell retained meta runtime buffers: "
            f"count={len(unresolved)}"
        )


def plan_ltx23_gemma_text_lora(path: Path) -> LTX23GemmaTextLoraPlan:
    """Prove the exact fixed prompt-LoRA closure without installing it."""

    from safetensors import safe_open

    probe = probe_artifact(Path(path))
    errors: list[str] = []
    if probe.format != "safetensors":
        errors.append("container is not SafeTensors")
    if probe.identity.size_bytes != _TEXT_LORA_SIZE_BYTES:
        errors.append("file size differs from the pinned artifact")
    if probe.schema_sha256 != LTX23_GEMMA_TEXT_LORA_SCHEMA_SHA256:
        errors.append("key/shape/dtype schema differs from the pinned artifact")
    if probe.tensor_count != _TEXT_LORA_TENSOR_COUNT or probe.tensor_dtypes != ("BF16",):
        errors.append("tensor count or stored dtypes differ from the pinned artifact")
    if errors:
        raise ValueError("LTX 2.3 Gemma text LoRA contract failed: " + "; ".join(errors))

    pairs: dict[str, dict[str, str]] = {}
    rank: int | None = None
    text_pairs: list[LTX23GemmaTextLoraTarget] = []
    with safe_open(str(probe.identity.path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            role = _lora_role(key)
            if role is None:
                raise ValueError(f"LTX Gemma text LoRA tensor is unsupported: {key}")
            stem, side = role
            entry = pairs.setdefault(stem, {})
            if side in entry:
                raise ValueError(f"LTX Gemma text LoRA has duplicate {side} tensor: {stem}")
            entry[side] = key

        text_targets: list[str] = []
        vision_targets: list[str] = []
        for stem, pair in sorted(pairs.items()):
            if set(pair) != {"down", "up"}:
                raise ValueError(f"LTX Gemma text LoRA pair is incomplete: {stem}")
            down_shape = tuple(handle.get_slice(pair["down"]).get_shape())
            up_shape = tuple(handle.get_slice(pair["up"]).get_shape())
            if (
                len(down_shape) != 2
                or len(up_shape) != 2
                or up_shape[1] != down_shape[0]
                or down_shape[0] <= 0
            ):
                raise ValueError(f"LTX Gemma text LoRA pair geometry is invalid: {stem}")
            if rank is None:
                rank = down_shape[0]
            elif rank != down_shape[0]:
                raise ValueError("LTX Gemma text LoRA rank is not uniform")
            if stem.startswith("model."):
                module_name = "model.language_model." + stem.removeprefix("model.")
                kind = "embedding" if stem == "model.embed_tokens" else "linear"
                if kind == "embedding" and (down_shape, up_shape) != ((64, 3840), (262208, 64)):
                    raise ValueError("LTX Gemma embed_tokens LoRA geometry changed")
                text_targets.append(module_name)
                text_pairs.append(
                    LTX23GemmaTextLoraTarget(
                        module_name=module_name,
                        down_key=pair["down"],
                        up_key=pair["up"],
                        kind=kind,
                    )
                )
            elif stem.startswith("vision_model."):
                vision_targets.append(stem)
            else:
                raise ValueError(f"LTX Gemma text LoRA target namespace is unsupported: {stem}")

    embedding = "model.language_model.embed_tokens"
    if (
        len(pairs) != (_TEXT_LORA_TARGET_COUNT + _VISION_LORA_TARGET_COUNT)
        or len(text_targets) != _TEXT_LORA_TARGET_COUNT
        or len(vision_targets) != _VISION_LORA_TARGET_COUNT
        or embedding not in text_targets
        or rank != 64
    ):
        raise ValueError("LTX Gemma text LoRA target coverage changed")
    return LTX23GemmaTextLoraPlan(
        probe.identity,
        probe.schema_sha256,
        tuple(text_targets),
        tuple(vision_targets),
        embedding,
        rank,
        tuple(text_pairs),
    )


def revalidate_ltx23_gemma_text_lora(plan: LTX23GemmaTextLoraPlan) -> bool:
    """Re-plan the fixed LoRA before a future Engine-native installer uses it."""

    try:
        refreshed = plan_ltx23_gemma_text_lora(plan.identity.path)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed == plan and revalidate_artifact(plan.identity)


def install_ltx23_gemma_text_lora(
    model: Any,
    plan: LTX23GemmaTextLoraPlan,
    *,
    adapter_name: str,
    strength: float = 1.0,
) -> LTX23GemmaTextLoraApplication:
    """Install all 337 required text pairs without modifying the mixed base.

    Vision pairs are intentionally not read: LTX prompt conditioning is text
    only.  The 336 linear targets must already be Kitchen-backed mixed linears;
    accepting an ordinary dense linear here would hide a materialization error.
    """

    if not _ADAPTER_NAME.fullmatch(adapter_name):
        raise ValueError("LTX Gemma LoRA adapter name is unsafe")
    if not revalidate_ltx23_gemma_text_lora(plan):
        raise ValueError("LTX 2.3 Gemma text LoRA changed after planning")
    if (
        len(plan.text_targets) != _TEXT_LORA_TARGET_COUNT
        or len(plan.pairs) != _TEXT_LORA_TARGET_COUNT
        or len(plan.ignored_vision_targets) != _VISION_LORA_TARGET_COUNT
        or plan.text_targets.count(plan.embedding_target) != 1
    ):
        raise ValueError("LTX Gemma text LoRA plan is incomplete")
    pair_names = tuple(pair.module_name for pair in plan.pairs)
    if pair_names != plan.text_targets:
        raise ValueError("LTX Gemma text LoRA pair order differs from its target contract")
    expected_quantized = set(getattr(model, "_latentslate_ltx23_gemma_quant_modules", {}))
    expected_linear = {pair.module_name for pair in plan.pairs if pair.kind == "linear"}
    if expected_quantized != expected_linear:
        raise RuntimeError("LTX Gemma text LoRA does not exactly cover mixed text linear modules")

    installed: list[str] = []
    original_embedding: nn.Embedding | None = None
    try:
        from safetensors import safe_open

        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            if not revalidate_artifact(plan.identity):
                raise ValueError("LTX 2.3 Gemma text LoRA changed before materialization")
            for pair in plan.pairs:
                down = handle.get_tensor(pair.down_key)
                up = handle.get_tensor(pair.up_key)
                if pair.kind == "embedding":
                    language = model.model.language_model
                    current = language.embed_tokens
                    if isinstance(current, nn.Embedding):
                        original_embedding = current
                        current = LTX23GemmaEmbeddingLora(current)
                        language.embed_tokens = current
                    if not isinstance(current, LTX23GemmaEmbeddingLora):
                        raise TypeError("LTX Gemma embed_tokens target changed after planning")
                    current.add_lora_adapter(adapter_name, down, up)
                elif pair.kind == "linear":
                    module = _lora_module(model, pair.module_name)
                    if not isinstance(module, _STORED_LINEAR_TYPES):
                        raise TypeError(f"LTX Gemma LoRA target is not a mixed stored Linear: {pair.module_name}")
                    module.add_lora_adapter(adapter_name, down, up, alpha=None)
                else:
                    raise ValueError(f"LTX Gemma LoRA target kind is unsupported: {pair.kind}")
                installed.append(pair.module_name)
        application = LTX23GemmaTextLoraApplication(
            model,
            plan,
            adapter_name,
            tuple(installed),
            original_embedding,
        )
        application.set_strength(strength)
        _retie_lm_head(model)
        return application
    except BaseException:
        for module_name in reversed(installed):
            try:
                _lora_module(model, module_name).delete_lora_adapter(adapter_name)
            except (AttributeError, KeyError):
                pass
        if original_embedding is not None:
            model.model.language_model.embed_tokens = original_embedding
        _retie_lm_head(model)
        raise


def _gemma_leaf_schedule(path: str, slots, source_values) -> LTX23LeafSchedule:
    """Classify one direct Gemma leaf without coupling capture to its topology."""

    parts = path.split(".") if path else []
    if parts and parts[0] == "layers":
        if (
            len(parts) < 2
            or not parts[1].isdigit()
            or str(int(parts[1])) != parts[1]
        ):
            raise ValueError("LTX Gemma layer leaf path is not canonical")
        group = f"layers.{int(parts[1])}"
    else:
        group = "root"
    is_patch = any("lora" in part.lower() or "adapter" in part.lower() for part in parts)
    if not source_values:
        return LTX23LeafSchedule(
            f"{group}.patch" if is_patch else group,
            tiny_force_resident=not is_patch,
            force_resident=(
                not is_patch
                and (path == "embed_tokens" or path.startswith("embed_tokens."))
            ),
        )
    file_backed = tuple(id(slot.cpu_value) in source_values for slot in slots)
    if is_patch:
        if any(file_backed):
            raise ValueError("LTX Gemma LoRA leaf unexpectedly owns base-file state")
        return LTX23LeafSchedule(f"{group}.patch", tiny_force_resident=False)
    if all(file_backed):
        return LTX23LeafSchedule(
            group,
            tiny_force_resident=True,
            # Gemma's outer shell consumes the tied embedding before the
            # language-model hook. Keep that one base allocation bound across
            # the node, then release it at ordinary text offload.
            force_resident=path == "embed_tokens" or path.startswith("embed_tokens."),
        )
    # Generated rotary/runtime buffers remain part of the base schedule even
    # though their bounded CPU tensors are not checkpoint-file spans.
    return LTX23LeafSchedule(group, tiny_force_resident=True)


_TEXT_DYNAMIC_REQUEST_COUNTERS = (
    "faults",
    "signature_hits",
    "signature_misses",
    "fault_none_temporaries",
    "pinned_copy_bytes",
    "pageable_copy_bytes",
    "transfer_events",
    "transfer_waits",
    "unpin_calls",
    "dirty_epoch",
    "lora_invalidations",
    "base_restores",
    "gathered_misses",
    "per_physical_misses",
    "packed_source_bytes",
    "gathered_h2d_bytes",
    "pressure_direct_transfers",
    "pressure_direct_bytes",
    "host_buffer_reuse_barriers",
    "host_source_pool_hits",
    "host_source_pool_misses",
    "host_source_pool_stale_rejections",
    "host_source_pool_warm_ram_pressure_bypasses",
    "host_source_pool_warm_zero_delta_extend_refusals",
    "host_source_pool_warm_registration_refusals",
    "host_source_pool_temporary_ram_pressure_bypasses",
    "host_source_pool_temporary_zero_delta_extend_refusals",
    "host_source_pool_temporary_registration_refusals",
    "base_file_read_calls",
    "base_file_read_bytes",
    "prefetch_calls",
)
_TEXT_SCHEDULER_REQUEST_COUNTERS = (
    "prefetch_groups",
    "prefetch_leaves",
    "deferred_waits",
    "force_resident_waits",
    "consumed_groups",
)


class LTX23GemmaMixedTextStage:
    """Engine-owned, per-leaf residency for LTX's mixed Gemma encoder.

    Authenticated file spans remain authoritative for base leaves, while
    immutable CPU LoRA leaves occupy distinct patch scheduling groups. Direct
    state leaves are bound only for their consuming root/layer forward, and every
    quantized Kitchen weight is reconstructed from its stored qdata/sidecars.
    Each text linear then follows Comfy's ``full_precision_mm`` contract: one
    bounded dense dequantized temporary feeds an ordinary linear operation.
    """

    def __init__(
        self,
        model: Any,
        execution_device: torch.device | str,
        *,
        dynamic_policy: str = "auto",
        progress: Callable[[float, str | None], None] | None = None,
    ) -> None:
        if not _is_ltx23_gemma_text_only(model):
            raise ValueError("LTX Gemma stage requires a text-only materialized model")
        self.model = model
        self.execution_device = torch.device(execution_device)
        if dynamic_policy not in {"auto", "required", "hooks"}:
            raise ValueError("LTX Gemma dynamic policy must be auto, required, or hooks")
        self.dynamic_policy = dynamic_policy
        self._progress = progress
        self.language_model = model.model.language_model
        self._source_descriptors: dict[int, Any] = dict(
            getattr(model, "_latentslate_ltx23_gemma_source_descriptors", {})
        )
        self._base_file_requested = bool(self._source_descriptors)
        layers = getattr(self.language_model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise RuntimeError("LTX Gemma text shell lacks its sequential language layers")
        self._layers = tuple(layers)
        if len({id(layer) for layer in self._layers}) != len(self._layers):
            raise RuntimeError("LTX Gemma text shell aliases language layers")
        self._root_storage = capture_ltx23_module_storage(
            self.language_model,
            exclude_children=frozenset({"layers"}),
            source_values=self._source_descriptors,
        )
        self._layer_storage = tuple(
            capture_ltx23_module_storage(
                layer, source_values=self._source_descriptors
            )
            for layer in self._layers
        )
        self._leaf_storage = capture_ltx23_leaf_storages(
            self.language_model,
            source_values=self._source_descriptors,
            schedule_resolver=_gemma_leaf_schedule,
        )
        self._leaf_by_path = {leaf.path: leaf for leaf in self._leaf_storage}
        self._base_leaf_paths = tuple(
            leaf.path
            for leaf in self._leaf_storage
            if all(not group.endswith(".patch") for group in leaf.schedule_groups)
        )
        self._patch_leaf_paths = tuple(
            leaf.path
            for leaf in self._leaf_storage
            if any(group.endswith(".patch") for group in leaf.schedule_groups)
        )
        if len(self._base_leaf_paths) + len(self._patch_leaf_paths) != len(
            self._leaf_storage
        ):
            raise ValueError("LTX Gemma alias leaf crosses base and patch source classes")
        self._schedule_order = tuple(
            group
            for index in range(len(self._layers) + 1)
            for group in (
                "root" if index == 0 else f"layers.{index - 1}",
                "root.patch" if index == 0 else f"layers.{index - 1}.patch",
            )
        )
        self._groups_with_leaves = {
            group for leaf in self._leaf_storage for group in leaf.schedule_groups
        }
        for nested in model.modules():
            if isinstance(nested, _STORED_LINEAR_TYPES):
                nested.set_execution_policy("strict_comfy_full_precision_mm")
        self._root_binding: LTX23ModuleBinding | None = None
        self._layer_binding: LTX23ModuleBinding | None = None
        self._root_dynamic_lease: DynamicResidencyLease | None = None
        self._layer_dynamic_lease: DynamicResidencyLease | None = None
        self._dynamic_backend: Any | None = None
        self._leaf_scheduler: LeafResidencyScheduler | None = None
        self._leaf_bindings: dict[str, LTX23ModuleBinding] = {}
        self._root_groups_active: list[str] = []
        self._layer_groups_active: list[str] = []
        self._patch_active = True
        self._request_index = 0
        self._request_baseline: dict[str, Any] = {}
        self._dynamic_diagnostics: dict[str, Any] | None = None
        self._dynamic_fallback_reason: str | None = None
        self._onload_backend_cleanup_attempted = False
        self._handles: list[Any] = []
        self._active = False
        self._owner_thread: int | None = None
        self._root_transitions = 0
        self._layer_transitions = 0
        self._transfer_streams: tuple[Any, ...] = ()
        self._next_transfer_stream = 0
        self._transfer_events = 0
        self._transfer_waits = 0
        self._async_transfer_fallbacks = 0
        self._host_registrations = _LTX23HostRegistrationLedger(
            _host_registration_budget_bytes()
        )
        self._host_registration_groups: set[int] = set()
        self._layer_compute_barriers = 0
        self._live_layer_bindings = 0
        self._live_layer_bytes = 0
        self._maximum_live_layer_bindings = 0
        self._maximum_live_layer_bytes = 0
        self._base_file_handle: Any | None = None
        self._base_file_handle_opened = 0
        self._base_file_handle_closed = 0
        self._base_file_fallback_reason: str | None = None

    @property
    def required_cuda_bytes(self) -> int:
        """Largest synchronous leaf set, excluding caller activation headroom."""

        forced = {
            leaf.path: leaf.storage.physical_bytes
            for leaf in self._leaf_storage
            if leaf.force_resident
        }
        group_bytes = {
            group: sum(
                leaf.storage.physical_bytes
                for leaf in self._leaf_storage
                if group in leaf.schedule_groups and leaf.path not in forced
            )
            for group in self._schedule_order
        }
        return sum(forced.values()) + max(group_bytes.values(), default=0)

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded residency evidence safe to persist in output metadata."""

        baseline = self._request_baseline
        patch_residency = self._patch_residency_snapshot()
        patch_baseline = baseline.get("patch_residency", {})
        patch_residency = {
            key: value - int(patch_baseline.get(key, 0))
            for key, value in patch_residency.items()
        }
        scheduler_proof = (
            self._leaf_scheduler.diagnostics()
            if self._leaf_scheduler is not None
            else None
        )
        dynamic_proof = (
            self._dynamic_backend.diagnostics()
            if self._dynamic_backend is not None
            and self.terminal_poison_reason() is None
            else self._dynamic_diagnostics
        )
        if scheduler_proof is not None:
            scheduler_proof = dict(scheduler_proof)
            scheduler_baseline = baseline.get("scheduler", {})
            for key in _TEXT_SCHEDULER_REQUEST_COUNTERS:
                scheduler_proof[key] -= int(scheduler_baseline.get(key, 0))
        if dynamic_proof is not None:
            dynamic_proof = {**dynamic_proof, "policy": self.dynamic_policy}
            dynamic_baseline = baseline.get("dynamic", {})
            dynamic_proof["request_delta"] = {
                key: int(dynamic_proof.get(key, 0)) - int(dynamic_baseline.get(key, 0))
                for key in _TEXT_DYNAMIC_REQUEST_COUNTERS
            }
            dynamic_proof["warm_request_index"] = self._request_index
            dynamic_proof.update(
                base_file_handle_live=self._base_file_handle is not None,
                base_file_handle_opened=self._base_file_handle_opened,
                base_file_handle_closed=self._base_file_handle_closed,
                base_file_fallback_reason=self._base_file_fallback_reason,
            )
        result = {
            "mode": (
                "dynamic_vbar_per_leaf"
                if dynamic_proof is not None
                else "layer_streamed_cpu_master"
            ),
            # The dynamic path faults roots at the outer Gemma hook because
            # embedding lookup precedes language_model.forward. The fallback
            # Engine-hook path keeps roots resident for the whole text stage.
            "root_activation": (
                "per_model_forward_fault"
                if dynamic_proof is not None
                else "stage_onload"
            ),
            "layer_count": len(self._layers),
            "root_weight_bytes": self._root_storage.physical_bytes,
            "largest_layer_weight_bytes": max(
                storage.physical_bytes for storage in self._layer_storage
            ),
            "required_weight_bytes": self.required_cuda_bytes,
            "root_transitions": self._root_transitions
            - int(baseline.get("root_transitions", 0)),
            "layer_transitions": self._layer_transitions
            - int(baseline.get("layer_transitions", 0)),
            "execution_policy": "strict_comfy_full_precision_mm",
            "patched_resident": patch_residency,
            "native_quantized_dispatches": sum(
                nested.native_dispatch_count
                for nested in self.model.modules()
                if isinstance(nested, _STORED_LINEAR_TYPES)
            )
            - int(baseline.get("native_quantized_dispatches", 0)),
            "full_precision_dispatches": sum(
                nested.full_precision_dispatch_count
                for nested in self.model.modules()
                if isinstance(nested, _STORED_LINEAR_TYPES)
            )
            - int(baseline.get("full_precision_dispatches", 0)),
            "transfer_mode": (
                "aimdo_two_stream_nonblocking"
                if dynamic_proof is not None
                else "two_stream_nonblocking"
                if self._transfer_streams
                else "blocking_cpu"
            ),
            "transfer_stream_count": (
                dynamic_proof["copy_stream_count"]
                if dynamic_proof is not None
                else len(self._transfer_streams)
            ),
            "transfer_events": (
                dynamic_proof["request_delta"]["transfer_events"]
                if dynamic_proof is not None
                else self._transfer_events - int(baseline.get("transfer_events", 0))
            ),
            "transfer_waits": (
                dynamic_proof["request_delta"]["transfer_waits"]
                if dynamic_proof is not None
                else self._transfer_waits - int(baseline.get("transfer_waits", 0))
            ),
            "async_transfer_fallbacks": self._async_transfer_fallbacks,
            "strict_cuda_parity": self.execution_device.type == "cuda"
            and (
                dynamic_proof is not None
                or len(self._transfer_streams) == 2
            ),
            "host_registration": (
                dynamic_proof["host_registration"]
                if dynamic_proof is not None
                else self._host_registrations.provenance()
            ),
            "layer_compute_barriers": self._layer_compute_barriers
            - int(baseline.get("layer_compute_barriers", 0)),
            "live_layer_bindings": self._live_layer_bindings,
            "live_layer_bytes": self._live_layer_bytes,
            "maximum_live_layer_bindings": self._maximum_live_layer_bindings,
            "maximum_live_layer_bytes": self._maximum_live_layer_bytes,
            "dynamic_vbar_prefetch": False,
            "leaf_allocation_count": (
                len(self._leaf_storage)
                if scheduler_proof is None
                else scheduler_proof["leaf_allocation_count"]
            ),
            "force_resident_leaf_count": sum(
                leaf.force_resident for leaf in self._leaf_storage
            ),
            "base_leaf_count": len(self._base_leaf_paths),
            "patch_leaf_count": len(self._patch_leaf_paths),
            "schedule_group_count": len(self._schedule_order),
            "leaf_scheduler": scheduler_proof,
            "warm_request_index": self._request_index,
        }
        if dynamic_proof is not None:
            result["dynamic_vram"] = dynamic_proof
        else:
            result["dynamic_vram"] = {
                "backend": "engine_hooks",
                "policy": self.dynamic_policy,
                "fallback_reason": self._dynamic_fallback_reason,
                "prefetch": False,
                "allocator_plugin": False,
                "base_file_requested": self._base_file_requested,
                "base_file_backed": False,
                "base_file_read_calls": 0,
                "base_file_read_bytes": 0,
                "base_file_handle_live": self._base_file_handle is not None,
                "base_file_handle_opened": self._base_file_handle_opened,
                "base_file_handle_closed": self._base_file_handle_closed,
                "base_file_fallback_reason": self._base_file_fallback_reason,
            }
        return result

    def _patch_residency_snapshot(self) -> dict[str, int]:
        linears = tuple(
            nested
            for nested in self.model.modules()
            if isinstance(nested, _STORED_LINEAR_TYPES)
        )
        embedding = self.language_model.embed_tokens
        return {
            "linear_merge_misses": sum(
                int(nested.patched_resident_merge_misses) for nested in linears
            ),
            "linear_hits": sum(int(nested.patched_resident_hits) for nested in linears),
            "linear_signature_none_rematerializations": sum(
                int(nested.signature_none_patch_rematerializations)
                for nested in linears
            ),
            "linear_requantize_writebacks": sum(
                int(nested.patched_resident_writebacks) for nested in linears
            ),
            "embedding_merge_misses": int(
                getattr(embedding, "patched_resident_merge_misses", 0)
            ),
            "embedding_hits": int(getattr(embedding, "patched_resident_hits", 0)),
            "embedding_signature_none_rematerializations": int(
                getattr(embedding, "signature_none_patch_rematerializations", 0)
            ),
            "embedding_writebacks": int(
                getattr(embedding, "patched_resident_writebacks", 0)
            ),
        }

    def terminal_poison_reason(self) -> str | None:
        scheduler = self._leaf_scheduler
        if scheduler is not None:
            reason = scheduler.terminal_poison_reason()
            if reason is not None:
                return reason
        backend = self._dynamic_backend
        if backend is None:
            return None
        return backend.terminal_poison_reason()

    def onload(self) -> None:
        if self._active:
            raise RuntimeError("LTX Gemma text residency is already active")
        if self._root_binding is not None or self._layer_binding is not None:
            raise RuntimeError("LTX Gemma text residency retained an active binding")
        self._owner_thread = threading.get_ident()
        self._onload_backend_cleanup_attempted = False
        self._patch_active = True
        try:
            self._initialize_backend()
            self._begin_request_diagnostics()
            self._handles.append(self.model.register_forward_pre_hook(self._model_pre))
            self._handles.append(
                self.model.register_forward_hook(self._model_post, always_call=True)
            )
            self._handles.append(self.language_model.register_forward_pre_hook(self._root_pre))
            self._handles.append(
                self.language_model.register_forward_hook(self._root_post, always_call=True)
            )
            for index, layer in enumerate(self._layers):
                self._handles.append(layer.register_forward_pre_hook(self._layer_pre(index)))
                self._handles.append(layer.register_forward_hook(self._layer_post, always_call=True))
            # ``Gemma3ForConditionalGeneration.forward`` calls
            # ``language_model.embed_tokens(input_ids)`` before delegating to
            # ``language_model.forward``.  In particular, the embedding LoRA
            # wrapper receives CUDA token indices first.  Bind every root
            # (embedding base, embedding LoRA tensors, norms, and direct root
            # state) before that outer forward can begin.
            if self._leaf_scheduler is not None:
                self._leaf_scheduler.onload()
                _retie_lm_head(self.model)
            elif self._dynamic_backend is None:
                self._register_storage_best_effort(self._root_storage)
                binding = self._copy_with_transfer(self._root_storage)
                binding.activate()
                self._root_binding = binding
                self._root_transitions += 1
                _retie_lm_head(self.model)
            self._active = True
        except BaseException as primary:
            self._remove_hooks()
            try:
                self._restore_cpu()
            except BaseException as cleanup_error:  # noqa: BLE001
                primary.add_note(
                    "LTX Gemma onload CPU restoration also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            finally:
                self._owner_thread = None
            if not self._onload_backend_cleanup_attempted:
                self._close_dynamic_backend(primary=primary)
            if self._dynamic_backend is None:
                self._cleanup_host_registrations(
                    primary=primary,
                    synchronized=False,
                )
            raise

    def _begin_request_diagnostics(self) -> None:
        backend = self._dynamic_backend
        scheduler = self._leaf_scheduler
        dynamic = {} if backend is None else backend.diagnostics()
        scheduled = {} if scheduler is None else scheduler.diagnostics()
        self._request_index += 1
        self._request_baseline = {
            "root_transitions": self._root_transitions,
            "layer_transitions": self._layer_transitions,
            "layer_compute_barriers": self._layer_compute_barriers,
            "transfer_events": self._transfer_events,
            "transfer_waits": self._transfer_waits,
            "native_quantized_dispatches": sum(
                nested.native_dispatch_count
                for nested in self.model.modules()
                if isinstance(nested, _STORED_LINEAR_TYPES)
            ),
            "full_precision_dispatches": sum(
                nested.full_precision_dispatch_count
                for nested in self.model.modules()
                if isinstance(nested, _STORED_LINEAR_TYPES)
            ),
            "patch_residency": self._patch_residency_snapshot(),
            "dynamic": {
                key: dynamic.get(key, 0) for key in _TEXT_DYNAMIC_REQUEST_COUNTERS
            },
            "scheduler": {
                key: scheduled.get(key, 0) for key in _TEXT_SCHEDULER_REQUEST_COUNTERS
            },
        }
        self._maximum_live_layer_bindings = 0
        self._maximum_live_layer_bytes = 0

    def offload(self) -> None:
        if not self._active:
            if self._leaf_scheduler is None:
                self._restore_cpu()
            return
        self._require_owner()
        self._remove_hooks()
        # Root weights remain live across all three text nodes. Preserve them
        # until every queued consumer has crossed the execution-stream barrier.
        try:
            if self.execution_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.execution_device)
        except BaseException as primary:
            if self._leaf_scheduler is not None:
                # Quiescence is unproven. Keep every binding, lease, VBAR,
                # source view, and file owner reachable until the persistent
                # GPU child takes its canonical hard OS exit. No restoration,
                # unpin, diagnostics query, or native close is safe here.
                self._leaf_scheduler.mark_terminal_poison(
                    "device_quiescence_failed"
                )
                self._active = False
                self._owner_thread = None
                raise DynamicResidencyPoisoned(
                    "device_quiescence_failed"
                ) from primary
            try:
                self._restore_cpu()
            except BaseException as cleanup_error:  # noqa: BLE001 - retain CUDA failure
                primary.add_note(
                    "LTX Gemma CPU restoration also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            finally:
                self._active = False
                self._owner_thread = None
            if self._leaf_scheduler is None:
                self._close_dynamic_backend(primary=primary)
            if self._dynamic_backend is None:
                self._cleanup_host_registrations(primary=primary, synchronized=False)
            raise
        try:
            if self._leaf_scheduler is not None:
                self._leaf_scheduler.clear_stage(release_force_resident=True)
                if self._leaf_bindings:
                    raise RuntimeError("LTX Gemma leaf bindings survived text offload")
                _retie_lm_head(self.model)
            else:
                self._restore_cpu()
                self._cleanup_host_registrations(synchronized=True)
        finally:
            self._active = False
            self._owner_thread = None
        if self.execution_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Strictly purge the dormant text identity during runtime unload."""

        if self._active:
            self.offload()
        scheduler = self._leaf_scheduler
        backend = self._dynamic_backend
        if scheduler is not None:
            # Assignment is deliberately after both calls: any failure retains
            # scheduler/backend/file ownership for the hard-child exit path.
            scheduler.close()
            if backend is not None:
                self._dynamic_diagnostics = backend.diagnostics()
            self._leaf_scheduler = None
            self._dynamic_backend = None
            self._close_base_file_handle()
        elif backend is not None:
            backend.close()
            self._dynamic_diagnostics = backend.diagnostics()
            self._dynamic_backend = None
            self._close_base_file_handle()
        else:
            self._cleanup_host_registrations(synchronized=True)

    def _root_pre(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._require_active_owner()
        if self._leaf_scheduler is not None:
            if "root" not in self._root_groups_active:
                raise RuntimeError(
                    "LTX Gemma base roots were not bound before the language-model forward"
                )
            return
        if self._root_binding is None or not self._root_binding.active:
            raise RuntimeError(
                "LTX Gemma text roots were not bound before the language-model forward"
            )

    def _model_pre(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._require_active_owner()
        scheduler = self._leaf_scheduler
        if scheduler is not None:
            if self._root_groups_active:
                raise RuntimeError("LTX Gemma per-leaf roots are non-reentrant")
            entered: list[str] = []
            try:
                for group in ("root", "root.patch"):
                    if group not in self._groups_with_leaves or (
                        group.endswith(".patch") and not self._patch_active
                    ):
                        continue
                    scheduler.enter(group)
                    entered.append(group)
                self._root_groups_active = entered
                self._root_transitions += 1
                _retie_lm_head(self.model)
            except BaseException:
                for group in reversed(entered):
                    scheduler.leave(group)
                raise
            return
        if self._dynamic_backend is None:
            return
        if self._root_binding is not None or self._root_dynamic_lease is not None:
            raise RuntimeError("LTX Gemma dynamic roots are non-reentrant")
        binding, lease = self._acquire_dynamic(self._root_storage)
        try:
            binding.activate()
            self._root_binding = binding
            self._root_dynamic_lease = lease
            self._root_transitions += 1
            _retie_lm_head(self.model)
        except BaseException as primary:
            self._release_dynamic(binding, lease, primary=primary)
            raise

    def _model_post(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self._leaf_scheduler is not None:
            self._leave_root_groups()
        elif self._dynamic_backend is not None:
            self._restore_root_cpu()
        return output

    def _root_post(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        try:
            self._require_active_owner()
            if self._layer_binding is not None or self._layer_groups_active:
                raise RuntimeError("LTX Gemma text layer was still resident at root completion")
        finally:
            # An exceptional language forward must not pin a partial CUDA copy.
            if self._leaf_scheduler is not None:
                self._leave_layer_groups()
            else:
                self._restore_layer_cpu()
        return output

    def _layer_pre(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._require_active_owner()
            scheduler = self._leaf_scheduler
            if scheduler is not None:
                if self._layer_groups_active:
                    raise RuntimeError("LTX Gemma per-leaf layer residency is non-reentrant")
                base_group = f"layers.{index}"
                groups = [base_group]
                if self._patch_active and f"{base_group}.patch" in self._groups_with_leaves:
                    groups.append(f"{base_group}.patch")
                entered: list[str] = []
                try:
                    for group in groups:
                        scheduler.enter(group)
                        entered.append(group)
                    self._layer_groups_active = entered
                    self._live_layer_bindings = 1
                    self._live_layer_bytes = sum(
                        leaf.storage.physical_bytes
                        for leaf in self._leaf_storage
                        if any(group in leaf.schedule_groups for group in entered)
                    )
                    self._maximum_live_layer_bindings = max(
                        self._maximum_live_layer_bindings, 1
                    )
                    self._maximum_live_layer_bytes = max(
                        self._maximum_live_layer_bytes, self._live_layer_bytes
                    )
                    self._layer_transitions += 1
                except BaseException:
                    for group in reversed(entered):
                        scheduler.leave(group)
                    raise
                return
            if self._layer_binding is not None:
                raise RuntimeError("LTX Gemma text layer residency is non-reentrant")
            storage = self._layer_storage[index]
            if self._dynamic_backend is None:
                self._register_storage_best_effort(storage)
                binding = self._copy_with_transfer(storage)
                lease = None
            else:
                binding, lease = self._acquire_dynamic(storage)
            try:
                binding.activate()
                self._layer_binding = binding
                self._layer_dynamic_lease = lease
                self._live_layer_bindings += 1
                self._live_layer_bytes += storage.physical_bytes
                self._maximum_live_layer_bindings = max(
                    self._maximum_live_layer_bindings, self._live_layer_bindings
                )
                self._maximum_live_layer_bytes = max(
                    self._maximum_live_layer_bytes, self._live_layer_bytes
                )
                if (
                    self._dynamic_backend is None
                    and self.execution_device.type == "cuda"
                    and torch.cuda.is_available()
                ):
                    binding.record_stream(torch.cuda.current_stream(self.execution_device))
                self._layer_transitions += 1
            except BaseException as primary:
                if self._layer_binding is binding:
                    try:
                        self._restore_layer_cpu()
                    except BaseException as cleanup_error:  # noqa: BLE001
                        primary.add_note(
                            "LTX Gemma layer cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                elif binding.active:
                    if lease is not None:
                        self._release_dynamic(binding, lease, primary=primary)
                    else:
                        try:
                            binding.restore_cpu()
                        except BaseException as cleanup_error:  # noqa: BLE001
                            primary.add_note(
                                "LTX Gemma layer binding cleanup also failed: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                elif lease is not None:
                    self._release_dynamic(binding, lease, primary=primary)
                raise

        return hook

    def _layer_post(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        try:
            self._require_active_owner()
        finally:
            if self._leaf_scheduler is not None:
                self._leave_layer_groups()
            else:
                self._restore_layer_cpu()
        return output

    def _leave_layer_groups(self) -> None:
        scheduler = self._leaf_scheduler
        if scheduler is None:
            return
        groups = tuple(self._layer_groups_active)
        try:
            for group in reversed(groups):
                scheduler.leave(group)
            if groups:
                self._layer_compute_barriers += 1
        finally:
            self._layer_groups_active.clear()
            self._live_layer_bindings = 0
            self._live_layer_bytes = 0

    def _leave_root_groups(self) -> None:
        scheduler = self._leaf_scheduler
        if scheduler is None:
            return
        groups = tuple(self._root_groups_active)
        try:
            for group in reversed(groups):
                scheduler.leave(group)
        finally:
            self._root_groups_active.clear()
            _retie_lm_head(self.model)

    def _restore_layer_cpu(self) -> None:
        if self._layer_binding is not None:
            binding = self._layer_binding
            lease = self._layer_dynamic_lease
            try:
                if lease is not None:
                    # Complete every consumer before rebinding the module to its
                    # authoritative CPU slots. Only then may AIMDO unpin/reuse
                    # the VBAR allocation.
                    self._dynamic_backend.synchronize(lease)
                    self._layer_compute_barriers += 1
                elif self.execution_device.type == "cuda" and torch.cuda.is_available():
                    # The H2D event makes the execution stream wait before using
                    # the binding. This compute event closes the other side of the
                    # lifetime: the host cannot release/reuse that device storage
                    # until all queued layer consumers have completed.
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.execution_device))
                    event.synchronize()
                    self._layer_compute_barriers += 1
            except BaseException as primary:
                # Attempt the broader device barrier before rebinding CPU
                # storage, but retain the original event failure either way.
                try:
                    torch.cuda.synchronize(self.execution_device)
                except BaseException as fallback_error:  # noqa: BLE001
                    primary.add_note(
                        "LTX Gemma fallback CUDA synchronization also failed: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                self._release_layer_binding(binding, primary=primary)
                raise
            self._release_layer_binding(binding, primary=None)
            if lease is not None:
                self._dynamic_backend.release(lease)

    def _release_layer_binding(
        self,
        binding: LTX23ModuleBinding,
        *,
        primary: BaseException | None,
    ) -> None:
        """Restore one binding and close counters without masking a CUDA failure."""

        restore_error: BaseException | None = None
        try:
            binding.restore_cpu()
        except BaseException as exc:  # noqa: BLE001 - try authoritative storage directly
            restore_error = exc
            try:
                binding.storage.restore_cpu()
                binding.active = False
            except BaseException as fallback_error:  # noqa: BLE001
                exc.add_note(
                    "LTX Gemma authoritative CPU storage restore also failed: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                )
        finally:
            self._layer_binding = None
            self._layer_dynamic_lease = None
            self._live_layer_bindings = 0
            self._live_layer_bytes = 0
        if restore_error is not None:
            if primary is not None:
                primary.add_note(
                    "LTX Gemma layer binding restoration also failed: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
            else:
                raise restore_error

    def _restore_cpu(self) -> None:
        primary: BaseException | None = None
        try:
            self._restore_layer_cpu()
        except BaseException as exc:  # noqa: BLE001 - continue restoring roots
            primary = exc
        try:
            self._restore_root_cpu()
            self._root_storage.restore_cpu()
            for storage in self._layer_storage:
                storage.restore_cpu()
            _retie_lm_head(self.model)
        except BaseException as cleanup_error:  # noqa: BLE001
            if primary is None:
                primary = cleanup_error
            else:
                primary.add_note(
                    "LTX Gemma root CPU restoration also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if primary is not None:
            raise primary

    def _restore_root_cpu(self) -> None:
        if self._root_binding is None:
            return
        binding = self._root_binding
        lease = self._root_dynamic_lease
        primary: BaseException | None = None
        if lease is not None:
            try:
                self._dynamic_backend.synchronize(lease)
            except BaseException as exc:  # noqa: BLE001 - continue authoritative restore
                primary = exc
        try:
            binding.restore_cpu()
        except BaseException as exc:  # noqa: BLE001 - preserve release failure
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"LTX Gemma root restoration also failed: {exc}")
        finally:
            self._root_binding = None
            self._root_dynamic_lease = None
            _retie_lm_head(self.model)
        if lease is not None:
            try:
                self._dynamic_backend.release(lease)
            except BaseException as exc:  # noqa: BLE001 - preserve earlier failure
                if primary is None:
                    primary = exc
                else:
                    primary.add_note(f"LTX Gemma root dynamic release also failed: {exc}")
        if primary is not None:
            raise primary

    def invalidate_patch_state(self, *, to_base: bool) -> None:
        """Invalidate AIMDO signatures at the LoRA/base patch-state boundary."""

        if self._leaf_scheduler is not None:
            if self._root_groups_active or self._layer_groups_active:
                raise RuntimeError("LTX Gemma cannot change patch state during a forward")
            if self._patch_leaf_paths:
                self._leaf_scheduler.invalidate(
                    reason="lora_to_base" if to_base else "patch_epoch",
                    # Resident BASE values now hold effective patched bytes.
                    # Recopy both classes while preserving the BASE source
                    # HostBuffer cache; PATCH source lanes are purged by the
                    # backend because this set includes patch groups.
                    paths=(*self._base_leaf_paths, *self._patch_leaf_paths),
                )
            self._patch_active = not to_base
            _retie_lm_head(self.model)
        elif self._dynamic_backend is not None:
            self._dynamic_backend.invalidate(
                reason="lora_to_base" if to_base else "patch_epoch"
            )

    def _activate_leaf(
        self,
        descriptor: LeafResidencyDescriptor,
        values: tuple[Any, ...],
    ) -> LTX23ModuleBinding:
        leaf = self._leaf_by_path[descriptor.path]
        if descriptor.path in self._leaf_bindings:
            raise RuntimeError("LTX Gemma leaf binding is already active")
        binding = LTX23ModuleBinding(leaf.storage, values, self.execution_device)
        binding.activate()
        self._leaf_bindings[descriptor.path] = binding
        if "root" in descriptor.schedule_groups:
            _retie_lm_head(self.model)
        return binding

    def _restore_leaf(
        self,
        descriptor: LeafResidencyDescriptor,
        binding: LTX23ModuleBinding,
    ) -> None:
        if self._leaf_bindings.get(descriptor.path) is not binding:
            raise RuntimeError("LTX Gemma leaf binding ownership changed")
        binding.restore_cpu()
        self._leaf_bindings.pop(descriptor.path)
        if "root" in descriptor.schedule_groups:
            _retie_lm_head(self.model)

    def _initialize_backend(self) -> None:
        if self._dynamic_backend is not None:
            if self._leaf_scheduler is None:
                raise RuntimeError("LTX Gemma retained a backend without its leaf scheduler")
            return
        if self.dynamic_policy == "hooks" or self.execution_device.type != "cuda":
            if self._source_descriptors:
                self._fallback_source_backed_cpu(
                    "aimdo_policy_or_device_unavailable: source-backed AIMDO disabled"
                )
            self._initialize_transfer_streams()
            return
        from .framework.residency.aimdo import AimdoDynamicResidency

        group_values = tuple(
            self._storage_values(leaf.storage) for leaf in self._leaf_storage
        )
        group_bytes = tuple(
            AimdoDynamicResidency.group_bytes(values) for values in group_values
        )
        virtual_bytes = sum(size + 512 for size in group_bytes)
        self._emit_aimdo_breadcrumb(
            "constructor_begin",
            requested_device=str(self.execution_device),
            group_count=len(group_values),
            staged_bytes=sum(group_bytes),
            virtual_bytes=virtual_bytes,
        )
        try:
            backend = AimdoDynamicResidency(
                self.execution_device,
                virtual_bytes=virtual_bytes,
                diagnostic=self._aimdo_backend_diagnostic,
            )
        except DynamicResidencyUnavailable as exc:
            if self.dynamic_policy == "required":
                raise
            self._dynamic_fallback_reason = str(exc)[:512]
            if self._source_descriptors:
                self._fallback_source_backed_cpu(
                    f"aimdo_backend_unavailable: {type(exc).__name__}: {exc}"
                )
            self._initialize_transfer_streams()
            return
        # Retain ownership before the first native allocation. Any subsequent
        # Python exception must close this exact backend once; otherwise its
        # ModelVBAR finalizer would become the accidental owner.
        self._dynamic_backend = backend
        try:
            # AIMDO resolves an unindexed production ``cuda`` request exactly
            # once. Every binding and stream operation uses this identity.
            self.execution_device = backend.device
            self._emit_aimdo_breadcrumb(
                "constructor_after",
                device=str(self.execution_device),
                current_device=self._current_cuda_device_diagnostic(),
                group_count=len(group_values),
                virtual_bytes=virtual_bytes,
            )
            descriptors = tuple(
                LeafResidencyDescriptor(
                    leaf.path,
                    leaf.schedule_groups,
                    values,
                    size,
                    leaf.force_resident,
                )
                for leaf, values, size in zip(
                    self._leaf_storage, group_values, group_bytes, strict=True
                )
            )
            self._emit_aimdo_breadcrumb(
                "leaf_allocation_begin",
                group_count=len(descriptors),
                staged_bytes=sum(group_bytes),
                virtual_bytes=virtual_bytes,
            )
            scheduler = LeafResidencyScheduler(
                backend,
                descriptors,
                schedule_order=self._schedule_order,
                activate=self._activate_leaf,
                restore=self._restore_leaf,
            )
            self._leaf_scheduler = scheduler
            self._emit_aimdo_breadcrumb(
                "leaf_allocation_after",
                group_count=len(descriptors),
                staged_bytes=sum(group_bytes),
                virtual_bytes=virtual_bytes,
            )
            self._emit_aimdo_breadcrumb(
                "prioritize_begin",
                group_count=len(descriptors),
                cumulative_bytes=sum(group_bytes),
                virtual_bytes=virtual_bytes,
            )
            backend.prioritize()
            self._emit_aimdo_breadcrumb(
                "prioritize_after",
                group_count=len(descriptors),
                cumulative_bytes=sum(group_bytes),
                virtual_bytes=virtual_bytes,
            )
            if self._source_descriptors:
                proof = backend.diagnostics()
                if proof["copy_strategy"] != "gathered_host_buffer":
                    fallback_reason = proof["copy_fallback_reason"]
                    if not isinstance(fallback_reason, str):
                        raise RuntimeError(
                            "AIMDO source-backed fallback lacks a capability reason"
                        )
                    if self.dynamic_policy == "required":
                        raise DynamicResidencyUnavailable(
                            "required AIMDO source-backed HostBuffer is unavailable: "
                            f"{fallback_reason}"
                        )
                    backend.close()
                    self._dynamic_backend = None
                    self._fallback_source_backed_cpu(fallback_reason)
                    self._initialize_transfer_streams()
                    return
                self._open_and_bind_base_file(backend)
        except BaseException as primary:
            self._onload_backend_cleanup_attempted = True
            self._close_dynamic_backend(primary=primary)
            raise

    def _aimdo_backend_diagnostic(
        self, phase: str, details: Mapping[str, object]
    ) -> None:
        self._emit_aimdo_breadcrumb(phase, **details)

    def _emit_aimdo_breadcrumb(self, phase: str, **details: object) -> None:
        progress = self._progress
        if progress is None:
            return
        safe_details = ", ".join(
            f"{key}={details[key]}" for key in sorted(details)
        )
        message = f"LTX AIMDO {phase}"
        if safe_details:
            message = f"{message} ({safe_details})"
        progress(0.0785, message[:768])

    @staticmethod
    def _current_cuda_device_diagnostic() -> int | str:
        try:
            return int(torch.cuda.current_device())
        except (RuntimeError, TypeError, ValueError):
            return "unavailable"

    def _storage_values(self, storage) -> tuple[Any, ...]:
        return tuple(
            self._source_descriptors.get(id(slot.cpu_value), slot.cpu_value)
            for slot in storage.slots
        )

    def _acquire_dynamic(
        self, storage
    ) -> tuple[LTX23ModuleBinding, DynamicResidencyLease]:
        lease = self._dynamic_backend.acquire(id(storage))
        return LTX23ModuleBinding(storage, lease.values, self.execution_device), lease

    def _release_dynamic(
        self,
        binding: LTX23ModuleBinding,
        lease: DynamicResidencyLease,
        *,
        primary: BaseException | None,
    ) -> None:
        release_error: BaseException | None = None
        try:
            self._dynamic_backend.synchronize(lease)
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary
            if primary is None:
                release_error = cleanup_error
            else:
                primary.add_note(f"LTX Gemma dynamic synchronize also failed: {cleanup_error}")
        try:
            binding.restore_cpu()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary
            if primary is None and release_error is None:
                release_error = cleanup_error
            elif primary is not None:
                primary.add_note(f"LTX Gemma dynamic restore also failed: {cleanup_error}")
            else:
                release_error.add_note(
                    f"LTX Gemma dynamic restore also failed: {cleanup_error}"
                )
        try:
            self._dynamic_backend.release(lease)
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary
            if primary is None and release_error is None:
                release_error = cleanup_error
            elif primary is not None:
                primary.add_note(f"LTX Gemma dynamic release also failed: {cleanup_error}")
            else:
                release_error.add_note(
                    f"LTX Gemma dynamic release also failed: {cleanup_error}"
                )
        if release_error is not None:
            raise release_error

    def _close_dynamic_backend(self, *, primary: BaseException) -> None:
        backend = self._dynamic_backend
        if backend is None:
            return
        try:
            if self._leaf_scheduler is not None:
                self._leaf_scheduler.close()
            else:
                backend.close()
            self._dynamic_diagnostics = backend.diagnostics()
            self._leaf_scheduler = None
            self._dynamic_backend = None
            self._close_base_file_handle()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary failure
            primary.add_note(f"LTX Gemma AIMDO cleanup also failed: {cleanup_error}")

    def _open_and_bind_base_file(self, backend: Any) -> None:
        plan = getattr(self.model, "_latentslate_ltx23_gemma_plan", None)
        if not isinstance(plan, LTX23GemmaMixedTextPlan) or not revalidate_ltx23_gemma_mixed_text_encoder(plan):
            raise ValueError("LTX Gemma base file changed before source-backed open")
        handle = plan.identity.path.open("rb")
        self._base_file_handle = handle
        self._base_file_handle_opened += 1
        if not revalidate_artifact(plan.identity):
            raise ValueError("LTX Gemma base file changed while opening source-backed stage")
        backend.bind_file_source("ltx23_gemma_base", handle)

    def _close_base_file_handle(self) -> None:
        handle = self._base_file_handle
        if handle is None:
            return
        handle.close()
        self._base_file_handle = None
        self._base_file_handle_closed += 1

    def _materialize_source_backed_cpu_fallback(self) -> None:
        plan = getattr(self.model, "_latentslate_ltx23_gemma_plan", None)
        if not isinstance(plan, LTX23GemmaMixedTextPlan) or not revalidate_ltx23_gemma_mixed_text_encoder(plan):
            raise ValueError("LTX Gemma base file changed before CPU fallback")
        from safetensors import safe_open

        replacements: dict[int, torch.Tensor] = {}
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            if not revalidate_artifact(plan.identity):
                raise ValueError("LTX Gemma base file changed during CPU fallback")
            for storage in (self._root_storage, *self._layer_storage):
                for slot in storage.slots:
                    descriptor = self._source_descriptors.get(id(slot.cpu_value))
                    if descriptor is None:
                        continue
                    key = id(descriptor)
                    value = replacements.get(key)
                    if value is None:
                        value = _materialize_file_backed_value(descriptor, handle)
                        replacements[key] = value
                    _assign_storage_slot(slot, value)
        self._source_descriptors.clear()
        self.model._latentslate_ltx23_gemma_source_descriptors = {}
        self.model._latentslate_ltx23_gemma_source_backed = False
        self._root_storage = capture_ltx23_module_storage(
            self.language_model, exclude_children=frozenset({"layers"})
        )
        self._layer_storage = tuple(
            capture_ltx23_module_storage(layer) for layer in self._layers
        )
        self._leaf_storage = capture_ltx23_leaf_storages(
            self.language_model,
            schedule_resolver=_gemma_leaf_schedule,
        )
        self._leaf_by_path = {leaf.path: leaf for leaf in self._leaf_storage}
        _retie_lm_head(self.model)

    def _fallback_source_backed_cpu(self, reason: str) -> None:
        if self._base_file_handle is not None:
            raise RuntimeError("LTX Gemma cannot fall back after base-file activation")
        self._base_file_fallback_reason = reason[:512]
        if self._dynamic_fallback_reason is None:
            self._dynamic_fallback_reason = self._base_file_fallback_reason
        self._materialize_source_backed_cpu_fallback()

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _initialize_transfer_streams(self) -> None:
        self._transfer_streams = ()
        if self.execution_device.type != "cuda":
            return
        if not torch.cuda.is_available():
            self._async_transfer_fallbacks += 1
            raise RuntimeError("strict Comfy text residency requires available CUDA")
        try:
            self._transfer_streams = (
                torch.cuda.Stream(device=self.execution_device),
                torch.cuda.Stream(device=self.execution_device),
            )
        except (RuntimeError, TypeError) as exc:
            self._async_transfer_fallbacks += 1
            raise RuntimeError(
                "strict Comfy text residency requires exactly two CUDA transfer streams"
            ) from exc

    def _register_storage_best_effort(self, storage) -> None:
        if self.execution_device.type != "cuda" or not torch.cuda.is_available():
            return
        if id(storage) in self._host_registration_groups:
            return
        self._host_registration_groups.add(id(storage))
        for slot in storage.slots:
            if slot.parameter:
                self._host_registrations.consider(slot.cpu_value)

    def _cleanup_host_registrations(
        self,
        *,
        primary: BaseException | None = None,
        synchronized: bool,
    ) -> None:
        if not self._host_registrations.owned:
            return
        if not synchronized:
            try:
                torch.cuda.synchronize(self.execution_device)
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(
                        "LTX Gemma host-unregister synchronization failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    return
                raise
        errors = self._host_registrations.unregister_owned()
        if errors:
            cleanup_error = errors[0]
            if primary is not None:
                primary.add_note(
                    "LTX Gemma CUDA host unregistration failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                return
            raise RuntimeError("LTX Gemma CUDA host unregistration failed") from cleanup_error

    def _copy_with_transfer(self, storage) -> LTX23ModuleBinding:
        if not self._transfer_streams:
            if self.execution_device.type == "cuda":
                raise RuntimeError("strict Comfy CUDA transfer streams are unavailable")
            return storage.copy_to(self.execution_device)
        if len(self._transfer_streams) != 2:
            raise RuntimeError("strict Comfy text residency requires two transfer streams")
        stream = self._transfer_streams[self._next_transfer_stream]
        self._next_transfer_stream = (self._next_transfer_stream + 1) % len(
            self._transfer_streams
        )
        with torch.cuda.stream(stream):
            binding = storage.copy_to(self.execution_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
            self._transfer_events += 1
        torch.cuda.current_stream(self.execution_device).wait_event(event)
        self._transfer_waits += 1
        return binding

    def _require_owner(self) -> None:
        if self._owner_thread != threading.get_ident():
            raise RuntimeError("LTX Gemma text residency moved across threads")

    def _require_active_owner(self) -> None:
        self._require_owner()
        if not self._active:
            raise RuntimeError("LTX Gemma text forward escaped its residency stage")


def _is_ltx23_gemma_text_only(model: Any) -> bool:
    """Recognize the deliberately partial Gemma shell without moving its meta state.

    The normal materializer records an ownership marker.  Cleanup must also be
    able to recognize the exact safe shape if an external pipeline wrapper has
    not preserved that Python-only attribute: a live language-model subset and
    meta state exclusively outside it.  This is deliberately narrower than a
    generic ``hasattr(model, "model")`` check so an unexpectedly complete or
    malformed Gemma cannot silently skip release of a CUDA-resident subtree.
    """

    try:
        language_model = model.model.language_model
        lm_head = model.lm_head
    except AttributeError:
        return False
    if not isinstance(language_model, nn.Module) or not isinstance(lm_head, nn.Module):
        return False
    state = model.state_dict()
    language_state = {
        name: value for name, value in state.items() if name.startswith("model.language_model.")
    }
    source_backed = bool(
        getattr(model, "_latentslate_ltx23_gemma_source_backed", False)
    )
    if not language_state or (
        not source_backed and any(value.is_meta for value in language_state.values())
    ):
        return False
    if any(buffer.is_meta for _, buffer in language_model.named_buffers()):
        return False
    if getattr(model, "_latentslate_ltx23_gemma_text_only", False):
        return True
    return any(
        value.is_meta and not name.startswith("model.language_model.")
        for name, value in state.items()
    )


def _materialize_file_backed_value(descriptor: Any, handle: Any) -> torch.Tensor:
    template = descriptor.template
    flatten = getattr(template, "__tensor_flatten__", None)
    if callable(flatten):
        names, context = flatten()
        if len(names) != len(descriptor.spans):
            raise RuntimeError("LTX Gemma fallback descriptor field count changed")
        fields = {
            name: handle.get_tensor(span.key)
            for name, span in zip(names, descriptor.spans, strict=True)
        }
        unflatten = getattr(type(template), "__tensor_unflatten__", None)
        if not callable(unflatten):
            raise TypeError("LTX Gemma fallback template cannot be rebuilt")
        value = unflatten(fields, context, 0, 0)
    else:
        if len(descriptor.spans) != 1:
            raise RuntimeError("LTX Gemma dense fallback descriptor is not singular")
        value = handle.get_tensor(descriptor.spans[0].key)
    if isinstance(template, nn.Parameter):
        if callable(flatten):
            value._is_param = True
            value.requires_grad_(template.requires_grad)
        else:
            value = nn.Parameter(value, requires_grad=template.requires_grad)
    return value


def _validate_quantized_weight(handle: Any, keys: set[str], stem: str, dtype: str) -> tuple[str, set[str]]:
    config_key = stem + ".comfy_quant"
    scale_key = stem + ".weight_scale"
    if dtype not in {"F8_E4M3", "U8"} or config_key not in keys or scale_key not in keys:
        raise ValueError(f"LTX Gemma quantized sidecars are incomplete: {stem}")
    raw_config = handle.get_tensor(config_key)
    if raw_config.dtype is not torch.uint8 or raw_config.ndim != 1:
        raise ValueError(f"LTX Gemma quant descriptor is invalid: {stem}")
    try:
        config = json.loads(raw_config.numpy().tobytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"LTX Gemma quant descriptor is malformed: {stem}") from exc
    expected_format = "float8_e4m3fn" if dtype == "F8_E4M3" else "nvfp4"
    if config != {"format": expected_format}:
        raise ValueError(f"LTX Gemma quant format is invalid: {stem}")
    weight_shape = tuple(handle.get_slice(stem + ".weight").get_shape())
    scale = handle.get_slice(scale_key)
    if expected_format == "float8_e4m3fn":
        if scale.get_dtype() != "F32" or scale.get_shape() != []:
            raise ValueError(f"LTX Gemma FP8 scale is invalid: {stem}")
        return expected_format, {config_key, scale_key}
    scale2_key = stem + ".weight_scale_2"
    if (
        len(weight_shape) != 2
        or weight_shape[1] % 8
        or scale.get_dtype() != "F8_E4M3"
        or tuple(scale.get_shape()) != (weight_shape[0], weight_shape[1] // 8)
        or scale2_key not in keys
        or handle.get_slice(scale2_key).get_dtype() != "F32"
        or handle.get_slice(scale2_key).get_shape() != []
    ):
        raise ValueError(f"LTX Gemma NVFP4 scales are invalid: {stem}")
    return expected_format, {config_key, scale_key, scale2_key}


def _validate_ignored_auxiliaries(ignored: list[str]) -> None:
    allowed = ("vision_model.", "multi_modal_projector.", "spiece_model")
    if any(not key.startswith(allowed) for key in ignored):
        raise ValueError("LTX Gemma non-text state contains an unclassified source")
    if sum(key.startswith("vision_model.") for key in ignored) != 437:
        raise ValueError("LTX Gemma vision state coverage changed")
    if {key for key in ignored if not key.startswith("vision_model.")} != {
        "multi_modal_projector.mm_input_projection_weight",
        "multi_modal_projector.mm_soft_emb_norm.weight",
        "spiece_model",
    }:
        raise ValueError("LTX Gemma projector/spiece state coverage changed")


def _lora_role(key: str) -> tuple[str, str] | None:
    for suffix, side in ((".lora_down.weight", "down"), (".lora_up.weight", "up")):
        if key.startswith(_TEXT_LORA_PREFIX) and key.endswith(suffix):
            return key[len(_TEXT_LORA_PREFIX) : -len(suffix)], side
    return None


def _lora_module(model: Any, module_name: str) -> Any:
    try:
        return model.get_submodule(module_name)
    except AttributeError as exc:
        raise TypeError(f"LTX Gemma LoRA target is missing: {module_name}") from exc


def _retie_lm_head(model: Any) -> None:
    """Keep the externally held generation head tied across wrapper/residency moves."""

    try:
        embedding = model.model.language_model.embed_tokens
        weight = embedding.weight
        model.lm_head.weight = weight
    except AttributeError as exc:
        raise TypeError("LTX Gemma model is missing its tied language head") from exc
    if model.lm_head.weight is not weight:
        raise RuntimeError("LTX Gemma language head no longer aliases embed_tokens")
