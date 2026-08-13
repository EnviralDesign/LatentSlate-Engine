"""Engine-owned Kitchen materialization for the LTX 2.3 mixed Gemma text path.

The optimized LTX artifact is one multimodal SafeTensors file.  LTX prompt
conditioning uses only its 626 ``model.*`` language-model weights; the vision
tower, projector, and SentencePiece payload are deliberately classified but
never loaded by this component.  Quantized language linears retain their
stored FP8/NVFP4 bytes and use Kitchen's native CUDA dispatch wrappers.  There
is no dequantized dense fallback for any quantized module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from .klein_stored_adapter import (
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    _restore_global_fp8_tensor,
    _restore_nvfp4_tensor,
    move_klein_module_storage,
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
_STORED_LINEAR_TYPES = (KleinStoredLinear, KleinStoredNVFP4Linear)


@dataclass(frozen=True, slots=True)
class LTX23GemmaMixedTextPlan:
    """Exact text-only subset of the LTX 2.3 mixed Gemma artifact."""

    identity: ArtifactIdentity
    schema_sha256: str
    quantized_formats: Mapping[str, str]
    dense_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    ignored_auxiliary_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LTX23GemmaTextLoraPlan:
    """Structural contract for LTX's fixed Gemma prompt LoRA.

    The official adapter includes an embedding target.  Engine has no
    embedding additive-adapter implementation yet, so this plan is deliberately
    non-executable.  Keeping that state explicit prevents a future caller from
    applying only the linear targets and silently changing prompt behavior.
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

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base(input_ids)
        # Gemma3TextScaledWordEmbedding applies this scale to the base lookup.
        # The additive delta is part of that same embedding output contract.
        embed_scale = getattr(self.base, "embed_scale", 1.0)
        active = False
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            output = output + torch.nn.functional.embedding(
                input_ids, adapter.up.to(dtype=output.dtype)
            ).matmul(adapter.down.to(dtype=output.dtype)) * adapter.strength * embed_scale
            active = True
        if active:
            self.lora_dispatch_count += 1
        return output

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

    return LTX23GemmaMixedTextPlan(
        probe.identity,
        probe.schema_sha256,
        MappingProxyType(dict(sorted(formats.items()))),
        tuple(sorted(dense)),
        tuple(sorted(auxiliary)),
        tuple(ignored),
    )


def revalidate_ltx23_gemma_mixed_text_encoder(plan: LTX23GemmaMixedTextPlan) -> bool:
    """Re-plan the header and identity before materializing any payload bytes."""

    try:
        refreshed = plan_ltx23_gemma_mixed_text_encoder(plan.identity.path)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed == plan and revalidate_artifact(plan.identity)


def load_ltx23_gemma_mixed_text_encoder(
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

    if not revalidate_ltx23_gemma_mixed_text_encoder(plan):
        raise ValueError("LTX 2.3 Gemma text artifact changed after planning")
    config = Gemma3Config.from_pretrained(Path(support_root), local_files_only=True)
    with init_empty_weights():
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
                weight = _restore_global_fp8_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    torch.bfloat16,
                )
                replacement: nn.Module = KleinStoredLinear(weight, input_scale=None)
            else:
                logical_shape = (qdata.shape[0], qdata.shape[1] * 2)
                if logical_shape != expected_shape:
                    raise RuntimeError(f"LTX Gemma NVFP4 shape mismatch: {stem}")
                weight = _restore_nvfp4_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    handle.get_tensor(stem + ".weight_scale_2"),
                    logical_shape,
                    torch.bfloat16,
                )
                replacement = KleinStoredNVFP4Linear(weight, input_scale=None)
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


class LTX23GemmaMixedTextStage:
    """Explicit whole-component residency for the language-model subset."""

    def __init__(self, model: Any, execution_device: torch.device | str) -> None:
        if not _is_ltx23_gemma_text_only(model):
            raise ValueError("LTX Gemma stage requires a text-only materialized model")
        self.model = model
        self.execution_device = torch.device(execution_device)

    def onload(self) -> None:
        move_klein_module_storage(self.model.model.language_model, self.execution_device)
        _retie_lm_head(self.model)

    def offload(self) -> None:
        move_klein_module_storage(self.model.model.language_model, "cpu")
        _retie_lm_head(self.model)


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
    if not language_state or any(value.is_meta for value in language_state.values()):
        return False
    if any(buffer.is_meta for _, buffer in language_model.named_buffers()):
        return False
    if getattr(model, "_latentslate_ltx23_gemma_text_only", False):
        return True
    return any(
        value.is_meta and not name.startswith("model.language_model.")
        for name, value in state.items()
    )


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
