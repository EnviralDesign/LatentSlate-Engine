from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.runtime import ltx23_kitchen_text as gemma
from latentslate_engine.runtime.framework.stored_quant import (
    StoredDenseLoraLinear,
    StoredFP8Linear,
    StoredNVFP4Linear,
)
from latentslate_engine.runtime.framework.stored_quant import execution as stored_execution
from latentslate_engine.stored_quant import restore_global_fp8_tensor, restore_nvfp4_tensor


class _Slice:
    def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
        self._dtype = dtype
        self._shape = shape

    def get_dtype(self) -> str:
        return self._dtype

    def get_shape(self) -> list[int]:
        return list(self._shape)


class _Handle:
    def __init__(
        self,
        entries: dict[str, tuple[str, tuple[int, ...]]],
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self.entries = entries
        self.tensors = tensors
        self.requested: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self) -> list[str]:
        return list(self.entries)

    def get_slice(self, key: str) -> _Slice:
        dtype, shape = self.entries[key]
        return _Slice(dtype, shape)

    def get_tensor(self, key: str) -> torch.Tensor:
        self.requested.append(key)
        return self.tensors[key]


def _descriptor(format_name: str) -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": format_name}).encode()), dtype=torch.uint8)


def _full_gemma_handle() -> _Handle:
    entries: dict[str, tuple[str, tuple[int, ...]]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for index in range(302):
        stem = f"model.nvfp4_{index}"
        entries[stem + ".weight"] = ("U8", (16, 8))
        entries[stem + ".weight_scale"] = ("F8_E4M3", (16, 1))
        entries[stem + ".weight_scale_2"] = ("F32", ())
        entries[stem + ".comfy_quant"] = ("U8", (19,))
        tensors[stem + ".comfy_quant"] = _descriptor("nvfp4")
    for index in range(34):
        stem = f"model.fp8_{index}"
        entries[stem + ".weight"] = ("F8_E4M3", (16, 16))
        entries[stem + ".weight_scale"] = ("F32", ())
        entries[stem + ".comfy_quant"] = ("U8", (27,))
        tensors[stem + ".comfy_quant"] = _descriptor("float8_e4m3fn")
    for index in range(290):
        entries[f"model.dense_{index}.weight"] = ("BF16", (16, 16))
    for index in range(437):
        entries[f"vision_model.fixture_{index}"] = ("BF16", (1,))
    entries["multi_modal_projector.mm_input_projection_weight"] = ("BF16", (1,))
    entries["multi_modal_projector.mm_soft_emb_norm.weight"] = ("BF16", (1,))
    entries["spiece_model"] = ("U8", (1,))
    assert len(entries) == 2040
    return _Handle(entries, tensors)


def _header_fixture(handle: _Handle) -> tuple[bytes, dict[str, object]]:
    widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    cursor = 0
    header: dict[str, object] = {}
    for key, (dtype, shape) in handle.entries.items():
        size = int(torch.Size(shape).numel()) * widths[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    return b"fixture-header", header


def _mixed_probe(path: Path) -> SimpleNamespace:
    identity = ArtifactIdentity(path.resolve(), gemma._GEMMA_SIZE_BYTES, 1, "header")
    return SimpleNamespace(
        format="safetensors",
        identity=identity,
        schema_sha256=gemma.LTX23_GEMMA_MIXED_SCHEMA_SHA256,
        tensor_count=2040,
        tensor_dtypes=("BF16", "F32", "F8_E4M3", "U8"),
    )


def test_plan_ltx23_gemma_mixed_text_encoder_proves_full_text_and_ignored_closures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gemma.safetensors"
    path.write_bytes(b"fixture")
    handle = _full_gemma_handle()
    monkeypatch.setattr(gemma, "probe_artifact", _mixed_probe)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: handle)
    monkeypatch.setattr(
        gemma,
        "read_safetensors_header_bytes",
        lambda *_args: _header_fixture(handle),
    )

    plan = gemma.plan_ltx23_gemma_mixed_text_encoder(path)

    assert len(plan.dense_sources) == 290
    assert len(plan.quantized_formats) == 336
    assert sum(value == "nvfp4" for value in plan.quantized_formats.values()) == 302
    assert sum(value == "float8_e4m3fn" for value in plan.quantized_formats.values()) == 34
    assert len(plan.auxiliary_sources) == 974
    assert len(plan.ignored_auxiliary_sources) == 440
    assert len(plan.base_spans) == 290 + 34 * 2 + 302 * 3
    assert all(span.offset >= 8 + len(b"fixture-header") for span in plan.base_spans.values())
    assert {
        key
        for stem, quant_format in plan.quantized_formats.items()
        for key in (
            (stem + ".weight", stem + ".weight_scale")
            if quant_format == "float8_e4m3fn"
            else (
                stem + ".weight",
                stem + ".weight_scale",
                stem + ".weight_scale_2",
            )
        )
    } | set(plan.dense_sources) == set(plan.base_spans)
    assert gemma.ltx23_gemma_source_to_transformers("model.layers.0.mlp.down_proj.weight") == (
        "model.language_model.layers.0.mlp.down_proj.weight"
    )


class _TinyGemma(nn.Module):
    def __init__(self, _config=None) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        language = self.model.language_model
        language.embed_tokens = nn.Embedding(16, 16)
        language.fp8 = nn.Linear(16, 16, bias=False)
        language.nvfp4 = nn.Linear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 16, bias=False)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def test_materializer_maps_model_namespace_to_gemma_language_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gemma.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, "h")
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 16), dtype=torch.bfloat16),
        "model.fp8.weight": torch.zeros((16, 16), dtype=torch.float8_e4m3fn),
        "model.fp8.weight_scale": torch.tensor(0.25, dtype=torch.float32),
        "model.nvfp4.weight": torch.zeros((16, 8), dtype=torch.uint8),
        "model.nvfp4.weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        "model.nvfp4.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
    }
    entries = {
        key: (
            "BF16"
            if value.dtype is torch.bfloat16
            else "F8_E4M3"
            if value.dtype is torch.float8_e4m3fn
            else "U8"
            if value.dtype is torch.uint8
            else "F32",
            tuple(value.shape),
        )
        for key, value in tensors.items()
    }
    plan = gemma.LTX23GemmaMixedTextPlan(
        identity,
        gemma.LTX23_GEMMA_MIXED_SCHEMA_SHA256,
        MappingProxyType({"model.fp8": "float8_e4m3fn", "model.nvfp4": "nvfp4"}),
        ("model.embed_tokens.weight",),
        ("model.fp8.weight_scale", "model.nvfp4.weight_scale", "model.nvfp4.weight_scale_2"),
        (),
    )
    monkeypatch.setattr(gemma, "revalidate_ltx23_gemma_mixed_text_encoder", lambda _plan: True)
    monkeypatch.setattr(gemma, "revalidate_artifact", lambda _identity: True)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: _Handle(entries, tensors))
    monkeypatch.setattr("accelerate.init_empty_weights", lambda: nullcontext())
    fake_config = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Gemma3Config=fake_config, Gemma3ForConditionalGeneration=_TinyGemma),
    )

    model = gemma.load_ltx23_gemma_mixed_text_encoder(
        plan, tmp_path, source_backed=False
    )

    assert isinstance(model.model.language_model.fp8, StoredFP8Linear)
    assert isinstance(model.model.language_model.nvfp4, StoredNVFP4Linear)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert dict(model._latentslate_ltx23_gemma_quant_modules) == {
        "model.language_model.fp8": "float8_e4m3fn",
        "model.language_model.nvfp4": "nvfp4",
    }


def test_file_backed_shell_uses_meta_kitchen_templates_without_payload_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo

    path = tmp_path / "gemma.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(
        path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, "h"
    )
    source_shapes = {
        "model.embed_tokens.weight": ("BF16", (16, 16)),
        "model.fp8.weight": ("F8_E4M3", (16, 16)),
        "model.fp8.weight_scale": ("F32", ()),
        "model.nvfp4.weight": ("U8", (16, 8)),
        "model.nvfp4.weight_scale": ("F8_E4M3", (16, 1)),
        "model.nvfp4.weight_scale_2": ("F32", ()),
    }
    widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    cursor = 64
    spans: dict[str, gemma.LTX23SafetensorSpan] = {}
    for key, (dtype, shape) in source_shapes.items():
        size = int(torch.Size(shape).numel()) * widths[dtype]
        spans[key] = gemma.LTX23SafetensorSpan(key, dtype, shape, cursor, size)
        cursor += size
    plan = gemma.LTX23GemmaMixedTextPlan(
        identity,
        gemma.LTX23_GEMMA_MIXED_SCHEMA_SHA256,
        MappingProxyType({"model.fp8": "float8_e4m3fn", "model.nvfp4": "nvfp4"}),
        ("model.embed_tokens.weight",),
        (
            "model.fp8.weight_scale",
            "model.nvfp4.weight_scale",
            "model.nvfp4.weight_scale_2",
        ),
        (),
        MappingProxyType(spans),
        56,
    )
    monkeypatch.setattr(
        gemma, "revalidate_ltx23_gemma_mixed_text_encoder", lambda _plan: True
    )
    fake_config = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Gemma3Config=fake_config, Gemma3ForConditionalGeneration=_TinyGemma),
    )
    monkeypatch.setattr(
        "safetensors.safe_open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("file-backed shell must not open the payload mapping")
        ),
    )

    model = gemma.load_ltx23_gemma_mixed_text_encoder(plan, tmp_path)

    assert model._latentslate_ltx23_gemma_source_backed is True
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert isinstance(model.model.language_model.fp8, StoredFP8Linear)
    assert isinstance(model.model.language_model.nvfp4, StoredNVFP4Linear)
    descriptors = model._latentslate_ltx23_gemma_source_descriptors
    assert len(descriptors) == 3
    dense = descriptors[id(model.model.language_model.embed_tokens.weight)]
    fp8 = descriptors[id(model.model.language_model.fp8.weight)]
    nvfp4 = descriptors[id(model.model.language_model.nvfp4.weight)]
    assert isinstance(dense, aimdo.AimdoFileBackedValue)
    assert dense.template.is_meta and len(dense.spans) == 1
    assert fp8.template.is_meta and nvfp4.template.is_meta
    assert fp8.template.__tensor_flatten__()[0] == ["_qdata", "_param_scale"]
    assert nvfp4.template.__tensor_flatten__()[0] == [
        "_qdata",
        "_param_scale",
        "_param_block_scale",
    ]
    assert [span.key for span in fp8.spans] == [
        "model.fp8.weight",
        "model.fp8.weight_scale",
    ]
    assert [span.key for span in nvfp4.spans] == [
        "model.nvfp4.weight",
        "model.nvfp4.weight_scale_2",
        "model.nvfp4.weight_scale",
    ]


def _full_lora_handle() -> _Handle:
    entries: dict[str, tuple[str, tuple[int, ...]]] = {}
    tensors: dict[str, torch.Tensor] = {}
    stems = ["model.embed_tokens"] + [f"model.linear_{index}" for index in range(336)]
    stems.extend(f"vision_model.linear_{index}" for index in range(163))
    for stem in stems:
        if stem == "model.embed_tokens":
            entries[_lora_key(stem, "down")] = ("BF16", (64, 3840))
            entries[_lora_key(stem, "up")] = ("BF16", (262208, 64))
        else:
            entries[_lora_key(stem, "down")] = ("BF16", (64, 16))
            entries[_lora_key(stem, "up")] = ("BF16", (16, 64))
    assert len(entries) == 1000
    return _Handle(entries, tensors)


def _lora_key(stem: str, side: str) -> str:
    return f"text_encoders.transformer.{stem}.lora_{side}.weight"


def test_ltx23_gemma_lora_plan_classifies_every_text_and_vision_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gemma-lora.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(path.resolve(), gemma._TEXT_LORA_SIZE_BYTES, 1, "header")
    probe = SimpleNamespace(
        format="safetensors",
        identity=identity,
        schema_sha256=gemma.LTX23_GEMMA_TEXT_LORA_SCHEMA_SHA256,
        tensor_count=1000,
        tensor_dtypes=("BF16",),
    )
    monkeypatch.setattr(gemma, "probe_artifact", lambda _path: probe)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: _full_lora_handle())

    plan = gemma.plan_ltx23_gemma_text_lora(path)

    assert len(plan.text_targets) == 337
    assert len(plan.ignored_vision_targets) == 163
    assert plan.embedding_target == "model.language_model.embed_tokens"
    assert plan.rank == 64
    embedding = next(pair for pair in plan.pairs if pair.kind == "embedding")
    assert embedding.module_name == "model.language_model.embed_tokens"
    assert embedding.down_key.endswith("model.embed_tokens.lora_down.weight")


def test_ltx23_gemma_embedding_and_linear_lora_are_additive_without_base_merge() -> None:
    base_embedding = nn.Embedding(6, 4)
    with torch.no_grad():
        base_embedding.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    embedding = gemma.LTX23GemmaEmbeddingLora(base_embedding)
    down = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]])
    up = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    embedding.add_lora_adapter("exact", down, up)
    ids = torch.tensor([[1, 4]])
    expected_embedding = torch.nn.functional.embedding(ids, base_embedding.weight) + (
        torch.nn.functional.embedding(ids, up) @ down
    )
    assert torch.equal(embedding(ids), expected_embedding)

    base_linear = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        base_linear.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    linear = StoredDenseLoraLinear(base_linear)
    linear_down = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]])
    linear_up = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    linear.add_lora_adapter("exact", linear_down, linear_up, alpha=None)
    linear.set_lora_strength("exact", 1.0)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected_linear = base_linear(inputs) + (inputs @ linear_down.T) @ linear_up.T
    assert torch.equal(linear(inputs), expected_linear)


@pytest.mark.parametrize("format_name", ("fp8", "nvfp4"))
def test_strict_comfy_text_execution_dequantizes_then_uses_ordinary_linear(
    format_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if format_name == "fp8":
        weight = restore_global_fp8_tensor(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]],
                dtype=torch.float8_e4m3fn,
            ),
            torch.tensor(0.25, dtype=torch.float32),
            torch.bfloat16,
        )
        linear = StoredFP8Linear(weight, input_scale=None)
        values = torch.tensor([[1.0, 0.5, -1.0, 2.0]], dtype=torch.bfloat16)
        assert weight.dequantize()[0, 0] == torch.tensor(0.25, dtype=torch.bfloat16)
    else:
        weight = restore_nvfp4_tensor(
            torch.full((128, 32), 0x77, dtype=torch.uint8),
            torch.ones((128, 4), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            (128, 64),
            torch.bfloat16,
        )
        linear = StoredNVFP4Linear(weight, input_scale=None)
        values = torch.ones((1, 64), dtype=torch.bfloat16)
        assert weight.dequantize()[0, 0] == torch.tensor(1.5, dtype=torch.bfloat16)
    linear.set_execution_policy("strict_comfy_full_precision_mm")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strict Comfy text execution called a quantized GEMM")

    monkeypatch.setattr(stored_execution, "_direct_kitchen_fp8_linear", forbidden)
    monkeypatch.setattr(stored_execution, "_direct_kitchen_nvfp4_linear", forbidden)
    observed = linear(values)

    expected = torch.nn.functional.linear(values, weight.dequantize().to(values.dtype))
    assert torch.equal(observed, expected)
    assert observed.device == values.device
    assert observed.dtype is torch.bfloat16
    assert linear.full_precision_dispatch_count == 1
    assert linear.native_dispatch_count == 0
    assert linear.rejected_dispatch_count == 0
    assert linear.dense_fallback_count == 0


class _TinyLoraGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        language = self.model.language_model
        language.embed_tokens = nn.Embedding(6, 4)
        for index in range(336):
            setattr(language, f"linear_{index}", StoredDenseLoraLinear(nn.Linear(4, 4, bias=False)))
        self.lm_head = nn.Linear(4, 6, bias=False)
        self.lm_head.weight = language.embed_tokens.weight
        self._latentslate_ltx23_gemma_quant_modules = MappingProxyType(
            {f"model.language_model.linear_{index}": "nvfp4" for index in range(336)}
        )


def test_ltx23_gemma_lora_installation_loads_only_text_pairs_and_proves_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lora.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, "header")
    pairs = [
        gemma.LTX23GemmaTextLoraTarget(
            "model.language_model.embed_tokens", "embed.down", "embed.up", "embedding"
        )
    ]
    pairs.extend(
        gemma.LTX23GemmaTextLoraTarget(
            f"model.language_model.linear_{index}", f"linear.{index}.down", f"linear.{index}.up", "linear"
        )
        for index in range(336)
    )
    entries: dict[str, tuple[str, tuple[int, ...]]] = {
        "embed.down": ("BF16", (2, 4)),
        "embed.up": ("BF16", (6, 2)),
    }
    tensors: dict[str, torch.Tensor] = {
        "embed.down": torch.ones((2, 4), dtype=torch.bfloat16),
        "embed.up": torch.ones((6, 2), dtype=torch.bfloat16),
    }
    for index in range(336):
        entries[f"linear.{index}.down"] = ("BF16", (2, 4))
        entries[f"linear.{index}.up"] = ("BF16", (4, 2))
        tensors[f"linear.{index}.down"] = torch.ones((2, 4), dtype=torch.bfloat16)
        tensors[f"linear.{index}.up"] = torch.ones((4, 2), dtype=torch.bfloat16)
    handle = _Handle(entries, tensors)
    plan = gemma.LTX23GemmaTextLoraPlan(
        identity,
        gemma.LTX23_GEMMA_TEXT_LORA_SCHEMA_SHA256,
        tuple(pair.module_name for pair in pairs),
        tuple(f"vision_model.linear_{index}" for index in range(163)),
        "model.language_model.embed_tokens",
        2,
        tuple(pairs),
    )
    model = _TinyLoraGemma()
    original_embedding = model.model.language_model.embed_tokens
    monkeypatch.setattr(gemma, "revalidate_ltx23_gemma_text_lora", lambda _plan: True)
    monkeypatch.setattr(gemma, "revalidate_artifact", lambda _identity: True)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: handle)
    monkeypatch.setattr(gemma, "_STORED_LINEAR_TYPES", (StoredDenseLoraLinear,))

    application = gemma.install_ltx23_gemma_text_lora(model, plan, adapter_name="prompt", strength=1.0)

    assert isinstance(model.model.language_model.embed_tokens, gemma.LTX23GemmaEmbeddingLora)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert set(handle.requested) == set(entries)
    before = application.dispatch_snapshot()
    model.model.language_model.embed_tokens(torch.tensor([[0, 1]]))
    for index in range(336):
        getattr(model.model.language_model, f"linear_{index}")(torch.ones((1, 4)))
    proof = application.verify_dispatch(before)
    assert proof["target_module_count"] == 337
    assert proof["total_dispatches"] == 337
    assert proof["ignored_vision_target_count"] == 163
    application.set_strength(0.0)
    base_snapshot = application.dispatch_snapshot()
    model.model.language_model.embed_tokens(torch.tensor([[0, 1]]))
    for index in range(336):
        getattr(model.model.language_model, f"linear_{index}")(torch.ones((1, 4)))
    assert application.dispatch_snapshot() == base_snapshot
    try:
        application.set_strength(1.0)
        raise RuntimeError("cancelled text phase")
    except RuntimeError:
        application.set_strength(0.0)
    assert all(
        adapter.strength == 0.0
        for module_name in application.target_modules
        for adapter in gemma._lora_module(model, module_name)._lora_adapters.values()
    )
    application.remove()
    assert model.model.language_model.embed_tokens is original_embedding
    assert model.lm_head.weight is original_embedding.weight


def test_ltx23_embedding_lora_accepts_real_scaled_embedding_subclass() -> None:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    base = Gemma3TextScaledWordEmbedding(8, 4, padding_idx=0)
    wrapped = gemma.LTX23GemmaEmbeddingLora(base)
    wrapped.add_lora_adapter(
        "prompt",
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((8, 2), dtype=torch.bfloat16),
    )

    input_ids = torch.tensor([[1, 2]])
    assert torch.equal(wrapped(input_ids), base(input_ids))
    assert wrapped.weight is base.weight


def test_ltx23_embedding_lora_scales_nonzero_delta_like_gemma_base() -> None:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    base = Gemma3TextScaledWordEmbedding(8, 4, padding_idx=0)
    with torch.no_grad():
        base.weight.zero_()
    wrapped = gemma.LTX23GemmaEmbeddingLora(base)
    down = torch.ones((1, 4), dtype=torch.float32)
    up = torch.zeros((8, 1), dtype=torch.float32)
    up[3, 0] = 2.0
    wrapped.add_lora_adapter("prompt", down, up)

    observed = wrapped(torch.tensor([[3]]))
    expected = torch.full_like(observed, 2.0 * base.embed_scale)
    assert torch.allclose(observed, expected)


class _TinyStreamedLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList((nn.Linear(4, 4), nn.Linear(4, 4)))
        self.norm = nn.LayerNorm(4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden).tanh()
        return self.norm(hidden)


class _TinyStreamedGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _TinyStreamedLanguage()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        self._latentslate_ltx23_gemma_text_only = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model(input_ids)

    def outer_prompt_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror Gemma's outer-shell lookup before ``language_model.forward``."""

        embedding = self.model.language_model.embed_tokens(input_ids)
        return embedding, self.model.language_model(input_ids)


class _FakeDynamicBackend:
    instances: ClassVar[list[_FakeDynamicBackend]] = []

    def __init__(self, _device, *, virtual_bytes: int, diagnostic=None) -> None:
        requested = torch.device(_device)
        self.device = (
            torch.device("cuda:0")
            if requested.type == "cuda" and requested.index is None
            else requested
        )
        self.virtual_bytes = virtual_bytes
        self.diagnostic = diagnostic
        self.groups = {}
        self.events = []
        self.closed = False
        self.active_layers = 0
        self.maximum_active_layers = 0
        type(self).instances.append(self)

    @property
    def allocation_started(self) -> bool:
        return True

    @staticmethod
    def group_bytes(values) -> int:
        return sum(value.nbytes for value in values)

    def allocate_group(self, key, values) -> None:
        self.groups[key] = values
        self.events.append(("allocate", key))

    def prioritize(self) -> None:
        self.events.append(("prioritize", None))

    def acquire(self, key):
        from latentslate_engine.runtime.framework.residency.dynamic import DynamicResidencyLease

        self.events.append(("acquire", key))
        if len(self.groups[key]) < len(next(iter(self.groups.values()))):
            self.active_layers += 1
            self.maximum_active_layers = max(self.maximum_active_layers, self.active_layers)
        return DynamicResidencyLease(self.groups[key], SimpleNamespace(key=key))

    def prefetch(self, key):
        raise AssertionError(f"synchronous Gemma must not prefetch {key}")

    def wait(self, lease) -> None:
        self.events.append(("wait", lease.token.key))

    def release(self, lease) -> None:
        self.events.append(("release", lease.token.key))
        if len(self.groups[lease.token.key]) < len(next(iter(self.groups.values()))):
            self.active_layers -= 1

    def synchronize(self, lease) -> None:
        self.events.append(("synchronize", lease.token.key))

    def invalidate(self, *, reason: str) -> None:
        self.events.append(("invalidate", reason))

    def invalidate_groups(self, keys, *, reason: str) -> None:
        self.events.append(("invalidate_groups", (tuple(keys), reason)))

    def close(self) -> None:
        self.events.append(("close", None))
        self.closed = True

    def diagnostics(self):
        return {
            "backend": "comfy-aimdo",
            "version": "0.4.15",
            "mode": "dynamic_vbar",
            "physical_bytes": 1,
            "staged_bytes": 1,
            "virtual_bytes": self.virtual_bytes,
            "allocation_count": len(self.groups),
            "live_allocations": 0 if self.closed else len(self.groups),
            "live_bytes": 0 if self.closed else self.virtual_bytes,
            "loaded_bytes": 0,
            "faults": 3,
            "signature_hits": 0,
            "signature_misses": 3,
            "fault_none_temporaries": 0,
            "pinned_copy_bytes": 1,
            "pageable_copy_bytes": 0,
            "transfer_events": 1,
            "transfer_waits": 1,
            "prioritize_calls": 1,
            "unpin_calls": 3,
            "free_calls": int(self.closed),
            "dirty_epoch": 1,
            "lora_invalidations": 1,
            "base_restores": 1,
            "copy_stream_count": 2,
            "prefetch": False,
            "prefetch_calls": 0,
            "allocator_plugin": False,
            "poisoned": False,
            "close_failed": False,
            "poison_reason": None,
            "host_registration": {"fixture": True},
        }

    def terminal_poison_reason(self):
        return None


@pytest.mark.parametrize(
    ("path", "expected_group"),
    (
        ("layers.0.self_attn.q_proj.base", "layers.0"),
        ("layers.47.post_attention_layernorm", "layers.47"),
        ("layers.0.self_attn.q_proj._lora_adapters.prompt.lora_A", "layers.0.patch"),
        ("layers.47.mlp.down_proj.adapter.weight", "layers.47.patch"),
        ("embed_tokens.base", "root"),
        ("norm.weight", "root"),
        ("rotary_emb.inv_freq", "root"),
        ("embed_tokens._lora_adapters.prompt.lora_B", "root.patch"),
    ),
)
def test_gemma_leaf_schedule_maps_production_direct_paths(
    path: str, expected_group: str
) -> None:
    schedule = gemma._gemma_leaf_schedule(path, (), {})

    assert schedule.group == expected_group


@pytest.mark.parametrize("path", ("layers", "layers.bad.weight", "layers.01.weight"))
def test_gemma_leaf_schedule_rejects_noncanonical_layer_paths(path: str) -> None:
    with pytest.raises(ValueError, match="layer leaf path is not canonical"):
        gemma._gemma_leaf_schedule(path, (), {})


def test_gemma_leaf_schedule_has_49_base_groups_and_mirrored_patch_order() -> None:
    base_paths = ["embed_tokens.base", *(
        f"layers.{index}.self_attn.q_proj.base" for index in range(48)
    )]
    patch_paths = ["embed_tokens._lora_adapters.prompt.lora_A", *(
        f"layers.{index}.self_attn.q_proj._lora_adapters.prompt.lora_A"
        for index in range(48)
    )]
    base_groups = tuple(
        gemma._gemma_leaf_schedule(path, (), {}).group for path in base_paths
    )
    patch_groups = tuple(
        gemma._gemma_leaf_schedule(path, (), {}).group for path in patch_paths
    )
    expected_base = ("root", *(f"layers.{index}" for index in range(48)))
    expected_patch = (
        "root.patch",
        *(f"layers.{index}.patch" for index in range(48)),
    )

    assert base_groups == expected_base
    assert patch_groups == expected_patch
    assert len(set(base_groups)) == len(set(patch_groups)) == 49
    assert tuple(
        group
        for pair in zip(base_groups, patch_groups, strict=True)
        for group in pair
    ) == tuple(
        group
        for index in range(49)
        for group in (
            "root" if index == 0 else f"layers.{index - 1}",
            "root.patch" if index == 0 else f"layers.{index - 1}.patch",
        )
    )


def test_ltx23_dynamic_stage_uses_per_leaf_backend_and_retains_warm_until_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    _FakeDynamicBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FakeDynamicBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(gemma.torch.cuda, "empty_cache", lambda: None)
    model = _TinyStreamedGemma()
    stage = gemma.LTX23GemmaMixedTextStage(model, "cuda:0", dynamic_policy="required")
    topology = {leaf.path: leaf.schedule_groups for leaf in stage._leaf_storage}

    assert topology == {
        "embed_tokens": ("root",),
        "layers.0": ("layers.0",),
        "layers.1": ("layers.1",),
        "norm": ("root",),
    }

    stage.onload()
    output = model(torch.tensor([[1, 2]]))
    stage.invalidate_patch_state(to_base=True)
    stage.offload()
    backend = _FakeDynamicBackend.instances[0]

    assert tuple(output.shape) == (1, 2, 4)
    assert len(backend.groups) == 4
    assert backend.active_layers == 0
    assert backend.events[-1] != ("close", None)
    proof = stage.diagnostics()
    assert proof["leaf_allocation_count"] == 4
    assert proof["leaf_allocation_count"] > proof["layer_count"] + 1
    assert proof["mode"] == "dynamic_vbar_per_leaf"
    assert proof["root_activation"] == "per_model_forward_fault"
    assert proof["dynamic_vram"]["policy"] == "required"
    assert proof["dynamic_vram"]["live_allocations"] == 4
    assert proof["dynamic_vram"]["prefetch_calls"] == 0
    assert proof["warm_request_index"] == 1
    assert proof["root_transitions"] == 1
    assert proof["layer_transitions"] == 2
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())

    stage.onload()
    model(torch.tensor([[1, 2]]))
    stage.offload()
    second_proof = stage.diagnostics()
    assert len(_FakeDynamicBackend.instances) == 1
    assert second_proof["warm_request_index"] == 2
    assert second_proof["root_transitions"] == 1
    assert second_proof["layer_transitions"] == 2
    assert second_proof["dynamic_vram"]["warm_request_index"] == 2
    stage.close()
    assert backend.events[-1] == ("close", None)
    assert stage.diagnostics()["dynamic_vram"]["live_allocations"] == 0


def test_ltx23_gemma_leaf_groups_separate_patch_and_reactivate_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    _FakeDynamicBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FakeDynamicBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(gemma.torch.cuda, "empty_cache", lambda: None)
    model = _TinyStreamedGemma()
    language = model.model.language_model
    embedding = gemma.LTX23GemmaEmbeddingLora(language.embed_tokens)
    embedding.add_lora_adapter(
        "prompt",
        torch.ones((1, 4), dtype=torch.float32),
        torch.ones((8, 1), dtype=torch.float32),
    )
    language.embed_tokens = embedding
    model.lm_head.weight = embedding.weight
    stage = gemma.LTX23GemmaMixedTextStage(
        model, "cuda:0", dynamic_policy="required"
    )

    groups = {leaf.path: leaf for leaf in stage._leaf_storage}
    patch_paths = [path for path in groups if "_lora_adapters" in path]
    assert patch_paths
    assert all(groups[path].schedule_groups == ("root.patch",) for path in patch_paths)
    assert all(not groups[path].force_resident for path in patch_paths)
    assert groups["embed_tokens.base"].schedule_groups == ("root",)
    assert groups["embed_tokens.base"].force_resident is True
    assert sum("embed_tokens.base" == path for path in groups) == 1
    assert all("lm_head" not in path for path in groups)

    stage.onload()
    backend = _FakeDynamicBackend.instances[0]
    model(torch.tensor([[1, 2]]))
    stage.invalidate_patch_state(to_base=True)
    transition_index = len(backend.events)
    model(torch.tensor([[1, 2]]))
    assert not any(
        event[0] == "acquire" and event[1] in patch_paths
        for event in backend.events[transition_index:]
    )
    invalidations = [event for event in backend.events if event[0] == "invalidate_groups"]
    assert invalidations == [
        ("invalidate_groups", (tuple(patch_paths), "lora_to_base"))
    ]
    stage.offload()

    stage.onload()
    request_two = len(backend.events)
    model(torch.tensor([[1, 2]]))
    assert any(
        event[0] == "acquire" and event[1] in patch_paths
        for event in backend.events[request_two:]
    )
    assert all(event[0] != "prefetch" for event in backend.events)
    stage.offload()
    stage.close()


@pytest.mark.parametrize("close_fails", [False, True], ids=["closed", "poison-retained"])
def test_ltx23_base_file_handle_closes_only_after_backend_quiescent_cleanup(
    close_fails: bool,
) -> None:
    events: list[str] = []
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")

    class _Backend:
        def close(self) -> None:
            events.append("backend-close")
            if close_fails:
                raise RuntimeError("quiescence failed")

        def diagnostics(self):
            return {"copy_stream_count": 2}

    class _Handle:
        def close(self) -> None:
            events.append("file-close")

    handle = _Handle()
    stage._dynamic_backend = _Backend()
    stage._base_file_handle = handle
    stage._base_file_handle_opened = 1

    if close_fails:
        with pytest.raises(RuntimeError, match="quiescence failed"):
            stage.close()
        assert events == ["backend-close"]
        assert stage._dynamic_backend is not None
        assert stage._base_file_handle is handle
    else:
        stage.close()
        assert events == ["backend-close", "file-close"]
        assert stage._dynamic_backend is None
        assert stage._base_file_handle is None
        assert stage._base_file_handle_closed == 1


@pytest.mark.parametrize("failure_phase", ["allocate", "prioritize"])
def test_ltx23_dynamic_initialization_failure_closes_owned_backend_once(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    class _FailingBackend(_FakeDynamicBackend):
        close_calls = 0

        def allocate_group(self, key, values) -> None:
            super().allocate_group(key, values)
            if failure_phase == "allocate" and len(self.groups) == 2:
                raise RuntimeError("fixture allocate failed")

        def prioritize(self) -> None:
            super().prioritize()
            if failure_phase == "prioritize":
                raise RuntimeError("fixture prioritize failed")

        def close(self) -> None:
            type(self).close_calls += 1
            super().close()

    _FailingBackend.instances.clear()
    _FailingBackend.close_calls = 0
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FailingBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="required"
    )

    with pytest.raises(RuntimeError, match=f"fixture {failure_phase} failed"):
        stage.onload()

    backend = _FailingBackend.instances[0]
    assert _FailingBackend.close_calls == 1
    assert backend.closed is True
    assert stage._dynamic_backend is None


def test_ltx23_dynamic_initialization_breadcrumbs_are_bounded_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    _FakeDynamicBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FakeDynamicBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(gemma.torch.cuda, "empty_cache", lambda: None)
    records: list[tuple[float, str | None]] = []
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(),
        "cuda",
        dynamic_policy="required",
        progress=lambda value, message: records.append((value, message)),
    )

    stage.onload()
    stage.offload()

    phases = [message.split(" (", 1)[0] for _value, message in records if message]
    assert phases == [
        "LTX AIMDO constructor_begin",
        "LTX AIMDO constructor_after",
        "LTX AIMDO leaf_allocation_begin",
        "LTX AIMDO leaf_allocation_after",
        "LTX AIMDO prioritize_begin",
        "LTX AIMDO prioritize_after",
    ]
    assert all(value == 0.0785 for value, _message in records)
    assert all(message is not None and len(message) <= 768 for _value, message in records)
    assert any("device=cuda:0" in message for _value, message in records if message)


def test_ltx23_post_constructor_progress_failure_closes_owned_backend_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    class _TrackedBackend(_FakeDynamicBackend):
        close_calls = 0

        def close(self) -> None:
            type(self).close_calls += 1
            super().close()

    def progress(_value: float, message: str | None) -> None:
        if message is not None and message.startswith("LTX AIMDO constructor_after"):
            raise RuntimeError("fixture progress failed")

    _TrackedBackend.instances.clear()
    _TrackedBackend.close_calls = 0
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _TrackedBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(),
        "cuda:0",
        dynamic_policy="required",
        progress=progress,
    )

    with pytest.raises(RuntimeError, match="fixture progress failed"):
        stage.onload()

    assert _TrackedBackend.close_calls == 1
    assert _TrackedBackend.instances[0].closed is True
    assert stage._dynamic_backend is None


def test_ltx23_dynamic_stage_retains_poisoned_backend_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    class _CloseFailedBackend(_FakeDynamicBackend):
        def close(self) -> None:
            self.events.append(("close-failed", None))
            raise RuntimeError("device quiescence failed")

        def diagnostics(self):
            proof = super().diagnostics()
            proof.update(
                poisoned=True,
                close_failed=True,
                poison_reason="device_quiescence_failed",
                loaded_bytes=None,
            )
            return proof

        def terminal_poison_reason(self):
            return "device_quiescence_failed"

    _CloseFailedBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _CloseFailedBackend)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(gemma.torch.cuda, "empty_cache", lambda: None)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="required"
    )
    stage.onload()

    stage.offload()
    with pytest.raises(RuntimeError, match="device quiescence failed"):
        stage.close()

    backend = _CloseFailedBackend.instances[0]
    assert stage._dynamic_backend is backend
    assert backend.closed is False
    assert stage.terminal_poison_reason() == "device_quiescence_failed"
    assert stage._leaf_scheduler is not None


def test_ltx23_leaf_stage_global_sync_failure_freezes_graph_for_hard_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module
    from latentslate_engine.runtime.framework.residency.dynamic import (
        DynamicResidencyPoisoned,
    )

    _FakeDynamicBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FakeDynamicBackend)
    monkeypatch.setattr(gemma.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gemma.torch.cuda, "synchronize", lambda *_args: None)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="required"
    )
    stage.onload()
    stage.model(torch.tensor([[1, 2]]))
    backend = _FakeDynamicBackend.instances[0]
    frozen_events = list(backend.events)
    frozen_bindings = dict(stage._leaf_bindings)

    def fail_sync(*_args) -> None:
        raise RuntimeError("fixture whole-device sync failed")

    monkeypatch.setattr(gemma.torch.cuda, "synchronize", fail_sync)
    with pytest.raises(DynamicResidencyPoisoned, match="device_quiescence_failed"):
        stage.offload()

    assert stage.terminal_poison_reason() == "device_quiescence_failed"
    assert stage._leaf_bindings == frozen_bindings
    assert backend.events == frozen_events
    backend.diagnostics = lambda: (_ for _ in ()).throw(
        AssertionError("poisoned diagnostics must not query the backend")
    )
    proof = stage.diagnostics()
    assert proof["leaf_scheduler"]["active_groups"] == 0
    for operation in (stage._leaf_scheduler.clear_stage, stage._leaf_scheduler.close):
        with pytest.raises(DynamicResidencyPoisoned, match="device_quiescence_failed"):
            operation()
    assert backend.events == frozen_events


def test_ltx23_dynamic_policy_falls_back_only_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module
    from latentslate_engine.runtime.framework.residency.dynamic import DynamicResidencyUnavailable

    class _Unavailable:
        group_bytes = staticmethod(lambda values: sum(value.nbytes for value in values))

        def __init__(self, *_args, **_kwargs) -> None:
            raise DynamicResidencyUnavailable("fixture incompatible")

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _Unavailable)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="auto"
    )
    monkeypatch.setattr(stage, "_initialize_transfer_streams", lambda: None)
    stage._initialize_backend()
    assert stage._dynamic_backend is None
    assert stage._dynamic_fallback_reason == "fixture incompatible"

    required = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="required"
    )
    with pytest.raises(DynamicResidencyUnavailable, match="fixture incompatible"):
        required._initialize_backend()

    class _AllocationFailed(_Unavailable):
        def __init__(self, *_args, **_kwargs) -> None:
            raise MemoryError("VBAR allocation failed")

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _AllocationFailed)
    no_midstage_fallback = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="auto"
    )
    with pytest.raises(MemoryError, match="VBAR allocation failed"):
        no_midstage_fallback._initialize_backend()


def test_ltx23_source_backed_setup_unavailable_materializes_authenticated_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module
    from latentslate_engine.runtime.framework.residency.dynamic import (
        DynamicResidencyUnavailable,
    )

    model = _TinyStreamedGemma()
    payload: dict[str, torch.Tensor] = {}
    descriptors: dict[int, aimdo_module.AimdoFileBackedValue] = {}
    offset = 64
    for module_name, module in model.model.language_model.named_modules():
        for name, value in tuple(module._parameters.items()):
            if value is None:
                continue
            key = f"model.{module_name}.{name}".replace("..", ".")
            source = value.detach().clone()
            payload[key] = source
            template = nn.Parameter(
                torch.empty(value.shape, dtype=value.dtype, device="meta"),
                requires_grad=value.requires_grad,
            )
            module._parameters[name] = template
            size = source.numel() * source.element_size()
            descriptors[id(template)] = aimdo_module.AimdoFileBackedValue(
                template,
                (
                    aimdo_module.AimdoFileSpan(
                        "ltx23_gemma_base",
                        key,
                        offset,
                        size,
                        source.dtype,
                        tuple(source.shape),
                    ),
                ),
            )
            offset += size
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    path = tmp_path / "base.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(
        path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, "h"
    )
    plan = gemma.LTX23GemmaMixedTextPlan(
        identity,
        "schema",
        MappingProxyType({}),
        tuple(payload),
        (),
        (),
        MappingProxyType({
            key: gemma.LTX23SafetensorSpan(
                key,
                "BF16" if tensor.dtype is torch.bfloat16 else "F32",
                tuple(tensor.shape),
                0,
                tensor.numel() * tensor.element_size(),
            )
            for key, tensor in payload.items()
        }),
        1,
    )
    model._latentslate_ltx23_gemma_source_backed = True
    model._latentslate_ltx23_gemma_source_descriptors = descriptors
    model._latentslate_ltx23_gemma_plan = plan

    class _Payload:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_tensor(self, key):
            return payload[key]

    class _Unavailable:
        group_bytes = staticmethod(aimdo_module.AimdoDynamicResidency.group_bytes)

        def __init__(self, *_args, **_kwargs) -> None:
            raise DynamicResidencyUnavailable("fixture backend unavailable")

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _Unavailable)
    monkeypatch.setattr(
        gemma, "revalidate_ltx23_gemma_mixed_text_encoder", lambda _plan: True
    )
    monkeypatch.setattr(gemma, "revalidate_artifact", lambda _identity: True)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: _Payload())
    stage = gemma.LTX23GemmaMixedTextStage(model, "cuda:0", dynamic_policy="auto")
    monkeypatch.setattr(stage, "_initialize_transfer_streams", lambda: None)

    stage._initialize_backend()

    proof = stage.diagnostics()["dynamic_vram"]
    assert stage._source_descriptors == {}
    assert all(not value.is_meta for value in model.model.language_model.parameters())
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert proof["base_file_requested"] is True
    assert proof["base_file_backed"] is False
    assert proof["base_file_fallback_reason"].startswith(
        "aimdo_backend_unavailable:"
    )
    assert proof["base_file_read_calls"] == proof["base_file_read_bytes"] == 0


@pytest.mark.parametrize(
    "fallback_reason",
    (
        "host_buffer_capability_unavailable: fixture import",
        "host_buffer_setup_failed: fixture allocation",
    ),
)
def test_ltx23_required_source_backed_host_buffer_failure_never_materializes_cpu(
    fallback_reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module
    from latentslate_engine.runtime.framework.residency.dynamic import (
        DynamicResidencyUnavailable,
    )

    class _FallbackBackend:
        instances: ClassVar[list[_FallbackBackend]] = []
        group_bytes = staticmethod(aimdo_module.AimdoDynamicResidency.group_bytes)

        def __init__(self, device, *, virtual_bytes, diagnostic=None) -> None:
            self.device = torch.device(device)
            self.virtual_bytes = virtual_bytes
            self.diagnostic = diagnostic
            self.closed = False
            type(self).instances.append(self)

        def allocate_group(self, _key, _values) -> None:
            pass

        def prioritize(self) -> None:
            pass

        def diagnostics(self):
            return {
                "copy_strategy": "per_physical",
                "copy_fallback_reason": fallback_reason,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FallbackBackend)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="required"
    )
    _install_test_file_descriptors(stage, aimdo_module)
    materialized: list[bool] = []
    monkeypatch.setattr(
        stage,
        "_materialize_source_backed_cpu_fallback",
        lambda: materialized.append(True),
    )

    with pytest.raises(
        DynamicResidencyUnavailable,
        match="required AIMDO source-backed HostBuffer is unavailable",
    ):
        stage._initialize_backend()

    assert materialized == []
    assert stage._source_descriptors
    assert _FallbackBackend.instances[0].closed is True
    assert stage._dynamic_backend is None


@pytest.mark.parametrize(
    "fallback_reason",
    (
        "host_buffer_capability_unavailable: fixture import",
        "host_buffer_setup_failed: fixture allocation",
    ),
)
def test_ltx23_auto_source_backed_host_buffer_failure_uses_cpu_fallback(
    fallback_reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    class _FallbackBackend:
        instances: ClassVar[list[_FallbackBackend]] = []
        group_bytes = staticmethod(aimdo_module.AimdoDynamicResidency.group_bytes)

        def __init__(self, device, *, virtual_bytes, diagnostic=None) -> None:
            self.device = torch.device(device)
            self.virtual_bytes = virtual_bytes
            self.diagnostic = diagnostic
            self.closed = False
            type(self).instances.append(self)

        def allocate_group(self, _key, _values) -> None:
            pass

        def prioritize(self) -> None:
            pass

        def diagnostics(self):
            return {
                "copy_strategy": "per_physical",
                "copy_fallback_reason": fallback_reason,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FallbackBackend)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:0", dynamic_policy="auto"
    )
    _install_test_file_descriptors(stage, aimdo_module)
    materialized: list[bool] = []

    def materialize() -> None:
        materialized.append(True)
        stage._source_descriptors.clear()

    monkeypatch.setattr(
        stage, "_materialize_source_backed_cpu_fallback", materialize
    )
    monkeypatch.setattr(stage, "_initialize_transfer_streams", lambda: None)

    stage._initialize_backend()

    assert materialized == [True]
    assert stage._base_file_fallback_reason == fallback_reason
    assert stage._dynamic_fallback_reason == fallback_reason
    assert _FallbackBackend.instances[0].closed is True
    assert stage._dynamic_backend is None


def _install_test_file_descriptors(stage, aimdo_module) -> None:
    offset = 64
    descriptors = {}
    for storage in (stage._root_storage, *stage._layer_storage):
        for slot in storage.slots:
            value = slot.cpu_value
            if id(value) in descriptors:
                continue
            template = torch.empty(value.shape, dtype=value.dtype, device="meta")
            if isinstance(value, nn.Parameter):
                template = nn.Parameter(
                    template, requires_grad=value.requires_grad
                )
            size = value.numel() * value.element_size()
            descriptors[id(value)] = aimdo_module.AimdoFileBackedValue(
                template,
                (
                    aimdo_module.AimdoFileSpan(
                        "ltx23_gemma_base",
                        f"fixture.{len(descriptors)}",
                        offset,
                        size,
                        value.dtype,
                        tuple(value.shape),
                    ),
                ),
            )
            offset += size
    stage._source_descriptors = descriptors
    stage._base_file_requested = True


def test_ltx23_dynamic_stage_adopts_canonical_backend_device_and_rejects_bad_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module
    from latentslate_engine.runtime.framework.residency.dynamic import (
        DynamicResidencyDeviceError,
    )

    _FakeDynamicBackend.instances.clear()
    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _FakeDynamicBackend)
    stage = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda", dynamic_policy="required"
    )
    stage._initialize_backend()

    assert stage.execution_device == torch.device("cuda:0")
    assert stage._dynamic_backend is _FakeDynamicBackend.instances[-1]
    stage._dynamic_backend.close()

    class _InvalidDevice:
        group_bytes = staticmethod(lambda values: sum(value.nbytes for value in values))

        def __init__(self, *_args, **_kwargs) -> None:
            raise DynamicResidencyDeviceError("fixture CUDA index is unavailable")

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", _InvalidDevice)
    invalid = gemma.LTX23GemmaMixedTextStage(
        _TinyStreamedGemma(), "cuda:9", dynamic_policy="auto"
    )
    with pytest.raises(DynamicResidencyDeviceError, match="index is unavailable"):
        invalid._initialize_backend()

    assert invalid._dynamic_backend is None
    assert invalid._dynamic_fallback_reason is None
    assert invalid._transfer_streams == ()


def test_ltx23_gemma_stage_streams_layers_and_retains_tied_head_until_offload() -> None:
    model = _TinyStreamedGemma()
    stage = gemma.LTX23GemmaMixedTextStage(model, "cpu")

    stage.onload()
    hidden = model.model.language_model(torch.tensor([[1, 2]]))
    logits = model.lm_head(hidden)
    second_hidden = model.model.language_model(torch.tensor([[3, 4]]))
    second_logits = model.lm_head(second_hidden)
    evidence = stage.diagnostics()
    stage.offload()

    assert tuple(logits.shape) == (1, 2, 8)
    assert tuple(second_logits.shape) == (1, 2, 8)
    assert evidence["mode"] == "layer_streamed_cpu_master"
    assert evidence["root_activation"] == "stage_onload"
    assert evidence["layer_count"] == 2
    assert evidence["root_transitions"] == 1
    assert evidence["layer_transitions"] == 4
    assert evidence["execution_policy"] == "strict_comfy_full_precision_mm"
    assert evidence["native_quantized_dispatches"] == 0
    assert evidence["full_precision_dispatches"] == 0
    assert evidence["transfer_mode"] == "blocking_cpu"
    assert evidence["transfer_stream_count"] == 0
    assert evidence["strict_cuda_parity"] is False
    assert evidence["maximum_live_layer_bindings"] == 1
    assert evidence["maximum_live_layer_bytes"] <= evidence["largest_layer_weight_bytes"]
    assert evidence["live_layer_bindings"] == evidence["live_layer_bytes"] == 0
    assert evidence["dynamic_vbar_prefetch"] is False
    assert evidence["required_weight_bytes"] == sum(
        leaf.storage.physical_bytes for leaf in stage._leaf_storage
    )
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_ltx23_gemma_stage_binds_nonzero_embedding_lora_before_outer_lookup() -> None:
    """The outer Gemma shell sees a live embedding LoRA before its first lookup.

    CPU-only CI cannot create CUDA token indices, so this exercises the same
    ordering invariant directly: ``onload`` binds all root storage, including
    the wrapper's nonzero adapter tensors, before the outer shell performs the
    embedding lookup that precedes ``language_model.forward``.
    """

    model = _TinyStreamedGemma()
    base = model.model.language_model.embed_tokens
    with torch.no_grad():
        base.weight.zero_()
    wrapped = gemma.LTX23GemmaEmbeddingLora(base)
    down = torch.ones((1, 4), dtype=torch.float32)
    up = torch.zeros((8, 1), dtype=torch.float32)
    up[3, 0] = 2.0
    wrapped.add_lora_adapter("prompt", down, up)
    model.model.language_model.embed_tokens = wrapped
    gemma._retie_lm_head(model)

    stage = gemma.LTX23GemmaMixedTextStage(model, "cpu")
    captured_names = {slot.name for slot in stage._root_storage.slots}
    assert {"weight", "down", "up"} <= captured_names

    stage.onload()
    assert stage._root_binding is not None
    assert stage._root_binding.active
    assert stage.diagnostics()["root_transitions"] == 1
    embedding, hidden = model.outer_prompt_forward(torch.tensor([[3]]))
    stage.offload()

    assert torch.allclose(embedding, torch.full_like(embedding, 2.0))
    assert tuple(hidden.shape) == (1, 1, 4)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight


def test_ltx23_gemma_stage_initializes_two_rotating_nonblocking_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    streams: list[SimpleNamespace] = []
    waits: list[object] = []

    def stream_factory(*, device):
        stream = SimpleNamespace(device=device, index=len(streams))
        streams.append(stream)
        return stream

    class _Event:
        def __init__(self) -> None:
            self.recorded_on = None

        def record(self, stream) -> None:
            self.recorded_on = stream

    current = SimpleNamespace(wait_event=lambda event: waits.append(event))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    stage._initialize_transfer_streams()

    copies: list[tuple[object, bool]] = []

    class _Storage:
        def copy_to(self, device, *, non_blocking=False):
            copies.append((device, non_blocking))
            return SimpleNamespace()

    stage._copy_with_transfer(_Storage())
    stage._copy_with_transfer(_Storage())

    assert len(stage._transfer_streams) == 2
    assert [stream.index for stream in streams] == [0, 1]
    assert copies == [(torch.device("cuda:0"), True)] * 2
    assert [event.recorded_on.index for event in waits] == [0, 1]
    assert stage._transfer_events == stage._transfer_waits == 2


def test_ltx23_gemma_stage_never_overlaps_live_layer_device_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    stage._active = True
    stage._owner_thread = __import__("threading").get_ident()
    live = 0
    maximum_live = 0
    compute_complete = False

    class _Binding:
        def __init__(self, storage) -> None:
            self.storage = storage
            self.active = False

        def activate(self) -> None:
            nonlocal live, maximum_live, compute_complete
            assert live == 0
            compute_complete = False
            self.active = True
            live += 1
            maximum_live = max(maximum_live, live)

        def record_stream(self, _stream) -> None:
            pass

        def restore_cpu(self) -> None:
            nonlocal live
            assert compute_complete, "layer storage released before compute completion"
            self.active = False
            live -= 1

    class _ComputeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            nonlocal compute_complete
            compute_complete = True

    current = SimpleNamespace()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _ComputeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    monkeypatch.setattr(stage, "_register_storage_best_effort", lambda _storage: None)
    monkeypatch.setattr(stage, "_copy_with_transfer", lambda storage: _Binding(storage))

    first_pre = stage._layer_pre(0)
    second_pre = stage._layer_pre(1)
    first_pre(stage._layers[0], ())
    assert live == 1
    with pytest.raises(RuntimeError, match="non-reentrant"):
        second_pre(stage._layers[1], ())
    stage._layer_post(stage._layers[0], (), object())
    assert live == 0
    second_pre(stage._layers[1], ())
    stage._layer_post(stage._layers[1], (), object())

    assert live == 0
    assert maximum_live == 1
    assert stage._maximum_live_layer_bindings == 1
    assert stage._maximum_live_layer_bytes <= max(
        storage.physical_bytes for storage in stage._layer_storage
    )
    assert stage._layer_compute_barriers == stage._layer_transitions == 2


@pytest.mark.parametrize("hook_name", ["model", "layer"])
def test_ltx23_dynamic_pre_hook_cleanup_never_masks_activation_primary(
    hook_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage._active = True
    stage._owner_thread = __import__("threading").get_ident()
    primary = ValueError(f"{hook_name} activation primary")
    lease = SimpleNamespace()

    class _Binding:
        active = False

        def activate(self) -> None:
            self.active = True
            raise primary

        def restore_cpu(self) -> None:
            raise RuntimeError("restore cleanup failed")

    class _Backend:
        def synchronize(self, _lease) -> None:
            raise RuntimeError("synchronize cleanup failed")

        def release(self, _lease) -> None:
            raise RuntimeError("release cleanup failed")

    binding = _Binding()
    stage._dynamic_backend = _Backend()
    monkeypatch.setattr(stage, "_acquire_dynamic", lambda _storage: (binding, lease))

    with pytest.raises(ValueError) as raised:
        if hook_name == "model":
            stage._model_pre(stage.model, ())
        else:
            stage._layer_pre(0)(stage._layers[0], ())

    assert raised.value is primary
    notes = "\n".join(primary.__notes__)
    assert "synchronize cleanup failed" in notes
    assert "restore cleanup failed" in notes
    assert "release cleanup failed" in notes


@pytest.mark.parametrize("failure_point", ("creation", "record", "synchronize"))
def test_ltx23_gemma_layer_event_failure_restores_binding_and_lifetime_state(
    failure_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    stage._active = True
    stage._owner_thread = __import__("threading").get_ident()
    restored: list[bool] = []
    fallback_barriers: list[torch.device] = []

    class _Binding:
        def __init__(self, storage) -> None:
            self.storage = storage
            self.active = False

        def activate(self) -> None:
            self.active = True

        def record_stream(self, _stream) -> None:
            pass

        def restore_cpu(self) -> None:
            self.active = False
            restored.append(True)

    class _FailingEvent:
        def __init__(self) -> None:
            if failure_point == "creation":
                raise RuntimeError("event creation failed")

        def record(self, _stream) -> None:
            if failure_point == "record":
                raise RuntimeError("event record failed")

        def synchronize(self) -> None:
            if failure_point == "synchronize":
                raise RuntimeError("event synchronize failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FailingEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: fallback_barriers.append(device),
    )
    monkeypatch.setattr(stage, "_register_storage_best_effort", lambda _storage: None)
    monkeypatch.setattr(stage, "_copy_with_transfer", lambda storage: _Binding(storage))
    stage._layer_pre(0)(stage._layers[0], ())

    with pytest.raises(RuntimeError, match=f"event {failure_point} failed"):
        stage._layer_post(stage._layers[0], (), object())

    assert restored == [True]
    assert fallback_barriers == [torch.device("cuda:0")]
    assert stage._layer_binding is None
    assert stage._live_layer_bindings == stage._live_layer_bytes == 0


def test_ltx23_gemma_offload_sync_failure_restores_root_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    stage._active = True
    stage._owner_thread = __import__("threading").get_ident()
    restored: list[bool] = []

    class _RootBinding:
        active = True

        def restore_cpu(self) -> None:
            self.active = False
            restored.append(True)

    stage._root_binding = _RootBinding()  # type: ignore[assignment]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(RuntimeError("offload sync failed")),
    )
    monkeypatch.setattr(stage, "_remove_hooks", lambda: None)

    with pytest.raises(RuntimeError, match="offload sync failed"):
        stage.offload()

    assert restored == [True]
    assert stage._root_binding is None
    assert stage._layer_binding is None
    assert stage._live_layer_bindings == stage._live_layer_bytes == 0
    assert stage._active is False
    assert stage._owner_thread is None


def test_ltx23_gemma_strict_cuda_stream_creation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stream unavailable")),
    )

    with pytest.raises(RuntimeError, match="exactly two CUDA transfer streams"):
        stage._initialize_transfer_streams()
    assert stage._async_transfer_fallbacks == 1


def test_ltx23_host_registration_is_in_place_deduplicated_and_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    class _Cudart:
        def cudaHostRegister(self, ptr, size, _flags):
            calls.append(("register", ptr, size))
            return 0

        def cudaHostUnregister(self, ptr):
            calls.append(("unregister", ptr, 0))
            return 0

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    value = nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    identity = id(value)
    pointer = value.data_ptr()
    alias = value.detach().view_as(value)
    before = value.clone()
    ledger = gemma._LTX23HostRegistrationLedger(1024)

    ledger.consider(value)
    ledger.consider(alias)

    assert id(value) == identity
    assert value.data_ptr() == pointer
    assert alias.data_ptr() == pointer
    assert torch.equal(value, before)
    assert calls == [("register", pointer, value.nbytes)]
    assert ledger.counts["successes"] == 1
    assert ledger.counts["deduplicated_aliases"] == 1
    assert ledger.unregister_owned() == []
    assert calls[-1] == ("unregister", pointer, 0)
    assert ledger.provenance()["owned_active"] == 0


def test_ltx23_host_registration_best_effort_categories_and_foreign_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unregisters: list[int] = []

    class _Cudart:
        def cudaHostRegister(self, ptr, _size, _flags):
            return 712 if ptr == 5 else 1

        def cudaHostUnregister(self, ptr):
            unregisters.append(ptr)
            return 0

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def candidate(
        ptr: int,
        size: int,
        *,
        contiguous: bool = True,
        pinned: bool = False,
    ):
        parameter_type = type(
            "Parameter",
            (),
            {
                "nbytes": size,
                "device": SimpleNamespace(type="cpu"),
                "data_ptr": lambda self: ptr,
                "is_contiguous": lambda self: contiguous,
                "is_pinned": lambda self: pinned,
            },
        )
        return parameter_type()

    ledger = gemma._LTX23HostRegistrationLedger(100)
    ledger.consider(candidate(1, 10, pinned=True))
    ledger.consider(candidate(2, 10, contiguous=False))
    ledger.consider(candidate(3, 200))
    ledger.consider(candidate(4, 20))
    ledger.consider(candidate(5, 20))

    proof = ledger.provenance()
    assert proof["already_registered"] == 2
    assert proof["ineligible"] == 2
    assert proof["failures"] == 1
    assert proof["categories"]["noncontiguous"] == 1
    assert proof["categories"]["budget_exceeded"] == 1
    assert proof["categories"]["register_error"] == 1
    assert ledger.unregister_owned() == []
    assert unregisters == []


def test_ltx23_host_unregister_failure_preserves_primary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cudart:
        def cudaHostRegister(self, _ptr, _size, _flags):
            return 0

        def cudaHostUnregister(self, _ptr):
            raise RuntimeError("unregister failed")

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    value = nn.Parameter(torch.ones(4), requires_grad=False)
    stage._host_registrations.consider(value)
    primary = RuntimeError("primary failure")

    stage._cleanup_host_registrations(primary=primary, synchronized=True)

    assert str(primary) == "primary failure"
    assert any("unregistration failed" in note for note in primary.__notes__)



def test_ltx23_gemma_stage_barriers_before_releasing_cuda_root_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = gemma.LTX23GemmaMixedTextStage(_TinyStreamedGemma(), "cpu")
    stage.execution_device = torch.device("cuda:0")
    stage._active = True
    stage._owner_thread = __import__("threading").get_ident()
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("barrier"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(stage, "_remove_hooks", lambda: events.append("hooks"))
    monkeypatch.setattr(stage, "_restore_cpu", lambda: events.append("restore"))

    stage.offload()

    assert events == ["hooks", "barrier", "restore", "empty"]


@pytest.mark.skipif(
    not os.environ.get("LATENTSLATE_LTX23_GEMMA_HEADER_PATH"),
    reason="set LATENTSLATE_LTX23_GEMMA_HEADER_PATH to opt into exact installed-header validation",
)
def test_opt_in_exact_ltx23_gemma_header_contract() -> None:
    plan = gemma.plan_ltx23_gemma_mixed_text_encoder(
        Path(os.environ["LATENTSLATE_LTX23_GEMMA_HEADER_PATH"])
    )

    assert len(plan.dense_sources) == 290
    assert len(plan.quantized_formats) == 336
    assert len(plan.ignored_auxiliary_sources) == 440


@pytest.mark.skipif(
    not os.environ.get("LATENTSLATE_LTX23_GEMMA_TEXT_LORA_HEADER_PATH"),
    reason="set LATENTSLATE_LTX23_GEMMA_TEXT_LORA_HEADER_PATH to opt into exact LoRA header validation",
)
def test_opt_in_exact_ltx23_gemma_text_lora_header_contract() -> None:
    plan = gemma.plan_ltx23_gemma_text_lora(
        Path(os.environ["LATENTSLATE_LTX23_GEMMA_TEXT_LORA_HEADER_PATH"])
    )

    assert len(plan.text_targets) == 337
    assert len(plan.ignored_vision_targets) == 163
    assert plan.embedding_target == "model.language_model.embed_tokens"
