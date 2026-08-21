from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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

    plan = gemma.plan_ltx23_gemma_mixed_text_encoder(path)

    assert len(plan.dense_sources) == 290
    assert len(plan.quantized_formats) == 336
    assert sum(value == "nvfp4" for value in plan.quantized_formats.values()) == 302
    assert sum(value == "float8_e4m3fn" for value in plan.quantized_formats.values()) == 34
    assert len(plan.auxiliary_sources) == 974
    assert len(plan.ignored_auxiliary_sources) == 440
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

    model = gemma.load_ltx23_gemma_mixed_text_encoder(plan, tmp_path)

    assert isinstance(model.model.language_model.fp8, StoredFP8Linear)
    assert isinstance(model.model.language_model.nvfp4, StoredNVFP4Linear)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert dict(model._latentslate_ltx23_gemma_quant_modules) == {
        "model.language_model.fp8": "float8_e4m3fn",
        "model.language_model.nvfp4": "nvfp4",
    }


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

    def outer_prompt_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror Gemma's outer-shell lookup before ``language_model.forward``."""

        embedding = self.model.language_model.embed_tokens(input_ids)
        return embedding, self.model.language_model(input_ids)


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
    assert evidence == {
        "mode": "layer_streamed_cpu_master",
        "root_activation": "stage_onload",
        "layer_count": 2,
        "root_weight_bytes": evidence["root_weight_bytes"],
        "largest_layer_weight_bytes": evidence["largest_layer_weight_bytes"],
        "required_weight_bytes": evidence["required_weight_bytes"],
        "root_transitions": 1,
        "layer_transitions": 4,
    }
    assert evidence["required_weight_bytes"] == (
        evidence["root_weight_bytes"] + evidence["largest_layer_weight_bytes"]
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
