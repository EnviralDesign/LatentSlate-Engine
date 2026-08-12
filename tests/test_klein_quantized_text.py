from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch
from torch import nn

from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.runtime import klein_quantized_text as mixed
from latentslate_engine.runtime.klein_stored_adapter import (
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
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
        return self.tensors[key]


def _full_contract_handle() -> _Handle:
    entries: dict[str, tuple[str, tuple[int, ...]]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for index in range(141):
        stem = f"model.fp8_{index}"
        entries[stem + ".weight"] = ("F8_E4M3", (16, 16))
        entries[stem + ".weight_scale"] = ("F32", ())
        entries[stem + ".comfy_quant"] = ("U8", (27,))
        tensors[stem + ".comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8
        )
    for index in range(85):
        stem = f"model.nvfp4_{index}"
        entries[stem + ".weight"] = ("U8", (16, 8))
        entries[stem + ".weight_scale"] = ("F8_E4M3", (16, 1))
        entries[stem + ".weight_scale_2"] = ("F32", ())
        entries[stem + ".comfy_quant"] = ("U8", (19,))
        tensors[stem + ".comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8
        )
    for index in range(172):
        entries[f"model.dense_{index}.weight"] = ("BF16", (16, 16))
    return _Handle(entries, tensors)


def test_mixed_qwen_plan_proves_exact_full_contract(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "qwen_3_8b_fp8mixed.safetensors"
    path.write_bytes(b"header fixture")
    identity = ArtifactIdentity(path.resolve(), mixed._SIZE_BYTES, 1, "header")
    probe = SimpleNamespace(
        format="safetensors",
        identity=identity,
        schema_sha256=mixed.KLEIN9_QWEN_MIXED_SCHEMA_SHA256,
        tensor_count=mixed._TENSOR_COUNT,
        tensor_dtypes=mixed._TENSOR_DTYPES,
    )
    handle = _full_contract_handle()
    monkeypatch.setattr(mixed, "probe_artifact", lambda _path: probe)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: handle)

    plan = mixed.plan_klein_mixed_text_encoder(path)

    assert plan.identity == identity
    assert len(plan.quantized_formats) == 226
    assert sum(value == "float8_e4m3fn" for value in plan.quantized_formats.values()) == 141
    assert sum(value == "nvfp4" for value in plan.quantized_formats.values()) == 85
    assert len(plan.dense_sources) == 172
    assert len(plan.auxiliary_sources) == 537


class _TinyQwen(nn.Module):
    def __init__(self, _config=None) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 16)
        self.model.fp8 = nn.Linear(16, 16, bias=False)
        self.model.nvfp4 = nn.Linear(16, 16, bias=False)
        self.model.norm = nn.LayerNorm(16, elementwise_affine=True, bias=False)
        self.lm_head = nn.Linear(16, 16, bias=False)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight


def test_mixed_qwen_materializer_restores_exact_native_modules(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "qwen.safetensors"
    path.write_bytes(b"fixture")
    identity = ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, "h")
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 16), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones((16,), dtype=torch.bfloat16),
        "model.fp8.weight": torch.zeros((16, 16), dtype=torch.float8_e4m3fn),
        "model.fp8.weight_scale": torch.tensor(0.25, dtype=torch.float32),
        "model.nvfp4.weight": torch.zeros((16, 8), dtype=torch.uint8),
        "model.nvfp4.weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        "model.nvfp4.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
    }
    entries = {
        key: (
            "BF16" if value.dtype is torch.bfloat16 else "F32",
            tuple(value.shape),
        )
        for key, value in tensors.items()
    }
    handle = _Handle(entries, tensors)
    plan = mixed.KleinMixedTextEncoderPlan(
        identity,
        mixed.KLEIN9_QWEN_MIXED_SCHEMA_SHA256,
        MappingProxyType(
            {"model.fp8": "float8_e4m3fn", "model.nvfp4": "nvfp4"}
        ),
        ("model.embed_tokens.weight", "model.norm.weight"),
        (
            "model.fp8.weight_scale",
            "model.nvfp4.weight_scale",
            "model.nvfp4.weight_scale_2",
        ),
    )
    monkeypatch.setattr(mixed, "revalidate_klein_mixed_text_encoder", lambda _plan: True)
    monkeypatch.setattr(mixed, "revalidate_artifact", lambda _identity: True)
    monkeypatch.setattr("safetensors.safe_open", lambda *_args, **_kwargs: handle)
    monkeypatch.setattr("accelerate.init_empty_weights", lambda: nullcontext())
    fake_config = SimpleNamespace(from_pretrained=lambda *_a, **_k: object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Qwen3Config=fake_config, Qwen3ForCausalLM=_TinyQwen),
    )

    model = mixed.load_klein_mixed_text_encoder(plan, tmp_path)

    assert isinstance(model.model.fp8, KleinStoredLinear)
    assert isinstance(model.model.nvfp4, KleinStoredNVFP4Linear)
    assert model.model.fp8.input_scale is None
    assert model.model.nvfp4.input_scale is None
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert dict(model._latentslate_klein_mixed_quant_modules) == {
        "model.fp8": "float8_e4m3fn",
        "model.nvfp4": "nvfp4",
    }
    assert not any(value.is_meta for value in model.state_dict().values())


def test_mixed_qwen_stage_requires_every_quantized_module_to_dispatch(monkeypatch):
    model = _TinyQwen()
    model._latentslate_klein_mixed_quant_modules = MappingProxyType(
        {"model.fp8": "float8_e4m3fn", "model.nvfp4": "nvfp4"}
    )
    model.model.fp8.native_dispatch_count = 0
    model.model.nvfp4.native_dispatch_count = 0
    monkeypatch.setattr(mixed, "move_klein_module_storage", lambda *_args: None)
    stage = mixed.KleinMixedTextEncoderStage(model, "cpu")

    stage.onload()
    model.model.fp8.native_dispatch_count += 1
    model.model.nvfp4.native_dispatch_count += 2
    proof = stage.verify_dispatch()
    stage.offload()

    assert proof == {
        "backend": "comfy-kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": 2,
        "total_dispatches": 3,
        "minimum_module_dispatches": 1,
        "maximum_module_dispatches": 2,
    }
