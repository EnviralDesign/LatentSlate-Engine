from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from latentslate_engine import z_image_turbo_recipe as contract
from latentslate_engine.artifacts import probe_artifact
from latentslate_engine.runtime import z_image_mixed_qwen as mixed_qwen
from latentslate_engine.runtime.z_image_turbo import (
    ZImagePhase,
    ZImageTurboCancelled,
    ZImageTurboLifecycle,
)
from latentslate_engine.tools.z_image_turbo import ZImageTurboTextToImageTool
from latentslate_engine.z_image_turbo_recipe import (
    ZImageDensePlan,
    ZImageTurboRecipe,
    ZImageTurboRuntimeRequest,
    _plan_transformer,
    z_image_turbo_schedule,
)


def _marker() -> torch.Tensor:
    return torch.tensor(
        list(
            json.dumps(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 4,
                    "per_row": True,
                }
            ).encode()
        ),
        dtype=torch.uint8,
    )


def _save_transformer(path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for index in range(202):
        stem = f"layers.{index}.attention.qkv"
        tensors[stem + ".weight"] = torch.tensor([[2, -2, 0, 0]], dtype=torch.int8)
        tensors[stem + ".weight_scale"] = torch.tensor([[0.25]])
        tensors[stem + ".comfy_quant"] = _marker()
    # The upstream header uses empty global metadata; every ConvRot fact comes
    # from the exact per-layer U8 marker payload.
    save_file(tensors, path, metadata={})


def test_official_catalog_is_exactly_one_turbo_t2i_contract():
    root = Path(__file__).parents[1] / "src/latentslate_engine"
    recipe = tomllib.loads(
        (
            root / "builtin_recipes/zimage/z-image-turbo-text-to-image-comfy-int8-convrot.toml"
        ).read_text()
    )["runnable_recipe"]
    assert recipe["family"] == "zimage"
    assert recipe["recipe"]["type"] == "z_image_turbo_t2i"
    assert recipe["recipe"]["operation"] == "comfy_turbo_t2i_int8_convrot"
    assert "fixed" not in recipe
    assert [item.key for item in ZImageTurboTextToImageTool().descriptor.inputs] == [
        "prompt",
        "seed",
    ]
    declarations = list((root / "builtin_resource_declarations").glob("zimage-*.toml"))
    assert len(declarations) == 3
    sources = [tomllib.loads(path.read_text())["resource"]["sources"][0] for path in declarations]
    assert {source["repo_id"] for source in sources} == {"Comfy-Org/z_image_turbo"}
    assert all(len(source["revision"]) == 40 and len(source["sha256"]) == 64 for source in sources)
    # Bounded real-header facts captured from the immutable revisions, not a
    # mutable filename heuristic. The synthetic files below preserve exactly
    # these counts/marker lengths while staying small enough for CI.
    assert (
        contract._Z_TRANSFORMER_HEADER_SHA256
        == "01e93cae3aa75eb2106025889f1a78df19628a95c433b45d9447562b04907814"
    )
    assert (
        contract._Z_QWEN_HEADER_SHA256
        == "7537b0cd31f4fc963d334b4f997cedee6f51c62aa8518b7b7a852b182144aed9"
    )


def test_stored_convrot_plan_materializes_without_dense_fallback(tmp_path: Path, monkeypatch):
    path = tmp_path / "z-int8.safetensors"
    _save_transformer(path)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    plan = _plan_transformer(path)
    assert plan.stored_layer_count == 202
    plan.require_stored_layout()
    restored = next(iter(plan.stored_layers.values())).materialize(torch.float32)
    assert restored.storage_dtype == torch.int8


def test_lifecycle_rejects_cancel_and_never_claims_warm_cache(tmp_path: Path, monkeypatch):
    transformer_path = tmp_path / "z-int8.safetensors"
    _save_transformer(transformer_path)
    raw, _ = contract._read_z_safetensors_header(transformer_path, transformer_path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        "latentslate_engine.runtime.z_image_mixed_qwen.revalidate_z_image_mixed_qwen",
        lambda _plan: True,
    )
    transformer = _plan_transformer(transformer_path)
    dense_identity = probe_artifact(transformer_path).identity
    dense = ZImageDensePlan(dense_identity, "synthetic", "text_encoder", 1)
    request = ZImageTurboRuntimeRequest(
        1,
        "comfy-org-z-image-turbo-int8-convrot",
        "comfy_turbo_t2i_int8_convrot",
        {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guidance_scale": 1.0,
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "negative_conditioning": "zero_out_positive",
        },
        {
            role: {
                "path": str(transformer_path.resolve()),
                "header_sha256": dense_identity.header_sha256,
                "source_revision": contract._IMMUTABLE_COMPONENTS[role][1],
                "source_filename": contract._IMMUTABLE_COMPONENTS[role][2],
                "source_sha256": contract._IMMUTABLE_COMPONENTS[role][3],
            }
            for role in ("transformer", "text_encoder", "vae")
        },
        {role: dense_identity for role in ("transformer", "text_encoder", "vae")},
        {
            "transformer": transformer,
            "text_encoder": dense,
            "vae": ZImageDensePlan(dense_identity, "synthetic", "vae", 1),
        },
    )
    lifecycle = ZImageTurboLifecycle(request)
    with pytest.raises(ZImageTurboCancelled):
        lifecycle.checkpoint(ZImagePhase.TEXT_ENCODER, lambda: True)
    assert lifecycle.ejected
    provenance = lifecycle.public_provenance()
    assert provenance["execution_cache"]["supported"] is False
    assert provenance["request_fingerprint"] == request.fingerprint
    assert provenance["components"] == request.public_component_manifest()
    assert provenance["native_transformer_dispatch"] == {
        "proven": False,
        "count": 0,
        "reason": "GPU execution has not been accepted",
    }


def test_lifecycle_requires_text_transformer_vae_order(tmp_path: Path, monkeypatch):
    transformer_path = tmp_path / "z-int8.safetensors"
    _save_transformer(transformer_path)
    raw, _ = contract._read_z_safetensors_header(transformer_path, transformer_path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        "latentslate_engine.runtime.z_image_mixed_qwen.revalidate_z_image_mixed_qwen",
        lambda _plan: True,
    )
    identity = probe_artifact(transformer_path).identity
    transformer = _plan_transformer(transformer_path)
    dense = ZImageDensePlan(identity, "synthetic", "text_encoder", 1)
    request = ZImageTurboRuntimeRequest(
        1,
        "comfy-org-z-image-turbo-int8-convrot",
        "comfy_turbo_t2i_int8_convrot",
        {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guidance_scale": 1.0,
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "negative_conditioning": "zero_out_positive",
        },
        {
            role: {
                "path": str(transformer_path.resolve()),
                "header_sha256": identity.header_sha256,
                "source_revision": contract._IMMUTABLE_COMPONENTS[role][1],
                "source_filename": contract._IMMUTABLE_COMPONENTS[role][2],
                "source_sha256": contract._IMMUTABLE_COMPONENTS[role][3],
            }
            for role in ("transformer", "text_encoder", "vae")
        },
        {role: identity for role in ("transformer", "text_encoder", "vae")},
        {
            "transformer": transformer,
            "text_encoder": dense,
            "vae": ZImageDensePlan(identity, "synthetic", "vae", 1),
        },
    )
    lifecycle = ZImageTurboLifecycle(request)
    with pytest.raises(ValueError, match="invalid after"):
        lifecycle.checkpoint(ZImagePhase.VAE, lambda: False)
    assert lifecycle.ejected


def test_schedule_rejects_any_base_or_edit_like_deviation():
    recipe = ZImageTurboRecipe.__new__(ZImageTurboRecipe)
    object.__setattr__(recipe, "width", 1024)
    object.__setattr__(recipe, "height", 1024)
    object.__setattr__(recipe, "steps", 9)
    object.__setattr__(recipe, "guidance_scale", 1.0)
    object.__setattr__(recipe, "sampling", "auraflow_shift_3")
    object.__setattr__(recipe, "sampler", "res_multistep")
    object.__setattr__(recipe, "scheduler", "simple")
    with pytest.raises(ValueError, match="exact Comfy schedule"):
        z_image_turbo_schedule(recipe)


def _save_qwen(path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    fp8_marker = torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)
    nvfp4_marker = torch.tensor(list(b'{"format": "nvfp4"}'), dtype=torch.uint8)
    for index in range(209):
        tensors[f"dense_{index}.weight"] = torch.zeros((8, 8), dtype=torch.bfloat16)
    for index in range(177):
        stem = f"fp8_{index}"
        tensors[stem + ".weight"] = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
        tensors[stem + ".weight_scale"] = torch.tensor(0.25, dtype=torch.float32)
        tensors[stem + ".comfy_quant"] = fp8_marker.clone()
    for index in range(12):
        stem = f"nvfp4_{index}"
        tensors[stem + ".weight"] = torch.zeros((8, 8), dtype=torch.uint8)
        tensors[stem + ".weight_scale"] = torch.ones((8, 1), dtype=torch.float8_e4m3fn)
        tensors[stem + ".weight_scale_2"] = torch.tensor(0.25, dtype=torch.float32)
        tensors[stem + ".comfy_quant"] = nvfp4_marker.clone()
    save_file(tensors, path)


def test_real_header_derived_mixed_qwen_closure_requires_209_177_12(tmp_path: Path, monkeypatch):
    path = tmp_path / "qwen_3_4b_fp8_mixed.safetensors"
    _save_qwen(path)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(mixed_qwen, "_Z_QWEN_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    plan = mixed_qwen.plan_z_image_mixed_qwen(path)
    assert (len(plan.dense_sources), len(plan.fp8_sources), len(plan.nvfp4_sources)) == (
        209,
        177,
        12,
    )
    assert len(plan.auxiliary_sources) == 390


def test_mixed_qwen_dispatch_requires_all_189_native_low_bit_modules():
    model = nn.Module()
    model.quantized = nn.ModuleList([nn.Identity() for _ in range(189)])
    names = {f"quantized.{index}": "fp8" if index < 177 else "nvfp4" for index in range(189)}
    for module in model.quantized:
        module.native_dispatch_count = 0
    model._latentslate_z_image_quant_modules = MappingProxyType(names)
    before = mixed_qwen.z_image_mixed_dispatch_snapshot(model)
    for module in model.quantized:
        module.native_dispatch_count += 1
    proof = mixed_qwen.verify_z_image_mixed_dispatch(model, before)
    assert proof == {
        "backend": "comfy-kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": 189,
        "total_dispatches": 189,
        "fp8_modules": 177,
        "nvfp4_modules": 12,
    }


def test_mixed_qwen_materializer_preserves_fp8_nvfp4_wrappers(tmp_path: Path, monkeypatch):
    path = tmp_path / "tiny-qwen.safetensors"
    save_file(
        {
            "model.dense.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
            "model.fp8.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            "model.fp8.weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "model.nvfp4.weight": torch.zeros((8, 8), dtype=torch.uint8),
            "model.nvfp4.weight_scale": torch.ones((8, 1), dtype=torch.float8_e4m3fn),
            "model.nvfp4.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
        },
        path,
    )
    identity = probe_artifact(path).identity

    class TinyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.dense = nn.Linear(8, 8, bias=False)
            self.model.fp8 = nn.Linear(8, 8, bias=False)
            self.model.nvfp4 = nn.Linear(16, 8, bias=False)

    plan = mixed_qwen.ZImageMixedQwenPlan(
        identity,
        "synthetic",
        "synthetic",
        MappingProxyType(
            {
                "model.dense.weight": "model.dense.weight",
                "model.fp8.weight": "model.fp8.weight",
                "model.nvfp4.weight": "model.nvfp4.weight",
            }
        ),
        ("model.fp8.weight",),
        ("model.nvfp4.weight",),
        ("model.dense.weight",),
        ("model.fp8.weight_scale", "model.nvfp4.weight_scale", "model.nvfp4.weight_scale_2"),
        "synthetic",
    )
    monkeypatch.setattr(mixed_qwen, "revalidate_z_image_mixed_qwen", lambda _plan: True)
    model = mixed_qwen.materialize_z_image_mixed_qwen(plan, TinyQwen())
    from latentslate_engine.runtime.klein_stored_adapter import (
        KleinStoredLinear,
        KleinStoredNVFP4Linear,
    )

    assert isinstance(model.model.fp8, KleinStoredLinear)
    assert isinstance(model.model.nvfp4, KleinStoredNVFP4Linear)


def test_mixed_qwen_materializer_allows_only_the_standard_tied_lm_head(tmp_path: Path, monkeypatch):
    path = tmp_path / "tiny-tied-qwen.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
            "model.fp8.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            "model.fp8.weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "model.nvfp4.weight": torch.zeros((8, 8), dtype=torch.uint8),
            "model.nvfp4.weight_scale": torch.ones((8, 1), dtype=torch.float8_e4m3fn),
            "model.nvfp4.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
        },
        path,
    )
    identity = probe_artifact(path).identity

    class TinyTiedQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(8, 8)
            self.model.fp8 = nn.Linear(8, 8, bias=False)
            self.model.nvfp4 = nn.Linear(16, 8, bias=False)
            self.lm_head = nn.Linear(8, 8, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    plan = mixed_qwen.ZImageMixedQwenPlan(
        identity,
        "synthetic",
        "synthetic",
        MappingProxyType(
            {
                "model.embed_tokens.weight": "model.embed_tokens.weight",
                "model.fp8.weight": "model.fp8.weight",
                "model.nvfp4.weight": "model.nvfp4.weight",
            }
        ),
        ("model.fp8.weight",),
        ("model.nvfp4.weight",),
        ("model.embed_tokens.weight",),
        ("model.fp8.weight_scale", "model.nvfp4.weight_scale", "model.nvfp4.weight_scale_2"),
        "synthetic",
    )
    monkeypatch.setattr(mixed_qwen, "revalidate_z_image_mixed_qwen", lambda _plan: True)
    model = mixed_qwen.materialize_z_image_mixed_qwen(plan, TinyTiedQwen())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not model.lm_head.weight.is_meta


def test_mixed_qwen_materializer_accepts_actual_tiny_qwen3_tied_shell(tmp_path: Path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    shell = Qwen3ForCausalLM(config)
    fp8_source = "model.layers.0.self_attn.q_proj.weight"
    nvfp4_source = "model.layers.0.self_attn.k_proj.weight"
    tensors = {
        key: torch.zeros_like(value, dtype=torch.bfloat16)
        for key, value in shell.state_dict().items()
        if key not in {"lm_head.weight", fp8_source, nvfp4_source}
    }
    tensors.update(
        {
            fp8_source: torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            fp8_source.removesuffix(".weight") + ".weight_scale": torch.tensor(
                0.25, dtype=torch.float32
            ),
            nvfp4_source: torch.zeros((8, 4), dtype=torch.uint8),
            nvfp4_source.removesuffix(".weight") + ".weight_scale": torch.ones(
                (8, 1), dtype=torch.float8_e4m3fn
            ),
            nvfp4_source.removesuffix(".weight") + ".weight_scale_2": torch.tensor(
                0.25, dtype=torch.float32
            ),
        }
    )
    path = tmp_path / "qwen3-tied-shell.safetensors"
    save_file(tensors, path)
    identity = probe_artifact(path).identity
    sources = tuple(sorted(key for key in tensors if key.endswith(".weight")))
    plan = mixed_qwen.ZImageMixedQwenPlan(
        identity,
        "synthetic",
        "synthetic",
        MappingProxyType({source: source for source in sources}),
        (fp8_source,),
        (nvfp4_source,),
        tuple(source for source in sources if source not in {fp8_source, nvfp4_source}),
        (
            fp8_source.removesuffix(".weight") + ".weight_scale",
            nvfp4_source.removesuffix(".weight") + ".weight_scale",
            nvfp4_source.removesuffix(".weight") + ".weight_scale_2",
        ),
        "synthetic",
    )
    monkeypatch.setattr(mixed_qwen, "revalidate_z_image_mixed_qwen", lambda _plan: True)
    model = mixed_qwen.materialize_z_image_mixed_qwen(plan, shell)
    assert model.lm_head.weight is model.model.embed_tokens.weight
