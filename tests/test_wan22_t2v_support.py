from __future__ import annotations

import json

import pytest

import latentslate_engine.runtime.wan22_i2v_support as shared_support
from latentslate_engine.runtime.umt5_stored_adapter import UMT5_XXL_CONFIG
from latentslate_engine.runtime.wan21_vae_adapter import WAN21_VAE_CONFIG
from latentslate_engine.runtime.wan22_i2v_support import plan_wan_i2v_support
from latentslate_engine.runtime.wan22_stored_adapter import WAN22_14B_T2V_CONFIG
from latentslate_engine.runtime.wan22_t2v_support import (
    plan_wan_t2v_support,
    revalidate_wan_t2v_support,
)


def _write_t2v_support(root):
    documents = {
        "model_index.json": {
            "_class_name": "WanPipeline",
            "boundary_ratio": 0.875,
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "transformer_2": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
        },
        "scheduler/scheduler_config.json": dict(shared_support._PINNED_SCHEDULER_CONFIG),
        "transformer/config.json": {"_class_name": "WanTransformer3DModel", **WAN22_14B_T2V_CONFIG},
        "transformer_2/config.json": {"_class_name": "WanTransformer3DModel", **WAN22_14B_T2V_CONFIG},
        "text_encoder/config.json": {
            "architectures": ["UMT5EncoderModel"],
            **{key: UMT5_XXL_CONFIG[key] for key in ("vocab_size", "d_model", "d_kv", "d_ff", "num_layers", "num_heads", "feed_forward_proj", "relative_attention_num_buckets", "relative_attention_max_distance", "layer_norm_epsilon")},
        },
        "vae/config.json": {
            "_class_name": "AutoencoderKLWan",
            **{key: WAN21_VAE_CONFIG[key] for key in ("base_dim", "dim_mult", "num_res_blocks", "temperal_downsample", "z_dim", "latents_mean", "latents_std")},
        },
    }
    for relative_path, document in documents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
    tokenizer = root / "tokenizer" / "spiece.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_bytes(b"synthetic sentencepiece")


def test_t2v_support_requires_wanpipeline_boundary_and_16_channel_configs(tmp_path, monkeypatch):
    _write_t2v_support(tmp_path)
    monkeypatch.setattr(
        shared_support.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "c" * 64})(),
    )
    plan = plan_wan_t2v_support(tmp_path)
    assert plan.boundary_ratio == 0.875
    assert len(plan.files) == 7

    document = json.loads((tmp_path / "transformer" / "config.json").read_text(encoding="utf-8"))
    document["in_channels"] = 36
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="transformer config mismatch"):
        plan_wan_t2v_support(tmp_path)


def test_t2v_revalidation_rejects_an_i2v_support_plan(tmp_path, monkeypatch):
    _write_t2v_support(tmp_path)
    monkeypatch.setattr(
        shared_support.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "d" * 64})(),
    )
    t2v = plan_wan_t2v_support(tmp_path)
    assert revalidate_wan_t2v_support(t2v)
    document = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    document["_class_name"] = "WanImageToVideoPipeline"
    document["boundary_ratio"] = 0.9
    (tmp_path / "model_index.json").write_text(json.dumps(document), encoding="utf-8")
    document = json.loads((tmp_path / "transformer" / "config.json").read_text(encoding="utf-8"))
    document["in_channels"] = 36
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "transformer_2" / "config.json").write_text(json.dumps(document), encoding="utf-8")
    i2v = plan_wan_i2v_support(tmp_path)
    assert not revalidate_wan_t2v_support(i2v)
