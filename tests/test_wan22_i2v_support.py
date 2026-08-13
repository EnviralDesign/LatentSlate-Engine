from __future__ import annotations

import json

import pytest

import latentslate_engine.runtime.wan22_i2v_support as support_module
from latentslate_engine.runtime.umt5_stored_adapter import UMT5_XXL_CONFIG
from latentslate_engine.runtime.wan21_vae_adapter import WAN21_VAE_CONFIG
from latentslate_engine.runtime.wan22_i2v_support import (
    plan_wan_i2v_support,
    revalidate_wan_i2v_support,
)
from latentslate_engine.runtime.wan22_stored_adapter import WAN22_14B_I2V_CONFIG


def _write_support(root):
    documents = {
        "model_index.json": {
            "_class_name": "WanImageToVideoPipeline",
            "boundary_ratio": 0.9,
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "transformer_2": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
        },
        "scheduler/scheduler_config.json": {
            "_class_name": "UniPCMultistepScheduler",
            "_diffusers_version": "0.35.0.dev0",
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "disable_corrector": [],
            "dynamic_thresholding_ratio": 0.995,
            "final_sigmas_type": "zero",
            "flow_shift": 3.0,
            "lower_order_final": True,
            "num_train_timesteps": 1000,
            "predict_x0": True,
            "prediction_type": "flow_prediction",
            "rescale_betas_zero_snr": False,
            "sample_max_value": 1.0,
            "use_flow_sigmas": True,
            "solver_order": 2,
            "solver_p": None,
            "solver_type": "bh2",
            "steps_offset": 0,
            "thresholding": False,
            "time_shift_type": "exponential",
            "timestep_spacing": "linspace",
            "trained_betas": None,
            "use_beta_sigmas": False,
            "use_dynamic_shifting": False,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        },
        "transformer/config.json": {"_class_name": "WanTransformer3DModel", **dict(WAN22_14B_I2V_CONFIG)},
        "transformer_2/config.json": {"_class_name": "WanTransformer3DModel", **dict(WAN22_14B_I2V_CONFIG)},
        "text_encoder/config.json": {
            "architectures": ["UMT5EncoderModel"],
            **{
                key: UMT5_XXL_CONFIG[key]
                for key in (
                    "vocab_size",
                    "d_model",
                    "d_kv",
                    "d_ff",
                    "num_layers",
                    "num_heads",
                    "feed_forward_proj",
                    "relative_attention_num_buckets",
                    "relative_attention_max_distance",
                    "layer_norm_epsilon",
                )
            },
        },
        "vae/config.json": {
            "_class_name": "AutoencoderKLWan",
            **{
                key: WAN21_VAE_CONFIG[key]
                for key in (
                    "base_dim",
                    "dim_mult",
                    "num_res_blocks",
                    "temperal_downsample",
                    "z_dim",
                    "latents_mean",
                    "latents_std",
                )
            },
        },
    }
    for relative_path, document in documents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
    tokenizer = root / "tokenizer" / "spiece.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_bytes(b"synthetic sentencepiece")


def test_support_plan_validates_and_revalidates_exact_files(tmp_path, monkeypatch):
    _write_support(tmp_path)
    monkeypatch.setattr(
        support_module.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "a" * 64})(),
    )
    plan = plan_wan_i2v_support(tmp_path)
    assert len(plan.files) == 7
    assert plan.boundary_ratio == 0.9
    assert plan.tokenizer_sha256 == "a" * 64
    assert revalidate_wan_i2v_support(plan)
    with pytest.raises(TypeError):
        plan.scheduler_config["flow_shift"] = 9.0
    scheduler = plan.load_scheduler()
    scheduler.set_timesteps(20)
    assert len(scheduler.timesteps) == 20

    (tmp_path / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    assert not revalidate_wan_i2v_support(plan)


def test_support_plan_rejects_config_mismatch_and_duplicate_json(tmp_path, monkeypatch):
    _write_support(tmp_path)
    monkeypatch.setattr(
        support_module.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "b" * 64})(),
    )
    path = tmp_path / "scheduler" / "scheduler_config.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["flow_shift"] = 9.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler config"):
        plan_wan_i2v_support(tmp_path)

    _write_support(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["solver_type"] = "bh1"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler config"):
        plan_wan_i2v_support(tmp_path)

    _write_support(tmp_path)
    path.write_text('{"flow_shift":3,"flow_shift":3}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        plan_wan_i2v_support(tmp_path)


def test_support_plan_rejects_escape_and_oversized_file(tmp_path, monkeypatch):
    _write_support(tmp_path)
    monkeypatch.setattr(
        support_module.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "c" * 64})(),
    )
    tokenizer = tmp_path / "tokenizer" / "spiece.model"
    tokenizer.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="safety limit"):
        plan_wan_i2v_support(tmp_path)


def test_support_plan_rejects_path_replacement_during_bound_read(tmp_path, monkeypatch):
    _write_support(tmp_path)
    monkeypatch.setattr(
        support_module.WanSentencePieceTokenizer,
        "from_bytes",
        lambda payload: type("Tokenizer", (), {"model_sha256": "d" * 64})(),
    )
    states = iter(((1, 1), (2, 1)))
    monkeypatch.setattr(support_module, "_file_state", lambda value: next(states))
    with pytest.raises(ValueError, match="changed while reading"):
        plan_wan_i2v_support(tmp_path)


def test_support_plan_rechecks_containment_after_open(tmp_path, monkeypatch):
    _write_support(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_bytes((tmp_path / "model_index.json").read_bytes())
    monkeypatch.setattr(support_module, "_opened_file_path", lambda handle: outside)
    with pytest.raises(ValueError, match="escapes its root after open"):
        plan_wan_i2v_support(tmp_path)
