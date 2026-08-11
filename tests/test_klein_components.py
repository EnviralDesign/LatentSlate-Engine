from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentslate_engine.runtime import klein_components


def _support_tree(tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    contracts = (
        klein_components._DISTILLED_SUPPORT_FILES
        if mode == "distilled"
        else klein_components._BASE_SUPPORT_FILES
    )
    root = tmp_path / mode
    for relative, (size, _digest) in contracts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "model_index.json":
            payload = {"_class_name": "Flux2KleinPipeline"}
            if mode == "distilled":
                payload["is_distilled"] = True
            raw = json.dumps(payload).encode()
        elif relative == "text_encoder/config.json":
            raw = json.dumps({"architectures": ["Qwen3ForCausalLM"]}).encode()
        elif relative == "vae/config.json":
            raw = json.dumps({"_class_name": "AutoencoderKLFlux2"}).encode()
        elif relative == "transformer/config.json":
            raw = json.dumps({"_class_name": "Flux2Transformer2DModel"}).encode()
        else:
            raw = b"x"
        path.write_bytes(raw + b" " * (size - len(raw)))

    expected_by_path = {
        str((root / relative).resolve()): digest for relative, (_size, digest) in contracts.items()
    }
    monkeypatch.setattr(
        klein_components,
        "_sha256_file",
        lambda path: expected_by_path[str(path.resolve())],
    )
    return root


@pytest.mark.parametrize("mode", ["base", "distilled"])
def test_klein_support_shell_is_mode_specific_and_weight_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    root = _support_tree(tmp_path, mode, monkeypatch)
    plan = klein_components.plan_klein_pipeline_support(root, mode)

    assert plan.mode == mode
    assert plan.root == root.resolve()
    assert len(plan.files) == 13

    other_mode = "base" if mode == "distilled" else "distilled"
    with pytest.raises(ValueError, match="identity mismatch|mode differs"):
        klein_components.plan_klein_pipeline_support(root, other_mode)


def test_klein_support_shell_rejects_any_unbounded_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _support_tree(tmp_path, "distilled", monkeypatch)
    (root / "text_encoder" / "model.safetensors").write_bytes(b"must not be here")

    with pytest.raises(ValueError, match="exact bounded shell"):
        klein_components.plan_klein_pipeline_support(root, "distilled")


def test_small_decoder_plan_is_exact_and_separate_from_full_vae(monkeypatch, tmp_path):
    captured = {}

    def fake_plan(path, **kwargs):
        captured.update(path=path, **kwargs)
        return "small-plan"

    monkeypatch.setattr(klein_components, "_plan_dense_component", fake_plan)
    path = tmp_path / "full_encoder_small_decoder.safetensors"

    assert klein_components.plan_klein_small_vae(path) == "small-plan"
    assert captured == {
        "path": path,
        "role": "vae",
        "architecture": "flux2_small_decoder_full_encoder",
        "size_bytes": 249_519_092,
        "schema_sha256": klein_components.KLEIN_SMALL_VAE_SCHEMA_SHA256,
        "tensor_count": 251,
        "tensor_dtypes": ("F32", "I64"),
        "contract": "native/fp32",
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("encoder.quant_conv.weight", "quant_conv.weight"),
        (
            "encoder.down.1.block.0.nin_shortcut.weight",
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
        ),
        (
            "decoder.up.3.upsample.conv.bias",
            "decoder.up_blocks.0.upsamplers.0.conv.bias",
        ),
        (
            "decoder.mid.attn_1.proj_out.weight",
            "decoder.mid_block.attentions.0.to_out.0.weight",
        ),
        ("bn.num_batches_tracked", "bn.num_batches_tracked"),
    ],
)
def test_small_decoder_key_mapping_is_bounded(source: str, expected: str):
    assert klein_components._map_small_vae_key(source) == expected


def test_small_decoder_key_mapping_rejects_unknown_keys():
    assert klein_components._map_small_vae_key("unexpected.weight") is None
