from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from latentslate_engine.runtime.ltx23_kitchen_contracts import (
    LTX23_DEV_FP8,
    LTX23_DISTILLED_FP8,
    LTX23_GEMMA_MIXED,
    LTX23_MODEL_LORA,
    LTX23_SPATIAL_UPSCALER,
    LTX23_TEXT_LORA,
    LTX23ArtifactContract,
    plan_ltx23_stored_artifact,
)


def _contract(name: str, path: Path, *, quantized: dict[str, int], components: dict[str, int]) -> LTX23ArtifactContract:
    from latentslate_engine.artifacts import probe_safetensors

    probe = probe_safetensors(path)
    return LTX23ArtifactContract(
        name=name,
        filename=path.name,
        size_bytes=probe.identity.size_bytes,
        source_sha256="synthetic",
        header_sha256=probe.identity.header_sha256,
        schema_sha256=probe.schema_sha256,
        tensor_count=probe.tensor_count,
        dtypes=probe.tensor_dtypes,
        quantization_contract="synthetic",
        component_counts=components,
        config={},
        expected_quantized_counts=quantized,
    )


def _checkpoint(path: Path, *, omit_input_scale: bool = False) -> None:
    stem = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
    tensors = {
        stem + ".weight": torch.zeros((16, 16), dtype=torch.float8_e4m3fn),
        stem + ".weight_scale": torch.ones((), dtype=torch.float32),
        "vae.decoder.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "audio_vae.decoder.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "vocoder.vocoder.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "text_embedding_projection.video.weight": torch.zeros((1,), dtype=torch.bfloat16),
    }
    if not omit_input_scale:
        tensors[stem + ".input_scale"] = torch.ones((), dtype=torch.float32)
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": {stem: {"format": "float8_e4m3fn"}}})}
    save_file(tensors, path, metadata=metadata)


def test_checkpoint_plan_derives_its_own_quantized_set_and_revalidates(tmp_path: Path):
    path = tmp_path / "checkpoint.safetensors"
    _checkpoint(path)
    contract = _contract(
        "checkpoint",
        path,
        quantized={"fp8": 1},
        components={"transformer": 3, "vae": 1, "audio_vae": 1, "vocoder": 1, "text_projection": 1},
    )

    plan = plan_ltx23_stored_artifact(path, contract)

    assert plan.available, plan.errors
    assert plan.quantized_layers == {"fp8": ("model.diffusion_model.transformer_blocks.0.attn1.to_q",)}
    assert plan.roles["model.diffusion_model.transformer_blocks.0.attn1.to_q.input_scale"] == "transformer/fp8_input_scale"
    assert plan.revalidate()


def test_checkpoint_plan_rejects_missing_quantization_companion(tmp_path: Path):
    path = tmp_path / "checkpoint.safetensors"
    _checkpoint(path, omit_input_scale=True)
    contract = _contract(
        "checkpoint",
        path,
        quantized={"fp8": 1},
        components={"transformer": 2, "vae": 1, "audio_vae": 1, "vocoder": 1, "text_projection": 1},
    )

    plan = plan_ltx23_stored_artifact(path, contract)

    assert not plan.available
    assert any("incomplete FP8" in error for error in plan.errors)


def test_text_lora_rejects_non_rank_64_pair(tmp_path: Path):
    path = tmp_path / "text-lora.safetensors"
    save_file(
        {
            "text_encoders.transformer.model.layers.0.self_attn.q_proj.lora_down.weight": torch.zeros((8, 16), dtype=torch.bfloat16),
            "text_encoders.transformer.model.layers.0.self_attn.q_proj.lora_up.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        },
        path,
    )
    # Use the actual fixed artifact class to exercise its specific parser; its
    # identity mismatch is expected, while the rank error proves the fail-closed
    # format rule is independent of pinned identity facts.
    plan = plan_ltx23_stored_artifact(path, LTX23_TEXT_LORA)

    assert not plan.available
    assert any("rank 64" in error for error in plan.errors)


def test_gemma_plan_rejects_incomplete_packed_nvfp4_group(tmp_path: Path):
    path = tmp_path / "gemma-mixed.safetensors"
    stem = "model.layers.0.self_attn.q_proj"
    save_file(
        {
            stem + ".weight": torch.zeros((16, 8), dtype=torch.uint8),
            stem + ".weight_scale": torch.zeros((16, 1), dtype=torch.float8_e4m3fn),
            stem + ".comfy_quant": torch.zeros((19,), dtype=torch.uint8),
            "model.embed_tokens.weight": torch.zeros((1,), dtype=torch.bfloat16),
        },
        path,
    )

    plan = plan_ltx23_stored_artifact(path, LTX23_GEMMA_MIXED)

    assert not plan.available
    assert any("incomplete nvfp4" in error for error in plan.errors)


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_LTX23_REAL_HEADERS") != "1",
    reason="set LATENTSLATE_LTX23_REAL_HEADERS=1 to inspect installed LTX headers",
)
@pytest.mark.parametrize(
    ("relative", "contract", "expected_quantized"),
    [
        ("models/ltx23/checkpoints/ltx-2.3-22b-dev-fp8.safetensors", LTX23_DEV_FP8, {"fp8": 1496}),
        ("models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors", LTX23_DISTILLED_FP8, {"fp8": 1462}),
        ("models/ltx23/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", LTX23_GEMMA_MIXED, {"fp8": 34, "nvfp4": 302}),
        ("loras/ltx23/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors", LTX23_MODEL_LORA, {}),
        ("loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors", LTX23_TEXT_LORA, {}),
        ("models/ltx23/latent_upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors", LTX23_SPATIAL_UPSCALER, {}),
    ],
)
def test_installed_exact_ltx23_headers(relative: str, contract: LTX23ArtifactContract, expected_quantized: dict[str, int]):
    path = Path(r"M:\LatentSlateEngineData") / relative
    if not path.is_file():
        pytest.skip(f"not installed: {path}")

    plan = plan_ltx23_stored_artifact(path, contract)

    assert plan.available, plan.errors
    assert {kind: len(layers) for kind, layers in plan.quantized_layers.items()} == expected_quantized
    assert plan.revalidate()
