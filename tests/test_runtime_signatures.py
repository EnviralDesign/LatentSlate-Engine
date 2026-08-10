from __future__ import annotations

import os
from pathlib import Path

from latentslate_engine.config import Settings
from latentslate_engine.runtime.kit import RuntimeDefaults, resolve_runtime_plan
from latentslate_engine.runtime.klein import resolve_klein_runtime_plan


def _rewrite_same_metadata(path: Path, content: bytes) -> None:
    stat = path.stat()
    assert len(content) == stat.st_size
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_nested_same_size_same_mtime_weight_change_invalidates_pipeline(tmp_path: Path):
    model = tmp_path / "model"
    nested = model / "transformer"
    nested.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    weight = nested / "model.safetensors"
    weight.write_bytes(b"A" * 1024)
    defaults = RuntimeDefaults(
        family="klein4b",
        model_id="test/model",
        model_path=model,
        model_format="diffusers",
        device="cuda",
        quantization="bf16",
        attention="native",
        offload="model",
    )

    first = resolve_runtime_plan(None, defaults)
    _rewrite_same_metadata(weight, b"B" * 1024)
    second = resolve_runtime_plan(None, defaults)

    assert first.pipeline_fingerprint != second.pipeline_fingerprint


def _klein9_settings(tmp_path: Path) -> tuple[Settings, Path, Path]:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    settings.ensure_directories()
    pipeline = settings.model_root / "klein9b" / "black-forest-labs--FLUX.2-klein-9B"
    pipeline.mkdir(parents=True)
    (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
    (pipeline / "vae").mkdir()
    (pipeline / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"V" * 256)

    transformer_repo = (
        settings.model_root
        / "klein9b"
        / "black-forest-labs--FLUX.2-klein-9b-nvfp4"
    )
    transformer_repo.mkdir(parents=True)
    transformer = transformer_repo / "flux-2-klein-9b-nvfp4.safetensors"
    transformer.write_bytes(b"T" * 512)

    text_encoder = settings.model_root / "klein9b" / "Qwen--Qwen3-8B-FP8"
    text_encoder.mkdir(parents=True)
    (text_encoder / "config.json").write_text("{}", encoding="utf-8")
    text_weight = text_encoder / "model-00001-of-00001.safetensors"
    text_weight.write_bytes(b"Q" * 512)
    return settings, transformer, text_weight


def test_builtin_klein9_transformer_and_text_encoder_changes_invalidate_plan(tmp_path: Path):
    settings, transformer, text_weight = _klein9_settings(tmp_path)
    first = resolve_klein_runtime_plan(settings, "klein9b", None)

    _rewrite_same_metadata(transformer, b"U" * 512)
    second = resolve_klein_runtime_plan(settings, "klein9b", None)
    assert first.pipeline_fingerprint != second.pipeline_fingerprint

    _rewrite_same_metadata(text_weight, b"R" * 512)
    third = resolve_klein_runtime_plan(settings, "klein9b", None)
    assert second.pipeline_fingerprint != third.pipeline_fingerprint
    assert {component.name for component in third.components} == {
        "model",
        "nvfp4_transformer",
        "text_encoder",
    }
