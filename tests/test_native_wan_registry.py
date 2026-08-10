from __future__ import annotations

from pathlib import Path

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.resources import discover_resources
from latentslate_engine.tools import default_registry
from latentslate_engine.tools import wan22_native as native_tool_module


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    return settings


def _write_support_tree(path: Path) -> Path:
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/spiece.model",
        "transformer/config.json",
        "transformer_2/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ):
        file = path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"{}" if file.suffix == ".json" else b"sentencepiece")
    return path


def test_hidden_native_base_does_not_probe_runtime_without_a_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        native_tool_module,
        "_native_runtime_availability",
        lambda: (_ for _ in ()).throw(AssertionError("native runtime should not be probed")),
    )

    registry = default_registry(settings, emit_warnings=False)

    assert all(
        descriptor.key != native_tool_module.NATIVE_WAN14B_I2V_KEY
        for descriptor in registry.descriptors()
    )


def test_explicit_pipeline_support_may_include_runtime_ignored_weights(tmp_path: Path):
    settings = _settings(tmp_path)
    support = _write_support_tree(settings.model_root / "wan22" / "invalid-support")
    (support / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
        b"dense"
    )
    (support / ".latentslate-resource.toml").write_text(
        '''
id = "model:wan22:invalid-support"
name = "Invalid support"
family = "wan22"
component = "pipeline_support"
format = "directory"
''',
        encoding="utf-8",
    )

    inventory = discover_resources(settings)

    descriptor = next(
        resource
        for resource in inventory.resources
        if resource.id == "model:wan22:invalid-support"
    )
    assert descriptor.component == "pipeline_support"
    assert descriptor.format.value == "directory"
    assert descriptor not in inventory.matching(kind=descriptor.kind, family="wan22")
