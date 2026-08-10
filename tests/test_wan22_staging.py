from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from latentslate_engine.config import Settings
from latentslate_engine.runtime import wan22_prompt_worker
from latentslate_engine.runtime.wan22 import Wan22Runtime, resolve_wan22_runtime_plan


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "wan22" / "Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    model.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text(
        '{"boundary_ratio": null, "expand_timesteps": true}',
        encoding="utf-8",
    )
    return settings


def test_staged_prompt_encoder_uses_pinned_diffusers_prompt_clean(monkeypatch):
    calls: list[str] = []

    diffusers = ModuleType("diffusers")
    diffusers.__path__ = []
    pipelines = ModuleType("diffusers.pipelines")
    pipelines.__path__ = []
    wan = ModuleType("diffusers.pipelines.wan")
    wan.__path__ = []
    pipeline_wan = ModuleType("diffusers.pipelines.wan.pipeline_wan")

    def prompt_clean(value: str) -> str:
        calls.append(value)
        return f"clean:{value.strip()}"

    pipeline_wan.prompt_clean = prompt_clean
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.wan", wan)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.pipelines.wan.pipeline_wan",
        pipeline_wan,
    )

    cleaned = wan22_prompt_worker._clean_prompt_pair(
        "  positive prompt  ",
        "  negative prompt  ",
    )

    assert cleaned == ("clean:positive prompt", "clean:negative prompt")
    assert calls == ["  positive prompt  ", "  negative prompt  "]


def test_prompt_conditioning_clones_file_backed_tensors(monkeypatch, tmp_path):
    events: list[tuple[str, str]] = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name

        def clone(self):
            events.append(("clone", self.name))
            return f"cloned:{self.name}"

    safetensors = ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *_args, **_kwargs: {
        "prompt_embeds": FakeTensor("positive"),
        "negative_prompt_embeds": FakeTensor("negative"),
    }
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    conditioning = Wan22Runtime._load_prompt_conditioning(
        tmp_path / "conditioning.safetensors"
    )

    assert conditioning == ("cloned:positive", "cloned:negative")
    assert events == [("clone", "positive"), ("clone", "negative")]


def test_wan_runtime_status_reports_resolved_quantization_and_offload(tmp_path):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    status = Wan22Runtime(settings, plan).status()

    assert status["quantization"] == "bf16"
    assert status["offload"] == "sequential"
    assert status["staged_text_encoder"] is True
