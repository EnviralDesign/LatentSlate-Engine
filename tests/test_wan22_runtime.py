from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import WorkflowKind
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.runtime.wan22 import (
    WAN22_MAX_FRAMES,
    WAN22_MIN_FRAMES,
    WAN22_SIZE_PRESETS,
    Wan22Runtime,
    frames_for_duration,
    resolve_wan22_runtime_plan,
)
from latentslate_engine.tools import wan22 as wan22_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest


def _settings(tmp_path: Path, *, profile: str = "bf16_sequential_offload") -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
        wan22_profile=profile,
    )
    settings.ensure_directories()
    model = settings.model_root / "wan22" / "Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    model.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text(
        '{"boundary_ratio": null, "expand_timesteps": true}',
        encoding="utf-8",
    )
    (model / "transformer").mkdir()
    (model / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    return settings


def _optimizations(**updates):
    values = {
        "attention": "inherit",
        "offload": "inherit",
        "quantization": "inherit",
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "compile_dynamic": False,
        "vae_tiling": "inherit",
        "vae_slicing": "inherit",
        "cache": "inherit",
        "group_offload_blocks": None,
        "group_offload_use_stream": False,
        "group_offload_record_stream": False,
        "low_cpu_mem_usage": True,
        "keep_pipeline_loaded": True,
    }
    values.update(updates)
    return values


def _support(*, torchao: bool = True):
    return SimpleNamespace(
        core_available=True,
        core_reason=None,
        torchao_available=torchao,
        torchao_reason=None if torchao else "TorchAO unavailable",
    )


def test_wan22_frame_counts_follow_temporal_contract():
    for duration in (1.0, 2.0, 5.0, 10.0):
        frames = frames_for_duration(duration)
        assert (frames - 1) % 4 == 0
        assert WAN22_MIN_FRAMES <= frames <= WAN22_MAX_FRAMES


def test_wan22_tool_follows_latentslate_taxonomy():
    descriptor = wan22_tools.Wan22TextToVideoTool().descriptor

    assert descriptor.name == "Text to Video"
    assert descriptor.key == "wan22.text_to_video"
    assert descriptor.workflow_kind == WorkflowKind.TEXT_TO_VIDEO
    assert descriptor.inputs[1].default == "1280x704"
    assert descriptor.inputs[2].default == 5.0
    assert {option.value for option in descriptor.inputs[1].options} == set(
        WAN22_SIZE_PRESETS
    )


def test_wan22_bundle_and_curated_defaults_are_unchanged(tmp_path):
    assert BUNDLES["wan22-basic"].repo_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    settings = _settings(tmp_path)
    assert settings.wan22_model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert settings.wan22_profile == "bf16_sequential_offload"


def test_wan22_capability_matrix_is_exact(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    capabilities = wan22_tools.Wan22TextToVideoTool().execution_capabilities()

    assert capabilities.model_formats == frozenset({"diffusers"})
    assert capabilities.lora_formats == frozenset()
    assert capabilities.attention_modes == frozenset({"native"})
    assert capabilities.offload_modes == frozenset({"sequential", "group_leaf", "model"})
    assert capabilities.quantization_modes == frozenset({"bf16", "int8"})
    assert capabilities.compile_modes == frozenset()
    assert capabilities.vae_tiling_modes == frozenset({"on"})
    assert capabilities.vae_slicing_modes == frozenset()
    assert capabilities.cache_modes == frozenset({"none", "prompt"})
    assert capabilities.residency_policy
    assert not capabilities.runtime_parameters


def test_wan22_capabilities_hide_int8_without_torchao(monkeypatch):
    monkeypatch.setattr(
        wan22_tools,
        "wan22_runtime_support",
        lambda: _support(torchao=False),
    )
    tool = wan22_tools.Wan22TextToVideoTool()

    assert tool.execution_capabilities().quantization_modes == frozenset({"bf16"})
    reasons = tool.validate_execution_request(
        ExecutionRequest(
            family="wan22",
            optimizations=_optimizations(
                quantization="int8",
                offload="model",
                vae_tiling="on",
            ),
        )
    )
    assert any("TorchAO unavailable" in reason for reason in reasons)


def test_wan22_accepts_only_reviewed_recovery_combinations(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    tool = wan22_tools.Wan22TextToVideoTool()

    safe_bf16 = ExecutionRequest(
        family="wan22",
        optimizations=_optimizations(
            quantization="bf16",
            offload="group_leaf",
            vae_tiling="on",
            cache="prompt",
        ),
    )
    safe_int8 = ExecutionRequest(
        family="wan22",
        optimizations=_optimizations(
            quantization="int8",
            offload="model",
            vae_tiling="on",
            cache="prompt",
        ),
    )
    unsafe_int8_group = ExecutionRequest(
        family="wan22",
        optimizations=_optimizations(
            quantization="int8",
            offload="group_leaf",
            vae_tiling="on",
        ),
    )
    unsafe_stream = ExecutionRequest(
        family="wan22",
        optimizations=_optimizations(
            quantization="bf16",
            offload="group_leaf",
            vae_tiling="on",
            group_offload_use_stream=True,
        ),
    )

    assert tool.validate_execution_request(safe_bf16) == []
    assert tool.validate_execution_request(safe_int8) == []
    assert any(
        "INT8 is implemented only with offload='model'" in reason
        for reason in tool.validate_execution_request(unsafe_int8_group)
    )
    assert any(
        "disable group-offload streams" in reason
        for reason in tool.validate_execution_request(unsafe_stream)
    )


def test_wan22_plans_are_fingerprinted_by_exact_load_recipe(tmp_path):
    settings = _settings(tmp_path)
    base = resolve_wan22_runtime_plan(settings, None)
    group_leaf = resolve_wan22_runtime_plan(
        settings,
        ExecutionPlan(
            variant_key="wan22.safe_bf16",
            family="wan22",
            optimizations=_optimizations(
                quantization="bf16",
                offload="group_leaf",
                vae_tiling="on",
                cache="prompt",
            ),
        ),
    )
    int8 = resolve_wan22_runtime_plan(
        settings,
        ExecutionPlan(
            variant_key="wan22.safe_int8",
            family="wan22",
            optimizations=_optimizations(
                quantization="int8",
                offload="model",
                vae_tiling="on",
                cache="prompt",
            ),
        ),
    )

    assert base.quantization == "bf16"
    assert base.offload == "sequential"
    assert base.vae_tiling == "on"
    assert base.cache == "prompt"
    assert group_leaf.pipeline_fingerprint != base.pipeline_fingerprint
    assert int8.pipeline_fingerprint != group_leaf.pipeline_fingerprint
    assert int8.low_cpu_mem_usage is False


def test_prompt_cache_miss_unloads_pipeline_before_isolated_encoder(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    runtime = Wan22Runtime(settings, plan)
    events = []

    class FakePipeline:
        def remove_all_hooks(self):
            events.append("unload")

    runtime._pipeline = FakePipeline()

    def fake_worker(_plan, prompt, *, check_cancelled):
        check_cancelled()
        events.append(("worker", prompt))
        return ("positive", "negative"), 1.25

    monkeypatch.setattr(runtime, "_run_prompt_worker", fake_worker)
    progress = lambda _value, _message: None
    check = lambda: None

    first, first_hit, first_stage = runtime._prompt_conditioning_cpu(
        plan,
        "a test shot",
        progress=progress,
        check_cancelled=check,
    )
    second, second_hit, second_stage = runtime._prompt_conditioning_cpu(
        plan,
        "a test shot",
        progress=progress,
        check_cancelled=check,
    )

    assert first == ("positive", "negative")
    assert second == first
    assert not first_hit
    assert second_hit
    assert first_stage["pipeline_unloaded_before_encode"] is True
    assert second_stage["worker_seconds"] == 0.0
    assert runtime._pipeline is None
    assert events == ["unload", ("worker", "a test shot")]


def test_wan22_pipeline_is_constructed_without_live_text_encoder(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    calls = []

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("transformer", Path(path), kwargs))
            return cls()

        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

        def reset_attention_backend(self):
            calls.append(("attention", "native"))

    class FakeVae:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("vae", Path(path), kwargs))
            return cls()

    class FakeScheduler:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("scheduler", Path(path), kwargs))
            return cls()

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(("pipeline", kwargs))
            self.transformer = kwargs["transformer"]
            self.vae = kwargs["vae"]

        def enable_vae_tiling(self):
            calls.append(("vae_tiling", "on"))

        def disable_vae_slicing(self):
            calls.append(("vae_slicing", "off"))

        def enable_sequential_cpu_offload(self, *, device):
            calls.append(("offload", "sequential", device))

        def set_progress_bar_config(self, **kwargs):
            calls.append(("progress", kwargs))

        def remove_all_hooks(self):
            pass

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan = FakeVae
    fake_diffusers.TorchAoConfig = lambda config: config
    fake_diffusers.UniPCMultistepScheduler = FakeScheduler
    fake_diffusers.WanPipeline = FakePipeline
    fake_diffusers.WanTransformer3DModel = FakeTransformer
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    runtime = Wan22Runtime(settings, plan)
    pipe = runtime._load_pipeline()

    pipeline_kwargs = next(call[1] for call in calls if call[0] == "pipeline")
    assert pipe is runtime._pipeline
    assert pipeline_kwargs["tokenizer"] is None
    assert pipeline_kwargs["text_encoder"] is None
    assert pipeline_kwargs["expand_timesteps"] is True
    assert ("offload", "sequential", "cuda") in calls


def test_wan22_pipeline_load_failure_releases_partial_components(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    cleaned = []

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            return cls()

        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

    class BrokenVae:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            raise RuntimeError("VAE load failed")

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan = BrokenVae
    fake_diffusers.TorchAoConfig = lambda config: config
    fake_diffusers.UniPCMultistepScheduler = object
    fake_diffusers.WanPipeline = object
    fake_diffusers.WanTransformer3DModel = FakeTransformer
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22.cleanup_accelerator_memory",
        lambda: cleaned.append(True),
    )

    runtime = Wan22Runtime(settings, plan)
    try:
        runtime._load_pipeline()
    except RuntimeError as exc:
        assert "VAE load failed" in str(exc)
    else:
        raise AssertionError("partial Wan pipeline load unexpectedly succeeded")

    assert runtime._pipeline is None
    assert cleaned == [True]


def test_wan22_int8_loader_uses_torchao_weight_only_config(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path, profile="int8_model_offload")
    plan = resolve_wan22_runtime_plan(settings, None)
    calls = []

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeInt8Config:
        def __init__(self, *, version):
            self.version = version

    fake_torchao_quantization = ModuleType("torchao.quantization")
    fake_torchao_quantization.Int8WeightOnlyConfig = FakeInt8Config
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_torchao_quantization)

    class FakeTorchAoConfig:
        def __init__(self, config):
            self.config = config

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((Path(path), kwargs))
            return cls()

        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.TorchAoConfig = FakeTorchAoConfig
    fake_diffusers.WanTransformer3DModel = FakeTransformer
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    transformer = Wan22Runtime._load_transformer(plan)

    assert isinstance(transformer, FakeTransformer)
    kwargs = calls[0][1]
    assert kwargs["low_cpu_mem_usage"] is False
    assert kwargs["quantization_config"].config.version == 2


def test_wan22_runtime_is_reused_by_pipeline_fingerprint(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    created = []

    class FakeRuntime:
        def __init__(self, runtime_settings, runtime_plan):
            self.settings = runtime_settings
            self.plan = runtime_plan
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(wan22_tools, "Wan22Runtime", FakeRuntime)
    context = SimpleNamespace(settings=settings)
    tool = wan22_tools.Wan22TextToVideoTool()

    first = tool._runtime(context, plan)
    second = tool._runtime(context, plan)

    assert first is second
    assert created == [first]
    RUNTIME_MANAGER.clear()
