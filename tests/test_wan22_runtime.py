from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from latentslate_engine.config import Settings
from latentslate_engine.protocol import InputRole, WorkflowKind
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.runtime.wan22 import (
    WAN22_MAX_FRAMES,
    WAN22_MIN_FRAMES,
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
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
        wan22_profile=profile,
    )
    settings.ensure_directories()
    model = settings.model_root / "wan22" / "Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    return settings


def _support():
    return SimpleNamespace(core_available=True, core_reason=None)


def test_wan22_frame_counts_follow_temporal_contract():
    for duration in (1.0, 2.0, 5.0, 10.0):
        frames = frames_for_duration(duration)
        assert (frames - 1) % 4 == 0
        assert WAN22_MIN_FRAMES <= frames <= WAN22_MAX_FRAMES


def test_wan22_tool_follows_latentslate_taxonomy():
    descriptor = wan22_tools.Wan22TextToVideoTool().descriptor
    assert descriptor.key == "wan22.text_to_video"
    assert descriptor.workflow_kind == WorkflowKind.TEXT_TO_VIDEO
    inputs = {item.key: item for item in descriptor.inputs}
    assert "size" not in inputs
    assert (inputs["width"].default, inputs["height"].default) == (1280, 704)
    assert inputs["width"].role == InputRole.WIDTH
    assert inputs["height"].role == InputRole.HEIGHT


def test_wan22_rejects_aligned_over_budget_canvas_before_loading_pipeline(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    runtime = Wan22Runtime(settings, plan)
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    try:
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "no-output.mp4",
            width=1280,
            height=721,
            duration_seconds=1.0,
            seed=0,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )
    except ValueError as exc:
        assert "pixel budget" in str(exc)
    else:
        raise AssertionError("over-budget dimensions reached the pipeline")


def test_wan22_advertises_bf16_and_engine_native_fp8(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    tool = wan22_tools.Wan22TextToVideoTool()
    assert tool.execution_capabilities().quantization_modes == frozenset({"bf16", "fp8"})
    errors = tool.validate_execution_request(
        ExecutionRequest(
            family="wan22",
            optimizations={"quantization": "int8", "offload": "model"},
        )
    )
    assert any("quantization mode 'int8' is not supported" in error for error in errors)


def test_wan5_recipe_provenance_names_the_exact_engine_pipeline(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)

    assert (
        wan22_tools.Wan22TextToVideoTool().variant_provenance("wan5_kitchen")["pipeline"]
        == "WanPipeline"
    )
    assert (
        wan22_tools.Wan22ImageToVideoTool().variant_provenance("wan5_kitchen")["pipeline"]
        == "WanImageToVideoPipeline"
    )


def test_wan5_catalog_availability_requires_direct_kitchen(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    original_find_spec = wan22_tools.importlib.util.find_spec
    monkeypatch.setattr(
        wan22_tools.importlib.util,
        "find_spec",
        lambda name: None if name == "comfy_kitchen" else original_find_spec(name),
    )

    available, reason = wan22_tools.Wan22TextToVideoTool().variant_recipe_availability(
        "wan5_kitchen"
    )

    assert available is False
    assert reason == "Install the direct Kitchen runtime dependency"


def test_wan5_catalog_availability_is_parent_light_and_true_with_prerequisites(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    monkeypatch.setattr(wan22_tools.os, "name", "nt")
    original_find_spec = wan22_tools.importlib.util.find_spec
    monkeypatch.setattr(
        wan22_tools.importlib.util,
        "find_spec",
        lambda name: object() if name == "comfy_kitchen" else original_find_spec(name),
    )

    available, reason = wan22_tools.Wan22TextToVideoTool().variant_recipe_availability(
        "wan5_kitchen"
    )

    assert available is True
    assert reason is None


def test_wan5_catalog_availability_requires_windows_job_objects(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    monkeypatch.setattr(wan22_tools.os, "name", "posix")

    available, reason = wan22_tools.Wan22TextToVideoTool().variant_recipe_availability(
        "wan5_kitchen"
    )

    assert available is False
    assert reason == "Engine-native Wan 5B Kitchen execution requires Windows Job Objects"


def test_wan5_catalog_availability_never_imports_torch(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    monkeypatch.setattr(wan22_tools.os, "name", "nt")
    original_find_spec = wan22_tools.importlib.util.find_spec
    monkeypatch.setattr(
        wan22_tools.importlib.util,
        "find_spec",
        lambda name: object() if name == "comfy_kitchen" else original_find_spec(name),
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    available, reason = wan22_tools.Wan22TextToVideoTool().variant_recipe_availability(
        "wan5_kitchen"
    )

    assert available is True
    assert reason is None
    assert "torch" not in sys.modules


def test_wan22_plan_records_native_bf16_artifact_metadata(tmp_path: Path):
    plan = resolve_wan22_runtime_plan(_settings(tmp_path), None)
    assert (plan.quantization, plan.model_precision, plan.model_quantization) == (
        "bf16",
        "bf16",
        "native",
    )
    assert plan.provenance()["model"]["precision"] == "bf16"


def test_wan22_plans_are_fingerprinted_by_exact_nonconversion_load_recipe(tmp_path: Path):
    settings = _settings(tmp_path)
    base = resolve_wan22_runtime_plan(settings, None)
    group_leaf = resolve_wan22_runtime_plan(
        settings,
        ExecutionPlan(
            variant_key="wan22.group-leaf",
            family="wan22",
            optimizations={"quantization": "bf16", "offload": "group_leaf", "vae_tiling": "on"},
        ),
    )
    assert (base.quantization, base.offload, base.vae_tiling, base.cache) == (
        "bf16",
        "sequential",
        "on",
        "prompt",
    )
    assert base.pipeline_fingerprint != group_leaf.pipeline_fingerprint


def test_wan22_rejects_legacy_runtime_conversion_profile(tmp_path: Path):
    try:
        resolve_wan22_runtime_plan(_settings(tmp_path, profile="int8_model_offload"), None)
    except RuntimeError as exc:
        assert "Unknown Wan 2.2 profile" in str(exc)
    else:
        raise AssertionError("runtime INT8 conversion profile was accepted")


def test_wan22_rejects_mismatched_selected_artifact(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "wan22" / "Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    execution = ExecutionPlan(
        variant_key="wan22.wrong-artifact",
        family="wan22",
        model_resource_id="model:wan22:unknown",
        model_path=model,
        model_format="diffusers",
        model_precision="unknown",
        model_quantization="unknown",
        optimizations={"quantization": "bf16", "offload": "sequential"},
    )
    try:
        resolve_wan22_runtime_plan(settings, execution)
    except ValueError as exc:
        assert "requires a native bf16 artifact" in str(exc)
    else:
        raise AssertionError("unknown artifact metadata was accepted")


def test_wan22_rejects_mixed_inherited_load_recipes(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    tool = wan22_tools.Wan22TextToVideoTool()
    reasons = tool.validate_execution_request(
        ExecutionRequest(
            family="wan22",
            optimizations={"quantization": "inherit", "offload": "sequential"},
        )
    )
    assert any("must either both inherit" in reason for reason in reasons)


def test_wan22_rejects_unsafe_streamed_group_offload(monkeypatch):
    monkeypatch.setattr(wan22_tools, "wan22_runtime_support", _support)
    reasons = wan22_tools.Wan22TextToVideoTool().validate_execution_request(
        ExecutionRequest(
            family="wan22",
            optimizations={
                "quantization": "bf16",
                "offload": "group_leaf",
                "vae_tiling": "on",
                "group_offload_use_stream": True,
            },
        )
    )
    assert any("disable group-offload streams" in reason for reason in reasons)


def test_prompt_cache_miss_unloads_pipeline_before_isolated_encoder(tmp_path, monkeypatch):
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
    first, first_hit, first_stage = runtime._prompt_conditioning_cpu(
        plan, "a test shot", progress=lambda *_: None, check_cancelled=lambda: None
    )
    second, second_hit, second_stage = runtime._prompt_conditioning_cpu(
        plan, "a test shot", progress=lambda *_: None, check_cancelled=lambda: None
    )

    assert first == second == ("positive", "negative")
    assert not first_hit and second_hit
    assert first_stage["pipeline_unloaded_before_encode"] is True
    assert second_stage["worker_seconds"] == 0.0
    assert events == ["unload", ("worker", "a test shot")]


def test_wan22_pipeline_is_constructed_without_live_text_encoder(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    calls = []
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16, fake_torch.float32 = object(), object()
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

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan, fake_diffusers.UniPCMultistepScheduler = FakeVae, FakeScheduler
    fake_diffusers.WanPipeline, fake_diffusers.WanTransformer3DModel = FakePipeline, FakeTransformer
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    runtime = Wan22Runtime(settings, plan)
    pipe = runtime._load_pipeline()
    pipeline_kwargs = next(call[1] for call in calls if call[0] == "pipeline")
    assert pipe is runtime._pipeline
    assert pipeline_kwargs["tokenizer"] is None and pipeline_kwargs["text_encoder"] is None
    assert ("offload", "sequential", "cuda") in calls
    assert "quantization_config" not in next(call[2] for call in calls if call[0] == "transformer")


def test_wan22_pipeline_load_failure_releases_partial_components(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    cleaned = []
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16, fake_torch.float32 = object(), object()
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
    fake_diffusers.AutoencoderKLWan, fake_diffusers.WanTransformer3DModel = (
        BrokenVae,
        FakeTransformer,
    )
    fake_diffusers.UniPCMultistepScheduler, fake_diffusers.WanPipeline = object, object
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22.cleanup_accelerator_memory", lambda: cleaned.append(True)
    )
    runtime = Wan22Runtime(settings, plan)
    try:
        runtime._load_pipeline()
    except RuntimeError as exc:
        assert "VAE load failed" in str(exc)
    else:
        raise AssertionError("partial Wan pipeline load unexpectedly succeeded")
    assert runtime._pipeline is None and cleaned == [True]


def test_wan22_runtime_is_reused_by_pipeline_fingerprint(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_wan22_runtime_plan(settings, None)
    created = []

    class FakeRuntime:
        def __init__(self, runtime_settings, runtime_plan):
            self.settings, self.plan = runtime_settings, runtime_plan
            created.append(self)

        def unload(self):
            pass

    monkeypatch.setattr(wan22_tools, "Wan22Runtime", FakeRuntime)
    context = SimpleNamespace(settings=settings)
    RUNTIME_MANAGER.clear()
    try:
        assert wan22_tools.Wan22TextToVideoTool()._runtime(
            context, plan
        ) is wan22_tools.Wan22TextToVideoTool()._runtime(context, plan)
    finally:
        RUNTIME_MANAGER.clear()
    assert len(created) == 1
