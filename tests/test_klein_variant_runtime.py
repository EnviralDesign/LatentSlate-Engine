from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.runtime import klein as klein_runtime
from latentslate_engine.runtime import klein_stored_adapter
from latentslate_engine.runtime.klein import resolve_klein_runtime_plan
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import klein as klein_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest, LoraExecution
from latentslate_engine.tools.klein import Klein4BTextToImageTool, KleinTextToImageTool


def _settings(tmp_path: Path, *, profile: str = "bf16_model_offload") -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
        klein4b_profile=profile,
        klein_profile=profile,
    )
    settings.ensure_directories()
    for family, repo in (
        ("klein4b", "black-forest-labs--FLUX.2-klein-4B"),
        ("klein9b", "black-forest-labs--FLUX.2-klein-9B"),
    ):
        model = settings.model_root / family / repo
        model.mkdir(parents=True)
        (model / "model_index.json").write_text("{}", encoding="utf-8")
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


def test_klein_advertises_only_currently_proven_artifacts():
    assert Klein4BTextToImageTool().execution_capabilities().quantization_modes == frozenset(
        {"native", "bf16", "fp8"}
    )
    assert Klein4BTextToImageTool().execution_capabilities().model_formats == frozenset(
        {"diffusers", "safetensors"}
    )
    assert KleinTextToImageTool().execution_capabilities().quantization_modes == frozenset(
        {"native", "bf16"}
    )


def test_klein_rejects_unimplemented_quantized_artifact_modes():
    request = ExecutionRequest(
        family="klein4b",
        optimizations={"quantization": "int8"},
    )
    reasons = Klein4BTextToImageTool().validate_execution_request(request)
    assert any("quantization mode 'int8' is not supported" in reason for reason in reasons)


def test_klein_stored_fp8_rejects_unproven_feature_combinations():
    tool = Klein4BTextToImageTool()
    request = ExecutionRequest(
        family="klein4b",
        model_override=True,
        model_formats=frozenset({"safetensors"}),
        loras=True,
        lora_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(
            quantization="fp8",
            attention="flash4_hub",
            offload="model",
            compile=True,
        ),
    )

    reasons = tool.validate_execution_request(request)

    assert any("native attention only" in reason for reason in reasons)
    assert any("staged residency" in reason for reason in reasons)
    assert any("does not yet support torch.compile" in reason for reason in reasons)
    assert any("FP8 LoRA execution" in reason for reason in reasons)

    no_stored_model = tool.validate_execution_request(
        ExecutionRequest(
            family="klein4b",
            optimizations=_optimizations(quantization="fp8", offload="staged"),
        )
    )
    assert any(
        "requires a stored SafeTensors model override" in reason for reason in no_stored_model
    )
    staged_bf16 = tool.validate_execution_request(
        ExecutionRequest(
            family="klein4b",
            optimizations=_optimizations(quantization="bf16", offload="staged"),
        )
    )
    assert any("reserved for a stored FP8 transformer" in reason for reason in staged_bf16)


def test_klein_default_plan_is_native_bf16_without_component_conversion(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = resolve_klein_runtime_plan(settings, "klein4b", None)

    assert plan.quantization == "bf16"
    assert plan.model_precision == "bf16"
    assert plan.model_quantization == "native"
    assert plan.provenance()["model"]["quantization"] == "native"


def test_klein_stored_fp8_plan_binds_artifact_and_bf16_pipeline_support(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    artifact = settings.model_root / "klein4b" / "stored-fp8.safetensors"
    artifact.write_bytes(b"stored-fp8-fixture")

    class AvailablePlan:
        def require_available(self):
            return None

    monkeypatch.setattr(
        klein_stored_adapter,
        "plan_comfy_klein_transformer",
        lambda path: AvailablePlan(),
    )
    monkeypatch.setattr(
        klein_runtime,
        "validate_diffusers_repository",
        lambda path, contract: None,
    )
    plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="klein4b.stored-fp8",
            family="klein4b",
            model_resource_id="model:klein4b:stored-fp8",
            model_path=artifact,
            model_format="safetensors",
            model_precision="fp8",
            model_quantization="native",
            optimizations=_optimizations(quantization="fp8"),
        ),
    )

    assert plan.quantization == "fp8"
    assert plan.offload == "staged"
    assert plan.component_path("model") == artifact.resolve()
    assert plan.component_path("pipeline_support").name == ("black-forest-labs--FLUX.2-klein-4B")


def test_legacy_conversion_profiles_are_rejected(tmp_path: Path):
    settings = _settings(tmp_path, profile="consumer_int8")
    try:
        resolve_klein_runtime_plan(settings, "klein4b", None)
    except RuntimeError as exc:
        assert "Unknown Klein profile" in str(exc)
    else:
        raise AssertionError("legacy conversion profile was accepted")


def test_klein_runtime_plan_rejects_mismatched_selected_artifact(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    plan = ExecutionPlan(
        variant_key="klein4b.wrong-artifact",
        family="klein4b",
        model_resource_id="model:klein4b:wrong",
        model_path=model,
        model_format="diffusers",
        model_precision="unknown",
        model_quantization="unknown",
        optimizations={"quantization": "bf16"},
    )
    try:
        resolve_klein_runtime_plan(settings, "klein4b", plan)
    except ValueError as exc:
        assert "requires a native bf16 artifact" in str(exc)
    else:
        raise AssertionError("unknown artifact metadata was accepted")


def test_klein_rejects_unsafe_nonconversion_mode_combinations():
    tool = Klein4BTextToImageTool()
    lora_compile = ExecutionRequest(
        family="klein4b",
        loras=True,
        lora_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(quantization="bf16", compile=True),
    )
    group_stream_tiling = ExecutionRequest(
        family="klein4b",
        optimizations=_optimizations(
            offload="group_block",
            group_offload_blocks=2,
            group_offload_use_stream=True,
            vae_tiling="on",
        ),
    )
    assert any(
        "LoRA switching is not supported on a compiled" in reason
        for reason in tool.validate_execution_request(lora_compile)
    )
    reasons = tool.validate_execution_request(group_stream_tiling)
    assert any("VAE tiling" in reason for reason in reasons)
    assert any("group_offload_blocks = 1" in reason for reason in reasons)


def test_model_override_and_dynamic_lora_share_pipeline_fingerprint(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "klein4b" / "custom-model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    lora_path = settings.lora_root / "klein4b" / "style.safetensors"
    lora_path.write_bytes(b"lora")
    common = {
        "family": "klein4b",
        "model_resource_id": "model:klein4b:custom-model",
        "model_path": model,
        "model_format": "diffusers",
        "model_precision": "bf16",
        "model_quantization": "native",
    }
    first = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="first",
            **common,
            optimizations=_optimizations(
                attention="flash4_hub", quantization="bf16", cache="prompt"
            ),
        ),
    )
    second = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="second",
            **common,
            loras=(LoraExecution("style", "lora:klein4b:style", lora_path, 0.6),),
            optimizations=_optimizations(
                attention="flash4_hub",
                quantization="bf16",
                cache="media",
                keep_pipeline_loaded=False,
            ),
        ),
    )
    assert first.pipeline_fingerprint == second.pipeline_fingerprint
    assert first.lora_signature != second.lora_signature
    assert second.cache == "media" and not second.keep_pipeline_loaded


def test_klein_runtime_manager_key_reuses_pipeline_across_lora_changes(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    model = settings.model_root / "klein4b" / "custom-model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    common = {
        "family": "klein4b",
        "model_resource_id": "model:klein4b:custom-model",
        "model_path": model,
        "model_format": "diffusers",
        "model_precision": "bf16",
        "model_quantization": "native",
    }
    first_plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="first", **common, optimizations=_optimizations(quantization="bf16")
        ),
    )
    second_plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="second",
            **common,
            optimizations=_optimizations(quantization="bf16", cache="none"),
        ),
    )
    accelerated_plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="accelerated",
            **common,
            optimizations=_optimizations(quantization="bf16", attention="flash4_hub"),
        ),
    )
    created = []

    class FakeRuntime:
        def __init__(self, _settings, variant, plan):
            self.variant, self.plan = variant, plan
            created.append(self)

        def unload(self):
            pass

    monkeypatch.setattr(klein_tools, "KleinRuntime", FakeRuntime)
    context = SimpleNamespace(settings=settings)
    tool = Klein4BTextToImageTool()
    RUNTIME_MANAGER.clear()
    try:
        first = tool._runtime(context, first_plan)
        second = tool._runtime(context, second_plan)
        accelerated = tool._runtime(context, accelerated_plan)
    finally:
        RUNTIME_MANAGER.clear()
    assert first is second
    assert accelerated is not first
    assert len(created) == 2


def test_klein_tool_evicts_poisoned_warm_runtime(tmp_path: Path, monkeypatch):
    tool = Klein4BTextToImageTool()
    plan = SimpleNamespace(keep_pipeline_loaded=True, provenance=dict)
    runtime = SimpleNamespace(
        generate=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("barrier failed")),
        residency_poisoned=lambda: True,
    )
    recorded: list[dict] = []
    evicted: list[object] = []
    context = SimpleNamespace(
        job_id="job",
        storage=SimpleNamespace(artifact_path=lambda *_: tmp_path / "output.png"),
        record_provenance=lambda **value: recorded.append(value),
        resolve_asset=lambda _asset_id: tmp_path / "unused.png",
        progress=lambda *_: None,
        check_cancelled=lambda: None,
    )
    monkeypatch.setattr(tool, "_resolve_plan", lambda _context: plan)
    monkeypatch.setattr(tool, "_runtime", lambda _context, _plan: runtime)
    monkeypatch.setattr(
        RUNTIME_MANAGER,
        "evict_runtime",
        lambda value, clear_cache: evicted.append(value) or "klein:poisoned",
    )

    with pytest.raises(RuntimeError, match="barrier failed"):
        tool._generate(context, {"prompt": "x", "size": "512x512", "seed": 1}, source_assets=[])

    assert evicted == [runtime]
    assert {"runtime_evicted_after_residency_failure": "klein:poisoned"} in recorded
