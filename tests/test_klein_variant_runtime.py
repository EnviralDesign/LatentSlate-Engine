from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.runtime.klein import resolve_klein_runtime_plan
from latentslate_engine.runtime.klein_support import KleinRuntimeSupport, nvfp4_host_status
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import klein as klein_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest, LoraExecution


def _support(
    *,
    core: bool = True,
    kernels: bool = True,
    peft: bool = True,
    torchao: bool = True,
    modelopt: bool = True,
    nvfp4: bool = True,
    capability: tuple[int, int] | None = (12, 0),
) -> KleinRuntimeSupport:
    return KleinRuntimeSupport(
        core_available=core,
        core_reason=None if core else "core import failed",
        kernels_available=kernels,
        kernels_reason=None if kernels else "kernels import failed",
        peft_available=peft,
        peft_reason=None if peft else "peft import failed",
        torchao_available=torchao,
        torchao_reason=None if torchao else "torchao import failed",
        modelopt_available=modelopt,
        modelopt_reason=None if modelopt else "modelopt import failed",
        cuda_available=capability is not None,
        cuda_capability=capability,
        nvfp4_available=nvfp4,
        nvfp4_reason=None if nvfp4 else "NVFP4 host unavailable",
        versions=(("torch", "test"),),
    )


@pytest.fixture(autouse=True)
def _full_klein_support(monkeypatch):
    monkeypatch.setattr(klein_tools, "klein_runtime_support", lambda: _support())


def _settings(tmp_path: Path) -> Settings:
    value = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    value.ensure_directories()
    model = (
        value.model_root
        / "klein4b"
        / "black-forest-labs--FLUX.2-klein-4B"
    )
    model.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    return value


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
        "group_offload_use_stream": True,
        "group_offload_record_stream": True,
        "low_cpu_mem_usage": True,
        "keep_pipeline_loaded": True,
    }
    values.update(updates)
    return values


def test_klein_capability_matrix_is_exact():
    capabilities4 = klein_tools.Klein4BTextToImageTool().execution_capabilities()
    capabilities9 = klein_tools.KleinTextToImageTool().execution_capabilities()

    assert capabilities4.model_formats == frozenset({"diffusers"})
    assert capabilities4.lora_formats == frozenset({"safetensors"})
    assert capabilities4.attention_modes == frozenset(
        {"native", "flash_hub", "flash3_hub", "flash4_hub", "sage_hub"}
    )
    assert capabilities4.offload_modes == frozenset(
        {"none", "model", "sequential", "group_block", "group_leaf"}
    )
    assert capabilities4.quantization_modes == frozenset(
        {"native", "bf16", "int8"}
    )
    assert capabilities9.quantization_modes == frozenset(
        {"native", "bf16", "int8", "nvfp4"}
    )
    assert capabilities4.compile_modes == frozenset(
        {"default", "reduce-overhead", "max-autotune"}
    )
    assert capabilities4.vae_tiling_modes == frozenset({"on", "off"})
    assert capabilities4.vae_slicing_modes == frozenset({"on", "off"})
    assert capabilities4.cache_modes == frozenset({"none", "prompt", "media"})
    assert capabilities4.residency_policy
    assert not capabilities4.load_policy
    assert not capabilities4.runtime_parameters


def test_klein_capabilities_follow_host_support(monkeypatch):
    monkeypatch.setattr(
        klein_tools,
        "klein_runtime_support",
        lambda: _support(kernels=False, peft=False, torchao=False, nvfp4=False),
    )
    capabilities4 = klein_tools.Klein4BTextToImageTool().execution_capabilities()
    capabilities9 = klein_tools.KleinTextToImageTool().execution_capabilities()

    assert capabilities4.attention_modes == frozenset({"native"})
    assert capabilities4.lora_formats == frozenset()
    assert capabilities4.quantization_modes == frozenset({"native", "bf16"})
    assert "nvfp4" not in capabilities9.quantization_modes


def test_nvfp4_host_probe_rejects_cpu_and_ada_and_accepts_blackwell():
    class FakeCuda:
        def __init__(self, available, capability=None):
            self.available = available
            self.capability = capability

        def is_available(self):
            return self.available

        def current_device(self):
            return 0

        def get_device_capability(self, _device):
            return self.capability

    cpu = SimpleNamespace(cuda=FakeCuda(False))
    ada = SimpleNamespace(cuda=FakeCuda(True, (8, 9)))
    blackwell = SimpleNamespace(cuda=FakeCuda(True, (12, 0)))

    assert nvfp4_host_status(cpu)[0] is False
    assert "visible CUDA GPU" in str(nvfp4_host_status(cpu)[2])
    assert nvfp4_host_status(ada)[0] is False
    assert "SM89" in str(nvfp4_host_status(ada)[2])
    assert nvfp4_host_status(blackwell) == (True, (12, 0), None)


def test_klein_base_descriptor_reports_real_import_failure(monkeypatch):
    monkeypatch.setattr(
        klein_tools,
        "klein_runtime_support",
        lambda: _support(core=False, nvfp4=False, capability=None),
    )
    descriptor = klein_tools.Klein4BTextToImageTool().descriptor

    assert descriptor.available is False
    assert descriptor.unavailable_reason == "core import failed"


def test_klein9_base_requires_nvfp4_but_variants_only_require_core(monkeypatch):
    monkeypatch.setattr(
        klein_tools,
        "klein_runtime_support",
        lambda: _support(nvfp4=False, capability=(8, 9)),
    )
    tool = klein_tools.KleinTextToImageTool()

    assert tool.descriptor.available is False
    assert tool.variant_base_availability() == (True, None)
    assert "nvfp4" not in tool.execution_capabilities().quantization_modes


def test_klein_accepts_implemented_modes_and_rejects_unimplemented_modes():
    tool = klein_tools.Klein4BTextToImageTool()
    accepted = ExecutionRequest(
        family="klein4b",
        optimizations=_optimizations(
            attention="native",
            offload="model",
            quantization="int8",
            vae_tiling="on",
            vae_slicing="off",
            cache="prompt",
        ),
    )
    rejected = ExecutionRequest(
        family="klein4b",
        optimizations=_optimizations(
            attention="sage",
            quantization="nvfp4",
        ),
    )

    assert tool.validate_execution_request(accepted) == []
    reasons = tool.validate_execution_request(rejected)
    assert "attention mode 'sage' is not supported" in " ".join(reasons)
    assert "quantization mode 'nvfp4' is not supported" in " ".join(reasons)


def test_klein_int8_plan_records_materialized_weight_loading(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="klein4b.int8",
            family="klein4b",
            optimizations=_optimizations(quantization="int8", offload="model"),
        ),
    )

    assert plan.quantization == "int8"
    assert plan.low_cpu_mem_usage is False
    assert plan.provenance()["optimizations"]["low_cpu_mem_usage"] is False


def test_klein_rejects_unsafe_mode_combinations():
    tool4 = klein_tools.Klein4BTextToImageTool()
    tool9 = klein_tools.KleinTextToImageTool()

    lora_compile = ExecutionRequest(
        family="klein4b",
        loras=True,
        lora_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(
            quantization="bf16",
            compile=True,
            compile_mode="reduce-overhead",
        ),
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
    nvfp4_override = ExecutionRequest(
        family="klein9b",
        model_override=True,
        model_formats=frozenset({"diffusers"}),
        optimizations=_optimizations(quantization="nvfp4"),
    )
    built_in_9b_int8 = ExecutionRequest(
        family="klein9b",
        optimizations=_optimizations(quantization="int8"),
    )
    inherited_9b_lora = ExecutionRequest(
        family="klein9b",
        loras=True,
        lora_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(),
    )

    assert any(
        "LoRA switching is not supported on a compiled transformer" in reason
        for reason in tool4.validate_execution_request(lora_compile)
    )
    group_reasons = tool4.validate_execution_request(group_stream_tiling)
    assert any("VAE tiling" in reason for reason in group_reasons)
    assert any("group_offload_blocks = 1" in reason for reason in group_reasons)
    assert any(
        "not an arbitrary model override" in reason
        for reason in tool9.validate_execution_request(nvfp4_override)
    )
    assert any(
        "requires a complete Diffusers model override" in reason
        for reason in tool9.validate_execution_request(built_in_9b_int8)
    )
    assert any(
        "must explicitly choose quantization" in reason
        for reason in tool9.validate_execution_request(inherited_9b_lora)
    )


def test_model_override_and_dynamic_lora_share_pipeline_fingerprint(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "klein4b" / "custom-model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    lora_path = settings.lora_root / "klein4b" / "style.safetensors"
    lora_path.write_bytes(b"lora")

    base_execution = ExecutionPlan(
        variant_key="klein4b.custom",
        family="klein4b",
        model_resource_id="model:klein4b:custom-model",
        model_path=model,
        model_format="diffusers",
        optimizations=_optimizations(
            attention="flash4_hub",
            offload="model",
            quantization="bf16",
            cache="prompt",
        ),
    )
    lora_execution = ExecutionPlan(
        variant_key="klein4b.custom.lora",
        family="klein4b",
        model_resource_id="model:klein4b:custom-model",
        model_path=model,
        model_format="diffusers",
        loras=(
            LoraExecution(
                slot="style",
                resource_id="lora:klein4b:style",
                path=lora_path,
                strength=0.6,
            ),
        ),
        optimizations=_optimizations(
            attention="flash4_hub",
            offload="model",
            quantization="bf16",
            cache="media",
            keep_pipeline_loaded=False,
        ),
    )
    first = resolve_klein_runtime_plan(settings, "klein4b", base_execution)
    second = resolve_klein_runtime_plan(settings, "klein4b", lora_execution)

    assert first.pipeline_fingerprint == second.pipeline_fingerprint
    assert first.model_resource_id == "model:klein4b:custom-model"
    assert first.attention == "flash4_hub"
    assert second.cache == "media"
    assert not second.keep_pipeline_loaded
    assert first.lora_signature != second.lora_signature


def test_klein_runtime_manager_key_reuses_pipeline_across_lora_changes(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    model = settings.model_root / "klein4b" / "custom-model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    execution = ExecutionPlan(
        variant_key="klein4b.custom",
        family="klein4b",
        model_resource_id="model:klein4b:custom-model",
        model_path=model,
        model_format="diffusers",
        optimizations=_optimizations(quantization="bf16", offload="model"),
    )
    first_plan = resolve_klein_runtime_plan(settings, "klein4b", execution)
    second_plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="klein4b.custom.other",
            family="klein4b",
            model_resource_id="model:klein4b:custom-model",
            model_path=model,
            model_format="diffusers",
            optimizations=_optimizations(
                quantization="bf16",
                offload="model",
                cache="none",
            ),
        ),
    )
    accelerated_plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="klein4b.custom.flash",
            family="klein4b",
            model_resource_id="model:klein4b:custom-model",
            model_path=model,
            model_format="diffusers",
            optimizations=_optimizations(
                quantization="bf16",
                offload="model",
                attention="flash4_hub",
            ),
        ),
    )
    created = []

    class FakeRuntime:
        def __init__(self, _settings, variant, plan):
            self.variant = variant
            self.plan = plan
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    monkeypatch.setattr(klein_tools, "KleinRuntime", FakeRuntime)
    tool = klein_tools.Klein4BTextToImageTool()
    context = SimpleNamespace(settings=settings)
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
