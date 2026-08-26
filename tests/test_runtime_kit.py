from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from latentslate_engine.runtime.kit import (
    LoraLifecycle,
    RuntimeDefaults,
    apply_attention_backend,
    apply_compile_policy,
    apply_vae_policy,
    is_cuda_oom,
    resolve_runtime_plan,
)
from latentslate_engine.runtime.manager import RuntimeManager
from latentslate_engine.tools.base import ConfiguredLora, ExecutionPlan, LoraExecution


def _defaults(tmp_path: Path) -> RuntimeDefaults:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    return RuntimeDefaults(
        family="klein4b",
        model_id="test/model",
        model_path=model,
        model_format="diffusers",
        device="cuda",
        quantization="bf16",
        attention="native",
        offload="model",
    )


def _execution(
    tmp_path: Path,
    *,
    attention: str = "inherit",
    cache: str = "inherit",
    keep_pipeline_loaded: bool = True,
    loras: tuple[LoraExecution, ...] = (),
    configured_loras: tuple[ConfiguredLora, ...] = (),
) -> ExecutionPlan:
    return ExecutionPlan(
        variant_key="test.variant",
        family="klein4b",
        loras=loras,
        configured_loras=configured_loras,
        optimizations={
            "attention": attention,
            "offload": "inherit",
            "quantization": "inherit",
            "compile": False,
            "compile_mode": "default",
            "compile_fullgraph": False,
            "compile_dynamic": False,
            "vae_tiling": "inherit",
            "vae_slicing": "inherit",
            "cache": cache,
            "group_offload_blocks": None,
            "group_offload_use_stream": False,
            "group_offload_record_stream": False,
            "low_cpu_mem_usage": True,
            "keep_pipeline_loaded": keep_pipeline_loaded,
        },
    )


def test_schema_only_variant_defaults_reuse_base_pipeline(tmp_path: Path):
    defaults = _defaults(tmp_path)
    base = resolve_runtime_plan(None, defaults)
    variant_defaults = resolve_runtime_plan(
        ExecutionPlan(
            variant_key="test.schema_only",
            family="klein4b",
            optimizations={
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
                # These are the variant grammar defaults. They are irrelevant unless
                # a group-offload mode is selected and must not split the pipeline cache.
                "group_offload_blocks": None,
                "group_offload_use_stream": True,
                "group_offload_record_stream": True,
                "low_cpu_mem_usage": True,
                "keep_pipeline_loaded": True,
            },
        ),
        defaults,
    )

    assert base.pipeline_fingerprint == variant_defaults.pipeline_fingerprint


def test_runtime_plan_rejects_cross_family_execution(tmp_path: Path):
    defaults = _defaults(tmp_path)
    execution = ExecutionPlan(
        variant_key="test.wrong_family",
        family="wan22",
        optimizations={},
    )

    try:
        resolve_runtime_plan(execution, defaults)
    except ValueError as exc:
        assert "does not match runtime family" in str(exc)
    else:
        raise AssertionError("cross-family runtime plan was accepted")


def test_pipeline_fingerprint_excludes_dynamic_loras_cache_and_residency(tmp_path: Path):
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_bytes(b"lora")
    lora = LoraExecution(
        slot="style",
        resource_id="lora:klein4b:style",
        path=lora_path,
        strength=0.7,
    )
    defaults = _defaults(tmp_path)
    base = resolve_runtime_plan(None, defaults)
    dynamic = resolve_runtime_plan(
        _execution(
            tmp_path,
            cache="none",
            keep_pipeline_loaded=False,
            loras=(lora,),
        ),
        defaults,
    )
    accelerated = resolve_runtime_plan(
        _execution(tmp_path, attention="sage_hub"),
        defaults,
    )

    assert base.pipeline_fingerprint == dynamic.pipeline_fingerprint
    assert base.pipeline_fingerprint != accelerated.pipeline_fingerprint
    assert base.lora_signature != dynamic.lora_signature
    provenance = dynamic.provenance()
    assert provenance["model"]["id"] == "test/model"
    assert "path" not in provenance["model"]
    assert provenance["optimizations"]["keep_pipeline_loaded"] is False


def test_zero_strength_lora_is_a_base_pipeline_bypass_with_separate_configuration(
    tmp_path: Path,
):
    path = tmp_path / "disabled.safetensors"
    path.write_bytes(b"disabled")
    disabled = LoraExecution("style", "lora:klein4b:disabled", path, 0.0)
    execution = _execution(
        tmp_path,
        loras=(disabled,),
        configured_loras=(ConfiguredLora("style", "lora:klein4b:disabled", 0.0, False),),
    )
    defaults = _defaults(tmp_path)
    plan = resolve_runtime_plan(execution, defaults)

    assert execution.loras == ()
    assert plan.loras == ()
    assert plan.lora_signature == resolve_runtime_plan(None, defaults).lora_signature
    assert plan.provenance()["loras"] == []
    assert plan.provenance()["configured_loras"] == [
        {
            "slot": "style",
            "resource_reference": "lora:klein4b:disabled",
            "strength": 0.0,
            "active": False,
        }
    ]


def test_record_stream_is_canonicalized_off_without_prefetch_stream(tmp_path: Path):
    defaults = _defaults(tmp_path)
    execution = ExecutionPlan(
        variant_key="test.group",
        family="klein4b",
        optimizations={
            "offload": "group_block",
            "group_offload_blocks": 1,
            "group_offload_use_stream": False,
            "group_offload_record_stream": True,
        },
    )
    plan = resolve_runtime_plan(execution, defaults)

    assert plan.group_offload_use_stream is False
    assert plan.group_offload_record_stream is False


def test_attention_backend_maps_public_variant_names():
    calls = []

    class Transformer:
        def set_attention_backend(self, backend):
            calls.append(("set", backend))

        def reset_attention_backend(self):
            calls.append(("reset", None))

    pipeline = SimpleNamespace(transformer=Transformer())

    assert apply_attention_backend(pipeline, "native") == "native"
    assert apply_attention_backend(pipeline, "flash3_hub") == "_flash_3_hub"
    assert apply_attention_backend(pipeline, "flash4_hub") == "flash_4_hub"
    assert apply_attention_backend(pipeline, "sage_hub") == "sage_hub"
    assert calls == [
        ("reset", None),
        ("set", "_flash_3_hub"),
        ("set", "flash_4_hub"),
        ("set", "sage_hub"),
    ]


def test_vae_policy_uses_pipeline_controls():
    calls = []

    class Pipeline:
        def enable_vae_tiling(self):
            calls.append("tiling:on")

        def disable_vae_tiling(self):
            calls.append("tiling:off")

        def enable_vae_slicing(self):
            calls.append("slicing:on")

        def disable_vae_slicing(self):
            calls.append("slicing:off")

    pipeline = Pipeline()
    apply_vae_policy(pipeline, tiling="on", slicing="off")
    apply_vae_policy(pipeline, tiling="off", slicing="on")
    assert calls == ["tiling:on", "slicing:off", "tiling:off", "slicing:on"]


def test_compile_prefers_repeated_block_compilation():
    calls = []

    class Transformer:
        _repeated_blocks = ("Block",)

        def compile_repeated_blocks(self, **kwargs):
            calls.append(kwargs)

    pipeline = SimpleNamespace(transformer=Transformer())
    scope = apply_compile_policy(
        pipeline,
        enabled=True,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    assert scope == "regional"
    assert calls == [{"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}]


def test_compile_falls_back_to_torch_compile(monkeypatch):
    compiled = object()
    calls = []
    fake_torch = ModuleType("torch")

    def compile_model(model, **kwargs):
        calls.append((model, kwargs))
        return compiled

    fake_torch.compile = compile_model
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    transformer = object()
    pipeline = SimpleNamespace(transformer=transformer)

    scope = apply_compile_policy(
        pipeline,
        enabled=True,
        mode="default",
        fullgraph=False,
        dynamic=True,
    )
    assert scope == "full"
    assert pipeline.transformer is compiled
    assert calls == [(transformer, {"mode": "default", "fullgraph": False, "dynamic": True})]


def test_lora_lifecycle_rejects_duplicate_or_excess_active_resources(tmp_path: Path):
    class Pipeline:
        def load_lora_weights(self, *_args, **_kwargs):
            raise AssertionError("invalid selection must fail before pipeline mutation")

        def set_adapters(self, *_args, **_kwargs):
            raise AssertionError("invalid selection must fail before pipeline mutation")

    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = LoraExecution("a", "lora:klein4b:first", first_path, 0.5)
    duplicate = LoraExecution("b", "lora:klein4b:first", first_path, 0.8)
    second = LoraExecution("b", "lora:klein4b:second", second_path, 0.8)

    lifecycle = LoraLifecycle(max_loaded=1)
    try:
        lifecycle.apply(Pipeline(), (first, duplicate), low_cpu_mem_usage=True)
    except ValueError as exc:
        assert "cannot be selected more than once" in str(exc)
    else:
        raise AssertionError("duplicate LoRA selection was accepted")

    try:
        lifecycle.apply(Pipeline(), (first, second), low_cpu_mem_usage=True)
    except ValueError as exc:
        assert "at most 1 simultaneously active LoRAs" in str(exc)
    else:
        raise AssertionError("excess active LoRAs were accepted")


def test_lora_lifecycle_loads_reuses_disables_and_evicts(tmp_path: Path):
    events = []

    class Pipeline:
        def load_lora_weights(self, directory, **kwargs):
            events.append(("load", directory, kwargs))

        def set_adapters(self, names, adapter_weights):
            events.append(("set", list(names), list(adapter_weights)))

        def enable_lora(self):
            events.append(("enable",))

        def disable_lora(self):
            events.append(("disable",))

        def delete_adapters(self, names):
            events.append(("delete", names))

    pipeline = Pipeline()
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = LoraExecution("style", "lora:klein4b:first", first_path, 0.5)
    second = LoraExecution("style", "lora:klein4b:second", second_path, 0.8)
    lifecycle = LoraLifecycle(max_loaded=1)

    first_status = lifecycle.apply(pipeline, (first,), low_cpu_mem_usage=True)
    reused_status = lifecycle.apply(pipeline, (first,), low_cpu_mem_usage=True)
    second_status = lifecycle.apply(pipeline, (second,), low_cpu_mem_usage=True)
    disabled_status = lifecycle.apply(pipeline, (), low_cpu_mem_usage=True)

    assert first_status["loaded_now"] == 1
    assert reused_status["reused"] == 1
    assert second_status["loaded_now"] == 1
    assert second_status["loaded"] == ["lora:klein4b:second"]
    assert disabled_status["active"] == []
    assert any(event[0] == "delete" for event in events)
    assert events[-1] == ("disable",)


def test_lora_lifecycle_does_not_load_or_activate_a_zero_strength_adapter(tmp_path: Path):
    events = []

    class Pipeline:
        def load_lora_weights(self, *_args, **_kwargs):
            events.append("load")

        def set_adapters(self, *_args, **_kwargs):
            events.append("set")

        def enable_lora(self):
            events.append("enable")

        def disable_lora(self):
            events.append("disable")

    path = tmp_path / "disabled.safetensors"
    path.write_bytes(b"disabled")
    status = LoraLifecycle().apply(
        Pipeline(),
        (LoraExecution("style", "lora:klein4b:disabled", path, 0.0),),
        low_cpu_mem_usage=True,
    )

    assert status["active"] == []
    assert status["loaded"] == []
    assert events == []


def test_lora_lifecycle_never_exceeds_peak_bound(tmp_path: Path):
    class Pipeline:
        def __init__(self):
            self.loaded = set()
            self.peak = 0
            self.events = []

        def load_lora_weights(self, _directory, *, adapter_name, **_kwargs):
            self.events.append(("load", adapter_name))
            self.loaded.add(adapter_name)
            self.peak = max(self.peak, len(self.loaded))

        def set_adapters(self, _names, adapter_weights):
            del adapter_weights

        def delete_adapters(self, names):
            names = [names] if isinstance(names, str) else list(names)
            for name in names:
                self.events.append(("delete", name))
                self.loaded.remove(name)

        def enable_lora(self):
            pass

    paths = []
    for name in ("a", "b", "c", "d"):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(name.encode())
        paths.append(path)
    loras = [
        LoraExecution(name, f"lora:klein4b:{name}", path, 1.0)
        for name, path in zip(("a", "b", "c", "d"), paths, strict=True)
    ]
    pipeline = Pipeline()
    lifecycle = LoraLifecycle(max_loaded=2)

    lifecycle.apply(pipeline, tuple(loras[:2]), low_cpu_mem_usage=True)
    event_boundary = len(pipeline.events)
    lifecycle.apply(pipeline, tuple(loras[2:]), low_cpu_mem_usage=True)

    assert pipeline.peak == 2
    assert len(pipeline.loaded) == 2
    swap_events = pipeline.events[event_boundary:]
    first_load = next(index for index, event in enumerate(swap_events) if event[0] == "load")
    assert all(event[0] == "delete" for event in swap_events[:first_load])


def test_lora_lifecycle_fails_closed_without_deletion_api(tmp_path: Path):
    class Pipeline:
        def __init__(self):
            self.loaded = []

        def load_lora_weights(self, _directory, *, adapter_name, **_kwargs):
            self.loaded.append(adapter_name)

        def set_adapters(self, _names, adapter_weights):
            del adapter_weights

        def enable_lora(self):
            pass

    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = LoraExecution("a", "lora:klein4b:first", first_path, 1.0)
    second = LoraExecution("b", "lora:klein4b:second", second_path, 1.0)
    pipeline = Pipeline()
    lifecycle = LoraLifecycle(max_loaded=1)

    lifecycle.apply(pipeline, (first,), low_cpu_mem_usage=True)
    try:
        lifecycle.apply(pipeline, (second,), low_cpu_mem_usage=True)
    except RuntimeError as exc:
        assert "does not implement delete_adapters" in str(exc)
    else:
        raise AssertionError("LoRA swap was allowed without a deletion API")

    assert len(pipeline.loaded) == 1
    assert lifecycle.status()["loaded"] == ["lora:klein4b:first"]


def test_cuda_oom_detection_walks_exception_chain():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 10 MiB"))
    try:
        try:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED")
        except RuntimeError as exc:
            raise ValueError("wrapped") from exc
    except ValueError as wrapped:
        assert is_cuda_oom(wrapped)
    assert not is_cuda_oom(RuntimeError("ordinary generation failure"))


def test_runtime_manager_bounds_inactive_wrappers_and_caches():
    class Runtime:
        def __init__(self):
            self.unloads = 0
            self.clears = 0

        def unload(self):
            self.unloads += 1

        def clear_cache(self):
            self.clears += 1

        def status(self):
            return {"loaded": self.unloads == 0, "active_worker": False}

    manager = RuntimeManager(max_wrappers=2)
    first = manager.activate("first", Runtime)
    manager.activate("second", Runtime)
    manager.activate("third", Runtime)

    status = manager.status()
    assert [runtime["key"] for runtime in status["runtimes"]] == ["third"]
    assert first.unloads >= 1
    assert first.clears == 1


def test_runtime_manager_records_cleanup_failures_without_masking_eviction():
    class BrokenRuntime:
        def unload(self):
            raise RuntimeError("unload failed")

        def clear_cache(self):
            raise RuntimeError("cache failed")

    manager = RuntimeManager()
    runtime = manager.activate("broken", BrokenRuntime)
    assert manager.unload_runtime(runtime)
    assert manager.evict_active() == "broken"
    status = manager.status()
    assert status["active_runtime"] is None
    assert status["runtimes"] == []
    assert any("unload broken" in error for error in status["cleanup_errors"])
    assert any("clear_cache broken" in error for error in status["cleanup_errors"])


def test_runtime_manager_can_unload_and_evict_poisoned_wrapper():
    class Runtime:
        def __init__(self):
            self.unloads = 0
            self.clears = 0

        def unload(self):
            self.unloads += 1

        def clear_cache(self):
            self.clears += 1

    manager = RuntimeManager()
    runtime = manager.activate(("family", "plan"), Runtime)
    assert manager.unload_runtime(runtime)
    assert runtime.unloads == 1
    label = manager.evict_active(clear_cache=True)
    assert label == "family:plan"
    assert runtime.unloads == 2
    assert runtime.clears == 1
    assert manager.status()["active_runtime"] is None
    assert manager.status()["runtimes"] == []
