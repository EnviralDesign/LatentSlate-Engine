from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.config import Settings
from latentslate_engine.klein_recipe import (
    Klein4RecipeComponent,
    Klein4RuntimeRequest,
    KleinStoredRecipe,
    validate_klein_stored_recipe,
)
from latentslate_engine.resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)
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
        {"native", "bf16", "fp8", "nvfp4"}
    )
    assert Klein4BTextToImageTool().execution_capabilities().model_formats == frozenset(
        {"diffusers", "safetensors"}
    )
    assert KleinTextToImageTool().execution_capabilities().quantization_modes == frozenset(
        {"native", "bf16", "fp8", "nvfp4"}
    )
    assert Klein4BTextToImageTool().execution_capabilities().recipe_types == frozenset(
        {"klein4_stored"}
    )
    assert KleinTextToImageTool().execution_capabilities().recipe_types == frozenset(
        {"klein9_stored"}
    )


def test_klein_rejects_unimplemented_quantized_artifact_modes():
    request = ExecutionRequest(
        family="klein4b",
        optimizations={"quantization": "int8"},
    )
    reasons = Klein4BTextToImageTool().validate_execution_request(request)
    assert any("quantization mode 'int8' is not supported" in reason for reason in reasons)


def test_klein_nvfp4_rejects_standalone_override_and_requires_typed_recipe(tmp_path: Path):
    tool = Klein4BTextToImageTool()
    request = ExecutionRequest(
        family="klein4b",
        model_override=True,
        model_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(quantization="nvfp4", offload="staged"),
    )
    assert any(
        "only through its typed component recipe" in reason
        for reason in tool.validate_execution_request(request)
    )
    resource = SimpleNamespace(
        component=None,
        format=klein_tools.ResourceFormat.SAFETENSORS,
        precision=klein_tools.ArtifactPrecision.FP4,
        quantization=klein_tools.ArtifactQuantization.NVFP4,
    )
    errors = tool.validate_model_resource(resource, tmp_path / "dropped.safetensors")
    assert any("typed recipe" in reason for reason in errors)


def test_klein_stored_fp8_accepts_lora_but_rejects_other_unproven_combinations():
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
    assert not any("LoRA execution" in reason for reason in reasons)

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
    assert any("reserved for a stored quantized transformer" in reason for reason in staged_bf16)

    component_recipe = tool.validate_execution_request(
        ExecutionRequest(
            family="klein4b",
            recipe_type="klein4_stored",
            optimizations=_optimizations(quantization="fp8", offload="staged", attention="native"),
        )
    )
    assert component_recipe == []

    nvfp4_recipe = tool.validate_execution_request(
        ExecutionRequest(
            family="klein4b",
            recipe_type="klein4_stored",
            optimizations=_optimizations(
                quantization="nvfp4", offload="staged", attention="native"
            ),
        )
    )
    assert nvfp4_recipe == []


def test_klein_component_recipe_plan_binds_all_roles_and_schedule(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    transformer = tmp_path / "transformer.safetensors"
    text_encoder = tmp_path / "qwen.safetensors"
    vae = tmp_path / "vae.safetensors"
    support = tmp_path / "support"
    transformer.write_bytes(b"transformer")
    text_encoder.write_bytes(b"text")
    vae.write_bytes(b"vae")
    support.mkdir()
    (support / "model_index.json").write_text("{}", encoding="utf-8")
    identities = {
        role: ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, role)
        for role, path in {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": vae,
        }.items()
    }
    request = Klein4RuntimeRequest(
        1,
        "klein4b",
        "base",
        "flux2-klein-base-4b",
        20,
        5.0,
        {
            "pipeline_support": {
                "resource_id": "model:klein4b:support",
                "path": str(support.resolve()),
            },
            "transformer": {
                "resource_id": "model:klein4b:base-transformer",
                "path": str(transformer.resolve()),
                "quantization_contract": "comfy_quant/float8_e4m3fn_global",
            },
            "text_encoder": {
                "resource_id": "model:klein4b:qwen",
                "path": str(text_encoder.resolve()),
            },
            "vae": {
                "resource_id": "model:klein4b:vae",
                "path": str(vae.resolve()),
            },
        },
        identities,
        SimpleNamespace(root=support),
        {},
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.revalidate_klein4_runtime_request",
        lambda value: value is request,
    )

    plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="flux2-klein-4b.image-to-image.comfy-base-fp8",
            family="klein4b",
            optimizations=_optimizations(
                quantization="fp8",
                offload="staged",
                attention="native",
            ),
            recipe=request,
        ),
    )

    assert plan.model_path == transformer.resolve()
    assert plan.model_resource_id == "model:klein4b:base-transformer"
    assert plan.quantization == "fp8" and plan.offload == "staged"
    assert plan.component_path("pipeline_support") == support.resolve()
    assert plan.component_path("text_encoder") == text_encoder.resolve()
    assert plan.component_path("vae") == vae.resolve()
    assert dict(plan.pipeline_parameters) == {
        "guidance_scale": 5.0,
        "recipe_fingerprint": request.fingerprint,
        "recipe_mode": "base",
        "steps": 20,
    }


def test_klein_installed_nvfp4_recipe_request_resolves_exact_runtime_plan(
    tmp_path: Path, monkeypatch
):
    settings = _settings(tmp_path)
    transformer = tmp_path / "transformer-nvfp4.safetensors"
    text_encoder = tmp_path / "qwen.safetensors"
    vae = tmp_path / "vae.safetensors"
    support = tmp_path / "support"
    for path in (transformer, text_encoder, vae):
        path.write_bytes(path.name.encode())
    support.mkdir()
    (support / "model_index.json").write_text("{}", encoding="utf-8")
    identities = {
        role: ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, role)
        for role, path in {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": vae,
        }.items()
    }
    request = Klein4RuntimeRequest(
        1,
        "klein4b",
        "distilled",
        "flux2-klein-4b-distilled",
        4,
        1.0,
        {
            "pipeline_support": {"resource_id": "support", "path": str(support.resolve())},
            "transformer": {
                "resource_id": "nvfp4-transformer",
                "path": str(transformer.resolve()),
                "quantization_contract": "comfy_quant/nvfp4_tensorcore",
            },
            "text_encoder": {"resource_id": "qwen", "path": str(text_encoder.resolve())},
            "vae": {"resource_id": "vae", "path": str(vae.resolve())},
        },
        identities,
        SimpleNamespace(root=support),
        {},
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.revalidate_klein4_runtime_request",
        lambda value: value is request,
    )

    plan = resolve_klein_runtime_plan(
        settings,
        "klein4b",
        ExecutionPlan(
            variant_key="flux2-klein-4b.text-to-image.bfl-distilled-nvfp4",
            family="klein4b",
            optimizations=_optimizations(
                quantization="nvfp4", offload="staged", attention="native"
            ),
            recipe=request,
        ),
    )

    assert plan.quantization == "nvfp4"
    assert plan.model_precision == "fp4"
    assert plan.model_quantization == "nvfp4"
    assert plan.model_resource_id == "nvfp4-transformer"
    assert dict(plan.pipeline_parameters)["steps"] == 4


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_klein9_typed_recipe_resolves_stored_runtime_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: str
):
    settings = _settings(tmp_path)
    transformer = tmp_path / f"transformer-{quantization}.safetensors"
    text_encoder = tmp_path / "qwen-8b-mixed.safetensors"
    vae = tmp_path / "small-vae.safetensors"
    support = tmp_path / "support-9b"
    for path in (transformer, text_encoder, vae):
        path.write_bytes(path.name.encode())
    support.mkdir()
    (support / "model_index.json").write_text("{}", encoding="utf-8")
    identities = {
        role: ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, role)
        for role, path in {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": vae,
        }.items()
    }
    request = Klein4RuntimeRequest(
        1,
        "klein9b",
        "distilled",
        "flux2-klein-9b-distilled",
        4,
        1.0,
        {
            "pipeline_support": {"resource_id": "support", "path": str(support.resolve())},
            "transformer": {
                "resource_id": f"{quantization}-transformer",
                "path": str(transformer.resolve()),
                "quantization_contract": (
                    "comfy_quant/nvfp4_tensorcore"
                    if quantization == "nvfp4"
                    else "comfy_quant/float8_e4m3fn_global"
                ),
            },
            "text_encoder": {
                "resource_id": "qwen-8b-mixed",
                "path": str(text_encoder.resolve()),
            },
            "vae": {"resource_id": "small-vae", "path": str(vae.resolve())},
        },
        identities,
        SimpleNamespace(root=support),
        {},
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.revalidate_klein4_runtime_request",
        lambda value: value is request,
    )

    plan = resolve_klein_runtime_plan(
        settings,
        "klein9b",
        ExecutionPlan(
            variant_key=f"flux2-klein-9b.text-to-image.bfl-distilled-{quantization}",
            family="klein9b",
            optimizations=_optimizations(
                quantization=quantization,
                offload="staged",
                attention="native",
            ),
            recipe=request,
        ),
    )

    assert plan.family == "klein9b"
    assert plan.quantization == quantization
    assert plan.model_resource_id == f"{quantization}-transformer"
    assert plan.component_path("text_encoder") == text_encoder.resolve()
    assert plan.component_path("vae") == vae.resolve()
    assert dict(plan.pipeline_parameters) == {
        "guidance_scale": 1.0,
        "recipe_fingerprint": request.fingerprint,
        "recipe_mode": "distilled",
        "steps": 4,
    }


def test_klein9_rejects_standalone_nvfp4_override(tmp_path: Path):
    tool = KleinTextToImageTool()
    request = ExecutionRequest(
        family="klein9b",
        model_override=True,
        model_formats=frozenset({"safetensors"}),
        optimizations=_optimizations(quantization="nvfp4", offload="staged"),
    )

    reasons = tool.validate_execution_request(request)

    assert any("requires its typed component recipe" in reason for reason in reasons)


def test_installed_nvfp4_recipe_validation_uses_exact_nvfp4_schema(
    tmp_path: Path, monkeypatch
):
    paths = {role: tmp_path / f"{role}.safetensors" for role in ("transformer", "text_encoder", "vae")}
    for path in paths.values():
        path.write_bytes(path.name.encode())
    support = tmp_path / "support"
    support.mkdir()
    descriptors = {
        "transformer": ResourceDescriptor(
            id="model:klein4b:nvfp4",
            kind=ResourceKind.MODEL,
            family="klein4b",
            name="NVFP4",
            relative_path="transformer.safetensors",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.FP4,
            quantization=ArtifactQuantization.NVFP4,
            size_bytes=paths["transformer"].stat().st_size,
            base_model="flux2-klein-4b-distilled",
            component="transformer",
            metadata={
                "architecture": "flux2_klein_4b_distilled",
                "quantization_contract": "comfy_quant/nvfp4_tensorcore",
                "schema_sha256": "c6683e31192ed861a3068673e41d89555caacdad2e4a3a7357e5e576dcaea9d6",
            },
        ),
        "text_encoder": ResourceDescriptor(
            id="model:klein4b:qwen",
            kind=ResourceKind.MODEL,
            family="klein4b",
            name="Qwen",
            relative_path="text_encoder.safetensors",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.BF16,
            quantization=ArtifactQuantization.NATIVE,
            size_bytes=paths["text_encoder"].stat().st_size,
            component="text_encoder",
            metadata={"architecture": "qwen3_4b", "quantization_contract": "native/bf16"},
        ),
        "vae": ResourceDescriptor(
            id="model:klein4b:vae",
            kind=ResourceKind.MODEL,
            family="klein4b",
            name="VAE",
            relative_path="vae.safetensors",
            format=ResourceFormat.SAFETENSORS,
            precision=ArtifactPrecision.FP32,
            quantization=ArtifactQuantization.NATIVE,
            size_bytes=paths["vae"].stat().st_size,
            component="vae",
            metadata={"architecture": "flux2_vae", "quantization_contract": "native/fp32"},
        ),
        "pipeline_support": ResourceDescriptor(
            id="model:klein4b:support",
            kind=ResourceKind.MODEL,
            family="klein4b",
            name="Support",
            relative_path="support",
            format=ResourceFormat.DIRECTORY,
            precision=ArtifactPrecision.BF16,
            quantization=ArtifactQuantization.NATIVE,
            size_bytes=0,
            component="pipeline_support",
        ),
    }
    inventory = ResourceInventory(
        list(descriptors.values()),
        {**paths, "pipeline_support": support},
    )
    # Inventory paths are keyed by resource ID.
    inventory.paths = {
        descriptors[role].id: path
        for role, path in {**paths, "pipeline_support": support}.items()
    }
    identities = {
        role: ArtifactIdentity(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns, role)
        for role, path in paths.items()
    }
    exact_schema = "c6683e31192ed861a3068673e41d89555caacdad2e4a3a7357e5e576dcaea9d6"
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.probe_artifact",
        lambda path: SimpleNamespace(schema_sha256=exact_schema),
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.plan_klein_pipeline_support",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        klein_stored_adapter,
        "plan_bfl_klein_nvfp4_transformer",
        lambda _path, _config: SimpleNamespace(
            identity=identities["transformer"], require_available=lambda: None
        ),
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.plan_klein_text_encoder",
        lambda _path: SimpleNamespace(identity=identities["text_encoder"]),
    )
    monkeypatch.setattr(
        "latentslate_engine.klein_recipe.plan_klein_vae",
        lambda _path: SimpleNamespace(identity=identities["vae"]),
    )
    recipe = KleinStoredRecipe(
        mode="distilled",
        base_model="flux2-klein-4b-distilled",
        steps=4,
        guidance_scale=1.0,
        **{
            role: Klein4RecipeComponent(descriptors[role], path)
            for role, path in {**paths, "pipeline_support": support}.items()
        },
    )

    validation = validate_klein_stored_recipe(recipe, inventory)

    assert validation.available, validation.errors


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
        tool._generate(
            context,
            {"prompt": "x", "width": 512, "height": 512, "seed": 1},
            source_assets=[],
        )

    assert evicted == [runtime]
    assert {"runtime_evicted_after_residency_failure": "klein:poisoned"} in recorded


def test_klein_tool_records_native_and_lora_dispatch_provenance(tmp_path: Path, monkeypatch):
    tool = Klein4BTextToImageTool()
    plan = SimpleNamespace(keep_pipeline_loaded=True, provenance=lambda: {"kind": "plan"})
    dispatch = {"status": "proven", "module_count": 4, "total_dispatch_delta": 16}
    runtime = SimpleNamespace(
        generate=lambda **_kwargs: {
            "pipeline_fingerprint": "runtime:klein4b:test",
            "cache": {"pipeline_warm": False},
            "pipeline_kit": {"stored_weight_contract": "comfy_quant/nvfp4_tensorcore"},
            "quantized_dispatch": dispatch,
            "text_encoder_quantized_dispatch": None,
            "residency_policy": {"mode": "full"},
            "reference_preprocessing": {"ordered": True},
            "loras": {"active": ["lora:klein4b:test"]},
            "lora_dispatch": dispatch,
        },
        residency_poisoned=lambda: False,
    )
    recorded: list[dict] = []
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

    tool._generate(
        context,
        {"prompt": "x", "width": 512, "height": 512, "seed": 1},
        source_assets=[],
    )

    result = next(entry["runtime_result"] for entry in recorded if "runtime_result" in entry)
    assert result["quantized_dispatch"] == dispatch
    assert result["lora_dispatch"] == dispatch
