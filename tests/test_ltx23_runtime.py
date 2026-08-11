import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import WorkflowKind
from latentslate_engine.runtime import diffusers_repository as repository_contracts
from latentslate_engine.runtime import ltx23 as ltx23_runtime
from latentslate_engine.runtime.diffusers_repository import LTX23_REPOSITORY_CONTRACT
from latentslate_engine.runtime.ltx23 import (
    LTX23_GUIDANCE_SCALE,
    LTX23_MAX_FRAMES,
    LTX23_MIN_FRAMES,
    LTX23_SIZE_PRESETS,
    LTX23_STEPS,
    frames_for_duration,
    resolve_ltx23_runtime_plan,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import ltx23 as ltx23_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest


@pytest.fixture(autouse=True)
def _use_synthetic_ltx23_contract(monkeypatch: pytest.MonkeyPatch):
    weights = []
    for component in LTX23_REPOSITORY_CONTRACT.weights:
        config = (
            {"architectures": [component.class_name]}
            if component.transformers_component
            else {"_class_name": component.class_name}
        )
        schema = {
            f"tensor_{index}": (dtype, (1,))
            for index, dtype in enumerate(sorted(component.required_dtypes))
        }
        weights.append(
            replace(
                component,
                schema_sha256=repository_contracts._schema_fingerprint(schema),
                config_sha256=repository_contracts._semantic_fingerprint(config),
            )
        )
    contract = replace(
        LTX23_REPOSITORY_CONTRACT,
        weights=tuple(weights),
        file_fingerprints=tuple(
            (
                relative,
                len(payload := (b"{}" if Path(relative).suffix == ".json" else b"fixture")),
                hashlib.sha256(payload).hexdigest(),
            )
            for relative in LTX23_REPOSITORY_CONTRACT.required_files
            if not relative.endswith("scheduler_config.json")
        ),
        json_fingerprints=(
            (
                "scheduler/scheduler_config.json",
                repository_contracts._semantic_fingerprint(
                    {"_class_name": "FlowMatchEulerDiscreteScheduler"}
                ),
            ),
        ),
    )
    monkeypatch.setattr(ltx23_runtime, "LTX23_REPOSITORY_CONTRACT", contract)


def _write_safetensors(
    path: Path,
    dtypes: frozenset[str],
    *,
    name_prefix: str = "tensor",
) -> None:
    widths = {"BF16": 2, "F32": 4}
    offset = 0
    header = {}
    payload = bytearray()
    for index, dtype in enumerate(sorted(dtypes)):
        width = widths[dtype]
        header[f"{name_prefix}_{index}"] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + width],
        }
        payload.extend(b"\0" * width)
        offset += width
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _write_ltx23_repository(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_index = {
        "_class_name": LTX23_REPOSITORY_CONTRACT.root_class,
        **{
            name: [library, class_name]
            for name, library, class_name in LTX23_REPOSITORY_CONTRACT.components
        },
    }
    (path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    for relative in LTX23_REPOSITORY_CONTRACT.required_files:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if target.suffix == ".json" else "fixture", encoding="utf-8")
    for component in LTX23_REPOSITORY_CONTRACT.weights:
        component_root = path / component.name
        component_root.mkdir(parents=True, exist_ok=True)
        config = (
            {"architectures": [component.class_name]}
            if component.transformers_component
            else {"_class_name": component.class_name}
        )
        (component_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        _write_safetensors(
            component_root / f"{component.weight_stem}.safetensors",
            component.required_dtypes,
        )
    (path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}), encoding="utf-8"
    )


def _settings(tmp_path: Path, *, profile: str = "bf16_sequential_offload") -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
        ltx23_profile=profile,
    )
    settings.ensure_directories()
    model = settings.model_root / "ltx23" / "diffusers--LTX-2.3-Distilled-Diffusers"
    _write_ltx23_repository(model)
    return settings


def test_ltx23_frame_counts_follow_temporal_contract():
    for duration in (1.0, 2.0, 5.0, 10.0, 20.0):
        frames = frames_for_duration(duration)
        assert frames % 8 == 1
        assert LTX23_MIN_FRAMES <= frames <= LTX23_MAX_FRAMES


def test_ltx23_tool_follows_latentslate_taxonomy():
    descriptor = ltx23_tools.LTX23TextToVideoTool().descriptor

    assert descriptor.name == "Text to Video"
    assert descriptor.key == "ltx23.text_to_video"
    assert descriptor.workflow_kind == WorkflowKind.TEXT_TO_VIDEO
    assert descriptor.inputs[1].default == "768x512"
    assert descriptor.inputs[2].default == 5.0
    assert {option.value for option in descriptor.inputs[1].options} == set(LTX23_SIZE_PRESETS)


def test_ltx23_bundle_and_defaults_use_converted_distilled_checkpoint(tmp_path):
    model_id = "diffusers/LTX-2.3-Distilled-Diffusers"
    assert BUNDLES["ltx23-basic"].repo_id == model_id
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    assert settings.ltx23_model_id == model_id
    assert settings.ltx23_profile == "bf16_sequential_offload"
    assert LTX23_STEPS == 8
    assert LTX23_GUIDANCE_SCALE == 1.0


def test_ltx23_plan_binds_complete_native_bf16_folder_and_fingerprint(tmp_path: Path):
    settings = _settings(tmp_path)
    default = resolve_ltx23_runtime_plan(settings, None)
    selected_path = settings.model_root / "ltx23" / "local-bf16"
    _write_ltx23_repository(selected_path)
    selected = resolve_ltx23_runtime_plan(
        settings,
        ExecutionPlan(
            variant_key="ltx23.local-bf16",
            family="ltx23",
            model_resource_id="model:ltx23:local-bf16",
            model_path=selected_path,
            model_format="diffusers",
            model_precision="bf16",
            model_quantization="native",
            optimizations={"quantization": "bf16", "offload": "model"},
        ),
    )

    assert default.model_path != selected.model_path
    assert selected.model_path == selected_path.resolve()
    assert selected.model_resource_id == "model:ltx23:local-bf16"
    assert selected.offload == "model"
    assert selected.pipeline_fingerprint != default.pipeline_fingerprint
    model_provenance = selected.provenance()["model"]
    assert model_provenance["id"] == "model:ltx23:local-bf16"
    assert model_provenance["format"] == "diffusers"
    assert model_provenance["precision"] == "bf16"
    assert model_provenance["quantization"] == "native"
    assert model_provenance["override"] is True
    assert model_provenance["components"][0]["name"] == "model"


def test_ltx23_plan_rejects_non_bf16_or_incomplete_selected_artifacts(tmp_path: Path):
    settings = _settings(tmp_path)
    complete = settings.model_root / "ltx23" / "local"
    _write_ltx23_repository(complete)
    unsupported = ExecutionPlan(
        variant_key="ltx23.gguf",
        family="ltx23",
        model_resource_id="model:ltx23:gguf",
        model_path=complete,
        model_format="gguf",
        model_precision="bf16",
        model_quantization="native",
        optimizations={"quantization": "bf16"},
    )
    try:
        resolve_ltx23_runtime_plan(settings, unsupported)
    except ValueError as exc:
        assert "complete Diffusers directories only" in str(exc)
    else:
        raise AssertionError("GGUF selection was accepted without an LTX loader")

    int8 = ExecutionPlan(
        variant_key="ltx23.int8",
        family="ltx23",
        model_resource_id="model:ltx23:int8",
        model_path=complete,
        model_format="diffusers",
        model_precision="unknown",
        model_quantization="int8",
        optimizations={"quantization": "int8"},
    )
    try:
        resolve_ltx23_runtime_plan(settings, int8)
    except ValueError as exc:
        assert "only a native BF16 artifact" in str(exc)
    else:
        raise AssertionError("INT8 selection was accepted without an LTX loader")

    incomplete = settings.model_root / "ltx23" / "incomplete"
    incomplete.mkdir()
    execution = ExecutionPlan(
        variant_key="ltx23.incomplete",
        family="ltx23",
        model_resource_id="model:ltx23:incomplete",
        model_path=incomplete,
        model_format="diffusers",
        model_precision="bf16",
        model_quantization="native",
        optimizations={"quantization": "bf16"},
    )
    try:
        resolve_ltx23_runtime_plan(settings, execution)
    except ValueError as exc:
        assert "Required repository file is missing" in str(exc)
    else:
        raise AssertionError("incomplete Diffusers directory was accepted")

    unannotated = ExecutionPlan(
        variant_key="ltx23.unannotated",
        family="ltx23",
        model_resource_id="model:ltx23:unannotated",
        model_path=complete,
        optimizations={"quantization": "bf16"},
    )
    try:
        resolve_ltx23_runtime_plan(settings, unannotated)
    except ValueError as exc:
        assert "must explicitly declare" in str(exc)
    else:
        raise AssertionError("unannotated selected folder was accepted")

    wrong_class = settings.model_root / "ltx23" / "wrong-class"
    _write_ltx23_repository(wrong_class)
    (wrong_class / "model_index.json").write_text(
        json.dumps({"_class_name": "UnrelatedPipeline"}), encoding="utf-8"
    )
    try:
        resolve_ltx23_runtime_plan(
            settings,
            ExecutionPlan(
                variant_key="ltx23.wrong-class",
                family="ltx23",
                model_path=wrong_class,
                model_format="diffusers",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
        )
    except ValueError as exc:
        assert "LTX2Pipeline" in str(exc)
    else:
        raise AssertionError("wrong LTX root class was accepted")


def test_ltx23_runtime_revalidates_repository_before_first_load(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_tools.LTX23Runtime(settings, plan)
    (plan.model_path / "tokenizer" / "tokenizer_config.json").write_text(
        '{"changed":true}', encoding="utf-8"
    )

    try:
        runtime._load_pipeline()
    except RuntimeError as exc:
        assert "changed after planning" in str(exc)
    else:
        raise AssertionError("mutated LTX repository was loaded under a stale plan")


def test_ltx23_plan_rejects_corrupt_tokenizer_payload(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "ltx23" / BUNDLES["ltx23-basic"].repo_id.replace("/", "--")
    (model / "tokenizer" / "tokenizer.model").write_bytes(b"fixture-corrupt")

    with pytest.raises(ValueError, match="support file differs"):
        resolve_ltx23_runtime_plan(settings, None)


def test_ltx23_plan_rejects_shard_index_that_omits_payload_tensor(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "ltx23" / "wrong-index"
    _write_ltx23_repository(model)
    component = LTX23_REPOSITORY_CONTRACT.weights[0]
    component_root = model / component.name
    (component_root / f"{component.weight_stem}.safetensors").unlink()
    shard_name = f"{component.weight_stem}-00001-of-00001.safetensors"
    _write_safetensors(component_root / shard_name, component.required_dtypes)
    (component_root / f"{component.weight_stem}.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"unrelated": shard_name}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="shard index does not exactly describe"):
        resolve_ltx23_runtime_plan(
            settings,
            ExecutionPlan(
                variant_key="ltx23.wrong-index",
                family="ltx23",
                model_path=model,
                model_format="diffusers",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
        )


def test_ltx23_advertises_only_complete_bf16_diffusers_execution(monkeypatch):
    monkeypatch.setattr(ltx23_tools, "_runtime_availability", lambda: (True, None))
    tool = ltx23_tools.LTX23TextToVideoTool()
    capabilities = tool.execution_capabilities()
    assert capabilities.model_formats == frozenset({"diffusers"})
    assert capabilities.quantization_modes == frozenset({"bf16"})
    assert capabilities.lora_formats == frozenset()
    errors = tool.validate_execution_request(
        ExecutionRequest(
            family="ltx23",
            model_override=True,
            model_formats=frozenset({"gguf"}),
            optimizations={"quantization": "gguf"},
        )
    )
    assert any("model override formats" in error for error in errors)
    assert any("quantization mode 'gguf'" in error for error in errors)


def test_ltx23_runtime_is_reused_per_resolved_model_selection(tmp_path: Path, monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings, plan):
            self.settings = settings
            self.plan = plan
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ltx23_tools, "LTX23Runtime", FakeRuntime)
    settings = _settings(tmp_path)
    selected_path = settings.model_root / "ltx23" / "selected"
    _write_ltx23_repository(selected_path)
    context = SimpleNamespace(settings=settings, execution=None)
    selected_context = SimpleNamespace(
        settings=settings,
        execution=ExecutionPlan(
            variant_key="ltx23.selected",
            family="ltx23",
            model_resource_id="model:ltx23:selected",
            model_path=selected_path,
            model_format="diffusers",
            model_precision="bf16",
            model_quantization="native",
            optimizations={"quantization": "bf16"},
        ),
    )
    tool = ltx23_tools.LTX23TextToVideoTool()

    first_plan = tool._resolve_plan(context)
    first = tool._runtime(context, first_plan)
    second = tool._runtime(context, first_plan)
    selected_plan = tool._resolve_plan(selected_context)
    selected = tool._runtime(selected_context, selected_plan)

    assert first is second
    assert created == [first, selected]
    assert selected.plan.model_path == selected_path.resolve()
    assert selected.plan.pipeline_fingerprint != first.plan.pipeline_fingerprint
    RUNTIME_MANAGER.clear()
