import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.runtime import diffusers_repository as repository_contracts
from latentslate_engine.runtime import h3 as h3_runtime
from latentslate_engine.runtime.diffusers_repository import H3_REPOSITORY_CONTRACT
from latentslate_engine.runtime.h3 import (
    H3_MAX_DURATION_SECONDS,
    H3_MAX_FRAMES,
    H3_MIN_FRAMES,
    frames_for_duration,
    resolve_h3_runtime_plan,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import h3 as h3_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest


@pytest.fixture(autouse=True)
def _use_synthetic_h3_contract(monkeypatch: pytest.MonkeyPatch):
    weights = []
    for component in H3_REPOSITORY_CONTRACT.weights:
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
        H3_REPOSITORY_CONTRACT,
        weights=tuple(weights),
        file_fingerprints=tuple(
            (
                relative,
                len(payload := (b"{}" if Path(relative).suffix == ".json" else b"fixture")),
                hashlib.sha256(payload).hexdigest(),
            )
            for relative in H3_REPOSITORY_CONTRACT.required_files
            if not relative.endswith("scheduler_config.json")
        ),
        json_fingerprints=(
            (
                "scheduler/scheduler_config.json",
                repository_contracts._semantic_fingerprint({"_class_name": "MiniMaxH3Scheduler"}),
            ),
            (
                "audio_scheduler/scheduler_config.json",
                repository_contracts._semantic_fingerprint({"_class_name": "MiniMaxH3Scheduler"}),
            ),
        ),
    )
    monkeypatch.setattr(h3_runtime, "H3_REPOSITORY_CONTRACT", contract)


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


def _write_h3_repository(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_index = {
        "_class_name": H3_REPOSITORY_CONTRACT.root_class,
        **{
            name: [library, class_name]
            for name, library, class_name in H3_REPOSITORY_CONTRACT.components
        },
    }
    for relative in ("model_index.json", *H3_REPOSITORY_CONTRACT.mirrored_model_indexes):
        (path / relative).write_text(json.dumps(model_index), encoding="utf-8")
    for relative in H3_REPOSITORY_CONTRACT.required_files:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if target.suffix == ".json" else "fixture", encoding="utf-8")
    for component in H3_REPOSITORY_CONTRACT.weights:
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
    for name in ("scheduler", "audio_scheduler"):
        (path / name / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3Scheduler"}), encoding="utf-8"
        )


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="MiniMaxAI/MiniMax-H3",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "h3" / "MiniMaxAI--MiniMax-H3"
    _write_h3_repository(model)
    return settings


def test_h3_duration_alignment_stays_inside_model_limits():
    assert frames_for_duration(5.0) == H3_MIN_FRAMES
    assert frames_for_duration(H3_MAX_DURATION_SECONDS) == H3_MAX_FRAMES
    assert frames_for_duration(15.0) == H3_MAX_FRAMES


def test_h3_frame_counts_follow_vae_contract():
    for duration in (5.0, 7.0, 10.0, 14.0, 15.0):
        frames = frames_for_duration(duration)
        assert frames % 17 == 5
        assert H3_MIN_FRAMES <= frames <= H3_MAX_FRAMES


def test_h3_tools_share_one_runtime_for_the_same_settings(tmp_path, monkeypatch):
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
    monkeypatch.setattr(h3_tools, "H3Runtime", FakeRuntime)
    context = SimpleNamespace(settings=_settings(tmp_path), execution=None)
    plan = h3_tools.H3TextToVideoTool()._resolve_plan(context)

    text_runtime = h3_tools.H3TextToVideoTool()._runtime(context, plan)
    keyframe_runtime = h3_tools.H3FirstLastFrameTool()._runtime(context, plan)

    assert text_runtime is keyframe_runtime
    assert created == [text_runtime]
    RUNTIME_MANAGER.clear()


def test_h3_plan_binds_selected_complete_bf16_folder(tmp_path: Path):
    settings = _settings(tmp_path)
    default = resolve_h3_runtime_plan(settings, None)
    selected_path = settings.model_root / "h3" / "local-bf16"
    _write_h3_repository(selected_path)
    selected = resolve_h3_runtime_plan(
        settings,
        ExecutionPlan(
            variant_key="h3.local-bf16",
            family="h3",
            model_resource_id="model:h3:local-bf16",
            model_path=selected_path,
            model_format="diffusers",
            model_precision="bf16",
            model_quantization="native",
            optimizations={"quantization": "bf16"},
        ),
    )

    assert selected.model_path == selected_path.resolve()
    assert selected.model_resource_id == "model:h3:local-bf16"
    assert selected.pipeline_fingerprint != default.pipeline_fingerprint
    assert selected.provenance()["model"]["override"] is True


def test_h3_plan_rejects_non_bf16_or_incomplete_selected_artifacts(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "h3" / "selected"
    _write_h3_repository(model)

    for execution, message in (
        (
            ExecutionPlan(
                variant_key="h3.gguf",
                family="h3",
                model_path=model,
                model_format="gguf",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
            "complete Diffusers directories only",
        ),
        (
            ExecutionPlan(
                variant_key="h3.int8",
                family="h3",
                model_path=model,
                model_format="diffusers",
                model_precision="unknown",
                model_quantization="int8",
                optimizations={"quantization": "int8"},
            ),
            "only a native BF16 artifact",
        ),
        (
            ExecutionPlan(
                variant_key="h3.unannotated",
                family="h3",
                model_path=model,
                optimizations={"quantization": "bf16"},
            ),
            "must explicitly declare",
        ),
    ):
        try:
            resolve_h3_runtime_plan(settings, execution)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"unsupported H3 execution was accepted: {execution}")

    incomplete = settings.model_root / "h3" / "incomplete"
    incomplete.mkdir()
    try:
        resolve_h3_runtime_plan(
            settings,
            ExecutionPlan(
                variant_key="h3.incomplete",
                family="h3",
                model_path=incomplete,
                model_format="diffusers",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
        )
    except ValueError as exc:
        assert "Required repository file is missing" in str(exc)
    else:
        raise AssertionError("incomplete H3 directory was accepted")

    wrong_class = settings.model_root / "h3" / "wrong-class"
    _write_h3_repository(wrong_class)
    (wrong_class / "model_index.json").write_text(
        json.dumps({"_class_name": "UnrelatedPipeline"}), encoding="utf-8"
    )
    try:
        resolve_h3_runtime_plan(
            settings,
            ExecutionPlan(
                variant_key="h3.wrong-class",
                family="h3",
                model_path=wrong_class,
                model_format="diffusers",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
        )
    except ValueError as exc:
        assert "MiniMaxH3ModularPipeline" in str(exc)
    else:
        raise AssertionError("wrong H3 root class was accepted")


def test_h3_runtime_revalidates_repository_before_first_load(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = resolve_h3_runtime_plan(settings, None)
    runtime = h3_tools.H3Runtime(settings, plan)
    (plan.model_path / "tokenizer" / "tokenizer_config.json").write_text(
        '{"changed":true}', encoding="utf-8"
    )

    try:
        runtime._load_bf16_auto_offload()
    except RuntimeError as exc:
        assert "changed after planning" in str(exc)
    else:
        raise AssertionError("mutated H3 repository was loaded under a stale plan")


def test_h3_fl2va_bundle_does_not_require_ignored_reference_transformer(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "h3" / "MiniMaxAI--MiniMax-H3"

    assert not (model / "transformer_ref").exists()
    assert resolve_h3_runtime_plan(settings, None).model_path == model.resolve()


def test_h3_plan_rejects_corrupt_tokenizer_payload(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "h3" / "MiniMaxAI--MiniMax-H3"
    (model / "tokenizer" / "tokenizer.json").write_text("{} ", encoding="utf-8")

    with pytest.raises(ValueError, match="support file differs"):
        resolve_h3_runtime_plan(settings, None)


def test_h3_plan_rejects_same_dtype_incomplete_tensor_schema(tmp_path: Path):
    settings = _settings(tmp_path)
    model = settings.model_root / "h3" / "wrong-schema"
    _write_h3_repository(model)
    text_contract = H3_REPOSITORY_CONTRACT.weights[0]
    _write_safetensors(
        model / "text_encoder" / "model.safetensors",
        text_contract.required_dtypes,
        name_prefix="wrong",
    )

    with pytest.raises(ValueError, match="tensor schema is incomplete or incompatible"):
        resolve_h3_runtime_plan(
            settings,
            ExecutionPlan(
                variant_key="h3.wrong-schema",
                family="h3",
                model_path=model,
                model_format="diffusers",
                model_precision="bf16",
                model_quantization="native",
                optimizations={"quantization": "bf16"},
            ),
        )


def test_h3_advertises_only_complete_bf16_diffusers_execution(monkeypatch):
    monkeypatch.setattr(h3_tools, "_runtime_availability", lambda: (True, None))
    tool = h3_tools.H3TextToVideoTool()
    capabilities = tool.execution_capabilities()
    assert capabilities.model_formats == frozenset({"diffusers"})
    assert capabilities.quantization_modes == frozenset({"bf16"})
    assert capabilities.offload_modes == frozenset()
    errors = tool.validate_execution_request(
        ExecutionRequest(
            family="h3",
            model_override=True,
            model_formats=frozenset({"gguf"}),
            optimizations={"quantization": "gguf"},
        )
    )
    assert any("model override formats" in error for error in errors)
    assert any("quantization mode 'gguf'" in error for error in errors)
