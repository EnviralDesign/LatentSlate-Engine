import hashlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.protocol import InputRole
from latentslate_engine.runtime import diffusers_repository as repository_contracts
from latentslate_engine.runtime import h3 as h3_runtime
from latentslate_engine.runtime.diffusers_repository import H3_REPOSITORY_CONTRACT
from latentslate_engine.runtime.h3 import (
    H3_MAX_DURATION_SECONDS,
    H3_MAX_FRAMES,
    H3_MAX_PIXELS,
    H3_MIN_FRAMES,
    frames_for_duration,
    resolve_h3_dimensions,
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


def _install_fake_h3_modules(monkeypatch: pytest.MonkeyPatch, calls: list[tuple]) -> None:
    class FakeGenerator:
        def __init__(self, device: str):
            self.device = device

        def manual_seed(self, seed: int):
            calls.append(("seed", self.device, seed))
            return self

    fake_torch = ModuleType("torch")
    fake_torch.Generator = FakeGenerator
    fake_torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeComponentsManager:
        def enable_auto_cpu_offload(self, *, device: str) -> None:
            calls.append(("offload", device))

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("from_pretrained", Path(path), kwargs))
            return cls()

        def load_components(self, **kwargs) -> None:
            calls.append(("load_components", kwargs))

        def __call__(self, **kwargs):
            calls.append(("generate", kwargs))
            return {
                "videos": ["video-frames"],
                "audio": ["audio-samples"],
                "sampling_rate": 48_000,
            }

        def remove_all_hooks(self) -> None:
            calls.append(("remove_all_hooks",))

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.ComponentsManager = FakeComponentsManager
    fake_diffusers.ModularPipeline = FakePipeline
    fake_utils = ModuleType("diffusers.utils")

    def load_image(path: str) -> str:
        calls.append(("load_image", path))
        return f"image:{path}"

    fake_utils.load_image = load_image
    fake_export_utils = ModuleType("diffusers.utils.export_utils")

    def encode_video(video, **kwargs) -> None:
        calls.append(("encode_video", video, kwargs))

    fake_export_utils.encode_video = encode_video
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "diffusers.utils.export_utils", fake_export_utils)


def test_h3_duration_alignment_stays_inside_model_limits():
    assert frames_for_duration(5.0) == H3_MIN_FRAMES
    assert frames_for_duration(H3_MAX_DURATION_SECONDS) == H3_MAX_FRAMES
    assert frames_for_duration(15.0) == H3_MAX_FRAMES


def test_h3_frame_counts_follow_vae_contract():
    for duration in (5.0, 7.0, 10.0, 14.0, 15.0):
        frames = frames_for_duration(duration)
        assert frames % 17 == 5
        assert H3_MIN_FRAMES <= frames <= H3_MAX_FRAMES


def test_h3_tools_expose_granular_canvas_and_explicit_legacy_step_policy():
    for tool in (h3_tools.H3TextToVideoTool(), h3_tools.H3FirstLastFrameTool()):
        descriptor = tool.descriptor
        inputs = {item.key: item for item in descriptor.inputs}

        assert descriptor.schema_revision == 2
        assert "quality" not in inputs
        assert (inputs["width"].default, inputs["height"].default) == (960, 544)
        assert inputs["width"].role == InputRole.WIDTH
        assert inputs["height"].role == InputRole.HEIGHT
        assert inputs["steps"].default == 20
        assert (inputs["steps"].ui.min, inputs["steps"].ui.max) == (1, 30)


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (64, None, "supplied together"),
        (None, None, "required"),
        (32, 64, "at least 64"),
        (64, 288, "1:4 to 4:1"),
        (1376, 768, "pixel budget"),
    ],
)
def test_h3_dimension_contract_rejects_invalid_canvases(width, height, message):
    with pytest.raises((TypeError, ValueError), match=message):
        resolve_h3_dimensions(width, height)

    assert 1344 * 768 == H3_MAX_PIXELS
    assert resolve_h3_dimensions(1344, 768).width == 1344


def test_h3_dimension_contract_normalizes_to_the_effective_32_pixel_canvas():
    dimensions = resolve_h3_dimensions(849, 495)

    assert dimensions.metadata() == {
        "requested_dimensions": {"width": 849, "height": 495},
        "effective_dimensions": {"width": 864, "height": 480},
    }


def test_h3_tools_keep_workflows_in_distinct_runtimes_and_unload_on_switch(tmp_path, monkeypatch):
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
    text_tool = h3_tools.H3TextToVideoTool()
    keyframe_tool = h3_tools.H3FirstLastFrameTool()
    text_plan = text_tool._resolve_plan(context)
    keyframe_plan = keyframe_tool._resolve_plan(context)

    text_runtime = text_tool._runtime(context, text_plan)
    assert text_tool._runtime(context, text_plan) is text_runtime
    keyframe_runtime = keyframe_tool._runtime(context, keyframe_plan)

    assert text_plan.pipeline_parameters == (("workflow", "t2va"),)
    assert keyframe_plan.pipeline_parameters == (("workflow", "fl2va"),)
    assert text_plan.pipeline_fingerprint != keyframe_plan.pipeline_fingerprint
    assert text_runtime is not keyframe_runtime
    assert text_runtime.unloaded is True
    assert created == [text_runtime, keyframe_runtime]
    RUNTIME_MANAGER.clear()


def test_h3_pipeline_parameters_are_immutable_and_keep_their_fingerprint(tmp_path):
    plan = resolve_h3_runtime_plan(_settings(tmp_path), None, workflow="t2va")
    fingerprint = plan.pipeline_fingerprint

    with pytest.raises(TypeError):
        plan.pipeline_parameters[0] = ("workflow", "fl2va")

    attempted_mutation = dict(plan.pipeline_parameters)
    attempted_mutation["workflow"] = "fl2va"
    assert plan.pipeline_parameters == (("workflow", "t2va"),)
    assert plan.pipeline_fingerprint == fingerprint


def test_h3_loads_the_modular_pipeline_for_the_selected_workflow(tmp_path, monkeypatch):
    calls = []
    settings = _settings(tmp_path)
    text_plan = resolve_h3_runtime_plan(settings, None, workflow="t2va")
    first_last_plan = resolve_h3_runtime_plan(settings, None, workflow="fl2va")
    _install_fake_h3_modules(monkeypatch, calls)

    text_runtime = h3_runtime.H3Runtime(
        settings,
        text_plan,
    )
    first_last_runtime = h3_runtime.H3Runtime(
        settings,
        first_last_plan,
    )
    text_runtime._load_pipeline()
    first_last_runtime._load_pipeline()

    assert [call[2]["workflow"] for call in calls if call[0] == "from_pretrained"] == [
        "t2va",
        "fl2va",
    ]
    assert [call for call in calls if call[0] == "offload"] == [
        ("offload", "cuda"),
        ("offload", "cuda"),
    ]


def test_h3_text_generation_never_enters_the_keyframe_path(tmp_path, monkeypatch):
    calls = []
    settings = _settings(tmp_path)
    plan = resolve_h3_runtime_plan(settings, None, workflow="t2va")
    _install_fake_h3_modules(monkeypatch, calls)
    runtime = h3_runtime.H3Runtime(settings, plan)

    metadata = runtime.generate(
        plan=plan,
        prompt="A thunderstorm over a lake",
        output_path=tmp_path / "text.mp4",
        width=849,
        height=495,
        steps=20,
        duration_seconds=5.0,
        seed=7,
        image_path=None,
        last_image_path=None,
        progress=lambda _progress, _message: None,
        check_cancelled=lambda: None,
    )

    generated = next(call[1] for call in calls if call[0] == "generate")
    assert generated["prompt"] == "A thunderstorm over a lake"
    assert generated["width"] == 864
    assert generated["height"] == 480
    assert generated["num_inference_steps"] == 20
    assert metadata["requested_dimensions"] == {"width": 849, "height": 495}
    assert metadata["effective_dimensions"] == {"width": 864, "height": 480}
    assert "image" not in generated
    assert "last_image" not in generated
    assert not [call for call in calls if call[0] == "load_image"]


def test_h3_first_last_generation_uses_start_and_optional_end_images(tmp_path, monkeypatch):
    calls = []
    settings = _settings(tmp_path)
    plan = resolve_h3_runtime_plan(settings, None, workflow="fl2va")
    _install_fake_h3_modules(monkeypatch, calls)
    runtime = h3_runtime.H3Runtime(settings, plan)
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"

    runtime.generate(
        plan=plan,
        prompt="A flower opens",
        output_path=tmp_path / "first-only.mp4",
        width=832,
        height=480,
        steps=16,
        duration_seconds=5.0,
        seed=8,
        image_path=start,
        last_image_path=None,
        progress=lambda _progress, _message: None,
        check_cancelled=lambda: None,
    )
    runtime.generate(
        plan=plan,
        prompt="A flower closes",
        output_path=tmp_path / "first-last.mp4",
        width=960,
        height=544,
        steps=30,
        duration_seconds=5.0,
        seed=9,
        image_path=start,
        last_image_path=end,
        progress=lambda _progress, _message: None,
        check_cancelled=lambda: None,
    )

    generated = [call[1] for call in calls if call[0] == "generate"]
    assert generated[0]["image"] == f"image:{start}"
    assert "last_image" not in generated[0]
    assert generated[1]["image"] == f"image:{start}"
    assert generated[1]["last_image"] == f"image:{end}"


@pytest.mark.parametrize(
    ("width", "height", "steps", "message"),
    [(1376, 768, 16, "pixel budget"), (832, 480, 31, "steps")],
)
def test_h3_rejects_canvas_and_step_budget_before_pipeline_load(
    tmp_path,
    monkeypatch,
    width,
    height,
    steps,
    message,
):
    settings = _settings(tmp_path)
    plan = resolve_h3_runtime_plan(settings, None, workflow="t2va")
    runtime = h3_runtime.H3Runtime(settings, plan)
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        runtime.generate(
            plan=plan,
            prompt="invalid request",
            output_path=tmp_path / "no-output.mp4",
            width=width,
            height=height,
            steps=steps,
            duration_seconds=5.0,
            seed=0,
            image_path=None,
            last_image_path=None,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )


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
