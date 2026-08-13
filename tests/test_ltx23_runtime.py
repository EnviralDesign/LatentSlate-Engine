import hashlib
import inspect
import json
import struct
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import UUID

import pytest

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.ltx23_kitchen_recipe import LTX23KitchenRuntimeRequest
from latentslate_engine.protocol import InputRole, WorkflowKind
from latentslate_engine.runtime import diffusers_repository as repository_contracts
from latentslate_engine.runtime import ltx23 as ltx23_runtime
from latentslate_engine.runtime.diffusers_repository import LTX23_REPOSITORY_CONTRACT
from latentslate_engine.runtime.ltx23 import (
    LTX23_DISTILLED_SIGMAS,
    LTX23_GUIDANCE_SCALE,
    LTX23_MAX_FRAMES,
    LTX23_MIN_FRAMES,
    LTX23_STEPS,
    frames_for_duration,
    resolve_ltx23_runtime_plan,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.storage import Storage
from latentslate_engine.tools import ltx23 as ltx23_tools
from latentslate_engine.tools.base import ExecutionPlan, ExecutionRequest, ToolContext


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
    inputs = {item.key: item for item in descriptor.inputs}
    assert "size" not in inputs
    assert (inputs["width"].default, inputs["height"].default) == (768, 512)
    assert inputs["width"].role == InputRole.WIDTH
    assert inputs["height"].role == InputRole.HEIGHT
    assert inputs["duration_seconds"].default == 5.0
    provenance = ltx23_tools.LTX23TextToVideoTool().variant_provenance("ltx23_kitchen")
    assert provenance["engine_default_dimensions"] == [768, 512]
    assert provenance["pinned_workflow_default_dimensions"] == [1280, 720]
    assert "divisible by the two-stage /64" in provenance["dimension_default_deviation"]
    available, reason = ltx23_tools.LTX23TextToVideoTool().variant_recipe_availability(
        "ltx23_kitchen"
    )
    assert available is True
    assert reason is None


def test_ltx23_first_frame_tool_has_a_distinct_no_endpoint_schema():
    descriptor = ltx23_tools.LTX23FirstFrameToVideoTool().descriptor
    inputs = {item.key: item for item in descriptor.inputs}

    assert descriptor.name == "First Frame to Video"
    assert descriptor.key == "ltx23.image_to_video"
    assert descriptor.schema_revision == 2
    assert descriptor.workflow_kind == WorkflowKind.IMAGE_TO_VIDEO
    assert inputs["start_image"].role == InputRole.START_IMAGE
    assert inputs["start_image"].required is True
    assert "end_image" not in inputs
    assert not {"audio", "keyframes"} & set(inputs)


def test_ltx23_first_last_frame_tool_requires_both_endpoint_inputs():
    descriptor = ltx23_tools.LTX23ImageToVideoTool().descriptor
    inputs = {item.key: item for item in descriptor.inputs}

    assert descriptor.name == "First and Last Frame to Video"
    assert descriptor.key == "ltx23.first_last_frame_to_video"
    assert descriptor.workflow_kind == WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO
    assert inputs["start_image"].required is True
    assert inputs["end_image"].role == InputRole.END_IMAGE
    assert inputs["end_image"].required is True


def test_ltx23_condition_constructor_accepts_the_pinned_root_components():
    from diffusers.pipelines.ltx2 import LTX2ConditionPipeline

    constructor_components = set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)
    root_components = {name for name, _library, _class_name in LTX23_REPOSITORY_CONTRACT.components}
    assert root_components <= constructor_components


@pytest.mark.parametrize(
    ("include_end", "expected_indices"),
    [(False, [0]), (True, [0, -1])],
)
def test_ltx23_condition_runtime_passes_first_and_optional_last_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_end: bool,
    expected_indices: list[int],
):
    import diffusers.utils as diffusers_utils
    import torch
    from diffusers.pipelines import ltx2

    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_runtime.LTX23ConditionRuntime(settings, plan)
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"
    start.write_bytes(b"start")
    end.write_bytes(b"end")
    calls: dict[str, object] = {}

    class FakeCondition:
        def __init__(self, *, frames, index, strength):
            self.frames = frames
            self.index = index
            self.strength = strength

    class FakePipeline:
        _execution_device = "cpu"
        vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48_000))

        def encode_prompt(self, **_kwargs):
            return ("prompt", "prompt-mask", "negative", "negative-mask")

        def __call__(self, **kwargs):
            calls.update(kwargs)
            return (["video"], [torch.zeros(1)])

    loaded = []
    encoded = []
    monkeypatch.setattr(ltx2, "LTX2VideoCondition", FakeCondition)
    monkeypatch.setattr(diffusers_utils, "load_image", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        diffusers_utils, "encode_video", lambda *args, **kwargs: encoded.append((args, kwargs))
    )
    monkeypatch.setattr(
        runtime, "_load_pipeline", lambda: loaded.append(FakePipeline()) or loaded[-1]
    )

    metadata = runtime.generate(
        plan=plan,
        prompt="a scene with sound",
        output_path=tmp_path / "output.mp4",
        width=769,
        height=513,
        duration_seconds=1.0,
        seed=42,
        start_image_path=start,
        end_image_path=end if include_end else None,
        progress=lambda *_: None,
        check_cancelled=lambda: None,
    )

    conditions = calls["conditions"]
    assert [condition.index for condition in conditions] == expected_indices
    assert [condition.strength for condition in conditions] == [1.0] * len(expected_indices)
    assert calls["width"] == 768
    assert calls["height"] == 512
    assert calls["frame_rate"] == 24.0
    assert calls["num_inference_steps"] == 8
    assert calls["guidance_scale"] == 1.0
    assert calls["sigmas"] == LTX23_DISTILLED_SIGMAS
    assert metadata["effective_dimensions"] == {"width": 768, "height": 512}
    assert metadata["has_audio"] is True
    assert metadata["sigmas"] == list(LTX23_DISTILLED_SIGMAS)
    assert metadata["conditioning"] == {
        "mode": "first_last_frame" if include_end else "first_frame",
        "start_frame": True,
        "end_frame": include_end,
        "ordered_indices": [0, -1] if include_end else [0],
    }
    assert encoded[0][1]["audio_sample_rate"] == 48_000


def test_ltx23_denoise_callback_reports_progress_and_checks_cancellation_without_mutation():
    checks: list[str] = []
    progress: list[tuple[float, str | None]] = []
    payload = {"latents": object(), "prompt_embeds": object()}

    result = ltx23_runtime._denoise_callback(
        lambda: checks.append("checked"), lambda value, message: progress.append((value, message))
    )(object(), 3, object(), payload)

    assert checks == ["checked"]
    assert progress == [(0.5, "Generating synchronized video and audio (4/8)")]
    assert result is payload


@pytest.mark.parametrize("include_end", [False, True])
def test_ltx23_runtime_registers_a_denoise_cancellation_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_end: bool,
):
    import diffusers.utils as diffusers_utils
    import torch
    from diffusers.pipelines import ltx2

    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = (
        ltx23_runtime.LTX23ConditionRuntime(settings, plan)
        if include_end
        else ltx23_runtime.LTX23Runtime(settings, plan)
    )
    calls: dict[str, object] = {}

    class FakePipeline:
        _execution_device = "cpu"
        vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48_000))

        def encode_prompt(self, **_kwargs):
            return ("prompt", "prompt-mask", "negative", "negative-mask")

        def __call__(self, **kwargs):
            calls.update(kwargs)
            kwargs["callback_on_step_end"](self, 0, 0, {"latents": "unchanged"})
            return (["video"], [torch.zeros(1)])

    encoded = []
    monkeypatch.setattr(
        diffusers_utils, "encode_video", lambda *args, **kwargs: encoded.append((args, kwargs))
    )
    if include_end:
        start = tmp_path / "start.png"
        end = tmp_path / "end.png"
        start.write_bytes(b"start")
        end.write_bytes(b"end")
        monkeypatch.setattr(ltx2, "LTX2VideoCondition", lambda **kwargs: kwargs)
        monkeypatch.setattr(diffusers_utils, "load_image", lambda path: f"loaded:{path}")
        runtime._load_pipeline = lambda: FakePipeline()  # type: ignore[method-assign]
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "out.mp4",
            width=768,
            height=512,
            duration_seconds=1.0,
            seed=0,
            start_image_path=start,
            end_image_path=end,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )
    else:
        runtime._load_pipeline = lambda: FakePipeline()  # type: ignore[method-assign]
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "out.mp4",
            width=768,
            height=512,
            duration_seconds=1.0,
            seed=0,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )

    assert callable(calls["callback_on_step_end"])
    assert encoded


def test_ltx23_condition_rejects_missing_reference_before_pipeline_load(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_runtime.LTX23ConditionRuntime(settings, plan)
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    with pytest.raises(ValueError, match="start image does not exist"):
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "no-output.mp4",
            width=768,
            height=512,
            duration_seconds=1.0,
            seed=0,
            start_image_path=tmp_path / "missing.png",
            end_image_path=None,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )


def test_ltx23_condition_rejects_aligned_over_budget_canvas_before_pipeline_load(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_runtime.LTX23ConditionRuntime(settings, plan)
    start = tmp_path / "start.png"
    start.write_bytes(b"start")
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    with pytest.raises(ValueError, match="pixel budget"):
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "no-output.mp4",
            width=1280,
            height=753,
            duration_seconds=1.0,
            seed=0,
            start_image_path=start,
            end_image_path=None,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )


def test_ltx23_condition_pipeline_loads_the_same_complete_repository(tmp_path, monkeypatch):
    import torch
    from diffusers.pipelines import ltx2

    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_runtime.LTX23ConditionRuntime(settings, plan)
    calls = []

    class FakePipeline:
        def __init__(self):
            self.vae = SimpleNamespace(enable_tiling=lambda: calls.append("tiling"))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return cls()

        def enable_sequential_cpu_offload(self, *, device):
            calls.append(("sequential", device))

        def set_progress_bar_config(self, **kwargs):
            calls.append(("progress", kwargs))

    monkeypatch.setattr(ltx2, "LTX2ConditionPipeline", FakePipeline)

    pipeline = runtime._load_pipeline()

    assert isinstance(pipeline, FakePipeline)
    assert calls[0] == (
        (plan.model_path,),
        {"dtype": torch.bfloat16, "low_cpu_mem_usage": plan.low_cpu_mem_usage},
    )
    assert ("sequential", settings.ltx23_device) in calls
    assert "tiling" in calls


def test_ltx23_rejects_aligned_over_budget_canvas_before_loading_pipeline(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    plan = resolve_ltx23_runtime_plan(settings, None)
    runtime = ltx23_runtime.LTX23Runtime(settings, plan)
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    with pytest.raises(ValueError, match="pixel budget"):
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "no-output.mp4",
            width=1280,
            height=753,
            duration_seconds=1.0,
            seed=0,
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )


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
    runtime = ltx23_runtime.LTX23Runtime(settings, plan)
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


def test_ltx23_separates_reference_and_engine_native_kitchen_capabilities(monkeypatch):
    monkeypatch.setattr(ltx23_tools, "_runtime_availability", lambda: (True, None))
    tool = ltx23_tools.LTX23TextToVideoTool()
    capabilities = tool.execution_capabilities()
    assert capabilities.model_formats == frozenset({"diffusers"})
    assert capabilities.quantization_modes == frozenset({"bf16", "fp8"})
    assert capabilities.recipe_types == frozenset({"ltx23_kitchen"})
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
    assert not tool.validate_execution_request(
        ExecutionRequest(
            family="ltx23",
            recipe_type="ltx23_kitchen",
            optimizations={
                "attention": "native",
                "offload": "staged",
                "quantization": "fp8",
                "cache": "none",
                "keep_pipeline_loaded": False,
            },
        )
    )
    reference_errors = tool.validate_execution_request(
        ExecutionRequest(
            family="ltx23",
            optimizations={"quantization": "fp8"},
        )
    )
    assert "LTX 2.3 Reference execution accepts BF16 only" in reference_errors


def test_ltx23_runtime_is_reused_per_resolved_model_selection(tmp_path: Path, monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings, plan, *, operation):
            self.settings = settings
            self.plan = plan
            self.operation = operation
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ltx23_tools, "ManagedLTX23Runtime", FakeRuntime)
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


def test_ltx23_condition_runtime_has_a_distinct_manager_identity(tmp_path: Path, monkeypatch):
    created = []

    class FakeManagedRuntime:
        def __init__(self, settings, plan, *, operation):
            self.settings = settings
            self.plan = plan
            created.append((operation, self))

        def unload(self):
            pass

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ltx23_tools, "ManagedLTX23Runtime", FakeManagedRuntime)
    settings = _settings(tmp_path)
    context = SimpleNamespace(settings=settings, execution=None)
    plan = ltx23_tools.LTX23TextToVideoTool()._resolve_plan(context)

    text_runtime = ltx23_tools.LTX23TextToVideoTool()._runtime(context, plan)
    first_runtime = ltx23_tools.LTX23FirstFrameToVideoTool()._runtime(context, plan)
    condition_runtime = ltx23_tools.LTX23ImageToVideoTool()._runtime(context, plan)

    assert text_runtime is not condition_runtime
    assert first_runtime is not condition_runtime
    assert [kind for kind, _runtime in created] == ["t2v", "first_frame", "first_last"]
    status = RUNTIME_MANAGER.status()
    assert {entry["key"] for entry in status["runtimes"]} == {
        f"ltx23:t2v:{plan.pipeline_fingerprint}",
        f"ltx23_condition:first_frame:{plan.pipeline_fingerprint}",
        f"ltx23_condition:first_last:{plan.pipeline_fingerprint}",
    }
    RUNTIME_MANAGER.clear()


@pytest.mark.parametrize(
    ("tool_type", "operation", "endpoint_count"),
    (
        (ltx23_tools.LTX23TextToVideoTool, "ltx23_dev_t2v", 0),
        (ltx23_tools.LTX23FirstFrameToVideoTool, "ltx23_dev_i2v", 1),
        (ltx23_tools.LTX23ImageToVideoTool, "ltx23_distilled_flf", 2),
    ),
)
def test_ltx23_kitchen_variants_use_only_the_engine_native_disposable_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_type,
    operation: str,
    endpoint_count: int,
) -> None:
    request = LTX23KitchenRuntimeRequest(
        schema_version=1,
        family="ltx23",
        operation=operation,
        base_model="Lightricks/LTX-2.3",
        components={},
        identities={},
        plans={},
    )
    calls: list[dict[str, object]] = []

    class FakeKitchenRuntime:
        def __init__(self, actual_request):
            assert actual_request is request

        def generate(self, **kwargs):
            calls.append(kwargs)
            output = Path(kwargs["output_path"])
            output.write_bytes(b"mp4")
            return SimpleNamespace(
                output_path=output,
                output_size_bytes=3,
                metadata={
                    "runtime": "engine-native/ltx23-kitchen",
                    "operation": operation,
                },
                worker_pid=1,
                worker_exit_code=0,
            )

        def status(self):
            return {
                "last_worker": {"outcome": "succeeded", "tree_empty": True},
                "cleanup_errors": [],
            }

        def unload(self):
            pass

        def clear_cache(self):
            pass

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ltx23_tools, "ManagedLTX23KitchenRuntime", FakeKitchenRuntime)
    tool = tool_type()
    monkeypatch.setattr(
        tool,
        "_resolve_plan",
        lambda _context: (_ for _ in ()).throw(AssertionError("native BF16 fallthrough")),
    )
    settings = _settings(tmp_path)
    storage = Storage(settings)
    inputs: dict[str, object] = {
        "prompt": "test",
        "width": 768,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 7,
    }
    for key in ("start_image", "end_image")[:endpoint_count]:
        asset = storage.store_asset(BytesIO(key.encode()), f"{key}.png", "image/png", 1024)
        inputs[key] = {"type": "asset", "asset_id": asset.id}
    context = ToolContext(
        job_id=UUID(int=endpoint_count + 1),
        settings=settings,
        storage=storage,
        cancel_event=Event(),
        progress=lambda _value, _message: None,
        execution=ExecutionPlan(
            variant_key=f"test.{operation}",
            family="ltx23",
            recipe=request,
        ),
    )

    artifacts = tool.run(context, inputs)

    assert len(calls) == 1
    assert len(artifacts) == 1
    assert artifacts[0].path.read_bytes() == b"mp4"
    if endpoint_count == 0:
        assert calls[0]["start_image_path"] is None
    else:
        assert isinstance(calls[0]["start_image_path"], Path)
    if endpoint_count < 2:
        assert calls[0]["end_image_path"] is None
    else:
        assert isinstance(calls[0]["end_image_path"], Path)
    assert context.runtime_provenance["runtime_plan"]["request_fingerprint"] == (
        request.fingerprint
    )
    assert context.runtime_provenance["runtime_result"]["worker"]["tree_empty"] is True
    RUNTIME_MANAGER.clear()
