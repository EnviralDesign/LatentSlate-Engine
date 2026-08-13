from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import UUID

import pytest

from latentslate_engine import resources as resources_module
from latentslate_engine import variants as variants_module
from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.config import Settings
from latentslate_engine.ltx23_comfy_recipe import (
    LTX23_COMFY_FLF_TEMPLATE_SHA256,
    LTX23_COMFY_FPS,
    LTX23_COMFY_I2V_TEMPLATE_SHA256,
    LTX23_COMFY_MAIN_SIGMAS,
    LTX23_COMFY_MODEL_LORA_STRENGTH,
    LTX23_COMFY_RUNTIME_REVISION,
    LTX23_COMFY_T2V_TEMPLATE_SHA256,
    LTX23_COMFY_UPSCALE_SIGMAS,
    LTX23ComfyRuntimeRequest,
    _expected_components,
    required_roles,
    template_sha256,
)
from latentslate_engine.resources import discover_resources
from latentslate_engine.runtime import ltx23_comfy as comfy_runtime
from latentslate_engine.runtime.ltx23_comfy import (
    LTX23ComfyRequest,
    ManagedLTX23ComfyRuntime,
    build_workflow,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.storage import Storage
from latentslate_engine.tools import default_registry
from latentslate_engine.tools.base import ExecutionPlan, ToolCancelled, ToolContext
from latentslate_engine.tools.ltx23 import (
    LTX23FirstFrameToVideoTool,
    LTX23ImageToVideoTool,
    LTX23TextToVideoTool,
    _comfy_runtime_availability,
)
from latentslate_engine.variants import VariantDefinition, _validate_ltx23_comfy_base_tool


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cpu",
    )


def _template_fixture() -> dict[str, object]:
    path = Path(__file__).with_name("fixtures") / "ltx23_comfy_template_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _submitted_graph_fixture() -> dict[str, object]:
    path = Path(__file__).with_name("fixtures") / "ltx23_comfy_submitted_graphs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize_submitted_graph(
    operation: str, graph: dict[str, dict[str, object]], *, prompt: str,
    staged_first: str | None, staged_last: str | None,
) -> dict[str, dict[str, object]]:
    """Canonicalize only the dynamic fields documented by the static fixture."""

    replacements = {
        prompt: "{{prompt}}",
        **({staged_first: "{{staged:first}}"} if staged_first else {}),
        **({staged_last: "{{staged:last}}"} if staged_last else {}),
        **{
            f"{role}.safetensors": f"{{{{model:{role}}}}}"
            for role in (
                "checkpoint", "text_encoder", "model_lora", "text_lora", "latent_upscaler"
            )
        },
    }

    def walk(value):
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return replacements.get(value, value)

    normalized = walk(graph)
    if operation.startswith("comfy_dev"):
        normalized["8"]["inputs"].update(
            width="{{width_half}}", height="{{height_half}}", length="{{frames}}"
        )
        normalized["9"]["inputs"]["frames_number"] = "{{frames}}"
        normalized["13"]["inputs"]["noise_seed"] = "{{seed}}"
        normalized["41"]["inputs"]["resize_type.width"] = "{{width}}"
        normalized["41"]["inputs"]["resize_type.height"] = "{{height}}"
        if operation == "comfy_dev_t2v":
            normalized["11"]["inputs"]["width"] = "{{width}}"
            normalized["11"]["inputs"]["height"] = "{{height}}"
    else:
        normalized["10"]["inputs"].update(
            width="{{width}}", height="{{height}}", length="{{frames}}"
        )
        normalized["14"]["inputs"]["frames_number"] = "{{frames}}"
        normalized["16"]["inputs"]["noise_seed"] = "{{seed}}"
        for node_id in ("27", "28"):
            normalized[node_id]["inputs"]["resize_type.width"] = "{{width}}"
            normalized[node_id]["inputs"]["resize_type.height"] = "{{height}}"
    return normalized


def _request(operation: str) -> LTX23ComfyRuntimeRequest:
    roles = required_roles(operation)  # type: ignore[arg-type]
    return LTX23ComfyRuntimeRequest(
        1,
        "ltx23",
        operation,  # type: ignore[arg-type]
        "Lightricks/LTX-2.3",
        {
            role: {
                "resource_id": f"{role}:test",
                "path": f"C:/models/{role}.safetensors",
                "size_bytes": 1,
                "mtime_ns": 1,
                "header_sha256": "a" * 64,
                "schema_sha256": "b" * 64,
            }
            for role in roles
        },
        {},
    )


def _valid_runtime_request(tmp_path: Path, operation: str = "comfy_dev_t2v") -> LTX23ComfyRuntimeRequest:
    """Create real tiny SafeTensors identities for pre-staging binding tests."""

    header = b'{"tensor":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    identities = {}
    components = {}
    expected = _expected_components(operation)
    for role in required_roles(operation):  # type: ignore[arg-type]
        path = tmp_path / f"{role}.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00")
        stat = path.stat()
        identity = ArtifactIdentity(path.resolve(), stat.st_size, stat.st_mtime_ns, hashlib.sha256(header).hexdigest())
        identities[role] = identity
        components[role] = {
            "resource_id": f"{role}:test", "path": str(identity.path), "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns, "header_sha256": identity.header_sha256,
            "schema_sha256": expected[role][3],
        }
    request = LTX23ComfyRuntimeRequest(1, "ltx23", operation, "Lightricks/LTX-2.3", components, identities)  # type: ignore[arg-type]
    assert comfy_runtime.revalidate_ltx23_comfy_runtime_request(request)
    return request


def test_ltx23_comfy_catalog_declares_three_immutable_operation_closures(tmp_path: Path) -> None:
    registry = default_registry(_settings(tmp_path), emit_warnings=False)
    assert not registry.variant_errors
    recipes = {entry.key: entry for entry in registry.variants}
    t2v = recipes["ltx-2-3.text-to-video.comfy-dev-fp8"]
    i2v = recipes["ltx-2-3.image-to-video.comfy-dev-fp8"]
    flf = recipes["ltx-2-3.first-last-frame-to-video.comfy-distilled-fp8"]
    assert t2v.recipe_type == "ltx23_comfy_dev_t2v"
    assert i2v.recipe_type == "ltx23_comfy_dev_i2v"
    assert flf.recipe_type == "ltx23_comfy_distilled_flf"
    assert set(t2v.recipe_resources) == set(required_roles("comfy_dev_t2v"))
    assert set(i2v.recipe_resources) == set(required_roles("comfy_dev_i2v"))
    assert set(flf.recipe_resources) == set(required_roles("comfy_distilled_flf"))
    assert "experimental" in t2v.tags and "experimental" in flf.tags
    tools = {tool.descriptor.key: tool for tool in registry.tools()}
    assert [item.key for item in tools[t2v.key].descriptor.inputs] == ["prompt", "width", "height", "duration_seconds", "seed"]
    assert tools[t2v.key].descriptor.inputs[1].ui.step == 64
    assert tools[t2v.key].descriptor.inputs[2].ui.step == 64
    assert tools[t2v.key].descriptor.inputs[3].ui.step == 1
    assert [item.key for item in tools[i2v.key].descriptor.inputs] == ["prompt", "start_image", "width", "height", "duration_seconds", "seed"]
    assert tools[i2v.key].descriptor.inputs[2].ui.step == 64
    assert tools[i2v.key].descriptor.inputs[3].ui.step == 64
    assert tools[i2v.key].descriptor.inputs[4].ui.step == 1
    assert [item.key for item in tools[flf.key].descriptor.inputs] == ["prompt", "start_image", "end_image", "width", "height", "duration_seconds", "seed"]
    assert tools[flf.key].descriptor.inputs[3].ui.step == 32
    assert tools[flf.key].descriptor.inputs[4].ui.step == 32
    assert tools[flf.key].descriptor.inputs[5].ui.step == 1
    assert tools[t2v.key].provenance() == {
        "runtime": "comfyui_disposable_worker",
        "pipeline": "official_comfy_ltx23_graph",
        "model_family": "ltx_2_3",
        "artifact_contract": "pinned_comfy_fp8_operation_closure",
        "operation": "comfy_dev_t2v",
        "variant_key": t2v.key,
        "variant_source": t2v.source_path,
        "variant_family": "ltx23",
    }
    assert tools[i2v.key].provenance()["operation"] == "comfy_dev_i2v"
    assert tools[flf.key].provenance()["operation"] == "comfy_distilled_flf"
    native = tools["ltx-2-3.text-to-video.native-distilled-bf16"].provenance()
    assert native["runtime"] == "diffusers_disposable_worker"
    assert native["artifact_contract"] == "complete_diffusers_bf16_native"


def test_ltx23_comfy_declarations_enrich_installed_canonical_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-install discovery must merge declarations into the path-derived IDs."""

    settings = _settings(tmp_path)
    relative_paths = {
        "model:ltx23:comfy/checkpoints/ltx-2.3-22b-dev-fp8": (
            "models/ltx23/comfy/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
        ),
        "model:ltx23:comfy/checkpoints/ltx-2.3-22b-distilled-fp8": (
            "models/ltx23/comfy/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
        ),
        "model:ltx23:comfy/text_encoders/gemma_3_12b_it_fp4_mixed": (
            "models/ltx23/comfy/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
        ),
        "model:ltx23:comfy/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1": (
            "models/ltx23/comfy/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ),
        "lora:ltx23:comfy/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16": (
            "loras/ltx23/comfy/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
        ),
        "lora:ltx23:comfy/gemma-3-12b-it-abliterated_lora_rank64_bf16": (
            "loras/ltx23/comfy/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
        ),
    }
    for relative_path in relative_paths.values():
        path = settings.home / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def available(resource, _path, **_kwargs):
        return resource.model_copy(update={"available": True, "unavailable_reason": None})

    monkeypatch.setattr(resources_module, "_with_artifact_availability", available)
    monkeypatch.setattr(
        variants_module,
        "validate_ltx23_comfy_recipe",
        lambda *_args, **_kwargs: SimpleNamespace(errors=()),
    )
    inventory = discover_resources(settings)
    assert not [error for error in inventory.errors if "ltx23-comfy" in error]
    resources = {resource.id: resource for resource in inventory.resources}
    for resource_id, relative_path in relative_paths.items():
        assert resources[resource_id].sources
        assert inventory.paths[resource_id] == settings.home / relative_path

    registry = default_registry(settings, emit_warnings=False)
    entries = {
        entry.key: entry
        for entry in registry.variants
        if entry.key.startswith("ltx-2-3.") and ".comfy-" in entry.key
    }
    assert len(entries) == 3
    assert all(entry.available for entry in entries.values())


@pytest.mark.parametrize(
    ("recipe_name", "incorrect_base_tool"),
    [
        ("ltx-2-3-text-to-video-comfy-dev-fp8.toml", "ltx23.image_to_video"),
        ("ltx-2-3-image-to-video-comfy-dev-fp8.toml", "ltx23.first_last_frame_to_video"),
        ("ltx-2-3-first-last-frame-to-video-comfy-distilled-fp8.toml", "ltx23.text_to_video"),
    ],
)
def test_ltx23_comfy_recipe_type_cannot_bind_the_wrong_public_tool(
    recipe_name: str,
    incorrect_base_tool: str,
) -> None:
    recipe_path = (
        Path(__file__).parents[1]
        / "src"
        / "latentslate_engine"
        / "builtin_recipes"
        / "ltx23"
        / recipe_name
    )
    raw = tomllib.loads(recipe_path.read_text(encoding="utf-8"))
    definition_data = raw.get("runnable_recipe", raw.get("variant", raw))
    definition_data["base_tool"] = incorrect_base_tool
    definition = VariantDefinition.model_validate(definition_data)
    with pytest.raises(ValueError, match="requires base_tool"):
        _validate_ltx23_comfy_base_tool(definition)


def test_ltx23_comfy_template_hashes_and_operation_schedules_are_distinct() -> None:
    fixture = _template_fixture()
    operations = fixture["operations"]
    assert template_sha256("comfy_dev_t2v") == LTX23_COMFY_T2V_TEMPLATE_SHA256
    assert template_sha256("comfy_dev_i2v") == LTX23_COMFY_I2V_TEMPLATE_SHA256
    assert template_sha256("comfy_distilled_flf") == LTX23_COMFY_FLF_TEMPLATE_SHA256
    assert len(LTX23_COMFY_MAIN_SIGMAS) == 9
    assert len(LTX23_COMFY_UPSCALE_SIGMAS) == 4
    assert LTX23_COMFY_RUNTIME_REVISION == fixture["runtime_revision"]
    assert template_sha256("comfy_dev_t2v") == operations["comfy_dev_t2v"]["raw_sha256"]
    assert template_sha256("comfy_dev_i2v") == operations["comfy_dev_i2v"]["raw_sha256"]
    assert template_sha256("comfy_distilled_flf") == operations["comfy_distilled_flf"]["raw_sha256"]
    assert comfy_runtime._COMFY_REQUIRED_SOURCE_BLOBS == fixture["required_source_blobs"]


def test_ltx23_comfy_dev_graph_keeps_two_stage_lora_and_audio_video_contract() -> None:
    recipe = _request("comfy_dev_t2v")
    graph = build_workflow(recipe, LTX23ComfyRequest("scene", 1280, 720, 1.0, 7), start_name=None, end_name=None)
    assert graph["2"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["2"]["inputs"]["strength_model"] == LTX23_COMFY_MODEL_LORA_STRENGTH
    assert graph["15"]["inputs"]["sigmas"] == ", ".join(str(value) for value in LTX23_COMFY_MAIN_SIGMAS)
    assert graph["26"]["inputs"]["sigmas"] == ", ".join(str(value) for value in LTX23_COMFY_UPSCALE_SIGMAS)
    assert graph["32"]["inputs"]["fps"] == LTX23_COMFY_FPS
    assert graph["10"]["inputs"]["image"] == ["40", 0]
    assert graph["22"]["inputs"]["image"] == ["40", 0]
    assert graph["40"]["inputs"] == {"image": ["42", 0], "img_compression": 18}
    assert graph["41"]["inputs"] == {
        "input": ["11", 0],
        "resize_type": "scale dimensions",
        "resize_type.width": 1280,
        "resize_type.height": 720,
        "resize_type.crop": "center",
        "scale_method": "lanczos",
    }
    assert graph["42"] == {
        "class_type": "ResizeImagesByLongerEdge",
        "inputs": {"images": ["41", 0], "longer_edge": 1536},
    }
    assert graph["17"]["inputs"]["positive"] == ["6", 0]
    assert graph["18"]["inputs"]["latent_image"] == ["12", 0]
    assert graph["16"]["inputs"] == {
        "positive": ["6", 0], "negative": ["6", 1], "latent": ["19", 0],
    }
    assert graph["27"]["inputs"]["positive"] == ["16", 0]
    assert graph["4"]["inputs"]["text"] == ["37", 0]
    assert graph["35"]["inputs"] == {
        "model": ["2", 0], "clip": ["3", 0], "lora_name": "text_lora.safetensors",
        "strength_model": 1.0, "strength_clip": 1.0,
    }
    assert graph["36"]["inputs"]["clip"] == ["35", 1]
    assert graph["37"]["inputs"] == {
        "on_false": ["34", 0], "on_true": ["36", 0], "switch": True,
    }
    assert graph["33"]["inputs"]["format"] == graph["33"]["inputs"]["codec"] == "auto"
    classes = [node["class_type"] for node in graph.values()]
    assert classes.count("LTXVPreprocess") == 1
    assert classes.count("ResizeImageMaskNode") == 1
    assert classes.count("ResizeImagesByLongerEdge") == 1
    assert classes.count("RandomNoise") == 2
    assert classes.count("SamplerCustomAdvanced") == 2
    assert classes.count("LTXVImgToVideoInplace") == 2


def test_ltx23_comfy_i2v_graph_has_the_same_official_two_stage_topology_but_a_real_first_frame() -> None:
    recipe = _request("comfy_dev_i2v")
    graph = build_workflow(
        recipe,
        LTX23ComfyRequest("scene", 1280, 720, 1.0, 7, Path("first.png")),
        start_name="first.png",
        end_name=None,
    )
    assert graph["11"] == {"class_type": "LoadImage", "inputs": {"image": "first.png"}}
    assert graph["10"]["inputs"] == {
        "vae": ["1", 2], "image": ["40", 0], "latent": ["8", 0], "strength": 0.7, "bypass": False,
    }
    assert graph["22"]["inputs"] == {
        "vae": ["1", 2], "image": ["40", 0], "latent": ["21", 0], "strength": 1.0, "bypass": False,
    }
    assert graph["36"]["inputs"]["image"] == ["42", 0]
    assert graph["37"]["inputs"] == {
        "on_false": ["34", 0], "on_true": ["36", 0], "switch": False,
    }
    # The deliberately non-sequential preprocess/resize IDs keep them distinct
    # from the sampler nodes that occupy the official template's 13/14 slots.
    assert set(graph) == {str(node_id) for node_id in range(1, 38)} | {"40", "41", "42"}


def test_ltx23_comfy_graph_edges_match_the_independent_raw_template_fixture() -> None:
    fixture = _template_fixture()["operations"]
    t2v = build_workflow(
        _request("comfy_dev_t2v"), LTX23ComfyRequest("scene", 1280, 720, 1.0, 7), start_name=None, end_name=None,
    )
    i2v = build_workflow(
        _request("comfy_dev_i2v"), LTX23ComfyRequest("scene", 1280, 720, 1.0, 7, Path("first.png")), start_name="first.png", end_name=None,
    )
    flf = build_workflow(
        _request("comfy_distilled_flf"), LTX23ComfyRequest("scene", 1280, 720, 1.0, 7, Path("first.png"), Path("last.png")), start_name="first.png", end_name="last.png",
    )
    assert t2v["40"]["inputs"]["image"] == fixture["comfy_dev_t2v"]["edges"]["preprocess_image"]
    assert t2v["16"]["inputs"]["latent"] == fixture["comfy_dev_t2v"]["edges"]["crop_latent"]
    assert t2v["17"]["inputs"]["positive"] == fixture["comfy_dev_t2v"]["edges"]["base_positive"]
    assert t2v["27"]["inputs"]["positive"] == fixture["comfy_dev_t2v"]["edges"]["refine_positive"]
    assert t2v["37"]["inputs"]["switch"] is fixture["comfy_dev_t2v"]["edges"]["prompt_switch"]
    assert i2v["36"]["inputs"]["image"] == fixture["comfy_dev_i2v"]["edges"]["enhancer_image"]
    assert i2v["37"]["inputs"]["switch"] is fixture["comfy_dev_i2v"]["edges"]["prompt_switch"]
    assert flf["11"]["inputs"]["frame_idx"] == fixture["comfy_distilled_flf"]["edges"]["first_frame_idx"]
    assert flf["12"]["inputs"]["frame_idx"] == fixture["comfy_distilled_flf"]["edges"]["last_frame_idx"]
    assert flf["21"]["inputs"]["av_latent"][1] == fixture["comfy_distilled_flf"]["edges"]["separate_sampler_slot"]
    for operation, graph in {
        "comfy_dev_t2v": t2v,
        "comfy_dev_i2v": i2v,
        "comfy_distilled_flf": flf,
    }.items():
        assert sorted({node["class_type"] for node in graph.values()}) == fixture[operation][
            "required_classes"
        ]


@pytest.mark.parametrize(
    ("operation", "start_name", "end_name"),
    [
        ("comfy_dev_t2v", None, None),
        ("comfy_dev_i2v", "job-first-a1.png", None),
        ("comfy_distilled_flf", "job-first-a1.png", "job-last-b2.png"),
    ],
)
def test_ltx23_comfy_complete_submitted_graph_matches_static_pinned_fixture(
    operation: str, start_name: str | None, end_name: str | None,
) -> None:
    fixture = _submitted_graph_fixture()
    assert fixture["source_revision"] == "8b2c08f297c63ffc73ce93f938b0f5139c0ed73f"
    expected = fixture["operations"][operation]
    assert expected["raw_template_sha256"] == template_sha256(operation)
    prompt = "A wholly different prompt used to prove placeholder canonicalization."
    graph = build_workflow(
        _request(operation),
        LTX23ComfyRequest(
            prompt, 1152, 640, 1.0, 99173,
            Path("private-first.png") if start_name else None,
            Path("private-last.png") if end_name else None,
        ),
        start_name=start_name,
        end_name=end_name,
    )
    assert _canonicalize_submitted_graph(
        operation, graph, prompt=prompt, staged_first=start_name, staged_last=end_name,
    ) == expected["normalized_submitted_graph"]


def test_ltx23_comfy_flf_graph_requires_ordered_endpoints_and_uses_one_stage_guides() -> None:
    recipe = _request("comfy_distilled_flf")
    graph = build_workflow(recipe, LTX23ComfyRequest("scene", 1280, 720, 1.0, 7, Path("first.png"), Path("last.png")), start_name="first.png", end_name="last.png")
    assert graph["11"]["inputs"]["frame_idx"] == 0
    assert graph["12"]["inputs"]["frame_idx"] == -1
    assert graph["11"]["inputs"] == {
        "positive": ["9", 0], "negative": ["9", 1], "vae": ["1", 2],
        "latent": ["10", 0], "image": ["7", 0], "frame_idx": 0, "strength": 0.7,
    }
    assert graph["12"]["inputs"] == {
        "positive": ["11", 0], "negative": ["11", 1], "vae": ["1", 2],
        "latent": ["11", 2], "image": ["8", 0], "frame_idx": -1, "strength": 0.7,
    }
    assert graph["7"]["inputs"]["image"] == ["27", 0]
    assert graph["8"]["inputs"]["image"] == ["28", 0]
    assert graph["7"]["inputs"]["img_compression"] == 25
    assert graph["8"]["inputs"]["img_compression"] == 25
    assert graph["27"]["inputs"] == {
        "input": ["5", 0],
        "resize_type": "scale dimensions",
        "resize_type.width": 1280,
        "resize_type.height": 720,
        "resize_type.crop": "center",
        "scale_method": "nearest-exact",
    }
    assert graph["28"]["inputs"]["input"] == ["6", 0]
    assert graph["26"]["inputs"] == {
        "positive": ["12", 0], "negative": ["12", 1], "latent": ["21", 0],
    }
    assert graph["22"]["inputs"]["samples"] == ["26", 2]
    assert graph["21"]["inputs"]["av_latent"] == ["20", 1]
    assert graph["25"]["inputs"]["format"] == graph["25"]["inputs"]["codec"] == "auto"
    assert graph["18"]["inputs"]["sigmas"] == ", ".join(str(value) for value in LTX23_COMFY_MAIN_SIGMAS)
    assert "LTXVLatentUpsampler" not in {node["class_type"] for node in graph.values()}
    classes = [node["class_type"] for node in graph.values()]
    assert classes.count("LoadImage") == 2
    assert classes.count("ResizeImageMaskNode") == 2
    assert classes.count("LTXVPreprocess") == 2
    assert classes.count("LTXVAddGuide") == 2
    assert classes.count("LTXVCropGuides") == 1
    with pytest.raises(ValueError, match="requires both"):
        build_workflow(recipe, LTX23ComfyRequest("scene", 1280, 720, 1.0, 7, Path("first.png")), start_name="first.png", end_name=None)


def test_ltx23_comfy_waits_only_for_the_expected_savevideo_output(monkeypatch) -> None:
    complete = {
        "job": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "other": {"videos": [{"filename": "wrong.mp4"}]},
                "33": {"videos": [{"filename": "right.mp4"}]},
            },
        },
    }
    monkeypatch.setattr(comfy_runtime, "_json_request", lambda *_args: complete)
    assert comfy_runtime._wait_video("http://unit", "job", lambda *_args: None, lambda: None, save_node_id="33") == {"filename": "right.mp4"}


def test_ltx23_comfy_queue_running_progress_binds_the_prompt_id(monkeypatch) -> None:
    calls = []
    responses = iter([
        {},
        {"queue_running": [[7, "other", {}, {}, {}]], "queue_pending": [[8, "job", {}, {}, {}]]},
        {},
        {"queue_running": [[9, "job", {}, {}, {}]], "queue_pending": []},
        {"job": {"status": {"status_str": "success", "completed": True}, "outputs": {"33": {"videos": [{"filename": "right.mp4"}]}}}},
    ])
    monkeypatch.setattr(comfy_runtime, "_json_request", lambda *_args: next(responses))
    monkeypatch.setattr(comfy_runtime.time, "sleep", lambda *_args: None)
    result = comfy_runtime._wait_video(
        "http://unit", "job", lambda _value, message: calls.append(message), lambda: None, save_node_id="33"
    )
    assert result == {"filename": "right.mp4"}
    assert calls == [comfy_runtime._QUEUE_RUNNING_MESSAGE]


def test_ltx23_comfy_mp4_contract_uses_observed_stream_facts(monkeypatch, tmp_path: Path) -> None:
    class Completed:
        stdout = """{"format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2"},"streams":[{"codec_type":"video","codec_name":"h264","width":1280,"height":720,"avg_frame_rate":"24/1","nb_read_frames":"25","duration":"1.0416667"},{"codec_type":"audio","sample_rate":"48000","channels":2,"duration":"1.0416667"}]}"""

    monkeypatch.setattr(comfy_runtime.subprocess, "run", lambda *_args, **_kwargs: Completed())
    observed = comfy_runtime._validate_mp4(tmp_path / "output.mp4", LTX23ComfyRequest("scene", 1280, 720, 1.0, 1))
    assert observed == {
        "codec": "h264", "width": 1280, "height": 720, "fps": 24, "frame_count": 25,
        "video_duration_seconds": pytest.approx(25 / 24), "audio_duration_seconds": pytest.approx(25 / 24),
        "sample_rate": 48000, "channels": 2,
    }


def test_ltx23_comfy_mp4_contract_rejects_non_mp4_container(monkeypatch, tmp_path: Path) -> None:
    class Completed:
        stdout = """{"format":{"format_name":"matroska,webm"},"streams":[]}"""

    monkeypatch.setattr(comfy_runtime.subprocess, "run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(RuntimeError, match="not an MP4"):
        comfy_runtime._validate_mp4(tmp_path / "output.mp4", LTX23ComfyRequest("scene", 1280, 720, 1.0, 1))


def test_ltx23_comfy_checkout_fails_closed_when_required_node_schema_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "ComfyUI"
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").touch()
    (root / "main.py").touch()
    with pytest.raises(RuntimeError, match="checkout revision|required graph support"):
        comfy_runtime._validate_comfy_checkout(root)


def test_ltx23_comfy_availability_requires_windows_job_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentslate_engine.tools.ltx23 as ltx_tool

    monkeypatch.setattr(ltx_tool.os, "name", "posix")
    available, reason = _comfy_runtime_availability(tmp_path)
    assert available is False
    assert reason is not None and "Windows Job Object" in reason


def test_ltx23_comfy_checkout_rejects_dirty_required_node_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ComfyUI"
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").touch()
    (root / "main.py").touch()
    for relative in comfy_runtime._COMFY_REQUIRED_SOURCE_BLOBS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path("C:/ComfyUI") / relative).read_bytes())
    monkeypatch.setattr(
        comfy_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": LTX23_COMFY_RUNTIME_REVISION})(),
    )
    # Keep all required names intact while changing a single source byte.
    monkeypatch.setattr(comfy_runtime, "_COMFY_REQUIRED_SOURCE_MARKERS", {})
    (root / "nodes.py").write_bytes((root / "nodes.py").read_bytes() + b"\n# local change\n")
    with pytest.raises(RuntimeError, match="required node source is modified"):
        comfy_runtime._validate_comfy_checkout(root)


@pytest.mark.skipif(not Path("C:/ComfyUI").is_dir(), reason="pinned local Comfy checkout")
def test_ltx23_comfy_checkout_accepts_the_exact_pinned_source_locations() -> None:
    assert comfy_runtime._COMFY_REQUIRED_SOURCE_BLOBS["comfy_extras/nodes_hunyuan.py"] == (
        "ce2997245840e6f038b3e0c26ceb900cb6d9910a"
    )
    assert "LatentUpscaleModelLoader" in comfy_runtime._COMFY_REQUIRED_SOURCE_MARKERS[
        "comfy_extras/nodes_hunyuan.py"
    ]
    assert "LatentUpscaleModelLoader" not in comfy_runtime._COMFY_REQUIRED_SOURCE_MARKERS[
        "comfy_extras/nodes_lt_upsampler.py"
    ]
    comfy_runtime._validate_comfy_checkout(Path("C:/ComfyUI"))


def test_ltx23_comfy_endpoint_oversize_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "too-large.png"
    source.write_bytes(b"12345")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(comfy_runtime, "_MAX_ENDPOINT_BYTES", 4)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("oversize source opened"))
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        comfy_runtime._stage_endpoint(source, workspace, "first", lambda: None)
    assert not (workspace / "input").exists()


def test_ltx23_comfy_endpoint_cancellation_removes_partial_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "first image.PNG"
    source.write_bytes(b"abcdefgh")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(comfy_runtime, "_IO_CHUNK_BYTES", 4)
    calls = 0

    def cancel_during_copy() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ToolCancelled("Generation canceled")

    with pytest.raises(ToolCancelled):
        comfy_runtime._stage_endpoint(source, workspace, "first", cancel_during_copy)
    assert list((workspace / "input").glob("*")) == []


def test_ltx23_comfy_generate_cancellation_during_staging_removes_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component_root = tmp_path / "components"
    component_root.mkdir()
    recipe = _valid_runtime_request(component_root, "comfy_dev_i2v")
    source = tmp_path / "first.png"
    source.write_bytes(b"abcdefgh")
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    runtime = ManagedLTX23ComfyRuntime(recipe, comfy_root=tmp_path)
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(comfy_runtime, "_validate_comfy_checkout", lambda _root: None)
    monkeypatch.setattr(comfy_runtime, "_workspace", lambda _recipe: workspace)
    monkeypatch.setattr(comfy_runtime, "_stage_components", lambda *_args: None)
    monkeypatch.setattr(comfy_runtime, "_IO_CHUNK_BYTES", 4)
    monkeypatch.setattr(comfy_runtime, "_start_comfy", lambda *_args, **_kwargs: pytest.fail("worker started"))
    calls = 0

    def cancel_during_staging() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise ToolCancelled("Generation canceled")

    with pytest.raises(ToolCancelled):
        runtime.generate(
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1, source),
            output_path=tmp_path / "output.mp4", progress=lambda *_args: None,
            check_cancelled=cancel_during_staging,
        )
    assert not workspace.exists()
    assert runtime.status()["last_worker"]["outcome"] == "canceled"
    assert runtime.status()["last_worker"]["spawned"] is False


def test_ltx23_comfy_staged_endpoint_names_are_private_unique_and_bind_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "FIRST weird name.PNG"
    last = tmp_path / "last odd.JPEG"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    values = iter([SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)])
    monkeypatch.setattr(comfy_runtime, "uuid4", lambda: next(values))
    first_name = comfy_runtime._stage_endpoint(first, tmp_path / "job", "first", lambda: None)
    last_name = comfy_runtime._stage_endpoint(last, tmp_path / "job", "last", lambda: None)
    assert first_name == f"latentslate-first-{'a' * 32}.png"
    assert last_name == f"latentslate-last-{'b' * 32}.jpeg"
    graph = build_workflow(
        _request("comfy_distilled_flf"),
        LTX23ComfyRequest("scene", 1280, 704, 1.0, 7, first, last),
        start_name=first_name,
        end_name=last_name,
    )
    assert graph["5"]["inputs"]["image"] == first_name
    assert graph["6"]["inputs"]["image"] == last_name
    assert (tmp_path / "job" / "input" / first_name).read_bytes() == b"first"
    assert (tmp_path / "job" / "input" / last_name).read_bytes() == b"last"


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes], *, content_length: int | None = None) -> None:
        self._chunks = iter(chunks)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return next(self._chunks, b"")


def test_ltx23_comfy_output_stream_overflow_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _ChunkedResponse([b"abcd", b"efgh", b""])
    monkeypatch.setattr(comfy_runtime, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(comfy_runtime, "_MAX_OUTPUT_BYTES", 6)
    monkeypatch.setattr(comfy_runtime, "_IO_CHUNK_BYTES", 4)
    output = tmp_path / "published" / "output.mp4"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(RuntimeError, match="bounded size"):
        comfy_runtime._download_validate_and_publish(
            "http://unit", {"filename": "x.mp4"}, output, workspace,
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1), lambda: None,
        )
    assert response.read_sizes and set(response.read_sizes) == {4}
    assert not output.exists()
    assert not (workspace / "download.mp4").exists()
    assert list(output.parent.glob("*.partial")) == []


def test_ltx23_comfy_late_output_cancellation_never_atomically_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _ChunkedResponse([b"video", b""])
    monkeypatch.setattr(comfy_runtime, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(comfy_runtime, "_validate_mp4", lambda *_args: {"codec": "h264"})
    replaced = False

    def replace(*_args) -> None:
        nonlocal replaced
        replaced = True

    monkeypatch.setattr(comfy_runtime.os, "replace", replace)
    calls = 0

    def cancel_immediately_before_replace() -> None:
        nonlocal calls
        calls += 1
        if calls == 10:
            raise ToolCancelled("Generation canceled")

    output = tmp_path / "published" / "output.mp4"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ToolCancelled):
        comfy_runtime._download_validate_and_publish(
            "http://unit", {"filename": "x.mp4"}, output, workspace,
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1),
            cancel_immediately_before_replace,
        )
    assert calls == 10
    assert replaced is False
    assert not output.exists()
    assert not (workspace / "download.mp4").exists()
    assert list(output.parent.glob("*.partial")) == []


@pytest.mark.parametrize(
    "comfy_request, message",
    [
        (LTX23ComfyRequest("scene", 1280, 960, 1.0, 1), "pixel area"),
        (LTX23ComfyRequest("scene", 800, 512, 1.0, 1), "divisible by 64"),
        (LTX23ComfyRequest("scene", 1280, 704, 0.5, 1), "between 1 and 10"),
        (LTX23ComfyRequest("scene", 1280, 704, 11.0, 1), "between 1 and 10"),
    ],
)
def test_ltx23_comfy_request_defensively_bounds_direct_runtime_calls(
    comfy_request: LTX23ComfyRequest,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        comfy_runtime._validate_request(comfy_request, "comfy_dev_t2v")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("family", "other"),
        ("base_model", "other/model"),
        ("component_fingerprint", "ltx23-comfy-components:sha256:tampered"),
        ("fingerprint", "ltx23-comfy:sha256:tampered"),
    ],
)
def test_ltx23_comfy_runtime_request_revalidation_rejects_canonical_field_tampering(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request = _valid_runtime_request(tmp_path)
    object.__setattr__(request, field, value)
    assert not comfy_runtime.revalidate_ltx23_comfy_runtime_request(request)


@pytest.mark.parametrize("component_field", ["path", "size_bytes", "mtime_ns", "header_sha256", "schema_sha256"])
def test_ltx23_comfy_runtime_request_revalidation_rejects_component_tampering(
    tmp_path: Path,
    component_field: str,
) -> None:
    request = _valid_runtime_request(tmp_path)
    components = {role: dict(value) for role, value in request.components.items()}
    components["checkpoint"][component_field] = "tampered" if component_field in {"path", "header_sha256", "schema_sha256"} else 999
    object.__setattr__(request, "components", components)
    assert not comfy_runtime.revalidate_ltx23_comfy_runtime_request(request)


def test_ltx23_comfy_job_assignment_failure_proves_only_root_exit(tmp_path: Path, monkeypatch) -> None:
    runtime = ManagedLTX23ComfyRuntime(_request("comfy_dev_t2v"), comfy_root=tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Process:
        pid = 9

        def __init__(self) -> None:
            self.running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout=None) -> int:
            self.running = False
            return 0

    process = Process()
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(comfy_runtime, "_validate_comfy_checkout", lambda _root: None)
    monkeypatch.setattr(comfy_runtime, "_workspace", lambda _recipe: workspace)
    monkeypatch.setattr(comfy_runtime, "_stage_components", lambda *_args: None)
    monkeypatch.setattr(comfy_runtime, "_start_comfy", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        comfy_runtime,
        "DisposableProcessTree",
        lambda _process: (_ for _ in ()).throw(OSError("assignment failed")),
    )
    with pytest.raises(OSError, match="assignment failed"):
        runtime.generate(
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1),
            output_path=tmp_path / "output.mp4",
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    last = runtime.status()["last_worker"]
    assert last["outcome"] == "failed"
    assert last["tree_empty"] is False
    assert last["root_exited"] is True


def test_ltx23_comfy_failed_disposal_does_not_claim_a_memory_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ManagedLTX23ComfyRuntime(_request("comfy_dev_t2v"), comfy_root=tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Process:
        pid = 19

        def poll(self):
            return None

        def terminate(self) -> None:
            raise OSError("termination failed")

    process = Process()
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(comfy_runtime, "_validate_comfy_checkout", lambda _root: None)
    monkeypatch.setattr(comfy_runtime, "_workspace", lambda _recipe: workspace)
    monkeypatch.setattr(comfy_runtime, "_stage_components", lambda *_args: None)
    monkeypatch.setattr(comfy_runtime, "_start_comfy", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        comfy_runtime, "DisposableProcessTree",
        lambda _process: (_ for _ in ()).throw(OSError("assignment failed")),
    )
    with pytest.raises(OSError, match="assignment failed") as raised:
        runtime.generate(
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1),
            output_path=tmp_path / "output.mp4", progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    last = runtime.status()["last_worker"]
    assert last["terminated"] is False
    assert last["tree_empty"] is False
    assert last["root_exited"] is False
    assert last["memory_boundary"] == "termination_unproven"
    assert "worker disposal failed" in "\n".join(raised.value.__notes__)


def test_ltx23_comfy_pre_spawn_failure_replaces_stale_worker_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ManagedLTX23ComfyRuntime(_valid_runtime_request(tmp_path), comfy_root=tmp_path)
    runtime._last_worker = {"outcome": "succeeded", "tree_empty": True}
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(comfy_runtime, "_validate_comfy_checkout", lambda _root: (_ for _ in ()).throw(RuntimeError("checkout failed")))
    with pytest.raises(RuntimeError, match="checkout failed"):
        runtime.generate(LTX23ComfyRequest("scene", 1280, 704, 1.0, 1), output_path=tmp_path / "out.mp4", progress=lambda *_args: None, check_cancelled=lambda: None)
    assert runtime.status()["last_worker"] == {
        "outcome": "failed", "spawned": False, "terminated": False, "tree_empty": False,
        "root_exited": False, "memory_boundary": "no_worker_spawned", "allocator_policy": None,
    }


def test_ltx23_comfy_rejects_concurrent_worker_ownership(tmp_path: Path, monkeypatch) -> None:
    runtime = ManagedLTX23ComfyRuntime(_request("comfy_dev_t2v"), comfy_root=tmp_path)
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    assert runtime._ownership.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            runtime.generate(
                LTX23ComfyRequest("scene", 1280, 704, 1.0, 1),
                output_path=tmp_path / "output.mp4",
                progress=lambda *_args: None,
                check_cancelled=lambda: None,
            )
    finally:
        runtime._ownership.release()


def test_ltx23_comfy_tool_cancellation_records_canceled_terminal_worker(tmp_path: Path, monkeypatch) -> None:
    runtime = ManagedLTX23ComfyRuntime(_request("comfy_dev_t2v"), comfy_root=tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Process:
        pid = 10

        def __init__(self) -> None:
            self.running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout=None) -> int:
            self.running = False
            return 0

    class Tree:
        def __init__(self, _process) -> None:
            pass

        def terminate(self) -> None:
            process.terminate()

        def wait_for_empty(self) -> None:
            return None

        def close(self) -> None:
            return None

    process = Process()
    monkeypatch.setattr(comfy_runtime, "revalidate_ltx23_comfy_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(comfy_runtime, "_validate_comfy_checkout", lambda _root: None)
    monkeypatch.setattr(comfy_runtime, "_workspace", lambda _recipe: workspace)
    monkeypatch.setattr(comfy_runtime, "_stage_components", lambda *_args: None)
    monkeypatch.setattr(comfy_runtime, "_start_comfy", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(comfy_runtime, "DisposableProcessTree", Tree)
    monkeypatch.setattr(
        comfy_runtime,
        "_wait_ready",
        lambda *_args: (_ for _ in ()).throw(ToolCancelled("Generation canceled")),
    )
    with pytest.raises(ToolCancelled):
        runtime.generate(
            LTX23ComfyRequest("scene", 1280, 704, 1.0, 1),
            output_path=tmp_path / "output.mp4",
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    last = runtime.status()["last_worker"]
    assert last["outcome"] == "canceled"
    assert last["tree_empty"] is True
    assert last["terminated"] is True


def test_ltx23_comfy_public_tool_retains_canceled_wrapper_for_manager_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolCancelled must not evict the sole status witness before API inspection."""

    settings = _settings(tmp_path)
    settings.ensure_directories()
    recipe = _request("comfy_dev_t2v")
    context = ToolContext(
        job_id=UUID(int=23),
        settings=settings,
        storage=Storage(settings),
        cancel_event=Event(),
        progress=lambda *_args: None,
        execution=ExecutionPlan(
            variant_key="ltx-2-3.text-to-video.comfy-dev-fp8",
            family="ltx23",
            recipe=recipe,
            optimizations={},
        ),
    )

    def cancel(self, *_args, **_kwargs):
        self._last_worker = {
            "outcome": "canceled",
            "terminated": True,
            "tree_empty": True,
            "memory_boundary": "disposable_process_exit",
        }
        raise ToolCancelled("Generation canceled")

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ManagedLTX23ComfyRuntime, "generate", cancel)
    try:
        with pytest.raises(ToolCancelled):
            LTX23TextToVideoTool().run(
                context,
                {"prompt": "scene", "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1},
            )
        statuses = RUNTIME_MANAGER.status()["runtimes"]
        assert len(statuses) == 1
        assert statuses[0]["key"] == (
            f"ltx23_comfy:comfy_dev_t2v:{recipe.component_fingerprint}"
        )
        assert statuses[0]["last_worker"]["outcome"] == "canceled"
        assert statuses[0]["last_worker"]["tree_empty"] is True
    finally:
        RUNTIME_MANAGER.clear()


@pytest.mark.parametrize(
    ("operation", "tool_class", "inputs"),
    [
        ("comfy_dev_t2v", LTX23TextToVideoTool, {"prompt": "scene", "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
        ("comfy_dev_i2v", LTX23FirstFrameToVideoTool, {"prompt": "scene", "start_image": {"type": "asset", "asset_id": str(UUID(int=1))}, "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
        ("comfy_distilled_flf", LTX23ImageToVideoTool, {"prompt": "scene", "start_image": {"type": "asset", "asset_id": str(UUID(int=1))}, "end_image": {"type": "asset", "asset_id": str(UUID(int=2))}, "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
    ],
)
def test_ltx23_comfy_public_tools_dispatch_each_operation_before_native_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    tool_class,
    inputs: dict[str, object],
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    recipe = _request(operation)
    context = ToolContext(
        job_id=UUID(int=89), settings=settings, storage=Storage(settings), cancel_event=Event(),
        progress=lambda *_args: None,
        execution=ExecutionPlan(variant_key="test", family="ltx23", recipe=recipe, optimizations={}),
    )
    tool = tool_class()
    called = []

    def dispatch(_context, _inputs):
        called.append(recipe.operation)
        raise ToolCancelled("Generation canceled")

    monkeypatch.setattr(tool, "_run_comfy" if operation == "comfy_dev_t2v" else "_run_comfy_condition", dispatch)
    monkeypatch.setattr(tool, "_resolve_plan", lambda *_args: pytest.fail("optimized dispatch entered native plan resolution"))
    with pytest.raises(ToolCancelled):
        tool.run(context, inputs)  # type: ignore[arg-type]
    assert called == [operation]


@pytest.mark.parametrize(
    ("operation", "tool_class", "inputs"),
    [
        ("comfy_dev_t2v", LTX23TextToVideoTool, {"prompt": "scene", "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
        ("comfy_dev_i2v", LTX23FirstFrameToVideoTool, {"prompt": "scene", "start_image": {"type": "asset", "asset_id": str(UUID(int=1))}, "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
        ("comfy_distilled_flf", LTX23ImageToVideoTool, {"prompt": "scene", "start_image": {"type": "asset", "asset_id": str(UUID(int=1))}, "end_image": {"type": "asset", "asset_id": str(UUID(int=2))}, "width": 1280, "height": 720, "duration_seconds": 1, "seed": 1}),
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_ltx23_comfy_public_operation_failures_retain_terminal_manager_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    tool_class,
    inputs: dict[str, object],
    cancel: bool,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    recipe = _request(operation)
    context = ToolContext(
        job_id=UUID(int=91), settings=settings, storage=Storage(settings), cancel_event=Event(),
        progress=lambda *_args: None,
        execution=ExecutionPlan(variant_key="test", family="ltx23", recipe=recipe, optimizations={}),
    )

    def terminal(self, *_args, **_kwargs):
        self._last_worker = {"outcome": "canceled" if cancel else "failed", "terminated": True, "tree_empty": True, "memory_boundary": "disposable_process_exit"}
        if cancel:
            raise ToolCancelled("Generation canceled")
        raise RuntimeError("synthetic failure")

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ManagedLTX23ComfyRuntime, "generate", terminal)
    monkeypatch.setattr(ToolContext, "resolve_asset", lambda *_args: tmp_path / "asset.png")
    try:
        with pytest.raises(ToolCancelled if cancel else RuntimeError):
            tool_class().run(context, inputs)  # type: ignore[arg-type]
        status = RUNTIME_MANAGER.status()["runtimes"]
        assert len(status) == 1
        assert status[0]["key"] == f"ltx23_comfy:{operation}:{recipe.component_fingerprint}"
        assert status[0]["last_worker"]["outcome"] == ("canceled" if cancel else "failed")
        assert status[0]["last_worker"]["tree_empty"] is True
    finally:
        RUNTIME_MANAGER.clear()
