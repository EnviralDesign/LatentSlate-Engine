from __future__ import annotations

import json
import struct
from pathlib import Path
from subprocess import CompletedProcess
from threading import Event
from uuid import UUID

import pytest

import latentslate_engine.runtime.wan5_comfy as runtime_module
import latentslate_engine.tools.wan5_comfy as tool_module
from latentslate_engine.artifacts import _shape_signals, _wan5_lora_signals, probe_safetensors
from latentslate_engine.config import Settings
from latentslate_engine.resources import ResourceDescriptor
from latentslate_engine.runtime.wan5_comfy import (
    WAN5_CFG,
    WAN5_SAMPLER,
    WAN5_SCHEDULER,
    WAN5_SHIFT,
    WAN5_STEPS,
    ManagedWan5ComfyRuntime,
    Wan5ComfyI2VRequest,
    Wan5ComfyLora,
    Wan5ComfyRequest,
    validate_wan5_comfy_request,
)
from latentslate_engine.storage import Storage
from latentslate_engine.tools.base import ConfiguredLora, ExecutionPlan, LoraExecution, ToolContext
from latentslate_engine.tools.wan5_comfy import (
    Wan5ComfyImageToVideoTool,
    Wan5ComfyTextToVideoTool,
    _source_image_manifest,
)
from latentslate_engine.wan22_ti2v5b_recipe import WAN5_COMFY_RUNTIME_REVISION, Wan5RuntimeRequest


def _safetensors(path: Path, entries: dict[str, tuple[str, list[int]]]) -> Path:
    offset = 0
    header: dict[str, object] = {}
    sizes = {"F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    for key, (dtype, shape) in entries.items():
        size = sizes[dtype]
        for value in shape:
            size *= value
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0" * offset)
    return path


def _runtime_recipe(tmp_path: Path, operation: str = "text_to_video") -> Wan5RuntimeRequest:
    files = {}
    identities = {}
    for role in ("transformer", "text_encoder", "vae"):
        path = _safetensors(
            tmp_path / f"{role}.safetensors",
            {f"{role}.weight": ("F16", [1])},
        )
        identity = probe_safetensors(path).identity
        identities[role] = identity
        files[role] = {
            "path": str(path),
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
        }
    return Wan5RuntimeRequest(1, "wan22", operation, "test", files, identities)


def test_artifact_probe_recognizes_exact_wan5_and_wan22_vae_signatures(tmp_path: Path):
    transformer = _safetensors(
        tmp_path / "transformer.safetensors",
        {
            **{f"blocks.{index}.self_attn.q.weight": ("F16", [1]) for index in range(30)},
            "blocks.0.cross_attn.k.weight": ("F16", [1]),
            "blocks.0.ffn.0.weight": ("F16", [14336, 3072]),
            "patch_embedding.weight": ("F16", [3072, 48, 1, 2, 2]),
            "head.modulation": ("F16", [1, 2, 3072]),
            "head.head.weight": ("F16", [192, 3072]),
        },
    )
    probe = probe_safetensors(transformer)
    assert probe.family_signals == ("wan22",)
    assert probe.architecture_signals == ("wan22_ti2v_5b_48ch_30block",)
    assert probe.component_signals == ("transformer",)

    vae = _safetensors(
        tmp_path / "vae.safetensors",
        {
            "decoder.middle.0.residual.0.gamma": ("F16", [1024, 1, 1, 1]),
            "decoder.upsamples.0.upsamples.0.residual.2.weight": ("F16", [1]),
            "decoder.conv1.weight": ("F16", [1024, 48, 3, 3, 3]),
            "encoder.head.2.weight": ("F16", [96, 640, 3, 3, 3]),
        },
    )
    probe = probe_safetensors(vae)
    assert probe.family_signals == ("wan22",)
    assert probe.architecture_signals == ("wan_vae_2_2_48ch",)
    assert probe.component_signals == ("vae",)


def test_t2v_request_contract_has_no_source_image():
    request = Wan5ComfyRequest(
        prompt="moving fox",
        width=128,
        height=96,
        num_frames=5,
    )
    validate_wan5_comfy_request(request, "text_to_video")
    assert "source_image" not in request.__dataclass_fields__


def test_wan5_zero_strength_lora_bypasses_the_strict_probe_and_runtime_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recipe = _runtime_recipe(tmp_path)
    disabled_path = tmp_path / "missing-disabled-lora.safetensors"
    execution = ExecutionPlan(
        variant_key="test.wan5.zero",
        family="wan22",
        loras=(LoraExecution("style", "lora:wan22:disabled", disabled_path, 0.0),),
        configured_loras=(
            ConfiguredLora("style", "lora:wan22:disabled", 0.0, False),
        ),
        recipe=recipe,
        optimizations={},
    )
    engine_home = tmp_path / "engine-home"
    config = Settings(
        home=engine_home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="unused",
        h3_device="cpu",
    )
    config.ensure_directories()
    context = ToolContext(
        job_id=UUID(int=42),
        settings=config,
        storage=Storage(config),
        cancel_event=Event(),
        progress=lambda _value, _message: None,
        execution=execution,
    )

    class Runtime:
        def generate(self, *_args, **_kwargs):
            return type("Result", (), {"provenance": {"graph_has_lora": False}})()

    monkeypatch.setattr(
        tool_module,
        "probe_artifact",
        lambda _path: pytest.fail("a zero-strength Wan5 LoRA must not be probed"),
    )
    monkeypatch.setattr(tool_module.RUNTIME_MANAGER, "activate", lambda *_args: Runtime())

    artifacts = Wan5ComfyTextToVideoTool().run(
        context,
        {
            "prompt": "base only",
            "negative_prompt": "",
            "num_frames": 5,
            "width": 128,
            "height": 96,
            "seed": 0,
        },
    )

    assert len(artifacts) == 1
    assert artifacts[0].metadata["lora"] is None
    assert artifacts[0].metadata["configured_loras"] == [
        {
            "slot": "style",
            "resource_reference": "lora:wan22:disabled",
            "strength": 0.0,
            "active": False,
        }
    ]
    assert context.runtime_provenance["configured_loras"] == artifacts[0].metadata["configured_loras"]
    assert context.runtime_provenance["runtime_result"]["graph_has_lora"] is False


def test_i2v_is_a_distinct_required_image_contract(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    request = Wan5ComfyI2VRequest(
        prompt="moving fox", source_image=source, width=128, height=96, num_frames=5
    )
    validate_wan5_comfy_request(request, "image_to_video")
    with pytest.raises(ValueError, match="requires its image request"):
        validate_wan5_comfy_request(
            Wan5ComfyRequest(prompt="moving fox", width=128, height=96, num_frames=5),
            "image_to_video",
        )
    t2v_inputs = {value.key: value for value in Wan5ComfyTextToVideoTool().descriptor.inputs}
    i2v_inputs = {value.key: value for value in Wan5ComfyImageToVideoTool().descriptor.inputs}
    assert "source_image" not in t2v_inputs
    assert i2v_inputs["source_image"].required is True


def test_workflow_is_the_frozen_official_comfy_schedule_and_conditioning(tmp_path: Path):
    files = {}
    identities = {}
    for role in ("transformer", "text_encoder", "vae"):
        path = _safetensors(tmp_path / f"{role}.safetensors", {f"{role}.weight": ("F16", [1])})
        identity = probe_safetensors(path).identity
        identities[role] = identity
        files[role] = {
            "path": str(path),
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
        }
    recipe = Wan5RuntimeRequest(
        1,
        "wan22",
        "text_to_video",
        "test",
        files,
        identities,
    )
    runtime = ManagedWan5ComfyRuntime(recipe, comfy_root=tmp_path)
    workflow = runtime._workflow(
        Wan5ComfyRequest(prompt="fox", width=128, height=96, num_frames=5, seed=7)
    )
    assert workflow["4"]["inputs"]["shift"] == WAN5_SHIFT
    assert workflow["8"]["inputs"] == {
        "model": ["4", 0],
        "positive": ["5", 0],
        "negative": ["6", 0],
        "latent_image": ["7", 0],
        "seed": 7,
        "steps": WAN5_STEPS,
        "cfg": WAN5_CFG,
        "sampler_name": WAN5_SAMPLER,
        "scheduler": WAN5_SCHEDULER,
        "denoise": 1.0,
    }
    assert "start_image" not in workflow["7"]["inputs"]

    i2v_recipe = Wan5RuntimeRequest(1, "wan22", "image_to_video", "test", files, identities)
    assert i2v_recipe.component_fingerprint == recipe.component_fingerprint
    assert i2v_recipe.fingerprint != recipe.fingerprint
    workflow = runtime._workflow(
        Wan5ComfyI2VRequest(
            prompt="fox",
            source_image=tmp_path / "source.png",
            width=128,
            height=96,
            num_frames=5,
            seed=7,
        ),
        upload_name="source.png",
    )
    assert workflow["11"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "source.png"},
    }
    assert workflow["7"]["inputs"]["start_image"] == ["11", 0]
    workflow = runtime._workflow(
        Wan5ComfyRequest(prompt="fox", width=128, height=96, num_frames=5),
        lora_name="exact-lora.safetensors",
        lora_strength=0.75,
    )
    assert workflow["12"] == {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "exact-lora.safetensors",
            "strength_model": 0.75,
        },
    }
    assert workflow["4"]["inputs"]["model"] == ["12", 0]


def test_component_workspace_ignores_os_temp_and_hardlinks_on_resource_volume(
    tmp_path: Path,
    monkeypatch,
):
    recipe = _runtime_recipe(tmp_path)
    runtime = ManagedWan5ComfyRuntime(recipe, comfy_root=tmp_path)
    real_temporary_directory = runtime_module.TemporaryDirectory
    requested_parents = []

    def tracked_temporary_directory(*args, **kwargs):
        requested_parents.append(Path(kwargs["dir"]))
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "TemporaryDirectory", tracked_temporary_directory)
    root = runtime._prepare_workspace()
    try:
        assert requested_parents == [tmp_path]
        for role, folder in (
            ("transformer", "diffusion_models"),
            ("text_encoder", "text_encoders"),
            ("vae", "vae"),
        ):
            source = Path(str(recipe.components[role]["path"]))
            staged = root / "models" / folder / source.name
            assert staged.is_file()
            assert staged.stat().st_dev == source.stat().st_dev
            assert staged.stat().st_ino == source.stat().st_ino
    finally:
        runtime.unload()


def test_cross_volume_lora_staging_fails_closed_without_copy(tmp_path: Path, monkeypatch):
    runtime = ManagedWan5ComfyRuntime(_runtime_recipe(tmp_path), comfy_root=tmp_path)
    runtime._workspace = __import__("tempfile").TemporaryDirectory(dir=tmp_path)
    lora_path = tmp_path / "lora.safetensors"
    lora_path.write_bytes(b"small fixture")
    real_stat = Path.stat

    def different_device_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == lora_path:
            return type("Stat", (), {"st_dev": result.st_dev + 1})()
        return result

    monkeypatch.setattr(Path, "stat", different_device_stat)
    monkeypatch.setattr(runtime_module.shutil, "copyfile", pytest.fail)
    monkeypatch.setattr(runtime_module.os, "link", pytest.fail)
    with pytest.raises(RuntimeError, match="component volume for zero-copy staging"):
        runtime._stage_lora(Wan5ComfyLora("lora:test", lora_path, 1.0, "a" * 64, "b" * 64, 32))
    runtime.unload()


def test_runtime_status_tracks_each_switched_recipe(tmp_path: Path, monkeypatch):
    t2v_recipe = _runtime_recipe(tmp_path)
    i2v_recipe = Wan5RuntimeRequest(
        1,
        "wan22",
        "image_to_video",
        "test",
        t2v_recipe.components,
        t2v_recipe.identities,
    )
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    runtime = ManagedWan5ComfyRuntime(t2v_recipe, comfy_root=tmp_path)
    monkeypatch.setattr(runtime_module, "revalidate_wan5_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(runtime, "_ensure_server", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_upload_image", lambda _source: "source.png")
    monkeypatch.setattr(runtime, "_queue_prompt", lambda _workflow: "prompt-id")
    monkeypatch.setattr(runtime, "_wait_for_output", lambda *_args, **_kwargs: {"filename": "x"})
    monkeypatch.setattr(runtime, "_download_output", lambda *_args, **_kwargs: None)

    assert runtime.status()["operation"] is None
    runtime.generate(
        Wan5ComfyI2VRequest(prompt="fox", source_image=source, width=128, height=96, num_frames=5),
        recipe=i2v_recipe,
        output_path=tmp_path / "i2v.webm",
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    assert runtime.status()["operation"] == "image_to_video"
    assert runtime.status()["recipe_fingerprint"] == i2v_recipe.fingerprint
    runtime.generate(
        Wan5ComfyRequest(prompt="fox", width=128, height=96, num_frames=5),
        recipe=t2v_recipe,
        output_path=tmp_path / "t2v.webm",
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    assert runtime.status()["operation"] == "text_to_video"
    assert runtime.status()["recipe_fingerprint"] == t2v_recipe.fingerprint


def test_managed_runtime_unloads_and_cleans_workspace(tmp_path: Path):
    runtime = ManagedWan5ComfyRuntime.__new__(ManagedWan5ComfyRuntime)
    runtime._lock = __import__("threading").RLock()
    runtime._process = None
    runtime._base_url = "http://127.0.0.1:1"
    log = (tmp_path / "worker.log").open("w", encoding="utf-8")
    runtime._log_handle = log
    runtime._workspace = __import__("tempfile").TemporaryDirectory(dir=tmp_path)
    workspace = Path(runtime._workspace.name)
    runtime.unload()
    assert runtime._base_url is None
    assert runtime._workspace is None
    assert not workspace.exists()
    assert log.closed


def test_managed_runtime_retries_transient_windows_workspace_lock(monkeypatch):
    class Workspace:
        attempts = 0

        def cleanup(self):
            self.attempts += 1
            if self.attempts == 1:
                raise PermissionError("transient Windows handle")

    runtime = ManagedWan5ComfyRuntime.__new__(ManagedWan5ComfyRuntime)
    runtime._lock = __import__("threading").RLock()
    runtime._process = None
    runtime._base_url = "http://127.0.0.1:1"
    runtime._log_handle = None
    runtime._workspace = Workspace()
    monkeypatch.setattr("latentslate_engine.runtime.wan5_comfy.time.sleep", lambda _: None)
    runtime.unload()
    assert runtime._workspace is None


def test_managed_runtime_rejects_an_unpinned_comfy_checkout(tmp_path: Path, monkeypatch):
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / ".venv" / "Scripts").mkdir(parents=True)
    (comfy_root / ".venv" / "Scripts" / "python.exe").touch()
    (comfy_root / "main.py").touch()
    runtime = ManagedWan5ComfyRuntime.__new__(ManagedWan5ComfyRuntime)
    runtime.comfy_root = comfy_root
    runtime._process = None
    runtime._base_url = None
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan5_comfy.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="wrong-revision\n"),
    )
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        runtime._ensure_server(check_cancelled=lambda: None)


def test_pinned_runtime_revision_is_an_immutable_git_commit():
    assert len(WAN5_COMFY_RUNTIME_REVISION) == 40
    int(WAN5_COMFY_RUNTIME_REVISION, 16)


def test_i2v_source_manifest_records_exact_center_crop_and_anchor(tmp_path: Path):
    from PIL import Image

    source = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), (20, 80, 160)).save(source)
    manifest = _source_image_manifest(source, target_width=128, target_height=96)
    assert manifest["width"] == 200
    assert manifest["height"] == 100
    assert manifest["preprocessing"] == {
        "node": "Wan22ImageToVideoLatent",
        "resize": "bilinear",
        "crop": "center",
        "crop_box": [33, 0, 167, 100],
        "target_width": 128,
        "target_height": 96,
        "vae_encode": True,
        "first_latent_anchor": True,
    }
    assert len(manifest["sha256"]) == 64


def test_wan5_lora_signature_requires_the_complete_30_block_topology():
    modules = (
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    )
    shapes = {}
    for block in range(30):
        for module in modules:
            input_dim, output_dim = (
                (14336, 3072)
                if module == "ffn.2"
                else (3072, 14336)
                if module == "ffn.0"
                else (3072, 3072)
            )
            stem = f"diffusion_model.blocks.{block}.{module}"
            shapes[f"{stem}.lora_A.weight"] = (32, input_dim)
            shapes[f"{stem}.lora_B.weight"] = (output_dim, 32)
    assert _wan5_lora_signals(list(shapes), shapes) == (
        ("wan22",),
        ("wan22_ti2v_5b_lora_30block",),
        ("lora",),
    )
    assert _shape_signals(shapes, list(shapes))["lora_rank"] == 32
    shapes.pop("diffusion_model.blocks.29.ffn.2.lora_B.weight")
    assert _wan5_lora_signals(list(shapes), shapes) == ((), (), ())


def test_wan5_lora_declared_rank_must_match_observed_header(monkeypatch, tmp_path: Path):
    resource = ResourceDescriptor.model_validate(
        {
            "id": "lora:wan22:test/rank",
            "kind": "lora",
            "family": "wan22",
            "name": "rank mismatch",
            "relative_path": "loras/wan22/rank.safetensors",
            "format": "safetensors",
            "precision": "bf16",
            "quantization": "native",
            "size_bytes": 1,
            "base_model": "Wan-AI/Wan2.2-TI2V-5B",
            "metadata": {
                "architecture": "wan22_ti2v_5b_lora_30block",
                "schema_sha256": "a" * 64,
                "rank": 64,
            },
            "sources": [
                {
                    "type": "huggingface",
                    "repo_id": "owner/repo",
                    "revision": "b" * 40,
                    "filename": "rank.safetensors",
                    "sha256": "c" * 64,
                }
            ],
        }
    )
    probe = type(
        "Probe",
        (),
        {
            "architecture_signals": ("wan22_ti2v_5b_lora_30block",),
            "component_signals": ("lora",),
            "schema_sha256": "a" * 64,
            "key_shape_signals": {"lora_rank": 32},
        },
    )()
    monkeypatch.setattr(tool_module, "probe_artifact", lambda _path: probe)
    errors = Wan5ComfyTextToVideoTool().validate_lora_resource(
        resource, tmp_path / "rank.safetensors"
    )
    assert errors == [
        "LoRA declared rank does not match its header topology: declared 64, observed 32"
    ]
