from __future__ import annotations

import json
import struct
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latentslate_engine.artifacts import _wan5_lora_signals, probe_safetensors
from latentslate_engine.runtime.wan5_comfy import (
    WAN5_CFG,
    WAN5_SAMPLER,
    WAN5_SCHEDULER,
    WAN5_SHIFT,
    WAN5_STEPS,
    ManagedWan5ComfyRuntime,
    Wan5ComfyI2VRequest,
    Wan5ComfyRequest,
    validate_wan5_comfy_request,
)
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


def test_artifact_probe_recognizes_exact_wan5_and_wan22_vae_signatures(tmp_path: Path):
    transformer = _safetensors(
        tmp_path / "transformer.safetensors",
        {
            **{
                f"blocks.{index}.self_attn.q.weight": ("F16", [1])
                for index in range(30)
            },
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

    i2v_recipe = Wan5RuntimeRequest(
        1, "wan22", "image_to_video", "test", files, identities
    )
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
    shapes.pop("diffusion_model.blocks.29.ffn.2.lora_B.weight")
    assert _wan5_lora_signals(list(shapes), shapes) == ((), (), ())
