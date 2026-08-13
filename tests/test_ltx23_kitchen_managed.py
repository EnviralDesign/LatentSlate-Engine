from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

import latentslate_engine.runtime.ltx23_kitchen_managed as managed_module
import latentslate_engine.runtime.ltx23_kitchen_worker as worker_module
from latentslate_engine.runtime.ltx23_kitchen_managed import ManagedLTX23KitchenRuntime


class _Request:
    operation = "ltx23_dev_t2v"
    fingerprint = "request-fingerprint"
    component_fingerprint = "component-fingerprint"

    def to_json_dict(self):
        return {
            "schema_version": 1,
            "family": "ltx23",
            "operation": self.operation,
            "base_model": "Lightricks/LTX-2.3",
            "execution_contract": {
                "workflow_revision": "2b7f823136606344f0bccce249898d771b809aa1",
                "workflow_sha256": (
                    "75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f"
                ),
                "node_semantics_revision": "725e6ec60621c6f001af04769173e7dbb3c53541",
                "kitchen_revision": "78e6dd22fe4ebe7bde5062e050a045dc3a244ee4",
                "pinned_workflow_default_width": 1280,
                "pinned_workflow_default_height": 720,
                "engine_acceptance_default_width": 768,
                "engine_acceptance_default_height": 512,
                "dimension_alignment": "dev=/64;distilled_flf=/32",
            },
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {"pipeline_support": {"path": "not-used-before-binding"}},
        }

    def public_component_manifest(self):
        return {"pipeline_support": {"component": "pipeline_support"}}


def _metadata(request: _Request, *, seed: int = 7, frames: int = 25) -> dict[str, object]:
    return {
        "family": "ltx23",
        "runtime": "engine-native/ltx23-kitchen",
        "operation": request.operation,
        "request_fingerprint": request.fingerprint,
        "component_fingerprint": request.component_fingerprint,
        "seed": seed,
        "width": 768,
        "height": 512,
        "num_frames": frames,
        "fps": 24,
        "audio_sample_rate": 48_000,
        "audio_channels": 2,
        "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_samples": 50_000,
        "video_duration_seconds": frames / 24,
        "audio_duration_seconds": 50_000 / 48_000,
        "output_sha256": "not-read-in-this-mocked-result",
        "components": request.public_component_manifest(),
        "native_fp8": {
            "complete": True,
            "modules": 2,
            "dispatched_modules": 2,
            "native_dispatch_count": 4,
            "dense_fallback_count": 0,
        },
        "native_text": {
            "backend": "comfy_kitchen/cuda/mixed-fp8-nvfp4",
            "module_count": 2,
            "total_dispatches": 4,
            "minimum_module_dispatches": 1,
        },
        "dense_base_dequantizations": 0,
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {name: tmp_path / name for name in ("request", "result", "progress", "gate")}


def test_worker_rejects_tamper_before_recipe_rehydration_or_heavy_import(
    tmp_path: Path, monkeypatch
):
    request = _Request()
    payload = managed_module._payload(
        request,
        {
            "prompt": "scene",
            "width": 768,
            "height": 512,
            "duration_seconds": 1.0,
            "num_frames": 25,
            "seed": 7,
            "start_image_path": None,
            "end_image_path": None,
            "output_path": str(tmp_path / "output.mp4"),
        },
        device="cuda",
    )
    tampered = copy.deepcopy(payload)
    tampered["generation"]["output_path"] = "C:/private/other.mp4"
    monkeypatch.setattr(
        worker_module,
        "_validate_generation_json",
        lambda _value: (_ for _ in ()).throw(AssertionError("parsed before binding")),
    )
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_bound_payload(tampered)


def test_worker_rejects_guide_content_changed_after_parent_binding(tmp_path: Path) -> None:
    guide = tmp_path / "guide.png"
    guide.write_bytes(b"first")
    generation = managed_module._generation(
        "ltx23_dev_i2v",
        prompt="scene",
        width=768,
        height=512,
        duration_seconds=1.0,
        num_frames=25,
        seed=7,
        start_image_path=guide,
        end_image_path=None,
        output_path=tmp_path / "output.mp4",
    )
    payload = managed_module._payload(_Request(), generation, device="cuda")
    guide.write_bytes(b"second")

    with pytest.raises(ValueError, match="endpoint changed"):
        worker_module._validate_bound_payload(payload)


def test_supervisor_recomputes_published_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "request_binding": "binding",
                "output_path": str(output),
                "output_size_bytes": 3,
                "metadata": {"output_sha256": "0" * 64},
                "allocator_policy": "expandable_segments:True",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="output hash"):
        managed_module._read_success(result, output, "binding")


def test_managed_success_proves_empty_tree_operation_and_native_metadata(
    tmp_path: Path, monkeypatch
):
    request = _Request()
    runtime = ManagedLTX23KitchenRuntime(request)  # type: ignore[arg-type]
    paths, events = _paths(tmp_path), []

    class _Process:
        pid = 42

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class _Tree:
        def __init__(self, process):
            self.process = process

        def wait_for_empty(self, timeout=15):
            events.append("empty")

        def close(self):
            events.append("close")

        def terminate(self):
            raise AssertionError("successful worker was terminated")

    output = tmp_path / "output.mp4"
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(managed_module, "_wait", lambda *_args: None)

    def result(_path, _expected, binding):
        output.write_bytes(b"mp4")
        return {
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": _metadata(request),
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    generated = runtime.generate(
        prompt="scene",
        output_path=output,
        width=768,
        height=512,
        duration_seconds=1,
        seed=7,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    assert generated.output_path == output.resolve()
    assert events == ["empty", "close"]
    status = runtime.status()
    assert status["last_worker"]["outcome"] == "succeeded"
    assert status["last_worker"]["tree_empty"] is True
    assert status["cache"] == {}
    assert all(not path.exists() for path in paths.values())


def test_cancellation_terminates_tree_and_removes_partial_output(tmp_path: Path, monkeypatch):
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths, events, output = _paths(tmp_path), [], tmp_path / "output.mp4"

    class _Process:
        pid = 99

        def poll(self):
            return None

    class _Tree:
        def __init__(self, process):
            self.process = process

        def terminate(self):
            events.append("terminate")

        def wait_for_empty(self, timeout=15):
            events.append("empty")

        def close(self):
            events.append("close")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)

    def cancel(*_args):
        output.write_bytes(b"partial")
        raise asyncio.CancelledError

    monkeypatch.setattr(managed_module, "_wait", cancel)
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert events == ["terminate", "empty", "close"]
    assert not output.exists()
    assert runtime.status()["last_worker"]["outcome"] == "canceled"


def test_tool_cancellation_is_classified_without_importing_the_tools_layer(
    tmp_path: Path, monkeypatch
):
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths, events = _paths(tmp_path), []

    class ToolCancelled(Exception):
        pass

    class _Process:
        pid = 101

        def poll(self):
            return 1 if events else None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            return 1

    class _Tree:
        def __init__(self, process):
            self.process = process

        def terminate(self):
            events.append("tree-terminate")

        def wait_for_empty(self, timeout=15):
            events.append("empty")

        def close(self):
            events.append("close")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(
        managed_module,
        "_wait",
        lambda *_args: (_ for _ in ()).throw(ToolCancelled()),
    )
    with pytest.raises(ToolCancelled):
        runtime.generate(
            prompt="scene",
            output_path=tmp_path / "output.mp4",
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert runtime.status()["last_worker"]["outcome"] == "canceled"


def test_generation_operation_identity_and_exact_temporal_alignment(tmp_path: Path):
    assert managed_module.frames_for_duration(1.0) == 25
    assert managed_module.frames_for_duration(1.05) == 33
    with pytest.raises(ValueError, match="must not receive"):
        managed_module._validate_generation(
            "ltx23_dev_t2v",
            {
                "prompt": "scene",
                "width": 768,
                "height": 512,
                "duration_seconds": 1.0,
                "num_frames": 25,
                "seed": 1,
                "start_image_path": "guide.png",
                "end_image_path": None,
                "start_image_identity": {
                    "size_bytes": 1,
                    "mtime_ns": 1,
                    "sha256": "0" * 64,
                },
                "end_image_identity": None,
                "output_path": str(tmp_path / "out.mp4"),
            },
        )
    with pytest.raises(ValueError, match="frame count"):
        managed_module._validate_generation(
            "ltx23_distilled_flf",
            {
                "prompt": "scene",
                "width": 768,
                "height": 512,
                "duration_seconds": 1.0,
                "num_frames": 33,
                "seed": 1,
                "start_image_path": None,
                "end_image_path": None,
                "start_image_identity": None,
                "end_image_identity": None,
                "output_path": str(tmp_path / "out.mp4"),
            },
        )


def test_failure_result_must_be_bound_and_output_cleanup_is_owned(tmp_path: Path):
    result = tmp_path / "result.json"
    result.write_text(
        '{"schema_version":1,"ok":false,"request_binding":"other","error_type":"RuntimeError","error":"private"}',
        encoding="utf-8",
    )
    assert (
        managed_module._worker_error(result, 1, "expected")
        == "LTX 2.3 Kitchen worker exited with code 1"
    )
    staging = tmp_path / ".out.mp4.part.tmp.mp4"
    staging.write_bytes(b"partial")
    assert managed_module._cleanup({}, tmp_path / "out.mp4") == []
    assert not staging.exists()


def test_worker_failure_publishes_only_safe_stage_diagnostics(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    fingerprint = "a" * 64
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "request_binding": "expected",
                "error_type": "TypeError",
                "error": "prompt and C:/private/path must never be returned",
                "failure_stage": "materialize_text_encoder",
                "error_fingerprint": fingerprint,
                "failure_location": "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder",
            }
        ),
        encoding="utf-8",
    )

    failure = managed_module._worker_failure(result, 1, "expected")
    assert failure == {
        "message": (
            "LTX 2.3 Kitchen worker failed (TypeError during materialize_text_encoder at "
            "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder; diagnostic "
            "aaaaaaaaaaaa)"
        ),
        "error_type": "TypeError",
        "stage": "materialize_text_encoder",
        "location": "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder",
        "fingerprint": fingerprint,
    }
    assert "private" not in failure["message"]
    assert "prompt" not in failure["message"]


def test_preexisting_output_is_never_deleted_on_pre_spawn_validation_failure(tmp_path: Path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fresh MP4"):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert output.read_bytes() == b"existing"
