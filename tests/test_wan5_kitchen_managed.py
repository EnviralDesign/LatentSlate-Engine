from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

import latentslate_engine.runtime.wan5_kitchen_managed as managed_module
import latentslate_engine.runtime.wan5_kitchen_worker as worker_module
from latentslate_engine.runtime.wan5_kitchen_managed import ManagedWan5KitchenRuntime


class _Request:
    operation = "wan5_t2v"
    fingerprint = "request-fingerprint"
    component_fingerprint = "component-fingerprint"

    def to_json_dict(self):
        return {
            "schema_version": 1,
            "family": "wan22",
            "operation": self.operation,
            "base_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "execution_contract": {},
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {},
        }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        **{name: tmp_path / name for name in ("request", "result", "progress", "gate")},
        "staging": tmp_path / "staging.mp4",
    }


def _metadata(request: _Request) -> dict[str, object]:
    return {
        "family": "wan22",
        "runtime": "engine-native/wan5-kitchen",
        "operation": request.operation,
        "request_fingerprint": request.fingerprint,
        "component_fingerprint": request.component_fingerprint,
        "seed": 7,
        "width": 1280,
        "height": 704,
        "frame_count": 25,
        "sampling": {
            "steps": 30,
            "source_schedule_steps": 31,
            "discard_penultimate_sigma": True,
            "guidance_scale": 5.0,
            "sampler_runtime": "diffusers/UniPCMultistepScheduler",
        },
        "kitchen_dispatch": {
            "proven": True,
            "fallback_calls": 0,
            "native_modules": 168,
            "expected_modules": 168,
        },
        "output": {"fps": 24, "frame_count": 25, "has_audio": False},
    }


def test_worker_rejects_tamper_before_recipe_rehydration(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    generation = managed_module._generation(
        request.operation,
        prompt="scene",
        output_path=tmp_path / "output.mp4",
        width=1280,
        height=704,
        num_frames=25,
        seed=7,
        start_image_path=None,
        staging_output_path=tmp_path / "staging.mp4",
    )
    payload = managed_module._payload(request, generation)  # type: ignore[arg-type]
    tampered = copy.deepcopy(payload)
    tampered["generation"]["seed"] = 8
    monkeypatch.setattr(
        worker_module,
        "_validate_generation",
        lambda *_args: (_ for _ in ()).throw(AssertionError("parsed before binding")),
    )

    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_binding(tampered)


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


def test_managed_success_proves_empty_worker_tree(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    runtime = ManagedWan5KitchenRuntime(request)  # type: ignore[arg-type]
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

        def wait_for_empty(self):
            events.append("empty")

        def close(self):
            events.append("close")

        def terminate(self):
            raise AssertionError("successful worker was terminated")

    output = tmp_path / "output.mp4"
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_wan5_kitchen_runtime_request", lambda _request: True
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
        width=1280,
        height=704,
        num_frames=25,
        seed=7,
        start_image_path=None,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )

    assert generated.output_path == output.resolve()
    assert events == ["empty", "close"]
    assert runtime.status()["last_worker"]["outcome"] == "succeeded"
    assert runtime.status()["last_worker"]["tree_empty"] is True
    assert all(not path.exists() for path in paths.values())


def test_cancellation_terminates_worker_and_removes_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = ManagedWan5KitchenRuntime(_Request())  # type: ignore[arg-type]
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

        def wait_for_empty(self):
            events.append("empty")

        def close(self):
            events.append("close")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_wan5_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)

    def cancel(*_args):
        output.write_bytes(b"partial")
        paths["staging"].write_bytes(b"partial staging")
        raise asyncio.CancelledError

    monkeypatch.setattr(managed_module, "_wait", cancel)
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=1280,
            height=704,
            num_frames=25,
            seed=7,
            start_image_path=None,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert events == ["terminate", "empty", "close"]
    assert not output.exists()
    assert not paths["staging"].exists()
    assert runtime.status()["last_worker"]["outcome"] == "canceled"
