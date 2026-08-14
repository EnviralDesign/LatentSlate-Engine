from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import latentslate_engine.runtime.ltx23_kitchen as kitchen_module
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
        "audio_duration_normalization": {
            "decoded_samples": 50_000,
            "target_samples": 50_000,
            "trimmed_samples": 0,
            "trailing_silence_samples": 0,
            "maximum_trailing_silence_samples": 1_920,
        },
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
        "residency_policy": {
            "mode": "grouped",
            "free_bytes": 16_000,
            "total_bytes": 16_000,
            "stored_bytes": 10_000,
            "reserved_headroom_bytes": 4_000,
            "stream_buffer_bytes": 1_000,
            "resident_weight_budget_bytes": 5_000,
            "reason": "test",
            "root_bytes": 1_000,
            "resident_block_count": 20,
            "resident_block_bytes": 4_000,
            "streamed_block_count": 28,
            "streamed_block_bytes": 5_000,
            "stream_buffer_count": 1,
            "streaming": "synchronous_cpu_master",
            "streamed_transitions": 28,
            "resident_refills": 1,
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / name
        for name in ("request", "result", "progress", "gate", "command")
    }


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


def test_worker_announces_start_before_runtime_import(tmp_path: Path, monkeypatch) -> None:
    request, result, progress, gate = (
        tmp_path / "request.json",
        tmp_path / "result.json",
        tmp_path / "progress.jsonl",
        tmp_path / "gate",
    )
    request.write_text("{}", encoding="utf-8")
    gate.touch()
    monkeypatch.setattr(
        worker_module,
        "_run",
        lambda *_args: {
            "request_binding": "binding",
            "output_path": str(tmp_path / "output.mp4"),
            "output_size_bytes": 1,
            "metadata": {},
            "allocator_policy": "expandable_segments:True",
        },
    )

    assert (
        worker_module.main(
            [
                "--request",
                str(request),
                "--result",
                str(result),
                "--progress",
                str(progress),
                "--start-gate",
                str(gate),
            ]
        )
        == 0
    )
    assert json.loads(progress.read_text(encoding="utf-8").splitlines()[0]) == {
        "message": "LTX worker started",
        "progress": 0.0,
    }


def test_worker_progress_log_phase_is_safe_and_specific() -> None:
    assert managed_module._worker_progress_phase("Importing LTX runtime") == "import_runtime"
    assert managed_module._worker_progress_phase("LTX denoise step 3/9") == "denoise"
    assert managed_module._worker_progress_phase("prompt: private scene") == "working"


def test_result_failure_is_detected_before_worker_process_exits(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "request_binding": "binding",
                "error_type": "TypeError",
                "error": "private child detail",
                "failure_stage": "materialize_text_encoder",
                "error_fingerprint": "a" * 64,
                "failure_location": "ltx23_kitchen_text.load",
            }
        ),
        encoding="utf-8",
    )
    assert managed_module._result_is_failure(result) is True
    failure = managed_module._worker_failure(result, 1, "binding")
    assert failure["stage"] == "materialize_text_encoder"
    assert "private child detail" not in failure["message"]


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
            return None

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
    monkeypatch.setattr(managed_module, "_wait_for_result", lambda *_args, **_kwargs: None)

    def result(_path, _expected, binding):
        output.write_bytes(b"mp4")
        return {
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": _metadata(request),
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    progress_events: list[tuple[float, str | None]] = []
    generated = runtime.generate(
        prompt="scene",
        output_path=output,
        width=768,
        height=512,
        duration_seconds=1,
        seed=7,
        progress=lambda value, message: progress_events.append((value, message)),
        check_cancelled=lambda: None,
    )
    assert generated.output_path == output.resolve()
    assert events == []
    status = runtime.status()
    assert status["last_worker"]["outcome"] == "succeeded"
    assert status["last_worker"]["tree_empty"] is False
    assert status["loaded"] is True
    assert status["cache"] == {"pipeline_warm": False}
    assert all(not paths[name].exists() for name in ("request", "result", "progress", "command"))
    assert paths["gate"].exists()
    assert progress_events[:2] == [
        (0.0, "Validating LTX runtime request"),
        (0.0, "Starting isolated LTX worker"),
    ]


def test_managed_session_reuses_one_worker_for_compatible_jobs(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    runtime = ManagedLTX23KitchenRuntime(request)  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    spawns: list[object] = []
    result_calls = 0

    class _Process:
        pid = 4242

        def poll(self):
            return None

    class _Tree:
        def __init__(self, _process):
            pass

        def terminate(self):
            pass

        def wait_for_empty(self, timeout=15):
            pass

        def close(self):
            pass

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (spawns.append(object()) or _Process()),
    )
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(managed_module, "_wait_for_result", lambda *_args, **_kwargs: None)

    def result(_path, expected, binding):
        nonlocal result_calls
        expected.write_bytes(b"mp4")
        metadata = _metadata(request, seed=result_calls + 1)
        metadata["cache"] = {"pipeline_warm": result_calls > 0}
        result_calls += 1
        return {
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": metadata,
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    first = runtime.generate(
        prompt="first", output_path=tmp_path / "first.mp4", width=768, height=512,
        duration_seconds=1, seed=1, progress=lambda *_args: None, check_cancelled=lambda: None,
    )
    second = runtime.generate(
        prompt="second", output_path=tmp_path / "second.mp4", width=768, height=512,
        duration_seconds=1, seed=2, progress=lambda *_args: None, check_cancelled=lambda: None,
    )

    assert len(spawns) == 1
    assert first.worker_pid == second.worker_pid == 4242
    assert first.worker_exit_code is None
    assert second.worker_exit_code is None
    assert first.metadata["cache"]["pipeline_warm"] is False
    assert second.metadata["cache"]["pipeline_warm"] is True
    assert runtime.status()["cache_support"] == {"prompt": False, "media": False}


def test_unload_clears_session_state_even_when_tree_close_fails(tmp_path: Path, monkeypatch) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)

    class _Process:
        pid = 77

        def poll(self):
            return None

    class _Tree:
        def terminate(self):
            raise OSError("terminate failed")

        def wait_for_empty(self, timeout=15):
            raise AssertionError("not reached")

        def close(self):
            raise OSError("close failed")

    session = managed_module._WorkerSession(_Process(), _Tree(), paths, "policy")
    runtime._session = session
    runtime._active_tree = session.tree
    with pytest.raises(OSError, match="terminate failed"):
        runtime.unload()
    assert runtime.status()["loaded"] is False
    assert runtime.status()["active_worker"] is False


@pytest.mark.parametrize("failure_point", ["popen", "tree"])
def test_failed_session_start_cleans_private_ipc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep")
    output = tmp_path / "target.mp4"
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    if failure_point == "popen":
        monkeypatch.setattr(
            managed_module.subprocess,
            "Popen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
        )
    else:
        class _Process:
            pid = 91

            def poll(self):
                return 0

        monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
        monkeypatch.setattr(
            managed_module,
            "DisposableProcessTree",
            lambda _process: (_ for _ in ()).throw(OSError("job object failed")),
        )
        monkeypatch.setattr(managed_module, "_terminate_direct_process", lambda _process: None)
    with pytest.raises(OSError):
        runtime.generate(
            prompt="scene", output_path=output, width=768, height=512, duration_seconds=1,
            seed=1, progress=lambda *_args: None, check_cancelled=lambda: None,
        )
    assert existing.read_bytes() == b"keep"
    assert all(not path.exists() for path in paths.values())


def test_unload_surfaces_close_only_error_after_clearing_session(tmp_path: Path) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)

    class _Process:
        pid = 78

        def poll(self):
            return None

    class _Tree:
        def terminate(self):
            pass

        def wait_for_empty(self, timeout=15):
            pass

        def close(self):
            raise OSError("close only")

    session = managed_module._WorkerSession(_Process(), _Tree(), paths, "policy")
    runtime._session = session
    runtime._active_tree = session.tree
    with pytest.raises(OSError, match="close only"):
        runtime.unload()
    assert runtime.status()["loaded"] is False


def test_ltx_runtime_reuses_materialized_components_then_unloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = kitchen_module.torch.device("cpu")
    runtime._components = None
    materialized: list[dict[str, object]] = []
    executed: list[dict[str, object]] = []
    released: list[dict[str, object]] = []
    components = {"transformer": object()}
    residency_closed: list[object] = []

    class _Residency:
        def __init__(self, transformer, _device):
            self.transformer = transformer

        def close(self):
            residency_closed.append(self.transformer)

    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True)
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", _Residency)
    monkeypatch.setattr(
        runtime,
        "_materialize",
        lambda *_args: (materialized.append(components) or components),
    )

    def execute(component_set, generation, **_kwargs):
        executed.append(component_set)
        return kitchen_module.LTX23KitchenResult(Path(generation.output_path), {})

    monkeypatch.setattr(runtime, "_execute", execute)
    monkeypatch.setattr(kitchen_module, "_release_components", lambda value, _device: released.append(value))
    first = SimpleNamespace(output_path=tmp_path / "one.mp4")
    second = SimpleNamespace(output_path=tmp_path / "two.mp4")
    runtime.generate(first, progress=lambda *_args: None, check_cancelled=lambda: None)
    runtime.generate(second, progress=lambda *_args: None, check_cancelled=lambda: None)
    runtime.unload()

    assert materialized == [components]
    assert executed == [components, components]
    assert released == [components]
    assert residency_closed == [components["transformer"]]


def test_worker_session_reuses_runtime_and_rejects_mismatched_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command_path = tmp_path / "command.json"
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.jsonl"
    outputs = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    initial, second, mismatch = object(), object(), object()
    request_data = {initial: "recipe-a", second: "recipe-a", mismatch: "recipe-b"}
    generations = {
        initial: {"output_path": str(outputs[0]), "prompt": "one", "width": 1, "height": 1, "num_frames": 1, "seed": 1, "start_image_path": None, "end_image_path": None, "start_image_identity": None, "end_image_identity": None},
        second: {"output_path": str(outputs[1]), "prompt": "two", "width": 1, "height": 1, "num_frames": 1, "seed": 2, "start_image_path": None, "end_image_path": None, "start_image_identity": None, "end_image_identity": None},
        mismatch: {"output_path": str(tmp_path / "bad.mp4"), "prompt": "bad", "width": 1, "height": 1, "num_frames": 1, "seed": 3, "start_image_path": None, "end_image_path": None, "start_image_identity": None, "end_image_identity": None},
    }
    writes: list[dict[str, object]] = []
    created: list[object] = []
    commands = iter((second, mismatch))

    monkeypatch.setattr(
        worker_module,
        "_validate_bound_payload",
        lambda payload: ({"recipe": request_data[payload]}, generations[payload], "cuda", f"binding-{generations[payload]['seed']}"),
    )
    monkeypatch.setattr(
        "latentslate_engine.ltx23_kitchen_recipe.rehydrate_ltx23_kitchen_runtime_request",
        lambda value: SimpleNamespace(operation="ltx23_dev_t2v", fingerprint=value["recipe"]),
    )

    class _Generation:
        def __init__(self, *args):
            self.output_path = args[1]

    class _Runtime:
        def __init__(self, request, **_kwargs):
            created.append(request)

        def generate(self, generation, **_kwargs):
            Path(generation.output_path).write_bytes(b"mp4")
            return SimpleNamespace(metadata={"cache": {"pipeline_warm": len(writes) > 0}})

    monkeypatch.setattr(kitchen_module, "LTX23KitchenGeneration", _Generation)
    monkeypatch.setattr(kitchen_module, "LTX23KitchenRuntime", _Runtime)
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(worker_module, "_wait_command", lambda _path: next(commands))
    monkeypatch.setattr(worker_module, "_write_json", lambda _path, value: writes.append(dict(value)))
    monkeypatch.setattr(worker_module, "_append_progress", lambda *_args: None)

    with pytest.raises(ValueError, match="does not match"):
        worker_module._run_session(initial, result_path, progress_path, command_path, worker_module._FailureContext())
    assert len(created) == 1
    assert [item["request_binding"] for item in writes] == ["binding-1", "binding-2"]
    assert writes[1]["metadata"]["cache"]["pipeline_warm"] is True


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

    monkeypatch.setattr(managed_module, "_wait_for_result", cancel)
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
        "_wait_for_result",
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


def test_worker_failure_publishes_safe_diagnostics_and_logs_detail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
        "log_detail": "prompt and C:/private/path must never be returned",
    }
    assert "private" not in failure["message"]
    assert "prompt" not in failure["message"]
    with caplog.at_level("ERROR"):
        managed_module._log_worker_failure(failure)
    assert "TypeError" in caplog.text
    assert "materialize_text_encoder" in caplog.text
    assert "prompt and C:/private/path must never be returned" in caplog.text


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
