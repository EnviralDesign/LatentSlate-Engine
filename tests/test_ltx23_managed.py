from __future__ import annotations

import asyncio
import copy
import threading
from pathlib import Path

import pytest

import latentslate_engine.runtime.ltx23_managed as managed_module
import latentslate_engine.runtime.ltx23_worker as worker_module
from latentslate_engine.config import Settings
from latentslate_engine.runtime.kit import ResolvedRuntimePlan, RuntimeComponent
from latentslate_engine.runtime.ltx23_managed import ManagedLTX23Runtime


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1,
        h3_model_id="unused",
        h3_profile="unused",
        h3_device="cpu",
        ltx23_device="cpu",
    )


def _plan(tmp_path: Path) -> ResolvedRuntimePlan:
    root = tmp_path / "model"
    root.mkdir()
    (root / "marker").write_text("immutable", encoding="utf-8")
    return ResolvedRuntimePlan(
        family="ltx23",
        variant_key="test",
        model_id="diffusers/LTX-2.3-Distilled-Diffusers",
        model_resource_id="model:ltx23:test",
        model_path=root,
        model_format="diffusers",
        model_precision="bf16",
        model_quantization="native",
        device="cpu",
        quantization="bf16",
        attention="native",
        offload="sequential",
        compile=False,
        compile_mode="default",
        compile_fullgraph=False,
        compile_dynamic=False,
        vae_tiling="on",
        vae_slicing="off",
        cache="prompt",
        group_offload_blocks=1,
        group_offload_use_stream=False,
        group_offload_record_stream=False,
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
        components=(RuntimeComponent.capture("model", root),),
    )


def test_ltx_managed_cancellation_is_checked_before_a_worker_starts(tmp_path: Path, monkeypatch):
    runtime = ManagedLTX23Runtime(_settings(tmp_path), _plan(tmp_path), operation="t2v")
    monkeypatch.setattr(
        managed_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker started")),
    )
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            plan=runtime.plan,
            prompt="scene",
            output_path=tmp_path / "output.mp4",
            width=768,
            height=512,
            duration_seconds=1,
            seed=1,
            progress=lambda *_args: None,
            check_cancelled=lambda: (_ for _ in ()).throw(asyncio.CancelledError()),
        )


def test_ltx_managed_success_requires_tree_empty_request_bound_result_and_allocator_policy(
    tmp_path: Path, monkeypatch
):
    plan = _plan(tmp_path)
    runtime = ManagedLTX23Runtime(_settings(tmp_path), plan, operation="t2v")
    output = tmp_path / "output.mp4"
    paths = {key: tmp_path / key for key in ("request", "result", "progress", "gate")}
    environment = {}
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    class Process:
        pid = 123

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    events = []

    class Tree:
        def __init__(self, process):
            self.process = process

        def wait_for_empty(self, timeout=15):
            events.append("empty")

        def close(self):
            events.append("close")

        def terminate(self):
            raise AssertionError("clean worker was terminated")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **kwargs: environment.update(kwargs["env"]) or Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", Tree)
    monkeypatch.setattr(managed_module, "_wait", lambda *_args: None)

    def result(_path, _output, binding):
        output.write_bytes(b"mp4")
        return {
            "metadata": {
                "pipeline_fingerprint": plan.pipeline_fingerprint,
                "seed": 1,
                "model_id": "model:ltx23:test",
            },
            "allocator_policy": "backend:cudaMallocAsync",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    metadata = runtime.generate(
        plan=plan, prompt="scene", output_path=output, width=768, height=512,
        duration_seconds=1, seed=1, progress=lambda *_args: None, check_cancelled=lambda: None,
    )
    assert metadata["pipeline_fingerprint"] == plan.pipeline_fingerprint
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
    assert events == ["empty", "close"]
    last_worker = runtime.status()["last_worker"]
    assert last_worker["memory_boundary"] == "disposable_process_exit"
    assert last_worker["outcome"] == "succeeded"
    assert last_worker["tree_empty"] is True
    assert last_worker["allocator_policy"] == "backend:cudaMallocAsync"
    assert all(not item.exists() for item in paths.values())


def test_ltx_managed_cancellation_terminates_tree_and_removes_partial_output(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    runtime = ManagedLTX23Runtime(_settings(tmp_path), plan, operation="t2v")
    output = tmp_path / "output.mp4"
    paths = {key: tmp_path / key for key in ("request", "result", "progress", "gate")}

    class Process:
        pid = 55

        def poll(self):
            return None

    events = []

    class Tree:
        def __init__(self, process):
            self.process = process

        def terminate(self):
            events.append("terminate")

        def wait_for_empty(self, timeout=15):
            events.append("empty")

        def close(self):
            events.append("close")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", Tree)

    def cancel(*_args):
        output.write_bytes(b"partial")
        raise asyncio.CancelledError()

    monkeypatch.setattr(managed_module, "_wait", cancel)
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            plan=plan, prompt="scene", output_path=output, width=768, height=512,
            duration_seconds=1, seed=1, progress=lambda *_args: None, check_cancelled=lambda: None,
        )
    assert events == ["terminate", "empty", "close"]
    assert not output.exists()
    last_worker = runtime.status()["last_worker"]
    assert last_worker["outcome"] == "canceled"
    assert last_worker["tree_empty"] is True


def test_ltx_worker_result_rejects_wrong_binding_or_output(tmp_path: Path):
    output = tmp_path / "expected.mp4"
    output.write_bytes(b"ok")
    result = tmp_path / "result.json"
    result.write_text(
        '{"schema_version":1,"ok":true,"request_binding":"other","output_path":"'
        + str(output).replace("\\", "\\\\")
        + '","output_size_bytes":2,"metadata":{},"allocator_policy":"expandable_segments:True"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        managed_module._read_success(result, output, "expected")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["generation"].update(seed=2),
        lambda payload: payload["generation"].update(output_path="C:/unexpected.mp4"),
        lambda payload: payload.update(operation="first_frame"),
        lambda payload: payload["settings"].update(device="cuda"),
        lambda payload: payload["plan"].update(attention="other"),
        lambda payload: payload["plan"].update(model_path="C:/unexpected-model"),
    ),
)
def test_ltx_worker_recomputes_binding_over_operation_paths_and_seed(tmp_path: Path, mutation):
    payload = managed_module._payload(
        _settings(tmp_path), _plan(tmp_path), operation="t2v", prompt="scene",
        output_path=tmp_path / "output.mp4", width=768, height=512, duration_seconds=1,
        seed=1, start_image_path=None, end_image_path=None,
    )
    tampered = copy.deepcopy(payload)
    mutation(tampered)
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_request_binding(
            tampered["schema_version"],
            tampered["operation"],
            tampered["settings"],
            tampered["plan"],
            tampered["generation"],
            tampered["request_binding"],
        )


def test_ltx_worker_binding_rejects_tampered_first_frame_endpoint(tmp_path: Path):
    first = tmp_path / "first.png"
    first.write_bytes(b"first")
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"replacement")
    payload = managed_module._payload(
        _settings(tmp_path), _plan(tmp_path), operation="first_frame", prompt="scene",
        output_path=tmp_path / "output.mp4", width=768, height=512, duration_seconds=1,
        seed=1, start_image_path=first, end_image_path=None,
    )
    payload["generation"]["start_image_path"] = str(replacement.resolve())
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_request_binding(
            payload["schema_version"],
            payload["operation"],
            payload["settings"],
            payload["plan"],
            payload["generation"],
            payload["request_binding"],
        )


def test_ltx_worker_binding_rejects_tampered_first_last_endpoint_order(tmp_path: Path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    payload = managed_module._payload(
        _settings(tmp_path), _plan(tmp_path), operation="first_last", prompt="scene",
        output_path=tmp_path / "output.mp4", width=768, height=512, duration_seconds=1,
        seed=1, start_image_path=first, end_image_path=last,
    )
    generation = payload["generation"]
    generation["start_image_path"], generation["end_image_path"] = (
        generation["end_image_path"], generation["start_image_path"]
    )
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_request_binding(
            payload["schema_version"], payload["operation"], payload["settings"],
            payload["plan"], payload["generation"], payload["request_binding"],
        )


def test_ltx_parent_rejects_unbound_failure_result(tmp_path: Path):
    result = tmp_path / "failure.json"
    result.write_text(
        '{"schema_version":1,"ok":false,"request_binding":"other",'
        '"error_type":"RuntimeError","error":"private path"}',
        encoding="utf-8",
    )
    assert managed_module._worker_error(result, 1, "expected") == "LTX worker exited with code 1"


def test_ltx_parent_accepts_only_bound_failure_without_exposing_worker_detail(tmp_path: Path):
    result = tmp_path / "failure.json"
    result.write_text(
        '{"schema_version":1,"ok":false,"request_binding":"expected",'
        '"error_type":"ValueError","error":"private path and prompt"}',
        encoding="utf-8",
    )
    assert managed_module._worker_error(result, 1, "expected") == "LTX worker failed (ValueError)"


def test_ltx_worker_checks_binding_before_settings_or_paths(tmp_path: Path, monkeypatch):
    payload = managed_module._payload(
        _settings(tmp_path), _plan(tmp_path), operation="t2v", prompt="scene",
        output_path=tmp_path / "output.mp4", width=768, height=512, duration_seconds=1,
        seed=1, start_image_path=None, end_image_path=None,
    )
    payload["settings"]["device"] = "cuda"
    monkeypatch.setattr(
        worker_module,
        "_settings",
        lambda _value: (_ for _ in ()).throw(AssertionError("settings were parsed before binding")),
    )
    with pytest.raises(ValueError, match="binding"):
        worker_module._run(payload, tmp_path / "progress.jsonl")


def test_ltx_managed_directly_terminates_worker_if_job_object_setup_fails(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    runtime = ManagedLTX23Runtime(_settings(tmp_path), plan, operation="t2v")
    paths = {key: tmp_path / key for key in ("request", "result", "progress", "gate")}
    events = []

    class Process:
        pid = 77

        def poll(self):
            return None if not events else 1

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append("wait")
            return 1

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        managed_module,
        "DisposableProcessTree",
        lambda _process: (_ for _ in ()).throw(OSError("assign failed")),
    )
    with pytest.raises(OSError, match="assign failed"):
        runtime.generate(
            plan=plan, prompt="scene", output_path=tmp_path / "output.mp4", width=768,
            height=512, duration_seconds=1, seed=1, progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert events == ["terminate", "wait"]
    assert runtime.status()["last_worker"]["outcome"] == "failed"
    assert runtime.status()["last_worker"]["tree_empty"] is True


def test_ltx_managed_serializes_concurrent_generation_ownership(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    runtime = ManagedLTX23Runtime(_settings(tmp_path), plan, operation="t2v")
    entered = threading.Event()
    release = threading.Event()
    paths = {key: tmp_path / key for key in ("request", "result", "progress", "gate")}

    class Process:
        pid = 88

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class Tree:
        def __init__(self, process):
            self.process = process

        def wait_for_empty(self, timeout=15):
            pass

        def close(self):
            pass

        def terminate(self):
            pass

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", Tree)

    def wait(*_args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(managed_module, "_wait", wait)
    def result(_path, output, _binding):
        output.write_bytes(b"ok")
        return {
            "metadata": {
                "pipeline_fingerprint": plan.pipeline_fingerprint,
                "seed": 1,
                "model_id": "model:ltx23:test",
            },
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    errors = []

    def first():
        try:
            runtime.generate(plan=plan, prompt="scene", output_path=tmp_path / "one.mp4", width=768, height=512, duration_seconds=1, seed=1, progress=lambda *_args: None, check_cancelled=lambda: None)
        except RuntimeError as exc:  # pragma: no cover - captured to fail assertion below.
            errors.append(exc)

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="already active"):
        runtime.generate(plan=plan, prompt="scene", output_path=tmp_path / "two.mp4", width=768, height=512, duration_seconds=1, seed=1, progress=lambda *_args: None, check_cancelled=lambda: None)
    release.set()
    thread.join(5)
    assert not errors and not thread.is_alive()


def test_ltx_cleanup_errors_are_observable_without_paths_or_request_content(tmp_path: Path, monkeypatch):
    path = tmp_path / "private-request.json"
    path.write_text("prompt should never be exposed", encoding="utf-8")
    output = tmp_path / "output.mp4"

    def fail_unlink(self, *args, **kwargs):
        if self == path:
            raise OSError("denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    assert managed_module._cleanup({"request": path}, output) == ["ipc"]
