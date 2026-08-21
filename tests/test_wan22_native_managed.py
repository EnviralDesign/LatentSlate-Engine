from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.runtime.framework.worker import PersistentWorkerPaths
from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest
from latentslate_engine.runtime.wan22_native_managed import ManagedNativeWanI2VRuntime


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.exited = False

    def poll(self) -> int | None:
        return 1 if self.exited else None


class _Supervisor:
    def __init__(self, root: Path, pid: int) -> None:
        self.paths = PersistentWorkerPaths(
            request=root / "request.json",
            result=root / "result.json",
            progress=root / "progress.jsonl",
            heartbeat=root / "heartbeat.jsonl",
            start_gate=root / "start-gate",
            command=root / "command.json",
            cancel=root / "cancel-requested",
        )
        self.process = _Process(pid)
        self.session = None
        self.failed_start = None
        self.events: list[str] = []
        self.fail_wait = False

    def start(self, _payload):
        self.events.append("start")
        self.session = SimpleNamespace(process=self.process)

    def send(self, _payload):
        self.events.append("send")

    def wait(self, *, progress, check_cancelled, policy):
        del policy
        self.events.append("wait")
        check_cancelled()
        if self.fail_wait:
            raise RuntimeError("child failed")
        progress({"completed": 1, "total": 1, "stage": "high"})

    def terminate(self):
        self.events.append("terminate")
        self.process.exited = True

    def close(self):
        self.events.append("close")
        self.session = None

    def cleanup_job(self):
        self.events.append("cleanup_job")
        return []

    def cleanup_session(self):
        self.events.append("cleanup_session")
        return []


def test_managed_wan_reuses_warm_child_then_poisons_and_recovers_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentslate_engine.runtime.wan22_native_managed as managed_module

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    recipe = SimpleNamespace(
        operation="wan22_i2v_base",
        fingerprint="recipe:test",
        to_json_dict=lambda: {"recipe": "test"},
        public_component_manifest=dict,
    )
    request = WanI2VRequest(
        image=object(),
        prompt="move",
        steps=20,
        high_guidance=3.5,
        low_guidance=3.5,
    )
    supervisors: list[_Supervisor] = []

    def make_supervisor(paths, _secret):
        supervisor = _Supervisor(paths.request.parent, len(supervisors) + 10)
        supervisors.append(supervisor)
        return supervisor

    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _value: True)
    monkeypatch.setattr(managed_module, "_secure_ipc_directory", lambda _root: None)
    monkeypatch.setattr(managed_module, "_supervisor", make_supervisor)
    monkeypatch.setattr(
        managed_module,
        "_read_persistent_result",
        lambda *_args, **_kwargs: {
            "output_size_bytes": 3,
            "stream_metadata": {},
            "provenance": {},
        },
    )
    monkeypatch.setattr(
        managed_module, "_validate_worker_provenance_against_request", lambda *_a, **_k: None
    )
    monkeypatch.setattr(managed_module, "_validate_stream_against_request", lambda *_a, **_k: None)

    runtime = ManagedNativeWanI2VRuntime(recipe)
    progress: list[str] = []

    def generate(name: str):
        return runtime.generate(
            request,
            source_image_path=source,
            output_path=tmp_path / name,
            device="cuda",
            fps=16,
            progress=lambda _completed, _total, stage: progress.append(stage),
            cancelled=lambda: False,
        )

    first = generate("first.mp4")
    second = generate("second.mp4")
    assert first.pipeline_warm is False
    assert second.pipeline_warm is True
    assert supervisors[0].events[:5] == ["start", "wait", "cleanup_job", "send", "wait"]

    supervisors[0].fail_wait = True
    with pytest.raises(RuntimeError, match="child failed"):
        generate("failed.mp4")
    assert runtime.status()["loaded"] is False
    assert supervisors[0].events[-3:] == ["terminate", "close", "cleanup_session"]

    recovered = generate("recovered.mp4")
    assert recovered.pipeline_warm is False
    assert len(supervisors) == 2
    runtime.unload()
    assert supervisors[1].events[-3:] == ["terminate", "close", "cleanup_session"]


def test_dead_idle_wan_child_is_discarded_before_status_reports_residency(
    tmp_path: Path,
) -> None:
    recipe = SimpleNamespace(fingerprint="recipe:dead", public_component_manifest=dict)
    runtime = ManagedNativeWanI2VRuntime(recipe)
    supervisor = _Supervisor(tmp_path, 17)
    supervisor.start({})
    supervisor.process.exited = True
    runtime._session = SimpleNamespace(
        supervisor=supervisor,
        device="cuda",
        session_binding="binding",
        secret=b"x" * 32,
        successful_jobs=1,
    )
    runtime._active_supervisor = supervisor

    status = runtime.status()

    assert status["loaded"] is False
    assert status["cache"] == {"pipeline_warm": False}
    assert supervisor.events[-3:] == ["terminate", "close", "cleanup_session"]


def test_poison_keeps_tree_cleanup_failure_authoritative_and_retains_original(
    tmp_path: Path,
) -> None:
    recipe = SimpleNamespace(fingerprint="recipe:cleanup", public_component_manifest=dict)
    runtime = ManagedNativeWanI2VRuntime(recipe)

    class _FailingSupervisor(_Supervisor):
        def terminate(self):
            raise RuntimeError("tree accounting failed")

        def close(self):
            raise OSError("close failed")

    supervisor = _FailingSupervisor(tmp_path, 23)
    supervisor.start({})
    session = SimpleNamespace(
        supervisor=supervisor,
        device="cuda",
        session_binding="binding",
        secret=b"x" * 32,
        successful_jobs=0,
    )
    runtime._session = session
    runtime._active_supervisor = supervisor
    original = RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="tree accounting failed") as caught:
        runtime._poison_session(session, original)

    notes = "\n".join(caught.value.__notes__)
    assert "original native Wan failure" in notes
    assert "close failed" in notes
    assert runtime._session is None


def test_late_cancellation_removes_owned_output_and_poisons_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentslate_engine.runtime.wan22_native_managed as managed_module

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    target = tmp_path / "target.mp4"
    recipe = SimpleNamespace(
        operation="wan22_i2v_base",
        fingerprint="recipe:cancel",
        to_json_dict=lambda: {"recipe": "test"},
        public_component_manifest=dict,
    )
    supervisor = _Supervisor(tmp_path / "session", 31)
    (tmp_path / "session").mkdir()
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _value: True)
    monkeypatch.setattr(managed_module, "_secure_ipc_directory", lambda _root: None)
    monkeypatch.setattr(managed_module, "_supervisor", lambda *_args: supervisor)
    monkeypatch.setattr(
        managed_module,
        "_wait_for_result",
        lambda *_args, **_kwargs: target.write_bytes(b"partial"),
    )
    checks = iter((False, True))
    runtime = ManagedNativeWanI2VRuntime(recipe)

    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            WanI2VRequest(
                image=object(),
                prompt="move",
                steps=20,
                high_guidance=3.5,
                low_guidance=3.5,
            ),
            source_image_path=source,
            output_path=target,
            device="cuda",
            fps=16,
            cancelled=lambda: next(checks),
        )

    assert not target.exists()
    assert runtime.status()["loaded"] is False
    assert supervisor.events[-3:] == ["terminate", "close", "cleanup_session"]
