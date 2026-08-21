from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import time
from pathlib import Path

import pytest

from latentslate_engine.runtime.framework.worker import (
    PersistentWatchdogPolicy,
    PersistentWorkerExited,
    PersistentWorkerPaths,
    PersistentWorkerStreamError,
    PersistentWorkerSupervisor,
    PersistentWorkerTimeout,
    WorkerJsonFileError,
    hmac_sha256,
    read_bounded_json,
    result_hmac_sha256,
)

FIXTURE = Path(__file__).parent / "fixtures" / "persistent_worker_fixture.py"
SECRET = bytes(range(32))


def _paths(root: Path) -> PersistentWorkerPaths:
    root.mkdir()
    return PersistentWorkerPaths(
        request=root / "request.json",
        result=root / "result.json",
        progress=root / "progress.jsonl",
        heartbeat=root / "heartbeat.jsonl",
        start_gate=root / "start.gate",
        command=root / "command.json",
        cancel=root / "cancel-requested",
    )


def _supervisor(root: Path) -> PersistentWorkerSupervisor:
    paths = _paths(root)
    environment = os.environ.copy()
    environment["LATENTSLATE_PERSISTENT_FIXTURE_SECRET"] = SECRET.hex()
    return PersistentWorkerSupervisor(
        command=(
            sys.executable,
            str(FIXTURE),
            "--request",
            str(paths.request),
            "--result",
            str(paths.result),
            "--progress",
            str(paths.progress),
            "--heartbeat",
            str(paths.heartbeat),
            "--start-gate",
            str(paths.start_gate),
            "--command",
            str(paths.command),
            "--cancel",
            str(paths.cancel),
        ),
        paths=paths,
        environment=environment,
    )


def _request(
    sequence: int,
    mode: str = "success",
    *,
    session_id: str = "fixture-session",
    marker: Path | None = None,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "session_id": session_id,
        "sequence": sequence,
        "mode": mode,
        "value": f"value-{sequence}",
        "marker_path": None if marker is None else str(marker),
    }
    return {**unsigned, "binding": hmac_sha256(unsigned, SECRET)}


def _policy(**overrides: float) -> PersistentWatchdogPolicy:
    values = {
        "hard_timeout_seconds": 5.0,
        "stage_timeout_seconds": 2.0,
        "heartbeat_timeout_seconds": 1.0,
        "cancel_grace_seconds": 0.05,
        "poll_seconds": 0.01,
        "maximum_stream_bytes": 2048,
        "maximum_stream_records": 128,
    }
    values.update(overrides)
    return PersistentWatchdogPolicy(**values)


def _wait(
    supervisor: PersistentWorkerSupervisor,
    progress: list[dict[str, object]] | None = None,
    *,
    policy: PersistentWatchdogPolicy | None = None,
    check_cancelled=lambda: None,
) -> dict[str, object]:
    supervisor.wait(
        progress=(lambda value: None) if progress is None else progress.append,
        check_cancelled=check_cancelled,
        policy=_policy() if policy is None else policy,
    )
    return read_bounded_json(supervisor.paths.result, maximum_bytes=2048)


def _verify_result(result: dict[str, object], request: dict[str, object]) -> None:
    assert result["request_binding"] == request["binding"]
    signature = result["result_binding"]
    unsigned = {key: value for key, value in result.items() if key != "result_binding"}
    assert isinstance(signature, str)
    assert hmac.compare_digest(signature, result_hmac_sha256(unsigned, SECRET))


def _destroy(supervisor: PersistentWorkerSupervisor) -> None:
    try:
        supervisor.terminate()
    finally:
        supervisor.close()


def test_persistent_fixture_cold_then_warm_reuses_exact_pid_and_load(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    first_request = _request(1)
    supervisor.start(first_request)
    first_progress: list[dict[str, object]] = []
    first = _wait(supervisor, first_progress)
    _verify_result(first, first_request)
    assert first["warm"] is False and first["loads"] == 1
    assert [record["progress"] for record in first_progress] == [0.25, 1.0]

    supervisor.cleanup_job()
    second_request = _request(2)
    supervisor.send(second_request)
    second_progress: list[dict[str, object]] = []
    second = _wait(supervisor, second_progress)
    _verify_result(second, second_request)
    assert second["worker_pid"] == first["worker_pid"]
    assert second["warm"] is True and second["loads"] == 1
    assert [record["progress"] for record in second_progress] == [0.25, 1.0]

    _destroy(supervisor)
    assert supervisor.cleanup_session() == []
    assert not (tmp_path / "ipc").exists()


def test_persistent_fixture_start_gate_failure_cleans_spawned_tree_and_ipc(tmp_path: Path) -> None:
    root = tmp_path / "ipc"
    supervisor = _supervisor(root)
    retained = root / "not-owned.txt"
    retained.write_text("retained", encoding="utf-8")
    supervisor.paths.start_gate.touch()
    with pytest.raises(FileExistsError):
        supervisor.start(_request(1))
    assert supervisor.session is None
    assert supervisor.failed_start is not None
    assert supervisor.failed_start.pid > 0
    assert supervisor.failed_start.exit_code is not None
    assert supervisor.failed_start.terminated is True
    assert supervisor.failed_start.tree_empty is True
    assert supervisor.failed_start.cleanup_errors == ("root:OSError",)
    assert set(root.iterdir()) == {retained}


@pytest.mark.parametrize("violation", ["replay", "skip", "tamper"])
def test_persistent_fixture_rejects_sequence_replay_duplicate_and_hmac_tamper(
    tmp_path: Path, violation: str
) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1))
    _wait(supervisor)
    supervisor.cleanup_job()
    command = _request(1 if violation == "replay" else 3)
    if violation == "tamper":
        command = _request(2)
        command["value"] = "tampered"
    supervisor.send(command)
    result = _wait(supervisor)
    _verify_result(result, command)
    assert result["ok"] is False
    assert result["failure_stage"] == "bind"
    assert "tampered" not in json.dumps(result)
    assert supervisor.session is not None
    supervisor.session.process.wait(timeout=5)
    assert supervisor.session.process.returncode == 1
    _destroy(supervisor)
    supervisor.cleanup_session()


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [("malformed_result", json.JSONDecodeError), ("oversized_result", WorkerJsonFileError)],
)
def test_persistent_fixture_parent_rejects_malformed_or_oversized_result(
    tmp_path: Path, mode: str, error_type: type[BaseException]
) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, mode))
    supervisor.wait(
        progress=lambda _value: None,
        check_cancelled=lambda: None,
        policy=_policy(),
    )
    with pytest.raises(error_type):
        read_bounded_json(supervisor.paths.result, maximum_bytes=2048)
    _destroy(supervisor)
    supervisor.cleanup_session()


def test_persistent_fixture_full_result_hmac_corruption_is_rejected(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    request = _request(1, "corrupt_result")
    supervisor.start(request)
    result = _wait(supervisor)
    with pytest.raises(AssertionError):
        _verify_result(result, request)
    _destroy(supervisor)
    supervisor.cleanup_session()


def test_persistent_fixture_rejects_truncated_progress_on_successful_exit(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, "truncated_progress"))
    with pytest.raises(PersistentWorkerStreamError) as raised:
        _wait(supervisor)
    assert raised.value.stream == "progress" and raised.value.reason == "truncated"
    assert supervisor.session is not None
    supervisor.session.process.wait(timeout=5)
    assert supervisor.session.process.returncode == 0
    _destroy(supervisor)
    supervisor.cleanup_session()


@pytest.mark.parametrize(
    ("mode", "stream"),
    [
        ("result_truncated_progress", "progress"),
        ("result_truncated_heartbeat", "heartbeat"),
    ],
)
def test_persistent_fixture_rejects_result_with_any_truncated_stream(
    tmp_path: Path, mode: str, stream: str
) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, mode))
    with pytest.raises(PersistentWorkerStreamError) as raised:
        _wait(supervisor)
    assert supervisor.paths.result.is_file()
    assert raised.value.stream == stream and raised.value.reason == "truncated"
    _destroy(supervisor)
    supervisor.cleanup_session()


@pytest.mark.parametrize(("mode", "exit_code"), [("crash_load", 8), ("crash_execute", 9)])
def test_persistent_fixture_reports_crash_before_or_during_execution(
    tmp_path: Path, mode: str, exit_code: int
) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, mode))
    with pytest.raises(PersistentWorkerExited):
        _wait(supervisor)
    assert supervisor.session is not None
    assert supervisor.session.process.wait(timeout=5) == exit_code
    _destroy(supervisor)
    supervisor.cleanup_session()


def test_persistent_fixture_safe_failure_unloads_once_and_fresh_session_recovers(
    tmp_path: Path,
) -> None:
    failed = _supervisor(tmp_path / "failed")
    request = _request(1, "unload_failure")
    failed.start(request)
    result = _wait(failed)
    _verify_result(result, request)
    assert result["ok"] is False and result["failure_stage"] == "unload"
    assert "private" not in json.dumps(result)
    assert (tmp_path / "failed" / "unload-count.txt").read_text(encoding="utf-8") == "1"
    _destroy(failed)
    failed.cleanup_session()

    recovered = _supervisor(tmp_path / "recovered")
    recovery_request = _request(1)
    recovered.start(recovery_request)
    recovery = _wait(recovered)
    _verify_result(recovery, recovery_request)
    assert recovery["ok"] is True
    _destroy(recovered)
    recovered.cleanup_session()


@pytest.mark.parametrize(
    ("mode", "clock", "overrides"),
    [
        ("heartbeat_stall", "heartbeat", {"heartbeat_timeout_seconds": 0.1}),
        ("stage_stall", "stage", {"stage_timeout_seconds": 0.1}),
        ("hard_busy", "hard", {"hard_timeout_seconds": 0.15}),
    ],
)
def test_persistent_fixture_watchdog_clocks_are_independent_and_hard_is_nonrenewable(
    tmp_path: Path, mode: str, clock: str, overrides: dict[str, float]
) -> None:
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, mode))
    with pytest.raises(PersistentWorkerTimeout) as raised:
        _wait(supervisor, policy=_policy(**overrides))
    assert raised.value.clock == clock
    assert supervisor.paths.cancel.is_file()
    _destroy(supervisor)
    supervisor.cleanup_session()


def test_persistent_fixture_cancel_marks_then_forces_complete_process_tree_exit(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant.pid"
    supervisor = _supervisor(tmp_path / "ipc")
    supervisor.start(_request(1, "ignore_cancel", marker=marker))

    def cancel_after_tree_started() -> None:
        if marker.exists():
            raise asyncio.CancelledError

    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        _wait(supervisor, check_cancelled=cancel_after_tree_started)
    assert supervisor.paths.cancel.is_file()
    assert time.monotonic() - started < 5
    _destroy(supervisor)
    assert supervisor.session is None or supervisor.session.process.poll() is not None
    supervisor.cleanup_session()


def test_persistent_worker_command_and_child_handler_are_static() -> None:
    parent_source = Path(sys.modules[PersistentWorkerSupervisor.__module__].__file__).read_text(
        encoding="utf-8"
    )
    child_source = FIXTURE.read_text(encoding="utf-8")
    assert "importlib" not in parent_source + child_source
    assert "runpy" not in parent_source + child_source
    assert "payload[\"handler\"]" not in parent_source + child_source
