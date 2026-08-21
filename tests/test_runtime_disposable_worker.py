from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from latentslate_engine.runtime.framework.worker import (
    DisposableWorkerExited,
    DisposableWorkerLimits,
    DisposableWorkerPaths,
    DisposableWorkerProgressTruncated,
    DisposableWorkerSupervisor,
    WorkerJsonFileError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "disposable_worker_fixture.py"


def _paths(tmp_path: Path) -> DisposableWorkerPaths:
    return DisposableWorkerPaths(
        request=tmp_path / "request.json",
        result=tmp_path / "result.json",
        progress=tmp_path / "progress.jsonl",
        start_gate=tmp_path / "start.gate",
    )


def _command(paths: DisposableWorkerPaths) -> tuple[str, ...]:
    return (
        sys.executable,
        str(FIXTURE),
        "--request",
        str(paths.request),
        "--result",
        str(paths.result),
        "--progress",
        str(paths.progress),
        "--start-gate",
        str(paths.start_gate),
    )


def _supervisor(
    tmp_path: Path, *, failure_outputs: tuple[Path, ...] = ()
) -> DisposableWorkerSupervisor:
    paths = _paths(tmp_path)
    return DisposableWorkerSupervisor(
        command=_command(paths),
        paths=paths,
        cleanup_paths={
            "request": paths.request,
            "result": paths.result,
            "progress": paths.progress,
            "gate": paths.start_gate,
        },
        failure_outputs=failure_outputs,
        limits=DisposableWorkerLimits(poll_seconds=0.02),
    )


def _request(
    mode: str,
    *,
    output: Path | None = None,
    unload_count: Path | None = None,
) -> dict[str, object]:
    return {
        "binding": "bound-request",
        "mode": mode,
        "value": "fixture-value",
        "output_path": None if output is None else str(output),
        "unload_count_path": None if unload_count is None else str(unload_count),
    }


def test_model_free_disposable_worker_cold_run_progress_and_cleanup(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    progress: list[dict[str, object]] = []

    result = supervisor.run(
        _request("success"),
        before_spawn=lambda: None,
        progress=progress.append,
        check_cancelled=lambda: None,
    )

    assert result == {
        "schema_version": 1,
        "ok": True,
        "request_binding": "bound-request",
        "value": "fixture-value",
    }
    assert progress == [
        {"message": "working", "progress": 0.25},
        {"message": "complete", "progress": 1.0},
    ]
    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "succeeded"
    assert supervisor.last_run.tree_empty is True
    assert supervisor.cleanup_errors == []
    assert not any(tmp_path.iterdir())


def test_model_free_disposable_worker_returns_only_safe_failure(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)

    with pytest.raises(DisposableWorkerExited) as raised:
        supervisor.run(
            _request("failure"),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=lambda: None,
        )

    failure = raised.value.result
    assert failure["ok"] is False
    assert failure["request_binding"] == "bound-request"
    assert failure["error_type"] == "RuntimeError"
    assert failure["failure_stage"] == "run"
    assert len(failure["error_fingerprint"]) == 64
    assert "private fixture detail" not in json.dumps(failure)
    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "failed"
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("request_bytes", "error_type"),
    [
        (b"not-json", "JSONDecodeError"),
        (b'{' + b'"padding":"' + b"x" * 2048 + b'"}', "ValueError"),
    ],
)
def test_model_free_child_fails_closed_for_malformed_or_oversize_request(
    tmp_path: Path, request_bytes: bytes, error_type: str
) -> None:
    paths = _paths(tmp_path)
    paths.request.write_bytes(request_bytes)
    paths.start_gate.touch()

    completed = subprocess.run(
        _command(paths),
        check=False,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode == 1
    failure = json.loads(paths.result.read_text(encoding="utf-8"))
    assert failure["ok"] is False
    assert failure["error_type"] == error_type
    assert failure["failure_stage"] == "read_request"
    assert "padding" not in json.dumps(failure)


def test_model_free_cancellation_forces_tree_empty_and_removes_owned_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial.bin"
    supervisor = _supervisor(tmp_path, failure_outputs=(output,))

    def cancel_after_child_started() -> None:
        if output.exists():
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        supervisor.run(
            _request("sleep", output=output),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=cancel_after_child_started,
        )

    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "canceled"
    assert supervisor.last_run.tree_empty is True
    assert not output.exists()
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed_result", json.JSONDecodeError),
        ("oversized_result", WorkerJsonFileError),
    ],
)
def test_model_free_parent_rejects_malformed_or_oversized_success_result(
    tmp_path: Path, mode: str, error_type: type[BaseException]
) -> None:
    supervisor = _supervisor(tmp_path)

    with pytest.raises(error_type):
        supervisor.run(
            _request(mode),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=lambda: None,
        )

    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "failed"
    assert supervisor.last_run.tree_empty is True
    assert supervisor.cleanup_errors == []
    assert not any(tmp_path.iterdir())


def test_model_free_parent_rejects_truncated_progress_after_successful_exit(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)

    with pytest.raises(DisposableWorkerProgressTruncated):
        supervisor.run(
            _request("truncated_progress"),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=lambda: None,
        )

    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "failed"
    assert supervisor.last_run.tree_empty is True
    assert supervisor.cleanup_errors == []
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(("mode", "exit_code"), [("crash_bind", 8), ("crash_run", 9)])
def test_model_free_parent_closes_tree_after_child_crash(
    tmp_path: Path, mode: str, exit_code: int
) -> None:
    supervisor = _supervisor(tmp_path)

    with pytest.raises(DisposableWorkerExited) as raised:
        supervisor.run(
            _request(mode),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=lambda: None,
        )

    assert raised.value.exit_code == exit_code
    assert raised.value.result is None
    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "failed"
    assert supervisor.last_run.tree_empty is True
    assert supervisor.cleanup_errors == []
    assert not any(tmp_path.iterdir())


def test_model_free_child_attempts_unload_exactly_once_when_unload_fails(
    tmp_path: Path,
) -> None:
    count_path = tmp_path / "unload-count.txt"
    supervisor = _supervisor(tmp_path)

    with pytest.raises(DisposableWorkerExited) as raised:
        supervisor.run(
            _request("unload_failure", unload_count=count_path),
            before_spawn=lambda: None,
            progress=lambda _record: None,
            check_cancelled=lambda: None,
        )

    assert raised.value.result["error_type"] == "RuntimeError"
    assert raised.value.result["failure_stage"] == "unload"
    assert "private unload detail" not in json.dumps(raised.value.result)
    assert count_path.read_text(encoding="utf-8") == "1"
    assert supervisor.last_run is not None
    assert supervisor.last_run.outcome == "failed"
    assert supervisor.last_run.tree_empty is True
    assert supervisor.cleanup_errors == []
    assert set(tmp_path.iterdir()) == {count_path}


def test_disposable_worker_command_is_fixed_before_request_publication() -> None:
    source = (Path(sys.modules[DisposableWorkerSupervisor.__module__].__file__)).read_text(
        encoding="utf-8"
    )

    assert "importlib" not in source
    assert "runpy" not in source
    assert "payload[" not in source
    assert "request[" not in source
