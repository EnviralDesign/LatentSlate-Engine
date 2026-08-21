"""Model-neutral supervision for one-shot isolated workers."""

from __future__ import annotations

import asyncio
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...windows_process import DisposableProcessTree
from .files import (
    WorkerJsonFileError,
    atomic_write_json,
    cleanup_atomic_write_siblings,
    read_bounded_json,
)
from .progress import JsonlCursor, drain_bounded_jsonl


class DisposableWorkerExited(RuntimeError):
    """A disposable child exited unsuccessfully with an optional result object."""

    def __init__(self, exit_code: int, result: Any | None) -> None:
        super().__init__(f"disposable worker exited with code {exit_code}")
        self.exit_code = exit_code
        self.result = result


class DisposableWorkerProgressTruncated(RuntimeError):
    """A successful child left an incomplete trailing progress record."""


@dataclass(frozen=True, slots=True)
class DisposableWorkerPaths:
    """Transport endpoints owned by one disposable worker invocation."""

    request: Path
    result: Path
    progress: Path
    start_gate: Path


@dataclass(frozen=True, slots=True)
class DisposableWorkerLimits:
    """Explicit file, record, polling, and shutdown bounds."""

    maximum_json_bytes: int = 1024 * 1024
    maximum_progress_bytes: int = 1024 * 1024
    maximum_progress_records: int = 4096
    poll_seconds: float = 0.1
    process_wait_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_json_bytes",
            "maximum_progress_bytes",
            "maximum_progress_records",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"disposable worker {name} must be a positive integer")
        for name in ("poll_seconds", "process_wait_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"disposable worker {name} must be numeric")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"disposable worker {name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class DisposableWorkerRunState:
    """Closed process facts retained by the model adapter for status reporting."""

    pid: int
    exit_code: int | None
    terminated: bool
    outcome: str
    tree_empty: bool


class DisposableWorkerSupervisor:
    """Run exactly one fixed child command in a kill-on-close process tree."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        paths: DisposableWorkerPaths,
        cleanup_paths: Mapping[str, Path],
        failure_outputs: Sequence[Path] = (),
        environment: Mapping[str, str] | None = None,
        limits: DisposableWorkerLimits | None = None,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("disposable worker command must contain non-empty strings")
        self._command = tuple(command)
        self.paths = paths
        self._cleanup_paths = dict(cleanup_paths)
        self._failure_outputs = tuple(Path(path) for path in failure_outputs)
        self._environment = None if environment is None else dict(environment)
        self._limits = DisposableWorkerLimits() if limits is None else limits
        self._active_tree: DisposableProcessTree | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.last_run: DisposableWorkerRunState | None = None
        self.cleanup_errors: list[str] = []

    @property
    def active(self) -> bool:
        return self._active_tree is not None

    def run(
        self,
        request: Mapping[str, Any],
        *,
        before_spawn: Callable[[], None],
        progress: Callable[[dict[str, Any]], None],
        check_cancelled: Callable[[], None],
    ) -> Any:
        """Publish a request, run the child, and return its bounded JSON result."""

        if self._process is not None or self._active_tree is not None:
            raise RuntimeError("disposable worker supervisor is already active")
        process: subprocess.Popen[bytes] | None = None
        tree: DisposableProcessTree | None = None
        try:
            atomic_write_json(self.paths.request, request)
            before_spawn()
            process = subprocess.Popen(
                self._command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=self._environment,
            )
            self._process = process
            try:
                tree = DisposableProcessTree(process)
            except BaseException:
                _terminate_direct(process, self._limits.process_wait_seconds)
                self.last_run = _run_state(process, "failed", tree_empty=process.poll() is not None)
                raise
            self._active_tree = tree
            self.paths.start_gate.touch(exist_ok=False)
            self._wait(process, progress, check_cancelled)
            exit_code = process.wait(timeout=self._limits.process_wait_seconds)
            if exit_code != 0:
                raise DisposableWorkerExited(exit_code, self._read_failure_result())
            result = read_bounded_json(
                self.paths.result, maximum_bytes=self._limits.maximum_json_bytes
            )
            tree.wait_for_empty()
            tree.close()
            tree = None
            self._active_tree = None
            self.last_run = _run_state(process, "succeeded", tree_empty=True)
            return result
        except BaseException as primary:
            if tree is not None or process is not None:
                tree_empty = _terminate_tree(tree, process, self._limits.process_wait_seconds)
                if process is not None:
                    self.last_run = _run_state(
                        process,
                        "canceled" if is_worker_cancellation(primary) else "failed",
                        tree_empty=tree_empty,
                    )
            if process is not None:
                for output in self._failure_outputs:
                    output.unlink(missing_ok=True)
            raise
        finally:
            self._active_tree = None
            self._process = None
            if tree is not None:
                try:
                    tree.close()
                except BaseException:  # noqa: BLE001, S110 - preserve primary result
                    pass
            self.cleanup()

    def terminate(self) -> None:
        """Force the active worker tree empty, if a run is in progress."""

        tree = self._active_tree
        if tree is None:
            return
        try:
            tree.terminate()
            tree.wait_for_empty()
        finally:
            tree.close()
            self._active_tree = None

    def cleanup(self) -> None:
        """Remove all invocation-owned paths and retain only safe error labels."""

        self.cleanup_errors = list(
            dict.fromkeys((*self.cleanup_errors, *_cleanup(self._cleanup_paths)))
        )

    def _wait(
        self,
        process: subprocess.Popen[bytes],
        progress: Callable[[dict[str, Any]], None],
        check_cancelled: Callable[[], None],
    ) -> None:
        cursor = JsonlCursor()
        while process.poll() is None:
            check_cancelled()
            cursor = self._drain_progress(cursor, progress)
            time.sleep(self._limits.poll_seconds)
        cursor = self._drain_progress(cursor, progress)
        if cursor.pending and process.poll() == 0:
            raise DisposableWorkerProgressTruncated

    def _drain_progress(
        self,
        cursor: JsonlCursor,
        progress: Callable[[dict[str, Any]], None],
    ) -> JsonlCursor:
        cursor, records = drain_bounded_jsonl(
            self.paths.progress,
            cursor,
            maximum_bytes=self._limits.maximum_progress_bytes,
            maximum_records=self._limits.maximum_progress_records,
        )
        for record in records:
            progress(record)
        return cursor

    def _read_failure_result(self) -> Any | None:
        try:
            return read_bounded_json(
                self.paths.result, maximum_bytes=self._limits.maximum_json_bytes
            )
        except (OSError, WorkerJsonFileError, UnicodeDecodeError, ValueError):
            return None


def is_worker_cancellation(exc: BaseException) -> bool:
    """Recognize the framework's supported cancellation exception families."""

    return isinstance(exc, asyncio.CancelledError) or any(
        cls.__name__ == "ToolCancelled" for cls in type(exc).__mro__
    )


def _terminate_tree(
    tree: DisposableProcessTree | None,
    process: subprocess.Popen[bytes] | None,
    wait_seconds: float,
) -> bool:
    if tree is not None:
        try:
            tree.terminate()
            tree.wait_for_empty()
            return True
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError):
            return False
    if process is not None:
        _terminate_direct(process, wait_seconds)
        return process.poll() is not None
    return False


def _terminate_direct(process: subprocess.Popen[bytes], wait_seconds: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=wait_seconds)


def _run_state(
    process: subprocess.Popen[bytes], outcome: str, *, tree_empty: bool
) -> DisposableWorkerRunState:
    exit_code = process.poll()
    return DisposableWorkerRunState(
        pid=process.pid,
        exit_code=exit_code,
        terminated=exit_code is not None,
        outcome=outcome,
        tree_empty=tree_empty,
    )


def _cleanup(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for label, path in paths.items():
        try:
            path.unlink(missing_ok=True)
            cleanup_atomic_write_siblings(path)
        except OSError:
            errors.append(f"{label}_cleanup_failed")
    return errors
