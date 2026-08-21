"""Model-neutral supervision for authenticated persistent workers."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...windows_process import DisposableProcessTree
from .files import atomic_write_json
from .progress import JsonlCursor, WorkerJsonlFileError, drain_bounded_jsonl


class PersistentWorkerExited(RuntimeError):
    """A persistent child exited before publishing a bounded result."""


class PersistentWorkerTimeout(RuntimeError):
    """One non-overlapping persistent-worker clock crossed its boundary."""

    def __init__(self, clock: str) -> None:
        if clock not in {"hard", "stage", "heartbeat"}:
            raise ValueError("persistent worker timeout clock is invalid")
        super().__init__(f"persistent worker {clock} clock expired")
        self.clock = clock


class PersistentWorkerStreamError(RuntimeError):
    """A progress or heartbeat stream violated its bounded transport."""

    def __init__(self, stream: str, reason: str = "invalid") -> None:
        if stream not in {"progress", "heartbeat"}:
            raise ValueError("persistent worker stream kind is invalid")
        super().__init__(f"persistent worker {stream} stream is invalid")
        self.stream = stream
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PersistentWorkerPaths:
    request: Path
    result: Path
    progress: Path
    heartbeat: Path
    start_gate: Path
    command: Path
    cancel: Path


@dataclass(frozen=True, slots=True)
class PersistentWatchdogPolicy:
    hard_timeout_seconds: float
    stage_timeout_seconds: float
    heartbeat_timeout_seconds: float
    cancel_grace_seconds: float
    poll_seconds: float = 0.1
    maximum_stream_bytes: int = 1024 * 1024
    maximum_stream_records: int = 4096


@dataclass(slots=True)
class PersistentWorkerSession:
    process: subprocess.Popen[bytes]
    tree: DisposableProcessTree
    paths: PersistentWorkerPaths


@dataclass(frozen=True, slots=True)
class PersistentWorkerFailedStart:
    """Bounded process facts retained when start-gate publication fails."""

    pid: int
    exit_code: int | None
    terminated: bool
    tree_empty: bool
    cleanup_errors: tuple[str, ...]


class PersistentWorkerSupervisor:
    """Own one fixed persistent child and poison it on every command error."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        paths: PersistentWorkerPaths,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("persistent worker command must contain non-empty strings")
        self._command = tuple(command)
        self.paths = paths
        self._environment = None if environment is None else dict(environment)
        self.session: PersistentWorkerSession | None = None
        self.failed_start: PersistentWorkerFailedStart | None = None

    def start(self, request: Mapping[str, Any]) -> PersistentWorkerSession:
        if self.session is not None:
            raise RuntimeError("persistent worker supervisor is already started")
        self.failed_start = None
        atomic_write_json(self.paths.request, request)
        try:
            process = subprocess.Popen(
                self._command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=self._environment,
            )
        except BaseException:
            self.cleanup_session()
            raise
        try:
            tree = DisposableProcessTree(process)
        except BaseException:
            process.kill()
            process.wait(timeout=15)
            self.cleanup_session()
            raise
        session = PersistentWorkerSession(process, tree, self.paths)
        self.session = session
        try:
            self.paths.start_gate.touch(exist_ok=False)
        except BaseException:
            errors: list[str] = []
            tree_empty = False
            try:
                tree.terminate()
            except BaseException as exc:  # noqa: BLE001 - preserve gate error
                errors.append(f"terminate:{type(exc).__name__}")
            try:
                process.wait(timeout=15)
            except BaseException as exc:  # noqa: BLE001 - preserve gate error
                errors.append(f"process_wait:{type(exc).__name__}")
            try:
                tree.wait_for_empty()
                tree_empty = True
            except BaseException as exc:  # noqa: BLE001 - preserve gate error
                errors.append(f"tree_wait:{type(exc).__name__}")
            try:
                tree.close()
            except BaseException as exc:  # noqa: BLE001 - preserve gate error
                errors.append(f"tree_close:{type(exc).__name__}")
            self.session = None
            errors.extend(self.cleanup_session())
            exit_code = process.poll()
            self.failed_start = PersistentWorkerFailedStart(
                pid=process.pid,
                exit_code=exit_code,
                terminated=exit_code is not None,
                tree_empty=tree_empty,
                cleanup_errors=tuple(errors[-16:]),
            )
            raise
        return session

    def send(self, command: Mapping[str, Any]) -> None:
        self._require_session()
        atomic_write_json(self.paths.command, command)

    def wait(
        self,
        *,
        progress: Callable[[dict[str, Any]], None],
        check_cancelled: Callable[[], None],
        policy: PersistentWatchdogPolicy,
    ) -> None:
        session = self._require_session()
        progress_cursor = JsonlCursor()
        heartbeat_cursor = JsonlCursor()
        started = time.monotonic()
        last_progress = started
        last_heartbeat = started

        def drain() -> tuple[bool, bool, bool]:
            nonlocal heartbeat_cursor, last_heartbeat, progress_cursor, last_progress
            heartbeat_cursor, heartbeat_seen = self._drain_heartbeat(
                heartbeat_cursor, policy
            )
            if heartbeat_seen:
                last_heartbeat = time.monotonic()
            progress_cursor, progress_seen = self._drain_progress(
                progress_cursor, progress, policy
            )
            if progress_seen:
                last_progress = time.monotonic()
            result = self.paths.result.is_file()
            if result:
                if heartbeat_cursor.pending:
                    raise PersistentWorkerStreamError("heartbeat", "truncated")
                if progress_cursor.pending:
                    raise PersistentWorkerStreamError("progress", "truncated")
            return result, heartbeat_seen, progress_seen

        def request_parent_cancel() -> None:
            try:
                check_cancelled()
            except BaseException:
                self.request_cancel_and_grace(policy.cancel_grace_seconds)
                raise

        def raise_exited() -> None:
            if heartbeat_cursor.pending:
                raise PersistentWorkerStreamError("heartbeat", "truncated")
            if progress_cursor.pending:
                raise PersistentWorkerStreamError("progress", "truncated")
            raise PersistentWorkerExited

        def timeout_recheck() -> tuple[bool, bool, bool]:
            result, heartbeat_seen, progress_seen = drain()
            if result:
                return True, heartbeat_seen, progress_seen
            request_parent_cancel()
            if session.process.poll() is not None:
                raise_exited()
            time.sleep(0)
            result, next_heartbeat, next_progress = drain()
            heartbeat_seen = heartbeat_seen or next_heartbeat
            progress_seen = progress_seen or next_progress
            if result:
                return True, heartbeat_seen, progress_seen
            request_parent_cancel()
            if session.process.poll() is not None:
                raise_exited()
            return False, heartbeat_seen, progress_seen

        while True:
            if drain()[0]:
                return
            request_parent_cancel()
            now = time.monotonic()
            if now - started > policy.hard_timeout_seconds:
                if timeout_recheck()[0]:
                    return
                self.request_cancel_and_grace(policy.cancel_grace_seconds)
                raise PersistentWorkerTimeout("hard")
            if now - last_heartbeat > policy.heartbeat_timeout_seconds:
                result, heartbeat_seen, _progress_seen = timeout_recheck()
                if result:
                    return
                if heartbeat_seen:
                    continue
                self.request_cancel_and_grace(policy.cancel_grace_seconds)
                raise PersistentWorkerTimeout("heartbeat")
            if now - last_progress > policy.stage_timeout_seconds:
                result, _heartbeat_seen, progress_seen = timeout_recheck()
                if result:
                    return
                if progress_seen:
                    continue
                self.request_cancel_and_grace(policy.cancel_grace_seconds)
                raise PersistentWorkerTimeout("stage")
            if session.process.poll() is not None:
                raise_exited()
            time.sleep(policy.poll_seconds)

    def request_cancel_and_grace(self, seconds: float) -> None:
        session = self._require_session()
        self.paths.cancel.touch(exist_ok=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.paths.result.is_file() or session.process.poll() is not None:
                return
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def terminate(self) -> None:
        session = self._require_session()
        session.tree.terminate()
        session.process.wait(timeout=15)
        session.tree.wait_for_empty()

    def close(self) -> None:
        session = self.session
        self.session = None
        if session is not None:
            session.tree.close()

    def cleanup_job(self) -> list[str]:
        return _cleanup_paths(
            {
                "command": self.paths.command,
                "result": self.paths.result,
                "progress": self.paths.progress,
                "heartbeat": self.paths.heartbeat,
            }
        )

    def cleanup_session(self) -> list[str]:
        errors = self.cleanup_job()
        errors.extend(
            _cleanup_paths(
                {
                    "request": self.paths.request,
                    "gate": self.paths.start_gate,
                    "cancel": self.paths.cancel,
                }
            )
        )
        try:
            self.paths.request.parent.rmdir()
        except OSError as exc:
            errors.append(f"root:{type(exc).__name__}")
        return errors[-16:]

    def _drain_progress(
        self,
        cursor: JsonlCursor,
        progress: Callable[[dict[str, Any]], None],
        policy: PersistentWatchdogPolicy,
    ) -> tuple[JsonlCursor, bool]:
        try:
            cursor, values = drain_bounded_jsonl(
                self.paths.progress,
                cursor,
                maximum_bytes=policy.maximum_stream_bytes,
                maximum_records=policy.maximum_stream_records,
            )
        except WorkerJsonlFileError as exc:
            raise PersistentWorkerStreamError("progress", exc.reason) from exc
        for value in values:
            progress(value)
        return cursor, bool(values)

    def _drain_heartbeat(
        self, cursor: JsonlCursor, policy: PersistentWatchdogPolicy
    ) -> tuple[JsonlCursor, bool]:
        try:
            cursor, values = drain_bounded_jsonl(
                self.paths.heartbeat,
                cursor,
                maximum_bytes=policy.maximum_stream_bytes,
                maximum_records=policy.maximum_stream_records,
            )
        except WorkerJsonlFileError as exc:
            raise PersistentWorkerStreamError("heartbeat", exc.reason) from exc
        if any(value != {"heartbeat": 1} for value in values):
            raise PersistentWorkerStreamError("heartbeat", "record_value")
        return cursor, bool(values)

    def _require_session(self) -> PersistentWorkerSession:
        if self.session is None:
            raise RuntimeError("persistent worker supervisor is not started")
        return self.session


def _cleanup_paths(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for label, path in paths.items():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{label}:{type(exc).__name__}")
    return errors
