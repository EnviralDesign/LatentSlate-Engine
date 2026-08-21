"""Model-neutral command-loop harness for persistent isolated workers."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Protocol, TypeVar

from .files import WorkerJsonFileError, atomic_write_json, read_bounded_json
from .progress import WorkerJsonlFileError, append_bounded_jsonl

BoundCommand = TypeVar("BoundCommand")
LoadedSession = TypeVar("LoadedSession")


@dataclass(frozen=True, slots=True)
class PersistentChildPaths:
    request: Path
    result: Path
    progress: Path
    heartbeat: Path
    start_gate: Path
    command: Path
    cancel: Path


@dataclass(slots=True)
class PersistentChildContext:
    paths: PersistentChildPaths
    maximum_bytes: int
    heartbeat_seconds: float
    protocol_error: Any
    binding: str = ""
    _last_progress: float = 0.0
    _heartbeat: tuple[Event, Thread] | None = None

    def publish_progress(self, value: float, message: str) -> None:
        value = max(self._last_progress, min(1.0, max(0.0, float(value))))
        self._last_progress = value
        try:
            append_bounded_jsonl(
                self.paths.progress,
                {"progress": float(value), "message": message},
                maximum_bytes=self.maximum_bytes,
            )
        except WorkerJsonlFileError as exc:
            raise self.protocol_error("progress_bound") from exc

    def reset_progress(self) -> None:
        self._last_progress = 0.0

    def start_heartbeat(self) -> None:
        if self._heartbeat is not None:
            raise RuntimeError("persistent child heartbeat is already active")
        stop = Event()

        def run() -> None:
            while not stop.is_set():
                try:
                    append_bounded_jsonl(
                        self.paths.heartbeat,
                        {"heartbeat": 1},
                        maximum_bytes=self.maximum_bytes,
                    )
                except WorkerJsonlFileError:
                    return
                stop.wait(self.heartbeat_seconds)

        thread = Thread(target=run, name="persistent-worker-heartbeat", daemon=True)
        thread.start()
        self._heartbeat = (stop, thread)

    def stop_heartbeat(self) -> None:
        heartbeat = self._heartbeat
        if heartbeat is None:
            return
        self._heartbeat = None
        heartbeat[0].set()
        heartbeat[1].join(timeout=2)
        if heartbeat[1].is_alive():
            raise self.protocol_error("heartbeat_stop")

    def publish_result(self, value: Mapping[str, Any]) -> None:
        self.stop_heartbeat()
        self.publish_progress(1.0, "Complete")
        atomic_write_json(self.paths.result, value)


class PersistentWorkerHandler(Protocol[BoundCommand, LoadedSession]):
    def bind_initial(
        self, payload: Any, context: PersistentChildContext
    ) -> BoundCommand: ...

    def load(
        self, command: BoundCommand, context: PersistentChildContext
    ) -> LoadedSession: ...

    def bind_command(
        self,
        payload: Any,
        session: LoadedSession,
        context: PersistentChildContext,
    ) -> BoundCommand: ...

    def execute(
        self,
        session: LoadedSession,
        command: BoundCommand,
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> Mapping[str, Any]: ...

    def unload(self, session: LoadedSession, context: PersistentChildContext) -> None: ...

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]: ...

    def protocol_error(self, reason: str) -> BaseException: ...


def parse_persistent_child_paths(
    argv: list[str] | None = None, *, description: str = "Persistent Engine worker"
) -> PersistentChildPaths:
    parser = argparse.ArgumentParser(description=description)
    for name in (
        "request",
        "result",
        "progress",
        "heartbeat",
        "start-gate",
        "command",
        "cancel",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    return PersistentChildPaths(
        **{
            name.replace("-", "_"): Path(getattr(args, name.replace("-", "_")))
            for name in (
                "request",
                "result",
                "progress",
                "heartbeat",
                "start-gate",
                "command",
                "cancel",
            )
        }
    )


def run_persistent_child(
    paths: PersistentChildPaths,
    handler: PersistentWorkerHandler[BoundCommand, LoadedSession],
    *,
    maximum_bytes: int = 1024 * 1024,
    heartbeat_seconds: float = 5.0,
) -> int:
    context = PersistentChildContext(
        paths=paths,
        maximum_bytes=maximum_bytes,
        heartbeat_seconds=heartbeat_seconds,
        protocol_error=handler.protocol_error,
    )
    session: LoadedSession | None = None
    try:
        _wait_file(paths.start_gate)
        initial = _read_payload(paths.request, maximum_bytes, handler)
        command = handler.bind_initial(initial, context)
        context.start_heartbeat()
        session = handler.load(command, context)
        result = handler.execute(session, command, context, cold=True)
        context.publish_result(result)
        while True:
            payload = _wait_payload(paths.command, maximum_bytes, handler)
            paths.command.unlink(missing_ok=True)
            command = handler.bind_command(payload, session, context)
            context.reset_progress()
            context.start_heartbeat()
            result = handler.execute(session, command, context, cold=False)
            context.publish_result(result)
    except BaseException as exc:  # noqa: BLE001 - bounded safe worker protocol
        try:
            context.stop_heartbeat()
        except BaseException:  # noqa: BLE001, S110 - retain primary failure
            pass
        if session is not None:
            try:
                handler.unload(session, context)
            except BaseException:  # noqa: BLE001, S110 - retain primary failure
                pass
        try:
            atomic_write_json(paths.result, handler.failure_result(exc, context))
        except BaseException:  # noqa: BLE001, S110 - no secondary channel
            pass
        return 1


def _wait_file(path: Path) -> None:
    while not path.is_file():
        time.sleep(0.01)


def _wait_payload(
    path: Path,
    maximum_bytes: int,
    handler: PersistentWorkerHandler[Any, Any],
) -> Any:
    while not path.is_file():
        time.sleep(0.02)
    return _read_payload(path, maximum_bytes, handler)


def _read_payload(
    path: Path,
    maximum_bytes: int,
    handler: PersistentWorkerHandler[Any, Any],
) -> Any:
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        raise handler.protocol_error("json_bound")
    try:
        value = read_bounded_json(path, maximum_bytes=maximum_bytes)
    except (OSError, UnicodeDecodeError, ValueError, WorkerJsonFileError) as exc:
        raise handler.protocol_error("json_invalid") from exc
    if not isinstance(value, Mapping):
        raise handler.protocol_error("json_type")
    return value
