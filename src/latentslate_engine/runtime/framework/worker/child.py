"""Model-neutral child harness for one-shot isolated workers."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from .files import WorkerJsonFileError, atomic_write_json, read_bounded_json
from .progress import WorkerJsonlFileError, append_bounded_jsonl

BoundRequest = TypeVar("BoundRequest")
LoadedRuntime = TypeVar("LoadedRuntime")


@dataclass(frozen=True, slots=True)
class DisposableChildPaths:
    """File endpoints parsed by a disposable child entry point."""

    request: Path
    result: Path
    progress: Path
    start_gate: Path


@dataclass(slots=True)
class DisposableChildContext:
    """Safe diagnostic state and bounded progress publisher for one child run."""

    paths: DisposableChildPaths
    maximum_progress_bytes: int
    stage_for_progress: Callable[[str | None], str]
    protocol_error: Callable[[str], BaseException]
    stage: str = "worker_startup"
    binding: str | None = None

    def publish_progress(self, progress: float, message: str | None) -> None:
        value = {"progress": progress, "message": message}
        if (
            isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not 0 <= float(progress) <= 1
            or not isinstance(message, (str, type(None)))
        ):
            raise self.protocol_error("invalid_progress")
        self.stage = self.stage_for_progress(message)
        try:
            append_bounded_jsonl(
                self.paths.progress,
                value,
                maximum_bytes=self.maximum_progress_bytes,
            )
        except WorkerJsonlFileError as exc:
            raise self.protocol_error("progress_bound") from exc


class DisposableWorkerHandler(Protocol[BoundRequest, LoadedRuntime]):
    """Static model adapter consumed by the generic child harness."""

    def bind_request(
        self, payload: Any, context: DisposableChildContext
    ) -> BoundRequest: ...

    def load(
        self, request: BoundRequest, context: DisposableChildContext
    ) -> LoadedRuntime: ...

    def run(
        self,
        runtime: LoadedRuntime,
        request: BoundRequest,
        context: DisposableChildContext,
    ) -> Mapping[str, Any]: ...

    def unload(
        self,
        runtime: LoadedRuntime,
        request: BoundRequest,
        context: DisposableChildContext,
    ) -> None: ...

    def failure_result(
        self, exc: BaseException, context: DisposableChildContext
    ) -> Mapping[str, Any]: ...

    def stage_for_progress(self, message: str | None) -> str: ...

    def protocol_error(self, reason: str) -> BaseException: ...


def parse_disposable_child_paths(
    argv: list[str] | None = None, *, description: str = "Disposable Engine worker"
) -> DisposableChildPaths:
    """Parse the closed four-path CLI shared by disposable child entry points."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    args = parser.parse_args(argv)
    return DisposableChildPaths(
        request=Path(args.request),
        result=Path(args.result),
        progress=Path(args.progress),
        start_gate=Path(args.start_gate),
    )


def run_disposable_child(
    paths: DisposableChildPaths,
    handler: DisposableWorkerHandler[BoundRequest, LoadedRuntime],
    *,
    maximum_json_bytes: int = 1024 * 1024,
    maximum_progress_bytes: int = 1024 * 1024,
    start_gate_timeout_seconds: float = 60.0,
) -> int:
    """Run a statically provided handler and publish one success or safe failure."""

    context = DisposableChildContext(
        paths=paths,
        maximum_progress_bytes=maximum_progress_bytes,
        stage_for_progress=handler.stage_for_progress,
        protocol_error=handler.protocol_error,
    )
    runtime: LoadedRuntime | None = None
    request: BoundRequest | None = None
    unload_attempted = False
    try:
        _wait_start_gate(
            paths.start_gate,
            timeout_seconds=start_gate_timeout_seconds,
            timeout_error=lambda: handler.protocol_error("start_gate_timeout"),
        )
        context.stage = "read_request"
        try:
            payload = read_bounded_json(paths.request, maximum_bytes=maximum_json_bytes)
        except WorkerJsonFileError as exc:
            raise handler.protocol_error("request_bound") from exc
        request = handler.bind_request(payload, context)
        runtime = handler.load(request, context)
        result = handler.run(runtime, request, context)
        unload_attempted = True
        handler.unload(runtime, request, context)
        runtime = None
        atomic_write_json(paths.result, result)
        return 0
    except BaseException as exc:  # noqa: BLE001 - bounded worker result protocol
        if runtime is not None and request is not None and not unload_attempted:
            unload_attempted = True
            failure_stage = context.stage
            try:
                handler.unload(runtime, request, context)
            except BaseException:  # noqa: BLE001, S110 - retain primary failure
                pass
            finally:
                context.stage = failure_stage
        try:
            atomic_write_json(paths.result, handler.failure_result(exc, context))
        except BaseException:  # noqa: BLE001, S110 - child cannot report further
            pass
        return 1


def _wait_start_gate(
    path: Path,
    *,
    timeout_seconds: float,
    timeout_error: Callable[[], BaseException],
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise timeout_error()
        time.sleep(0.02)
