"""Tiny static handler used to exercise the disposable worker framework."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latentslate_engine.runtime.framework.worker import (
    DisposableChildContext,
    parse_disposable_child_paths,
    run_disposable_child,
)


class FixtureHandler:
    def bind_request(
        self, payload: Any, context: DisposableChildContext
    ) -> dict[str, Any]:
        context.stage = "bind"
        if not isinstance(payload, dict) or set(payload) != {
            "binding",
            "mode",
            "value",
            "output_path",
            "unload_count_path",
        }:
            raise ValueError("fixture request is invalid")
        if payload["mode"] not in {
            "success",
            "failure",
            "sleep",
            "crash_bind",
            "crash_run",
            "truncated_progress",
            "malformed_result",
            "oversized_result",
            "unload_failure",
        }:
            raise ValueError("fixture mode is invalid")
        context.binding = str(payload["binding"])
        if payload["mode"] == "crash_bind":
            os._exit(8)
        return payload

    def load(
        self, request: dict[str, Any], context: DisposableChildContext
    ) -> object:
        context.stage = "load"
        return object()

    def run(
        self,
        runtime: object,
        request: dict[str, Any],
        context: DisposableChildContext,
    ) -> Mapping[str, Any]:
        context.stage = "run"
        context.publish_progress(0.25, "working")
        if request["mode"] == "crash_run":
            os._exit(9)
        if request["mode"] == "truncated_progress":
            with context.paths.progress.open("ab") as stream:
                stream.write(b'{"progress":0.5')
                stream.flush()
            os._exit(0)
        if request["mode"] == "malformed_result":
            context.paths.result.write_bytes(b"not-json")
            os._exit(0)
        if request["output_path"] is not None:
            Path(request["output_path"]).write_bytes(b"partial")
        if request["mode"] == "failure":
            raise RuntimeError("private fixture detail")
        if request["mode"] == "sleep":
            time.sleep(30)
        context.publish_progress(1.0, "complete")
        result = {
            "schema_version": 1,
            "ok": True,
            "request_binding": context.binding,
            "value": request["value"],
        }
        if request["mode"] == "oversized_result":
            result["padding"] = "x" * (1024 * 1024)
        return result

    def unload(
        self,
        runtime: object,
        request: dict[str, Any],
        context: DisposableChildContext,
    ) -> None:
        context.stage = "unload"
        count_path = request["unload_count_path"]
        if count_path is not None:
            path = Path(count_path)
            count = int(path.read_text(encoding="utf-8")) if path.exists() else 0
            path.write_text(str(count + 1), encoding="utf-8")
        if request["mode"] == "unload_failure":
            raise RuntimeError("private unload detail")

    def failure_result(
        self, exc: BaseException, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        fingerprint = hashlib.sha256(
            f"{type(exc).__name__}:{exc}".encode()
        ).hexdigest()
        return {
            "schema_version": 1,
            "ok": False,
            "request_binding": context.binding,
            "error_type": type(exc).__name__,
            "failure_stage": context.stage,
            "error_fingerprint": fingerprint,
        }

    def stage_for_progress(self, message: str | None) -> str:
        return "complete" if message == "complete" else "run"

    def protocol_error(self, reason: str) -> BaseException:
        if reason == "start_gate_timeout":
            return TimeoutError("fixture gate timeout")
        return ValueError(f"fixture protocol error: {reason}")


def main(argv: list[str] | None = None) -> int:
    return run_disposable_child(
        parse_disposable_child_paths(argv),
        FixtureHandler(),
        maximum_json_bytes=1024,
        maximum_progress_bytes=1024,
        start_gate_timeout_seconds=2.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
