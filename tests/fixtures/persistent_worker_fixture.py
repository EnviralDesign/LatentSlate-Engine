"""Tiny static authenticated handler for the persistent worker framework."""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latentslate_engine.runtime.framework.worker import (
    PersistentChildContext,
    atomic_write_json,
    hmac_sha256,
    parse_persistent_child_paths,
    result_hmac_sha256,
    run_persistent_child,
)

_SECRET = bytes.fromhex(os.environ["LATENTSLATE_PERSISTENT_FIXTURE_SECRET"])
_MODES = {
    "success",
    "failure",
    "crash_load",
    "crash_execute",
    "malformed_result",
    "oversized_result",
    "corrupt_result",
    "truncated_progress",
    "result_truncated_progress",
    "result_truncated_heartbeat",
    "heartbeat_stall",
    "stage_stall",
    "hard_busy",
    "ignore_cancel",
    "unload_failure",
}


def _binding(payload: Mapping[str, Any]) -> str:
    return hmac_sha256({key: value for key, value in payload.items() if key != "binding"}, _SECRET)


class FixtureHandler:
    def __init__(self) -> None:
        self.stage = "bind"
        self.unload_failure = False

    def bind_initial(
        self, payload: Any, context: PersistentChildContext
    ) -> dict[str, Any]:
        command = self._bind(payload, context)
        if command["sequence"] != 1:
            raise ValueError("fixture initial sequence is invalid")
        return command

    def load(
        self, command: dict[str, Any], context: PersistentChildContext
    ) -> dict[str, Any]:
        self.stage = "load"
        if command["mode"] == "crash_load":
            os._exit(8)
        return {"session_id": command["session_id"], "sequence": 0, "loads": 1}

    def bind_command(
        self,
        payload: Any,
        session: dict[str, Any],
        context: PersistentChildContext,
    ) -> dict[str, Any]:
        command = self._bind(payload, context)
        if command["session_id"] != session["session_id"]:
            raise ValueError("fixture command session is invalid")
        if command["sequence"] != session["sequence"] + 1:
            raise ValueError("fixture command sequence is invalid")
        return command

    def execute(
        self,
        session: dict[str, Any],
        command: dict[str, Any],
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> Mapping[str, Any]:
        self.stage = "execute"
        session["sequence"] = command["sequence"]
        context.publish_progress(0.25, "working")
        mode = command["mode"]
        if mode == "crash_execute":
            os._exit(9)
        if mode == "malformed_result":
            context.stop_heartbeat()
            context.paths.result.write_bytes(b"not-json")
            os._exit(0)
        if mode == "truncated_progress":
            with context.paths.progress.open("ab") as stream:
                stream.write(b'{"progress":0.5')
                stream.flush()
            context.stop_heartbeat()
            os._exit(0)
        if mode == "result_truncated_progress":
            context.stop_heartbeat()
            with context.paths.progress.open("ab") as stream:
                stream.write(b'{"progress":0.5')
                stream.flush()
            atomic_write_json(
                context.paths.result,
                self._success_result(session, command, context, cold=cold),
            )
            os._exit(0)
        if mode == "result_truncated_heartbeat":
            context.stop_heartbeat()
            with context.paths.heartbeat.open("ab") as stream:
                stream.write(b'{"heartbeat":')
                stream.flush()
            atomic_write_json(
                context.paths.result,
                self._success_result(session, command, context, cold=cold),
            )
            os._exit(0)
        if mode == "failure":
            raise RuntimeError("private fixture failure")
        if mode == "unload_failure":
            self.unload_failure = True
            raise RuntimeError("private primary failure")
        if mode == "heartbeat_stall":
            context.stop_heartbeat()
            time.sleep(30)
        if mode == "stage_stall":
            time.sleep(30)
        if mode == "hard_busy":
            while True:
                context.publish_progress(0.5, "still working")
                time.sleep(0.02)
        if mode == "ignore_cancel":
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            marker = command["marker_path"]
            if marker is not None:
                Path(marker).write_text(str(child.pid), encoding="utf-8")
            time.sleep(30)
        return self._success_result(session, command, context, cold=cold)

    def _success_result(
        self,
        session: dict[str, Any],
        command: dict[str, Any],
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> dict[str, Any]:
        unsigned: dict[str, Any] = {
            "schema_version": 1,
            "ok": True,
            "request_binding": context.binding,
            "session_id": session["session_id"],
            "sequence": command["sequence"],
            "value": command["value"],
            "worker_pid": os.getpid(),
            "loads": session["loads"],
            "warm": not cold,
        }
        if command["mode"] == "oversized_result":
            unsigned["padding"] = "x" * 4096
        signature = (
            "0" * 64
            if command["mode"] == "corrupt_result"
            else result_hmac_sha256(unsigned, _SECRET)
        )
        return {**unsigned, "result_binding": signature}

    def unload(
        self, session: dict[str, Any], context: PersistentChildContext
    ) -> None:
        count_path = context.paths.request.parent / "unload-count.txt"
        count = int(count_path.read_text(encoding="utf-8")) if count_path.exists() else 0
        count_path.write_text(str(count + 1), encoding="utf-8")
        if self.unload_failure:
            self.stage = "unload"
            raise RuntimeError("private unload failure")

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]:
        unsigned = {
            "schema_version": 1,
            "ok": False,
            "request_binding": context.binding,
            "error_type": type(exc).__name__,
            "failure_stage": self.stage,
            "error_fingerprint": hashlib.sha256(
                f"{type(exc).__name__}:{self.stage}".encode()
            ).hexdigest(),
        }
        return {**unsigned, "result_binding": result_hmac_sha256(unsigned, _SECRET)}

    def protocol_error(self, reason: str) -> BaseException:
        self.stage = "protocol"
        return ValueError(f"fixture protocol error: {reason}")

    def _bind(self, payload: Any, context: PersistentChildContext) -> dict[str, Any]:
        self.stage = "bind"
        if isinstance(payload, Mapping) and isinstance(payload.get("binding"), str):
            context.binding = payload["binding"]
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "session_id",
            "sequence",
            "mode",
            "value",
            "marker_path",
            "binding",
        }:
            raise ValueError("fixture request is invalid")
        if (
            payload["schema_version"] != 1
            or not isinstance(payload["session_id"], str)
            or isinstance(payload["sequence"], bool)
            or not isinstance(payload["sequence"], int)
            or payload["mode"] not in _MODES
            or not isinstance(payload["value"], str)
            or (payload["marker_path"] is not None and not isinstance(payload["marker_path"], str))
            or not hmac.compare_digest(payload["binding"], _binding(payload))
        ):
            raise ValueError("fixture request does not bind")
        return payload


def main(argv: list[str] | None = None) -> int:
    return run_persistent_child(
        parse_persistent_child_paths(argv),
        FixtureHandler(),
        maximum_bytes=2048,
        heartbeat_seconds=0.02,
    )


if __name__ == "__main__":
    raise SystemExit(main())
