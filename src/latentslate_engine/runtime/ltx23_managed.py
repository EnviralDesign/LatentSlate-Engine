"""Disposable-process supervisor for native LTX 2.3 generation.

The LTX BF16 closure can leave almost all 16 GB of CUDA allocation resident after
Python-level teardown on Windows.  The Engine process therefore owns only this
small supervisor; a fresh worker owns loading, denoising, and muxing for one job.
Worker-tree exit is the authoritative CUDA/heap release boundary.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import Settings
from .framework.worker import (
    JsonlCursor,
    WorkerJsonFileError,
    WorkerJsonlFileError,
    atomic_write_json,
    drain_bounded_jsonl,
    read_bounded_json,
    sha256_fingerprint,
)
from .kit import ResolvedRuntimePlan
from .windows_process import DisposableProcessTree

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_OPERATIONS = frozenset({"t2v", "first_frame", "first_last"})


@dataclass(frozen=True, slots=True)
class ManagedLTX23Result:
    metadata: dict[str, Any]
    worker_pid: int
    worker_exit_code: int


class ManagedLTX23Runtime:
    """Run one identity-bound LTX operation in a kill-on-close worker tree."""

    def __init__(self, settings: Settings, plan: ResolvedRuntimePlan, *, operation: str) -> None:
        if plan.family != "ltx23" or operation not in _OPERATIONS:
            raise ValueError("managed LTX runtime requires a supported LTX operation")
        self.settings = settings
        self.plan = plan
        self.operation = operation
        self._active_tree: DisposableProcessTree | None = None
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []
        self._ownership = Lock()

    def generate(
        self,
        *,
        plan: ResolvedRuntimePlan,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        duration_seconds: float,
        seed: int,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
        start_image_path: Path | None = None,
        end_image_path: Path | None = None,
    ) -> dict[str, Any]:
        check_cancelled()
        self.plan.assert_same_pipeline(plan)
        _validate_operation_paths(self.operation, start_image_path, end_image_path)
        plan.revalidate_components()
        payload = _payload(
            self.settings,
            plan,
            operation=self.operation,
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            seed=seed,
            start_image_path=start_image_path,
            end_image_path=end_image_path,
        )
        paths = _paths(output_path)
        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("managed LTX worker is already active")
        process: subprocess.Popen[bytes] | None = None
        tree: DisposableProcessTree | None = None
        allocator_policy: str | None = None
        try:
            if self._active_tree is not None:
                raise RuntimeError("managed LTX worker is already active")
            _require_fresh(paths)
            _write_json(paths["request"], payload)
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            allocator_policy = env.get("PYTORCH_CUDA_ALLOC_CONF")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "latentslate_engine.runtime.ltx23_worker",
                    "--request",
                    str(paths["request"]),
                    "--result",
                    str(paths["result"]),
                    "--progress",
                    str(paths["progress"]),
                    "--start-gate",
                    str(paths["gate"]),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=env,
            )
            tree = DisposableProcessTree(process)
            self._active_tree = tree
            paths["gate"].touch(exist_ok=False)
            _wait(process, paths["progress"], progress, check_cancelled)
            exit_code = process.wait(timeout=5)
            if exit_code != 0:
                raise RuntimeError(
                    _worker_error(paths["result"], exit_code, payload["request_binding"])
                )
            result = _read_success(paths["result"], output_path, payload["request_binding"])
            _validate_metadata(result["metadata"], payload)
            tree.wait_for_empty()
            tree.close()
            tree = None
            self._active_tree = None
            self._last_worker = {
                "pid": process.pid,
                "exit_code": exit_code,
                "terminated": True,
                "outcome": "succeeded",
                "tree_empty": True,
                "memory_boundary": "disposable_process_exit",
                "allocator_policy": result["allocator_policy"],
            }
            return result["metadata"]
        except BaseException as primary:
            tree_empty = _terminate_tree(tree, process, primary)
            if process is not None:
                self._last_worker = {
                    "pid": process.pid,
                    "exit_code": process.poll(),
                    "terminated": True,
                    "outcome": "canceled" if isinstance(primary, asyncio.CancelledError) else "failed",
                    "tree_empty": tree_empty,
                    "memory_boundary": "disposable_process_exit",
                    "allocator_policy": allocator_policy,
                }
            _remove_output(output_path, primary)
            raise
        finally:
            self._active_tree = None
            if tree is not None:
                _close_tree(tree)
            self._cleanup_errors = _cleanup(paths, output_path)
            self._ownership.release()

    def status(self) -> dict[str, Any]:
        return {
            "family": "ltx23",
            "runtime": "ltx23_disposable_worker",
            "pipeline_fingerprint": self.plan.pipeline_fingerprint,
            "loaded": False,
            "active_worker": self._active_tree is not None,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        """No heavyweight state or tensor cache is retained by the parent."""

    def unload(self) -> None:
        tree = self._active_tree
        if tree is None:
            return
        try:
            tree.terminate()
            tree.wait_for_empty()
        finally:
            tree.close()
            self._active_tree = None


def _validate_operation_paths(operation: str, start: Path | None, end: Path | None) -> None:
    if operation == "t2v" and (start is not None or end is not None):
        raise ValueError("LTX T2V does not accept endpoint images")
    if operation == "first_frame" and (start is None or end is not None):
        raise ValueError("LTX first-frame generation requires only a start image")
    if operation == "first_last" and (start is None or end is None):
        raise ValueError("LTX first+last generation requires both endpoint images")
    for label, path in (("start", start), ("end", end)):
        if path is not None and not Path(path).is_file():
            raise ValueError(f"LTX {label} image does not exist or is not a file: {path}")


def _payload(settings: Settings, plan: ResolvedRuntimePlan, **request: Any) -> dict[str, Any]:
    serialized_plan = {
        "pipeline_fingerprint": plan.pipeline_fingerprint,
        "model_id": plan.model_id,
        "model_resource_id": plan.model_resource_id,
        "model_path": str(plan.model_path),
        "model_format": plan.model_format,
        "model_precision": plan.model_precision,
        "model_quantization": plan.model_quantization,
        "device": plan.device,
        "quantization": plan.quantization,
        "attention": plan.attention,
        "offload": plan.offload,
        "vae_tiling": plan.vae_tiling,
        "vae_slicing": plan.vae_slicing,
        "cache": "none",
        "low_cpu_mem_usage": plan.low_cpu_mem_usage,
        "components": [
            {"name": item.name, "path": str(item.path), "signature": item.signature}
            for item in plan.components
        ],
    }
    generation = {
        "prompt": request["prompt"],
        "width": request["width"],
        "height": request["height"],
        "duration_seconds": request["duration_seconds"],
        "seed": request["seed"],
        "start_image_path": None
        if request["start_image_path"] is None
        else str(Path(request["start_image_path"]).resolve(strict=True)),
        "end_image_path": None
        if request["end_image_path"] is None
        else str(Path(request["end_image_path"]).resolve(strict=True)),
        "output_path": str(Path(request["output_path"]).resolve(strict=False)),
    }
    unsigned = {
        "schema_version": _SCHEMA_VERSION,
        "operation": request["operation"],
        "settings": {
            "home": str(settings.home),
            "model_id": settings.ltx23_model_id,
            "profile": settings.ltx23_profile,
            "device": settings.ltx23_device,
        },
        "plan": serialized_plan,
        "generation": generation,
    }
    return {**unsigned, "request_binding": _fingerprint(unsigned)}


def _paths(output_path: Path) -> dict[str, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    return {key: output.parent / f".{stem}.ltx23-worker-{key}{suffix}" for key, suffix in {
        "request": ".json", "result": ".json", "progress": ".jsonl", "gate": ""
    }.items()}


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_fingerprint(value)


def _require_fresh(paths: Mapping[str, Path]) -> None:
    stale = [name for name, path in paths.items() if path.exists()]
    if stale:
        raise RuntimeError("LTX worker IPC paths already exist: " + ", ".join(stale))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonFileError:
        raise RuntimeError("LTX worker result is missing or exceeds its bound")


def _read_success(path: Path, output: Path, binding: str) -> dict[str, Any]:
    result = _read_json(path)
    if not isinstance(result, dict) or set(result) != {
        "schema_version", "ok", "request_binding", "output_path", "output_size_bytes", "metadata",
        "allocator_policy",
    } or result["schema_version"] != _SCHEMA_VERSION or result["ok"] is not True:
        raise RuntimeError("LTX worker returned an invalid success result")
    if result["request_binding"] != binding:
        raise RuntimeError("LTX worker result does not bind to this request")
    expected = Path(output).resolve(strict=True)
    if Path(result["output_path"]).resolve(strict=True) != expected:
        raise RuntimeError("LTX worker published an unexpected output path")
    if not isinstance(result["output_size_bytes"], int) or result["output_size_bytes"] != expected.stat().st_size or result["output_size_bytes"] <= 0:
        raise RuntimeError("LTX worker output size is invalid")
    if not isinstance(result["metadata"], dict):
        raise TypeError("LTX worker metadata is invalid")
    if not isinstance(result["allocator_policy"], str) or not result["allocator_policy"]:
        raise TypeError("LTX worker allocator policy is invalid")
    return result


def _validate_metadata(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    plan = payload["plan"]
    generation = payload["generation"]
    if metadata.get("pipeline_fingerprint") != plan["pipeline_fingerprint"]:
        raise RuntimeError("LTX worker metadata has an unexpected pipeline fingerprint")
    if metadata.get("seed") != generation["seed"] or metadata.get("model_id") != (
        plan["model_resource_id"] or plan["model_id"]
    ):
        raise RuntimeError("LTX worker metadata is not bound to the requested model/seed")
    if payload["operation"] == "t2v" and "conditioning" in metadata:
        raise RuntimeError("LTX T2V worker unexpectedly returned conditioning")
    expected_mode = {"first_frame": "first_frame", "first_last": "first_last_frame"}.get(payload["operation"])
    if expected_mode is not None and metadata.get("conditioning", {}).get("mode") != expected_mode:
        raise RuntimeError("LTX worker conditioning does not match its operation")


def _wait(process: subprocess.Popen[bytes], progress_path: Path, progress: Callable[[float, str | None], None], cancelled: Callable[[], None]) -> None:
    offset = 0
    records = 0
    pending = b""
    while process.poll() is None:
        cancelled()
        offset, pending, records = _drain(progress_path, offset, pending, records, progress)
        time.sleep(_POLL_SECONDS)
    _offset, pending, _records = _drain(progress_path, offset, pending, records, progress)
    if pending and process.poll() == 0:
        raise RuntimeError("LTX worker ended with a truncated progress record")


def _drain(path: Path, offset: int, pending: bytes, records: int, callback: Callable[[float, str | None], None]) -> tuple[int, bytes, int]:
    try:
        cursor, items = drain_bounded_jsonl(
            path,
            JsonlCursor(offset=offset, pending=pending, records=records),
            maximum_bytes=_MAX_PROGRESS_BYTES,
            maximum_records=_MAX_PROGRESS_RECORDS,
        )
    except WorkerJsonlFileError as exc:
        raise RuntimeError("LTX worker progress is invalid or exceeds its bound") from exc
    for item in items:
        if not isinstance(item, dict) or set(item) != {"progress", "message"} or not isinstance(item["progress"], (int, float)) or not isinstance(item["message"], (str, type(None))):
            raise RuntimeError("LTX worker progress record is invalid")
        callback(float(item["progress"]), item["message"])
    return cursor.offset, cursor.pending, cursor.records


def _worker_error(path: Path, exit_code: int, binding: str) -> str:
    try:
        result = _read_json(path)
        if (
            isinstance(result, dict)
            and set(result) == {"schema_version", "ok", "request_binding", "error_type", "error"}
            and result.get("schema_version") == _SCHEMA_VERSION
            and result.get("ok") is False
            and result.get("request_binding") == binding
            and isinstance(result.get("error_type"), str)
            and isinstance(result.get("error"), str)
        ):
            return f"LTX worker failed ({result['error_type']})"
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return f"LTX worker exited with code {exit_code}"
    return f"LTX worker exited with code {exit_code}"


def _terminate_tree(
    tree: DisposableProcessTree | None,
    process: subprocess.Popen[bytes] | None,
    primary: BaseException,
) -> bool:
    try:
        if tree is not None:
            tree.terminate()
            tree.wait_for_empty()
            return True
        _terminate_direct_process(process)
        return True
    except BaseException as safety_error:
        safety_error.add_note(f"while handling LTX worker failure: {type(primary).__name__}: {primary}")
        raise


def _terminate_direct_process(process: subprocess.Popen[bytes] | None) -> None:
    """Fallback when Job Object creation/assignment failed after Popen."""

    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if process.poll() is None:
        raise RuntimeError("LTX worker direct-process termination was not confirmed")


def _remove_output(path: Path, primary: BaseException) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"LTX partial output cleanup failed: {exc}")


def _close_tree(tree: DisposableProcessTree) -> None:
    try:
        tree.close()
    except OSError as exc:
        primary = sys.exc_info()[1]
        if primary is None:
            raise
        primary.add_note(f"LTX worker Job Object close failed: {exc}")


def _cleanup(paths: Mapping[str, Path], output: Path) -> list[str]:
    errors: list[str] = []
    for path in paths.values():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            errors.append("ipc")
    parent = Path(output).parent
    prefix = f".{Path(output).name}."
    for temp in parent.glob(f"{prefix}*.tmp.mp4"):
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            errors.append("staging")
    return sorted(set(errors))
