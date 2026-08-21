"""Disposable-process supervisor for native LTX 2.3 generation.

The LTX BF16 closure can leave almost all 16 GB of CUDA allocation resident after
Python-level teardown on Windows.  The Engine process therefore owns only this
small supervisor; a fresh worker owns loading, denoising, and muxing for one job.
Worker-tree exit is the authoritative CUDA/heap release boundary.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import Settings
from .framework.worker import (
    DisposableWorkerExited,
    DisposableWorkerLimits,
    DisposableWorkerPaths,
    DisposableWorkerProgressTruncated,
    DisposableWorkerRunState,
    DisposableWorkerSupervisor,
    WorkerJsonFileError,
    WorkerJsonlFileError,
    is_worker_cancellation,
    read_bounded_json,
    sha256_fingerprint,
)
from .kit import ResolvedRuntimePlan

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
        self._active_supervisor: DisposableWorkerSupervisor | None = None
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
        endpoints = DisposableWorkerPaths(
            request=paths["request"],
            result=paths["result"],
            progress=paths["progress"],
            start_gate=paths["gate"],
        )
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        allocator_policy: str | None = env["PYTORCH_CUDA_ALLOC_CONF"]
        supervisor = DisposableWorkerSupervisor(
            command=(
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
            ),
            paths=endpoints,
            cleanup_paths=paths,
            failure_outputs=(Path(output_path),),
            environment=env,
            limits=DisposableWorkerLimits(
                maximum_json_bytes=_MAX_JSON_BYTES,
                maximum_progress_bytes=_MAX_PROGRESS_BYTES,
                maximum_progress_records=_MAX_PROGRESS_RECORDS,
                poll_seconds=_POLL_SECONDS,
            ),
        )
        try:
            _require_fresh(paths)
            self._active_supervisor = supervisor
            try:
                raw_result = supervisor.run(
                    payload,
                    before_spawn=plan.revalidate_components,
                    progress=lambda value: _publish_progress(value, progress),
                    check_cancelled=check_cancelled,
                )
            except DisposableWorkerExited as exc:
                raise RuntimeError(
                    _worker_error_value(
                        exc.result, exc.exit_code, payload["request_binding"]
                    )
                ) from exc
            except WorkerJsonlFileError as exc:
                raise RuntimeError(
                    "LTX worker progress is invalid or exceeds its bound"
                ) from exc
            except DisposableWorkerProgressTruncated as exc:
                raise RuntimeError(
                    "LTX worker ended with a truncated progress record"
                ) from exc
            except WorkerJsonFileError as exc:
                raise RuntimeError(
                    "LTX worker result is missing or exceeds its bound"
                ) from exc
            result = _validate_success(
                raw_result, output_path, payload["request_binding"]
            )
            _validate_metadata(result["metadata"], payload)
            allocator_policy = result["allocator_policy"]
            state = supervisor.last_run
            if state is None or state.exit_code is None:
                raise RuntimeError("LTX worker exited without a closed run state")
            self._last_worker = _last_worker(state, allocator_policy=allocator_policy)
            return result["metadata"]
        except BaseException as primary:
            if supervisor.last_run is not None:
                self._last_worker = _last_worker(
                    supervisor.last_run,
                    outcome="canceled" if is_worker_cancellation(primary) else "failed",
                    allocator_policy=allocator_policy,
                )
            raise
        finally:
            self._active_supervisor = None
            supervisor.cleanup()
            self._cleanup_errors = list(supervisor.cleanup_errors)
            self._ownership.release()

    def status(self) -> dict[str, Any]:
        return {
            "family": "ltx23",
            "runtime": "ltx23_disposable_worker",
            "pipeline_fingerprint": self.plan.pipeline_fingerprint,
            "loaded": False,
            "active_worker": self._active_supervisor is not None,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        """No heavyweight state or tensor cache is retained by the parent."""

    def unload(self) -> None:
        supervisor = self._active_supervisor
        if supervisor is None:
            return
        supervisor.terminate()
        self._active_supervisor = None


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


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonFileError:
        raise RuntimeError("LTX worker result is missing or exceeds its bound")


def _read_success(path: Path, output: Path, binding: str) -> dict[str, Any]:
    return _validate_success(_read_json(path), output, binding)


def _validate_success(result: Any, output: Path, binding: str) -> dict[str, Any]:
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


def _publish_progress(
    value: Any, callback: Callable[[float, str | None], None]
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"progress", "message"}
        or isinstance(value["progress"], bool)
        or not isinstance(value["progress"], (int, float))
        or not isinstance(value["message"], (str, type(None)))
    ):
        raise RuntimeError("LTX worker progress record is invalid")
    callback(float(value["progress"]), value["message"])


def _worker_error(path: Path, exit_code: int, binding: str) -> str:
    try:
        return _worker_error_value(_read_json(path), exit_code, binding)
    except (OSError, RuntimeError, ValueError):
        return f"LTX worker exited with code {exit_code}"


def _worker_error_value(result: Any, exit_code: int, binding: str) -> str:
    if (
        isinstance(result, dict)
        and set(result)
        == {"schema_version", "ok", "request_binding", "error_type", "error"}
        and result.get("schema_version") == _SCHEMA_VERSION
        and result.get("ok") is False
        and result.get("request_binding") == binding
        and isinstance(result.get("error_type"), str)
        and isinstance(result.get("error"), str)
    ):
        return f"LTX worker failed ({result['error_type']})"
    return f"LTX worker exited with code {exit_code}"


def _last_worker(
    state: DisposableWorkerRunState,
    *,
    allocator_policy: str | None,
    outcome: str | None = None,
) -> dict[str, object]:
    return {
        "pid": state.pid,
        "exit_code": state.exit_code,
        "terminated": state.terminated,
        "outcome": state.outcome if outcome is None else outcome,
        "tree_empty": state.tree_empty,
        "memory_boundary": "disposable_process_exit",
        "allocator_policy": allocator_policy,
    }
