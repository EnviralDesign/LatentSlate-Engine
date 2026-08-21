"""Disposable parent supervisor for Engine-native Wan 2.2 TI2V 5B."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..wan5_kitchen_recipe import (
    WAN5_FPS,
    WAN5_GUIDANCE_SCALE,
    WAN5_STEPS,
    Wan5KitchenRuntimeRequest,
    revalidate_wan5_kitchen_runtime_request,
)
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
    sha256_fingerprint,
)

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_LOGGER = logging.getLogger(__name__)
_WAN5_UNIPC_SOLVER_BRIDGE = "comfy/uni_pc-vp-flow-v1"
_WAN5_UNIPC_TERMINAL_SIGMA = 0.001
_OBSERVED_DURATION_TOLERANCE_SECONDS = 1e-9


@dataclass(frozen=True, slots=True)
class ManagedWan5KitchenResult:
    output_path: Path
    output_size_bytes: int
    metadata: dict[str, Any]
    worker_pid: int
    worker_exit_code: int


class ManagedWan5KitchenRuntime:
    """Run exactly one generation in a fresh kill-on-close worker tree."""

    def __init__(self, request: Wan5KitchenRuntimeRequest) -> None:
        if request.operation not in {"wan5_t2v", "wan5_i2v"}:
            raise ValueError("managed Wan 5B runtime requires a supported operation")
        self.request = request
        self._active_supervisor: DisposableWorkerSupervisor | None = None
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []
        self._ownership = Lock()

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        start_image_path: Path | None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> ManagedWan5KitchenResult:
        check_cancelled()
        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("managed Wan 5B worker is already active")
        paths = _paths(output_path)
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
                "latentslate_engine.runtime.wan5_kitchen_worker",
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
            generation = _generation(
                self.request.operation,
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                num_frames=num_frames,
                seed=seed,
                start_image_path=start_image_path,
                staging_output_path=paths["staging"],
            )
            _validate_generation(generation, self.request.operation)
            _require_fresh(paths)
            payload = _payload(self.request, generation)
            self._active_supervisor = supervisor
            try:
                raw_result = supervisor.run(
                    payload,
                    before_spawn=lambda: _revalidate_before_spawn(self.request),
                    progress=lambda value: progress(
                        float(value["progress"]), value.get("message")
                    ),
                    check_cancelled=check_cancelled,
                )
            except DisposableWorkerExited as exc:
                failure = _worker_failure_value(
                    exc.result, exc.exit_code, payload["request_binding"]
                )
                _log_worker_failure(failure)
                raise RuntimeError(failure["message"])
            except WorkerJsonlFileError as exc:
                raise RuntimeError(
                    "Wan 5B worker progress is invalid or exceeds its bound"
                ) from exc
            except DisposableWorkerProgressTruncated as exc:
                raise RuntimeError(
                    "Wan 5B worker ended with a truncated progress record"
                ) from exc
            except WorkerJsonFileError as exc:
                raise RuntimeError(
                    "Wan 5B worker JSON is missing or exceeds its bound"
                ) from exc
            result = _validate_success(raw_result, output_path, payload["request_binding"])
            _validate_metadata(result["metadata"], self.request, generation)
            allocator_policy = result["allocator_policy"]
            state = supervisor.last_run
            assert state is not None
            self._last_worker = _last_worker(state, allocator_policy=allocator_policy)
            return ManagedWan5KitchenResult(
                Path(output_path).resolve(strict=True),
                result["output_size_bytes"],
                result["metadata"],
                state.pid,
                int(state.exit_code),
            )
        except BaseException as primary:
            if supervisor.last_run is not None:
                self._last_worker = _last_worker(
                    supervisor.last_run,
                    outcome="canceled" if is_worker_cancellation(primary) else "failed",
                    allocator_policy=allocator_policy,
                )
            if supervisor.last_run is not None:
                Path(output_path).unlink(missing_ok=True)
            raise
        finally:
            self._active_supervisor = None
            supervisor.cleanup()
            self._cleanup_errors = list(supervisor.cleanup_errors)
            self._ownership.release()

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "runtime": "engine-native/wan5-kitchen-disposable-worker",
            "request_fingerprint": self.request.fingerprint,
            "component_fingerprint": self.request.component_fingerprint,
            "loaded": False,
            "active_worker": self._active_supervisor is not None,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False, "tensor": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        pass

    def unload(self) -> None:
        supervisor = self._active_supervisor
        if supervisor is None:
            return
        supervisor.terminate()
        self._active_supervisor = None


def _generation(operation: str, **values: Any) -> dict[str, object]:
    start = _endpoint(values["start_image_path"])
    return {
        "operation": operation,
        "prompt": values["prompt"],
        "width": values["width"],
        "height": values["height"],
        "num_frames": values["num_frames"],
        "seed": values["seed"],
        "output_path": str(Path(values["output_path"]).resolve(strict=False)),
        "staging_output_path": str(Path(values["staging_output_path"]).resolve(strict=False)),
        "start_image_path": None if start is None else start["path"],
        "start_image_identity": None if start is None else start["identity"],
    }


def _validate_generation(value: Mapping[str, object], operation: str) -> None:
    if (
        value.get("operation") != operation
        or not isinstance(value.get("prompt"), str)
        or not str(value["prompt"]).strip()
    ):
        raise ValueError("Wan 5B managed generation is invalid")
    for key in ("width", "height", "num_frames", "seed"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int):
            raise TypeError(f"Wan 5B {key} must be an integer")
    width, height, frames, seed = (
        int(value["width"]),
        int(value["height"]),
        int(value["num_frames"]),
        int(value["seed"]),
    )
    if width <= 0 or height <= 0 or width % 32 or height % 32 or width * height > 901_120:
        raise ValueError("Wan 5B dimensions are outside the exact runtime contract")
    if not 25 <= frames <= 121 or frames % 4 != 1 or seed < 0:
        raise ValueError("Wan 5B frames or seed are outside the exact runtime contract")
    start, identity = value.get("start_image_path"), value.get("start_image_identity")
    if operation == "wan5_t2v":
        if start is not None or identity is not None:
            raise ValueError("Wan 5B T2V cannot receive an image")
    elif (
        not isinstance(start, str)
        or not isinstance(identity, Mapping)
        or _endpoint_identity(Path(start)) != dict(identity)
    ):
        raise ValueError("Wan 5B I2V requires an identity-bound image")
    output = value.get("output_path")
    staging = value.get("staging_output_path")
    if (
        not isinstance(output, str)
        or Path(output).exists()
        or Path(output).suffix.lower() != ".mp4"
        or not isinstance(staging, str)
        or Path(staging).exists()
        or Path(staging).suffix.lower() != ".mp4"
        or Path(staging).resolve(strict=False).parent != Path(output).resolve(strict=False).parent
        or Path(staging).resolve(strict=False) == Path(output).resolve(strict=False)
    ):
        raise ValueError("Wan 5B output and staging paths must be fresh sibling MP4 paths")


def _payload(
    request: Wan5KitchenRuntimeRequest, generation: Mapping[str, object]
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "request": request.to_json_dict(),
        "generation": dict(generation),
        "device": "cuda",
    }
    return {**unsigned, "request_binding": _fingerprint(unsigned)}


def _paths(output_path: Path) -> dict[str, Path]:
    output = Path(output_path).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    return {
        key: output.parent / f".{output.stem}.wan5-kitchen-worker-{key}{suffix}"
        for key, suffix in {
            "request": ".json",
            "result": ".json",
            "progress": ".jsonl",
            "gate": "",
            "staging": ".mp4",
        }.items()
    }


def _endpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("Wan 5B endpoint is not a file")
    return {"path": str(candidate), "identity": _endpoint_identity(candidate)}


def _endpoint_identity(path: Path) -> dict[str, int | str]:
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Wan 5B endpoint changed while measuring identity")
    return {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def _fingerprint(value: Mapping[str, object]) -> str:
    return sha256_fingerprint(value)


def _require_fresh(paths: Mapping[str, Path]) -> None:
    stale = sorted(name for name, path in paths.items() if path.exists())
    if stale:
        raise RuntimeError("Wan 5B worker IPC paths already exist: " + ", ".join(stale))


def _validate_success(result: Any, output: Path, binding: str) -> dict[str, Any]:
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "schema_version",
            "ok",
            "request_binding",
            "output_path",
            "output_size_bytes",
            "metadata",
            "allocator_policy",
        }
        or result.get("schema_version") != _SCHEMA_VERSION
        or result.get("ok") is not True
        or result.get("request_binding") != binding
    ):
        raise RuntimeError("Wan 5B worker returned an invalid success result")
    expected_output = Path(output).resolve(strict=True)
    if Path(result["output_path"]).resolve(strict=True) != expected_output:
        raise RuntimeError("Wan 5B worker published an unexpected output")
    if result.get("output_size_bytes") != expected_output.stat().st_size:
        raise RuntimeError("Wan 5B worker output size is invalid")
    if not isinstance(result.get("metadata"), dict) or not isinstance(
        result.get("allocator_policy"), str
    ):
        raise TypeError("Wan 5B worker metadata is invalid")
    if result["metadata"].get("output_sha256") != _sha256_file(expected_output):
        raise RuntimeError("Wan 5B worker output hash is invalid")
    return result


def _revalidate_before_spawn(request: Wan5KitchenRuntimeRequest) -> None:
    if not revalidate_wan5_kitchen_runtime_request(request):
        raise RuntimeError("Wan 5B request changed immediately before worker spawn")


def _validate_metadata(
    metadata: Mapping[str, object],
    request: Wan5KitchenRuntimeRequest,
    generation: Mapping[str, object],
) -> None:
    sampling = metadata.get("sampling")
    dispatch = metadata.get("kitchen_dispatch")
    output = metadata.get("output")
    time_base = output.get("time_base") if isinstance(output, Mapping) else None
    observed_fps = output.get("fps") if isinstance(output, Mapping) else None
    observed_frames = output.get("frame_count") if isinstance(output, Mapping) else None
    observed_duration = output.get("duration_seconds") if isinstance(output, Mapping) else None
    if (
        metadata.get("family") != "wan22"
        or metadata.get("runtime") != "engine-native/wan5-kitchen"
        or metadata.get("operation") != request.operation
        or metadata.get("request_fingerprint") != request.fingerprint
        or metadata.get("component_fingerprint") != request.component_fingerprint
        or metadata.get("seed") != generation["seed"]
        or metadata.get("width") != generation["width"]
        or metadata.get("height") != generation["height"]
        or metadata.get("frame_count") != observed_frames
        or metadata.get("fps") != observed_fps
        or metadata.get("duration_seconds") != observed_duration
        or not isinstance(sampling, Mapping)
        or sampling.get("steps") != WAN5_STEPS
        or sampling.get("source_schedule_steps") != WAN5_STEPS + 1
        or sampling.get("discard_penultimate_sigma") is not True
        or sampling.get("guidance_scale") != WAN5_GUIDANCE_SCALE
        or sampling.get("sampler_runtime") != "diffusers/UniPCMultistepScheduler"
        or sampling.get("solver_bridge") != _WAN5_UNIPC_SOLVER_BRIDGE
        or sampling.get("terminal_vp_sigma") != _WAN5_UNIPC_TERMINAL_SIGMA
        or not isinstance(dispatch, Mapping)
        or dispatch.get("proven") is not True
        or dispatch.get("fallback_calls") != 0
        or dispatch.get("native_modules") != dispatch.get("expected_modules")
        or not isinstance(output, Mapping)
        or not isinstance(observed_fps, (int, float))
        or isinstance(observed_fps, bool)
        or not math.isclose(float(observed_fps), WAN5_FPS, abs_tol=0.01)
        or observed_frames != generation["num_frames"]
        or not isinstance(observed_duration, (int, float))
        or isinstance(observed_duration, bool)
        or not math.isclose(
            float(observed_duration),
            int(observed_frames) / float(observed_fps),
            abs_tol=0.001,
        )
        or not isinstance(output.get("duration"), int)
        or isinstance(output.get("duration"), bool)
        or output["duration"] <= 0
        or not isinstance(time_base, Mapping)
        or set(time_base) != {"numerator", "denominator"}
        or not isinstance(time_base.get("numerator"), int)
        or not isinstance(time_base.get("denominator"), int)
        or isinstance(time_base.get("numerator"), bool)
        or isinstance(time_base.get("denominator"), bool)
        or time_base["numerator"] <= 0
        or time_base["denominator"] <= 0
        or not math.isclose(
            float(observed_duration),
            output["duration"] * time_base["numerator"] / time_base["denominator"],
            rel_tol=0.0,
            abs_tol=_OBSERVED_DURATION_TOLERANCE_SECONDS,
        )
        or output.get("has_audio") is not False
    ):
        raise RuntimeError("Wan 5B worker metadata differs from its bound request")


def _worker_failure_value(result: Any, exit_code: int, binding: str) -> dict[str, Any]:
    """Validate one bounded child failure object without reopening its IPC file."""

    if result is None:
        return {"message": f"Wan 5B worker exited with code {exit_code}"}
    legacy_fields = {"schema_version", "ok", "request_binding", "error_type"}
    diagnostic_fields = legacy_fields | {
        "failure_stage",
        "error_fingerprint",
        "failure_location",
    }
    numerical_fields = diagnostic_fields | {
        "numerical_boundary",
        "denoise_step",
        "transformer_pass",
    }
    result_fields = frozenset(result) if isinstance(result, dict) else frozenset()
    if (
        isinstance(result, dict)
        and result.get("ok") is False
        and result.get("request_binding") == binding
        and isinstance(result.get("error_type"), str)
    ):
        if result_fields in {frozenset(diagnostic_fields), frozenset(numerical_fields)} and (
            _valid_failure_diagnostic(result)
            and (
                set(result) == diagnostic_fields or _valid_numerical_diagnostic(result)
            )
        ):
            failure: dict[str, Any] = {
                "message": (
                    f"Wan 5B worker failed ({result['error_type']} during "
                    f"{result['failure_stage']} at {result['failure_location']}; "
                    f"diagnostic {result['error_fingerprint'][:12]})"
                ),
                "error_type": result["error_type"],
                "stage": result["failure_stage"],
                "location": result["failure_location"],
                "fingerprint": result["error_fingerprint"],
            }
            if set(result) == numerical_fields:
                failure.update(
                    {
                        "boundary": result["numerical_boundary"],
                        "step": result["denoise_step"],
                        "transformer_pass": result["transformer_pass"],
                    }
                )
                failure["message"] = (
                    failure["message"][:-1]
                    + f"; boundary {result['numerical_boundary']} step {result['denoise_step']}"
                    + (
                        ""
                        if result["transformer_pass"] is None
                        else f" pass {result['transformer_pass']}"
                    )
                    + ")"
                )
            return failure
        if set(result) == legacy_fields:
            return {
                "message": f"Wan 5B worker failed ({result['error_type']})",
                "error_type": result["error_type"],
            }
    return {"message": f"Wan 5B worker exited with code {exit_code} and an invalid failure result"}


def _log_worker_failure(failure: Mapping[str, Any]) -> None:
    """Log only the closed diagnostic fields, never child exception text."""

    _LOGGER.error(
        "Wan 5B Kitchen child failure: type=%s stage=%s location=%s diagnostic=%s "
        "boundary=%s step=%s pass=%s",
        failure.get("error_type", "unknown"),
        failure.get("stage", "unknown"),
        failure.get("location", "unknown"),
        failure.get("fingerprint", "unknown"),
        failure.get("boundary", "unknown"),
        failure.get("step", "unknown"),
        failure.get("transformer_pass", "unknown"),
    )


def _valid_failure_diagnostic(value: Mapping[str, Any]) -> bool:
    stage = value.get("failure_stage")
    fingerprint = value.get("error_fingerprint")
    location = value.get("failure_location")
    return (
        isinstance(stage, str)
        and stage.replace("_", "").isalnum()
        and len(stage) <= 80
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
        and isinstance(location, str)
        and location.replace("_", "").replace(".", "").isalnum()
        and len(location) <= 160
    )


def _valid_numerical_diagnostic(value: Mapping[str, Any]) -> bool:
    return (
        value.get("numerical_boundary")
        in {
            "transformer_noise_prediction",
            "guided_noise_prediction",
            "scheduler_output_latents",
            "denoise_latents",
        }
        and isinstance(value.get("denoise_step"), int)
        and not isinstance(value.get("denoise_step"), bool)
        and 1 <= value["denoise_step"] <= WAN5_STEPS
        and value.get("transformer_pass") in {None, "conditional", "unconditional"}
    )


def _last_worker(
    state: DisposableWorkerRunState,
    *,
    outcome: str | None = None,
    allocator_policy: str | None,
) -> dict[str, object]:
    return {
        "pid": state.pid,
        "exit_code": state.exit_code,
        "terminated": state.terminated,
        "outcome": state.outcome if outcome is None else outcome,
        "tree_empty": state.tree_empty,
        "memory_boundary": "disposable_process_exit" if state.tree_empty else "unproven",
        "allocator_policy": allocator_policy,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
