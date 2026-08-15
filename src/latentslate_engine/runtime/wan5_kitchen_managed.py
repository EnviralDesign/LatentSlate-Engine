"""Disposable parent supervisor for Engine-native Wan 2.2 TI2V 5B."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
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
from .windows_process import DisposableProcessTree

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
        self._active_tree: DisposableProcessTree | None = None
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
        process: subprocess.Popen[bytes] | None = None
        tree: DisposableProcessTree | None = None
        paths = _paths(output_path)
        allocator_policy: str | None = None
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
            _write_json(paths["request"], payload)
            if not revalidate_wan5_kitchen_runtime_request(self.request):
                raise RuntimeError("Wan 5B request changed immediately before worker spawn")
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            allocator_policy = env["PYTORCH_CUDA_ALLOC_CONF"]
            process = subprocess.Popen(
                [
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
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=env,
            )
            try:
                tree = DisposableProcessTree(process)
            except BaseException:
                _terminate_direct(process)
                self._last_worker = _last_worker(
                    process,
                    "failed",
                    tree_empty=process.poll() is not None,
                    allocator_policy=allocator_policy,
                )
                raise
            self._active_tree = tree
            paths["gate"].touch(exist_ok=False)
            _wait(process, paths["progress"], progress, check_cancelled)
            exit_code = process.wait(timeout=5)
            if exit_code != 0:
                failure = _worker_failure(paths["result"], exit_code, payload["request_binding"])
                _log_worker_failure(failure)
                raise RuntimeError(failure["message"])
            result = _read_success(paths["result"], output_path, payload["request_binding"])
            _validate_metadata(result["metadata"], self.request, generation)
            tree.wait_for_empty()
            tree.close()
            tree = None
            self._active_tree = None
            self._last_worker = _last_worker(
                process,
                "succeeded",
                tree_empty=True,
                allocator_policy=result["allocator_policy"],
            )
            return ManagedWan5KitchenResult(
                Path(output_path).resolve(strict=True),
                result["output_size_bytes"],
                result["metadata"],
                process.pid,
                exit_code,
            )
        except BaseException as primary:
            if tree is not None or process is not None:
                tree_empty = _terminate_tree(tree, process)
                if process is not None:
                    self._last_worker = _last_worker(
                        process,
                        "canceled" if _is_cancellation(primary) else "failed",
                        tree_empty=tree_empty,
                        allocator_policy=allocator_policy,
                    )
            if process is not None:
                Path(output_path).unlink(missing_ok=True)
            raise
        finally:
            self._active_tree = None
            if tree is not None:
                try:
                    tree.close()
                except BaseException:  # noqa: BLE001, S110 - preserve the primary result
                    pass
            self._cleanup_errors = _cleanup(paths)
            self._ownership.release()

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "runtime": "engine-native/wan5-kitchen-disposable-worker",
            "request_fingerprint": self.request.fingerprint,
            "component_fingerprint": self.request.component_fingerprint,
            "loaded": False,
            "active_worker": self._active_tree is not None,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False, "tensor": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        pass

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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _require_fresh(paths: Mapping[str, Path]) -> None:
    stale = sorted(name for name, path in paths.items() if path.exists())
    if stale:
        raise RuntimeError("Wan 5B worker IPC paths already exist: " + ", ".join(stale))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise RuntimeError("Wan 5B worker JSON is missing or exceeds its bound")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_success(path: Path, output: Path, binding: str) -> dict[str, Any]:
    result = _read_json(path)
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


def _wait(
    process: subprocess.Popen[bytes],
    progress_path: Path,
    progress: Callable[[float, str | None], None],
    check_cancelled: Callable[[], None],
) -> None:
    position = 0
    records = 0
    while process.poll() is None:
        check_cancelled()
        position, records = _drain_progress(progress_path, position, records, progress)
        time.sleep(_POLL_SECONDS)
    _drain_progress(progress_path, position, records, progress)


def _drain_progress(
    path: Path,
    position: int,
    records: int,
    progress: Callable[[float, str | None], None],
) -> tuple[int, int]:
    if not path.exists():
        return position, records
    if path.stat().st_size > _MAX_PROGRESS_BYTES:
        raise RuntimeError("Wan 5B worker progress exceeds its bound")
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(position)
        while line := stream.readline():
            records += 1
            if records > _MAX_PROGRESS_RECORDS:
                raise RuntimeError("Wan 5B worker progress record count exceeds its bound")
            value = json.loads(line)
            progress(float(value["progress"]), value.get("message"))
        return stream.tell(), records


def _worker_error(path: Path, exit_code: int, binding: str) -> str:
    """Return the public message for one already-sanitized worker failure."""

    return _worker_failure(path, exit_code, binding)["message"]


def _worker_failure(path: Path, exit_code: int, binding: str) -> dict[str, Any]:
    """Read the child's closed, privacy-safe diagnostic protocol."""

    try:
        result = _read_json(path)
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
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


def _terminate_tree(
    tree: DisposableProcessTree | None, process: subprocess.Popen[bytes] | None
) -> bool:
    if tree is not None:
        try:
            tree.terminate()
            tree.wait_for_empty()
            return True
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError):
            return False
    if process is not None:
        _terminate_direct(process)
        return process.poll() is not None
    return False


def _terminate_direct(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _last_worker(
    process: subprocess.Popen[bytes],
    outcome: str,
    *,
    tree_empty: bool,
    allocator_policy: str | None,
) -> dict[str, object]:
    return {
        "pid": process.pid,
        "exit_code": process.poll(),
        "terminated": process.poll() is not None,
        "outcome": outcome,
        "tree_empty": tree_empty,
        "memory_boundary": "disposable_process_exit" if tree_empty else "unproven",
        "allocator_policy": allocator_policy,
    }


def _cleanup(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for label, path in paths.items():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            errors.append(f"{label}_cleanup_failed")
    return errors


def _is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, asyncio.CancelledError) or any(
        cls.__name__ == "ToolCancelled" for cls in type(exc).__mro__
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
