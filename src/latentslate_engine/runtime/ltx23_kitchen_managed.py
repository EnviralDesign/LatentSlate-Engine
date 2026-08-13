"""Disposable supervisor for the Engine-native LTX 2.3 Kitchen runtime.

The parent deliberately owns no tensors, pipelines, or Kitchen imports.  A
fresh worker process receives one fully-bound JSON request, performs exactly
one generation, and exits inside a Windows kill-on-close job object.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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

from ..ltx23_kitchen_recipe import (
    LTX23KitchenRuntimeRequest,
    revalidate_ltx23_kitchen_runtime_request,
)
from .windows_process import DisposableProcessTree

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_FPS = 24
_AUDIO_SAMPLE_RATE = 48_000
_AUDIO_CHANNELS = 2
_OPERATIONS = frozenset({"ltx23_dev_t2v", "ltx23_dev_i2v", "ltx23_distilled_flf"})


@dataclass(frozen=True, slots=True)
class ManagedLTX23KitchenResult:
    """A result accepted only after the complete disposable worker tree exited."""

    output_path: Path
    output_size_bytes: int
    metadata: dict[str, Any]
    worker_pid: int
    worker_exit_code: int


class ManagedLTX23KitchenRuntime:
    """Run one fully-bound Kitchen request in a fresh disposable worker per job."""

    def __init__(self, request: LTX23KitchenRuntimeRequest) -> None:
        if request.operation not in _OPERATIONS:
            raise ValueError("managed LTX 2.3 Kitchen runtime requires a supported operation")
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
        duration_seconds: float,
        seed: int,
        start_image_path: Path | None = None,
        end_image_path: Path | None = None,
        device: str = "cuda",
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> ManagedLTX23KitchenResult:
        """Generate one MP4 without retaining any worker or tensor cache in the parent."""

        check_cancelled()
        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("managed LTX 2.3 Kitchen worker is already active")
        process: subprocess.Popen[bytes] | None = None
        tree: DisposableProcessTree | None = None
        paths = _paths(output_path)
        allocator_policy: str | None = None
        try:
            if self._active_tree is not None:
                raise RuntimeError("managed LTX 2.3 Kitchen worker is already active")
            num_frames = frames_for_duration(duration_seconds)
            generation = _generation(
                self.request.operation,
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                num_frames=num_frames,
                seed=seed,
                start_image_path=start_image_path,
                end_image_path=end_image_path,
            )
            _validate_generation(self.request.operation, generation)
            if device != "cuda":
                raise ValueError("LTX 2.3 Kitchen worker requires direct CUDA execution")
            _require_fresh(paths)
            payload = _payload(self.request, generation, device=device)
            _write_json(paths["request"], payload)
            # This is deliberately the final identity/path revalidation before
            # Popen.  A changed artifact never reaches a worker import.
            if not revalidate_ltx23_kitchen_runtime_request(self.request):
                raise RuntimeError(
                    "LTX 2.3 Kitchen request changed immediately before worker spawn"
                )
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            allocator_policy = env["PYTORCH_CUDA_ALLOC_CONF"]
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "latentslate_engine.runtime.ltx23_kitchen_worker",
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
                _terminate_direct_process(process)
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
                raise RuntimeError(
                    _worker_error(paths["result"], exit_code, payload["request_binding"])
                )
            result = _read_success(paths["result"], output_path, payload["request_binding"])
            _validate_metadata(result["metadata"], self.request, generation)
            tree.wait_for_empty()
            tree.close()
            tree = None
            self._active_tree = None
            self._last_worker = _last_worker(
                process, "succeeded", tree_empty=True, allocator_policy=result["allocator_policy"]
            )
            return ManagedLTX23KitchenResult(
                Path(output_path).resolve(strict=True),
                result["output_size_bytes"],
                result["metadata"],
                process.pid,
                exit_code,
            )
        except BaseException as primary:
            if tree is not None or process is not None:
                tree_empty = _terminate_tree(tree, process, primary)
                if process is not None:
                    self._last_worker = _last_worker(
                        process,
                        "canceled" if _is_cancellation(primary) else "failed",
                        tree_empty=tree_empty,
                        allocator_policy=allocator_policy,
                    )
            # Validation/spawn failures must never remove a pre-existing user
            # target. Once Popen succeeds the target was proved fresh above and
            # is owned by this disposable job, including partial worker output.
            if process is not None:
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
            "runtime": "engine-native/ltx23-kitchen-disposable-worker",
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
        """There is no parent-side tensor or media cache to clear."""

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


def frames_for_duration(duration_seconds: float) -> int:
    """Derive the exact 24 fps ``8k+1`` LTX frame count from a requested duration."""

    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise TypeError("LTX 2.3 duration must be numeric")
    if (
        not math.isfinite(float(duration_seconds))
        or duration_seconds < 1.0
        or duration_seconds > 10.0
    ):
        raise ValueError("LTX 2.3 duration must be finite and within 1..10 seconds")
    requested = math.ceil(float(duration_seconds) * _FPS)
    frames = math.ceil((requested - 1) / 8) * 8 + 1
    if not 25 <= frames <= 241 or frames % 8 != 1:
        raise AssertionError("LTX 2.3 temporal alignment contract was violated")
    return frames


def _generation(operation: str, **value: Any) -> dict[str, object]:
    start = _endpoint(value["start_image_path"])
    end = _endpoint(value["end_image_path"])
    return {
        "prompt": value["prompt"],
        "width": value["width"],
        "height": value["height"],
        "duration_seconds": value["duration_seconds"],
        "num_frames": value["num_frames"],
        "seed": value["seed"],
        "start_image_path": None if start is None else start["path"],
        "end_image_path": None if end is None else end["path"],
        "start_image_identity": None if start is None else start["identity"],
        "end_image_identity": None if end is None else end["identity"],
        "output_path": str(Path(value["output_path"]).resolve(strict=False)),
    }


def _validate_generation(operation: str, value: Mapping[str, object]) -> None:
    if (
        operation not in _OPERATIONS
        or set(value)
        != {
            "prompt",
            "width",
            "height",
            "duration_seconds",
            "num_frames",
            "seed",
            "start_image_path",
            "end_image_path",
            "start_image_identity",
            "end_image_identity",
            "output_path",
        }
        or not isinstance(value["prompt"], str)
        or not value["prompt"].strip()
    ):
        raise ValueError("LTX 2.3 Kitchen generation is invalid")
    width, height, seed, frames = (
        value["width"],
        value["height"],
        value["seed"],
        value["num_frames"],
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (width, height, seed, frames)
    ):
        raise TypeError("LTX 2.3 Kitchen generation integer fields are invalid")
    divisor = 64 if operation.startswith("ltx23_dev_") else 32
    if width <= 0 or height <= 0 or width % divisor or height % divisor:
        raise ValueError("LTX 2.3 Kitchen dimensions do not meet the operation alignment")
    if frames != frames_for_duration(value["duration_seconds"]):
        raise ValueError("LTX 2.3 Kitchen frame count does not bind its duration")
    start, end = value["start_image_path"], value["end_image_path"]
    start_identity, end_identity = value["start_image_identity"], value["end_image_identity"]
    expected = {
        "ltx23_dev_t2v": (None, None),
        "ltx23_dev_i2v": (str, None),
        "ltx23_distilled_flf": (str, str),
    }[operation]
    for actual, identity, expected_type, label in (
        (start, start_identity, expected[0], "start"),
        (end, end_identity, expected[1], "end"),
    ):
        if expected_type is None:
            if actual is not None or identity is not None:
                raise ValueError(f"LTX 2.3 Kitchen {operation} must not receive a {label} image")
        elif (
            not isinstance(actual, expected_type)
            or not Path(actual).is_file()
            or not isinstance(identity, Mapping)
            or _endpoint_identity(Path(actual)) != dict(identity)
        ):
            raise ValueError(
                f"LTX 2.3 Kitchen {operation} requires an identity-bound {label} image"
            )
    output = value["output_path"]
    if (
        not isinstance(output, str)
        or Path(output).suffix.lower() != ".mp4"
        or Path(output).exists()
    ):
        raise ValueError("LTX 2.3 Kitchen output must be a fresh MP4 path")


def _payload(
    request: LTX23KitchenRuntimeRequest, generation: Mapping[str, object], *, device: str
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "request": request.to_json_dict(),
        "generation": dict(generation),
        "device": device,
    }
    return {**unsigned, "request_binding": _fingerprint(unsigned)}


def _paths(output_path: Path) -> dict[str, Path]:
    output = Path(output_path).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    return {
        key: output.parent / f".{output.stem}.ltx23-kitchen-worker-{key}{suffix}"
        for key, suffix in {
            "request": ".json",
            "result": ".json",
            "progress": ".jsonl",
            "gate": "",
        }.items()
    }


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _endpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("LTX 2.3 guide is not a file")
    return {"path": str(candidate), "identity": _endpoint_identity(candidate)}


def _endpoint_identity(path: Path) -> dict[str, int | str]:
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("LTX 2.3 guide changed while its identity was measured")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fresh(paths: Mapping[str, Path]) -> None:
    stale = sorted(name for name, path in paths.items() if path.exists())
    if stale:
        raise RuntimeError("LTX 2.3 Kitchen worker IPC paths already exist: " + ", ".join(stale))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise RuntimeError("LTX 2.3 Kitchen worker JSON is missing or exceeds its bound")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_success(path: Path, output: Path, binding: str) -> dict[str, Any]:
    result = _read_json(path)
    expected = {
        "schema_version",
        "ok",
        "request_binding",
        "output_path",
        "output_size_bytes",
        "metadata",
        "allocator_policy",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected
        or result.get("schema_version") != _SCHEMA_VERSION
        or result.get("ok") is not True
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker returned an invalid success result")
    if result["request_binding"] != binding:
        raise RuntimeError("LTX 2.3 Kitchen worker result does not bind to this request")
    expected_output = Path(output).resolve(strict=True)
    if Path(result["output_path"]).resolve(strict=True) != expected_output:
        raise RuntimeError("LTX 2.3 Kitchen worker published an unexpected output path")
    if (
        not isinstance(result["output_size_bytes"], int)
        or result["output_size_bytes"] != expected_output.stat().st_size
        or result["output_size_bytes"] <= 0
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker output size is invalid")
    if (
        not isinstance(result["metadata"], dict)
        or not isinstance(result["allocator_policy"], str)
        or not result["allocator_policy"]
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker result metadata is invalid")
    if result["metadata"].get("output_sha256") != _sha256_file(expected_output):
        raise RuntimeError("LTX 2.3 Kitchen worker output hash is invalid")
    return result


def _validate_metadata(
    metadata: Mapping[str, object],
    request: LTX23KitchenRuntimeRequest,
    generation: Mapping[str, object],
) -> None:
    exact = {
        "family": "ltx23",
        "runtime": "engine-native/ltx23-kitchen",
        "operation": request.operation,
        "request_fingerprint": request.fingerprint,
        "component_fingerprint": request.component_fingerprint,
        "seed": generation["seed"],
        "width": generation["width"],
        "height": generation["height"],
        "num_frames": generation["num_frames"],
        "fps": _FPS,
        "audio_sample_rate": _AUDIO_SAMPLE_RATE,
        "audio_channels": _AUDIO_CHANNELS,
    }
    if (
        any(metadata.get(key) != value for key, value in exact.items())
        or metadata.get("components") != request.public_component_manifest()
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker metadata is not bound to the complete request")
    observed = {
        "container_format": str,
        "video_codec": str,
        "audio_codec": str,
        "audio_samples": int,
        "video_duration_seconds": (int, float),
        "audio_duration_seconds": (int, float),
        "output_sha256": str,
    }
    if any(not isinstance(metadata.get(key), kind) for key, kind in observed.items()):
        raise RuntimeError("LTX 2.3 Kitchen worker media provenance is incomplete")
    if (
        "mp4" not in str(metadata["container_format"]).split(",")
        or metadata["video_codec"] != "h264"
        or metadata["audio_codec"] != "aac"
        or metadata["audio_samples"] <= 0
        or abs(
            float(metadata["video_duration_seconds"]) - float(metadata["audio_duration_seconds"])
        )
        > (1 / _FPS + 1024 / _AUDIO_SAMPLE_RATE)
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker media provenance is invalid")
    proof = metadata.get("native_fp8")
    if (
        not isinstance(proof, Mapping)
        or proof.get("complete") is not True
        or not isinstance(proof.get("modules"), int)
        or proof["modules"] <= 0
        or proof.get("dispatched_modules") != proof["modules"]
        or proof.get("dense_fallback_count") != 0
        or not isinstance(proof.get("native_dispatch_count"), int)
        or proof["native_dispatch_count"] <= 0
        or metadata.get("dense_base_dequantizations") != 0
    ):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker did not prove direct native dispatch without dense fallback"
        )
    text_proof = metadata.get("native_text")
    if (
        not isinstance(text_proof, Mapping)
        or not isinstance(text_proof.get("backend"), str)
        or not isinstance(text_proof.get("module_count"), int)
        or text_proof["module_count"] <= 0
        or not isinstance(text_proof.get("total_dispatches"), int)
        or text_proof["total_dispatches"] <= 0
        or not isinstance(text_proof.get("minimum_module_dispatches"), int)
        or text_proof["minimum_module_dispatches"] <= 0
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker did not prove complete native text dispatch")


def _wait(
    process: subprocess.Popen[bytes],
    path: Path,
    progress: Callable[[float, str | None], None],
    cancelled: Callable[[], None],
) -> None:
    offset = records = 0
    pending = b""
    previous = -1.0
    while process.poll() is None:
        cancelled()
        offset, pending, records, previous = _drain(
            path, offset, pending, records, previous, progress
        )
        time.sleep(_POLL_SECONDS)
    _offset, pending, _records, _previous = _drain(
        path, offset, pending, records, previous, progress
    )
    if pending and process.poll() == 0:
        raise RuntimeError("LTX 2.3 Kitchen worker ended with a truncated progress record")


def _drain(
    path: Path,
    offset: int,
    pending: bytes,
    records: int,
    previous: float,
    callback: Callable[[float, str | None], None],
) -> tuple[int, bytes, int, float]:
    if not path.is_file():
        return offset, pending, records, previous
    if path.stat().st_size > _MAX_PROGRESS_BYTES:
        raise RuntimeError("LTX 2.3 Kitchen worker progress exceeds its bound")
    with path.open("rb") as stream:
        stream.seek(offset)
        chunk = stream.read()
        offset = stream.tell()
    lines = (pending + chunk).split(b"\n")
    pending = lines.pop()
    for raw in lines:
        if not raw:
            continue
        records += 1
        if records > _MAX_PROGRESS_RECORDS or len(raw) > 4096:
            raise RuntimeError("LTX 2.3 Kitchen worker progress exceeds its bound")
        item = json.loads(raw)
        if (
            not isinstance(item, dict)
            or set(item) != {"progress", "message"}
            or isinstance(item.get("progress"), bool)
            or not isinstance(item.get("progress"), (int, float))
            or not 0.0 <= float(item["progress"]) <= 1.0
            or float(item["progress"]) < previous
            or not isinstance(item.get("message"), (str, type(None)))
        ):
            raise RuntimeError("LTX 2.3 Kitchen worker progress record is invalid")
        previous = float(item["progress"])
        callback(previous, item["message"])
    return offset, pending, records, previous


def _worker_error(path: Path, exit_code: int, binding: str) -> str:
    try:
        value = _read_json(path)
        if (
            isinstance(value, dict)
            and set(value) == {"schema_version", "ok", "request_binding", "error_type", "error"}
            and value.get("schema_version") == _SCHEMA_VERSION
            and value.get("ok") is False
            and value.get("request_binding") == binding
            and isinstance(value.get("error_type"), str)
            and isinstance(value.get("error"), str)
        ):
            return f"LTX 2.3 Kitchen worker failed ({value['error_type']})"
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        pass
    return f"LTX 2.3 Kitchen worker exited with code {exit_code}"


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
        return process is None or process.poll() is not None
    except BaseException as exc:
        exc.add_note(
            f"while handling LTX 2.3 Kitchen worker failure: {type(primary).__name__}: {primary}"
        )
        raise


def _terminate_direct_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if process.poll() is None:
        raise RuntimeError("LTX 2.3 Kitchen worker direct-process termination was not confirmed")


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
        "memory_boundary": "disposable_process_exit",
        "allocator_policy": allocator_policy,
    }


def _is_cancellation(exc: BaseException) -> bool:
    """Classify the protocol cancellation without coupling runtime to tools.base.

    ``ToolContext.check_cancelled`` raises ``tools.base.ToolCancelled``.  The
    runtime must not import that layer (tools depend on runtimes), so retain the
    narrow public exception identity by name alongside asyncio cancellation.
    """

    return isinstance(exc, asyncio.CancelledError) or any(
        item.__name__ == "ToolCancelled" for item in type(exc).__mro__
    )


def _remove_output(path: Path, primary: BaseException) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"LTX 2.3 Kitchen partial output cleanup failed: {exc}")


def _close_tree(tree: DisposableProcessTree) -> None:
    primary = sys.exc_info()[1]
    try:
        tree.close()
    except OSError as exc:
        if primary is None:
            raise
        primary.add_note(f"LTX 2.3 Kitchen worker Job Object close failed: {exc}")


def _cleanup(paths: Mapping[str, Path], output: Path) -> list[str]:
    errors: list[str] = []
    for path in paths.values():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            errors.append("ipc")
    parent = Path(output).parent
    prefix = f".{Path(output).name}."
    for temporary in parent.glob(f"{prefix}*.tmp.mp4"):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            errors.append("staging")
    return sorted(set(errors))
