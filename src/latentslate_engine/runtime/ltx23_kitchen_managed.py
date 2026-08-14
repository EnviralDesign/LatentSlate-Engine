"""Persistent isolated supervisor for the Engine-native LTX 2.3 Kitchen runtime.

The parent deliberately owns no tensors, pipelines, or Kitchen imports.  A
one request-bound worker process receives serial fully-bound generation
commands and remains inside a Windows kill-on-close job object until unload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..ltx23_kitchen_recipe import (
    LTX23KitchenRuntimeRequest,
    ltx23_kitchen_dimension_alignment,
    revalidate_ltx23_kitchen_runtime_request,
    validate_ltx23_kitchen_dimensions,
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
_AUDIO_DECODER_TAIL_SAMPLES = 1_920
_AAC_PACKET_SAMPLES = 1_024
_OPERATIONS = frozenset({"ltx23_dev_t2v", "ltx23_dev_i2v", "ltx23_distilled_flf"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManagedLTX23KitchenResult:
    """A result accepted from the exact active isolated worker session."""

    output_path: Path
    output_size_bytes: int
    metadata: dict[str, Any]
    worker_pid: int
    worker_exit_code: int | None


@dataclass(slots=True)
class _WorkerSession:
    process: subprocess.Popen[bytes]
    tree: DisposableProcessTree
    paths: dict[str, Path]
    allocator_policy: str


class ManagedLTX23KitchenRuntime:
    """Retain one isolated worker for compatible jobs of one exact recipe."""

    def __init__(self, request: LTX23KitchenRuntimeRequest) -> None:
        if request.operation not in _OPERATIONS:
            raise ValueError("managed LTX 2.3 Kitchen runtime requires a supported operation")
        self.request = request
        self._active_tree: DisposableProcessTree | None = None
        self._session: _WorkerSession | None = None
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []
        self._ownership = Lock()
        self._job_active = False

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
        """Generate one MP4, reusing a healthy isolated worker when compatible."""

        check_cancelled()
        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("managed LTX 2.3 Kitchen worker is already active")
        self._job_active = True
        session: _WorkerSession | None = None
        worker_failure: dict[str, str] | None = None
        try:
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
            progress(0.0, "Validating LTX runtime request")
            preflight_started = time.perf_counter()
            payload = _payload(self.request, generation, device=device)
            if not revalidate_ltx23_kitchen_runtime_request(self.request):
                raise RuntimeError(
                    "LTX 2.3 Kitchen request changed immediately before worker dispatch"
                )
            _LOGGER.info(
                "LTX 2.3 Kitchen preflight completed: operation=%s elapsed_seconds=%.3f",
                self.request.operation,
                time.perf_counter() - preflight_started,
            )
            session = self._session
            if session is None:
                paths = _paths(output_path)
                _require_fresh(paths)
                _write_json(paths["request"], payload)
                try:
                    session = self._spawn_session(paths)
                except BaseException:
                    self._cleanup_errors = _cleanup_session(paths)
                    raise
                self._session = session
                self._active_tree = session.tree
                paths["gate"].touch(exist_ok=False)
                progress(0.0, "Starting isolated LTX worker")
                _LOGGER.info(
                    "LTX 2.3 Kitchen worker spawned: operation=%s pid=%s",
                    self.request.operation,
                    session.process.pid,
                )
            else:
                _require_job_fresh(session.paths)
                progress(0.0, "Dispatching to warmed LTX worker")
                _write_json(session.paths["command"], payload)
            _wait_for_result(
                session.process,
                session.paths["progress"],
                session.paths["result"],
                progress,
                check_cancelled,
                self.request.operation,
            )
            if _result_is_failure(session.paths["result"]):
                worker_failure = _worker_failure(
                    session.paths["result"], session.process.poll() or 1, payload["request_binding"]
                )
                _log_worker_failure(worker_failure)
                raise RuntimeError(worker_failure["message"])
            if session.process.poll() is not None:
                raise RuntimeError("LTX 2.3 Kitchen worker exited before its success result was accepted")
            result = _read_success(session.paths["result"], output_path, payload["request_binding"])
            _validate_metadata(result["metadata"], self.request, generation)
            self._last_worker = _last_worker(
                session.process,
                "succeeded",
                tree_empty=False,
                allocator_policy=result["allocator_policy"],
            )
            self._last_worker["pipeline_warm"] = bool(
                result["metadata"].get("cache", {}).get("pipeline_warm")
            )
            self._cleanup_errors = _cleanup_job(session.paths)
            return ManagedLTX23KitchenResult(
                Path(output_path).resolve(strict=True),
                result["output_size_bytes"],
                result["metadata"],
                session.process.pid,
                None,
            )
        except BaseException as primary:
            if session is not None:
                try:
                    tree_empty = _terminate_tree(session.tree, session.process, primary)
                    self._last_worker = _last_worker(
                        session.process,
                        "canceled" if _is_cancellation(primary) else "failed",
                        tree_empty=tree_empty,
                        allocator_policy=session.allocator_policy,
                        worker_failure=worker_failure,
                    )
                finally:
                    self._session = None
                    self._active_tree = None
                    try:
                        _close_tree(session.tree)
                    finally:
                        self._cleanup_errors = _cleanup_session(session.paths)
            # Validation/spawn failures must never remove a pre-existing user
            # target. Once Popen succeeds the target was proved fresh above and
            # is owned by this disposable job, including partial worker output.
            if session is not None:
                _remove_output(output_path, primary)
            raise
        finally:
            self._job_active = False
            self._ownership.release()

    def _spawn_session(self, paths: dict[str, Path]) -> _WorkerSession:
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "latentslate_engine.runtime.ltx23_kitchen_worker",
                "--request", str(paths["request"]), "--result", str(paths["result"]),
                "--progress", str(paths["progress"]), "--start-gate", str(paths["gate"]),
                "--command", str(paths["command"]),
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=env,
        )
        try:
            tree = DisposableProcessTree(process)
        except BaseException:
            _terminate_direct_process(process)
            raise
        return _WorkerSession(process, tree, paths, env["PYTORCH_CUDA_ALLOC_CONF"])

    def status(self) -> dict[str, Any]:
        return {
            "family": "ltx23",
            "runtime": "engine-native/ltx23-kitchen-persistent-worker",
            "request_fingerprint": self.request.fingerprint,
            "component_fingerprint": self.request.component_fingerprint,
            "loaded": self._session is not None,
            "active_worker": self._job_active,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False},
            "cache": {
                "pipeline_warm": bool(
                    self._last_worker and self._last_worker.get("pipeline_warm", False)
                )
            },
        }

    def clear_cache(self) -> None:
        """LTX has no prompt/media cache; retain warmed model components."""

    def unload(self) -> None:
        session = self._session
        if session is None:
            return
        primary: BaseException | None = None
        try:
            session.tree.terminate()
            session.tree.wait_for_empty()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            except OSError as close_error:
                if primary is None:
                    self._cleanup_errors = [*self._cleanup_errors, "job_object_close"][-16:]
                    raise
                primary.add_note(f"LTX 2.3 Kitchen worker Job Object close failed: {close_error}")
            finally:
                self._cleanup_errors = _cleanup_session(session.paths)


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
    try:
        validate_ltx23_kitchen_dimensions(operation, width=width, height=height)
    except ValueError as exc:
        alignment = ltx23_kitchen_dimension_alignment(operation)
        raise ValueError(
            "LTX 2.3 Kitchen dimensions do not meet the operation alignment "
            f"(requires /{alignment}; received {width}x{height})"
        ) from exc
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
    # Session commands can contain user prompts and asset paths. Keep them in
    # an unpredictable owner-scoped temporary directory, never beside a public
    # output path with a guessable filename.
    root = Path(tempfile.mkdtemp(prefix="latentslate-ltx23-"))
    try:
        os.chmod(root, 0o700)
    except OSError:
        # Windows ACL inheritance remains the owner boundary where POSIX mode
        # bits are unavailable; the random directory capability still binds IPC.
        pass
    return {
        key: root / f"{key}{suffix}"
        for key, suffix in {
            "request": ".json",
            "result": ".json",
            "progress": ".jsonl",
            "gate": "",
            "command": ".json",
        }.items()
    }


def _result_is_failure(path: Path) -> bool:
    """Recognize a terminal child failure before relying on process scheduling."""

    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("ok") is False


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


def _require_job_fresh(paths: Mapping[str, Path]) -> None:
    stale = sorted(name for name in ("command", "result", "progress") if paths[name].exists())
    if stale:
        raise RuntimeError("LTX 2.3 Kitchen worker job IPC paths already exist: " + ", ".join(stale))


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
        > (_AAC_PACKET_SAMPLES / _AUDIO_SAMPLE_RATE)
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker media provenance is invalid")
    normalization = metadata.get("audio_duration_normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("LTX 2.3 Kitchen worker audio duration normalization is missing")
    normalization_keys = {
        "decoded_samples",
        "target_samples",
        "trimmed_samples",
        "trailing_silence_samples",
        "maximum_trailing_silence_samples",
    }
    if (
        set(normalization) != normalization_keys
        or any(not isinstance(normalization[key], int) for key in normalization_keys)
        or normalization["decoded_samples"] <= 0
        or normalization["target_samples"] != generation["num_frames"] * _AUDIO_SAMPLE_RATE // _FPS
        or normalization["trimmed_samples"] < 0
        or normalization["trailing_silence_samples"] < 0
        or normalization["maximum_trailing_silence_samples"] != _AUDIO_DECODER_TAIL_SAMPLES
        or normalization["trailing_silence_samples"] > _AUDIO_DECODER_TAIL_SAMPLES
        or normalization["decoded_samples"]
        - normalization["trimmed_samples"]
        + normalization["trailing_silence_samples"]
        != normalization["target_samples"]
        or (
            normalization["decoded_samples"] != normalization["target_samples"]
            and bool(normalization["trimmed_samples"])
            == bool(normalization["trailing_silence_samples"])
        )
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker audio duration normalization is invalid")
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
    residency = metadata.get("residency_policy")
    integer_fields = (
        "stored_bytes",
        "root_bytes",
        "resident_weight_budget_bytes",
        "resident_block_count",
        "resident_block_bytes",
        "streamed_block_count",
        "streamed_block_bytes",
        "stream_buffer_bytes",
        "stream_buffer_count",
        "streamed_transitions",
        "resident_refills",
    )
    if (
        not isinstance(residency, Mapping)
        or residency.get("mode") not in {"full", "grouped"}
        or residency.get("streaming") != "synchronous_cpu_master"
        or any(
            not isinstance(residency.get(key), int) or residency[key] < 0
            for key in integer_fields
        )
        or residency["stream_buffer_count"] != int(residency["streamed_block_count"] > 0)
        or residency["resident_block_count"] + residency["streamed_block_count"] != 48
        or residency["root_bytes"]
        + residency["resident_block_bytes"]
        + residency["streamed_block_bytes"]
        != residency["stored_bytes"]
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker residency provenance is invalid")


def _wait(
    process: subprocess.Popen[bytes],
    path: Path,
    progress: Callable[[float, str | None], None],
    cancelled: Callable[[], None],
    operation: str,
) -> None:
    offset = records = 0
    pending = b""
    previous = -1.0
    started_at = time.perf_counter()
    while process.poll() is None:
        cancelled()
        offset, pending, records, previous = _drain(
            path,
            offset,
            pending,
            records,
            previous,
            progress,
            operation=operation,
            started_at=started_at,
        )
        time.sleep(_POLL_SECONDS)
    _offset, pending, _records, _previous = _drain(
        path,
        offset,
        pending,
        records,
        previous,
        progress,
        operation=operation,
        started_at=started_at,
    )
    if pending and process.poll() == 0:
        raise RuntimeError("LTX 2.3 Kitchen worker ended with a truncated progress record")


def _wait_for_result(
    process: subprocess.Popen[bytes],
    progress_path: Path,
    result_path: Path,
    progress: Callable[[float, str | None], None],
    cancelled: Callable[[], None],
    operation: str,
) -> None:
    """Drain one session job until its atomically published response arrives."""

    offset = records = 0
    pending = b""
    previous = -1.0
    started_at = time.perf_counter()
    while not result_path.is_file() and process.poll() is None:
        cancelled()
        offset, pending, records, previous = _drain(
            progress_path, offset, pending, records, previous, progress,
            operation=operation, started_at=started_at,
        )
        time.sleep(_POLL_SECONDS)
    _drain(
        progress_path, offset, pending, records, previous, progress,
        operation=operation, started_at=started_at,
    )


def _drain(
    path: Path,
    offset: int,
    pending: bytes,
    records: int,
    previous: float,
    callback: Callable[[float, str | None], None],
    *,
    operation: str,
    started_at: float,
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
        _LOGGER.info(
            "LTX 2.3 Kitchen worker progress: operation=%s phase=%s progress=%.3f "
            "elapsed_seconds=%.3f",
            operation,
            _worker_progress_phase(item["message"]),
            previous,
            time.perf_counter() - started_at,
        )
    return offset, pending, records, previous


def _worker_progress_phase(message: str | None) -> str:
    """Map worker-owned IPC text to a log-safe phase name.

    The server log intentionally never repeats arbitrary worker IPC text, so
    an invalid or future message cannot disclose prompts or local paths.
    """

    if message is None:
        return "working"
    prefixes = (
        ("LTX worker started", "worker_started"),
        ("Rehydrating LTX recipe", "rehydrate_recipe"),
        ("Importing LTX runtime", "import_runtime"),
        ("Preparing LTX generation", "prepare_generation"),
        ("Materializing LTX transformer", "materialize_transformer"),
        ("Materializing LTX connectors", "materialize_connectors"),
        ("Materializing LTX video VAE", "materialize_video_vae"),
        ("Materializing LTX audio VAE", "materialize_audio_vae"),
        ("Materializing LTX vocoder", "materialize_vocoder"),
        ("Materializing LTX latent upsampler", "materialize_latent_upsampler"),
        ("Materializing LTX text encoder", "materialize_text_encoder"),
        ("Installing LTX model LoRA", "install_model_lora"),
        ("Installing LTX text LoRA", "install_text_lora"),
        ("Enhancing prompt", "enhance_prompt"),
        ("Encoding prompt", "encode_prompt"),
        ("LTX denoise step", "denoise"),
        ("Upscaling LTX video latents", "upscale_latents"),
        ("Decoding LTX video and audio", "decode_media"),
        ("Muxing 24 fps", "mux_output"),
        ("LTX 2.3 output ready", "verify_output"),
    )
    return next((phase for prefix, phase in prefixes if message.startswith(prefix)), "working")


def _worker_error(path: Path, exit_code: int, binding: str) -> str:
    """Return the public message for one already-sanitized worker failure."""

    return _worker_failure(path, exit_code, binding)["message"]


def _worker_failure(path: Path, exit_code: int, binding: str) -> dict[str, str]:
    """Read bounded worker diagnostics while never surfacing child exception text.

    The child persists its raw exception only in disposable IPC for local
    debugging. The Engine publishes the operation stage, internal function
    identifier, and a deterministic fingerprint instead, so prompts and local
    filesystem paths cannot leak through a provider error.
    """

    try:
        value = _read_json(path)
        legacy_fields = {"schema_version", "ok", "request_binding", "error_type", "error"}
        diagnostic_fields = legacy_fields | {
            "failure_stage",
            "error_fingerprint",
            "failure_location",
        }
        if (
            isinstance(value, dict)
            and (set(value) == legacy_fields or set(value) == diagnostic_fields)
            and value.get("schema_version") == _SCHEMA_VERSION
            and value.get("ok") is False
            and value.get("request_binding") == binding
            and isinstance(value.get("error_type"), str)
            and isinstance(value.get("error"), str)
        ):
            if set(value) == diagnostic_fields and _valid_failure_diagnostic(value):
                return {
                    "message": (
                        f"LTX 2.3 Kitchen worker failed ({value['error_type']} during "
                        f"{value['failure_stage']} at {value['failure_location']}; "
                        f"diagnostic {value['error_fingerprint'][:12]})"
                    ),
                    "error_type": value["error_type"],
                    "stage": value["failure_stage"],
                    "location": value["failure_location"],
                    "fingerprint": value["error_fingerprint"],
                    "log_detail": value["error"],
                }
            return {
                "message": f"LTX 2.3 Kitchen worker failed ({value['error_type']})",
                "error_type": value["error_type"],
                "log_detail": value["error"],
            }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        pass
    return {"message": f"LTX 2.3 Kitchen worker exited with code {exit_code}"}


def _log_worker_failure(failure: Mapping[str, str]) -> None:
    """Emit the bounded child detail to local server logs, not the public API.

    Disposable workers cannot rely on inherited stderr.  Their result protocol
    therefore carries a capped exception string that the supervisor logs before
    deleting IPC.  Public job errors retain the safer stage/location/fingerprint
    message, while operators still receive the actionable underlying exception.
    """

    detail = failure.get("log_detail")
    if not detail:
        return
    _LOGGER.error(
        "LTX 2.3 Kitchen child failure: type=%s stage=%s location=%s "
        "diagnostic=%s detail=%s",
        failure.get("error_type", "unknown"),
        failure.get("stage", "unknown"),
        failure.get("location", "unknown"),
        failure.get("fingerprint", "unknown"),
        detail,
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
    worker_failure: Mapping[str, str] | None = None,
) -> dict[str, object]:
    status: dict[str, object] = {
        "pid": process.pid,
        "exit_code": process.poll(),
        "terminated": process.poll() is not None,
        "outcome": outcome,
        "tree_empty": tree_empty,
        "memory_boundary": (
            "disposable_process_exit" if tree_empty else "persistent_process_residency"
        ),
        "allocator_policy": allocator_policy,
    }
    if worker_failure is not None:
        status["failure"] = {
            key: worker_failure[key]
            for key in ("error_type", "stage", "location", "fingerprint")
            if key in worker_failure
        }
    return status


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


def _cleanup_job(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for name in ("request", "result", "progress", "command"):
        try:
            paths[name].unlink(missing_ok=True)
        except OSError:
            errors.append("ipc")
    return sorted(set(errors))


def _cleanup_session(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for path in paths.values():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            errors.append("ipc")
    roots = {path.parent for path in paths.values()}
    for root in roots:
        try:
            root.rmdir()
        except OSError:
            # Test doubles can intentionally share a directory with unrelated
            # files; production session directories contain only this IPC.
            if root.name.startswith("latentslate-ltx23-"):
                errors.append("ipc")
    return sorted(set(errors))
