"""Persistent isolated supervisor for the Engine-native LTX 2.3 Kitchen runtime.

The parent deliberately owns no tensors, pipelines, or Kitchen imports.  A
one request-bound worker process receives serial fully-bound generation
commands and remains inside a Windows kill-on-close job object until unload.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import secrets
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
from .framework.worker import (
    PersistentWatchdogPolicy,
    PersistentWorkerExited,
    PersistentWorkerPaths,
    PersistentWorkerStreamError,
    PersistentWorkerSupervisor,
    PersistentWorkerTimeout,
    WorkerJsonFileError,
    canonical_json,
    hmac_sha256,
    is_worker_cancellation,
    read_bounded_json,
    result_hmac_sha256,
)

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_HARD_TIMEOUT_SECONDS = 60 * 60
_STAGE_TIMEOUT_SECONDS = 20 * 60
_HEARTBEAT_TIMEOUT_SECONDS = 45
_CANCEL_GRACE_SECONDS = 5
_FPS = 24
_AUDIO_SAMPLE_RATE = 48_000
_AUDIO_CHANNELS = 2
_AUDIO_SOURCE_SAMPLE_RATE = 16_000
_AUDIO_MEL_HOP_LENGTH = 160
_AUDIO_TEMPORAL_COMPRESSION_RATIO = 4
_AUDIO_DURATION_POLICY = "source_derived_exact_duration_v1"
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
    supervisor: PersistentWorkerSupervisor
    allocator_policy: str
    secret: bytes


class ManagedLTX23KitchenRuntime:
    """Retain one isolated worker for compatible jobs of one exact recipe."""

    def __init__(self, request: LTX23KitchenRuntimeRequest) -> None:
        if request.operation not in _OPERATIONS:
            raise ValueError("managed LTX 2.3 Kitchen runtime requires a supported operation")
        self.request = request
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
            session = self._session
            secret = session.secret if session is not None else secrets.token_bytes(32)
            payload = _payload(self.request, generation, device=device, secret=secret)
            if not revalidate_ltx23_kitchen_runtime_request(self.request):
                raise RuntimeError(
                    "LTX 2.3 Kitchen request changed immediately before worker dispatch"
                )
            _LOGGER.info(
                "LTX 2.3 Kitchen preflight completed: operation=%s elapsed_seconds=%.3f",
                self.request.operation,
                time.perf_counter() - preflight_started,
            )
            if session is None:
                paths = _paths(output_path)
                _require_fresh(paths)
                supervisor, allocator_policy = _supervisor(
                    _persistent_paths(paths), secret
                )
                try:
                    supervisor.start(payload)
                except BaseException:
                    failed = supervisor.failed_start
                    self._cleanup_errors = (
                        list(failed.cleanup_errors)
                        if failed is not None
                        else supervisor.cleanup_session()
                    )
                    raise
                session = _WorkerSession(
                    supervisor,
                    allocator_policy,
                    secret,
                )
                self._session = session
                progress(0.0, "Starting isolated LTX worker")
                _LOGGER.info(
                    "LTX 2.3 Kitchen worker spawned: operation=%s pid=%s",
                    self.request.operation,
                    _process(session).pid,
                )
            else:
                _require_job_fresh(_path_mapping(session.supervisor.paths))
                progress(0.0, "Dispatching to warmed LTX worker")
                session.supervisor.send(payload)
            _wait_for_result(session.supervisor, progress, check_cancelled, self.request.operation)
            paths = _path_mapping(session.supervisor.paths)
            result = _read_result(
                paths["result"], output_path, payload["request_binding"], secret
            )
            if result["ok"] is False:
                worker_failure = _worker_failure_result(result)
                _log_worker_failure(worker_failure)
                raise RuntimeError(worker_failure["message"])
            process = _process(session)
            if process.poll() is not None:
                raise RuntimeError("LTX 2.3 Kitchen worker exited before its success result was accepted")
            _validate_metadata(result["metadata"], self.request, generation)
            self._last_worker = _last_worker(
                process,
                "succeeded",
                tree_empty=False,
                allocator_policy=result["allocator_policy"],
            )
            self._last_worker["pipeline_warm"] = bool(
                result["metadata"].get("cache", {}).get("pipeline_warm")
            )
            self._cleanup_errors = session.supervisor.cleanup_job()
            return ManagedLTX23KitchenResult(
                Path(output_path).resolve(strict=True),
                result["output_size_bytes"],
                result["metadata"],
                process.pid,
                None,
            )
        except BaseException as primary:
            if session is not None:
                try:
                    tree_empty = _terminate_supervisor(session.supervisor, primary)
                    self._last_worker = _last_worker(
                        _process(session),
                        "canceled" if is_worker_cancellation(primary) else "failed",
                        tree_empty=tree_empty,
                        allocator_policy=session.allocator_policy,
                        worker_failure=worker_failure,
                    )
                finally:
                    self._session = None
                    try:
                        session.supervisor.close()
                    finally:
                        self._cleanup_errors = session.supervisor.cleanup_session()
            # Validation/spawn failures must never remove a pre-existing user
            # target. Once Popen succeeds the target was proved fresh above and
            # is owned by this disposable job, including partial worker output.
            if session is not None:
                _remove_output(output_path, primary)
            raise
        finally:
            self._job_active = False
            self._ownership.release()

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
            session.supervisor.terminate()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._session = None
            try:
                session.supervisor.close()
            except OSError as close_error:
                if primary is None:
                    self._cleanup_errors = [*self._cleanup_errors, "job_object_close"][-16:]
                    raise
                primary.add_note(f"LTX 2.3 Kitchen worker Job Object close failed: {close_error}")
            finally:
                self._cleanup_errors = session.supervisor.cleanup_session()


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
    request: LTX23KitchenRuntimeRequest,
    generation: Mapping[str, object],
    *,
    device: str,
    secret: bytes,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "request": request.to_json_dict(),
        "generation": dict(generation),
        "device": device,
    }
    return {**unsigned, "request_binding": _binding(unsigned, secret)}


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
            "heartbeat": ".jsonl",
            "gate": "",
            "command": ".json",
            "cancel": "",
        }.items()
    }


def _persistent_paths(paths: Mapping[str, Path]) -> PersistentWorkerPaths:
    return PersistentWorkerPaths(
        request=paths["request"],
        result=paths["result"],
        progress=paths["progress"],
        heartbeat=paths["heartbeat"],
        start_gate=paths["gate"],
        command=paths["command"],
        cancel=paths["cancel"],
    )


def _path_mapping(paths: PersistentWorkerPaths) -> dict[str, Path]:
    return {
        "request": paths.request,
        "result": paths.result,
        "progress": paths.progress,
        "heartbeat": paths.heartbeat,
        "gate": paths.start_gate,
        "command": paths.command,
        "cancel": paths.cancel,
    }


def _supervisor(
    paths: PersistentWorkerPaths, secret: bytes
) -> tuple[PersistentWorkerSupervisor, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["LATENTSLATE_LTX23_IPC_SECRET"] = secret.hex()
    return (
        PersistentWorkerSupervisor(
            command=(
                sys.executable,
                "-m",
                "latentslate_engine.runtime.ltx23_kitchen_worker",
                "--request",
                str(paths.request),
                "--result",
                str(paths.result),
                "--progress",
                str(paths.progress),
                "--heartbeat",
                str(paths.heartbeat),
                "--start-gate",
                str(paths.start_gate),
                "--command",
                str(paths.command),
                "--cancel",
                str(paths.cancel),
            ),
            paths=paths,
            environment=env,
        ),
        env["PYTORCH_CUDA_ALLOC_CONF"],
    )


def _process(session: _WorkerSession) -> Any:
    active = session.supervisor.session
    if active is None:
        raise RuntimeError("LTX 2.3 Kitchen worker session is closed")
    return active.process


def _binding(value: Mapping[str, object], secret: bytes) -> str:
    return hmac_sha256(value, secret)


def _result_binding(value: Mapping[str, object], secret: bytes) -> str:
    return result_hmac_sha256(value, secret)


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return canonical_json(value)


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


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonFileError:
        raise RuntimeError("LTX 2.3 Kitchen worker JSON is missing or exceeds its bound")


def _read_result(path: Path, output: Path, binding: str, secret: bytes) -> dict[str, Any]:
    try:
        result = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker result does not bind to its command"
        ) from None
    if not isinstance(result, dict):
        raise RuntimeError(  # noqa: TRY004 - uniform authenticated protocol failure.
            "LTX 2.3 Kitchen worker result does not bind to its command"
        )
    result_binding = result.get("result_binding")
    try:
        expected_result_binding = _result_binding(result, secret)
    except (TypeError, ValueError):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker result does not bind to its command"
        ) from None
    if not isinstance(result_binding, str) or not hmac.compare_digest(
        result_binding, expected_result_binding
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker result does not bind to its command")
    if (
        result.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(result.get("request_binding"), str)
        or not hmac.compare_digest(result["request_binding"], binding)
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker result does not bind to its command")
    failure_fields = {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "error_type",
        "error",
        "failure_stage",
        "error_fingerprint",
        "failure_location",
    }
    if result.get("ok") is False:
        if (
            set(result) != failure_fields
            or not isinstance(result.get("error_type"), str)
            or not result["error_type"].replace("_", "").isalnum()
            or len(result["error_type"]) > 80
            or not isinstance(result.get("error"), str)
            or len(result["error"]) > 4096
            or not _valid_failure_diagnostic(result)
        ):
            raise RuntimeError("LTX 2.3 Kitchen worker failure result is invalid")
        return result
    success_fields = {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "output_path",
        "output_size_bytes",
        "metadata",
        "allocator_policy",
    }
    if (
        set(result) != success_fields
        or result.get("ok") is not True
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker returned an invalid success result")
    try:
        expected_output = Path(output).resolve(strict=True)
        result_output = Path(result["output_path"]).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("LTX 2.3 Kitchen worker returned an invalid success result") from exc
    if result_output != expected_output:
        raise RuntimeError("LTX 2.3 Kitchen worker published an unexpected output path")
    if (
        isinstance(result["output_size_bytes"], bool)
        or not isinstance(result["output_size_bytes"], int)
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
    if any(
        isinstance(metadata.get(key), bool) or not isinstance(metadata.get(key), kind)
        for key, kind in observed.items()
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker media provenance is incomplete")
    video_duration = metadata["num_frames"] / metadata["fps"]
    audio_duration = metadata["audio_samples"] / metadata["audio_sample_rate"]
    target_audio_samples = metadata["num_frames"] * metadata["audio_sample_rate"] // metadata["fps"]
    if (
        "mp4" not in str(metadata["container_format"]).split(",")
        or metadata["video_codec"] != "h264"
        or metadata["audio_codec"] != "aac"
        or metadata["audio_samples"] <= 0
        or not math.isfinite(float(metadata["video_duration_seconds"]))
        or not math.isfinite(float(metadata["audio_duration_seconds"]))
        or metadata["video_duration_seconds"] != video_duration
        or metadata["audio_duration_seconds"] != audio_duration
        or not target_audio_samples
        <= metadata["audio_samples"]
        < target_audio_samples + _AAC_PACKET_SAMPLES
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker media provenance is invalid")
    normalization = metadata.get("audio_duration_normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("LTX 2.3 Kitchen worker audio duration normalization is missing")
    normalization_keys = {
        "policy",
        "reason",
        "video_frames",
        "audio_latent_frames",
        "expected_audio_latent_frames",
        "audio_latent_channels",
        "audio_latent_mel_bins",
        "decoded_mel_frames",
        "expected_mel_frames",
        "decoded_mel_channels",
        "decoded_mel_bins",
        "decoded_samples",
        "expected_decoded_samples",
        "target_samples",
        "fps",
        "audio_channels",
        "source_sample_rate",
        "output_sample_rate",
        "mel_hop_length",
        "temporal_compression_ratio",
        "causality_axis",
        "is_causal",
        "trimmed_samples",
        "padded_samples",
    }
    integer_keys = normalization_keys - {"policy", "reason", "causality_axis", "is_causal"}
    if (
        set(normalization) != normalization_keys
        or normalization["policy"] != _AUDIO_DURATION_POLICY
        or normalization["reason"] != "independent_audio_grid_causal_tail"
        or normalization["causality_axis"] != "height"
        or normalization["is_causal"] is not True
        or any(
            isinstance(normalization[key], bool) or not isinstance(normalization[key], int)
            for key in integer_keys
        )
        or normalization["source_sample_rate"] != _AUDIO_SOURCE_SAMPLE_RATE
        or normalization["output_sample_rate"] != _AUDIO_SAMPLE_RATE
        or normalization["output_sample_rate"] != metadata["audio_sample_rate"]
        or normalization["mel_hop_length"] != _AUDIO_MEL_HOP_LENGTH
        or normalization["temporal_compression_ratio"]
        != _AUDIO_TEMPORAL_COMPRESSION_RATIO
        or normalization["fps"] != _FPS
        or normalization["fps"] != metadata["fps"]
        or normalization["audio_latent_frames"] <= 0
        or normalization["expected_audio_latent_frames"]
        != round((normalization["video_frames"] / normalization["fps"]) * 25)
        or normalization["audio_latent_frames"]
        != normalization["expected_audio_latent_frames"]
        or normalization["audio_latent_channels"] != 8
        or normalization["audio_latent_mel_bins"] != 16
        or normalization["decoded_mel_frames"] <= 0
        or normalization["expected_mel_frames"]
        != normalization["audio_latent_frames"] * normalization["temporal_compression_ratio"]
        - (normalization["temporal_compression_ratio"] - 1)
        or normalization["decoded_mel_frames"] != normalization["expected_mel_frames"]
        or normalization["decoded_mel_channels"] != 2
        or normalization["decoded_mel_bins"] != 64
        or normalization["decoded_samples"] <= 0
        or normalization["expected_decoded_samples"]
        != normalization["expected_mel_frames"]
        * normalization["mel_hop_length"]
        * normalization["output_sample_rate"]
        // normalization["source_sample_rate"]
        or normalization["decoded_samples"] != normalization["expected_decoded_samples"]
        or normalization["video_frames"] != generation["num_frames"]
        or normalization["video_frames"] != metadata["num_frames"]
        or normalization["audio_channels"] != _AUDIO_CHANNELS
        or normalization["audio_channels"] != metadata["audio_channels"]
        or normalization["target_samples"]
        != normalization["video_frames"]
        * normalization["output_sample_rate"]
        // normalization["fps"]
        or normalization["trimmed_samples"] < 0
        or normalization["padded_samples"] < 0
        or normalization["decoded_samples"]
        - normalization["trimmed_samples"]
        + normalization["padded_samples"]
        != normalization["target_samples"]
        or (
            normalization["trimmed_samples"]
            != max(0, normalization["decoded_samples"] - normalization["target_samples"])
        )
        or (
            normalization["padded_samples"]
            != max(0, normalization["target_samples"] - normalization["decoded_samples"])
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


def _wait_for_result(
    supervisor: PersistentWorkerSupervisor,
    progress: Callable[[float, str | None], None],
    cancelled: Callable[[], None],
    operation: str,
) -> None:
    previous = [-1.0]
    started_at = time.perf_counter()
    try:
        supervisor.wait(
            progress=lambda value: _consume_progress(
                value,
                progress,
                operation=operation,
                started_at=started_at,
                previous=previous,
            ),
            check_cancelled=cancelled,
            policy=PersistentWatchdogPolicy(
                hard_timeout_seconds=_HARD_TIMEOUT_SECONDS,
                stage_timeout_seconds=_STAGE_TIMEOUT_SECONDS,
                heartbeat_timeout_seconds=_HEARTBEAT_TIMEOUT_SECONDS,
                cancel_grace_seconds=_CANCEL_GRACE_SECONDS,
                poll_seconds=_POLL_SECONDS,
                maximum_stream_bytes=_MAX_PROGRESS_BYTES,
                maximum_stream_records=_MAX_PROGRESS_RECORDS,
            ),
        )
    except PersistentWorkerExited as exc:
        raise RuntimeError(
            "LTX 2.3 Kitchen worker exited without a bounded result"
        ) from exc
    except PersistentWorkerStreamError as exc:
        raise RuntimeError(
            f"LTX 2.3 Kitchen worker {exc.stream} is invalid or exceeds its bound"
        ) from exc
    except PersistentWorkerTimeout as exc:
        messages = {
            "hard": "LTX 2.3 Kitchen generation exceeded its bounded deadline",
            "stage": "LTX 2.3 Kitchen worker stage exceeded its bounded deadline",
            "heartbeat": "LTX 2.3 Kitchen worker heartbeat became stale",
        }
        raise RuntimeError(messages[exc.clock]) from exc


def _consume_progress(
    item: Mapping[str, Any],
    callback: Callable[[float, str | None], None],
    *,
    operation: str,
    started_at: float,
    previous: list[float],
) -> None:
    if (
        not isinstance(item, Mapping)
        or set(item) != {"progress", "message"}
        or isinstance(item.get("progress"), bool)
        or not isinstance(item.get("progress"), (int, float))
        or not 0.0 <= float(item["progress"]) <= 1.0
        or float(item["progress"]) < previous[0]
        or not isinstance(item.get("message"), (str, type(None)))
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker progress record is invalid")
    value = float(item["progress"])
    previous[0] = value
    callback(value, item["message"])
    _LOGGER.info(
        "LTX 2.3 Kitchen worker progress: operation=%s phase=%s progress=%.3f "
        "elapsed_seconds=%.3f",
        operation,
        _worker_progress_phase(item["message"]),
        value,
        time.perf_counter() - started_at,
    )


def _worker_progress_phase(message: str | None) -> str:
    """Map worker-owned IPC text to a log-safe phase name.

    The server log intentionally never repeats arbitrary worker IPC text, so
    an invalid or future message cannot disclose prompts or local paths.
    """

    if message is None:
        return "working"
    prefixes = (
        ("LTX worker started", "worker_started"),
        ("Validating LTX worker request", "validate_bound_request"),
        ("Rehydrating LTX recipe", "rehydrate_recipe"),
        ("Importing LTX runtime", "import_runtime"),
        ("Preparing LTX generation", "prepare_generation"),
        ("Inspecting LTX transformer artifact", "inspect_transformer"),
        ("Building LTX transformer shell", "build_transformer_shell"),
        ("Planning LTX transformer materialization", "plan_transformer_materialization"),
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


def _worker_error(path: Path, exit_code: int, binding: str, secret: bytes) -> str:
    """Return the public message for one already-sanitized worker failure."""

    return _worker_failure(path, exit_code, binding, secret)["message"]


def _worker_failure(path: Path, exit_code: int, binding: str, secret: bytes) -> dict[str, str]:
    """Read bounded worker diagnostics while never surfacing child exception text.

    The child persists its raw exception only in disposable IPC for local
    debugging. The Engine publishes the operation stage, internal function
    identifier, and a deterministic fingerprint instead, so prompts and local
    filesystem paths cannot leak through a provider error.
    """

    try:
        value = _read_result(path, Path(), binding, secret)
        if value["ok"] is False:
            return _worker_failure_result(value)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        pass
    return {"message": f"LTX 2.3 Kitchen worker exited with code {exit_code}"}


def _worker_failure_result(value: Mapping[str, Any]) -> dict[str, str]:
    """Render one already-authenticated, exact-schema child failure."""

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


def _terminate_supervisor(
    supervisor: PersistentWorkerSupervisor, primary: BaseException
) -> bool:
    try:
        supervisor.terminate()
        return True
    except BaseException as exc:
        exc.add_note(
            f"while handling LTX 2.3 Kitchen worker failure: {type(primary).__name__}: {primary}"
        )
        raise


def _last_worker(
    process: Any,
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


def _remove_output(path: Path, primary: BaseException) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"LTX 2.3 Kitchen partial output cleanup failed: {exc}")
