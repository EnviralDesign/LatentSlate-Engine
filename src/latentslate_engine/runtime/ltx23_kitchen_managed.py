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
from itertools import pairwise
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

_SCHEMA_VERSION = 2
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_HARD_TIMEOUT_SECONDS = 60 * 60
_STAGE_TIMEOUT_SECONDS = 20 * 60
_HEARTBEAT_TIMEOUT_SECONDS = 45
_CANCEL_GRACE_SECONDS = 5
_FPS = 25
_AUDIO_SAMPLE_RATE = 48_000
_AUDIO_CHANNELS = 2
_AUDIO_SOURCE_SAMPLE_RATE = 16_000
_AUDIO_MEL_HOP_LENGTH = 160
_AUDIO_TEMPORAL_COMPRESSION_RATIO = 4
_AUDIO_DURATION_POLICY = "source_derived_exact_duration_v1"
_AAC_PACKET_SAMPLES = 1_024
_PROMPT_CACHE_MAX_BYTES = 1024 * 1024**2
_PROMPT_CACHE_MAX_ENTRIES = 8
_AIMDO_VBAR_PAGE_BYTES = 32 * 1024**2
_AIMDO_POISON_EXIT_CODE = 86
_AIMDO_POISON_REASONS = frozenset(
    {
        "device_quiescence_failed",
        "failed_fill_quiescence_failed",
        "host_source_pool_structural_failure",
        "host_source_pool_setup_cleanup_failed",
        "ltx23_av_dynamic_initialization_cleanup_failed",
        "retirement_release_failed",
        "retirement_cleanup_failed",
        "retirement_query_failed",
        "retirement_quiescence_failed",
        "stage_prepare_failed",
    }
)
_AIMDO_STAGE_PREPARE_INTEGER_FIELDS = frozenset(
    {
        "stage_prepare_calls",
        "stage_prepare_requested_bytes",
        "stage_prepare_pending_before",
        "stage_prepare_pending_after",
        "stage_prepare_loaded_before",
        "stage_prepare_loaded_after",
        "stage_prepare_trim_requested",
        "stage_prepare_trim_freed",
        "stage_prepare_cuda_allocated_before",
        "stage_prepare_cuda_reserved_before",
        "stage_prepare_cuda_free_before",
        "stage_prepare_cuda_allocated_after",
        "stage_prepare_cuda_reserved_after",
        "stage_prepare_cuda_free_after",
    }
)
_AIMDO_STREAM_RETIREMENT_INTEGER_FIELDS = frozenset(
    {
        "reverse_stream_waits",
        "retirement_batches",
        "retirement_polls",
        "retirement_completions",
        "pending_retirement_batches",
    }
)
_TEXT_TIMING_PHASES = (
    "materialization",
    "text_onload",
    "enhancement",
    "positive_encode",
    "negative_encode",
    "text_offload",
    "downstream",
    "prompt_cache_publish",
)
_TWO_STAGE_MEMORY_PHASES = (
    "after_text_offload",
    "after_stage1",
    "after_latent_upscaling",
    "after_stage2",
    "after_decode",
    "after_transient_clearing",
    "after_prompt_cache_publication",
)
_SINGLE_STAGE_MEMORY_PHASES = (
    "after_text_offload",
    "after_main_denoise",
    "after_decode",
    "after_transient_clearing",
    "after_prompt_cache_publication",
)
_DEV_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
_PROMPT_ENHANCEMENT_SYSTEM_SHA256 = (
    "f00b22f47dad68358f5c2c7396c701db95095cf26dc3dbd6b5556eab04692071"
)
_PROMPT_ENHANCEMENT_TEMPLATE = "comfy_ltx2_gemma3_manual_v1"
_PROMPT_ENHANCEMENT_STOP_TOKEN_ID = 106
_PROMPT_ENHANCEMENT_SEED = 0
_PROMPT_ENHANCEMENT_MAX_NEW_TOKENS = 2_048
_PROMPT_ENHANCEMENT_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
}
_I2V_GUIDE_LONGER_EDGE = 1_536
_I2V_GUIDE_CRF = 18
_I2V_GUIDE_PRESET = "veryfast"
_I2V_GUIDE_PIXEL_FORMAT = "yuv420p"
_LTX23_REFILL_FAILURE_REASONS = frozenset(
    {
        "unbound_root_exceeds_target",
        "resident_trim_failed",
        "binding_acquire_failed",
    }
)
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

    def __init__(
        self,
        request: LTX23KitchenRuntimeRequest,
        *,
        cache_policy: str = "none",
    ) -> None:
        if request.operation not in _OPERATIONS:
            raise ValueError("managed LTX 2.3 Kitchen runtime requires a supported operation")
        if cache_policy not in {"none", "prompt"}:
            raise ValueError("managed LTX 2.3 Kitchen cache policy must be 'none' or 'prompt'")
        self.request = request
        self.cache_policy = cache_policy
        self._session: _WorkerSession | None = None
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []
        self._last_cache = _empty_cache_status(cache_policy)
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
        worker_failure: dict[str, Any] | None = None
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
            payload = _payload(
                self.request,
                generation,
                device=device,
                cache_policy=self.cache_policy,
                secret=secret,
            )
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
                supervisor, allocator_policy = _supervisor(_persistent_paths(paths), secret)
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
            result = _read_result(paths["result"], output_path, payload["request_binding"], secret)
            if result["ok"] is False:
                worker_failure = _worker_failure_result(result)
                _log_worker_failure(worker_failure)
                raise RuntimeError(worker_failure["message"])
            process = _process(session)
            if process.poll() is not None:
                raise RuntimeError(
                    "LTX 2.3 Kitchen worker exited before its success result was accepted"
                )
            _validate_metadata(result["metadata"], self.request, generation, self.cache_policy)
            self._last_worker = _last_worker(
                process,
                "succeeded",
                tree_empty=False,
                allocator_policy=result["allocator_policy"],
            )
            self._last_worker["pipeline_warm"] = bool(
                result["metadata"].get("cache", {}).get("pipeline_warm")
            )
            self._last_cache = dict(result["metadata"].get("cache", {}))
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
                    self._last_cache = _empty_cache_status(self.cache_policy)
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
            "cache_support": {"prompt": True, "media": False},
            "cache": dict(self._last_cache),
        }

    def clear_cache(self) -> None:
        """Clear the persistent child's bounded prompt cache without unloading it."""

        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("managed LTX 2.3 Kitchen worker is already active")
        self._job_active = True
        session = self._session
        if session is None:
            self._last_cache = _empty_cache_status(self.cache_policy)
            self._job_active = False
            self._ownership.release()
            return
        try:
            paths = _path_mapping(session.supervisor.paths)
            _require_job_fresh(paths)
            payload = _clear_cache_payload(
                self.request,
                cache_policy=self.cache_policy,
                secret=session.secret,
            )
            session.supervisor.send(payload)
            _wait_for_result(
                session.supervisor,
                lambda *_args: None,
                lambda: None,
                self.request.operation,
            )
            result = _read_clear_cache_result(
                paths["result"],
                payload["request_binding"],
                session.secret,
                self.cache_policy,
            )
            if result["ok"] is False:
                raise RuntimeError(_worker_failure_result(result)["message"])
            if _process(session).poll() is not None:
                raise RuntimeError("LTX 2.3 Kitchen worker exited during cache clear")
            self._last_cache = dict(result["cache"])
            self._cleanup_errors = session.supervisor.cleanup_job()
        except BaseException as primary:
            try:
                _terminate_supervisor(session.supervisor, primary)
            finally:
                self._session = None
                self._last_cache = _empty_cache_status(self.cache_policy)
                try:
                    session.supervisor.close()
                finally:
                    self._cleanup_errors = session.supervisor.cleanup_session()
            raise
        finally:
            self._job_active = False
            self._ownership.release()

    def unload(self) -> None:
        session = self._session
        if session is None:
            self._last_cache = _empty_cache_status(self.cache_policy)
            return
        primary: BaseException | None = None
        try:
            session.supervisor.terminate()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._session = None
            self._last_cache = _empty_cache_status(self.cache_policy)
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
    """Floor the Comfy 25 fps request onto the effective ``8k+1`` decode grid."""

    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise TypeError("LTX 2.3 duration must be numeric")
    if (
        not math.isfinite(float(duration_seconds))
        or duration_seconds < 1.0
        or duration_seconds > 10.0
    ):
        raise ValueError("LTX 2.3 duration must be finite and within 1..10 seconds")
    requested = requested_length_for_duration(duration_seconds)
    frames = ((requested - 1) // 8) * 8 + 1
    if not 25 <= frames <= 249 or frames % 8 != 1:
        raise AssertionError("LTX 2.3 temporal alignment contract was violated")
    return frames


def _empty_cache_status(cache_policy: str) -> dict[str, Any]:
    return {
        "pipeline_warm": False,
        "policy": cache_policy,
        "prompt_hit": False,
        "prompt_published": False,
        "media_hit": False,
        "prompt": {
            "name": "prompt",
            "enabled": cache_policy == "prompt",
            "entries": 0,
            "bytes": 0,
            "max_bytes": _PROMPT_CACHE_MAX_BYTES,
            "max_entries": _PROMPT_CACHE_MAX_ENTRIES,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": None,
        },
    }


def requested_length_for_duration(duration_seconds: float) -> int:
    """Return the source request length before LTX temporal latent flooring."""

    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise TypeError("LTX 2.3 duration must be numeric")
    if (
        not math.isfinite(float(duration_seconds))
        or duration_seconds < 1.0
        or duration_seconds > 10.0
    ):
        raise ValueError("LTX 2.3 duration must be finite and within 1..10 seconds")
    return math.floor(float(duration_seconds) * _FPS) + 1


def _generation(operation: str, **value: Any) -> dict[str, object]:
    start = _endpoint(value["start_image_path"])
    end = _endpoint(value["end_image_path"])
    return {
        "prompt": value["prompt"],
        "width": value["width"],
        "height": value["height"],
        "duration_seconds": value["duration_seconds"],
        "requested_num_frames": requested_length_for_duration(value["duration_seconds"]),
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
            "requested_num_frames",
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
    width, height, seed, requested_frames, frames = (
        value["width"],
        value["height"],
        value["seed"],
        value["requested_num_frames"],
        value["num_frames"],
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (width, height, seed, requested_frames, frames)
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
    if requested_frames != requested_length_for_duration(value["duration_seconds"]):
        raise ValueError("LTX 2.3 Kitchen requested frame count does not bind its duration")
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
    cache_policy: str,
    secret: bytes,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "command": "generate",
        "request": request.to_json_dict(),
        "generation": dict(generation),
        "device": device,
        "cache_policy": cache_policy,
    }
    return {**unsigned, "request_binding": _binding(unsigned, secret)}


def _clear_cache_payload(
    request: LTX23KitchenRuntimeRequest,
    *,
    cache_policy: str,
    secret: bytes,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "command": "clear_cache",
        "request_fingerprint": request.fingerprint,
        "cache_policy": cache_policy,
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
        raise RuntimeError(
            "LTX 2.3 Kitchen worker job IPC paths already exist: " + ", ".join(stale)
        )


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonFileError:
        raise RuntimeError("LTX 2.3 Kitchen worker JSON is missing or exceeds its bound")


def _read_result(path: Path, output: Path, binding: str, secret: bytes) -> dict[str, Any]:
    try:
        result = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
        raise RuntimeError("LTX 2.3 Kitchen worker result does not bind to its command") from None
    if not isinstance(result, dict):
        raise RuntimeError(  # noqa: TRY004 - uniform authenticated protocol failure.
            "LTX 2.3 Kitchen worker result does not bind to its command"
        )
    result_binding = result.get("result_binding")
    try:
        expected_result_binding = _result_binding(result, secret)
    except (TypeError, ValueError):
        raise RuntimeError("LTX 2.3 Kitchen worker result does not bind to its command") from None
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
        "cleanup_stage",
        "aimdo_counters",
    }
    if result.get("ok") is False:
        terminal_fields = failure_fields | {
            "terminal_exit_code",
            "poison_reason",
            "poison_origin",
        }
        fields = frozenset(result)
        terminal = fields == frozenset(terminal_fields)
        if (
            fields not in {frozenset(failure_fields), frozenset(terminal_fields)}
            or not isinstance(result.get("error_type"), str)
            or not result["error_type"].replace("_", "").isalnum()
            or len(result["error_type"]) > 80
            or not isinstance(result.get("error"), str)
            or len(result["error"]) > 4096
            or not _valid_failure_diagnostic(result)
            or not _valid_failure_aimdo_counters(result.get("aimdo_counters"))
            or (result.get("error_type") == "LTX23KitchenWorkerPoisoned" and not terminal)
            or (
                terminal
                and (
                    result.get("terminal_exit_code") != _AIMDO_POISON_EXIT_CODE
                    or result.get("poison_reason") not in _AIMDO_POISON_REASONS
                    or result.get("poison_origin") not in {"primary", "cleanup"}
                    or (
                        result.get("poison_origin") == "primary"
                        and (
                            result.get("error_type") != "LTX23KitchenWorkerPoisoned"
                            or result.get("cleanup_stage") == "unload_runtime"
                        )
                    )
                    or (
                        result.get("poison_origin") == "cleanup"
                        and (
                            result.get("error_type") == "LTX23KitchenWorkerPoisoned"
                            or result.get("cleanup_stage") != "unload_runtime"
                        )
                    )
                )
            )
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
    if set(result) != success_fields or result.get("ok") is not True:
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


def _read_clear_cache_result(
    path: Path,
    binding: str,
    secret: bytes,
    cache_policy: str,
) -> dict[str, Any]:
    """Accept one authenticated cache-clear result without an output artifact."""

    try:
        result = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker cache-clear result does not bind to its command"
        ) from None
    if not isinstance(result, dict):
        raise RuntimeError(  # noqa: TRY004 - uniform authenticated protocol failure.
            "LTX 2.3 Kitchen worker cache-clear result does not bind to its command"
        )
    result_binding = result.get("result_binding")
    try:
        expected_result_binding = _result_binding(result, secret)
    except (TypeError, ValueError):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker cache-clear result does not bind to its command"
        ) from None
    if (
        not isinstance(result_binding, str)
        or not hmac.compare_digest(result_binding, expected_result_binding)
        or result.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(result.get("request_binding"), str)
        or not hmac.compare_digest(result["request_binding"], binding)
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker cache-clear result does not bind to its command")
    if result.get("ok") is False:
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
            "cleanup_stage",
            "aimdo_counters",
        }
        if (
            set(result) != failure_fields
            or not isinstance(result.get("error_type"), str)
            or not result["error_type"].replace("_", "").isalnum()
            or len(result["error_type"]) > 80
            or not isinstance(result.get("error"), str)
            or len(result["error"]) > 4096
            or not _valid_failure_diagnostic(result)
            or not _valid_failure_aimdo_counters(result.get("aimdo_counters"))
        ):
            raise RuntimeError("LTX 2.3 Kitchen worker cache-clear failure is invalid")
        return result
    expected = {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "command",
        "cache",
    }
    cache = result.get("cache")
    if (
        set(result) != expected
        or result.get("ok") is not True
        or result.get("command") != "clear_cache"
        or not _valid_cache_status(
            cache,
            expected_policy=cache_policy,
            expected_pipeline_warm=True,
            expected_prompt_hit=False,
            require_empty=True,
        )
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker returned an invalid cache-clear result")
    return result


def _validate_metadata(
    metadata: Mapping[str, object],
    request: LTX23KitchenRuntimeRequest,
    generation: Mapping[str, object],
    cache_policy: str = "none",
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
        "requested_num_frames": generation["requested_num_frames"],
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
        "requested_duration_seconds": (int, float),
        "effective_duration_seconds": (int, float),
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
        or metadata["requested_duration_seconds"] != generation["duration_seconds"]
        or metadata["effective_duration_seconds"] != video_duration
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
        or normalization["temporal_compression_ratio"] != _AUDIO_TEMPORAL_COMPRESSION_RATIO
        or normalization["fps"] != _FPS
        or normalization["fps"] != metadata["fps"]
        or normalization["audio_latent_frames"] <= 0
        or normalization["expected_audio_latent_frames"]
        != round((normalization["video_frames"] / normalization["fps"]) * 25)
        or normalization["audio_latent_frames"] != normalization["expected_audio_latent_frames"]
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
    if not _valid_native_text_proof(text_proof):
        raise RuntimeError(
            "LTX 2.3 Kitchen worker did not prove strict full-precision text execution"
        )
    if not _valid_cache_status(metadata.get("cache"), expected_policy=cache_policy):
        raise RuntimeError("LTX 2.3 Kitchen worker cache provenance is invalid")
    if not _valid_negative_encoding(
        metadata.get("negative_encoding"), request.operation, metadata.get("negative_prompt")
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker negative text provenance is invalid")
    if not _valid_prompt_enhancement_provenance(metadata, request.operation):
        raise RuntimeError("LTX 2.3 Kitchen worker prompt-enhancement provenance is invalid")
    if not _valid_i2v_guide_preprocessing(
        metadata.get("guide_preprocessing"), request.operation, generation
    ):
        raise RuntimeError("LTX 2.3 Kitchen worker guide preprocessing provenance is invalid")
    text_patch_state = metadata.get("text_patch_state")
    if not _valid_text_patch_state(text_patch_state, request.operation):
        raise RuntimeError("LTX 2.3 Kitchen worker text patch-state provenance is invalid")
    lora_entry_transitions, lora_to_base_transitions = _text_patch_transition_counts(
        text_patch_state
    )
    if not _valid_text_residency(
        metadata.get("text_residency"),
        lora_entry_transitions=lora_entry_transitions,
        lora_to_base_transitions=lora_to_base_transitions,
    ):
        _LOGGER.error(
            "LTX 2.3 Kitchen worker rejected text residency proof: %s",
            canonical_json(
                _text_residency_rejection_summary(metadata.get("text_residency"))
            ).decode("utf-8"),
        )
        raise RuntimeError("LTX 2.3 Kitchen worker text residency provenance is invalid")
    if not _valid_timings(metadata.get("timings")):
        raise RuntimeError("LTX 2.3 Kitchen worker timing provenance is invalid")
    if not _valid_memory_telemetry(metadata.get("memory_telemetry"), request.operation):
        raise RuntimeError("LTX 2.3 Kitchen worker memory telemetry is invalid")
    text_lora = metadata.get("text_lora")
    if request.operation.startswith("ltx23_dev_"):
        if not _valid_text_lora_proof(
            text_lora,
            expected_active=True,
        ):
            raise RuntimeError("LTX 2.3 Kitchen worker text LoRA provenance is invalid")
    elif text_lora is not None:
        raise RuntimeError("LTX 2.3 Kitchen worker unexpectedly reported a text LoRA")
    residency = metadata.get("residency_policy")
    if not _valid_transformer_residency(residency, request.operation):
        _LOGGER.error(
            "LTX 2.3 Kitchen worker rejected transformer residency proof: %s",
            canonical_json(residency).decode("utf-8"),
        )
        raise RuntimeError("LTX 2.3 Kitchen worker residency provenance is invalid")


def _text_residency_rejection_summary(value: object) -> dict[str, object]:
    """Return bounded, non-prompt diagnostics for an invalid worker proof."""

    encoded = canonical_json(value)
    summary: dict[str, object] = {
        "proof_sha256": hashlib.sha256(encoded).hexdigest(),
        "proof_bytes": len(encoded),
        "proof_type": type(value).__name__,
    }
    if not isinstance(value, Mapping):
        return summary
    for key in (
        "mode",
        "root_transitions",
        "layer_count",
        "layer_transitions",
        "full_precision_dispatches",
        "live_layer_bindings",
        "transfer_events",
        "transfer_waits",
        "warm_request_index",
    ):
        summary[key] = value.get(key)
    dynamic = value.get("dynamic_vram")
    if isinstance(dynamic, Mapping):
        summary["dynamic_vram"] = {
            key: dynamic.get(key)
            for key in (
                "loaded_bytes",
                "live_bytes",
                "faults",
                "signature_hits",
                "signature_misses",
                "fault_none_temporaries",
                "unpin_calls",
                "transfer_events",
                "transfer_waits",
                "host_source_pool_hits",
                "host_source_pool_misses",
                "host_source_pool_poisoned",
                "poisoned",
                "close_failed",
            )
        }
    return summary


def _valid_transformer_residency(value: object, operation: str) -> bool:
    """Validate only the stable Phase-1 direct AIMDO facts.

    Scheduler groups, source-pool lanes, retirement batches, and predictive stage
    budgets were implementation details of the retired AV path and are not worker
    protocol contracts.
    """

    if not isinstance(value, Mapping):
        return False
    expected = {
        "mode",
        "stored_bytes",
        "base_stored_bytes",
        "companion_stored_bytes",
        "leaf_allocation_count",
        "force_resident_leaf_count",
        "block_count",
        "prefetch",
        "base_file_backed",
        "base_file_handle_live",
        "dynamic_vram",
    }
    if set(value) != expected:
        return False
    if (
        value.get("mode") != "comfy_direct_leaf_vbar"
        or value.get("prefetch") is not True
        or value.get("base_file_backed") is not True
        or value.get("base_file_handle_live") is not True
        or value.get("block_count") != 48
        or not isinstance(value.get("stored_bytes"), int)
        or isinstance(value.get("stored_bytes"), bool)
        or value["stored_bytes"] <= 0
        or not isinstance(value.get("base_stored_bytes"), int)
        or isinstance(value.get("base_stored_bytes"), bool)
        or value["base_stored_bytes"] <= 0
        or not isinstance(value.get("companion_stored_bytes"), int)
        or isinstance(value.get("companion_stored_bytes"), bool)
        or not isinstance(value.get("leaf_allocation_count"), int)
        or value["leaf_allocation_count"] <= 0
        or not isinstance(value.get("force_resident_leaf_count"), int)
        or not 0 < value["force_resident_leaf_count"] < value["leaf_allocation_count"]
    ):
        return False
    if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}:
        if value["companion_stored_bytes"] <= 0:
            return False
    elif operation == "ltx23_distilled_flf":
        if value["companion_stored_bytes"] != 0:
            return False
    else:
        return False
    if value["stored_bytes"] != (
        value["base_stored_bytes"] + value["companion_stored_bytes"]
    ):
        return False
    dynamic = value.get("dynamic_vram")
    if not isinstance(dynamic, Mapping):
        return False
    integer_fields = {
        "allocation_count",
        "loaded_bytes",
        "faults",
        "signature_hits",
        "signature_misses",
        "fault_none_temporaries",
        "gathered_h2d_bytes",
        "base_file_read_calls",
        "base_file_read_bytes",
        "prefetch_calls",
        "unpin_calls",
        "cleanup_calls",
        "forward_stream_waits",
        "reverse_stream_waits",
    }
    if set(dynamic) != integer_fields | {"backend", "version", "mode", "poison_reason"}:
        return False
    if any(
        not isinstance(dynamic.get(key), int)
        or isinstance(dynamic.get(key), bool)
        or dynamic[key] < 0
        for key in integer_fields
    ):
        return False
    return bool(
        dynamic.get("backend") == "comfy-aimdo"
        and dynamic.get("version") == "0.4.15"
        and dynamic.get("mode") == "ltx23_av_direct"
        and dynamic.get("poison_reason") is None
        and dynamic["allocation_count"]
        == value["leaf_allocation_count"] - value["force_resident_leaf_count"]
        and dynamic["faults"] == dynamic["signature_hits"] + dynamic["signature_misses"]
        and dynamic["fault_none_temporaries"] <= dynamic["signature_misses"]
        and dynamic["prefetch_calls"] >= dynamic["faults"]
        and dynamic["forward_stream_waits"] <= dynamic["prefetch_calls"]
        and dynamic["reverse_stream_waits"] == dynamic["forward_stream_waits"]
        and dynamic["base_file_read_bytes"] <= dynamic["gathered_h2d_bytes"]
        and dynamic["base_file_read_calls"] > 0
        and dynamic["base_file_read_bytes"] > 0
        and dynamic["signature_misses"] > 0
        and dynamic["cleanup_calls"] == 0
    )


def _valid_native_text_proof(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("provenance") == "cached_prompt_conditioning":
        return value.get("dispatch_performed") is False and _valid_native_text_proof(
            value.get("source_proof")
        )
    return bool(
        value.get("backend") == "engine-native/comfy-strict-full-precision-mm"
        and value.get("policy") == "full_precision_mm"
        and isinstance(value.get("module_count"), int)
        and value["module_count"] > 0
        and isinstance(value.get("total_dispatches"), int)
        and value["total_dispatches"] > 0
        and isinstance(value.get("minimum_module_dispatches"), int)
        and value["minimum_module_dispatches"] > 0
        and isinstance(value.get("maximum_module_dispatches"), int)
        and value["maximum_module_dispatches"] >= value["minimum_module_dispatches"]
        and value.get("native_quantized_dispatches") == 0
        and value.get("rejected_dispatches") == 0
        and value.get("dense_fallback_dispatches") == 0
    )


def _valid_negative_encoding(value: object, operation: str, prompt: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("provenance") == "cached_prompt_conditioning":
        return value.get("dispatch_performed") is False and _valid_negative_encoding(
            value.get("source_proof"), operation, prompt
        )
    expected_prompt = _DEV_NEGATIVE_PROMPT if operation.startswith("ltx23_dev_") else prompt
    if not isinstance(expected_prompt, str):
        return False
    return bool(
        value.get("prompt_sha256") == hashlib.sha256(expected_prompt.encode()).hexdigest()
        and value.get("max_sequence_length") == 1024
        and value.get("dtype") == "bfloat16"
        and value.get("mask_dtype") in {"int64", "bool"}
        and value.get("finite") is True
        and value.get("encoded") is True
        and value.get("used_for_cfg") is False
        and _valid_conditioning_shapes(value.get("embeds_shape"), value.get("mask_shape"))
    )


def _valid_conditioning_shapes(embeds: object, mask: object) -> bool:
    return bool(
        isinstance(embeds, list)
        and len(embeds) == 3
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in embeds
        )
        and embeds == [1, 1024, 188160]
        and isinstance(mask, list)
        and mask == [1, 1024]
    )


def _valid_prompt_enhancement_provenance(metadata: Mapping[str, object], operation: str) -> bool:
    enabled = operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}
    expected = {
        "prompt_enhanced": enabled,
        "prompt_enhancement_system_sha256": (
            _PROMPT_ENHANCEMENT_SYSTEM_SHA256 if enabled else None
        ),
        "prompt_enhancement_seed": _PROMPT_ENHANCEMENT_SEED if enabled else None,
        "prompt_enhancement_max_new_tokens": (
            _PROMPT_ENHANCEMENT_MAX_NEW_TOKENS if enabled else None
        ),
        "prompt_enhancement_stop_token_id": (
            _PROMPT_ENHANCEMENT_STOP_TOKEN_ID if enabled else None
        ),
        "prompt_enhancement_template": _PROMPT_ENHANCEMENT_TEMPLATE if enabled else None,
        "prompt_enhancement_generation_settings": (
            _PROMPT_ENHANCEMENT_GENERATION_SETTINGS if enabled else None
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    memory = metadata.get("prompt_enhancement_memory")
    if not enabled:
        return memory is None
    cache = metadata.get("cache")
    cached = isinstance(cache, Mapping) and cache.get("prompt_hit") is True
    return _valid_prompt_enhancement_memory(memory, cached=cached)


def _valid_prompt_enhancement_memory(value: object, *, cached: bool = False) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("policy") == "not_required_prompt_cache_hit":
        return bool(
            cached
            and set(value) == {"policy", "source_proof"}
            and _valid_prompt_enhancement_memory(value.get("source_proof"))
        )
    if cached:
        return False
    expected = {
        "policy",
        "cache_present",
        "cache_type",
        "cuda_allocated_before_bytes",
        "cuda_allocated_after_bytes",
        "cuda_allocated_released_bytes",
        "template",
        "stop_token_id",
        "generation_settings",
        "decoded_suffix_nonempty",
        "think_block_removed",
        "fallback_to_source_prompt",
    }
    if (
        set(value) != expected
        or value.get("policy") != "release_after_prompt_enhancement"
        or not isinstance(value.get("cache_present"), bool)
        or value.get("template") != _PROMPT_ENHANCEMENT_TEMPLATE
        or value.get("stop_token_id") != _PROMPT_ENHANCEMENT_STOP_TOKEN_ID
        or value.get("generation_settings") != _PROMPT_ENHANCEMENT_GENERATION_SETTINGS
        or any(
            not isinstance(value.get(field), bool)
            for field in (
                "decoded_suffix_nonempty",
                "think_block_removed",
                "fallback_to_source_prompt",
            )
        )
        or (not value["decoded_suffix_nonempty"] and not value["fallback_to_source_prompt"])
        or (value["think_block_removed"] and not value["decoded_suffix_nonempty"])
        or (value["cache_present"] and not isinstance(value.get("cache_type"), str))
        or (not value["cache_present"] and value.get("cache_type") is not None)
    ):
        return False
    before = value.get("cuda_allocated_before_bytes")
    after = value.get("cuda_allocated_after_bytes")
    released = value.get("cuda_allocated_released_bytes")
    if before is None or after is None:
        return before is None and after is None and released in {None, 0}
    return bool(
        isinstance(before, int)
        and not isinstance(before, bool)
        and before >= 0
        and isinstance(after, int)
        and not isinstance(after, bool)
        and after >= 0
        and isinstance(released, int)
        and not isinstance(released, bool)
        and released == max(0, before - after)
    )


def _valid_i2v_guide_preprocessing(
    value: object,
    operation: str,
    generation: Mapping[str, object],
) -> bool:
    """Authenticate the operation-local pinned-Comfy I2V guide chain."""

    if operation != "ltx23_dev_i2v":
        return value is None
    expected_keys = {
        "policy",
        "ordering",
        "source_size",
        "source_file_identity",
        "center_crop_box",
        "resize_dimensions_size",
        "resize_dimensions_method",
        "longer_edge",
        "longer_edge_size",
        "compression_codec",
        "compression_crf",
        "compression_preset",
        "compression_pixel_format",
        "operation_image_size",
        "operation_image_identity_sha256",
        "stage_image_identities",
        "stage_dimensions",
        "stage_strengths",
        "shared_operation_image",
        "persistent_guide_cache",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        return False
    width = generation.get("width")
    height = generation.get("height")
    source_size = value.get("source_size")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or not isinstance(source_size, list)
        or len(source_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in source_size
        )
    ):
        return False
    source_width, source_height = source_size
    old_aspect = source_width / source_height
    new_aspect = width / height
    crop_x = 0
    crop_y = 0
    if old_aspect > new_aspect:
        crop_x = round((source_width - source_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        crop_y = round((source_height - source_height * (old_aspect / new_aspect)) / 2)
    expected_crop = [
        crop_x,
        crop_y,
        source_width - crop_x,
        source_height - crop_y,
    ]
    if width > height:
        longer_size = [
            _I2V_GUIDE_LONGER_EDGE,
            int(height * (_I2V_GUIDE_LONGER_EDGE / width)),
        ]
    else:
        longer_size = [
            int(width * (_I2V_GUIDE_LONGER_EDGE / height)),
            _I2V_GUIDE_LONGER_EDGE,
        ]
    operation_size = [longer_size[0] // 2 * 2, longer_size[1] // 2 * 2]
    identity = value.get("operation_image_identity_sha256")
    return bool(
        value.get("policy") == "pinned_comfy_i2v_guide_v1"
        and value.get("ordering")
        == [
            "resize_dimensions_center_lanczos",
            "resize_longer_edge_1536_pil_lanczos",
            "h264_single_frame_roundtrip",
        ]
        and value.get("source_file_identity") == generation.get("start_image_identity")
        and value.get("center_crop_box") == expected_crop
        and value.get("resize_dimensions_size") == [width, height]
        and value.get("resize_dimensions_method") == "pil_lanczos_common_upscale_uint8"
        and value.get("longer_edge") == _I2V_GUIDE_LONGER_EDGE
        and value.get("longer_edge_size") == longer_size
        and value.get("compression_codec") == "libx264"
        and value.get("compression_crf") == _I2V_GUIDE_CRF
        and value.get("compression_preset") == _I2V_GUIDE_PRESET
        and value.get("compression_pixel_format") == _I2V_GUIDE_PIXEL_FORMAT
        and value.get("operation_image_size") == operation_size
        and isinstance(identity, str)
        and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity)
        and value.get("stage_image_identities") == [identity, identity]
        and value.get("stage_dimensions") == [[width // 2, height // 2], [width, height]]
        and value.get("stage_strengths") == [0.7, 1.0]
        and value.get("shared_operation_image") is True
        and value.get("persistent_guide_cache") is False
    )


def _valid_text_patch_state(value: object, operation: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("provenance") == "cached_prompt_conditioning":
        return value.get("dispatch_performed") is False and _valid_text_patch_state(
            value.get("source_proof"), operation
        )
    expected_policy = (
        "prompt_enhancement_only"
        if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}
        else "base_only"
    )
    return bool(
        value.get("policy") == expected_policy
        and value.get("lora_strength_enhancement")
        == (1.0 if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"} else None)
        and value.get("lora_strength_positive") == 0.0
        and value.get("lora_strength_negative") == 0.0
        and value.get("lora_entry_transitions")
        == (1 if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"} else 0)
        and value.get("lora_to_base_transitions")
        == (1 if operation.startswith("ltx23_dev_") else 0)
        and value.get("restored_base_on_exit") is True
    )


def _text_patch_transition_counts(value: object) -> tuple[int, int]:
    while isinstance(value, Mapping) and value.get("provenance") == "cached_prompt_conditioning":
        value = value.get("source_proof")
    if not isinstance(value, Mapping):
        return (-1, -1)
    entry = value.get("lora_entry_transitions")
    restored = value.get("lora_to_base_transitions")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in (entry, restored)):
        return (-1, -1)
    return (entry, restored)


def _valid_patched_resident_diagnostics(value: object) -> bool:
    expected = {
        "linear_merge_misses",
        "linear_hits",
        "linear_signature_none_rematerializations",
        "linear_requantize_writebacks",
        "embedding_merge_misses",
        "embedding_hits",
        "embedding_signature_none_rematerializations",
        "embedding_writebacks",
    }
    return bool(
        isinstance(value, Mapping)
        and set(value) == expected
        and all(
            isinstance(value[key], int) and not isinstance(value[key], bool) and value[key] >= 0
            for key in expected
        )
        and value["linear_merge_misses"] == value["linear_requantize_writebacks"]
        and value["embedding_merge_misses"] == value["embedding_writebacks"]
    )


def _valid_text_residency(
    value: object,
    *,
    lora_entry_transitions: int | None = None,
    lora_to_base_transitions: int | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("mode") == "not_required_prompt_cache_hit":
        return _valid_text_residency(
            value.get("source_proof"),
            lora_entry_transitions=lora_entry_transitions,
            lora_to_base_transitions=lora_to_base_transitions,
        )
    expected = {
        "mode",
        "root_activation",
        "layer_count",
        "root_weight_bytes",
        "largest_layer_weight_bytes",
        "required_weight_bytes",
        "root_transitions",
        "layer_transitions",
        "execution_policy",
        "patched_resident",
        "native_quantized_dispatches",
        "full_precision_dispatches",
        "transfer_mode",
        "transfer_stream_count",
        "transfer_events",
        "transfer_waits",
        "async_transfer_fallbacks",
        "strict_cuda_parity",
        "host_registration",
        "layer_compute_barriers",
        "live_layer_bindings",
        "live_layer_bytes",
        "maximum_live_layer_bindings",
        "maximum_live_layer_bytes",
        "dynamic_vbar_prefetch",
        "leaf_allocation_count",
        "force_resident_leaf_count",
        "base_leaf_count",
        "patch_leaf_count",
        "schedule_group_count",
        "leaf_scheduler",
        "warm_request_index",
        "dynamic_vram",
    }
    if set(value) != expected:
        return False
    integer_fields = (
        "layer_count",
        "root_weight_bytes",
        "largest_layer_weight_bytes",
        "required_weight_bytes",
        "root_transitions",
        "layer_transitions",
        "full_precision_dispatches",
        "native_quantized_dispatches",
        "transfer_stream_count",
        "transfer_events",
        "transfer_waits",
        "async_transfer_fallbacks",
        "layer_compute_barriers",
        "live_layer_bindings",
        "live_layer_bytes",
        "maximum_live_layer_bindings",
        "maximum_live_layer_bytes",
        "leaf_allocation_count",
        "force_resident_leaf_count",
        "base_leaf_count",
        "patch_leaf_count",
        "schedule_group_count",
        "warm_request_index",
    )
    common = bool(
        value.get("mode") in {"layer_streamed_cpu_master", "dynamic_vbar_per_leaf"}
        and value.get("execution_policy") == "strict_comfy_full_precision_mm"
        and _valid_patched_resident_diagnostics(value.get("patched_resident"))
        and value.get("strict_cuda_parity") is True
        and value.get("dynamic_vbar_prefetch") is False
        and all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and value[key] >= 0
            for key in integer_fields
        )
        and value["layer_count"] > 0
        and value["warm_request_index"] >= 1
        and value["leaf_allocation_count"] > value["layer_count"] + 1
        and 0 <= value["force_resident_leaf_count"] <= value["leaf_allocation_count"]
        and value["base_leaf_count"] + value["patch_leaf_count"] == value["leaf_allocation_count"]
        and value["schedule_group_count"] == 2 * (value["layer_count"] + 1)
        and value["root_weight_bytes"] > 0
        and value["largest_layer_weight_bytes"] > 0
        and 0
        < value["required_weight_bytes"]
        <= value["root_weight_bytes"] + value["largest_layer_weight_bytes"]
        and value["full_precision_dispatches"] > 0
        and value["native_quantized_dispatches"] == 0
        and value["root_transitions"] > 0
        and value["layer_transitions"] > 0
        and value["transfer_stream_count"] == 2
        and value["transfer_events"] == value["transfer_waits"]
        and value["async_transfer_fallbacks"] == 0
        and value["layer_compute_barriers"] == value["layer_transitions"]
        and value["live_layer_bindings"] == 0
        and value["live_layer_bytes"] == 0
        and value["maximum_live_layer_bindings"] == 1
        and 0 < value["maximum_live_layer_bytes"] <= value["largest_layer_weight_bytes"]
    )
    if not common:
        return False
    if value.get("mode") == "dynamic_vbar_per_leaf":
        dynamic = value.get("dynamic_vram")
        scheduler = value.get("leaf_scheduler")
        return bool(
            value.get("root_activation") == "per_model_forward_fault"
            and _valid_text_forward_accounting(value, lora_to_base_transitions)
            and value.get("transfer_mode") == "aimdo_two_stream_nonblocking"
            and _valid_dynamic_text_residency(
                dynamic,
                value["layer_count"],
                root_transitions=value["root_transitions"],
                layer_transitions=value["layer_transitions"],
                lora_entry_transitions=lora_entry_transitions,
                lora_to_base_transitions=lora_to_base_transitions,
                leaf_allocation_count=value["leaf_allocation_count"],
                force_resident_leaf_count=value["force_resident_leaf_count"],
                base_leaf_count=value["base_leaf_count"],
                patch_leaf_count=value["patch_leaf_count"],
                scheduler=scheduler,
                warm_request_index=value["warm_request_index"],
            )
            and value["transfer_events"] == dynamic["request_delta"]["transfer_events"]
            and value["transfer_waits"] == dynamic["request_delta"]["transfer_waits"]
            and value["transfer_stream_count"] == dynamic["copy_stream_count"]
            and value.get("host_registration") == dynamic.get("host_registration")
            and _valid_text_host_registration(
                value.get("host_registration"),
                lifecycle="residency_stage_through_synchronized_close",
                allow_idle=dynamic.get("copy_strategy") == "gathered_host_buffer",
            )
        )
    return bool(
        value.get("mode") == "layer_streamed_cpu_master"
        and value.get("leaf_scheduler") is None
        and value.get("root_activation") == "stage_onload"
        and value.get("transfer_mode") == "two_stream_nonblocking"
        and _valid_text_host_registration(value.get("host_registration"))
        and _valid_text_dynamic_fallback(value.get("dynamic_vram"))
        and value.get("dynamic_vbar_prefetch") is False
        and value["root_transitions"] == 1
        and value["transfer_events"] == value["root_transitions"] + value["layer_transitions"]
    )


def _valid_text_forward_accounting(
    value: Mapping[str, Any], lora_to_base_transitions: int | None
) -> bool:
    """Validate variable-length autoregressive Gemma execution exactly."""

    root_transitions = value["root_transitions"]
    minimum_root_transitions = 3 if lora_to_base_transitions == 1 else 2
    # Enhancement is autoregressive: each generated token executes another
    # complete Gemma forward before positive and negative conditioning.  The
    # token count varies, but every root forward must traverse all layers and
    # produce the same number of full-precision Kitchen dispatches.
    return bool(
        root_transitions >= minimum_root_transitions
        and value["layer_transitions"] == root_transitions * value["layer_count"]
        and value["full_precision_dispatches"] % root_transitions == 0
    )


def _valid_text_dynamic_fallback(value: object) -> bool:
    expected = {
        "backend",
        "policy",
        "fallback_reason",
        "prefetch",
        "allocator_plugin",
        "base_file_requested",
        "base_file_backed",
        "base_file_read_calls",
        "base_file_read_bytes",
        "base_file_handle_live",
        "base_file_handle_opened",
        "base_file_handle_closed",
        "base_file_fallback_reason",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    base_requested = value.get("base_file_requested")
    base_fallback = value.get("base_file_fallback_reason")
    return bool(
        value.get("backend") == "engine_hooks"
        and value.get("policy") in {"auto", "hooks"}
        and (
            value.get("fallback_reason") is None
            or isinstance(value.get("fallback_reason"), str)
            and 0 < len(value["fallback_reason"]) <= 512
        )
        and (value.get("fallback_reason") is not None if value.get("policy") == "auto" else True)
        and value.get("prefetch") is False
        and value.get("allocator_plugin") is False
        and isinstance(base_requested, bool)
        and value.get("base_file_backed") is False
        and value.get("base_file_read_calls") == 0
        and value.get("base_file_read_bytes") == 0
        and value.get("base_file_handle_live") is False
        and value.get("base_file_handle_opened") == 0
        and value.get("base_file_handle_closed") == 0
        and (
            _valid_base_file_fallback_reason(base_fallback)
            if base_requested
            else base_fallback is None
        )
    )


def _valid_dynamic_text_residency(
    value: object,
    layer_count: int,
    *,
    root_transitions: int,
    layer_transitions: int,
    lora_entry_transitions: int | None,
    lora_to_base_transitions: int | None,
    leaf_allocation_count: int,
    force_resident_leaf_count: int,
    base_leaf_count: int,
    patch_leaf_count: int,
    scheduler: object,
    warm_request_index: int,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    integer_keys = {
        "physical_bytes",
        "staged_bytes",
        "virtual_bytes",
        "allocation_count",
        "live_allocations",
        "live_bytes",
        "loaded_bytes",
        "faults",
        "signature_hits",
        "signature_misses",
        "fault_none_temporaries",
        "pinned_copy_bytes",
        "pageable_copy_bytes",
        "transfer_events",
        "transfer_waits",
        "prioritize_calls",
        "unpin_calls",
        "free_calls",
        "dirty_epoch",
        "lora_invalidations",
        "base_restores",
        "copy_stream_count",
        "host_buffer_capacity_bytes",
        "host_buffer_allocations",
        "host_buffer_unregistrations",
        "host_buffer_frees",
        "gathered_misses",
        "per_physical_misses",
        "packed_source_bytes",
        "gathered_h2d_bytes",
        "pressure_direct_transfers",
        "pressure_direct_bytes",
        "host_buffer_reuse_barriers",
        "host_source_pool_generation",
        "host_source_pool_lane_count",
        "host_source_pool_capacity_bytes",
        "host_source_pool_retained_slices",
        "host_source_pool_retained_bytes",
        "host_source_pool_temporary_slices",
        "host_source_pool_temporary_bytes",
        "host_source_pool_hits",
        "host_source_pool_misses",
        "host_source_pool_stale_rejections",
        "host_source_pool_warm_ram_pressure_bypasses",
        "host_source_pool_warm_zero_delta_extend_refusals",
        "host_source_pool_warm_registration_refusals",
        "host_source_pool_temporary_ram_pressure_bypasses",
        "host_source_pool_temporary_zero_delta_extend_refusals",
        "host_source_pool_temporary_registration_refusals",
        "base_file_read_calls",
        "base_file_read_bytes",
        "base_file_handle_opened",
        "base_file_handle_closed",
        "prefetch_calls",
    }
    for optional_fields in (
        _AIMDO_STREAM_RETIREMENT_INTEGER_FIELDS,
        _AIMDO_STAGE_PREPARE_INTEGER_FIELDS,
    ):
        present = set(value) & optional_fields
        if present and present != optional_fields:
            return False
        integer_keys |= present
    expected = integer_keys | {
        "backend",
        "version",
        "mode",
        "prefetch",
        "allocator_plugin",
        "poisoned",
        "close_failed",
        "poison_reason",
        "copy_strategy",
        "copy_fallback_reason",
        "gathered_host_buffer_requested",
        "host_buffer_live",
        "host_tensor_view_live",
        "host_buffer_transfer_pending",
        "host_source_pool_poisoned",
        "host_source_pool_poison_reason",
        "host_source_registration",
        "base_file_backed",
        "base_file_source_live",
        "base_file_handle_live",
        "base_file_fallback_reason",
        "host_registration",
        "policy",
        "request_delta",
        "warm_request_index",
    }
    if (
        set(value) != expected
        or value.get("backend") != "comfy-aimdo"
        or value.get("version") != "0.4.15"
        or value.get("mode") != "dynamic_vbar"
        or value.get("policy") not in {"auto", "required"}
        or value.get("prefetch") is not False
        or value.get("allocator_plugin") is not False
        or value.get("poisoned") is not False
        or value.get("close_failed") is not False
        or value.get("poison_reason") is not None
        or value.get("host_source_pool_poisoned") is not False
        or value.get("host_source_pool_poison_reason") is not None
        or value.get("warm_request_index") != warm_request_index
        or not _valid_text_dynamic_request_delta(value.get("request_delta"))
        or any(
            not isinstance(value.get(key), int)
            or isinstance(value.get(key), bool)
            or value[key] < 0
            for key in integer_keys
        )
        or not _valid_text_host_registration(
            value.get("host_registration"),
            lifecycle="residency_stage_through_synchronized_close",
            allow_idle=value.get("copy_strategy") == "gathered_host_buffer",
        )
    ):
        return False
    request = value["request_delta"]
    return bool(
        value["physical_bytes"] > 0
        and value["physical_bytes"] <= value["staged_bytes"] <= value["virtual_bytes"]
        and leaf_allocation_count == value["allocation_count"] > layer_count + 1
        and 0 < force_resident_leaf_count <= leaf_allocation_count
        and base_leaf_count + patch_leaf_count == leaf_allocation_count
        and base_leaf_count > layer_count + 1
        and (patch_leaf_count > 0 if lora_to_base_transitions else True)
        and _valid_text_leaf_scheduler(
            scheduler,
            allocation_count=leaf_allocation_count,
            force_resident_count=force_resident_leaf_count,
            layer_count=layer_count,
            faults=request["faults"],
        )
        and value["live_allocations"] == value["allocation_count"]
        and value["live_bytes"] == value["virtual_bytes"]
        # AIMDO reports loaded physical pages, while live_bytes is the logical
        # VBAR size.  The pinned 0.4.15 ABI uses 32 MiB VBAR pages, so a fully
        # resident mapping is rounded up to the next native page boundary.
        and value["loaded_bytes"]
        <= math.ceil(value["live_bytes"] / _AIMDO_VBAR_PAGE_BYTES) * _AIMDO_VBAR_PAGE_BYTES
        and request["faults"] >= root_transitions + layer_transitions
        and value["faults"] == value["signature_hits"] + value["signature_misses"]
        and request["faults"] == request["signature_hits"] + request["signature_misses"]
        and (request["signature_misses"] > 0 if warm_request_index == 1 else True)
        and request["fault_none_temporaries"] <= request["signature_misses"]
        and value["pinned_copy_bytes"] + value["pressure_direct_bytes"] >= value["physical_bytes"]
        and value["pressure_direct_transfers"]
        == value["host_source_pool_warm_ram_pressure_bypasses"]
        + value["host_source_pool_warm_zero_delta_extend_refusals"]
        + value["host_source_pool_warm_registration_refusals"]
        + value["host_source_pool_temporary_ram_pressure_bypasses"]
        + value["host_source_pool_temporary_zero_delta_extend_refusals"]
        + value["host_source_pool_temporary_registration_refusals"]
        and value["transfer_events"] == value["transfer_waits"]
        and value["transfer_events"] >= value["signature_misses"]
        and request["transfer_events"] == request["transfer_waits"]
        and request["transfer_events"] >= request["signature_misses"]
        and value["prioritize_calls"] == 1
        and request["unpin_calls"] == request["faults"] - request["fault_none_temporaries"]
        and value["free_calls"] == 0
        and value["lora_invalidations"] == value["base_restores"]
        and value["dirty_epoch"] >= value["lora_invalidations"]
        and value["dirty_epoch"] >= request["dirty_epoch"]
        and request["lora_invalidations"] == lora_to_base_transitions
        and request["base_restores"] == lora_to_base_transitions
        and (
            request["dirty_epoch"] in {lora_to_base_transitions, lora_to_base_transitions + 1}
            if lora_entry_transitions is None
            else request["dirty_epoch"] == lora_entry_transitions + lora_to_base_transitions
        )
        and value["copy_stream_count"] == 2
        and value["prefetch_calls"] == 0
        and request["prefetch_calls"] == 0
        and request["host_source_pool_hits"]
        + request["host_source_pool_misses"]
        + request["pressure_direct_transfers"]
        == request["gathered_misses"]
        and request["base_file_read_bytes"] <= request["packed_source_bytes"]
        and request["pressure_direct_transfers"] <= request["gathered_misses"]
        and request["pressure_direct_bytes"] <= request["gathered_h2d_bytes"]
        and (request["pressure_direct_transfers"] == 0) == (request["pressure_direct_bytes"] == 0)
        and request["pageable_copy_bytes"] <= request["pressure_direct_bytes"]
        and request["pressure_direct_transfers"]
        == request["host_source_pool_warm_ram_pressure_bypasses"]
        + request["host_source_pool_warm_zero_delta_extend_refusals"]
        + request["host_source_pool_warm_registration_refusals"]
        + request["host_source_pool_temporary_ram_pressure_bypasses"]
        + request["host_source_pool_temporary_zero_delta_extend_refusals"]
        + request["host_source_pool_temporary_registration_refusals"]
        and _valid_dynamic_copy_proof(value)
        and _valid_host_source_registration(
            value.get("host_source_registration"),
            source_misses=value["host_source_pool_misses"],
            capacity_bytes=value["host_source_pool_capacity_bytes"],
            expect_closed=False,
            pool_poison_reason=value.get("host_source_pool_poison_reason"),
        )
        and _valid_success_base_file_proof(value)
    )


_TEXT_DYNAMIC_REQUEST_DELTA_KEYS = frozenset(
    {
        "faults",
        "signature_hits",
        "signature_misses",
        "fault_none_temporaries",
        "pinned_copy_bytes",
        "pageable_copy_bytes",
        "transfer_events",
        "transfer_waits",
        "unpin_calls",
        "dirty_epoch",
        "lora_invalidations",
        "base_restores",
        "gathered_misses",
        "per_physical_misses",
        "packed_source_bytes",
        "gathered_h2d_bytes",
        "pressure_direct_transfers",
        "pressure_direct_bytes",
        "host_buffer_reuse_barriers",
        "host_source_pool_hits",
        "host_source_pool_misses",
        "host_source_pool_stale_rejections",
        "host_source_pool_warm_ram_pressure_bypasses",
        "host_source_pool_warm_zero_delta_extend_refusals",
        "host_source_pool_warm_registration_refusals",
        "host_source_pool_temporary_ram_pressure_bypasses",
        "host_source_pool_temporary_zero_delta_extend_refusals",
        "host_source_pool_temporary_registration_refusals",
        "base_file_read_calls",
        "base_file_read_bytes",
        "prefetch_calls",
    }
)


def _valid_text_dynamic_request_delta(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == _TEXT_DYNAMIC_REQUEST_DELTA_KEYS
        and all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and value[key] >= 0
            for key in _TEXT_DYNAMIC_REQUEST_DELTA_KEYS
        )
    )


def _valid_text_leaf_scheduler(
    value: object,
    *,
    allocation_count: int,
    force_resident_count: int,
    layer_count: int,
    faults: int,
) -> bool:
    expected = {
        "leaf_allocation_count",
        "force_resident_leaf_count",
        "schedule_group_count",
        "prefetch_groups",
        "prefetch_leaves",
        "deferred_waits",
        "force_resident_waits",
        "consumed_groups",
        "active_groups",
        "pending_prefetch",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    integer_keys = expected - {"pending_prefetch"}
    return bool(
        all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and value[key] >= 0
            for key in integer_keys
        )
        and value["leaf_allocation_count"] == allocation_count
        and value["force_resident_leaf_count"] == force_resident_count
        and value["schedule_group_count"] == 2 * (layer_count + 1)
        and value["prefetch_groups"] == 0
        and value["prefetch_leaves"] == 0
        and value["force_resident_waits"] >= force_resident_count
        and value["force_resident_waits"] % force_resident_count == 0
        and value["deferred_waits"] + value["force_resident_waits"] == faults
        and value["consumed_groups"] > 0
        and value["active_groups"] == 0
        and value["pending_prefetch"] is False
    )


_HOST_SOURCE_REGISTRATION_KEYS = frozenset(
    {
        "policy",
        "budget_bytes",
        "attempts",
        "attempt_bytes",
        "successes",
        "failures",
        "failure_bytes",
        "registered_bytes",
        "unregistered_bytes",
        "live_bytes",
        "peak_bytes",
        "state_proven",
    }
)


def _valid_host_source_registration(
    value: object,
    *,
    source_misses: int,
    capacity_bytes: int,
    expect_closed: bool,
    allow_poisoned: bool = False,
    pool_poison_reason: object = None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != _HOST_SOURCE_REGISTRATION_KEYS:
        return False
    integer_fields = _HOST_SOURCE_REGISTRATION_KEYS - {"policy", "state_proven"}
    if (
        value.get("policy") != "aimdo_hostbuffer_registered_append"
        or not isinstance(value.get("state_proven"), bool)
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
            for field in integer_fields
        )
    ):
        return False
    proven = value["state_proven"] is True
    return bool(
        (proven or allow_poisoned)
        and value["attempts"] == value["successes"] + value["failures"]
        and value["attempt_bytes"] == value["registered_bytes"] + value["failure_bytes"]
        and source_misses <= value["successes"] <= value["attempts"]
        and (allow_poisoned or value["successes"] == source_misses)
        and value["successes"] - source_misses <= 1
        and (
            value["successes"] == source_misses
            or pool_poison_reason
            in {"host_buffer_view_validation_failed", "host_buffer_rollback_failed"}
        )
        and value["unregistered_bytes"] <= value["registered_bytes"]
        and value["live_bytes"] <= value["peak_bytes"] <= capacity_bytes
        and value["peak_bytes"] <= value["budget_bytes"]
        and (
            not proven
            or value["live_bytes"] == value["registered_bytes"] - value["unregistered_bytes"]
        )
        and (
            not expect_closed
            or value["live_bytes"] == 0
            and value["unregistered_bytes"] == value["registered_bytes"]
        )
    )


def _valid_dynamic_copy_proof(value: Mapping[str, Any]) -> bool:
    strategy = value.get("copy_strategy")
    fallback = value.get("copy_fallback_reason")
    common = bool(
        value.get("gathered_host_buffer_requested") is True
        and strategy in {"gathered_host_buffer", "per_physical"}
        and (fallback is None or _valid_host_buffer_fallback_reason(fallback))
        and value["host_buffer_allocations"] <= 4
        and value["gathered_misses"] + value["per_physical_misses"] == value["signature_misses"]
        and value["pressure_direct_transfers"] <= value["gathered_misses"]
        and value["pressure_direct_bytes"] <= value["gathered_h2d_bytes"]
        and (value["pressure_direct_transfers"] == 0) == (value["pressure_direct_bytes"] == 0)
        and value["pressure_direct_transfers"]
        == value["host_source_pool_warm_ram_pressure_bypasses"]
        + value["host_source_pool_warm_zero_delta_extend_refusals"]
        + value["host_source_pool_warm_registration_refusals"]
        + value["host_source_pool_temporary_ram_pressure_bypasses"]
        + value["host_source_pool_temporary_zero_delta_extend_refusals"]
        + value["host_source_pool_temporary_registration_refusals"]
    )
    if not common:
        return False
    if strategy == "gathered_host_buffer":
        return bool(
            fallback is None
            and value.get("host_buffer_live") is True
            and value.get("host_tensor_view_live") is True
            and value.get("host_buffer_transfer_pending") is False
            and 1 <= value["host_buffer_allocations"] <= 4
            and value["host_buffer_unregistrations"] == 0
            and value["host_buffer_frees"] == 0
            and value["host_source_pool_lane_count"] == value["host_buffer_allocations"]
            and value["host_source_pool_generation"] >= 1
            and value["host_source_pool_capacity_bytes"] >= value["host_buffer_capacity_bytes"]
            and value["host_source_pool_capacity_bytes"] <= 2 * value["virtual_bytes"]
            and value["host_source_pool_hits"]
            + value["host_source_pool_misses"]
            + value["pressure_direct_transfers"]
            == value["gathered_misses"]
            and value["host_source_pool_stale_rejections"] == 0
            and (
                value["pressure_direct_transfers"] == value["gathered_misses"]
                or value["host_source_pool_retained_slices"] > 0
                and value["host_source_pool_retained_bytes"] > 0
            )
            and value["host_source_pool_temporary_slices"] == 0
            and value["host_source_pool_temporary_bytes"] == 0
            and 0 < value["host_buffer_capacity_bytes"] <= value["virtual_bytes"]
            and value["gathered_misses"] == value["signature_misses"]
            and value["per_physical_misses"] == 0
            and value["transfer_events"] == value["signature_misses"]
            and value["pageable_copy_bytes"] <= value["pressure_direct_bytes"]
            and value["pinned_copy_bytes"] + value["pressure_direct_bytes"]
            == value["gathered_h2d_bytes"]
            and value["packed_source_bytes"] >= value["physical_bytes"]
            and value["gathered_h2d_bytes"] >= value["packed_source_bytes"]
            and value["gathered_h2d_bytes"]
            <= value["host_buffer_capacity_bytes"] * value["gathered_misses"]
            and value["host_buffer_reuse_barriers"] == 0
            and _idle_text_host_registration(value.get("host_registration"))
        )
    return bool(
        _valid_host_buffer_fallback_reason(fallback)
        and value.get("host_buffer_live") is False
        and value.get("host_tensor_view_live") is False
        and value.get("host_buffer_transfer_pending") is False
        and value["host_buffer_allocations"] == 0
        and value["host_buffer_unregistrations"] == 0
        and value["host_buffer_frees"] == 0
        and value["gathered_misses"] == 0
        and value["per_physical_misses"] == value["signature_misses"]
        and value["packed_source_bytes"] == 0
        and value["gathered_h2d_bytes"] == 0
        and value["pressure_direct_transfers"] == 0
        and value["pressure_direct_bytes"] == 0
        and value["host_source_pool_warm_ram_pressure_bypasses"] == 0
        and value["host_source_pool_warm_zero_delta_extend_refusals"] == 0
        and value["host_source_pool_warm_registration_refusals"] == 0
        and value["host_source_pool_temporary_ram_pressure_bypasses"] == 0
        and value["host_source_pool_temporary_zero_delta_extend_refusals"] == 0
        and value["host_source_pool_temporary_registration_refusals"] == 0
        and value["host_buffer_reuse_barriers"] == 0
        and _valid_clean_host_source_pool_fallback(value)
    )


def _valid_clean_host_source_pool_fallback(value: Mapping[str, Any]) -> bool:
    allocations = value["host_buffer_allocations"]
    common = bool(
        value.get("host_source_pool_poisoned") is False
        and value.get("host_source_pool_poison_reason") is None
        and value["host_source_pool_retained_slices"] == 0
        and value["host_source_pool_retained_bytes"] == 0
        and value["host_source_pool_temporary_slices"] == 0
        and value["host_source_pool_temporary_bytes"] == 0
        and value["host_source_pool_hits"] == 0
        and value["host_source_pool_misses"] == 0
        and value["host_source_pool_stale_rejections"] == 0
    )
    if not common:
        return False
    if allocations == 0:
        return bool(
            value["host_buffer_unregistrations"] == 0
            and value["host_buffer_frees"] == 0
            and value["host_source_pool_generation"] == 0
            and value["host_source_pool_lane_count"] == 0
            and value["host_source_pool_capacity_bytes"] == 0
        )
    return bool(
        1 <= allocations <= 4
        and 0 <= value["host_buffer_unregistrations"] <= allocations
        and value["host_buffer_frees"] == allocations
        and value["host_source_pool_generation"] >= 2
        and value["host_source_pool_lane_count"] == allocations
        and value["host_source_pool_capacity_bytes"] > 0
    )


def _valid_success_base_file_proof(value: Mapping[str, Any]) -> bool:
    if value.get("copy_strategy") == "gathered_host_buffer":
        return bool(
            value.get("base_file_backed") is True
            and value.get("base_file_source_live") is True
            and value.get("base_file_handle_live") is True
            and value.get("base_file_fallback_reason") is None
            and value["base_file_read_calls"] > 0
            and 0 < value["base_file_read_bytes"] <= value["packed_source_bytes"]
            and value["base_file_handle_opened"] == 1
            and value["base_file_handle_closed"] == 0
        )
    return bool(
        value.get("base_file_backed") is False
        and value.get("base_file_source_live") is False
        and value.get("base_file_handle_live") is False
        and value.get("base_file_fallback_reason") is None
        and value["base_file_read_calls"] == 0
        and value["base_file_read_bytes"] == 0
        and value["base_file_handle_opened"] == 0
        and value["base_file_handle_closed"] == 0
    )


_HOST_BUFFER_FALLBACK_PREFIXES = (
    "host_buffer_capability_unavailable:",
    "host_buffer_setup_failed:",
)

_BASE_FILE_FALLBACK_PREFIXES = (
    "aimdo_backend_unavailable:",
    "aimdo_policy_or_device_unavailable:",
    *_HOST_BUFFER_FALLBACK_PREFIXES,
)


def _valid_host_buffer_fallback_reason(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= 512
        and value.startswith(_HOST_BUFFER_FALLBACK_PREFIXES)
    )


def _valid_base_file_fallback_reason(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= 512
        and value.startswith(_BASE_FILE_FALLBACK_PREFIXES)
    )


def _idle_text_host_registration(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    ignored = {"policy", "lifecycle", "budget_bytes", "categories"}
    counters = (field for field in value if field not in ignored)
    categories = value.get("categories")
    return bool(
        all(value[field] == 0 for field in counters)
        and isinstance(categories, Mapping)
        and all(count == 0 for count in categories.values())
    )


def _valid_text_host_registration(
    value: object,
    *,
    lifecycle: str = "text_stage_onload_through_synchronized_offload",
    allow_idle: bool = False,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    counter_keys = {
        "candidates",
        "candidate_bytes",
        "deduplicated_aliases",
        "already_registered",
        "already_registered_bytes",
        "attempts",
        "attempt_bytes",
        "successes",
        "registered_bytes",
        "failures",
        "failure_bytes",
        "ineligible",
        "ineligible_bytes",
        "unregistered",
        "unregistered_bytes",
        "unregister_failures",
        "unregister_failure_bytes",
        "owned_active",
        "owned_active_bytes",
    }
    category_keys = {
        "unsupported_type",
        "non_cpu",
        "noncontiguous",
        "zero_pointer",
        "budget_exceeded",
        "eligibility_error",
        "register_error",
        "unregister_error",
    }
    categories = value.get("categories")
    expected_keys = counter_keys | {"policy", "lifecycle", "budget_bytes", "categories"}
    if (
        set(value) != expected_keys
        or value.get("policy") != "comfy_best_effort_in_place_cuda_host_register"
        or value.get("lifecycle") != lifecycle
        or isinstance(value.get("budget_bytes"), bool)
        or not isinstance(value.get("budget_bytes"), int)
        or value["budget_bytes"] <= 0
        or not isinstance(categories, Mapping)
        or set(categories) != category_keys
        or any(
            isinstance(value.get(key), bool)
            or not isinstance(value.get(key), int)
            or value[key] < 0
            for key in counter_keys
        )
        or any(
            isinstance(categories.get(key), bool)
            or not isinstance(categories.get(key), int)
            or categories[key] < 0
            for key in category_keys
        )
    ):
        return False
    unique_candidates = value["candidates"] - value["deduplicated_aliases"]
    ineligible_categories = category_keys - {"register_error", "unregister_error"}
    if allow_idle and _idle_text_host_registration(value):
        return True
    return bool(
        unique_candidates >= 0
        and value["candidates"] > 0
        and unique_candidates
        == value["already_registered"] + value["attempts"] + value["ineligible"]
        and value["successes"] + value["failures"] == value["attempts"]
        and value["registered_bytes"] + value["failure_bytes"] == value["attempt_bytes"]
        and value["candidate_bytes"]
        >= value["attempt_bytes"] + value["ineligible_bytes"] + value["already_registered_bytes"]
        and value["registered_bytes"] <= value["budget_bytes"]
        and (value["successes"] == 0) == (value["registered_bytes"] == 0)
        and (value["failures"] == 0) == (value["failure_bytes"] == 0)
        and (value["already_registered"] == 0) == (value["already_registered_bytes"] == 0)
        and value["owned_active"] == 0
        and value["owned_active_bytes"] == 0
        and value["unregistered"] == value["successes"]
        and value["unregistered_bytes"] == value["registered_bytes"]
        and value["unregister_failures"] == 0
        and value["unregister_failure_bytes"] == 0
        and sum(categories[key] for key in ineligible_categories) == value["ineligible"]
        and categories["register_error"] == value["failures"]
        and categories["unregister_error"] == value["unregister_failures"]
    )


def _valid_timings(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    phases = value.get("phases")
    cumulative = value.get("cumulative")
    if (
        value.get("clock") != "time.perf_counter"
        or value.get("unit") != "seconds"
        or not isinstance(phases, Mapping)
        or not isinstance(cumulative, Mapping)
        or set(phases) != set(_TEXT_TIMING_PHASES)
        or set(cumulative) != set(_TEXT_TIMING_PHASES)
    ):
        return False
    values: list[float] = []
    for phase in _TEXT_TIMING_PHASES:
        duration = phases[phase]
        total = cumulative[phase]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
            or isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(float(total))
            or total < 0
        ):
            return False
        values.append(float(total))
    total_seconds = value.get("total_seconds")
    return bool(
        all(right >= left for left, right in pairwise(values))
        and not isinstance(total_seconds, bool)
        and isinstance(total_seconds, (int, float))
        and math.isfinite(float(total_seconds))
        and float(total_seconds) >= values[-1]
    )


def _valid_memory_telemetry(value: object, operation: str) -> bool:
    expected_phases = (
        _SINGLE_STAGE_MEMORY_PHASES
        if operation == "ltx23_distilled_flf"
        else _TWO_STAGE_MEMORY_PHASES
        if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}
        else None
    )
    if (
        expected_phases is None
        or not isinstance(value, Mapping)
        or set(value) != {"schema_version", "timestamp_clock", "elapsed_clock", "samples"}
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("timestamp_clock") != "time.time_ns"
        or value.get("elapsed_clock") != "time.perf_counter_ns"
        or not isinstance(value.get("samples"), list)
        or len(value["samples"]) != len(expected_phases)
    ):
        return False
    prior_elapsed = -1
    for sequence, (phase, sample) in enumerate(zip(expected_phases, value["samples"], strict=True)):
        if (
            not isinstance(sample, Mapping)
            or set(sample)
            != {
                "sequence",
                "phase",
                "timestamp_unix_ns",
                "elapsed_ns",
                "process",
                "system",
                "cuda",
            }
            or type(sample.get("sequence")) is not int
            or sample.get("sequence") != sequence
            or sample.get("phase") != phase
            or isinstance(sample.get("timestamp_unix_ns"), bool)
            or not isinstance(sample.get("timestamp_unix_ns"), int)
            or sample["timestamp_unix_ns"] <= 0
            or isinstance(sample.get("elapsed_ns"), bool)
            or not isinstance(sample.get("elapsed_ns"), int)
            or sample["elapsed_ns"] < prior_elapsed
            or not _valid_process_memory_observation(sample.get("process"))
            or not _valid_system_memory_observation(sample.get("system"))
            or not _valid_cuda_memory_observation(sample.get("cuda"))
        ):
            return False
        prior_elapsed = sample["elapsed_ns"]
    return True


def _valid_observation_error(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= 128
        and value.isidentifier()
        and value.isascii()
    )


def _valid_process_memory_observation(value: object) -> bool:
    fields = {"pid", "private_bytes", "working_set_bytes"}
    if not isinstance(value, Mapping) or set(value) != {"status", "error", *fields}:
        return False
    if value.get("status") == "error":
        return _valid_observation_error(value.get("error")) and all(
            value.get(field) is None for field in fields
        )
    return bool(
        value.get("status") == "ok"
        and value.get("error") is None
        and all(
            not isinstance(value.get(field), bool) and isinstance(value.get(field), int)
            for field in fields
        )
        and value["pid"] > 0
        and value["private_bytes"] > 0
        and value["working_set_bytes"] > 0
    )


def _valid_system_memory_observation(value: object) -> bool:
    fields = {
        "total_physical_bytes",
        "free_physical_bytes",
        "used_physical_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != {"status", "error", *fields}:
        return False
    if value.get("status") == "error":
        return _valid_observation_error(value.get("error")) and all(
            value.get(field) is None for field in fields
        )
    return bool(
        value.get("status") == "ok"
        and value.get("error") is None
        and all(
            not isinstance(value.get(field), bool) and isinstance(value.get(field), int)
            for field in fields
        )
        and value["total_physical_bytes"] > 0
        and 0 <= value["free_physical_bytes"] <= value["total_physical_bytes"]
        and value["used_physical_bytes"] >= 0
        and value["free_physical_bytes"] + value["used_physical_bytes"]
        == value["total_physical_bytes"]
    )


def _valid_cuda_memory_observation(value: object) -> bool:
    fields = {"allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes"}
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status", "error", "device", *fields}
        or not isinstance(value.get("device"), str)
        or not (
            value["device"] == "cuda"
            or value["device"].startswith("cuda:")
            and value["device"][5:].isdigit()
        )
    ):
        return False
    if value.get("status") == "error":
        return _valid_observation_error(value.get("error")) and all(
            value.get(field) is None for field in fields
        )
    return bool(
        value.get("status") == "ok"
        and value.get("error") is None
        and value["device"].startswith("cuda:")
        and all(
            not isinstance(value.get(field), bool) and isinstance(value.get(field), int)
            for field in fields
        )
        and 0 <= value["allocated_bytes"] <= value["reserved_bytes"] <= value["total_bytes"]
        and 0 <= value["free_bytes"] <= value["total_bytes"]
        and value["total_bytes"] > 0
    )


def _valid_cache_status(
    value: object,
    *,
    expected_policy: str,
    expected_pipeline_warm: bool | None = None,
    expected_prompt_hit: bool | None = None,
    require_empty: bool = False,
) -> bool:
    cache_keys = {
        "pipeline_warm",
        "policy",
        "prompt_hit",
        "prompt_published",
        "media_hit",
        "prompt",
    }
    prompt_keys = {
        "name",
        "enabled",
        "entries",
        "bytes",
        "max_bytes",
        "max_entries",
        "hits",
        "misses",
        "evictions",
        "hit_rate",
    }
    if not isinstance(value, Mapping) or set(value) != cache_keys:
        return False
    pipeline_warm = value.get("pipeline_warm")
    prompt_hit = value.get("prompt_hit")
    prompt_published = value.get("prompt_published")
    prompt = value.get("prompt")
    if (
        expected_policy not in {"none", "prompt"}
        or value.get("policy") != expected_policy
        or not isinstance(pipeline_warm, bool)
        or not isinstance(prompt_hit, bool)
        or not isinstance(prompt_published, bool)
        or value.get("media_hit") is not False
        or not isinstance(prompt, Mapping)
        or set(prompt) != prompt_keys
        or prompt.get("name") != "prompt"
        or not isinstance(prompt.get("enabled"), bool)
        or prompt.get("enabled") != (expected_policy == "prompt")
        or prompt.get("max_bytes") != _PROMPT_CACHE_MAX_BYTES
        or prompt.get("max_entries") != _PROMPT_CACHE_MAX_ENTRIES
    ):
        return False
    counter_keys = ("entries", "bytes", "hits", "misses", "evictions")
    if any(
        isinstance(prompt.get(key), bool) or not isinstance(prompt.get(key), int) or prompt[key] < 0
        for key in counter_keys
    ):
        return False
    if prompt["entries"] > _PROMPT_CACHE_MAX_ENTRIES or prompt["bytes"] > _PROMPT_CACHE_MAX_BYTES:
        return False
    requests = prompt["hits"] + prompt["misses"]
    hit_rate = prompt["hit_rate"]
    if requests == 0:
        if hit_rate is not None:
            return False
    elif (
        isinstance(hit_rate, bool)
        or not isinstance(hit_rate, (int, float))
        or not math.isfinite(float(hit_rate))
        or not math.isclose(
            float(hit_rate),
            prompt["hits"] / requests,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        return False
    if prompt_hit:
        if prompt_published or prompt["hits"] <= 0:
            return False
    elif prompt["misses"] <= 0:
        return False
    if prompt_published and (
        expected_policy != "prompt" or prompt["entries"] <= 0 or prompt["bytes"] <= 0
    ):
        return False
    if prompt_hit and (prompt["entries"] <= 0 or prompt["bytes"] <= 0):
        return False
    if expected_policy == "none" and any(
        prompt[key] != 0 for key in ("entries", "bytes", "hits", "evictions")
    ):
        return False
    if expected_policy == "none" and prompt_published:
        return False
    if expected_pipeline_warm is not None and pipeline_warm is not expected_pipeline_warm:
        return False
    if expected_prompt_hit is not None and prompt_hit is not expected_prompt_hit:
        return False
    if require_empty:
        return not prompt_published and prompt["entries"] == 0 and prompt["bytes"] == 0
    return expected_policy == "none" or prompt_hit or prompt_published


def _valid_text_lora_proof(value: object, *, expected_active: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("provenance") == "cached_prompt_conditioning":
        return value.get("dispatch_performed") is False and _valid_text_lora_proof(
            value.get("source_proof"), expected_active=expected_active
        )
    return bool(
        value.get("backend") == "engine-native/additive-lora"
        and value.get("policy")
        == ("prompt_enhancement_only" if expected_active else "installed_inactive_base_encode")
        and isinstance(value.get("target_module_count"), int)
        and value["target_module_count"] > 0
        and isinstance(value.get("total_dispatches"), int)
        and (value["total_dispatches"] > 0) is expected_active
        and isinstance(value.get("minimum_target_dispatches"), int)
        and (value["minimum_target_dispatches"] > 0) is expected_active
        and isinstance(value.get("maximum_target_dispatches"), int)
        and (value["maximum_target_dispatches"] > 0) is expected_active
    )


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
        raise RuntimeError("LTX 2.3 Kitchen worker exited without a bounded result") from exc
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
        "LTX 2.3 Kitchen worker progress: operation=%s phase=%s progress=%.3f elapsed_seconds=%.3f",
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
        ("Materialized LTX components", "materialization_complete"),
        ("Reusing warmed LTX components", "materialization_reused"),
        ("Preparing streamed LTX text encoder", "prepare_text_streaming"),
        ("Prepared streamed LTX text encoder", "prepare_text_streaming_complete"),
        ("LTX AIMDO ", "aimdo_text_residency"),
        ("Enhancing prompt", "enhance_prompt"),
        ("Encoding positive prompt", "encode_positive_prompt"),
        ("Encoding negative prompt", "encode_negative_prompt"),
        ("Offloaded base Gemma", "offload_text"),
        ("LTX denoise step", "denoise"),
        ("Upscaling LTX video latents", "upscale_latents"),
        ("Decoding LTX video and audio", "decode_media"),
        ("Muxing 25 fps", "mux_output"),
        ("Completed LTX downstream phases", "downstream_complete"),
        ("Published LTX prompt cache", "prompt_cache_publish"),
        ("Skipped LTX prompt cache publication", "prompt_cache_reused"),
        ("LTX 2.3 output ready", "verify_output"),
    )
    return next((phase for prefix, phase in prefixes if message.startswith(prefix)), "working")


def _worker_error(path: Path, exit_code: int, binding: str, secret: bytes) -> str:
    """Return the public message for one already-sanitized worker failure."""

    return _worker_failure(path, exit_code, binding, secret)["message"]


def _worker_failure(path: Path, exit_code: int, binding: str, secret: bytes) -> dict[str, Any]:
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


def _worker_failure_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Render one already-authenticated, exact-schema child failure."""

    result = {
        "message": (
            f"LTX 2.3 Kitchen worker failed ({value['error_type']} during "
            f"{value['failure_stage']} at {value['failure_location']}; "
            f"diagnostic {value['error_fingerprint'][:12]})"
        ),
        "error_type": value["error_type"],
        "stage": value["failure_stage"],
        "location": value["failure_location"],
        "fingerprint": value["error_fingerprint"],
        "cleanup_stage": value["cleanup_stage"],
        "aimdo_counters": value["aimdo_counters"],
    }
    if "poison_reason" in value:
        result["poison_reason"] = value["poison_reason"]
        result["poison_origin"] = value["poison_origin"]
    return result


def _canonical_worker_poison(
    failure: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return only terminal poison proof already bound to its primary origin."""

    reason = failure.get("poison_reason")
    origin = failure.get("poison_origin")
    error_type = failure.get("error_type")
    cleanup_stage = failure.get("cleanup_stage")
    if reason not in _AIMDO_POISON_REASONS:
        return None
    if origin == "primary":
        return (
            (reason, origin)
            if error_type == "LTX23KitchenWorkerPoisoned" and cleanup_stage != "unload_runtime"
            else None
        )
    if origin == "cleanup":
        return (
            (reason, origin)
            if error_type != "LTX23KitchenWorkerPoisoned" and cleanup_stage == "unload_runtime"
            else None
        )
    return None


def _log_worker_failure(failure: Mapping[str, Any]) -> None:
    """Log only categorized proof; authenticated IPC retains raw child detail."""

    poison = _canonical_worker_poison(failure)
    _LOGGER.error(
        "LTX 2.3 Kitchen child failure: type=%s stage=%s location=%s "
        "cleanup_stage=%s diagnostic=%s poison_reason=%s poison_origin=%s "
        "detail=withheld_authenticated_ipc",
        failure.get("error_type", "unknown"),
        failure.get("stage", "unknown"),
        failure.get("location", "unknown"),
        failure.get("cleanup_stage") or "none",
        failure.get("fingerprint", "unknown"),
        "none" if poison is None else poison[0],
        "none" if poison is None else poison[1],
    )


def _valid_failure_diagnostic(value: Mapping[str, Any]) -> bool:
    stage = value.get("failure_stage")
    fingerprint = value.get("error_fingerprint")
    location = value.get("failure_location")
    cleanup_stage = value.get("cleanup_stage")
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
        and (
            cleanup_stage is None
            or isinstance(cleanup_stage, str)
            and cleanup_stage.replace("_", "").isalnum()
            and len(cleanup_stage) <= 80
        )
    )


_FAILURE_AIMDO_INTEGER_FIELDS = frozenset(
    {
        "physical_bytes",
        "staged_bytes",
        "virtual_bytes",
        "allocation_count",
        "live_allocations",
        "live_bytes",
        "loaded_bytes",
        "faults",
        "signature_hits",
        "signature_misses",
        "fault_none_temporaries",
        "pinned_copy_bytes",
        "pageable_copy_bytes",
        "transfer_events",
        "transfer_waits",
        "prioritize_calls",
        "unpin_calls",
        "free_calls",
        "dirty_epoch",
        "lora_invalidations",
        "base_restores",
        "copy_stream_count",
        "host_buffer_capacity_bytes",
        "host_buffer_allocations",
        "host_buffer_unregistrations",
        "host_buffer_frees",
        "gathered_misses",
        "per_physical_misses",
        "packed_source_bytes",
        "gathered_h2d_bytes",
        "pressure_direct_transfers",
        "pressure_direct_bytes",
        "host_buffer_reuse_barriers",
        "host_source_pool_generation",
        "host_source_pool_lane_count",
        "host_source_pool_capacity_bytes",
        "host_source_pool_retained_slices",
        "host_source_pool_retained_bytes",
        "host_source_pool_temporary_slices",
        "host_source_pool_temporary_bytes",
        "host_source_pool_hits",
        "host_source_pool_misses",
        "host_source_pool_stale_rejections",
        "host_source_pool_warm_ram_pressure_bypasses",
        "host_source_pool_warm_zero_delta_extend_refusals",
        "host_source_pool_warm_registration_refusals",
        "host_source_pool_temporary_ram_pressure_bypasses",
        "host_source_pool_temporary_zero_delta_extend_refusals",
        "host_source_pool_temporary_registration_refusals",
        "base_file_read_calls",
        "base_file_read_bytes",
        "base_file_handle_opened",
        "base_file_handle_closed",
    }
)


def _valid_failure_aimdo_counters(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    expected = _FAILURE_AIMDO_INTEGER_FIELDS | {
        "backend",
        "version",
        "mode",
        "policy",
        "poisoned",
        "close_failed",
        "poison_reason",
        "copy_strategy",
        "copy_fallback_reason",
        "gathered_host_buffer_requested",
        "host_buffer_live",
        "host_tensor_view_live",
        "host_buffer_transfer_pending",
        "host_source_pool_poisoned",
        "host_source_pool_poison_reason",
        "host_source_registration",
        "base_file_backed",
        "base_file_source_live",
        "base_file_handle_live",
        "base_file_fallback_reason",
        "refill_failure_reason",
        "refill_target_bytes",
        "refill_root_already_bound",
        "refill_resident_bytes",
    }
    return bool(
        set(value) == expected
        and value.get("backend") == "comfy-aimdo"
        and value.get("version") == "0.4.15"
        and value.get("mode") == "dynamic_vbar"
        and value.get("policy") in {"auto", "required"}
        and value.get("copy_strategy") in {"gathered_host_buffer", "per_physical"}
        and (
            value.get("copy_fallback_reason") is None
            or _valid_host_buffer_fallback_reason(value.get("copy_fallback_reason"))
        )
        and isinstance(value.get("gathered_host_buffer_requested"), bool)
        and isinstance(value.get("host_buffer_live"), bool)
        and isinstance(value.get("host_tensor_view_live"), bool)
        and isinstance(value.get("host_buffer_transfer_pending"), bool)
        and isinstance(value.get("host_source_pool_poisoned"), bool)
        and (
            value.get("host_source_pool_poison_reason") is None
            or isinstance(value.get("host_source_pool_poison_reason"), str)
            and 0 < len(value["host_source_pool_poison_reason"]) <= 80
            and value["host_source_pool_poison_reason"].replace("_", "").isalnum()
        )
        and isinstance(value.get("base_file_backed"), bool)
        and isinstance(value.get("base_file_source_live"), bool)
        and isinstance(value.get("base_file_handle_live"), bool)
        and (
            value.get("base_file_fallback_reason") is None
            or _valid_base_file_fallback_reason(value.get("base_file_fallback_reason"))
        )
        and (
            value.get("refill_failure_reason") is None
            and value.get("refill_target_bytes") is None
            and value.get("refill_root_already_bound") is None
            and value.get("refill_resident_bytes") is None
            or value.get("refill_failure_reason") in _LTX23_REFILL_FAILURE_REASONS
            and isinstance(value.get("refill_target_bytes"), int)
            and not isinstance(value.get("refill_target_bytes"), bool)
            and value["refill_target_bytes"] >= 0
            and isinstance(value.get("refill_root_already_bound"), bool)
            and isinstance(value.get("refill_resident_bytes"), int)
            and not isinstance(value.get("refill_resident_bytes"), bool)
            and value["refill_resident_bytes"] >= 0
            and (
                value["refill_failure_reason"] != "unbound_root_exceeds_target"
                or value["refill_root_already_bound"] is False
            )
        )
        and (not value["host_tensor_view_live"] or value["host_buffer_live"])
        and (
            not value["host_buffer_transfer_pending"]
            or value["host_buffer_live"]
            and value["host_tensor_view_live"]
            and value["copy_strategy"] == "gathered_host_buffer"
        )
        and value["host_buffer_allocations"] <= 4
        and value["host_buffer_unregistrations"] <= value["host_buffer_allocations"]
        and value["host_buffer_frees"] <= value["host_buffer_unregistrations"]
        and value["gathered_misses"] <= value["signature_misses"]
        and value["per_physical_misses"] <= value["signature_misses"]
        and value["pressure_direct_transfers"] <= value["gathered_misses"]
        and value["pressure_direct_bytes"] <= value["gathered_h2d_bytes"]
        and (value["pressure_direct_transfers"] == 0) == (value["pressure_direct_bytes"] == 0)
        and value["pressure_direct_transfers"]
        <= value["host_source_pool_warm_ram_pressure_bypasses"]
        + value["host_source_pool_warm_zero_delta_extend_refusals"]
        + value["host_source_pool_warm_registration_refusals"]
        + value["host_source_pool_temporary_ram_pressure_bypasses"]
        + value["host_source_pool_temporary_zero_delta_extend_refusals"]
        + value["host_source_pool_temporary_registration_refusals"]
        <= value["pressure_direct_transfers"] + 1
        and value["host_buffer_reuse_barriers"] <= value["gathered_misses"]
        and value["host_source_pool_lane_count"] == value["host_buffer_allocations"]
        and value["host_source_pool_capacity_bytes"] >= value["host_buffer_capacity_bytes"]
        and value["host_source_pool_capacity_bytes"] <= 2 * value["virtual_bytes"]
        and value["gathered_misses"]
        <= value["host_source_pool_hits"]
        + value["host_source_pool_misses"]
        + value["pressure_direct_transfers"]
        <= value["signature_misses"]
        and value["host_source_pool_retained_bytes"] + value["host_source_pool_temporary_bytes"]
        <= value["host_source_pool_capacity_bytes"]
        and _valid_host_source_registration(
            value.get("host_source_registration"),
            source_misses=value["host_source_pool_misses"],
            capacity_bytes=value["host_source_pool_capacity_bytes"],
            expect_closed=not value["host_buffer_live"],
            allow_poisoned=(
                value.get("poison_reason")
                in {
                    "host_source_pool_structural_failure",
                    "host_source_pool_setup_cleanup_failed",
                }
                and value.get("host_source_pool_poisoned") is True
            ),
            pool_poison_reason=value.get("host_source_pool_poison_reason"),
        )
        and value["base_file_handle_opened"] <= 1
        and value["base_file_handle_closed"] <= value["base_file_handle_opened"]
        and value["base_file_handle_live"]
        == (value["base_file_handle_opened"] > value["base_file_handle_closed"])
        and (not value["base_file_source_live"] or value["base_file_handle_live"])
        and (not value["base_file_source_live"] or value["base_file_backed"])
        and (value["base_file_read_calls"] == 0) == (value["base_file_read_bytes"] == 0)
        and (value["base_file_read_calls"] == 0 or value["base_file_backed"])
        and (
            value["base_file_backed"]
            and value["base_file_handle_opened"] == 1
            and value["base_file_fallback_reason"] is None
            or not value["base_file_backed"]
            and value["base_file_read_calls"] == 0
        )
        and (
            value["copy_strategy"] == "gathered_host_buffer"
            and value["gathered_host_buffer_requested"] is True
            and value["copy_fallback_reason"] is None
            and 1 <= value["host_buffer_allocations"] <= 4
            and value["host_source_pool_generation"] >= 1
            and value["host_source_pool_stale_rejections"] == 0
            and value["host_buffer_reuse_barriers"] == 0
            and value["host_buffer_capacity_bytes"] > 0
            and value["per_physical_misses"] == 0
            and value["pinned_copy_bytes"] + value["pressure_direct_bytes"]
            == value["gathered_h2d_bytes"]
            and value["pageable_copy_bytes"] <= value["pressure_direct_bytes"]
            and value["packed_source_bytes"] <= value["gathered_h2d_bytes"]
            and value["gathered_h2d_bytes"]
            <= value["host_buffer_capacity_bytes"] * value["gathered_misses"]
            or value["copy_strategy"] == "per_physical"
            and (
                value["copy_fallback_reason"] is None
                or _valid_host_buffer_fallback_reason(value["copy_fallback_reason"])
            )
            and value["gathered_misses"] == 0
            and value["packed_source_bytes"] == 0
            and value["gathered_h2d_bytes"] == 0
            and value["host_buffer_reuse_barriers"] == 0
            and _valid_clean_host_source_pool_fallback(value)
        )
        and all(
            (
                field == "loaded_bytes"
                and value.get(field) is None
                or isinstance(value.get(field), int)
                and not isinstance(value.get(field), bool)
                and value[field] >= 0
            )
            for field in _FAILURE_AIMDO_INTEGER_FIELDS
        )
        and isinstance(value.get("poisoned"), bool)
        and isinstance(value.get("close_failed"), bool)
        and (
            value.get("poison_reason") is None
            or value.get("poison_reason") in _AIMDO_POISON_REASONS
        )
    )


def _terminate_supervisor(supervisor: PersistentWorkerSupervisor, primary: BaseException) -> bool:
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
    worker_failure: Mapping[str, Any] | None = None,
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
        failure_status: dict[str, object] = {
            key: worker_failure[key]
            for key in ("error_type", "stage", "location", "fingerprint")
            if key in worker_failure
        }
        if worker_failure.get("cleanup_stage") is not None:
            failure_status["cleanup_stage"] = worker_failure["cleanup_stage"]
        if worker_failure.get("aimdo_counters") is not None:
            failure_status["aimdo_counters"] = dict(worker_failure["aimdo_counters"])
        poison = _canonical_worker_poison(worker_failure)
        if poison is not None:
            failure_status["poison_reason"], failure_status["poison_origin"] = poison
        status["failure"] = failure_status
    return status


def _remove_output(path: Path, primary: BaseException) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"LTX 2.3 Kitchen partial output cleanup failed: {exc}")
