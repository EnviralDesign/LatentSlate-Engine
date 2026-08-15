"""Private persistent-worker supervisor for the exact Z-Image Turbo recipe.

The Engine parent deliberately owns only an authenticated process capability.
All model shells, CPU masters and CUDA residency remain in the child, so a
failed/cancelled command can be resolved by destroying its complete Job Object.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..safe_errors import SafeJobFailure
from ..z_image_turbo_recipe import (
    ZImageTurboRuntimeRequest,
    revalidate_z_image_turbo_runtime_request,
)
from . import z_image_cuda_health as _cuda_health
from .windows_process import DisposableProcessTree

_SCHEMA = 1
_MAX_BYTES = 1024 * 1024
_POLL = 0.1
_GENERATION_TIMEOUT_SECONDS = 30 * 60
_STAGE_TIMEOUT_SECONDS = 12 * 60
_HEARTBEAT_STALE_SECONDS = 45
_PROGRESS_STALE_SECONDS = (
    _STAGE_TIMEOUT_SECONDS  # compatibility: this is a stage, not liveness clock.
)
_CANCEL_GRACE_SECONDS = 5
_LOGGER = logging.getLogger(__name__)
_CUDA_HEALTH_PHASES = (
    "pre_import",
    "post_tokenizer",
    "post_qwen",
    "post_nextdit",
    "post_vae",
    "post_core",
    "pre_qwen_preflight",
)
_CUDA_HEALTH_STAGES = frozenset(f"cuda_health_{phase}" for phase in _CUDA_HEALTH_PHASES)
_CUDA_ERROR_CODES = frozenset(
    {
        "cuda_oom",
        "illegal_memory_access",
        "invalid_argument",
        "operation_not_supported",
        "driver_error",
        "unknown_runtime",
    }
)
_QWEN_FAILURE_STAGES = {
    *(f"conditioning.edge_{index:02d}" for index in range(7, 21)),
    *(
        f"conditioning.preflight_{kind}_{substage}"
        for kind in ("fp8", "nvfp4")
        for substage in (
            "cuda_sync",
            "uint8_allocate",
            "ordinary_uint8_copy",
            "ordinary_uint8_sync",
            "ordinary_uint8_readback",
            "origin_flat_prepare",
            "origin_uint8_copy",
            "flat_dtype_view",
            "shape_restore",
            "scale_move",
            "bit_verify",
            "direct_fp32_dequant",
            "f_linear",
            "validate",
        )
    ),
    "conditioning.embedding",
    "conditioning.mask",
    "conditioning.rope",
    "conditioning.final_norm",
    *(f"conditioning.block_{index:02d}" for index in range(36)),
    *(f"conditioning.linear_{index:03d}" for index in range(252)),
}
_FAILURE_STAGES = frozenset(
    {
        "auth",
        "canonical_validation",
        "device_contract",
        "rehydrate",
        "runtime_import",
        "tokenizer",
        "qwen_materialize",
        "nextdit_materialize",
        "transformer_onload",
        "vae_materialize",
        "core_ready",
        "conditioning",
        "noise",
        "sampling",
        "decode",
        "publish",
        *_CUDA_HEALTH_STAGES,
        *_QWEN_FAILURE_STAGES,
    }
)
_FAILURE_LOCATIONS = frozenset(
    {
        "z_image_turbo_worker._read_json",
        "z_image_turbo_worker._secret",
        "z_image_turbo_worker._validate",
        "z_image_turbo_worker._resolve_worker_cuda_device",
        "z_image_turbo_recipe.rehydrate_z_image_turbo_runtime_request",
        "z_image_turbo_worker._load_core",
        "z_image_turbo_worker._execute",
        "z_image_turbo_worker._validate_artifact",
        "z_image_turbo.generate",
    }
)
_SAFE_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "BaseException",
        "EOFError",
        "Exception",
        "FileNotFoundError",
        "ImportError",
        "IsADirectoryError",
        "JSONDecodeError",
        "KeyError",
        "MemoryError",
        "ModuleNotFoundError",
        "NotADirectoryError",
        "OSError",
        "OutOfMemoryError",
        "OverflowError",
        "PermissionError",
        "RuntimeError",
        "SystemExit",
        "TypeError",
        "UnicodeDecodeError",
        "ValueError",
        "ZImageDecodeCancelled",
        "ZImagePngPublicationCancelled",
        "ZImageSamplingCancelled",
        "ZImageTurboCancelled",
    }
)


class ZImageWorkerTimeout(RuntimeError):
    """A private worker failed to make bounded forward progress."""


class ZImageWorkerFailure(SafeJobFailure):
    """Already-sanitized failure provenance from an authenticated child result."""

    def __init__(
        self,
        error_type: str,
        stage: str,
        location: str,
        fingerprint: str,
        cuda_error_code: str | None = None,
        cuda_health_completed: tuple[str, ...] = (),
        cuda_health_phase: str | None = None,
        cuda_health_substage: str | None = None,
    ) -> None:
        self.error_type = error_type
        self.stage = stage
        self.location = location
        self.fingerprint = fingerprint
        self.cuda_error_code = cuda_error_code
        self.cuda_health_completed = cuda_health_completed
        self.cuda_health_phase = cuda_health_phase
        self.cuda_health_substage = cuda_health_substage
        message = (
            f"Z-Image worker failed ({error_type} during {stage} at {location}; "
            f"diagnostic {fingerprint[:12]}"
            + (f"; cuda {cuda_error_code}" if cuda_error_code is not None else "")
            + ")"
        )
        super().__init__(
            code="generation_failed",
            message=message,
            error_type=error_type,
            diagnostic=fingerprint[:12],
        )


@dataclass(frozen=True, slots=True)
class ManagedZImageResult:
    output_path: Path
    output_size_bytes: int
    metadata: dict[str, Any]
    worker_pid: int
    pipeline_warm: bool


@dataclass(slots=True)
class _Session:
    process: subprocess.Popen[bytes]
    tree: DisposableProcessTree
    paths: dict[str, Path]
    device: str
    binding: str
    secret: bytes
    successful_jobs: int = 0
    execution_device: str | None = None


class ManagedZImageTurboRuntime:
    """One serial, exact-recipe session whose child is poisoned on every error."""

    def __init__(
        self,
        request: ZImageTurboRuntimeRequest,
        *,
        generation_timeout_seconds: float = _GENERATION_TIMEOUT_SECONDS,
        progress_stale_seconds: float = _PROGRESS_STALE_SECONDS,
        cancel_grace_seconds: float = _CANCEL_GRACE_SECONDS,
    ) -> None:
        self.request = request
        self._session: _Session | None = None
        self._active_tree: DisposableProcessTree | None = None
        self._job_active = False
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []
        self._generation_timeout_seconds = _bounded_timeout(generation_timeout_seconds)
        self._progress_stale_seconds = _bounded_timeout(progress_stale_seconds)
        self._cancel_grace_seconds = _bounded_timeout(cancel_grace_seconds)

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        output_path: Path,
        device: str,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> ManagedZImageResult:
        check_cancelled()
        if self._job_active:
            raise RuntimeError("Z-Image worker is already active")
        if not revalidate_z_image_turbo_runtime_request(self.request):
            raise RuntimeError("Z-Image request changed after catalog validation")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("Z-Image requires a non-empty prompt and non-negative integer seed")
        target = Path(output_path).resolve(strict=False)
        if target.suffix.lower() != ".png" or target.exists():
            raise ValueError("Z-Image output path must be a fresh PNG")
        self._job_active = True
        session = self._session
        owns_output = False
        try:
            if session is not None and session.process.poll() is not None:
                self._discard(session)
                session = None
            secret = session.secret if session else secrets.token_bytes(32)
            payload = _payload(self.request, prompt, seed, target, device, secret)
            if session is None:
                paths = _paths()
                _require_fresh(paths, initial=True)
                _write_json(paths["request"], payload)
                session = self._spawn(paths, str(device), str(payload["session_binding"]), secret)
                self._session = session
                self._active_tree = session.tree
                paths["gate"].touch(exist_ok=False)
            else:
                if session.device != str(device) or not hmac.compare_digest(
                    session.binding, str(payload["session_binding"])
                ):
                    raise RuntimeError("Z-Image command does not match the loaded session")
                _require_fresh(session.paths)
                _write_json(session.paths["command"], payload)
            owns_output = True
            _wait(
                session,
                progress,
                check_cancelled,
                generation_timeout_seconds=self._generation_timeout_seconds,
                stage_timeout_seconds=self._progress_stale_seconds,
                cancel_grace_seconds=self._cancel_grace_seconds,
            )
            check_cancelled()  # late cancellation is never allowed to publish an artifact.
            result = _read_result(
                session.paths["result"],
                target,
                payload["request_binding"],
                secret,
            )
            _validate_metadata(result["metadata"], self.request, seed, session.device)
            session.execution_device = str(result["metadata"]["execution_device"])
            if session.process.poll() is not None:
                raise RuntimeError("Z-Image worker exited before success was accepted")
            warm = session.successful_jobs > 0
            session.successful_jobs += 1
            self._last_worker = {
                "pid": session.process.pid,
                "exit_code": None,
                "terminated": False,
                "outcome": "succeeded",
                "pipeline_warm": warm,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
            self._cleanup_errors = _cleanup_job(session.paths)
            return ManagedZImageResult(
                target,
                int(result["output_size_bytes"]),
                dict(result["metadata"]),
                session.process.pid,
                warm,
            )
        except BaseException as exc:
            if session is not None:
                self._poison(session, exc)
            if owns_output:
                target.unlink(missing_ok=True)
            raise
        finally:
            self._job_active = False

    def _spawn(self, paths: dict[str, Path], device: str, binding: str, secret: bytes) -> _Session:
        env = os.environ.copy()
        env["LATENTSLATE_ZIMAGE_IPC_SECRET"] = secret.hex()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "latentslate_engine.runtime.z_image_turbo_worker",
                "--request",
                str(paths["request"]),
                "--result",
                str(paths["result"]),
                "--progress",
                str(paths["progress"]),
                "--heartbeat",
                str(paths["heartbeat"]),
                "--start-gate",
                str(paths["gate"]),
                "--command",
                str(paths["command"]),
                "--cancel",
                str(paths["cancel"]),
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
            process.kill()
            raise
        return _Session(process, tree, paths, device, binding, secret)

    def _poison(self, session: _Session, primary: BaseException) -> None:
        try:
            _terminate(session.tree, session.process)
            self._last_worker = {
                "pid": session.process.pid,
                "exit_code": session.process.poll(),
                "terminated": True,
                "outcome": _outcome(primary),
                "timeout": isinstance(primary, ZImageWorkerTimeout),
                "pipeline_warm": False,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
            if isinstance(primary, ZImageWorkerFailure):
                self._last_worker.update(
                    {
                        "failure_stage": primary.stage,
                        "failure_location": primary.location,
                        "diagnostic": primary.fingerprint[:12],
                    }
                )
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            finally:
                self._cleanup_errors = _cleanup_session(session.paths)

    def _discard(self, session: _Session) -> None:
        exit_code = session.process.poll()
        self._last_worker = {
            "pid": session.process.pid,
            "exit_code": exit_code,
            "terminated": True,
            "outcome": "dead_idle",
            "pipeline_warm": False,
            "memory_boundary": "persistent_exact_recipe_worker",
        }
        try:
            _terminate(session.tree, session.process)
        except BaseException as exc:  # noqa: BLE001 - status must retain a usable recovery path.
            self._cleanup_errors = [*self._cleanup_errors, f"dead_worker:{type(exc).__name__}"][
                -16:
            ]
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            finally:
                self._cleanup_errors = [*self._cleanup_errors, *_cleanup_session(session.paths)][
                    -16:
                ]

    def status(self) -> dict[str, Any]:
        session = self._session
        if session is not None and not self._job_active and session.process.poll() is not None:
            self._discard(session)
            session = None
        return {
            "family": "zimage",
            "runtime": "engine-native/z-image-turbo-persistent-worker",
            "recipe_fingerprint": self.request.fingerprint,
            "loaded": session is not None,
            "active_worker": self._job_active,
            "worker_pid": None if session is None else session.process.pid,
            "execution_device": None if session is None else session.execution_device,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "components": self.request.public_component_manifest(),
            "cache_support": {"prompt": False, "media": False, "tensor": True},
            "cache": {"pipeline_warm": bool(session and session.successful_jobs > 0)},
        }

    def clear_cache(self) -> None:
        self.unload()

    def unload(self) -> None:
        session = self._session
        if session is not None:
            try:
                _terminate(session.tree, session.process)
                self._last_worker = {
                    "pid": session.process.pid,
                    "exit_code": session.process.poll(),
                    "terminated": True,
                    "outcome": "unloaded",
                    "pipeline_warm": False,
                    "memory_boundary": "persistent_exact_recipe_worker",
                }
            finally:
                self._session = None
                self._active_tree = None
                try:
                    session.tree.close()
                finally:
                    self._cleanup_errors = _cleanup_session(session.paths)


def _binding(value: Mapping[str, object], secret: bytes) -> str:
    return hmac.new(
        secret, json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256
    ).hexdigest()


def _result_binding(value: Mapping[str, object], secret: bytes) -> str:
    """Bind the complete result envelope, including output and provenance."""

    unsigned = {key: item for key, item in value.items() if key != "result_binding"}
    return hmac.new(
        secret,
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        hashlib.sha256,
    ).hexdigest()


def _payload(
    request: ZImageTurboRuntimeRequest,
    prompt: str,
    seed: int,
    output: Path,
    device: str,
    secret: bytes,
) -> dict[str, object]:
    recipe = request.to_json_dict()
    session = {
        "recipe": recipe,
        "device": str(device),
        "dtype": "bfloat16",
        "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    }
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA,
        **session,
        "session_binding": _binding(session, secret),
        "output_path": str(output),
        "generation": {"prompt": prompt, "seed": seed},
    }
    return {**unsigned, "request_binding": _binding(unsigned, secret)}


def _paths() -> dict[str, Path]:
    root = Path(tempfile.mkdtemp(prefix="latentslate-zimage-"))
    _secure(root)
    return {
        key: root / name
        for key, name in {
            "request": "request.json",
            "result": "result.json",
            "progress": "progress.jsonl",
            "heartbeat": "heartbeat.jsonl",
            "gate": "start-gate",
            "command": "command.json",
            "cancel": "cancel-requested",
        }.items()
    }


def _secure(root: Path) -> None:
    if os.name != "nt":
        os.chmod(root, 0o700)
        return
    advapi, kernel = (
        ctypes.WinDLL("advapi32", use_last_error=True),
        ctypes.WinDLL("kernel32", use_last_error=True),
    )
    descriptor = ctypes.c_void_p()
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None
    ):
        raise OSError(ctypes.get_last_error(), "Z-Image IPC DACL conversion failed")
    try:
        if not advapi.SetFileSecurityW(str(root), 0x00000004 | 0x80000000, descriptor):
            raise OSError(ctypes.get_last_error(), "Z-Image IPC DACL application failed")
    finally:
        if kernel.LocalFree(descriptor):
            raise OSError(ctypes.get_last_error(), "Z-Image IPC DACL free failed")


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


def _require_fresh(paths: Mapping[str, Path], *, initial: bool = False) -> None:
    names = (
        ("request", "result", "progress", "heartbeat", "gate", "command", "cancel")
        if initial
        else ("result", "progress", "command")
    )
    if stale := [name for name in names if paths[name].exists()]:
        raise RuntimeError("Z-Image worker IPC paths already exist: " + ", ".join(stale))


def _wait(
    session: _Session,
    progress: Callable[[float, str | None], None],
    check_cancelled: Callable[[], None],
    *,
    generation_timeout_seconds: float,
    stage_timeout_seconds: float,
    cancel_grace_seconds: float,
) -> None:
    offset = 0
    started = time.monotonic()
    last_progress = started
    last_heartbeat = started
    heartbeat_offset = 0
    def drain() -> tuple[bool, bool, bool]:
        """Consume all queued records before making a terminal decision."""

        nonlocal heartbeat_offset, last_heartbeat, offset, last_progress
        heartbeat_offset, heartbeat_seen = _drain_heartbeats(
            session.paths["heartbeat"], heartbeat_offset
        )
        if heartbeat_seen:
            last_heartbeat = time.monotonic()
        offset, progress_seen = _drain_progress(session.paths["progress"], offset, progress)
        if progress_seen:
            last_progress = time.monotonic()

        return session.paths["result"].is_file(), heartbeat_seen, progress_seen

    def request_parent_cancel() -> None:
        try:
            check_cancelled()
        except BaseException:
            session.paths["cancel"].touch(exist_ok=True)
            _await_cancel_grace(session, cancel_grace_seconds)
            raise

    def timeout_recheck() -> tuple[bool, bool, bool]:
        """Give the child one ordered, zero-yield chance to win a boundary race."""

        result, heartbeat_seen, progress_seen = drain()
        if result:
            return True, heartbeat_seen, progress_seen
        request_parent_cancel()
        if session.process.poll() is not None:
            raise RuntimeError("Z-Image worker exited without a bounded result")
        # A child that completed between the final stat and its atomic replace
        # must not be killed merely because the parent crossed a clock boundary.
        time.sleep(0)
        result, next_heartbeat, next_progress = drain()
        heartbeat_seen = heartbeat_seen or next_heartbeat
        progress_seen = progress_seen or next_progress
        if result:
            return True, heartbeat_seen, progress_seen
        request_parent_cancel()
        if session.process.poll() is not None:
            raise RuntimeError("Z-Image worker exited without a bounded result")
        return False, heartbeat_seen, progress_seen

    while True:
        # Drain queued IPC first: a just-written record must win over a timeout.
        if drain()[0]:
            return
        request_parent_cancel()
        now = time.monotonic()
        if now - started > generation_timeout_seconds:
            result, _heartbeat_seen, _progress_seen = timeout_recheck()
            if result:
                return
            _request_cancel_then_grace(session, cancel_grace_seconds)
            raise ZImageWorkerTimeout("Z-Image generation exceeded its bounded deadline")
        if now - last_heartbeat > _HEARTBEAT_STALE_SECONDS:
            result, heartbeat_seen, _progress_seen = timeout_recheck()
            if result:
                return
            if heartbeat_seen:
                # A fresh heartbeat was atomically appended at the boundary.
                # It changes only heartbeat liveness, never stage or hard clocks.
                continue
            _request_cancel_then_grace(session, cancel_grace_seconds)
            raise ZImageWorkerTimeout("Z-Image worker heartbeat became stale")
        if now - last_progress > stage_timeout_seconds:
            result, _heartbeat_seen, progress_seen = timeout_recheck()
            if result:
                return
            if progress_seen:
                # Progress is the sole renewable stage clock; heartbeat is not.
                continue
            _request_cancel_then_grace(session, cancel_grace_seconds)
            raise ZImageWorkerTimeout("Z-Image worker stage exceeded its bounded deadline")
        if session.process.poll() is not None:
            raise RuntimeError("Z-Image worker exited without a bounded result")
        time.sleep(_POLL)


def _drain_progress(
    path: Path, offset: int, progress: Callable[[float, str | None], None]
) -> tuple[int, bool]:
    if not path.is_file():
        return offset, False
    if path.stat().st_size > _MAX_BYTES:
        raise RuntimeError("Z-Image worker progress exceeds its bound")
    with path.open(encoding="utf-8") as stream:
        stream.seek(offset)
        seen = False
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Z-Image worker progress is invalid") from exc
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("progress"), (int, float))
                or not math.isfinite(float(value["progress"]))
                or not 0 <= float(value["progress"]) <= 1
            ):
                raise TypeError("Z-Image worker progress is invalid")
            progress(
                float(value["progress"]),
                value.get("message") if isinstance(value.get("message"), str) else None,
            )
            seen = True
        return stream.tell(), seen


def _drain_heartbeats(path: Path, offset: int) -> tuple[int, bool]:
    if not path.is_file():
        return offset, False
    if path.stat().st_size > _MAX_BYTES:
        raise RuntimeError("Z-Image worker heartbeat exceeds its bound")
    with path.open(encoding="utf-8") as stream:
        stream.seek(offset)
        seen = False
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Z-Image worker heartbeat is invalid") from exc
            if not isinstance(value, dict) or value != {"heartbeat": 1}:
                raise RuntimeError("Z-Image worker heartbeat is invalid")
            seen = True
        return stream.tell(), seen


def _read_result(path: Path, output: Path, binding: object, secret: bytes) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise RuntimeError("Z-Image worker result is missing or exceeds its bound")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Z-Image worker result is invalid") from exc
    # Authenticate the full, canonical unsigned result before reading any
    # worker-supplied content (including result kind, output metadata, or
    # diagnostic labels).  Exact schemas below still prevent future field
    # additions from becoming silently trusted.
    if not isinstance(result, dict):
        raise RuntimeError("Z-Image worker result does not bind its command")  # noqa: TRY004 - protocol error is intentionally uniform.
    result_binding = result.get("result_binding")
    try:
        expected_result_binding = _result_binding(result, secret)
    except (TypeError, ValueError):
        raise RuntimeError("Z-Image worker result does not bind its command") from None
    if not isinstance(result_binding, str) or not hmac.compare_digest(
        result_binding, expected_result_binding
    ):
        raise RuntimeError("Z-Image worker result does not bind its command")
    if (
        result.get("schema_version") != _SCHEMA
        or not isinstance(binding, str)
        or not isinstance(result.get("request_binding"), str)
        or not hmac.compare_digest(result["request_binding"], binding)
    ):
        raise RuntimeError("Z-Image worker result does not bind its command")
    if result.get("ok") is False:
        failure = _validate_failure_result(result, binding)
        if failure is None:
            raise RuntimeError("Z-Image worker failure result is invalid")
        if "cuda_error_code" in failure:
            _LOGGER.error(
                "Z-Image synthetic CUDA health failure: type=%s code=%s",
                failure["error_type"],
                failure["cuda_error_code"],
            )
        else:
            _LOGGER.error(
                "Z-Image worker failure: type=%s stage=%s location=%s diagnostic=%s",
                failure["error_type"],
                failure["failure_stage"],
                failure["failure_location"],
                failure["error_fingerprint"],
            )
        raise ZImageWorkerFailure(
            failure["error_type"],
            failure["failure_stage"],
            failure["failure_location"],
            failure["error_fingerprint"],
            failure.get("cuda_error_code"),
            tuple(failure.get("cuda_health_completed", ())),
            failure.get("cuda_health_phase"),
            failure.get("cuda_health_substage"),
        )
    expected = {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "output_path",
        "output_size_bytes",
        "output_sha256",
        "metadata",
    }
    try:
        result_output = Path(str(result.get("output_path", ""))).resolve(strict=True)
        expected_output = output.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Z-Image worker success result is invalid") from exc
    if set(result) != expected or result.get("ok") is not True or result_output != expected_output:
        raise RuntimeError("Z-Image worker success result is invalid")
    _validate_png(output, result)
    if not isinstance(result["metadata"], dict):
        raise TypeError("Z-Image worker metadata is invalid")
    return result


def _validate_failure_result(
    value: Mapping[str, Any], binding: str
) -> dict[str, str] | None:
    """Accept only the fixed child error schema; reject additions and tampering."""

    required = {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "error_type",
        "failure_stage",
        "failure_location",
        "error_fingerprint",
    }
    health_failure = value.get("failure_stage") in _CUDA_HEALTH_STAGES
    if health_failure:
        required.update(
            {
                "cuda_error_code",
                "cuda_health_completed",
                "cuda_health_phase",
                "cuda_health_substage",
            }
        )
    keys = set(value)
    if keys != required:
        return None
    if (
        value.get("schema_version") != _SCHEMA
        or value.get("ok") is not False
        or not isinstance(value.get("request_binding"), str)
        or not hmac.compare_digest(value["request_binding"], binding)
        or not isinstance(value.get("error_type"), str)
        or value.get("error_type") not in _SAFE_EXCEPTION_TYPES
        or value.get("failure_stage") not in _FAILURE_STAGES
        or value.get("failure_location") not in _FAILURE_LOCATIONS
        or (
            health_failure
            and value.get("cuda_error_code") not in _CUDA_ERROR_CODES
        )
        or (
            health_failure
            and (
                value.get("cuda_health_phase") not in _CUDA_HEALTH_PHASES
                or value.get("failure_stage")
                != f"cuda_health_{value.get('cuda_health_phase')}"
                or value.get("cuda_health_substage")
                not in _cuda_health._HEALTH_SUBSTAGES
                or not isinstance(value.get("cuda_health_completed"), list)
                or value.get("cuda_health_completed")
                != list(
                    _CUDA_HEALTH_PHASES[
                        : _CUDA_HEALTH_PHASES.index(value.get("cuda_health_phase"))
                    ]
                )
            )
        )
        or not isinstance(value.get("error_fingerprint"), str)
        or len(value["error_fingerprint"]) != 64
        or any(character not in "0123456789abcdef" for character in value["error_fingerprint"])
    ):
        return None
    result = {
        key: value[key]
        for key in ("error_type", "failure_stage", "failure_location", "error_fingerprint")
    }
    if health_failure:
        result["cuda_error_code"] = value["cuda_error_code"]
        result["cuda_health_completed"] = value["cuda_health_completed"]
        result["cuda_health_phase"] = value["cuda_health_phase"]
        result["cuda_health_substage"] = value["cuda_health_substage"]
    return result


def _validate_png(path: Path, result: Mapping[str, Any]) -> None:
    from PIL import Image

    if (
        path.stat().st_size <= 0
        or result.get("output_size_bytes") != path.stat().st_size
        or result.get("output_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("Z-Image worker PNG identity is invalid")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError("Z-Image worker output is not an RGB 1024 PNG")


def _validate_metadata(
    metadata: Mapping[str, Any],
    request: ZImageTurboRuntimeRequest,
    seed: int,
    requested_device: str,
) -> None:
    qwen, transformer = (
        metadata.get("qwen_dispatch"),
        metadata.get("transformer_dispatch"),
    )
    cuda_health = metadata.get("cuda_health")
    qwen_ints = (
        "module_count",
        "fp8_modules",
        "nvfp4_modules",
        "dequantized_modules",
        "f_linear_modules",
        "min_module_dequant_delta",
        "max_module_dequant_delta",
        "total_dequantizations",
        "total_f_linear_calls",
        "rejected_dispatch_count",
        "dense_checkpoint_fallback_count",
        "scaled_mm_calls",
    )
    qwen_int_values = {
        key: qwen.get(key) if isinstance(qwen, Mapping) else None for key in qwen_ints
    }
    qwen_counts_are_ints = all(
        isinstance(value, int) and not isinstance(value, bool) for value in qwen_int_values.values()
    )
    execution_device = metadata.get("execution_device")
    execution_device_is_indexed = (
        isinstance(execution_device, str)
        and execution_device.startswith("cuda:")
        and execution_device.removeprefix("cuda:").isdigit()
    )
    if (
        metadata.get("family") != "zimage"
        or metadata.get("runtime") != "engine-native/z-image-turbo"
        or metadata.get("request_fingerprint") != request.fingerprint
        or metadata.get("seed") != seed
        or metadata.get("requested_device") != str(requested_device)
        or not execution_device_is_indexed
        or (
            str(requested_device).startswith("cuda:")
            and execution_device != str(requested_device)
        )
        or metadata.get("schedule") != dict(request.schedule)
        or cuda_health != {phase: "pass" for phase in _CUDA_HEALTH_PHASES}
        or not isinstance(qwen, Mapping)
        or not isinstance(transformer, Mapping)
        or not qwen_counts_are_ints
        or qwen_int_values["module_count"] != 189
        or qwen_int_values["fp8_modules"] != 177
        or qwen_int_values["nvfp4_modules"] != 12
        or qwen_int_values["dequantized_modules"] != 189
        or qwen_int_values["f_linear_modules"] != 189
        or qwen.get("complete") is not True
        or qwen.get("contract") != "full_precision_mm"
        or qwen.get("backend")
        != "comfy-kitchen/public-direct-fp32-dequant+torch/f.linear"
        or qwen_int_values["dense_checkpoint_fallback_count"] != 0
        or qwen_int_values["rejected_dispatch_count"] != 0
        or qwen.get("activation_quantized") is not False
        or qwen_int_values["scaled_mm_calls"] != 0
        or qwen.get("per_op_residency") is not True
        or qwen.get("stored_transport") != "source-backed-raw-byte/vbar-equivalent"
        or qwen.get("full_module_cuda_onload") is not False
        or qwen.get("cpu_master_retained") is not True
        or qwen.get("first_linear_preflight") is not True
        or qwen.get("first_linear_format") != "fp8"
        or qwen.get("first_linear_logical_shape") != "4096x2560"
        or qwen.get("first_linear_storage_dtype") != "float8_e4m3fn"
        or qwen.get("first_linear_compute_dtype") != "float32"
        or qwen.get("first_linear_output_shape") != "1x1x4096"
        or qwen.get("first_linear_backend")
        != "comfy_kitchen.dequantize_per_tensor_fp8+torch/f.linear"
        or qwen.get("first_linear_layout_registered") is not True
        or qwen.get("first_linear_transfer")
        != "source-backed-raw-byte/current-stream/blocking"
        or qwen.get("first_linear_transport_equivalence") != "vbar-output-equivalent"
        or qwen.get("first_linear_bit_identity") is not True
        or qwen.get("first_linear_byte_count") != 10_485_760
        or qwen.get("first_linear_logical_wrapper_cast") is not False
        or qwen.get("first_linear_dequant_contract") != "public-direct-fp32"
        or qwen_int_values["min_module_dequant_delta"] <= 0
        or qwen_int_values["max_module_dequant_delta"] < qwen_int_values["min_module_dequant_delta"]
        or qwen_int_values["total_dequantizations"] < 189 * qwen_int_values["min_module_dequant_delta"]
        or qwen_int_values["total_dequantizations"] > 189 * qwen_int_values["max_module_dequant_delta"]
        or qwen_int_values["total_f_linear_calls"] != qwen_int_values["total_dequantizations"]
        or transformer.get("module_count") != 202
        or transformer.get("dense_fallback_count") != 0
        or transformer.get("rejected_dispatch_count") != 0
    ):
        raise RuntimeError("Z-Image worker provenance differs from the exact request")


def _terminate(tree: DisposableProcessTree, process: subprocess.Popen[bytes]) -> None:
    tree.terminate()
    process.wait(timeout=15)
    tree.wait_for_empty()


def _cleanup_job(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for key in ("command", "result", "progress", "heartbeat"):
        try:
            paths[key].unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{key}:{type(exc).__name__}")
    return errors[-16:]


def _cleanup_session(paths: Mapping[str, Path]) -> list[str]:
    errors = _cleanup_job(paths)
    for key in ("request", "gate", "cancel"):
        try:
            paths[key].unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{key}:{type(exc).__name__}")
    try:
        paths["request"].parent.rmdir()
    except OSError as exc:
        errors.append(f"root:{type(exc).__name__}")
    return errors[-16:]


def _await_cancel_grace(session: _Session, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if session.paths["result"].is_file() or session.process.poll() is not None:
            return
        time.sleep(min(_POLL, max(0.0, deadline - time.monotonic())))


def _request_cancel_then_grace(session: _Session, seconds: float) -> None:
    session.paths["cancel"].touch(exist_ok=True)
    _await_cancel_grace(session, seconds)


def _bounded_timeout(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Z-Image worker timeout must be a finite positive number")
    return float(value)


def _outcome(exc: BaseException) -> str:
    if isinstance(exc, ZImageWorkerTimeout):
        return "timed_out"
    if isinstance(exc, asyncio.CancelledError) or any(
        cls.__name__ == "ToolCancelled" for cls in type(exc).__mro__
    ):
        return "canceled"
    return "failed"
