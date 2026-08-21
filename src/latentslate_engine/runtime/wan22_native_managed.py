"""Shared persistent-worker adapter for identity-bound native Wan 14B runtimes."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import math
import os
import secrets
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..wan22_recipe import Wan22RuntimeRequest, revalidate_runtime_request
from .framework.worker import (
    PersistentWatchdogPolicy,
    PersistentWorkerExited,
    PersistentWorkerPaths,
    PersistentWorkerStreamError,
    PersistentWorkerSupervisor,
    PersistentWorkerTimeout,
    WorkerJsonFileError,
    hmac_sha256,
    read_bounded_json,
    result_hmac_sha256,
)

_WORKER_SCHEMA_VERSION = 1
_MAX_WORKER_RESULT_BYTES = 1024 * 1024
_MAX_WORKER_PROGRESS_BYTES = 1024 * 1024
_MAX_WORKER_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
_GENERATION_TIMEOUT_SECONDS = 4 * 60 * 60
_STAGE_TIMEOUT_SECONDS = 45 * 60
_HEARTBEAT_TIMEOUT_SECONDS = 45
_CANCEL_GRACE_SECONDS = 3
_WORKER_PROVENANCE_STRING_FIELDS = frozenset(
    {
        "support_fingerprint",
        "tokenizer_sha256",
        "transformer_high_header_sha256",
        "transformer_low_header_sha256",
        "text_encoder_header_sha256",
        "vae_header_sha256",
        "transformer_high_contract",
        "transformer_low_contract",
        "text_encoder_contract",
        "stage_policy",
        "sampler",
        "scheduler",
    }
)
_WORKER_PROVENANCE_INTEGER_FIELDS = frozenset(
    {
        "steps",
        "seed",
        "transformer_high_size_bytes",
        "transformer_low_size_bytes",
        "text_encoder_size_bytes",
        "vae_size_bytes",
        "transformer_high_mtime_ns",
        "transformer_low_mtime_ns",
        "text_encoder_mtime_ns",
        "vae_mtime_ns",
    }
)
_WORKER_PROVENANCE_FIELDS = (
    _WORKER_PROVENANCE_STRING_FIELDS
    | _WORKER_PROVENANCE_INTEGER_FIELDS
    | {"shift", "configured_loras", "active_loras", "lora_dispatch", "transformer_dispatch"}
)


@dataclass(frozen=True, slots=True)
class NativeWanWorkerResult:
    """Bounded data returned by the exact persistent native Wan session."""

    output_path: Path
    output_size_bytes: int
    stream_metadata: dict[str, int | float | str | bool]
    provenance: dict[str, object]
    worker_pid: int
    worker_exit_code: int | None
    pipeline_warm: bool = False


def _cleanup_owned_encoder_temps(
    output_path: Path, *, primary: BaseException | None = None
) -> None:
    """Remove only atomic-MP4 staging files belonging to this exact target."""

    try:
        target = Path(output_path).resolve(strict=False)
        parent = target.parent.resolve(strict=True)
        prefix = f".{target.name}."
        for candidate in parent.glob(f"{prefix}*.tmp.mp4"):
            resolved = candidate.resolve(strict=False)
            if resolved.parent != parent or not candidate.is_file():
                continue
            candidate.unlink()
    except BaseException as exc:  # noqa: BLE001 - cleanup must never hide job state.
        if primary is not None:
            primary.add_note(f"native Wan encoder staging cleanup also failed: {exc}")


def _remove_output_or_note(output_path: Path, primary: BaseException) -> None:
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"native Wan partial output cleanup failed: {exc}")


def _validate_worker_provenance(value: Mapping[str, object]) -> None:
    """Accept only the fixed public worker result contract, never arbitrary JSON."""

    if set(value) != _WORKER_PROVENANCE_FIELDS:
        raise RuntimeError("native Wan worker provenance schema is invalid")
    for key in _WORKER_PROVENANCE_STRING_FIELDS:
        if not isinstance(value[key], str) or not value[key]:
            raise RuntimeError(f"native Wan worker provenance {key} is invalid")
    for key in _WORKER_PROVENANCE_INTEGER_FIELDS:
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise RuntimeError(f"native Wan worker provenance {key} is invalid")
    shift = value["shift"]
    if (
        isinstance(shift, bool)
        or not isinstance(shift, (int, float))
        or not math.isfinite(float(shift))
    ):
        raise RuntimeError("native Wan worker provenance shift is invalid")
    if not isinstance(value["configured_loras"], list) or not isinstance(
        value["active_loras"], list
    ):
        raise TypeError("native Wan worker provenance LoRA stacks are invalid")
    dispatch = value["lora_dispatch"]
    if not isinstance(dispatch, dict) or set(dispatch) != {"high", "low"}:
        raise RuntimeError("native Wan worker provenance LoRA dispatch is invalid")
    for stage, item in dispatch.items():
        if not isinstance(item, dict) or set(item) != {
            "target_module_count",
            "dispatch_call_count",
        }:
            raise RuntimeError(f"native Wan worker {stage} LoRA dispatch is invalid")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in item.values()
        ):
            raise RuntimeError(f"native Wan worker {stage} LoRA dispatch is invalid")
    _validate_transformer_dispatch(value["transformer_dispatch"])


def _validate_transformer_dispatch(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"high", "low"}:
        raise RuntimeError("native Wan worker transformer dispatch is invalid")
    for stage, proof in value.items():
        if not isinstance(proof, dict) or set(proof) != {
            "fp8_module_count",
            "fp8_modules",
            "int8_module_count",
            "int8_modules",
            "dense_fallback_count",
            "rejected_count",
        }:
            raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
        for key in (
            "fp8_module_count",
            "int8_module_count",
            "dense_fallback_count",
            "rejected_count",
        ):
            if isinstance(proof[key], bool) or not isinstance(proof[key], int) or proof[key] < 0:
                raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
        for kind, delta_key in (
            ("fp8_modules", "native_dispatch_delta"),
            ("int8_modules", "int8_dispatch_delta"),
        ):
            modules = proof[kind]
            if not isinstance(modules, dict) or len(modules) != proof[f"{kind[:-8]}_module_count"]:
                raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
            for name, counts in modules.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(counts, dict)
                    or set(counts) != {delta_key, "rejected_delta", "dense_fallback_delta"}
                ):
                    raise RuntimeError(
                        f"native Wan worker {stage} transformer module proof is invalid"
                    )
                if any(
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                    for count in counts.values()
                ):
                    raise RuntimeError(
                        f"native Wan worker {stage} transformer module proof is invalid"
                    )
                if (
                    counts[delta_key] <= 0
                    or counts["rejected_delta"] != 0
                    or counts["dense_fallback_delta"] != 0
                ):
                    raise RuntimeError(
                        f"native Wan worker {stage} transformer module proof is not clean"
                    )
        if proof["dense_fallback_count"] != 0 or proof["rejected_count"] != 0:
            raise RuntimeError(
                f"native Wan worker {stage} transformer proof has fallback or rejection"
            )


def _validate_stream_metadata(value: Mapping[str, object]) -> None:
    required = {
        "width",
        "height",
        "frame_count",
        "fps",
        "duration_seconds",
        "has_audio",
        "codec_name",
        "pixel_format",
    }
    if set(value) != required:
        raise RuntimeError("native Wan worker stream metadata schema is invalid")
    for key in ("width", "height", "frame_count", "fps"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise RuntimeError("native Wan worker stream metadata is invalid")
    if (
        value["has_audio"] is not False
        or not isinstance(value["codec_name"], str)
        or not value["codec_name"]
        or not isinstance(value["pixel_format"], str)
        or not value["pixel_format"]
    ):
        raise RuntimeError("native Wan worker stream metadata is invalid")
    duration = value["duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise RuntimeError("native Wan worker stream metadata is invalid")
    expected_duration = value["frame_count"] / value["fps"]
    if not math.isclose(float(duration), expected_duration, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("native Wan worker stream duration does not match frames and FPS")


def _validate_stream_against_request(
    value: Mapping[str, object], request: Any, *, fps: int
) -> None:
    _validate_stream_metadata(value)
    if (
        value["width"] != request.width
        or value["height"] != request.height
        or value["frame_count"] != request.num_frames
        or value["fps"] != fps
        or value["has_audio"] is not False
    ):
        raise RuntimeError("native Wan worker observed stream does not match the requested output")


def _validate_worker_provenance_against_request(
    value: Mapping[str, object], request: Wan22RuntimeRequest, *, expected_seed: int
) -> None:
    """Bind child result facts to the parent-revalidated resource identities."""

    _validate_worker_provenance(value)
    from ..wan22_recipe import wan22_i2v_operation

    operation = wan22_i2v_operation(request.operation)
    fixed = {
        "steps": operation["steps"],
        "stage_policy": operation["stage_policy"],
        "sampler": operation["sampler"],
        "scheduler": operation["scheduler"],
        "shift": operation["shift"],
        "seed": expected_seed,
    }
    for key, expected in fixed.items():
        if value[key] != expected:
            raise RuntimeError(f"native Wan worker provenance {key} does not match recipe")
    support = request.support_plan
    if support is None or (
        value["support_fingerprint"] != support.fingerprint
        or value["tokenizer_sha256"] != support.tokenizer_sha256
    ):
        raise RuntimeError("native Wan worker provenance support identity does not match recipe")
    role_prefixes = {
        "transformer_high_noise": "transformer_high",
        "transformer_low_noise": "transformer_low",
        "text_encoder": "text_encoder",
        "vae": "vae",
    }
    for role, prefix in role_prefixes.items():
        identity = request.identities.get(role)
        if identity is None or (
            value[f"{prefix}_header_sha256"] != identity.header_sha256
            or value[f"{prefix}_size_bytes"] != identity.size_bytes
            or value[f"{prefix}_mtime_ns"] != identity.mtime_ns
        ):
            raise RuntimeError(
                f"native Wan worker provenance {role} identity does not match recipe"
            )
    for role, prefix in (
        ("transformer_high_noise", "transformer_high"),
        ("transformer_low_noise", "transformer_low"),
        ("text_encoder", "text_encoder"),
    ):
        expected_contract = request.components[role].get("quantization_contract")
        if value[f"{prefix}_contract"] != expected_contract:
            raise RuntimeError(
                f"native Wan worker provenance {role} contract does not match recipe"
            )
    from .wan22_stored_adapter import expected_wan_stored_module_targets

    for stage, role in (("high", "transformer_high_noise"), ("low", "transformer_low_noise")):
        plan = request.adapter_plans.get(role)
        if plan is None:
            raise RuntimeError(f"native Wan worker {stage} transformer plan is missing")
        expected_modules = set(expected_wan_stored_module_targets(plan))
        proof = value["transformer_dispatch"][stage]
        is_int8 = plan.artifact_contract == "comfy_quant/int8_tensorwise_convrot"
        reported = proof["int8_modules"] if is_int8 else proof["fp8_modules"]
        unexpected_kind = proof["fp8_modules"] if is_int8 else proof["int8_modules"]
        if set(reported) != expected_modules or unexpected_kind:
            raise RuntimeError(
                f"native Wan worker {stage} transformer module proof does not bind its plan"
            )
    expected_configured = [dict(item) for item in request.configured_loras]
    expected_active = [item.public_dict() for item in request.active_loras]
    if value["configured_loras"] != expected_configured or value["active_loras"] != expected_active:
        raise RuntimeError("native Wan worker provenance LoRA stacks do not match recipe")
    expected_by_stage = {
        stage: sum(1 for item in request.active_loras if item.stage == stage)
        for stage in ("high", "low")
    }
    for stage, active_count in expected_by_stage.items():
        dispatch = value["lora_dispatch"][stage]
        if active_count and (
            dispatch["target_module_count"] <= 0 or dispatch["dispatch_call_count"] <= 0
        ):
            raise RuntimeError(f"native Wan worker {stage} LoRA did not dispatch")
        if not active_count and dispatch != {"target_module_count": 0, "dispatch_call_count": 0}:
            raise RuntimeError(f"native Wan worker reported unexpected {stage} LoRA dispatch")


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_WORKER_RESULT_BYTES)
    except WorkerJsonFileError:
        raise ValueError("native Wan worker result is missing or exceeds its bound")


@dataclass(slots=True)
class _WanWorkerSession:
    supervisor: PersistentWorkerSupervisor
    device: str
    session_binding: str
    secret: bytes
    successful_jobs: int = 0


def _binding(value: Mapping[str, object], secret: bytes) -> str:
    """Stable binding for a command, without trusting a caller supplied hash."""

    return hmac_sha256(value, secret)


def _endpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("native Wan endpoint is not a file")
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _persistent_worker_payload(
    recipe: Wan22RuntimeRequest,
    generation: Any,
    *,
    source_image_path: Path | None,
    end_image_path: Path | None,
    output_path: Path,
    device: str,
    fps: int,
    secret: bytes,
) -> dict[str, object]:
    """Construct a complete command that the child revalidates before use."""

    target = Path(output_path).resolve(strict=False)
    if target.exists():
        raise ValueError("native Wan output path must be fresh")
    session = {
        "recipe": recipe.to_json_dict(),
        "operation": getattr(recipe, "operation", "wan22_i2v_base"),
        "device": str(device),
    }
    unsigned: dict[str, object] = {
        "schema_version": _WORKER_SCHEMA_VERSION,
        **session,
        "session_binding": _binding(session, secret),
        "source_endpoint": _endpoint(source_image_path),
        "end_endpoint": _endpoint(end_image_path),
        "output_path": str(target),
        "fps": fps,
        "generation": {
            "prompt": generation.prompt,
            "negative_prompt": generation.negative_prompt,
            "num_frames": generation.num_frames,
            "height": generation.height,
            "width": generation.width,
            "steps": generation.steps,
            "seed": generation.seed,
            "stage_policy": generation.stage_policy,
            "high_guidance": generation.high_guidance,
            "low_guidance": generation.low_guidance,
        },
    }
    return {**unsigned, "request_binding": _binding(unsigned, secret)}


def _worker_paths(_output_path: Path) -> PersistentWorkerPaths:
    """Create owner-scoped, capability-style IPC outside public artifacts."""

    root = Path(tempfile.mkdtemp(prefix="latentslate-wan14-"))
    try:
        _secure_ipc_directory(root)
    except BaseException:
        root.rmdir()
        raise
    return PersistentWorkerPaths(
        request=root / "request.json",
        result=root / "result.json",
        progress=root / "progress.jsonl",
        heartbeat=root / "heartbeat.jsonl",
        start_gate=root / "start-gate",
        command=root / "command.json",
        cancel=root / "cancel-requested",
    )


def _secure_ipc_directory(root: Path) -> None:
    """Fail closed unless the random IPC capability dir is owner/SYSTEM only."""

    if os.name != "nt":
        try:
            os.chmod(root, 0o700)
        except OSError as exc:
            raise RuntimeError("unable to secure native Wan IPC directory") from exc
        return
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    descriptor = ctypes.c_void_p()
    # Protected DACL: current owner and SYSTEM only, inheritable by every IPC
    # file. No inherited LogonSessionId/users/groups/world access can read
    # prompts, asset paths, or capabilities after request/command creation.
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None
    ):
        raise OSError(ctypes.get_last_error(), "native Wan IPC DACL conversion failed")
    try:
        security_information = 0x00000004 | 0x80000000  # DACL + protected DACL
        if not advapi.SetFileSecurityW(str(root), security_information, descriptor):
            raise OSError(ctypes.get_last_error(), "native Wan IPC DACL application failed")
    finally:
        if kernel.LocalFree(descriptor):
            # The DACL was already applied, but leaking a Windows allocation is
            # not an acceptable long-lived session state.
            raise OSError(ctypes.get_last_error(), "native Wan IPC DACL free failed")


def _require_fresh(paths: PersistentWorkerPaths, *, initial: bool) -> None:
    endpoints = (
        (
            ("request", paths.request),
            ("result", paths.result),
            ("progress", paths.progress),
            ("heartbeat", paths.heartbeat),
            ("gate", paths.start_gate),
            ("command", paths.command),
            ("cancel", paths.cancel),
        )
        if initial
        else (
            ("result", paths.result),
            ("progress", paths.progress),
            ("heartbeat", paths.heartbeat),
            ("command", paths.command),
        )
    )
    stale = [label for label, path in endpoints if path.exists()]
    if stale:
        raise RuntimeError("native Wan worker IPC paths already exist: " + ", ".join(stale))


def _read_persistent_result(
    path: Path, *, expected_output: Path, expected_binding: str, secret: bytes
) -> dict[str, Any]:
    result = _read_json(path)
    if not isinstance(result, dict):
        raise TypeError("native Wan worker returned an invalid result")
    result_binding = result.get("result_binding")
    if not isinstance(result_binding, str) or not hmac.compare_digest(
        result_binding, result_hmac_sha256(result, secret)
    ):
        raise RuntimeError("native Wan worker result authentication is invalid")
    if result.get("ok") is False:
        expected_failure = {
            "schema_version",
            "ok",
            "request_binding",
            "error_type",
            "error",
            "result_binding",
        }
        if (
            set(result) != expected_failure
            or result.get("schema_version") != _WORKER_SCHEMA_VERSION
            or not isinstance(result.get("request_binding"), str)
            or not hmac.compare_digest(result["request_binding"], expected_binding)
            or not isinstance(result.get("error_type"), str)
            or not isinstance(result.get("error"), str)
        ):
            raise RuntimeError("native Wan worker returned an invalid failure result")
        raise RuntimeError(
            f"native Wan worker failed ({result.get('error_type', 'error')}): "
            f"{str(result.get('error', 'unknown failure'))[:4096]}"
        )
    expected = {
        "schema_version",
        "ok",
        "request_binding",
        "output_path",
        "output_size_bytes",
        "stream_metadata",
        "provenance",
        "result_binding",
    }
    if set(result) != expected or result["schema_version"] != _WORKER_SCHEMA_VERSION:
        raise RuntimeError("native Wan worker returned an invalid success result")
    if not isinstance(result["request_binding"], str) or not hmac.compare_digest(
        result["request_binding"], expected_binding
    ):
        raise RuntimeError("native Wan worker result does not bind its command")
    if not isinstance(result["output_path"], str):
        raise TypeError("native Wan worker output path is invalid")
    if Path(result["output_path"]).resolve(strict=True) != expected_output.resolve(strict=True):
        raise RuntimeError("native Wan worker published an unexpected output path")
    size = result["output_size_bytes"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or size != expected_output.stat().st_size
        or not isinstance(result["provenance"], dict)
        or not isinstance(result["stream_metadata"], dict)
    ):
        raise RuntimeError("native Wan worker output/provenance is invalid")
    _validate_worker_provenance(result["provenance"])
    _validate_stream_metadata(result["stream_metadata"])
    return result


def _supervisor(paths: PersistentWorkerPaths, secret: bytes) -> PersistentWorkerSupervisor:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["LATENTSLATE_WAN14_IPC_SECRET"] = secret.hex()
    return PersistentWorkerSupervisor(
        command=(
            sys.executable,
            "-m",
            "latentslate_engine.runtime.wan22_native_worker",
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
    )


def _process(session: _WanWorkerSession):
    worker = session.supervisor.session
    if worker is None:
        raise RuntimeError("native Wan worker session is closed")
    return worker.process


def _consume_progress(value: Mapping[str, Any], callback: Any) -> None:
    if set(value) == {"progress", "message"}:
        # The shared harness emits a terminal transport record. Wan stage
        # progress remains the established completed/total/stage contract.
        return
    if set(value) != {"completed", "total", "stage"}:
        raise ValueError("native Wan worker progress record is invalid")
    completed, total, stage = value["completed"], value["total"], value["stage"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= completed <= total
        or not isinstance(stage, str)
    ):
        raise ValueError("native Wan worker progress values are invalid")
    if callback is not None:
        callback(completed, total, stage)


def _wait_for_result(
    supervisor: PersistentWorkerSupervisor, *, progress: Any, cancelled: Any
) -> None:
    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError

    try:
        supervisor.wait(
            progress=lambda value: _consume_progress(value, progress),
            check_cancelled=check_cancelled,
            policy=PersistentWatchdogPolicy(
                hard_timeout_seconds=_GENERATION_TIMEOUT_SECONDS,
                stage_timeout_seconds=_STAGE_TIMEOUT_SECONDS,
                heartbeat_timeout_seconds=_HEARTBEAT_TIMEOUT_SECONDS,
                cancel_grace_seconds=_CANCEL_GRACE_SECONDS,
                poll_seconds=_POLL_SECONDS,
                maximum_stream_bytes=_MAX_WORKER_PROGRESS_BYTES,
                maximum_stream_records=_MAX_WORKER_PROGRESS_RECORDS,
            ),
        )
    except PersistentWorkerExited as exc:
        raise RuntimeError("native Wan worker exited without a bounded result") from exc
    except PersistentWorkerStreamError as exc:
        raise RuntimeError(f"native Wan worker {exc.stream} stream is invalid") from exc
    except PersistentWorkerTimeout as exc:
        raise RuntimeError(f"native Wan worker {exc.clock} timeout expired") from exc


class ManagedNativeWanI2VRuntime:
    """One exact component/device-bound native Wan worker session.

    CPU-master components live only in the isolated child.  Its normal runtime
    contexts continue to move only active stages to VRAM; successful commands
    keep the child and CPU masters resident, while every failure, timeout, or
    cancellation destroys the complete Job Object tree.
    """

    def __init__(self, request: Wan22RuntimeRequest) -> None:
        self.request = request
        self._session: _WanWorkerSession | None = None
        self._active_supervisor: PersistentWorkerSupervisor | None = None
        self._job_active = False
        self._last_worker: dict[str, object] | None = None
        self._cleanup_errors: list[str] = []

    def generate(
        self,
        generation_request: Any,
        *,
        source_image_path: Path | None,
        end_image_path: Path | None = None,
        output_path: Path,
        device: str,
        fps: int,
        progress: Any = None,
        cancelled: Any = None,
    ) -> NativeWanWorkerResult:
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        if self._job_active:
            raise RuntimeError("native Wan worker is already active")
        if not revalidate_runtime_request(self.request):
            raise RuntimeError("native Wan recipe changed after catalog validation")
        _validate_parent_generation(
            getattr(self.request, "operation", "wan22_i2v_base"),
            generation_request,
            source_image_path,
            end_image_path,
        )
        target = Path(output_path).resolve(strict=False)
        self._job_active = True
        session: _WanWorkerSession | None = self._session
        owns_output = False
        try:
            if session is not None and _process(session).poll() is not None:
                self._discard_dead_session(session)
                session = None
            secret = session.secret if session is not None else secrets.token_bytes(32)
            payload = _persistent_worker_payload(
                self.request,
                generation_request,
                source_image_path=source_image_path,
                end_image_path=end_image_path,
                output_path=output_path,
                device=device,
                fps=fps,
                secret=secret,
            )
            if session is None:
                paths = _worker_paths(target)
                _require_fresh(paths, initial=True)
                supervisor = _supervisor(paths, secret)
                try:
                    supervisor.start(payload)
                except BaseException:
                    self._record_failed_start(supervisor)
                    raise
                session = _WanWorkerSession(
                    supervisor, str(device), str(payload["session_binding"]), secret
                )
                self._session = session
                self._active_supervisor = supervisor
            else:
                if session.device != str(device) or not hmac.compare_digest(
                    session.session_binding, str(payload["session_binding"])
                ):
                    raise RuntimeError("native Wan command does not match the loaded session")
                _require_fresh(session.supervisor.paths, initial=False)
                session.supervisor.send(payload)
            owns_output = True
            _wait_for_result(session.supervisor, progress=progress, cancelled=cancelled)
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            result = _read_persistent_result(
                session.supervisor.paths.result,
                expected_output=target,
                expected_binding=str(payload["request_binding"]),
                secret=secret,
            )
            _validate_worker_provenance_against_request(
                result["provenance"], self.request, expected_seed=generation_request.seed
            )
            _validate_stream_against_request(result["stream_metadata"], generation_request, fps=fps)
            if _process(session).poll() is not None:
                raise RuntimeError(
                    "native Wan worker exited before its success result was accepted"
                )
            warm = session.successful_jobs > 0
            session.successful_jobs += 1
            self._last_worker = {
                "pid": _process(session).pid,
                "exit_code": None,
                "terminated": False,
                "outcome": "succeeded",
                "pipeline_warm": warm,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
            self._cleanup_errors = session.supervisor.cleanup_job()
            return NativeWanWorkerResult(
                target,
                result["output_size_bytes"],
                result["stream_metadata"],
                result["provenance"],
                _process(session).pid,
                None,
                warm,
            )
        except BaseException as primary:
            termination_error: BaseException | None = None
            if session is not None:
                try:
                    self._poison_session(session, primary)
                except BaseException as exc:  # noqa: BLE001 - cleanup proof is authoritative.
                    termination_error = exc
            if owns_output:
                _remove_output_or_note(target, primary)
            _cleanup_owned_encoder_temps(target, primary=primary)
            if termination_error is not None:
                raise termination_error
            raise
        finally:
            self._job_active = False

    def _record_failed_start(self, supervisor: PersistentWorkerSupervisor) -> None:
        state = supervisor.failed_start
        if state is None:
            return
        self._last_worker = {
            "pid": state.pid,
            "exit_code": state.exit_code,
            "terminated": state.terminated,
            "tree_empty": state.tree_empty,
            "outcome": "failed",
            "pipeline_warm": False,
            "memory_boundary": "persistent_exact_recipe_worker",
        }
        self._cleanup_errors = list(state.cleanup_errors)

    def _poison_session(self, session: _WanWorkerSession, primary: BaseException) -> None:
        process = _process(session)
        termination_error: BaseException | None = None
        try:
            session.supervisor.terminate()
            self._last_worker = {
                "pid": process.pid,
                "exit_code": process.poll(),
                "terminated": True,
                "outcome": "canceled" if isinstance(primary, asyncio.CancelledError) else "failed",
                "pipeline_warm": False,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
        except BaseException as exc:  # noqa: BLE001 - cleanup must retain job failure.
            exc.add_note(
                f"while handling original native Wan failure: {type(primary).__name__}: {primary}"
            )
            termination_error = exc
        finally:
            self._session = None
            self._active_supervisor = None
            try:
                session.supervisor.close()
            except BaseException as exc:  # noqa: BLE001
                (termination_error or primary).add_note(
                    f"native Wan persistent worker Job Object close failed: {exc}"
                )
            cleanup = session.supervisor.cleanup_session()
            self._cleanup_errors = cleanup
            if cleanup:
                (termination_error or primary).add_note(
                    "native Wan persistent IPC cleanup failed: " + ", ".join(cleanup)
                )
        if termination_error is not None:
            raise termination_error

    def _discard_dead_session(self, session: _WanWorkerSession) -> None:
        """A dead idle root is never reported warm or reused."""

        try:
            session.supervisor.terminate()
        except BaseException as exc:  # noqa: BLE001 - status/next job remains usable.
            self._cleanup_errors = [*self._cleanup_errors, f"dead_worker:{type(exc).__name__}"][
                -16:
            ]
        finally:
            self._session = None
            self._active_supervisor = None
            try:
                session.supervisor.close()
            except OSError as exc:
                self._cleanup_errors = [*self._cleanup_errors, f"dead_close:{type(exc).__name__}"][
                    -16:
                ]
            self._cleanup_errors = [
                *self._cleanup_errors,
                *session.supervisor.cleanup_session(),
            ][-16:]

    def status(self) -> dict[str, Any]:
        session = self._session
        if session is not None and not self._job_active and _process(session).poll() is not None:
            self._discard_dead_session(session)
        return {
            "family": "wan22",
            "runtime": "engine-native/wan22-persistent-worker",
            "recipe_fingerprint": self.request.fingerprint,
            "loaded": self._session is not None,
            "active_worker": self._job_active,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "components": self.request.public_component_manifest(),
            "cache_support": {"prompt": False, "media": False},
            "cache": {"pipeline_warm": bool(self._session and self._session.successful_jobs > 0)},
        }

    def clear_cache(self) -> None:
        """Clear no model residency: repeat renders deliberately stay warm."""

    def unload(self) -> None:
        session = self._session
        if session is None:
            return
        process = _process(session)
        try:
            session.supervisor.terminate()
            self._last_worker = {
                "pid": process.pid,
                "exit_code": process.poll(),
                "terminated": True,
                "outcome": "unloaded",
                "pipeline_warm": False,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
        finally:
            self._session = None
            self._active_supervisor = None
            try:
                session.supervisor.close()
            finally:
                self._cleanup_errors = session.supervisor.cleanup_session()


def _validate_parent_generation(
    operation: str, generation: Any, source: Path | None, end: Path | None
) -> None:
    if operation.startswith("wan22_t2v_"):
        from .wan22_t2v_runtime import WanT2VRequest, validate_wan_t2v_request

        if not isinstance(generation, WanT2VRequest) or source is not None or end is not None:
            raise TypeError("native Wan T2V requires a text-only generation request")
        validate_wan_t2v_request(generation)
    elif operation.startswith("wan22_flf_"):
        from .wan22_flf_runtime import WanFLFRequest, validate_wan_flf_request

        if not isinstance(generation, WanFLFRequest) or source is None or end is None:
            raise TypeError("native Wan FLF requires start and end images")
        if Path(source).resolve(strict=False) == Path(end).resolve(strict=False):
            raise ValueError("native Wan FLF start and end images must be distinct paths")
        validate_wan_flf_request(generation)
    else:
        from .wan22_i2v_runtime import validate_wan_i2v_request

        if end is not None:
            raise TypeError("native Wan I2V does not accept an end image")
        validate_wan_i2v_request(generation)
