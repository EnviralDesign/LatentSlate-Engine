"""Disposable-process supervisor for the identity-bound native Wan 14B I2V runtime."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..wan22_recipe import Wan22RuntimeRequest, revalidate_runtime_request
from .framework.worker import (
    JsonlCursor,
    WorkerJsonFileError,
    WorkerJsonlFileError,
    atomic_write_json,
    drain_bounded_jsonl,
    hmac_sha256,
    read_bounded_json,
)
from .windows_process import DisposableProcessTree

_WORKER_SCHEMA_VERSION = 1
_MAX_WORKER_RESULT_BYTES = 1024 * 1024
_MAX_WORKER_PROGRESS_BYTES = 1024 * 1024
_MAX_WORKER_PROGRESS_RECORDS = 4096
_POLL_SECONDS = 0.1
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


class _DisposableNativeWanI2VRuntime:
    """Supervise one disposable native Wan process for every generation.

    Wan's two experts and text encoder can leave tens of GiB in the Windows
    native/PyTorch CPU allocator after Python object teardown. The Engine parent
    intentionally never materializes those components; each worker owns loading,
    denoising, and MP4 encoding, then exits. A clean worker exit is the only
    release state this wrapper reports as successful.
    """

    def __init__(self, request: Wan22RuntimeRequest) -> None:
        self.request = request
        self._active_tree: DisposableProcessTree | None = None
        self._last_worker: dict[str, object] | None = None

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
        if self._active_tree is not None:
            raise RuntimeError("native Wan worker is already active")
        if not revalidate_runtime_request(self.request):
            raise RuntimeError("native Wan recipe changed after catalog validation")
        # Preserve the cheap fail-fast boundary before a child is started. This
        # imports only request validation, never a model materializer.
        operation = getattr(self.request, "operation", "wan22_i2v_base")
        if operation.startswith("wan22_t2v_"):
            from .wan22_t2v_runtime import WanT2VRequest, validate_wan_t2v_request

            if (
                not isinstance(generation_request, WanT2VRequest)
                or source_image_path is not None
                or end_image_path is not None
            ):
                raise TypeError("native Wan T2V requires a text-only generation request")
            validate_wan_t2v_request(generation_request)
        elif operation.startswith("wan22_flf_"):
            from .wan22_flf_runtime import WanFLFRequest, validate_wan_flf_request

            if (
                not isinstance(generation_request, WanFLFRequest)
                or source_image_path is None
                or end_image_path is None
            ):
                raise TypeError("native Wan FLF requires start and end images")
            if Path(source_image_path).resolve(strict=False) == Path(end_image_path).resolve(
                strict=False
            ):
                raise ValueError("native Wan FLF start and end images must be distinct paths")
            validate_wan_flf_request(generation_request)
        else:
            from .wan22_i2v_runtime import validate_wan_i2v_request

            if end_image_path is not None:
                raise TypeError("native Wan I2V does not accept an end image")
            validate_wan_i2v_request(generation_request)
        payload = _worker_payload(
            self.request,
            generation_request,
            source_image_path=source_image_path,
            end_image_path=end_image_path,
            output_path=output_path,
            device=device,
            fps=fps,
        )
        paths = _worker_paths(output_path)
        process: subprocess.Popen[bytes] | None = None
        tree: DisposableProcessTree | None = None
        worker_pid: int | None = None
        try:
            _require_fresh_ipc_paths(paths)
            _write_json(paths["request"], payload)
            command = [
                sys.executable,
                "-m",
                "latentslate_engine.runtime.wan22_native_worker",
                "--request",
                str(paths["request"]),
                "--result",
                str(paths["result"]),
                "--progress",
                str(paths["progress"]),
                "--start-gate",
                str(paths["gate"]),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                # Worker failures use the bounded result JSON. Do not retain an
                # unbounded native-library stderr file alongside a job artifact.
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            worker_pid = process.pid
            tree = DisposableProcessTree(process)
            self._active_tree = tree
            # The child does not import a native Wan runtime until this gate is
            # opened, after it is safely attached to the kill-on-close job.
            paths["gate"].touch(exist_ok=False)
            _wait_for_worker(
                process,
                progress_path=paths["progress"],
                progress=progress,
                cancelled=cancelled,
            )
            exit_code = process.wait(timeout=5)
            if exit_code != 0:
                raise RuntimeError(_worker_error(paths["result"], exit_code))
            result = _read_success_result(paths["result"], expected_output=output_path)
            _validate_worker_provenance_against_request(
                result["provenance"], self.request, expected_seed=generation_request.seed
            )
            _validate_stream_against_request(result["stream_metadata"], generation_request, fps=fps)
            # Popen only proves the direct worker exited. Query the Job Object
            # before closing it, so success means every descendant is gone too.
            tree.wait_for_empty()
            tree.close()
            tree = None
            self._active_tree = None
            self._last_worker = {
                "pid": worker_pid,
                "exit_code": exit_code,
                "terminated": True,
                "memory_boundary": "disposable_process_exit",
            }
            return NativeWanWorkerResult(
                output_path=output_path,
                output_size_bytes=result["output_size_bytes"],
                stream_metadata=result["stream_metadata"],
                provenance=result["provenance"],
                worker_pid=worker_pid,
                worker_exit_code=exit_code,
            )
        except asyncio.CancelledError as primary:
            _terminate_or_raise_safety_failure(tree, process, primary)
            _record_worker_or_note(self, worker_pid, process, primary)
            _remove_output_or_note(output_path, primary)
            raise
        except BaseException as primary:
            _terminate_or_raise_safety_failure(tree, process, primary)
            _record_worker_or_note(self, worker_pid, process, primary)
            _remove_output_or_note(output_path, primary)
            raise
        finally:
            self._active_tree = None
            if tree is not None:
                _close_tree_preserving_primary_exception(tree)
            _cleanup_owned_encoder_temps(output_path, primary=sys.exc_info()[1])
            _cleanup_ipc_paths(paths)

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "runtime": "native_wan_i2v_14b_disposable_worker",
            "recipe_fingerprint": self.request.fingerprint,
            # There is no parent-held heavyweight runtime. `loaded=false` means
            # exactly that, while active_worker identifies the live job process.
            "loaded": False,
            "active_worker": self._active_tree is not None,
            "last_worker": self._last_worker,
            "components": self.request.public_component_manifest(),
            "cache_support": {"prompt": False, "media": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        """The disposable native worker has no parent-side tensor cache."""

    def unload(self) -> None:
        """Terminate any live worker tree; no model state exists in this process."""

        tree = self._active_tree
        if tree is not None:
            termination_error: BaseException | None = None
            try:
                # Do not mistake a dead root for an empty worker tree: ffmpeg
                # or another inheriting descendant can still own the large job.
                _terminate_worker(tree, tree.process)
            except BaseException as exc:  # noqa: BLE001 - tree-empty proof is authoritative.
                termination_error = exc
            finally:
                try:
                    tree.close()
                except OSError as close_error:
                    if termination_error is None:
                        raise
                    termination_error.add_note(
                        f"native Wan worker Job Object close also failed: {close_error}"
                    )
                self._active_tree = None
            if termination_error is not None:
                raise termination_error

    def _record_terminated_worker(
        self,
        worker_pid: int | None,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        if worker_pid is None or process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            raise RuntimeError("native Wan worker termination was not confirmed")
        self._last_worker = {
            "pid": worker_pid,
            "exit_code": exit_code,
            "terminated": True,
            "memory_boundary": "disposable_process_exit",
        }


def _worker_payload(
    recipe: Wan22RuntimeRequest,
    generation: Any,
    *,
    source_image_path: Path | None,
    end_image_path: Path | None,
    output_path: Path,
    device: str,
    fps: int,
) -> dict[str, object]:
    source = (
        None if source_image_path is None else str(Path(source_image_path).resolve(strict=True))
    )
    end = None if end_image_path is None else str(Path(end_image_path).resolve(strict=True))
    target = Path(output_path).resolve(strict=False)
    return {
        "schema_version": _WORKER_SCHEMA_VERSION,
        "recipe": recipe.to_json_dict(),
        "source_image_path": source,
        "end_image_path": end,
        "output_path": str(target),
        "device": str(device),
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


def _worker_paths(output_path: Path) -> dict[str, Path]:
    root = Path(output_path).parent
    root.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem
    return {
        "request": root / f".{stem}.wan14-worker-request.json",
        "result": root / f".{stem}.wan14-worker-result.json",
        "progress": root / f".{stem}.wan14-worker-progress.jsonl",
        "gate": root / f".{stem}.wan14-worker-start-gate",
    }


def _require_fresh_ipc_paths(paths: Mapping[str, Path]) -> None:
    stale = [key for key, path in paths.items() if path.exists()]
    if stale:
        raise RuntimeError("native Wan worker IPC paths already exist: " + ", ".join(sorted(stale)))


def _cleanup_ipc_paths(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Private IPC cleanup must not replace a generation/cancellation
            # result. A later unique job cannot consume these stale paths.
            pass


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


def _close_tree_preserving_primary_exception(tree: DisposableProcessTree) -> None:
    """Do not obscure a generation error with a post-terminal handle-close error."""

    primary = sys.exc_info()[1]
    try:
        tree.close()
    except OSError as exc:
        if primary is None:
            raise
        primary.add_note(f"native Wan worker Job Object close also failed: {exc}")


def _terminate_or_raise_safety_failure(
    tree: DisposableProcessTree | None,
    process: subprocess.Popen[bytes] | None,
    primary: BaseException,
) -> None:
    """Keep a failed tree-empty proof primary, with the original failure retained."""

    try:
        _terminate_worker(tree, process)
    except BaseException as safety_error:
        safety_error.add_note(
            f"while handling original native Wan failure: {type(primary).__name__}: {primary}"
        )
        raise


def _record_worker_or_note(
    managed: _DisposableNativeWanI2VRuntime,
    worker_pid: int | None,
    process: subprocess.Popen[bytes] | None,
    primary: BaseException,
) -> None:
    try:
        managed._record_terminated_worker(worker_pid, process)
    except Exception as exc:  # noqa: BLE001 - terminal proof has already succeeded.
        primary.add_note(f"native Wan worker termination metadata was unavailable: {exc}")


def _remove_output_or_note(output_path: Path, primary: BaseException) -> None:
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        primary.add_note(f"native Wan partial output cleanup failed: {exc}")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_write_json(path, value)


def _wait_for_worker(
    process: subprocess.Popen[bytes],
    *,
    progress_path: Path,
    progress: Any,
    cancelled: Any,
) -> None:
    offset = 0
    records = 0
    pending = b""
    while process.poll() is None:
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        offset, pending, records = _drain_progress(
            progress_path,
            offset,
            pending,
            records,
            progress,
        )
        time.sleep(_POLL_SECONDS)
    _offset, pending, _records = _drain_progress(
        progress_path,
        offset,
        pending,
        records,
        progress,
    )
    if pending and process.poll() == 0:
        raise ValueError("native Wan worker ended with a truncated progress record")


def _drain_progress(
    path: Path,
    offset: int,
    pending: bytes,
    records: int,
    callback: Any,
) -> tuple[int, bytes, int]:
    try:
        cursor, items = drain_bounded_jsonl(
            path,
            JsonlCursor(offset=offset, pending=pending, records=records),
            maximum_bytes=_MAX_WORKER_PROGRESS_BYTES,
            maximum_records=_MAX_WORKER_PROGRESS_RECORDS,
        )
    except WorkerJsonlFileError as exc:
        raise ValueError("native Wan worker progress is invalid or exceeds its bound") from exc
    for item in items:
        if not isinstance(item, dict) or set(item) != {"completed", "total", "stage"}:
            raise ValueError("native Wan worker progress record is invalid")
        completed = item["completed"]
        total = item["total"]
        stage = item["stage"]
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
    return cursor.offset, cursor.pending, cursor.records


def _read_success_result(path: Path, *, expected_output: Path) -> dict[str, Any]:
    result = _read_json(path)
    if not isinstance(result, dict) or set(result) != {
        "schema_version",
        "ok",
        "output_path",
        "output_size_bytes",
        "stream_metadata",
        "provenance",
    }:
        raise RuntimeError("native Wan worker returned an invalid success result")
    if result["schema_version"] != _WORKER_SCHEMA_VERSION or result["ok"] is not True:
        raise RuntimeError("native Wan worker did not report success")
    if not isinstance(result["output_path"], str):
        raise TypeError("native Wan worker output path is invalid")
    if Path(result["output_path"]).resolve(strict=True) != expected_output.resolve(strict=True):
        raise RuntimeError("native Wan worker published an unexpected output path")
    if (
        isinstance(result["output_size_bytes"], bool)
        or not isinstance(result["output_size_bytes"], int)
        or result["output_size_bytes"] != expected_output.stat().st_size
        or result["output_size_bytes"] <= 0
        or not isinstance(result["provenance"], dict)
        or not isinstance(result["stream_metadata"], dict)
    ):
        raise RuntimeError("native Wan worker output/provenance is invalid")
    _validate_worker_provenance(result["provenance"])
    _validate_stream_metadata(result["stream_metadata"])
    return result


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
            "fp8_module_count", "fp8_modules", "int8_module_count", "int8_modules",
            "dense_fallback_count", "rejected_count",
        }:
            raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
        for key in ("fp8_module_count", "int8_module_count", "dense_fallback_count", "rejected_count"):
            if isinstance(proof[key], bool) or not isinstance(proof[key], int) or proof[key] < 0:
                raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
        for kind, delta_key in (("fp8_modules", "native_dispatch_delta"), ("int8_modules", "int8_dispatch_delta")):
            modules = proof[kind]
            if not isinstance(modules, dict) or len(modules) != proof[f"{kind[:-8]}_module_count"]:
                raise RuntimeError(f"native Wan worker {stage} transformer dispatch is invalid")
            for name, counts in modules.items():
                if not isinstance(name, str) or not name or not isinstance(counts, dict) or set(counts) != {delta_key, "rejected_delta", "dense_fallback_delta"}:
                    raise RuntimeError(f"native Wan worker {stage} transformer module proof is invalid")
                if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values()):
                    raise RuntimeError(f"native Wan worker {stage} transformer module proof is invalid")
                if counts[delta_key] <= 0 or counts["rejected_delta"] != 0 or counts["dense_fallback_delta"] != 0:
                    raise RuntimeError(f"native Wan worker {stage} transformer module proof is not clean")
        if proof["dense_fallback_count"] != 0 or proof["rejected_count"] != 0:
            raise RuntimeError(f"native Wan worker {stage} transformer proof has fallback or rejection")


def _validate_stream_metadata(value: Mapping[str, object]) -> None:
    required = {
        "width", "height", "frame_count", "fps", "duration_seconds", "has_audio", "codec_name",
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


def _validate_stream_against_request(value: Mapping[str, object], request: Any, *, fps: int) -> None:
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
            raise RuntimeError(f"native Wan worker {stage} transformer module proof does not bind its plan")
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


def _worker_error(result_path: Path, exit_code: int) -> str:
    try:
        result = _read_json(result_path)
        if (
            isinstance(result, dict)
            and result.get("schema_version") == _WORKER_SCHEMA_VERSION
            and result.get("ok") is False
            and isinstance(result.get("error"), str)
        ):
            return (
                f"native Wan worker failed ({result.get('error_type', 'error')}): {result['error']}"
            )
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return f"native Wan worker exited with code {exit_code} without a valid result"


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_WORKER_RESULT_BYTES)
    except WorkerJsonFileError:
        raise ValueError("native Wan worker result is missing or exceeds its bound")


def _terminate_worker(
    tree: DisposableProcessTree | None,
    process: subprocess.Popen[bytes] | None,
) -> None:
    active = tree.process if tree is not None else process
    if active is None:
        return
    if tree is not None:
        # The root may already have exited while a descendant is still running.
        # The Job Object accounting is authoritative for that case.
        if tree.active_processes() != 0:
            tree.terminate()
    elif active.poll() is None:
        active.terminate()
    if active.poll() is None:
        try:
            active.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # A second Job Object termination is intentional: do not let the API
            # report cancellation/failure while an orphaned model worker survives.
            if tree is not None:
                tree.terminate()
            else:
                active.kill()
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "native Wan worker did not terminate after cancellation"
                ) from exc
        if active.poll() is None:
            raise RuntimeError("native Wan worker termination was not confirmed")
    if tree is not None:
        tree.wait_for_empty()


# A native Wan component set is too large to rebuild for each render.  The
# original one-shot supervisor above is intentionally left as a source-history
# compatible set of validation helpers; the definitions below replace its
# disposable entry point with an exact, private, long-lived session.


@dataclass(slots=True)
class _WanWorkerSession:
    process: subprocess.Popen[bytes]
    tree: DisposableProcessTree
    paths: dict[str, Path]
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


def _worker_paths(_output_path: Path) -> dict[str, Path]:
    """Create owner-scoped, capability-style IPC outside public artifacts."""

    root = Path(tempfile.mkdtemp(prefix="latentslate-wan14-"))
    try:
        _secure_ipc_directory(root)
    except BaseException:
        root.rmdir()
        raise
    return {
        "request": root / "request.json",
        "result": root / "result.json",
        "progress": root / "progress.jsonl",
        "gate": root / "start-gate",
        "command": root / "command.json",
    }


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


def _cleanup_persistent_job(paths: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    for key in ("command", "result", "progress"):
        try:
            paths[key].unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{key}:{type(exc).__name__}")
        for temporary in paths[key].parent.glob(f".{paths[key].name}.*.tmp"):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{key}_tmp:{type(exc).__name__}")
    return errors


def _cleanup_persistent_session(paths: Mapping[str, Path]) -> list[str]:
    errors = _cleanup_persistent_job(paths)
    for key in ("request", "gate"):
        try:
            paths[key].unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{key}:{type(exc).__name__}")
        for temporary in paths[key].parent.glob(f".{paths[key].name}.*.tmp"):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{key}_tmp:{type(exc).__name__}")
    root = paths["request"].parent
    try:
        root.rmdir()
    except OSError as exc:
        errors.append(f"root:{type(exc).__name__}")
    return errors[-16:]


def _require_fresh_job(paths: Mapping[str, Path]) -> None:
    stale = [key for key in ("command", "result", "progress") if paths[key].exists()]
    if stale:
        raise RuntimeError("native Wan worker job IPC paths already exist: " + ", ".join(stale))


def _read_persistent_result(
    path: Path, *, expected_output: Path, expected_binding: str
) -> dict[str, Any]:
    result = _read_json(path)
    if not isinstance(result, dict):
        raise TypeError("native Wan worker returned an invalid result")
    if result.get("ok") is False:
        expected_failure = {
            "schema_version", "ok", "request_binding", "error_type", "error"
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
        "schema_version", "ok", "request_binding", "output_path", "output_size_bytes",
        "stream_metadata", "provenance",
    }
    if set(result) != expected or result["schema_version"] != _WORKER_SCHEMA_VERSION:
        raise RuntimeError("native Wan worker returned an invalid success result")
    if result["request_binding"] != expected_binding:
        raise RuntimeError("native Wan worker result does not bind its command")
    # Reuse the established output/provenance and stream validators after
    # checking the extra session command binding.
    legacy = dict(result)
    legacy.pop("request_binding")
    temporary = path.with_suffix(".legacy-check.json")
    try:
        _write_json(temporary, legacy)
        return _read_success_result(temporary, expected_output=expected_output)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_persistent_result(
    session: _WanWorkerSession,
    *,
    progress: Any,
    cancelled: Any,
) -> None:
    offset = 0
    records = 0
    pending = b""
    while not session.paths["result"].is_file():
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        if session.process.poll() is not None:
            raise RuntimeError(_worker_error(session.paths["result"], session.process.poll() or 1))
        offset, pending, records = _drain_progress(
            session.paths["progress"], offset, pending, records, progress
        )
        time.sleep(_POLL_SECONDS)
    _offset, pending, _records = _drain_progress(
        session.paths["progress"], offset, pending, records, progress
    )
    if pending:
        raise ValueError("native Wan worker ended with a truncated progress record")


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
        self._active_tree: DisposableProcessTree | None = None
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
            generation_request, source_image_path, end_image_path,
        )
        target = Path(output_path).resolve(strict=False)
        self._job_active = True
        session: _WanWorkerSession | None = self._session
        owns_output = False
        try:
            if session is not None and session.process.poll() is not None:
                self._discard_dead_session(session)
                session = None
            secret = session.secret if session is not None else secrets.token_bytes(32)
            payload = _persistent_worker_payload(
                self.request, generation_request, source_image_path=source_image_path,
                end_image_path=end_image_path, output_path=output_path, device=device, fps=fps,
                secret=secret,
            )
            if session is None:
                paths = _worker_paths(target)
                _require_fresh_ipc_paths(paths)
                _write_json(paths["request"], payload)
                try:
                    session = self._spawn_session(
                        paths, str(device), str(payload["session_binding"]), secret
                    )
                except BaseException:
                    self._cleanup_errors = _cleanup_persistent_session(paths)
                    raise
                self._session = session
                self._active_tree = session.tree
                paths["gate"].touch(exist_ok=False)
            else:
                if session.device != str(device) or session.session_binding != payload["session_binding"]:
                    raise RuntimeError("native Wan command does not match the loaded session")
                _require_fresh_job(session.paths)
                _write_json(session.paths["command"], payload)
            owns_output = True
            _wait_for_persistent_result(session, progress=progress, cancelled=cancelled)
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            result = _read_persistent_result(
                session.paths["result"], expected_output=target,
                expected_binding=str(payload["request_binding"]),
            )
            _validate_worker_provenance_against_request(
                result["provenance"], self.request, expected_seed=generation_request.seed
            )
            _validate_stream_against_request(result["stream_metadata"], generation_request, fps=fps)
            if session.process.poll() is not None:
                raise RuntimeError("native Wan worker exited before its success result was accepted")
            warm = session.successful_jobs > 0
            session.successful_jobs += 1
            self._last_worker = {
                "pid": session.process.pid, "exit_code": None, "terminated": False,
                "outcome": "succeeded", "pipeline_warm": warm,
                "memory_boundary": "persistent_exact_recipe_worker",
            }
            self._cleanup_errors = _cleanup_persistent_job(session.paths)
            return NativeWanWorkerResult(
                target, result["output_size_bytes"], result["stream_metadata"], result["provenance"],
                session.process.pid, None, warm,
            )
        except BaseException as primary:
            if session is not None:
                self._poison_session(session, primary)
            if owns_output:
                _remove_output_or_note(target, primary)
            _cleanup_owned_encoder_temps(target, primary=primary)
            raise
        finally:
            self._job_active = False

    def _spawn_session(
        self, paths: dict[str, Path], device: str, binding: str, secret: bytes
    ) -> _WanWorkerSession:
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["LATENTSLATE_WAN14_IPC_SECRET"] = secret.hex()
        process = subprocess.Popen(
            [sys.executable, "-m", "latentslate_engine.runtime.wan22_native_worker",
             "--request", str(paths["request"]), "--result", str(paths["result"]),
             "--progress", str(paths["progress"]), "--start-gate", str(paths["gate"]),
             "--command", str(paths["command"])],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=env,
        )
        try:
            tree = DisposableProcessTree(process)
        except BaseException:
            _terminate_worker(None, process)
            raise
        return _WanWorkerSession(process, tree, paths, device, binding, secret)

    def _poison_session(self, session: _WanWorkerSession, primary: BaseException) -> None:
        try:
            _terminate_worker(session.tree, session.process)
            self._last_worker = {
                "pid": session.process.pid, "exit_code": session.process.poll(), "terminated": True,
                "outcome": "canceled" if isinstance(primary, asyncio.CancelledError) else "failed",
                "pipeline_warm": False, "memory_boundary": "persistent_exact_recipe_worker",
            }
        except BaseException as exc:  # noqa: BLE001 - cleanup must retain job failure.
            primary.add_note(f"native Wan persistent worker termination failed: {exc}")
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            except BaseException as exc:  # noqa: BLE001
                primary.add_note(f"native Wan persistent worker Job Object close failed: {exc}")
            cleanup = _cleanup_persistent_session(session.paths)
            self._cleanup_errors = cleanup
            if cleanup:
                primary.add_note("native Wan persistent IPC cleanup failed: " + ", ".join(cleanup))

    def _discard_dead_session(self, session: _WanWorkerSession) -> None:
        """A dead idle root is never reported warm or reused."""

        try:
            _terminate_worker(session.tree, session.process)
        except BaseException as exc:  # noqa: BLE001 - status/next job remains usable.
            self._cleanup_errors = [*self._cleanup_errors, f"dead_worker:{type(exc).__name__}"][-16:]
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            except OSError as exc:
                self._cleanup_errors = [*self._cleanup_errors, f"dead_close:{type(exc).__name__}"][-16:]
            self._cleanup_errors = [*self._cleanup_errors, *_cleanup_persistent_session(session.paths)][-16:]

    def status(self) -> dict[str, Any]:
        session = self._session
        if session is not None and not self._job_active and session.process.poll() is not None:
            self._discard_dead_session(session)
        return {
            "family": "wan22", "runtime": "engine-native/wan22-persistent-worker",
            "recipe_fingerprint": self.request.fingerprint,
            "loaded": self._session is not None, "active_worker": self._job_active,
            "last_worker": self._last_worker, "cleanup_errors": list(self._cleanup_errors),
            "components": self.request.public_component_manifest(),
            "cache_support": {"prompt": False, "media": False},
            "cache": {"pipeline_warm": bool(self._session and self._session.successful_jobs > 0)},
        }

    def clear_cache(self) -> None:
        """Clear no model residency: repeat renders deliberately stay warm."""

    def unload(self) -> None:
        session = getattr(self, "_session", None)
        if session is None:
            # Compatibility/safety path for a partially constructed session:
            # never abandon a Job Object merely because bookkeeping was
            # interrupted before _session was published.
            tree = self._active_tree
            if tree is not None:
                primary: BaseException | None = None
                try:
                    _terminate_worker(tree, tree.process)
                except BaseException as exc:
                    primary = exc
                    raise
                finally:
                    self._active_tree = None
                    try:
                        tree.close()
                    except OSError as exc:
                        if primary is None:
                            raise
                        primary.add_note(f"native Wan persistent worker Job Object close failed: {exc}")
            return
        primary: BaseException | None = None
        try:
            _terminate_worker(session.tree, session.process)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._session = None
            self._active_tree = None
            try:
                session.tree.close()
            except OSError as exc:
                if primary is None:
                    raise
                primary.add_note(f"native Wan persistent worker Job Object close failed: {exc}")
            finally:
                self._cleanup_errors = _cleanup_persistent_session(session.paths)


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
