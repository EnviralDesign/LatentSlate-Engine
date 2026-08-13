"""Disposable-process supervisor for the identity-bound native Wan 14B I2V runtime."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..wan22_recipe import Wan22RuntimeRequest, revalidate_runtime_request
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
    | {"shift", "configured_loras", "active_loras", "lora_dispatch"}
)


@dataclass(frozen=True, slots=True)
class NativeWanWorkerResult:
    """Only the bounded data returned after a worker has exited cleanly."""

    output_path: Path
    output_size_bytes: int
    provenance: dict[str, object]
    worker_pid: int
    worker_exit_code: int


class ManagedNativeWanI2VRuntime:
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
        operation = getattr(self.request, "operation", "comfy_i2v_base")
        if operation.startswith("comfy_t2v_"):
            from .wan22_t2v_runtime import WanT2VRequest, validate_wan_t2v_request

            if not isinstance(generation_request, WanT2VRequest) or source_image_path is not None or end_image_path is not None:
                raise TypeError("native Wan T2V requires a text-only generation request")
            validate_wan_t2v_request(generation_request)
        elif operation == "comfy_i2v_flf_base":
            from .wan22_flf_runtime import WanFLFRequest, validate_wan_flf_request

            if not isinstance(generation_request, WanFLFRequest) or source_image_path is None or end_image_path is None:
                raise TypeError("native Wan FLF requires start and end images")
            if Path(source_image_path).resolve(strict=False) == Path(end_image_path).resolve(strict=False):
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
    source = None if source_image_path is None else str(Path(source_image_path).resolve(strict=True))
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
    managed: ManagedNativeWanI2VRuntime,
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    if not path.is_file():
        return offset, pending, records
    if path.stat().st_size > _MAX_WORKER_PROGRESS_BYTES:
        raise ValueError("native Wan worker progress exceeds its aggregate bound")
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
        if records > _MAX_WORKER_PROGRESS_RECORDS:
            raise ValueError("native Wan worker progress exceeds its record bound")
        if len(raw) > 4096:
            raise ValueError("native Wan worker progress record exceeds its bound")
        item = json.loads(raw)
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
    return offset, pending, records


def _read_success_result(path: Path, *, expected_output: Path) -> dict[str, Any]:
    result = _read_json(path)
    if not isinstance(result, dict) or set(result) != {
        "schema_version",
        "ok",
        "output_path",
        "output_size_bytes",
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
    ):
        raise RuntimeError("native Wan worker output/provenance is invalid")
    _validate_worker_provenance(result["provenance"])
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
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or float(shift) != 5.0:
        raise RuntimeError("native Wan worker provenance shift is invalid")
    if not isinstance(value["configured_loras"], list) or not isinstance(value["active_loras"], list):
        raise TypeError("native Wan worker provenance LoRA stacks are invalid")
    dispatch = value["lora_dispatch"]
    if not isinstance(dispatch, dict) or set(dispatch) != {"high", "low"}:
        raise RuntimeError("native Wan worker provenance LoRA dispatch is invalid")
    for stage, item in dispatch.items():
        if not isinstance(item, dict) or set(item) != {"target_module_count", "dispatch_call_count"}:
            raise RuntimeError(f"native Wan worker {stage} LoRA dispatch is invalid")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in item.values()
        ):
            raise RuntimeError(f"native Wan worker {stage} LoRA dispatch is invalid")


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
    if not path.is_file() or path.stat().st_size > _MAX_WORKER_RESULT_BYTES:
        raise ValueError("native Wan worker result is missing or exceeds its bound")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


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
