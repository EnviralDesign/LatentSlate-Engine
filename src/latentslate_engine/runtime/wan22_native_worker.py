"""Private persistent native Wan 14B worker entry point.

The Engine parent never imports the native Wan runtime.  One child owns one
exact recipe, operation, LoRA stack and device; compatible commands reuse that
loaded CPU-master component set while normal runtime contexts stage VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_WORKER_SCHEMA_VERSION = 1
_MAX_RESULT_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024


class _BoundWorkerFailure(RuntimeError):
    """Carry the current command's untrusted binding to the strict failure record."""

    def __init__(self, binding: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.request_binding = binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LatentSlate disposable Wan 14B worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result)
    progress_path = Path(args.progress)
    gate_path = Path(args.start_gate)
    secret = os.environ.pop("LATENTSLATE_WAN14_IPC_SECRET", "")
    binding = ""
    try:
        _wait_for_start_gate(gate_path)
        payload = _read_json(request_path, _MAX_RESULT_BYTES)
        binding = _untrusted_binding(payload)
        return _run_persistent_session(payload, result_path, progress_path, Path(args.command), secret)
    except BaseException as exc:  # noqa: BLE001 - child must publish fatal worker failures.
        _write_json(
            result_path,
            {
                "schema_version": _WORKER_SCHEMA_VERSION,
                "ok": False,
                "request_binding": binding,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
            },
        )
        print(f"native Wan worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _run(payload: Mapping[str, Any], progress_path: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "recipe",
        "source_image_path",
        "end_image_path",
        "output_path",
        "device",
        "fps",
        "generation",
    }:
        raise ValueError("native Wan worker request is not canonical")
    if payload["schema_version"] != _WORKER_SCHEMA_VERSION:
        raise ValueError("native Wan worker request schema_version is unsupported")
    output_path = _absolute_output(payload["output_path"])
    device = payload["device"]
    fps = payload["fps"]
    generation = payload["generation"]
    if (
        not isinstance(device, str)
        or not device
        or fps != 16
        or not isinstance(generation, Mapping)
    ):
        raise ValueError("native Wan worker execution settings are invalid")
    from ..wan22_recipe import rehydrate_native_wan22_i2v_14b_runtime_request
    from .video_output import encode_rgb_video_tensor, validate_encoded_video_stream
    from .wan22_i2v_runtime import WanI2VArtifactPaths

    recipe = rehydrate_native_wan22_i2v_14b_runtime_request(payload["recipe"])
    _validate_fixed_operation(generation, operation=recipe.operation)
    request_kwargs = {
        "prompt": _required_text(generation, "prompt"),
        "negative_prompt": _optional_text(generation, "negative_prompt"),
        "num_frames": _required_int(generation, "num_frames"),
        "height": _required_int(generation, "height"),
        "width": _required_int(generation, "width"),
        "steps": _required_int(generation, "steps"),
        "seed": _required_int(generation, "seed"),
        "stage_policy": _required_text(generation, "stage_policy"),
        "high_guidance": _required_number(generation, "high_guidance"),
        "low_guidance": _required_number(generation, "low_guidance"),
    }
    if recipe.operation.startswith("wan22_t2v_"):
        if payload["source_image_path"] is not None or payload["end_image_path"] is not None:
            raise ValueError("native Wan T2V worker must not receive a source image")
        from .wan22_t2v_runtime import NativeWanT2VRuntime, WanT2VRequest

        request = WanT2VRequest(**request_kwargs)
        runtime_type = NativeWanT2VRuntime
    elif recipe.operation.startswith("wan22_flf_"):
        start_path = _absolute_file(payload["source_image_path"], "source_image_path")
        end_path = _absolute_file(payload["end_image_path"], "end_image_path")
        if start_path == end_path:
            raise ValueError("native Wan FLF start and end images must be distinct paths")
        from .wan22_flf_runtime import NativeWanFLFRuntime, WanFLFRequest

        request = WanFLFRequest(
            start_image=_load_rgb(start_path),
            end_image=_load_rgb(end_path),
            operation=recipe.operation,
            **request_kwargs,
        )
        runtime_type = NativeWanFLFRuntime
    else:
        if payload["end_image_path"] is not None:
            raise ValueError("native Wan I2V worker must not receive an end image")
        source_path = _absolute_file(payload["source_image_path"], "source_image_path")
        from .wan22_i2v_runtime import NativeWanI2VRuntime, WanI2VRequest

        request = WanI2VRequest(image=_load_rgb(source_path), **request_kwargs)
        runtime_type = NativeWanI2VRuntime
    paths = WanI2VArtifactPaths(
        support=recipe.support_plan.root,
        transformer_high=recipe.identities["transformer_high_noise"].path,
        transformer_low=recipe.identities["transformer_low_noise"].path,
        text_encoder=recipe.identities["text_encoder"].path,
        vae=recipe.identities["vae"].path,
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(completed: int, total: int, stage: str) -> None:
        _append_progress(progress_path, {"completed": completed, "total": total, "stage": stage})

    runtime = runtime_type.load(
        paths,
        support_plan=recipe.support_plan,
        adapter_plans=recipe.adapter_plans,
        configured_loras=recipe.configured_loras,
        active_loras=recipe.active_loras,
    )
    try:
        result = runtime.generate(request, device=device, progress=progress)
        encode_rgb_video_tensor(result.video, fps=fps, output_path=output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError("native Wan worker did not publish an MP4")
        stream_metadata = validate_encoded_video_stream(
            output_path,
            width=request.width,
            height=request.height,
            frame_count=request.num_frames,
            fps=fps,
        )
        return {
            "output_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "stream_metadata": stream_metadata,
            "provenance": _public_provenance(result.provenance),
        }
    finally:
        # The child is immediately discarded, but this avoids keeping the video
        # graph alive while result JSON is serialized on an exceptional path.
        runtime.release()


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _public_provenance(provenance: Any) -> dict[str, object]:
    return {
        "support_fingerprint": provenance.support_fingerprint,
        "tokenizer_sha256": provenance.tokenizer_sha256,
        "transformer_high_header_sha256": provenance.transformer_high_header_sha256,
        "transformer_low_header_sha256": provenance.transformer_low_header_sha256,
        "text_encoder_header_sha256": provenance.text_encoder_header_sha256,
        "vae_header_sha256": provenance.vae_header_sha256,
        "transformer_high_contract": provenance.transformer_high_contract,
        "transformer_low_contract": provenance.transformer_low_contract,
        "text_encoder_contract": provenance.text_encoder_contract,
        "stage_policy": provenance.stage_policy,
        "steps": provenance.steps,
        "seed": provenance.seed,
        "sampler": provenance.sampler,
        "scheduler": provenance.scheduler,
        "shift": provenance.shift,
        "transformer_high_size_bytes": provenance.transformer_high_size_bytes,
        "transformer_low_size_bytes": provenance.transformer_low_size_bytes,
        "text_encoder_size_bytes": provenance.text_encoder_size_bytes,
        "vae_size_bytes": provenance.vae_size_bytes,
        "transformer_high_mtime_ns": provenance.transformer_high_mtime_ns,
        "transformer_low_mtime_ns": provenance.transformer_low_mtime_ns,
        "text_encoder_mtime_ns": provenance.text_encoder_mtime_ns,
        "vae_mtime_ns": provenance.vae_mtime_ns,
        "configured_loras": [dict(item) for item in provenance.configured_loras],
        "active_loras": [dict(item) for item in provenance.active_loras],
        "lora_dispatch": {
            stage: dict(value) for stage, value in (provenance.lora_dispatch or {}).items()
        },
        "transformer_dispatch": {
            stage: dict(value) for stage, value in (provenance.transformer_dispatch or {}).items()
        },
    }


def _wait_for_start_gate(path: Path) -> None:
    deadline = time.monotonic() + 60.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("native Wan worker start gate was not opened")
        time.sleep(0.02)


def _read_json(path: Path, maximum: int) -> Any:
    if path.stat().st_size > maximum:
        raise ValueError("native Wan worker JSON exceeds its bounded size")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _append_progress(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if len(encoded.encode("utf-8")) > 4096:
        raise ValueError("native Wan worker progress record exceeds its bound")
    if path.exists() and path.stat().st_size + len(encoded.encode("utf-8")) > _MAX_PROGRESS_BYTES:
        raise ValueError("native Wan worker progress exceeds its aggregate bound")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def _absolute_file(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"native Wan worker {label} is invalid")
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"native Wan worker {label} is not a file")
    return path


def _absolute_output(value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("native Wan worker output_path is invalid")
    path = Path(value).resolve(strict=False)
    if path.suffix.lower() != ".mp4":
        raise ValueError("native Wan worker output_path must be an MP4")
    return path


def _required_text(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"native Wan worker generation {key} is invalid")
    return value


def _optional_text(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"native Wan worker generation {key} is invalid")
    return value


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"native Wan worker generation {key} is invalid")
    return value


def _required_number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"native Wan worker generation {key} is invalid")
    return float(value)


def _validate_fixed_operation(
    generation: Mapping[str, Any], *, operation: str = "wan22_i2v_base"
) -> None:
    """Reject a bypassed caller that would change this built-in recipe's graph."""

    from ..wan22_recipe import wan22_i2v_operation

    expected = {
        key: value
        for key, value in wan22_i2v_operation(operation).items()
        if key in {"steps", "stage_policy", "high_guidance", "low_guidance"}
    }
    for key, value in expected.items():
        if generation.get(key) != value:
            raise ValueError(
                f"native Wan 14B I2V requires the pinned operation {key}={value!r}"
            )


def _command_binding(value: Mapping[str, Any], secret: bytes) -> str:
    unsigned = dict(value)
    unsigned.pop("request_binding", None)
    return hmac.new(
        secret, json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _session_binding(recipe: object, operation: object, device: object, secret: bytes) -> str:
    return hmac.new(
        secret, json.dumps(
            {"recipe": recipe, "operation": operation, "device": device},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _worker_secret(value: str) -> bytes:
    if len(value) != 64:
        raise ValueError("native Wan worker capability secret is invalid")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("native Wan worker capability secret is invalid") from exc


def _untrusted_binding(value: object) -> str:
    if isinstance(value, Mapping) and isinstance(value.get("request_binding"), str):
        return value["request_binding"]
    return ""


def _validate_endpoint(value: object, label: str, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"native Wan worker requires {label}")
        return None
    if not isinstance(value, Mapping) or set(value) != {"path", "size_bytes", "mtime_ns", "sha256"}:
        raise ValueError(f"native Wan worker {label} is invalid")
    path_value, size, mtime, digest = (
        value["path"], value["size_bytes"], value["mtime_ns"], value["sha256"]
    )
    if (
        not isinstance(path_value, str) or isinstance(size, bool) or not isinstance(size, int)
        or isinstance(mtime, bool) or not isinstance(mtime, int)
        or not isinstance(digest, str) or len(digest) != 64
    ):
        raise ValueError(f"native Wan worker {label} is invalid")
    path = Path(path_value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"native Wan worker {label} is not a file")
    stat = path.stat()
    if stat.st_size != size or stat.st_mtime_ns != mtime:
        raise RuntimeError(f"native Wan worker {label} changed after dispatch")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"native Wan worker {label} changed after dispatch")
    return path


def _validate_persistent_payload(
    payload: Mapping[str, Any], secret: bytes
) -> tuple[dict[str, Any], str, Path | None, Path | None, Path]:
    expected = {
        "schema_version", "recipe", "operation", "device", "session_binding",
        "source_endpoint", "end_endpoint", "output_path", "fps", "generation", "request_binding",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("native Wan worker request is not canonical")
    if payload["schema_version"] != _WORKER_SCHEMA_VERSION or payload["fps"] != 16:
        raise ValueError("native Wan worker request schema is unsupported")
    if not isinstance(payload["recipe"], dict) or not isinstance(payload["operation"], str):
        raise TypeError("native Wan worker recipe is invalid")
    if not isinstance(payload["device"], str) or not payload["device"]:
        raise ValueError("native Wan worker device is invalid")
    if not isinstance(payload["session_binding"], str) or payload["session_binding"] != _session_binding(
        payload["recipe"], payload["operation"], payload["device"], secret
    ):
        raise ValueError("native Wan worker session binding is invalid")
    if not isinstance(payload["request_binding"], str) or not hmac.compare_digest(
        payload["request_binding"], _command_binding(payload, secret)
    ):
        raise ValueError("native Wan worker command binding is invalid")
    if not isinstance(payload["generation"], Mapping):
        raise TypeError("native Wan worker generation is invalid")
    _validate_fixed_operation(payload["generation"], operation=payload["operation"])
    output = _absolute_output(payload["output_path"])
    if output.exists():
        raise ValueError("native Wan worker output path must be fresh")
    op = payload["operation"]
    source = _validate_endpoint(payload["source_endpoint"], "source endpoint", required=not op.startswith("wan22_t2v_"))
    end = _validate_endpoint(payload["end_endpoint"], "end endpoint", required=op.startswith("wan22_flf_"))
    if op.startswith("wan22_t2v_") and (source is not None or end is not None):
        raise ValueError("native Wan T2V worker must not receive endpoints")
    if op.startswith("wan22_i2v_") and end is not None:
        raise ValueError("native Wan I2V worker must not receive an end endpoint")
    if op.startswith("wan22_flf_") and source == end:
        raise ValueError("native Wan FLF endpoints must be distinct")
    return dict(payload["generation"]), str(payload["request_binding"]), source, end, output


def _run_persistent_session(
    payload: Mapping[str, Any], result_path: Path, progress_path: Path, command_path: Path, secret_text: str
) -> int:
    """Publish a binding-checked failure result for every command failure."""

    binding = _untrusted_binding(payload)
    try:
        return _run_persistent_session_inner(payload, result_path, progress_path, command_path, secret_text)
    except BaseException as exc:  # noqa: BLE001 - strict bounded child protocol.
        binding = getattr(exc, "request_binding", binding)
        _write_json(result_path, {
            "schema_version": _WORKER_SCHEMA_VERSION, "ok": False, "request_binding": binding,
            "error_type": type(exc).__name__, "error": str(exc)[:4096],
        })
        return 1


def _run_persistent_session_inner(
    payload: Mapping[str, Any], result_path: Path, progress_path: Path, command_path: Path, secret_text: str
) -> int:
    """Load exactly once, then serve serial, immutable-session-bound commands."""

    secret = _worker_secret(secret_text)
    generation, request_binding, source, end, output = _validate_persistent_payload(payload, secret)
    from ..wan22_recipe import rehydrate_native_wan22_i2v_14b_runtime_request
    from .wan22_i2v_runtime import WanI2VArtifactPaths

    recipe = rehydrate_native_wan22_i2v_14b_runtime_request(payload["recipe"])
    if recipe.operation != payload["operation"]:
        raise RuntimeError("native Wan worker operation does not bind its recipe")
    session_identity = (payload["session_binding"], payload["device"], payload["recipe"])
    paths = WanI2VArtifactPaths(
        support=recipe.support_plan.root,
        transformer_high=recipe.identities["transformer_high_noise"].path,
        transformer_low=recipe.identities["transformer_low_noise"].path,
        text_encoder=recipe.identities["text_encoder"].path,
        vae=recipe.identities["vae"].path,
    )
    runtime_type = _runtime_type(recipe.operation)
    runtime = runtime_type.load(
        paths, support_plan=recipe.support_plan, adapter_plans=recipe.adapter_plans,
        configured_loras=recipe.configured_loras, active_loras=recipe.active_loras,
    )
    try:
        _execute_persistent_command(
            runtime, recipe.operation, generation, request_binding, source, end, output,
            str(payload["device"]), result_path, progress_path,
        )
        while True:
            command = _wait_command(command_path)
            command_path.unlink(missing_ok=True)
            command_binding = _untrusted_binding(command)
            try:
                next_generation, next_binding, next_source, next_end, next_output = _validate_persistent_payload(command, secret)
                if (
                    command["session_binding"], command["device"], command["recipe"]
                ) != session_identity:
                    raise ValueError("native Wan worker command does not match its loaded session")
                _execute_persistent_command(
                    runtime, recipe.operation, next_generation, next_binding, next_source, next_end,
                    next_output, str(payload["device"]), result_path, progress_path,
                )
            except BaseException as exc:  # session must poison on command failure.
                raise _BoundWorkerFailure(command_binding, exc) from exc
    finally:
        primary = sys.exc_info()[1]
        try:
            runtime.release()
        except BaseException as exc:  # do not hide model/generation failure.
            if primary is None:
                raise
            primary.add_note(f"native Wan worker runtime release failed: {exc}")


def _runtime_type(operation: str):
    if operation.startswith("wan22_t2v_"):
        from .wan22_t2v_runtime import NativeWanT2VRuntime
        return NativeWanT2VRuntime
    if operation.startswith("wan22_flf_"):
        from .wan22_flf_runtime import NativeWanFLFRuntime
        return NativeWanFLFRuntime
    from .wan22_i2v_runtime import NativeWanI2VRuntime
    return NativeWanI2VRuntime


def _execute_persistent_command(
    runtime: Any, operation: str, generation: Mapping[str, Any], request_binding: str,
    source: Path | None, end: Path | None, output: Path, device: str,
    result_path: Path, progress_path: Path,
) -> None:
    from .video_output import encode_rgb_video_tensor, validate_encoded_video_stream
    request_kwargs = {
        "prompt": _required_text(generation, "prompt"), "negative_prompt": _optional_text(generation, "negative_prompt"),
        "num_frames": _required_int(generation, "num_frames"), "height": _required_int(generation, "height"),
        "width": _required_int(generation, "width"), "steps": _required_int(generation, "steps"),
        "seed": _required_int(generation, "seed"), "stage_policy": _required_text(generation, "stage_policy"),
        "high_guidance": _required_number(generation, "high_guidance"), "low_guidance": _required_number(generation, "low_guidance"),
    }
    if operation.startswith("wan22_t2v_"):
        from .wan22_t2v_runtime import WanT2VRequest
        request = WanT2VRequest(**request_kwargs)
    elif operation.startswith("wan22_flf_"):
        from .wan22_flf_runtime import WanFLFRequest
        request = WanFLFRequest(start_image=_load_rgb(source), end_image=_load_rgb(end), operation=operation, **request_kwargs)
    else:
        from .wan22_i2v_runtime import WanI2VRequest
        request = WanI2VRequest(image=_load_rgb(source), **request_kwargs)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def report(completed: int, total: int, stage: str) -> None:
        _append_progress(progress_path, {"completed": completed, "total": total, "stage": stage})

    generated = runtime.generate(request, device=device, progress=report)
    encode_rgb_video_tensor(generated.video, fps=16, output_path=output)
    stream = validate_encoded_video_stream(output, width=request.width, height=request.height, frame_count=request.num_frames, fps=16)
    _write_json(result_path, {
        "schema_version": _WORKER_SCHEMA_VERSION, "ok": True, "request_binding": request_binding,
        "output_path": str(output), "output_size_bytes": output.stat().st_size,
        "stream_metadata": stream, "provenance": _public_provenance(generated.provenance),
    })


def _wait_command(path: Path) -> Mapping[str, Any]:
    while not path.is_file():
        time.sleep(0.02)
    return _read_json(path, _MAX_RESULT_BYTES)


if __name__ == "__main__":  # pragma: no cover - exercised through the supervisor
    raise SystemExit(main())
