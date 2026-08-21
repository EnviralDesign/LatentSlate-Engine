"""Private persistent native Wan 14B worker entry point.

The Engine parent never imports the native Wan runtime.  One child owns one
exact recipe, operation, LoRA stack and device; compatible commands reuse that
loaded CPU-master component set while normal runtime contexts stage VRAM.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework.worker import (
    PersistentChildContext,
    hmac_sha256,
    parse_persistent_child_paths,
    result_hmac_sha256,
    run_persistent_child,
)

_WORKER_SCHEMA_VERSION = 1
_MAX_RESULT_BYTES = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    paths = parse_persistent_child_paths(argv, description="LatentSlate persistent Wan 14B worker")
    secret = _worker_secret(os.environ.pop("LATENTSLATE_WAN14_IPC_SECRET", ""))
    return run_persistent_child(
        paths,
        _WanWorkerHandler(secret),
        maximum_bytes=_MAX_RESULT_BYTES,
    )


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
            raise ValueError(f"native Wan 14B I2V requires the pinned operation {key}={value!r}")


def _command_binding(value: Mapping[str, Any], secret: bytes) -> str:
    unsigned = dict(value)
    unsigned.pop("request_binding", None)
    return hmac_sha256(unsigned, secret)


def _session_binding(recipe: object, operation: object, device: object, secret: bytes) -> str:
    return hmac_sha256(
        {"recipe": recipe, "operation": operation, "device": device},
        secret,
    )


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
        value["path"],
        value["size_bytes"],
        value["mtime_ns"],
        value["sha256"],
    )
    if (
        not isinstance(path_value, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or isinstance(mtime, bool)
        or not isinstance(mtime, int)
        or not isinstance(digest, str)
        or len(digest) != 64
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
        "schema_version",
        "recipe",
        "operation",
        "device",
        "session_binding",
        "source_endpoint",
        "end_endpoint",
        "output_path",
        "fps",
        "generation",
        "request_binding",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("native Wan worker request is not canonical")
    if payload["schema_version"] != _WORKER_SCHEMA_VERSION or payload["fps"] != 16:
        raise ValueError("native Wan worker request schema is unsupported")
    if not isinstance(payload["recipe"], dict) or not isinstance(payload["operation"], str):
        raise TypeError("native Wan worker recipe is invalid")
    if not isinstance(payload["device"], str) or not payload["device"]:
        raise ValueError("native Wan worker device is invalid")
    expected_session_binding = _session_binding(
        payload["recipe"], payload["operation"], payload["device"], secret
    )
    if not isinstance(payload["session_binding"], str) or not hmac.compare_digest(
        payload["session_binding"], expected_session_binding
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
    source = _validate_endpoint(
        payload["source_endpoint"], "source endpoint", required=not op.startswith("wan22_t2v_")
    )
    end = _validate_endpoint(
        payload["end_endpoint"], "end endpoint", required=op.startswith("wan22_flf_")
    )
    if op.startswith("wan22_t2v_") and (source is not None or end is not None):
        raise ValueError("native Wan T2V worker must not receive endpoints")
    if op.startswith("wan22_i2v_") and end is not None:
        raise ValueError("native Wan I2V worker must not receive an end endpoint")
    if op.startswith("wan22_flf_") and source == end:
        raise ValueError("native Wan FLF endpoints must be distinct")
    return dict(payload["generation"]), str(payload["request_binding"]), source, end, output


@dataclass(frozen=True, slots=True)
class _WanCommand:
    payload: dict[str, Any]
    generation: dict[str, Any]
    request_binding: str
    source: Path | None
    end: Path | None
    output: Path


@dataclass(slots=True)
class _WanSession:
    runtime: Any
    recipe: Any
    identity: tuple[object, object, object]


class _WanWorkerHandler:
    """Wan-specific adapter for the model-neutral persistent child loop."""

    def __init__(self, secret: bytes) -> None:
        self.secret = secret

    def protocol_error(self, reason: str) -> BaseException:
        return ValueError(f"native Wan worker protocol violation: {reason}")

    def bind_initial(self, payload: Any, context: PersistentChildContext) -> _WanCommand:
        return self._bind(payload, context)

    def bind_command(
        self, payload: Any, session: _WanSession, context: PersistentChildContext
    ) -> _WanCommand:
        command = self._bind(payload, context)
        identity = (
            command.payload["session_binding"],
            command.payload["device"],
            command.payload["recipe"],
        )
        if identity != session.identity:
            raise ValueError("native Wan worker command does not match its loaded session")
        return command

    def _bind(self, payload: Any, context: PersistentChildContext) -> _WanCommand:
        context.binding = _untrusted_binding(payload)
        if not isinstance(payload, Mapping):
            raise TypeError("native Wan worker request is not canonical")
        generation, binding, source, end, output = _validate_persistent_payload(
            payload, self.secret
        )
        context.binding = binding
        return _WanCommand(dict(payload), generation, binding, source, end, output)

    def load(self, command: _WanCommand, context: PersistentChildContext) -> _WanSession:
        def load_progress(completed: int, total: int, stage: str) -> None:
            _publish_stage(context, completed, total, stage)

        load_progress(1, 1000, "Validating native Wan request")
        load_progress(2, 1000, "Rehydrating native Wan recipe")
        from ..wan22_recipe import rehydrate_native_wan22_i2v_14b_runtime_request
        from .wan22_i2v_runtime import WanI2VArtifactPaths

        payload = command.payload
        recipe = rehydrate_native_wan22_i2v_14b_runtime_request(payload["recipe"])
        if recipe.operation != payload["operation"]:
            raise RuntimeError("native Wan worker operation does not bind its recipe")
        paths = WanI2VArtifactPaths(
            support=recipe.support_plan.root,
            transformer_high=recipe.identities["transformer_high_noise"].path,
            transformer_low=recipe.identities["transformer_low_noise"].path,
            text_encoder=recipe.identities["text_encoder"].path,
            vae=recipe.identities["vae"].path,
        )
        load_progress(3, 1000, "Importing native Wan runtime")
        runtime = _runtime_type(recipe.operation).load(
            paths,
            support_plan=recipe.support_plan,
            adapter_plans=recipe.adapter_plans,
            configured_loras=recipe.configured_loras,
            active_loras=recipe.active_loras,
            load_progress=load_progress,
        )
        return _WanSession(
            runtime,
            recipe,
            (payload["session_binding"], payload["device"], payload["recipe"]),
        )

    def execute(
        self,
        session: _WanSession,
        command: _WanCommand,
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> Mapping[str, Any]:
        del cold
        result = _execute_persistent_command(
            session.runtime,
            session.recipe.operation,
            command.generation,
            command.request_binding,
            command.source,
            command.end,
            command.output,
            str(command.payload["device"]),
            context,
        )
        return _authenticated_result(result, self.secret)

    def unload(self, session: _WanSession, context: PersistentChildContext) -> None:
        del context
        session.runtime.release()

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]:
        error_type, message = _safe_failure(exc)
        return _authenticated_result(
            {
                "schema_version": _WORKER_SCHEMA_VERSION,
                "ok": False,
                "request_binding": context.binding,
                "error_type": error_type,
                "error": message,
            },
            self.secret,
        )


def _authenticated_result(value: Mapping[str, Any], secret: bytes) -> dict[str, Any]:
    result = dict(value)
    result["result_binding"] = result_hmac_sha256(result, secret)
    return result


def _safe_failure(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, asyncio.CancelledError):
        return "CancelledError", "native Wan generation was canceled"
    if isinstance(exc, MemoryError):
        return "MemoryError", "native Wan worker ran out of memory"
    if isinstance(exc, TimeoutError):
        return "TimeoutError", "native Wan worker timed out"
    if isinstance(exc, (TypeError, ValueError)):
        return type(exc).__name__, "native Wan worker request is invalid"
    return "RuntimeError", "native Wan generation failed"


def _publish_stage(context: PersistentChildContext, completed: int, total: int, stage: str) -> None:
    if context.paths.cancel.is_file():
        raise asyncio.CancelledError
    context.publish_progress_record(
        {"completed": completed, "total": total, "stage": stage},
    )


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
    runtime: Any,
    operation: str,
    generation: Mapping[str, Any],
    request_binding: str,
    source: Path | None,
    end: Path | None,
    output: Path,
    device: str,
    context: PersistentChildContext,
) -> dict[str, Any]:
    from .video_output import encode_rgb_video_tensor, validate_encoded_video_stream

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
    if operation.startswith("wan22_t2v_"):
        from .wan22_t2v_runtime import WanT2VRequest

        request = WanT2VRequest(**request_kwargs)
    elif operation.startswith("wan22_flf_"):
        from .wan22_flf_runtime import WanFLFRequest

        request = WanFLFRequest(
            start_image=_load_rgb(source),
            end_image=_load_rgb(end),
            operation=operation,
            **request_kwargs,
        )
    else:
        from .wan22_i2v_runtime import WanI2VRequest

        request = WanI2VRequest(image=_load_rgb(source), **request_kwargs)

    def report(completed: int, total: int, stage: str) -> None:
        _publish_stage(context, completed, total, stage)

    generated = runtime.generate(request, device=device, progress=report)
    if context.paths.cancel.is_file():
        raise asyncio.CancelledError
    encode_rgb_video_tensor(generated.video, fps=16, output_path=output)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("native Wan worker did not publish an MP4")
    stream = validate_encoded_video_stream(
        output,
        width=request.width,
        height=request.height,
        frame_count=request.num_frames,
        fps=16,
    )
    return {
        "schema_version": _WORKER_SCHEMA_VERSION,
        "ok": True,
        "request_binding": request_binding,
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "stream_metadata": stream,
        "provenance": _public_provenance(generated.provenance),
    }


if __name__ == "__main__":  # pragma: no cover - exercised through the supervisor
    raise SystemExit(main())
