"""One-shot Engine-native Wan 2.2 TI2V 5B generation worker."""

from __future__ import annotations

import hashlib
import os
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework.worker import (
    DisposableChildContext,
    parse_disposable_child_paths,
    run_disposable_child,
    sha256_fingerprint,
)

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _BoundRequest:
    request: Any
    generation: dict[str, Any]
    device: str
    binding: str


class _Wan5KitchenHandler:
    """Wan-specific request, runtime, result, and diagnostic contract."""

    def bind_request(self, payload: Any, context: DisposableChildContext) -> _BoundRequest:
        context.stage = "validate_bound_request"
        binding = _validate_binding(payload)
        context.binding = binding
        context.stage = "rehydrate_recipe"
        from ..wan5_kitchen_recipe import rehydrate_wan5_kitchen_runtime_request

        request = rehydrate_wan5_kitchen_runtime_request(payload["request"])
        generation = _validate_generation(payload["generation"], request.operation)
        return _BoundRequest(request, generation, payload["device"], binding)

    def load(self, request: _BoundRequest, context: DisposableChildContext) -> Any:
        # Heavy dependencies stay behind complete request and endpoint validation.
        context.stage = "import_runtime"
        from .wan5_kitchen import Wan5KitchenRuntime

        context.stage = "initialize_runtime"
        return Wan5KitchenRuntime(request.request, device=request.device)

    def run(
        self, runtime: Any, request: _BoundRequest, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        from .wan5_kitchen import Wan5KitchenGeneration

        context.stage = "generate"
        result = runtime.generate(
            Wan5KitchenGeneration(**request.generation),
            progress=context.publish_progress,
            check_cancelled=lambda: None,
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "ok": True,
            "request_binding": request.binding,
            "output_path": str(result.output_path),
            "output_size_bytes": result.output_path.stat().st_size,
            "metadata": result.metadata,
            "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        }

    def unload(
        self,
        runtime: Any,
        request: _BoundRequest,
        context: DisposableChildContext,
    ) -> None:
        # Process exit is the accepted Wan 5 disposal and memory-release boundary.
        pass

    def failure_result(
        self, exc: BaseException, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "ok": False,
            "request_binding": context.binding,
            "error_type": type(exc).__name__,
            **_failure_diagnostic(exc, context),
        }

    def stage_for_progress(self, message: str | None) -> str:
        return _progress_stage(message)

    def protocol_error(self, reason: str) -> BaseException:
        errors: dict[str, BaseException] = {
            "start_gate_timeout": TimeoutError("Wan 5B worker start gate was not opened"),
            "request_bound": ValueError(
                "Wan 5B worker request is missing or exceeds its bound"
            ),
            "invalid_progress": ValueError("Wan 5B worker progress is invalid"),
            "progress_bound": ValueError("Wan 5B worker progress exceeds its bound"),
        }
        try:
            return errors[reason]
        except KeyError as exc:
            raise ValueError("unknown disposable worker protocol error") from exc


def main(argv: list[str] | None = None) -> int:
    paths = parse_disposable_child_paths(
        argv, description="LatentSlate disposable Wan 5B Kitchen worker"
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return run_disposable_child(
        paths,
        _Wan5KitchenHandler(),
        maximum_json_bytes=_MAX_JSON_BYTES,
        maximum_progress_bytes=_MAX_PROGRESS_BYTES,
    )


def _failure_diagnostic(
    exc: BaseException, failure: DisposableChildContext
) -> dict[str, Any]:
    """Return safe error provenance without persisting exception text."""

    location = "worker"
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        name = Path(frame.filename).stem
        candidate = f"{name}.{frame.name}"
        if name == "child":
            if frame.name == "_wait_start_gate":
                candidate = "wan5_kitchen_worker._wait_gate"
            elif frame.name == "run_disposable_child" and failure.stage == "read_request":
                candidate = "wan5_kitchen_worker._read_json"
            elif frame.name == "publish_progress":
                candidate = "wan5_kitchen_worker._append_progress"
        if candidate.replace("_", "").replace(".", "").isalnum() and len(candidate) <= 160:
            location = candidate
            break
    digest = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()
    diagnostic: dict[str, Any] = {
        "failure_stage": failure.stage,
        "error_fingerprint": digest,
        "failure_location": location,
    }
    boundary = getattr(exc, "numerical_boundary", None)
    step = getattr(exc, "denoise_step", None)
    transformer_pass = getattr(exc, "transformer_pass", None)
    if (
        boundary in {
            "transformer_noise_prediction",
            "guided_noise_prediction",
            "scheduler_output_latents",
            "denoise_latents",
        }
        and isinstance(step, int)
        and not isinstance(step, bool)
        and 1 <= step <= 30
        and transformer_pass in {None, "conditional", "unconditional"}
    ):
        diagnostic.update(
            {
                "numerical_boundary": boundary,
                "denoise_step": step,
                "transformer_pass": transformer_pass,
            }
        )
    return diagnostic


def _progress_stage(message: str | None) -> str:
    """Map fixed Engine progress messages to safe, stable failure stages."""

    if message is None:
        return "generate"
    prefixes = (
        ("Materializing Wan 2.2 prompt encoder", "materialize_text_encoder"),
        ("Materializing Wan 2.2 transformer and VAE", "materialize_model_components"),
        ("Validating Wan 2.2 text conditioning", "validate_text_conditioning"),
        ("Preprocessing Wan 2.2 guide image", "guide_preprocess"),
        ("Encoding Wan 2.2 guide image", "guide_vae_encode"),
        ("Prepared Wan 2.2 guide latent", "guide_latent_ready"),
        ("Generating Wan 2.2 video", "generate"),
        ("Validating Wan 2.2 decoded frames", "validate_decoded_frames"),
        ("Encoding Wan 2.2 MP4", "encode_mp4"),
        ("Complete", "verify_output"),
    )
    return next((stage for prefix, stage in prefixes if message.startswith(prefix)), "generate")


def _validate_binding(payload: Any) -> str:
    expected = {"schema_version", "request", "generation", "device", "request_binding"}
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("device") != "cuda"
        or not isinstance(payload.get("request"), dict)
        or not isinstance(payload.get("generation"), dict)
        or not isinstance(payload.get("request_binding"), str)
    ):
        raise ValueError("Wan 5B worker payload is invalid")
    unsigned = {key: payload[key] for key in expected - {"request_binding"}}
    binding = _fingerprint(unsigned)
    if payload["request_binding"] != binding:
        raise ValueError("Wan 5B worker request binding differs")
    return binding


def _validate_generation(value: Mapping[str, object], operation: str) -> dict[str, Any]:
    expected = {
        "operation",
        "prompt",
        "width",
        "height",
        "num_frames",
        "seed",
        "output_path",
        "staging_output_path",
        "start_image_path",
        "start_image_identity",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("operation") != operation
    ):
        raise ValueError("Wan 5B worker generation is invalid")
    output = value.get("output_path")
    staging = value.get("staging_output_path")
    if (
        not isinstance(output, str)
        or Path(output).exists()
        or Path(output).suffix.lower() != ".mp4"
        or not isinstance(staging, str)
        or Path(staging).exists()
        or Path(staging).suffix.lower() != ".mp4"
        or Path(staging).resolve(strict=False).parent != Path(output).resolve(strict=False).parent
        or Path(staging).resolve(strict=False) == Path(output).resolve(strict=False)
    ):
        raise ValueError("Wan 5B worker output or staging path is invalid")
    start = value.get("start_image_path")
    identity = value.get("start_image_identity")
    if operation == "wan5_t2v":
        if start is not None or identity is not None:
            raise ValueError("Wan 5B T2V worker cannot receive an image")
    elif (
        not isinstance(start, str)
        or not isinstance(identity, Mapping)
        or _endpoint_identity(Path(start)) != dict(identity)
    ):
        raise ValueError("Wan 5B I2V endpoint identity changed")
    return {
        "operation": operation,
        "prompt": value["prompt"],
        "width": value["width"],
        "height": value["height"],
        "num_frames": value["num_frames"],
        "seed": value["seed"],
        "output_path": Path(output),
        "staging_output_path": Path(staging),
        "start_image_path": None if start is None else Path(start),
        "start_image_identity": None if identity is None else dict(identity),
    }


def _endpoint_identity(path: Path) -> dict[str, int | str]:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("Wan 5B worker endpoint is not a file")
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Wan 5B worker endpoint changed during validation")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_fingerprint(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
