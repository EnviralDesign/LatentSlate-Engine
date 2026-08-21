"""One-shot Engine-native LTX 2.3 Kitchen generation worker."""

from __future__ import annotations

import hashlib
import hmac
import os
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework.worker import (
    PersistentChildContext,
    canonical_json,
    hmac_sha256,
    parse_persistent_child_paths,
    result_hmac_sha256,
    run_persistent_child,
)

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024


@dataclass(slots=True)
class _FailureContext:
    """Bounded diagnostic state that never contains a prompt, asset, or path."""

    stage: str = "worker_startup"
    binding: str | None = None


@dataclass(frozen=True, slots=True)
class _BoundCommand:
    request: Any
    generation: Mapping[str, Any]
    device: str
    binding: str


@dataclass(frozen=True, slots=True)
class _LoadedSession:
    request: Any
    runtime: Any
    generation_type: Any
    validate_generation: Any


class _LTX23KitchenHandler:
    def __init__(self, secret: bytes) -> None:
        self.secret = secret
        self.failure = _FailureContext()

    def bind_initial(
        self, payload: Any, context: PersistentChildContext
    ) -> _BoundCommand:
        context.publish_progress(0.001, "LTX worker started")
        return self._bind(payload, context)

    def load(
        self, command: _BoundCommand, context: PersistentChildContext
    ) -> _LoadedSession:
        self.failure.stage = "import_runtime"
        context.publish_progress(0.004, "Importing LTX runtime")
        from .ltx23_kitchen import (
            LTX23KitchenGeneration,
            LTX23KitchenRuntime,
            validate_ltx23_kitchen_generation,
        )

        self.failure.stage = "initialize_runtime"
        return _LoadedSession(
            command.request,
            LTX23KitchenRuntime(command.request, device=command.device),
            LTX23KitchenGeneration,
            validate_ltx23_kitchen_generation,
        )

    def bind_command(
        self,
        payload: Any,
        session: _LoadedSession,
        context: PersistentChildContext,
    ) -> _BoundCommand:
        command = self._bind(payload, context)
        if command.request.fingerprint != session.request.fingerprint:
            raise ValueError(
                "LTX 2.3 Kitchen worker command recipe does not match its session"
            )
        return command

    def execute(
        self,
        session: _LoadedSession,
        command: _BoundCommand,
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> Mapping[str, Any]:
        self.failure.binding = command.binding
        self.failure.stage = "build_generation"
        context.publish_progress(0.005, "Preparing LTX generation")
        value = command.generation
        output = Path(value["output_path"]).resolve(strict=False)
        built = session.generation_type(
            value["prompt"],
            output,
            value["width"],
            value["height"],
            value["num_frames"],
            value["seed"],
            None
            if value["start_image_path"] is None
            else Path(value["start_image_path"]),
            None
            if value["end_image_path"] is None
            else Path(value["end_image_path"]),
            value["start_image_identity"],
            value["end_image_identity"],
        )
        self.failure.stage = "validate_generation"
        session.validate_generation(session.request.operation, built)

        def report(value: float, message: str | None) -> None:
            self.failure.stage = _progress_stage(message)
            context.publish_progress(value, message)

        self.failure.stage = "generate"
        generated = session.runtime.generate(
            built, progress=report, check_cancelled=lambda: None
        )
        self.failure.stage = "verify_output"
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("LTX 2.3 Kitchen worker did not publish an MP4")
        return _signed_result(
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": True,
                "request_binding": command.binding,
                "output_path": str(output),
                "output_size_bytes": output.stat().st_size,
                "metadata": dict(generated.metadata),
                "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            },
            self.secret,
        )

    def unload(self, session: _LoadedSession, context: PersistentChildContext) -> None:
        self.failure.stage = "unload_runtime"
        session.runtime.unload()

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]:
        return _signed_result(
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": False,
                "request_binding": self.failure.binding or context.binding or None,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
                **_failure_diagnostic(exc, self.failure),
            },
            self.secret,
        )

    def protocol_error(self, reason: str) -> BaseException:
        return ValueError(f"LTX 2.3 Kitchen worker protocol violation: {reason}")

    def _bind(
        self, payload: Any, context: PersistentChildContext
    ) -> _BoundCommand:
        self.failure.stage = "validate_bound_request"
        context.publish_progress(0.002, "Validating LTX worker request")
        request_data, generation, device, binding = _validate_bound_payload(
            payload, self.secret
        )
        self.failure.binding = binding
        context.binding = binding
        self.failure.stage = "rehydrate_recipe"
        context.publish_progress(0.003, "Rehydrating LTX recipe")
        from ..ltx23_kitchen_recipe import rehydrate_ltx23_kitchen_runtime_request

        request = rehydrate_ltx23_kitchen_runtime_request(request_data)
        return _BoundCommand(request, generation, device, binding)


def main(argv: list[str] | None = None) -> int:
    paths = parse_persistent_child_paths(
        argv, description="LatentSlate persistent Engine-native LTX 2.3 Kitchen worker"
    )
    secret_text = os.environ.pop("LATENTSLATE_LTX23_IPC_SECRET", "")
    secret = _secret(secret_text)
    # This precedes every torch, diffusers, and Comfy Kitchen import.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return run_persistent_child(
        paths, _LTX23KitchenHandler(secret), maximum_bytes=_MAX_PROGRESS_BYTES
    )


def _progress_stage(message: str | None) -> str:
    """Map Engine-owned progress text to a safe, stable error phase."""

    if message is None:
        return "generate"
    prefixes = (
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
        ("Preparing streamed LTX text encoder", "prepare_text_streaming"),
        ("Enhancing prompt", "enhance_prompt"),
        ("Encoding prompt", "encode_prompt"),
        ("Upscaling LTX video latents", "upscale_latents"),
        ("Decoding LTX video and audio", "decode_media"),
        ("Muxing 24 fps", "mux_output"),
        ("LTX denoise step", "denoise"),
        ("LTX 2.3 output ready", "verify_output"),
    )
    return next((stage for prefix, stage in prefixes if message.startswith(prefix)), "generate")


def _failure_diagnostic(exc: BaseException, failure: _FailureContext) -> dict[str, str]:
    """Return useful cross-process provenance without returning sensitive exception text."""

    digest = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()
    location = "worker"
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        name = Path(frame.filename).stem
        if name.startswith("ltx23_"):
            location = f"{name}.{frame.name}"
            break
    return {
        "failure_stage": failure.stage,
        "error_fingerprint": digest,
        "failure_location": location,
    }


def _validate_bound_payload(
    payload: Mapping[str, Any],
    secret: bytes,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str]:
    expected = {"schema_version", "request", "generation", "device", "request_binding"}
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("LTX 2.3 Kitchen worker request is not canonical")
    request, generation, device, binding = (
        payload["request"],
        payload["generation"],
        payload["device"],
        payload["request_binding"],
    )
    if (
        not isinstance(request, Mapping)
        or not isinstance(generation, Mapping)
        or device != "cuda"
        or not isinstance(binding, str)
    ):
        raise ValueError("LTX 2.3 Kitchen worker request fields are invalid")
    unsigned = {
        "schema_version": payload["schema_version"],
        "request": request,
        "generation": generation,
        "device": device,
    }
    if not hmac.compare_digest(binding, _binding(unsigned, secret)):
        raise ValueError(
            "LTX 2.3 Kitchen worker request binding does not match its canonical payload"
        )
    _validate_generation_json(generation)
    return request, generation, device, binding


def _validate_generation_json(value: Mapping[str, Any]) -> None:
    expected = {
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
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not isinstance(value.get("prompt"), str)
        or not value["prompt"].strip()
        or not isinstance(value.get("output_path"), str)
    ):
        raise ValueError("LTX 2.3 Kitchen worker generation is invalid")
    for key in ("width", "height", "num_frames", "seed"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise TypeError("LTX 2.3 Kitchen worker generation integer fields are invalid")
    if isinstance(value["duration_seconds"], bool) or not isinstance(
        value["duration_seconds"], (int, float)
    ):
        raise TypeError("LTX 2.3 Kitchen worker duration is invalid")
    if any(
        item is not None and not isinstance(item, str)
        for item in (value["start_image_path"], value["end_image_path"])
    ):
        raise TypeError("LTX 2.3 Kitchen worker endpoint fields are invalid")
    for path, identity in (
        (value["start_image_path"], value["start_image_identity"]),
        (value["end_image_path"], value["end_image_identity"]),
    ):
        if path is None:
            if identity is not None:
                raise ValueError("LTX 2.3 Kitchen worker endpoint identity lacks a path")
            continue
        if not isinstance(identity, Mapping) or _endpoint_identity(Path(path)) != dict(identity):
            raise ValueError("LTX 2.3 Kitchen worker endpoint changed after request binding")


def _endpoint_identity(path: Path) -> dict[str, int | str]:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("LTX 2.3 Kitchen worker endpoint is not a file")
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("LTX 2.3 Kitchen worker endpoint changed during validation")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _binding(value: Mapping[str, Any], secret: bytes) -> str:
    return hmac_sha256(value, secret)


def _result_binding(value: Mapping[str, Any], secret: bytes) -> str:
    return result_hmac_sha256(value, secret)


def _signed_result(value: dict[str, Any], secret: bytes) -> dict[str, Any]:
    return {**value, "result_binding": _result_binding(value, secret)}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value)


def _secret(value: str) -> bytes:
    if len(value) != 64:
        raise ValueError("LTX 2.3 Kitchen worker capability secret is invalid")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("LTX 2.3 Kitchen worker capability secret is invalid") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
