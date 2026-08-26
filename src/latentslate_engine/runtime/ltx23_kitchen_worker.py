"""One-shot Engine-native LTX 2.3 Kitchen generation worker."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework.worker import (
    PersistentChildContext,
    PersistentChildTerminalExit,
    canonical_json,
    hmac_sha256,
    parse_persistent_child_paths,
    result_hmac_sha256,
    run_persistent_child,
)

_SCHEMA_VERSION = 2
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
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


@dataclass(slots=True)
class _FailureContext:
    """Bounded diagnostic state that never contains a prompt, asset, or path."""

    stage: str = "worker_startup"
    binding: str | None = None
    cleanup_stage: str | None = None
    aimdo_counters: Mapping[str, object] | None = None
    terminal_poison_reason: str | None = None
    terminal_poison_origin: str | None = None


@dataclass(frozen=True, slots=True)
class _BoundCommand:
    kind: str
    request: Any | None
    generation: Mapping[str, Any] | None
    device: str | None
    cache_policy: str
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
        if command.kind != "generate" or command.request is None or command.device is None:
            raise ValueError("LTX 2.3 Kitchen initial worker command must generate")
        return _LoadedSession(
            command.request,
            LTX23KitchenRuntime(
                command.request,
                device=command.device,
                cache_policy=command.cache_policy,
            ),
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
        if command.cache_policy != session.runtime.cache_policy:
            raise ValueError("LTX 2.3 Kitchen worker cache policy does not match its session")
        if command.kind == "clear_cache":
            if command.request != session.request.fingerprint:
                raise ValueError(
                    "LTX 2.3 Kitchen cache-clear command recipe does not match its session"
                )
            return command
        if command.request is None or command.request.fingerprint != session.request.fingerprint:
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
        self.failure.cleanup_stage = None
        self.failure.aimdo_counters = None
        self.failure.terminal_poison_reason = None
        self.failure.terminal_poison_origin = None
        if command.kind == "clear_cache":
            self.failure.stage = "clear_cache"
            session.runtime.clear_cache()
            return _signed_result(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "ok": True,
                    "request_binding": command.binding,
                    "command": "clear_cache",
                    "cache": session.runtime.cache_status(),
                },
                self.secret,
            )
        self.failure.stage = "build_generation"
        context.publish_progress(0.005, "Preparing LTX generation")
        value = command.generation
        if value is None:
            raise ValueError("LTX 2.3 Kitchen generate command has no generation")
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
            next_stage = _progress_stage(message)
            if next_stage == "offload_text":
                self.failure.cleanup_stage = next_stage
            else:
                self.failure.stage = next_stage
                self.failure.cleanup_stage = None
            context.publish_progress(value, message)

        self.failure.stage = "generate"
        try:
            generated = session.runtime.generate(
                built, progress=report, check_cancelled=lambda: None
            )
        except BaseException:
            self.failure.aimdo_counters = session.runtime.failure_aimdo_counters()
            raise
        try:
            metadata = dict(generated.metadata)
            metadata["requested_num_frames"] = value["requested_num_frames"]
            metadata["requested_duration_seconds"] = value["duration_seconds"]
            metadata["effective_duration_seconds"] = value["num_frames"] / 25
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
                    "metadata": metadata,
                    "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                },
                self.secret,
            )
        except BaseException as primary:
            try:
                output.unlink(missing_ok=True)
            except BaseException as cleanup_error:  # noqa: BLE001 - retain primary error
                primary.add_note(f"LTX worker output cleanup also failed: {cleanup_error}")
            raise

    def unload(self, session: _LoadedSession, context: PersistentChildContext) -> None:
        # Persistent-child cleanup runs after the primary exception has already
        # established its stage. Keep that diagnostic stable and report unload
        # only through the separate cleanup field.
        self.failure.cleanup_stage = "unload_runtime"
        session.runtime.unload()

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]:
        poison_reason = self.failure.terminal_poison_reason or _poison_reason(exc)
        return _signed_result(
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": False,
                "request_binding": self.failure.binding or context.binding or None,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
                **_failure_diagnostic(exc, self.failure),
                "cleanup_stage": self.failure.cleanup_stage,
                "aimdo_counters": self.failure.aimdo_counters,
                **(
                    {
                        "terminal_exit_code": _AIMDO_POISON_EXIT_CODE,
                        "poison_reason": poison_reason,
                        "poison_origin": self.failure.terminal_poison_origin
                        or "primary",
                    }
                    if poison_reason is not None
                    else {}
                ),
            },
            self.secret,
        )

    def terminal_exit_status(
        self,
        exc: BaseException,
        _session: _LoadedSession,
        _context: PersistentChildContext,
    ) -> PersistentChildTerminalExit | None:
        # Runtime residency diagnostics intentionally retain descriptive text,
        # but the terminal protocol accepts only bounded canonical tokens.  A
        # descriptive runtime reason must never become an exit reason; recover
        # the backend's canonical token from the exception chain instead.
        reason = _poison_reason(exc)
        self.failure.terminal_poison_reason = reason
        self.failure.terminal_poison_origin = (
            None
            if reason is None
            else "cleanup"
            if self.failure.cleanup_stage == "unload_runtime"
            else "primary"
        )
        return (
            None
            if reason is None
            else PersistentChildTerminalExit(_AIMDO_POISON_EXIT_CODE, reason)
        )

    def protocol_error(self, reason: str) -> BaseException:
        return ValueError(f"LTX 2.3 Kitchen worker protocol violation: {reason}")

    def _bind(
        self, payload: Any, context: PersistentChildContext
    ) -> _BoundCommand:
        self.failure.stage = "validate_bound_request"
        context.publish_progress(0.002, "Validating LTX worker request")
        bound = _validate_bound_payload(payload, self.secret)
        if bound[0] == "clear_cache":
            kind, fingerprint, cache_policy, binding = bound
            if not hasattr(context, "binding"):
                raise ValueError("LTX 2.3 Kitchen worker context is invalid")
            self.failure.binding = binding
            context.binding = binding
            return _BoundCommand(kind, fingerprint, None, None, cache_policy, binding)
        kind, request_data, generation, device, cache_policy, binding = bound
        self.failure.binding = binding
        context.binding = binding
        self.failure.stage = "rehydrate_recipe"
        context.publish_progress(0.003, "Rehydrating LTX recipe")
        from ..ltx23_kitchen_recipe import rehydrate_ltx23_kitchen_runtime_request

        request = rehydrate_ltx23_kitchen_runtime_request(request_data)
        return _BoundCommand(kind, request, generation, device, cache_policy, binding)


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
        ("Materialized LTX components", "materialization_complete"),
        ("Reusing warmed LTX components", "materialization_reused"),
        ("Preparing streamed LTX text encoder", "prepare_text_streaming"),
        ("Prepared streamed LTX text encoder", "prepare_text_streaming_complete"),
        ("Enhancing prompt", "enhance_prompt"),
        ("Encoding positive prompt", "encode_positive_prompt"),
        ("Encoding negative prompt", "encode_negative_prompt"),
        ("Offloaded base Gemma", "offload_text"),
        ("Upscaling LTX video latents", "upscale_latents"),
        ("Decoding LTX video and audio", "decode_media"),
        ("Muxing 25 fps", "mux_output"),
        ("Completed LTX downstream phases", "downstream_complete"),
        ("Published LTX prompt cache", "prompt_cache_publish"),
        ("Skipped LTX prompt cache publication", "prompt_cache_reused"),
        ("LTX denoise step", "denoise"),
        ("LTX 2.3 output ready", "verify_output"),
    )
    return next((stage for prefix, stage in prefixes if message.startswith(prefix)), "generate")


def _failure_diagnostic(exc: BaseException, failure: _FailureContext) -> dict[str, str]:
    """Return useful cross-process provenance without returning sensitive exception text."""

    digest = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()
    location = "worker"
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        name = _failure_token(Path(frame.filename).stem, fallback="worker")
        if name.startswith("ltx23_"):
            function = _failure_token(frame.name, fallback="module")
            location = f"{name}.{function}"
            break
    return {
        "failure_stage": failure.stage,
        "error_fingerprint": digest,
        "failure_location": location,
    }


def _failure_token(value: str, *, fallback: str) -> str:
    """Normalize traceback labels to the authenticated diagnostic alphabet."""

    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return (token or fallback)[:72]


def _validate_bound_payload(
    payload: Mapping[str, Any],
    secret: bytes,
) -> tuple[Any, ...]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("LTX 2.3 Kitchen worker request is not canonical")
    if payload.get("command") == "clear_cache":
        expected = {
            "schema_version",
            "command",
            "request_fingerprint",
            "cache_policy",
            "request_binding",
        }
        if set(payload) != expected:
            raise ValueError("LTX 2.3 Kitchen worker cache-clear request is not canonical")
        fingerprint, cache_policy, binding = (
            payload["request_fingerprint"],
            payload["cache_policy"],
            payload["request_binding"],
        )
        unsigned = {key: payload[key] for key in expected - {"request_binding"}}
        if (
            not isinstance(fingerprint, str)
            or cache_policy not in {"none", "prompt"}
            or not isinstance(binding, str)
            or not hmac.compare_digest(binding, _binding(unsigned, secret))
        ):
            raise ValueError("LTX 2.3 Kitchen worker cache-clear binding is invalid")
        return "clear_cache", fingerprint, cache_policy, binding
    expected = {
        "schema_version",
        "command",
        "request",
        "generation",
        "device",
        "cache_policy",
        "request_binding",
    }
    if (
        set(payload) != expected
        or payload.get("command") != "generate"
    ):
        raise ValueError("LTX 2.3 Kitchen worker request is not canonical")
    request, generation, device, cache_policy, binding = (
        payload["request"],
        payload["generation"],
        payload["device"],
        payload["cache_policy"],
        payload["request_binding"],
    )
    if (
        not isinstance(request, Mapping)
        or not isinstance(generation, Mapping)
        or device != "cuda"
        or cache_policy not in {"none", "prompt"}
        or not isinstance(binding, str)
    ):
        raise ValueError("LTX 2.3 Kitchen worker request fields are invalid")
    unsigned = {
        "schema_version": payload["schema_version"],
        "command": payload["command"],
        "request": request,
        "generation": generation,
        "device": device,
        "cache_policy": cache_policy,
    }
    if not hmac.compare_digest(binding, _binding(unsigned, secret)):
        raise ValueError(
            "LTX 2.3 Kitchen worker request binding does not match its canonical payload"
        )
    _validate_generation_json(generation)
    return "generate", request, generation, device, cache_policy, binding


def _poison_reason(exc: BaseException) -> str | None:
    """Find the bounded child-terminal reason without importing AIMDO."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reason = getattr(current, "reason", None)
        if (
            type(current).__name__
            in {"LTX23KitchenWorkerPoisoned", "DynamicResidencyPoisoned"}
            and reason in _AIMDO_POISON_REASONS
        ):
            return reason
        current = current.__cause__ or current.__context__
    return None


def _validate_generation_json(value: Mapping[str, Any]) -> None:
    expected = {
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
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not isinstance(value.get("prompt"), str)
        or not value["prompt"].strip()
        or not isinstance(value.get("output_path"), str)
    ):
        raise ValueError("LTX 2.3 Kitchen worker generation is invalid")
    for key in ("width", "height", "requested_num_frames", "num_frames", "seed"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise TypeError("LTX 2.3 Kitchen worker generation integer fields are invalid")
    if isinstance(value["duration_seconds"], bool) or not isinstance(
        value["duration_seconds"], (int, float)
    ):
        raise TypeError("LTX 2.3 Kitchen worker duration is invalid")
    duration = float(value["duration_seconds"])
    if not math.isfinite(duration) or not 1.0 <= duration <= 10.0:
        raise ValueError("LTX 2.3 Kitchen worker duration is outside 1..10 seconds")
    requested = math.floor(duration * 25) + 1
    effective = ((requested - 1) // 8) * 8 + 1
    if (
        value["requested_num_frames"] != requested
        or value["num_frames"] != effective
    ):
        raise ValueError("LTX 2.3 Kitchen worker temporal request is invalid")
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
