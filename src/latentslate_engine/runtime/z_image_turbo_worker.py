"""Authenticated persistent child for the exact Z-Image Turbo core.

This module intentionally imports no Z model materializer until after the
private HMAC capability, canonical request, and on-disk identities are proven.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import z_image_cuda_health as _cuda_health
from .framework.worker import (
    PersistentChildContext,
    hmac_sha256,
    parse_persistent_child_paths,
    result_hmac_sha256,
    run_persistent_child,
)

_SCHEMA = 1
_MAX_BYTES = 1024 * 1024
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
        "lora_install",
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
_SAFE_EXCEPTION_CLASSES: dict[type[BaseException], str] = {
    AssertionError: "AssertionError",
    AttributeError: "AttributeError",
    BaseException: "BaseException",
    EOFError: "EOFError",
    Exception: "Exception",
    FileNotFoundError: "FileNotFoundError",
    ImportError: "ImportError",
    IsADirectoryError: "IsADirectoryError",
    json.JSONDecodeError: "JSONDecodeError",
    KeyError: "KeyError",
    MemoryError: "MemoryError",
    ModuleNotFoundError: "ModuleNotFoundError",
    NotADirectoryError: "NotADirectoryError",
    OSError: "OSError",
    OverflowError: "OverflowError",
    PermissionError: "PermissionError",
    RuntimeError: "RuntimeError",
    SystemExit: "SystemExit",
    TypeError: "TypeError",
    UnicodeDecodeError: "UnicodeDecodeError",
    ValueError: "ValueError",
}


@dataclass(slots=True)
class _FailureContext:
    """Only closed diagnostic labels; never request content or local paths."""

    stage: str = "canonical_validation"
    location: str = "z_image_turbo_worker._read_json"
    binding: str = ""


class _BoundFailure(RuntimeError):
    def __init__(self, binding: str, cause: BaseException, failure: _FailureContext) -> None:
        # Do not make ``cause`` part of this exception's text: parent process
        # logs and provider errors must never recover a prompt or filesystem path.
        super().__init__("bound worker failure")
        self.request_binding = binding
        self.cause = cause
        self.failure = failure


class _CudaHealthFailure(RuntimeError):
    """Closed synthetic-only CUDA failure; never retains exception text."""

    def __init__(
        self,
        phase: str,
        substage: str,
        code: str,
        error_type: str,
        completed: tuple[str, ...] = (),
    ) -> None:
        safe_types = frozenset({*_SAFE_EXCEPTION_CLASSES.values(), "OutOfMemoryError"})
        if (
            phase not in _CUDA_HEALTH_PHASES
            or substage not in _cuda_health._HEALTH_SUBSTAGES
            or code not in _CUDA_ERROR_CODES
        ):
            raise ValueError("invalid synthetic CUDA failure classification")
        if error_type not in safe_types:
            error_type = "Exception"
        super().__init__("synthetic CUDA health checkpoint failed")
        self.phase = phase
        self.substage = substage
        self.code = code
        self.error_type = error_type
        self.completed = completed


def _classify_synthetic_cuda_error(exc: BaseException) -> str:
    """Classify a synthetic CUDA exception without returning hostile content."""

    type_module = type(exc).__module__
    type_name = type(exc).__name__
    if type_module.startswith("torch") and type_name == "OutOfMemoryError":
        return "cuda_oom"
    if type(exc) is not RuntimeError:
        return "unknown_runtime"
    try:
        message = str(exc).lower()
    except BaseException:  # noqa: BLE001 - hostile synthetic exceptions collapse closed.
        return "unknown_runtime"
    if "out of memory" in message:
        return "cuda_oom"
    if "illegal memory access" in message:
        return "illegal_memory_access"
    if "invalid argument" in message:
        return "invalid_argument"
    if "operation not supported" in message or "not supported" in message:
        return "operation_not_supported"
    if any(
        marker in message
        for marker in (
            "cuda driver",
            "driver version",
            "driver initialization",
            "system driver mismatch",
        )
    ):
        return "driver_error"
    return "unknown_runtime"


def _resolve_worker_cuda_device(torch_module: Any, requested: str) -> Any:
    """Resolve one authenticated request to the worker's concrete current CUDA device."""

    try:
        parsed = torch_module.device(requested)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Z-Image worker execution device is invalid") from exc
    if parsed.type != "cuda":
        raise ValueError("Z-Image worker execution device must be CUDA")
    if not torch_module.cuda.is_available():
        raise ValueError("Z-Image worker CUDA execution device is unavailable")
    try:
        device_count = int(torch_module.cuda.device_count())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Z-Image worker CUDA execution device is unavailable") from exc
    if parsed.index is not None:
        if parsed.index < 0 or parsed.index >= device_count:
            raise ValueError("Z-Image worker execution device index is invalid")
        return parsed
    try:
        current_index = int(torch_module.cuda.current_device())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Z-Image worker current CUDA device is unavailable") from exc
    if current_index < 0 or current_index >= device_count:
        raise ValueError("Z-Image worker current CUDA device is invalid")
    return torch_module.device("cuda", current_index)


def _z_image_cuda_health_check(
    torch_module: Any,
    device: Any,
    phase: str,
) -> dict[str, str | int | bool]:
    """Run one bounded model-free synchronous CPU-uint8/CUDA-copy checkpoint."""

    if phase not in _CUDA_HEALTH_PHASES:
        raise ValueError("invalid CUDA health phase")
    active_substage = "sync_before"

    def checkpoint(substage: str) -> None:
        nonlocal active_substage
        active_substage = substage

    try:
        return _cuda_health.z_image_cuda_health_check(
            torch_module,
            device,
            checkpoint=checkpoint,
        )
    except BaseException as exc:  # noqa: BLE001 - every synthetic CUDA failure is classified.
        error_type = (
            "OutOfMemoryError"
            if type(exc).__module__.startswith("torch")
            and type(exc).__name__ == "OutOfMemoryError"
            else _SAFE_EXCEPTION_CLASSES.get(type(exc), "Exception")
        )
        raise _CudaHealthFailure(
            phase,
            active_substage,
            _classify_synthetic_cuda_error(exc),
            error_type,
        ) from None


@dataclass(frozen=True, slots=True)
class _BoundCommand:
    recipe_json: dict[str, Any]
    device: str
    generation: dict[str, Any]
    output: Path
    binding: str


@dataclass(slots=True)
class _LoadedSession:
    recipe_json: dict[str, Any]
    recipe: Any
    core: Any
    identity: tuple[object, object]


class _ZImageHandler:
    def __init__(self, secret_text: str) -> None:
        self.secret_text = secret_text
        self.failure = _FailureContext()
        try:
            self.result_secret = _secret(secret_text)
        except ValueError:
            self.result_secret = b""
        self.secret: bytes | None = None

    def bind_initial(
        self, payload: Any, context: PersistentChildContext
    ) -> _BoundCommand:
        self.failure.stage = "auth"
        self.failure.location = "z_image_turbo_worker._secret"
        self.secret = _secret(self.secret_text)
        command = self._bind(payload)
        context.binding = command.binding
        return command

    def load(
        self, command: _BoundCommand, context: PersistentChildContext
    ) -> _LoadedSession:
        context.publish_progress(0.01, "Validating Z-Image request")
        self.failure.stage = "rehydrate"
        self.failure.location = "z_image_turbo_recipe.rehydrate_z_image_turbo_runtime_request"
        from ..z_image_turbo_recipe import rehydrate_z_image_turbo_runtime_request

        recipe = rehydrate_z_image_turbo_runtime_request(command.recipe_json)
        context.publish_progress(0.02, "Materializing exact Z-Image components")
        core = _load_core(recipe, command.device, context, self.failure)
        assert self.secret is not None
        canonical_device = str(core.execution_device)
        identity = (
            _execution_session_binding(command.recipe_json, canonical_device, self.secret),
            _execution_runtime_key(command.recipe_json, canonical_device),
        )
        return _LoadedSession(command.recipe_json, recipe, core, identity)

    def bind_command(
        self,
        payload: Any,
        session: _LoadedSession,
        context: PersistentChildContext,
    ) -> _BoundCommand:
        next_binding = _untrusted_binding(payload)
        try:
            command = self._bind(payload)
            assert self.secret is not None
            canonical_device = str(session.core.execution_device)
            next_identity = (
                _execution_session_binding(
                    command.recipe_json, canonical_device, self.secret
                ),
                _execution_runtime_key(command.recipe_json, canonical_device),
            )
            if next_identity != session.identity:
                raise ValueError("Z-Image command does not match its loaded session")
            context.binding = command.binding
            return command
        except BaseException as exc:
            raise _BoundFailure(next_binding, exc, self.failure) from exc

    def execute(
        self,
        session: _LoadedSession,
        command: _BoundCommand,
        context: PersistentChildContext,
        *,
        cold: bool,
    ) -> Mapping[str, Any]:
        assert self.secret is not None
        return _execute(
            session.core,
            session.recipe,
            command.generation,
            command.output,
            command.binding,
            context,
            cold=cold,
            failure=self.failure,
            secret=self.secret,
        )

    def unload(self, session: _LoadedSession, context: PersistentChildContext) -> None:
        session.core = None

    def failure_result(
        self, exc: BaseException, context: PersistentChildContext
    ) -> Mapping[str, Any]:
        return _failure_result(
            exc,
            self.failure,
            context.binding,
            self.result_secret,
        )

    def protocol_error(self, reason: str) -> BaseException:
        if reason == "json_bound":
            return ValueError("Z-Image worker JSON is missing or exceeds its bound")
        if reason == "json_type":
            return TypeError("Z-Image worker JSON is invalid")
        if reason == "json_invalid":
            return ValueError("Z-Image worker JSON is invalid")
        if reason == "progress_bound":
            return RuntimeError("Z-Image worker progress exceeds its bound")
        if reason == "heartbeat_stop":
            return RuntimeError("Z-Image heartbeat did not stop before result publication")
        return RuntimeError("Z-Image worker protocol is invalid")

    def _bind(self, payload: Any) -> _BoundCommand:
        assert self.secret is not None
        self.failure.stage = "canonical_validation"
        self.failure.location = "z_image_turbo_worker._read_json"
        binding = _untrusted_binding(payload)
        self.failure.binding = binding
        recipe_json, device, generation, output, binding, _session_binding = _validate(
            payload, self.secret, self.failure
        )
        self.failure.binding = binding
        return _BoundCommand(recipe_json, device, generation, output, binding)


def main(argv: list[str] | None = None) -> int:
    paths = parse_persistent_child_paths(
        argv, description="LatentSlate Z-Image Turbo private worker"
    )
    handler = _ZImageHandler(os.environ.pop("LATENTSLATE_ZIMAGE_IPC_SECRET", ""))
    return run_persistent_child(paths, handler, maximum_bytes=_MAX_BYTES)


def _load_core(
    recipe: Any,
    device: str,
    context: PersistentChildContext,
    failure: _FailureContext,
):
    failure.stage = "runtime_import"
    failure.location = "z_image_turbo_worker._load_core"
    import torch

    failure.stage = "device_contract"
    failure.location = "z_image_turbo_worker._resolve_worker_cuda_device"
    requested_device = device
    device = _resolve_worker_cuda_device(torch, requested_device)
    failure.location = "z_image_turbo_worker._load_core"

    health_passes: list[str] = []

    def health(phase: str) -> None:
        failure.stage = f"cuda_health_{phase}"
        try:
            _z_image_cuda_health_check(torch, device, phase)
        except _CudaHealthFailure as exc:
            exc.completed = tuple(health_passes)
            raise
        health_passes.append(phase)

    health("pre_import")

    from .z_image_conditioning import build_z_image_qwen_tokenizer
    from .z_image_nextdit import materialize_z_image_nextdit
    from .z_image_qwen_runtime import (
        build_z_image_mixed_qwen_shell,
        materialize_z_image_mixed_qwen,
    )
    from .z_image_stored_lora import ZImageFixedLoraLifecycle
    from .z_image_turbo import ZImageTurboCore
    from .z_image_vae import materialize_z_image_flux_ae

    support = recipe.plans["pipeline_support"]
    failure.stage = "tokenizer"
    context.publish_progress(0.03, "Preparing tokenizer")
    tokenizer = build_z_image_qwen_tokenizer(support)
    health("post_tokenizer")
    context.publish_progress(0.04, "Tokenizer ready")
    report = lambda base, span: (
        lambda done, total: context.publish_progress(
            base + span * done / total, "Materializing component"
        )
    )
    failure.stage = "qwen_materialize"
    context.publish_progress(0.045, "Materializing Qwen")
    qwen = materialize_z_image_mixed_qwen(
        recipe.plans["text_encoder"],
        build_z_image_mixed_qwen_shell(support),
        progress=report(0.045, 0.025),
        cancelled=context.paths.cancel.is_file,
    )
    health("post_qwen")
    context.publish_progress(0.07, "Qwen ready")
    failure.stage = "nextdit_materialize"
    context.publish_progress(0.075, "Materializing NextDiT")
    transformer = materialize_z_image_nextdit(
        recipe.plans["transformer"],
        progress=report(0.075, 0.025),
        cancelled=context.paths.cancel.is_file,
    )
    health("post_nextdit")
    context.publish_progress(0.10, "NextDiT ready")
    fixed_lora = None
    if "style_lora" in recipe.plans:
        failure.stage = "lora_install"
        context.publish_progress(0.101, "Installing fixed Z-Image LoRA")
        fixed_lora = ZImageFixedLoraLifecycle()
        fixed_lora.install(
            transformer,
            recipe.plans["style_lora"],
            cancelled=context.paths.cancel.is_file,
        )
        context.publish_progress(0.102, "Fixed Z-Image LoRA ready")
    failure.stage = "vae_materialize"
    context.publish_progress(0.102, "Materializing Flux AE")
    vae = materialize_z_image_flux_ae(
        recipe.plans["vae"],
        progress=report(0.102, 0.012),
        cancelled=context.paths.cancel.is_file,
    )
    health("post_vae")
    context.publish_progress(0.114, "Flux AE ready")
    failure.stage = "core_ready"
    core = ZImageTurboCore(
        recipe,
        tokenizer=tokenizer,
        text_encoder=qwen,
        transformer=transformer,
        vae=vae,
        execution_device=device,
        fixed_lora=fixed_lora,
    )
    core._latentslate_requested_device = requested_device
    core._latentslate_execution_device = str(device)
    core._latentslate_runtime_key = (
        recipe.fingerprint,
        str(device),
        "bfloat16",
        "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    )
    health("post_core")
    core._latentslate_cuda_health_passes = tuple(health_passes)
    context.publish_progress(0.12, "Core ready")
    return core


def _execute(
    core: Any,
    recipe: Any,
    generation: Mapping[str, Any],
    output: Path,
    binding: str,
    context: PersistentChildContext,
    cold: bool,
    failure: _FailureContext | None = None,
    secret: bytes | None = None,
) -> Mapping[str, Any]:
    if failure is None:
        failure = _FailureContext("conditioning", "z_image_turbo.generate")
    prompt, seed = generation["prompt"], generation["seed"]
    import torch

    failure.stage = "cuda_health_pre_qwen_preflight"
    failure.location = "z_image_turbo_worker._execute"
    health_passes = tuple(getattr(core, "_latentslate_cuda_health_passes", ()))
    try:
        _z_image_cuda_health_check(
            torch, str(core.execution_device), "pre_qwen_preflight"
        )
    except _CudaHealthFailure as exc:
        exc.completed = health_passes
        raise
    failure.stage = "conditioning"
    failure.location = "z_image_turbo.generate"
    result = core.generate(
        prompt=prompt,
        seed=seed,
        output_path=output,
        cancelled=context.paths.cancel.is_file,
        progress=lambda value, message: _execution_progress(
            context, value, message, cold=cold
        ),
        failure_stage=lambda stage: _set_core_failure_stage(failure, stage),
    )
    artifact = result.artifact
    failure.stage = "publish"
    failure.location = "z_image_turbo_worker._validate_artifact"
    _validate_artifact(artifact, output)
    metadata = {
        "family": "zimage",
        "runtime": "engine-native/z-image-turbo",
        "request_fingerprint": recipe.fingerprint,
        "components": recipe.public_component_manifest(),
        "seed": seed,
        "schedule": dict(recipe.schedule),
        "qwen_dispatch": result.qwen_dispatch,
        "transformer_dispatch": result.transformer_dispatch,
        "lora_dispatch": getattr(result, "lora_dispatch", None),
        "cuda_health": {
            phase: "pass"
            for phase in (*getattr(core, "_latentslate_cuda_health_passes", ()), "pre_qwen_preflight")
        },
        "requested_device": str(getattr(core, "_latentslate_requested_device", "")),
        "execution_device": str(core.execution_device),
        "phases": list(result.phases),
        "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    }
    return _signed_result(
        {
            "schema_version": _SCHEMA,
            "ok": True,
            "request_binding": binding,
            "output_path": str(artifact.path.resolve(strict=True)),
            "output_size_bytes": artifact.size_bytes,
            "output_sha256": artifact.sha256,
            "metadata": metadata,
        },
        secret,
    )


def _execution_progress(
    context: PersistentChildContext, value: float, message: str, *, cold: bool
) -> None:
    """Translate core phases into the public cold/warm lifecycle scale.

    Cold loading owns 0-.12.  Both cold and warm inference reserve .12-.92
    for prompt/sampling and .92-1 for decode/publication.  A warm session
    therefore begins at its first meaningful inference stage, not at a fake
    materialization percentage.
    """

    del cold  # The scale intentionally stays identical once inference begins.
    raw = min(1.0, max(0.0, float(value)))
    if message == "Complete":
        # Core completion is validated again by _execute before public 1.0.
        mapped = 0.999
    elif message.startswith("Decoding Z-Image"):
        mapped = 0.92
    elif message.startswith(("Sampling Z-Image", "Z-Image step ")):
        # Core's eight sampler callbacks cover [.12, .82].
        mapped = 0.12 + 0.80 * min(1.0, max(0.0, (raw - 0.12) / 0.70))
    else:
        # Text encoding is the boundary between ready/loading and sampling.
        mapped = 0.12
    context.publish_progress(mapped, message)


def _validate_artifact(artifact: Any, output: Path) -> None:
    """Require the core's already-validated PNG identity before public completion."""

    try:
        actual = Path(artifact.path).resolve(strict=True)
        expected = output.resolve(strict=True)
        size = actual.stat().st_size
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    except (AttributeError, OSError) as exc:
        raise RuntimeError("Z-Image core did not publish a validated PNG") from exc
    if actual != expected or size != int(artifact.size_bytes) or digest != artifact.sha256:
        raise RuntimeError("Z-Image core PNG identity changed before worker publication")


def _validate(
    payload: object, secret: bytes, failure: _FailureContext | None = None
) -> tuple[dict[str, Any], str, dict[str, Any], Path, str, str]:
    if failure is not None:
        failure.stage = "canonical_validation"
        failure.location = "z_image_turbo_worker._validate"
    expected = {
        "schema_version",
        "recipe",
        "device",
        "dtype",
        "execution",
        "session_binding",
        "output_path",
        "generation",
        "request_binding",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version") != _SCHEMA
    ):
        raise ValueError("Z-Image worker request is not canonical")
    if (
        not isinstance(payload.get("recipe"), dict)
        or not isinstance(payload.get("device"), str)
        or payload.get("dtype") != "bfloat16"
        or payload.get("execution")
        != "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise"
    ):
        raise ValueError("Z-Image worker session is invalid")
    session = {
        "recipe": payload["recipe"],
        "device": payload["device"],
        "dtype": payload["dtype"],
        "execution": payload["execution"],
    }
    if failure is not None:
        failure.stage = "auth"
    if not isinstance(payload.get("session_binding"), str) or not hmac.compare_digest(
        payload["session_binding"], _binding(session, secret)
    ):
        raise ValueError("Z-Image worker session binding is invalid")
    if not isinstance(payload.get("request_binding"), str) or not hmac.compare_digest(
        payload["request_binding"],
        _binding(
            {key: value for key, value in payload.items() if key != "request_binding"}, secret
        ),
    ):
        raise ValueError("Z-Image worker command binding is invalid")
    if failure is not None:
        failure.stage = "canonical_validation"
    generation = payload.get("generation")
    if (
        not isinstance(generation, Mapping)
        or set(generation) != {"prompt", "seed"}
        or not isinstance(generation["prompt"], str)
        or not generation["prompt"].strip()
        or isinstance(generation["seed"], bool)
        or not isinstance(generation["seed"], int)
        or generation["seed"] < 0
    ):
        raise ValueError("Z-Image worker generation is invalid")
    if not isinstance(payload.get("output_path"), str):
        if failure is not None:
            failure.stage = "canonical_validation"
        raise TypeError("Z-Image worker output path is invalid")
    output = Path(payload["output_path"]).resolve(strict=False)
    if output.suffix.lower() != ".png" or output.exists():
        raise ValueError("Z-Image worker output must be a fresh PNG")
    _validate_recipe_manifest(payload["recipe"])
    return (
        dict(payload["recipe"]),
        payload["device"],
        dict(generation),
        output,
        payload["request_binding"],
        payload["session_binding"],
    )


def _set_core_failure_stage(failure: _FailureContext, stage: str) -> None:
    """Accept only Engine-owned phase names from the core's observer."""

    if stage not in {
        "conditioning",
        "transformer_onload",
        "noise",
        "sampling",
        "decode",
        "publish",
        *_QWEN_FAILURE_STAGES,
    }:
        raise ValueError("Z-Image core reported an invalid diagnostic stage")
    failure.stage = stage
    failure.location = "z_image_turbo.generate"


def _failure_result(
    exc: BaseException, failure: _FailureContext, fallback_binding: str, secret: bytes | None = None
) -> dict[str, object]:
    """Serialize exactly the public-safe diagnostic envelope, never exception text."""

    if isinstance(exc, _BoundFailure):
        cause, context, binding = exc.cause, exc.failure, exc.request_binding
    else:
        cause, context, binding = exc, failure, failure.binding or fallback_binding
    # Do not consult exception text, args, repr, or custom instance attributes:
    # they are a hostile-content channel.  Unknown classes collapse to the
    # fixed public label before fingerprinting.
    cuda_error_code = None
    cuda_health_completed: tuple[str, ...] = ()
    cuda_health_phase = None
    cuda_health_substage = None
    if isinstance(cause, _CudaHealthFailure):
        error_type = cause.error_type
        cuda_error_code = cause.code
        cuda_health_completed = cause.completed
        cuda_health_phase = cause.phase
        cuda_health_substage = cause.substage
    else:
        error_type = _SAFE_EXCEPTION_CLASSES.get(type(cause), "Exception")
    stage = context.stage if context.stage in _FAILURE_STAGES else "canonical_validation"
    location = (
        context.location
        if context.location in _FAILURE_LOCATIONS
        else "z_image_turbo_worker._read_json"
    )
    fingerprint_payload = f"zimage-worker-failure-v1:{error_type}:{stage}:{location}"
    if cuda_error_code is not None:
        fingerprint_payload += f":{cuda_error_code}"
        fingerprint_payload += (
            f":{cuda_health_phase}:{cuda_health_substage}:"
            + ",".join(cuda_health_completed)
        )
    fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()
    value: dict[str, object] = {
        "schema_version": _SCHEMA,
        "ok": False,
        "request_binding": binding if isinstance(binding, str) else "",
        "error_type": error_type,
        "failure_stage": stage,
        "failure_location": location,
        "error_fingerprint": fingerprint,
    }
    if cuda_error_code is not None:
        value["cuda_error_code"] = cuda_error_code
        value["cuda_health_completed"] = list(cuda_health_completed)
        value["cuda_health_phase"] = cuda_health_phase
        value["cuda_health_substage"] = cuda_health_substage
    return _signed_result(value, secret)


def _signed_result(value: dict[str, object], secret: bytes | None) -> dict[str, object]:
    """Return the exact result envelope authenticated by the session capability."""

    key = secret if isinstance(secret, bytes) else b""
    return {**value, "result_binding": _result_binding(value, key)}


def _validate_recipe_manifest(value: object) -> None:
    """Cheap structural/fingerprint check; it imports neither planners nor torch."""

    expected = {
        "schema_version",
        "family",
        "operation",
        "base_model",
        "execution_contract",
        "schedule",
        "components",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Z-Image worker recipe manifest is invalid")
    schedule = {
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "guider": "basic",
        "sampling": "auraflow_shift_3",
        "sampler": "res_multistep",
        "scheduler": "simple",
    }
    contract = {
        "guider": "basic_positive_only",
        "noise": "cpu_fp32_manual_seed_then_transfer",
        "sampler": "res_multistep",
        "scheduler": "simple",
        "shift": 3,
        "output": "png_rgb_1024",
    }
    components = value.get("components")
    if (
        value.get("schema_version") != 1
        or value.get("family") != "zimage"
        or value.get("operation") != "zimage_turbo_t2i_int8_convrot"
        or not isinstance(value.get("base_model"), str)
        or not isinstance(value.get("execution_contract"), Mapping)
        or dict(value["execution_contract"]) != contract
        or not isinstance(value.get("schedule"), Mapping)
        or dict(value["schedule"]) != schedule
        or not isinstance(components, Mapping)
        or frozenset(components)
        not in {
            frozenset({"pipeline_support", "transformer", "text_encoder", "vae"}),
            frozenset(
                {"pipeline_support", "transformer", "text_encoder", "vae", "style_lora"}
            ),
        }
        or not all(isinstance(component, Mapping) for component in components.values())
        or not isinstance(value.get("fingerprint"), str)
    ):
        raise ValueError("Z-Image worker recipe manifest is invalid")
    fingerprint_payload = {
        "schema_version": value["schema_version"],
        "base_model": value["base_model"],
        "operation": value["operation"],
        "schedule": dict(sorted(value["schedule"].items())),
        "components": {role: dict(component) for role, component in sorted(components.items())},
    }
    digest = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if not hmac.compare_digest(value["fingerprint"], f"z-image-turbo:sha256:{digest}"):
        raise ValueError("Z-Image worker recipe fingerprint is invalid")


def _execution_session_binding(
    recipe: Mapping[str, object], canonical_device: str, secret: bytes
) -> str:
    """Bind the loaded session to the concrete device, never the bare request alias."""

    return _binding(
        {
            "recipe": dict(recipe),
            "device": canonical_device,
            "dtype": "bfloat16",
            "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
        },
        secret,
    )


def _execution_runtime_key(
    recipe: Mapping[str, object], canonical_device: str
) -> tuple[object, str, str, str]:
    """Return the exact loaded-runtime identity with an indexed CUDA device."""

    return (
        recipe["fingerprint"],
        canonical_device,
        "bfloat16",
        "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    )


def _binding(value: Mapping[str, object], secret: bytes) -> str:
    return hmac_sha256(value, secret)


def _result_binding(value: Mapping[str, object], secret: bytes) -> str:
    return result_hmac_sha256(value, secret)


def _secret(value: str) -> bytes:
    if len(value) != 64:
        raise ValueError("Z-Image worker capability secret is invalid")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Z-Image worker capability secret is invalid") from exc


def _untrusted_binding(value: object) -> str:
    return (
        value.get("request_binding", "")
        if isinstance(value, Mapping) and isinstance(value.get("request_binding"), str)
        else ""
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
