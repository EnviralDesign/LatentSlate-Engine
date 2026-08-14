"""One-shot Engine-native LTX 2.3 Kitchen generation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024


@dataclass(slots=True)
class _FailureContext:
    """Bounded diagnostic state that never contains a prompt, asset, or path."""

    stage: str = "worker_startup"
    binding: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LatentSlate disposable Engine-native LTX 2.3 Kitchen worker"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    parser.add_argument("--command")
    args = parser.parse_args(argv)
    request_path, result_path, progress_path, gate_path = map(
        Path, (args.request, args.result, args.progress, args.start_gate)
    )
    binding: str | None = None
    failure = _FailureContext()
    try:
        _wait_gate(gate_path)
        # This is deliberately before every heavyweight import. A disposable
        # worker can spend several seconds loading Python/CUDA packages before
        # it can materialize a component; reporting the boundary makes that
        # cold-start interval truthful rather than looking like a stuck queue.
        _append_progress(progress_path, {"progress": 0.0, "message": "LTX worker started"})
        # This precedes every torch, diffusers, and Comfy Kitchen import.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        payload = _read_json(request_path)
        if isinstance(payload, Mapping) and isinstance(payload.get("request_binding"), str):
            binding = payload["request_binding"]
        if args.command:
            return _run_session(payload, result_path, progress_path, Path(args.command), failure)
        outcome = _run(payload, progress_path, failure)
        _write_json(result_path, {"schema_version": _SCHEMA_VERSION, "ok": True, **outcome})
        return 0
    except BaseException as exc:  # noqa: BLE001 - bounded child error protocol.
        _write_json(
            result_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": False,
                "request_binding": failure.binding or binding,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
                **_failure_diagnostic(exc, failure),
            },
        )
        return 1


def _run(
    payload: Mapping[str, Any],
    progress_path: Path,
    failure: _FailureContext,
) -> dict[str, Any]:
    # No path access or heavy import is permitted before this exact JSON binding.
    failure.stage = "validate_bound_request"
    request_data, generation, device, binding = _validate_bound_payload(payload)
    failure.stage = "rehydrate_recipe"
    _append_progress(progress_path, {"progress": 0.0, "message": "Rehydrating LTX recipe"})
    from ..ltx23_kitchen_recipe import rehydrate_ltx23_kitchen_runtime_request

    request = rehydrate_ltx23_kitchen_runtime_request(request_data)
    failure.stage = "import_runtime"
    _append_progress(progress_path, {"progress": 0.0, "message": "Importing LTX runtime"})
    from .ltx23_kitchen import (
        LTX23KitchenGeneration,
        LTX23KitchenRuntime,
        validate_ltx23_kitchen_generation,
    )

    failure.stage = "build_generation"
    _append_progress(progress_path, {"progress": 0.0, "message": "Preparing LTX generation"})
    output = Path(generation["output_path"]).resolve(strict=False)
    built = LTX23KitchenGeneration(
        generation["prompt"],
        output,
        generation["width"],
        generation["height"],
        generation["num_frames"],
        generation["seed"],
        None if generation["start_image_path"] is None else Path(generation["start_image_path"]),
        None if generation["end_image_path"] is None else Path(generation["end_image_path"]),
        generation["start_image_identity"],
        generation["end_image_identity"],
    )
    failure.stage = "validate_generation"
    validate_ltx23_kitchen_generation(request.operation, built)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def report(value: float, message: str | None) -> None:
        failure.stage = _progress_stage(message)
        _append_progress(progress_path, {"progress": value, "message": message})

    failure.stage = "initialize_runtime"
    runtime = LTX23KitchenRuntime(request, device=device)
    failure.stage = "generate"
    result = runtime.generate(built, progress=report, check_cancelled=lambda: None)
    failure.stage = "verify_output"
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("LTX 2.3 Kitchen worker did not publish an MP4")
    return {
        "request_binding": binding,
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "metadata": dict(result.metadata),
        "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    }


def _run_session(
    payload: Mapping[str, Any],
    result_path: Path,
    progress_path: Path,
    command_path: Path,
    failure: _FailureContext,
) -> int:
    """Serve serial commands for one exact request-bound LTX component session."""

    failure.stage = "validate_bound_request"
    request_data, generation, device, binding = _validate_bound_payload(payload)
    failure.binding = binding
    failure.stage = "rehydrate_recipe"
    _append_progress(progress_path, {"progress": 0.0, "message": "Rehydrating LTX recipe"})
    from ..ltx23_kitchen_recipe import rehydrate_ltx23_kitchen_runtime_request

    request = rehydrate_ltx23_kitchen_runtime_request(request_data)
    request_fingerprint = request.fingerprint
    failure.stage = "import_runtime"
    _append_progress(progress_path, {"progress": 0.0, "message": "Importing LTX runtime"})
    from .ltx23_kitchen import (
        LTX23KitchenGeneration,
        LTX23KitchenRuntime,
        validate_ltx23_kitchen_generation,
    )

    runtime = LTX23KitchenRuntime(request, device=device)

    def execute(value: Mapping[str, Any], request_binding: str) -> None:
        failure.binding = request_binding
        failure.stage = "build_generation"
        _append_progress(progress_path, {"progress": 0.0, "message": "Preparing LTX generation"})
        output = Path(value["output_path"]).resolve(strict=False)
        built = LTX23KitchenGeneration(
            value["prompt"], output, value["width"], value["height"], value["num_frames"], value["seed"],
            None if value["start_image_path"] is None else Path(value["start_image_path"]),
            None if value["end_image_path"] is None else Path(value["end_image_path"]),
            value["start_image_identity"], value["end_image_identity"],
        )
        failure.stage = "validate_generation"
        validate_ltx23_kitchen_generation(request.operation, built)

        def report(value: float, message: str | None) -> None:
            failure.stage = _progress_stage(message)
            _append_progress(progress_path, {"progress": value, "message": message})

        failure.stage = "generate"
        generated = runtime.generate(built, progress=report, check_cancelled=lambda: None)
        failure.stage = "verify_output"
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("LTX 2.3 Kitchen worker did not publish an MP4")
        _write_json(
            result_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": True,
                "request_binding": request_binding,
                "output_path": str(output),
                "output_size_bytes": output.stat().st_size,
                "metadata": dict(generated.metadata),
                "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            },
        )

    execute(generation, binding)
    while True:
        next_payload = _wait_command(command_path)
        command_path.unlink(missing_ok=True)
        request_data, generation, device, binding = _validate_bound_payload(next_payload)
        failure.binding = binding
        if device != "cuda":
            raise ValueError("LTX 2.3 Kitchen worker requires direct CUDA execution")
        next_request = rehydrate_ltx23_kitchen_runtime_request(request_data)
        if next_request.fingerprint != request_fingerprint:
            raise ValueError("LTX 2.3 Kitchen worker command recipe does not match its session")
        execute(generation, binding)


def _progress_stage(message: str | None) -> str:
    """Map Engine-owned progress text to a safe, stable error phase."""

    if message is None:
        return "generate"
    prefixes = (
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
    if binding != _fingerprint(unsigned):
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


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _wait_gate(path: Path) -> None:
    deadline = time.monotonic() + 60.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("LTX 2.3 Kitchen worker start gate was not opened")
        time.sleep(0.02)


def _wait_command(path: Path) -> Mapping[str, Any]:
    """Wait for one parent-owned atomic command file in a live session."""

    while not path.is_file():
        time.sleep(0.02)
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError("LTX 2.3 Kitchen worker command is invalid")
    return value


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("LTX 2.3 Kitchen worker request is missing or exceeds its bound")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_progress(path: Path, value: Mapping[str, Any]) -> None:
    progress = value.get("progress")
    if (
        set(value) != {"progress", "message"}
        or isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not 0 <= float(progress) <= 1
        or not isinstance(value.get("message"), (str, type(None)))
    ):
        raise ValueError("LTX 2.3 Kitchen worker progress is invalid")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if len(raw.encode()) > 4096 or (
        path.exists() and path.stat().st_size + len(raw.encode()) > _MAX_PROGRESS_BYTES
    ):
        raise ValueError("LTX 2.3 Kitchen worker progress exceeds its bound")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
