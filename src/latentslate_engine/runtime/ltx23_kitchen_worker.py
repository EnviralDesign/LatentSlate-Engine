"""One-shot Engine-native LTX 2.3 Kitchen generation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LatentSlate disposable Engine-native LTX 2.3 Kitchen worker"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    args = parser.parse_args(argv)
    request_path, result_path, progress_path, gate_path = map(
        Path, (args.request, args.result, args.progress, args.start_gate)
    )
    binding: str | None = None
    try:
        _wait_gate(gate_path)
        # This precedes every torch, diffusers, and Comfy Kitchen import.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        payload = _read_json(request_path)
        if isinstance(payload, Mapping) and isinstance(payload.get("request_binding"), str):
            binding = payload["request_binding"]
        outcome = _run(payload, progress_path)
        _write_json(result_path, {"schema_version": _SCHEMA_VERSION, "ok": True, **outcome})
        return 0
    except BaseException as exc:  # noqa: BLE001 - bounded child error protocol.
        _write_json(
            result_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": False,
                "request_binding": binding,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
            },
        )
        return 1


def _run(payload: Mapping[str, Any], progress_path: Path) -> dict[str, Any]:
    # No path access or heavy import is permitted before this exact JSON binding.
    request_data, generation, device, binding = _validate_bound_payload(payload)
    from ..ltx23_kitchen_recipe import rehydrate_ltx23_kitchen_runtime_request

    request = rehydrate_ltx23_kitchen_runtime_request(request_data)
    from .ltx23_kitchen import (
        LTX23KitchenGeneration,
        LTX23KitchenRuntime,
        validate_ltx23_kitchen_generation,
    )

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
    validate_ltx23_kitchen_generation(request.operation, built)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def report(value: float, message: str | None) -> None:
        _append_progress(progress_path, {"progress": value, "message": message})

    runtime = LTX23KitchenRuntime(request, device=device)
    result = runtime.generate(built, progress=report, check_cancelled=lambda: None)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("LTX 2.3 Kitchen worker did not publish an MP4")
    return {
        "request_binding": binding,
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "metadata": dict(result.metadata),
        "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
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
