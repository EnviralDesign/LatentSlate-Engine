"""One-shot Engine-native Wan 2.2 TI2V 5B generation worker."""

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
    parser = argparse.ArgumentParser(description="LatentSlate disposable Wan 5B Kitchen worker")
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
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        payload = _read_json(request_path)
        binding = _validate_binding(payload)
        from ..wan5_kitchen_recipe import rehydrate_wan5_kitchen_runtime_request

        request = rehydrate_wan5_kitchen_runtime_request(payload["request"])
        generation = _validate_generation(payload["generation"], request.operation)
        # Heavy runtime dependencies are deliberately imported only after the
        # complete request, artifact, endpoint, and output contract is bound.
        from .wan5_kitchen import Wan5KitchenGeneration, Wan5KitchenRuntime

        runtime = Wan5KitchenRuntime(request, device=payload["device"])
        result = runtime.generate(
            Wan5KitchenGeneration(**generation),
            progress=lambda progress, message: _append_progress(
                progress_path, {"progress": progress, "message": message}
            ),
            check_cancelled=lambda: None,
        )
        metadata = result.metadata
        _write_json(
            result_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "ok": True,
                "request_binding": binding,
                "output_path": str(result.output_path),
                "output_size_bytes": result.output_path.stat().st_size,
                "metadata": metadata,
                "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            },
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - bounded worker result protocol
        try:
            _write_json(
                result_path,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "ok": False,
                    "request_binding": binding,
                    "error_type": type(exc).__name__,
                },
            )
        except BaseException:  # noqa: BLE001, S110 - worker cannot report further
            pass
        return 1


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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _wait_gate(path: Path) -> None:
    deadline = time.monotonic() + 60.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Wan 5B worker start gate was not opened")
        time.sleep(0.02)


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("Wan 5B worker request is missing or exceeds its bound")
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
        raise ValueError("Wan 5B worker progress is invalid")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if len(raw.encode()) > 4096 or (
        path.exists() and path.stat().st_size + len(raw.encode()) > _MAX_PROGRESS_BYTES
    ):
        raise ValueError("Wan 5B worker progress exceeds its bound")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
