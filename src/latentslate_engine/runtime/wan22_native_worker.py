"""Disposable native Wan 14B I2V worker entry point.

The parent Engine process deliberately never imports or materializes the native
Wan runtime. This module is invoked in one short-lived Python process per job;
process exit is therefore the authoritative release of its large Windows CPU
heap and any encoder descendants.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LatentSlate disposable Wan 14B worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result)
    progress_path = Path(args.progress)
    gate_path = Path(args.start_gate)
    try:
        _wait_for_start_gate(gate_path)
        payload = _read_json(request_path, _MAX_RESULT_BYTES)
        result = _run(payload, progress_path)
        _write_json(result_path, {"schema_version": _WORKER_SCHEMA_VERSION, "ok": True, **result})
        return 0
    except BaseException as exc:  # noqa: BLE001 - child must publish fatal worker failures.
        _write_json(
            result_path,
            {
                "schema_version": _WORKER_SCHEMA_VERSION,
                "ok": False,
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
        "output_path",
        "device",
        "fps",
        "generation",
    }:
        raise ValueError("native Wan worker request is not canonical")
    if payload["schema_version"] != _WORKER_SCHEMA_VERSION:
        raise ValueError("native Wan worker request schema_version is unsupported")
    source_path = _absolute_file(payload["source_image_path"], "source_image_path")
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
    from .video_output import encode_rgb_video_tensor
    from .wan22_i2v_runtime import NativeWanI2VRuntime, WanI2VArtifactPaths, WanI2VRequest

    recipe = rehydrate_native_wan22_i2v_14b_runtime_request(payload["recipe"])
    _validate_fixed_operation(generation, operation=recipe.operation)
    request = WanI2VRequest(
        image=_load_rgb(source_path),
        prompt=_required_text(generation, "prompt"),
        negative_prompt=_optional_text(generation, "negative_prompt"),
        num_frames=_required_int(generation, "num_frames"),
        height=_required_int(generation, "height"),
        width=_required_int(generation, "width"),
        steps=_required_int(generation, "steps"),
        seed=_required_int(generation, "seed"),
        stage_policy=_required_text(generation, "stage_policy"),
        high_guidance=_required_number(generation, "high_guidance"),
        low_guidance=_required_number(generation, "low_guidance"),
    )
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

    runtime = NativeWanI2VRuntime.load(
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
        return {
            "output_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
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
    generation: Mapping[str, Any], *, operation: str = "comfy_i2v_base"
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
                f"native Wan 14B I2V requires the pinned Comfy operation {key}={value!r}"
            )


if __name__ == "__main__":  # pragma: no cover - exercised through the supervisor
    raise SystemExit(main())
