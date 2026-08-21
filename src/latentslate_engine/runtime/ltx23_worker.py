"""One-shot LTX 2.3 worker; imports torch only after allocator policy is set."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .framework.worker import (
    WorkerJsonFileError,
    WorkerJsonlFileError,
    append_bounded_jsonl,
    atomic_write_json,
    read_bounded_json,
    sha256_fingerprint,
)

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LatentSlate disposable LTX 2.3 worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--start-gate", required=True)
    args = parser.parse_args(argv)
    request, result, progress, gate = map(Path, (args.request, args.result, args.progress, args.start_gate))
    request_binding: str | None = None
    try:
        _wait_gate(gate)
        # Must precede every torch/diffusers import in this process.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        payload = _read_json(request)
        if isinstance(payload, Mapping) and isinstance(payload.get("request_binding"), str):
            request_binding = payload["request_binding"]
        outcome = _run(payload, progress)
        _write_json(result, {"schema_version": _SCHEMA_VERSION, "ok": True, **outcome})
        return 0
    except BaseException as exc:  # noqa: BLE001 - child must report bounded failure data.
        _write_json(result, {
            "schema_version": _SCHEMA_VERSION,
            "ok": False,
            "request_binding": request_binding,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4096],
        })
        return 1


def _run(payload: Mapping[str, Any], progress_path: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "operation", "settings", "plan", "generation", "request_binding"
    } or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("LTX worker request is not canonical")
    operation, settings_data, plan_data, generation, binding = (
        payload["operation"], payload["settings"], payload["plan"], payload["generation"], payload["request_binding"]
    )
    if operation not in {"t2v", "first_frame", "first_last"} or not isinstance(binding, str):
        raise ValueError("LTX worker operation/binding is invalid")
    _validate_request_binding(
        payload["schema_version"], operation, settings_data, plan_data, generation, binding
    )
    settings = _settings(settings_data)
    plan = _plan(plan_data)
    _validate_generation(generation, operation)
    from .ltx23 import LTX23ConditionRuntime, LTX23Runtime

    output = Path(generation["output_path"]).resolve(strict=False)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def report(value: float, message: str | None) -> None:
        _append_progress(progress_path, {"progress": value, "message": message})

    runtime = LTX23Runtime(settings, plan) if operation == "t2v" else LTX23ConditionRuntime(settings, plan)
    try:
        common = {
            "plan": plan,
            "prompt": generation["prompt"],
            "output_path": output,
            "width": generation["width"],
            "height": generation["height"],
            "duration_seconds": generation["duration_seconds"],
            "seed": generation["seed"],
            "progress": report,
            "check_cancelled": lambda: None,
        }
        if operation == "t2v":
            metadata = runtime.generate(**common)
        else:
            metadata = runtime.generate(
                **common,
                start_image_path=Path(generation["start_image_path"]),
                end_image_path=None if operation == "first_frame" else Path(generation["end_image_path"]),
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("LTX worker did not publish an MP4")
        return {
            "request_binding": binding,
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "metadata": metadata,
            "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        }
    finally:
        runtime.unload()


def _settings(value: Any):
    if not isinstance(value, Mapping) or set(value) != {"home", "model_id", "profile", "device"}:
        raise ValueError("LTX worker settings are invalid")
    if not all(isinstance(value[key], str) and value[key] for key in value):
        raise ValueError("LTX worker settings values are invalid")
    from ..config import Settings

    return Settings(
        home=Path(value["home"]), token=None, max_upload_bytes=1,
        h3_model_id="worker-unused", h3_profile="worker-unused", h3_device="cpu",
        ltx23_model_id=value["model_id"], ltx23_profile=value["profile"], ltx23_device=value["device"],
        cache_enabled=False,
    )


def _plan(value: Any):
    expected = {
        "pipeline_fingerprint", "model_id", "model_resource_id", "model_path", "model_format",
        "model_precision", "model_quantization", "device", "quantization", "attention", "offload",
        "vae_tiling", "vae_slicing", "cache", "low_cpu_mem_usage", "components",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("LTX worker plan is invalid")
    from .kit import ResolvedRuntimePlan, RuntimeComponent

    components = value["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("LTX worker plan components are invalid")
    rebuilt = []
    for item in components:
        if not isinstance(item, Mapping) or set(item) != {"name", "path", "signature"}:
            raise ValueError("LTX worker component is invalid")
        if not isinstance(item["name"], str) or not isinstance(item["path"], str) or not isinstance(item["signature"], dict):
            raise TypeError("LTX worker component fields are invalid")
        rebuilt.append(RuntimeComponent(item["name"], Path(item["path"]).resolve(strict=True), item["signature"]))
    plan = ResolvedRuntimePlan(
        family="ltx23", variant_key=None, model_id=_string(value, "model_id"),
        model_resource_id=value["model_resource_id"] if isinstance(value["model_resource_id"], str) else None,
        model_path=Path(_string(value, "model_path")).resolve(strict=True), model_format=_string(value, "model_format"),
        model_precision=_string(value, "model_precision"), model_quantization=_string(value, "model_quantization"),
        device=_string(value, "device"), quantization=_string(value, "quantization"), attention=_string(value, "attention"),
        offload=_string(value, "offload"), compile=False, compile_mode="default", compile_fullgraph=False,
        compile_dynamic=False, vae_tiling=_string(value, "vae_tiling"), vae_slicing=_string(value, "vae_slicing"),
        cache="none", group_offload_blocks=1, group_offload_use_stream=False, group_offload_record_stream=False,
        low_cpu_mem_usage=bool(value["low_cpu_mem_usage"]), keep_pipeline_loaded=False, components=tuple(rebuilt),
    )
    if plan.pipeline_fingerprint != value["pipeline_fingerprint"]:
        raise ValueError("LTX worker plan fingerprint does not match request")
    plan.revalidate_components()
    return plan


def _validate_generation(value: Any, operation: str) -> None:
    expected = {"prompt", "width", "height", "duration_seconds", "seed", "start_image_path", "end_image_path", "output_path"}
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value["prompt"], str) or not value["prompt"].strip():
        raise ValueError("LTX worker generation is invalid")
    if isinstance(value["width"], bool) or not isinstance(value["width"], int) or isinstance(value["height"], bool) or not isinstance(value["height"], int) or isinstance(value["seed"], bool) or not isinstance(value["seed"], int) or not isinstance(value["duration_seconds"], (int, float)):
        raise TypeError("LTX worker generation parameters are invalid")
    start, end = value["start_image_path"], value["end_image_path"]
    if operation == "t2v" and (start is not None or end is not None):
        raise ValueError("LTX T2V worker received endpoint images")
    if operation == "first_frame" and (not isinstance(start, str) or end is not None):
        raise ValueError("LTX first-frame worker endpoints are invalid")
    if operation == "first_last" and (not isinstance(start, str) or not isinstance(end, str)):
        raise ValueError("LTX first+last worker endpoints are invalid")
    for path in (start, end):
        if path is not None and not Path(path).is_file():
            raise ValueError("LTX worker endpoint image does not exist")
    if not isinstance(value["output_path"], str) or Path(value["output_path"]).suffix.lower() != ".mp4":
        raise ValueError("LTX worker output path is invalid")


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item:
        raise ValueError(f"LTX worker plan {key} is invalid")
    return item


def _request_binding(
    schema_version: object,
    operation: object,
    settings: object,
    plan: object,
    generation: object,
) -> str:
    value = {
        "schema_version": schema_version,
        "operation": operation,
        "settings": settings,
        "plan": plan,
        "generation": generation,
    }
    return sha256_fingerprint(value)


def _validate_request_binding(
    schema_version: object,
    operation: object,
    settings: object,
    plan: object,
    generation: object,
    binding: object,
) -> None:
    if not isinstance(binding, str) or binding != _request_binding(
        schema_version, operation, settings, plan, generation
    ):
        raise ValueError("LTX worker request binding does not match its canonical payload")


def _wait_gate(path: Path) -> None:
    deadline = time.monotonic() + 60
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("LTX worker start gate was not opened")
        time.sleep(0.02)


def _read_json(path: Path) -> Any:
    try:
        return read_bounded_json(path, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonFileError:
        raise ValueError("LTX worker request is missing or exceeds its bound")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _append_progress(path: Path, value: Mapping[str, Any]) -> None:
    try:
        append_bounded_jsonl(path, value, maximum_bytes=_MAX_JSON_BYTES)
    except WorkerJsonlFileError as exc:
        raise ValueError("LTX worker progress exceeds its bound") from exc


if __name__ == "__main__":  # pragma: no cover - child-only entry point
    raise SystemExit(main())
