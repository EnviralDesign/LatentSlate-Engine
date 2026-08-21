"""One-shot LTX 2.3 worker; imports torch only after allocator policy is set."""

from __future__ import annotations

import os
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


@dataclass(frozen=True, slots=True)
class _BoundRequest:
    operation: str
    settings: Any
    plan: Any
    generation: dict[str, Any]
    binding: str


class _LTX23Handler:
    """LTX-specific validation and generation behind the shared child harness."""

    def bind_request(
        self, payload: Any, context: DisposableChildContext
    ) -> _BoundRequest:
        context.stage = "validate_bound_request"
        return _bind_request(payload, context)

    def load(self, request: _BoundRequest, context: DisposableChildContext) -> Any:
        context.stage = "import_runtime"
        from .ltx23 import LTX23ConditionRuntime, LTX23Runtime

        context.stage = "initialize_runtime"
        if request.operation == "t2v":
            return LTX23Runtime(request.settings, request.plan)
        return LTX23ConditionRuntime(request.settings, request.plan)

    def run(
        self, runtime: Any, request: _BoundRequest, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        generation = request.generation
        output = Path(generation["output_path"]).resolve(strict=False)
        common = {
            "plan": request.plan,
            "prompt": generation["prompt"],
            "output_path": output,
            "width": generation["width"],
            "height": generation["height"],
            "duration_seconds": generation["duration_seconds"],
            "seed": generation["seed"],
            "progress": context.publish_progress,
            "check_cancelled": lambda: None,
        }
        context.stage = "generate"
        if request.operation == "t2v":
            metadata = runtime.generate(**common)
        else:
            metadata = runtime.generate(
                **common,
                start_image_path=Path(generation["start_image_path"]),
                end_image_path=None
                if request.operation == "first_frame"
                else Path(generation["end_image_path"]),
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("LTX worker did not publish an MP4")
        return {
            "schema_version": _SCHEMA_VERSION,
            "ok": True,
            "request_binding": request.binding,
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "metadata": metadata,
            "allocator_policy": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        }

    def unload(
        self, runtime: Any, request: _BoundRequest, context: DisposableChildContext
    ) -> None:
        context.stage = "unload_runtime"
        runtime.unload()

    def failure_result(
        self, exc: BaseException, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "ok": False,
            "request_binding": context.binding,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4096],
        }

    def stage_for_progress(self, message: str | None) -> str:
        return "generate"

    def protocol_error(self, reason: str) -> BaseException:
        errors: dict[str, BaseException] = {
            "start_gate_timeout": TimeoutError("LTX worker start gate was not opened"),
            "request_bound": ValueError(
                "LTX worker request is missing or exceeds its bound"
            ),
            "invalid_progress": ValueError("LTX worker progress is invalid"),
            "progress_bound": ValueError("LTX worker progress exceeds its bound"),
        }
        try:
            return errors[reason]
        except KeyError as exc:
            raise ValueError("unknown disposable worker protocol error") from exc


def main(argv: list[str] | None = None) -> int:
    paths = parse_disposable_child_paths(
        argv, description="LatentSlate disposable LTX 2.3 worker"
    )
    # Must precede every torch/diffusers import in this process.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return run_disposable_child(
        paths,
        _LTX23Handler(),
        maximum_json_bytes=_MAX_JSON_BYTES,
        maximum_progress_bytes=_MAX_JSON_BYTES,
    )


def _bind_request(payload: Any, context: DisposableChildContext) -> _BoundRequest:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "operation", "settings", "plan", "generation", "request_binding"
    } or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("LTX worker request is not canonical")
    operation, settings_data, plan_data, generation, binding = (
        payload["operation"], payload["settings"], payload["plan"], payload["generation"], payload["request_binding"]
    )
    if operation not in {"t2v", "first_frame", "first_last"} or not isinstance(binding, str):
        raise ValueError("LTX worker operation/binding is invalid")
    context.binding = binding
    _validate_request_binding(
        payload["schema_version"], operation, settings_data, plan_data, generation, binding
    )
    settings = _settings(settings_data)
    plan = _plan(plan_data)
    _validate_generation(generation, operation)
    return _BoundRequest(operation, settings, plan, dict(generation), binding)


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


if __name__ == "__main__":  # pragma: no cover - child-only entry point
    raise SystemExit(main())
