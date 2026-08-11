from __future__ import annotations

import gc
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .signatures import path_signature

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan, LoraExecution


_ATTENTION_BACKENDS = {
    "native": "native",
    "flash_hub": "flash_hub",
    "flash3_hub": "_flash_3_hub",
    "flash4_hub": "flash_4_hub",
    "sage_hub": "sage_hub",
}

PipelineParameterValue = str | int | float | bool | None
PipelineParameters = tuple[tuple[str, PipelineParameterValue], ...]


def _canonical_pipeline_parameters(
    parameters: Iterable[tuple[str, PipelineParameterValue]],
) -> PipelineParameters:
    """Validate and freeze pipeline-load inputs for stable manager keying."""

    canonical: dict[str, PipelineParameterValue] = {}
    for parameter in parameters:
        if not isinstance(parameter, tuple) or len(parameter) != 2:
            raise ValueError("Pipeline parameters must be (name, JSON scalar) pairs")
        name, value = parameter
        if not isinstance(name, str) or not name:
            raise ValueError("Pipeline parameter names must be non-empty strings")
        if name in canonical:
            raise ValueError(f"Pipeline parameter {name!r} is declared more than once")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"Pipeline parameter {name!r} must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Pipeline parameter {name!r} must be finite")
        canonical[name] = value
    return tuple(sorted(canonical.items()))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def stable_fingerprint(namespace: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class RuntimeComponent:
    name: str
    path: Path
    signature: dict[str, Any]

    @classmethod
    def capture(cls, name: str, path: Path) -> RuntimeComponent:
        resolved = Path(path).resolve(strict=True)
        return cls(name=name, path=resolved, signature=path_signature(resolved))

    def provenance(self) -> dict[str, Any]:
        signature = self.signature
        return {
            "name": self.name,
            "kind": signature.get("kind"),
            "files": signature.get("files"),
            "bytes": signature.get("bytes", signature.get("size")),
            "digest": signature.get("manifest_digest", signature.get("digest")),
        }

    def revalidate(self) -> None:
        """Fail if this component changed after its runtime plan was captured."""

        if path_signature(self.path) != self.signature:
            raise RuntimeError(
                f"Runtime component {self.name!r} changed after planning; "
                "refresh resources and resolve a new execution plan"
            )


@dataclass(frozen=True, slots=True)
class RuntimeDefaults:
    family: str
    model_id: str
    model_path: Path
    model_format: str
    device: str
    quantization: str
    attention: str
    offload: str
    artifact_precision: str = "bf16"
    artifact_quantization: str = "native"
    vae_tiling: str = "off"
    vae_slicing: str = "off"
    cache: str = "both"
    low_cpu_mem_usage: bool = True
    keep_pipeline_loaded: bool = True
    group_offload_blocks: int = 1
    group_offload_use_stream: bool = False
    group_offload_record_stream: bool = False
    component_paths: tuple[tuple[str, Path], ...] = ()
    # Pipeline-load inputs must remain immutable after manager keying. Dynamic,
    # request-scoped values belong in runtime_parameters instead.
    pipeline_parameters: PipelineParameters = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline_parameters",
            _canonical_pipeline_parameters(self.pipeline_parameters),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRuntimePlan:
    family: str
    variant_key: str | None
    model_id: str
    model_resource_id: str | None
    model_path: Path
    model_format: str
    model_precision: str
    model_quantization: str
    device: str
    quantization: str
    attention: str
    offload: str
    compile: bool
    compile_mode: str
    compile_fullgraph: bool
    compile_dynamic: bool
    vae_tiling: str
    vae_slicing: str
    cache: str
    group_offload_blocks: int
    group_offload_use_stream: bool
    group_offload_record_stream: bool
    low_cpu_mem_usage: bool
    keep_pipeline_loaded: bool
    components: tuple[RuntimeComponent, ...] = ()
    loras: tuple[LoraExecution, ...] = ()
    pipeline_parameters: PipelineParameters = ()
    runtime_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline_parameters",
            _canonical_pipeline_parameters(self.pipeline_parameters),
        )

    @property
    def cache_prompt(self) -> bool:
        return self.cache in {"both", "prompt"}

    @property
    def cache_media(self) -> bool:
        return self.cache in {"both", "media"}

    @property
    def pipeline_payload(self) -> dict[str, Any]:
        """Values that require a distinct loaded/compiled pipeline instance."""

        return {
            "family": self.family,
            "model_id": self.model_id,
            "model_resource_id": self.model_resource_id,
            "components": [
                {"name": component.name, "signature": component.signature}
                for component in self.components
            ],
            "model_format": self.model_format,
            "model_precision": self.model_precision,
            "model_quantization": self.model_quantization,
            "device": self.device,
            "quantization": self.quantization,
            "attention": self.attention,
            "offload": self.offload,
            "compile": self.compile,
            "compile_mode": self.compile_mode if self.compile else None,
            "compile_fullgraph": self.compile_fullgraph if self.compile else None,
            "compile_dynamic": self.compile_dynamic if self.compile else None,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "group_offload_blocks": (
                self.group_offload_blocks if self.offload == "group_block" else None
            ),
            "group_offload_use_stream": (
                self.group_offload_use_stream
                if self.offload in {"group_block", "group_leaf"}
                else None
            ),
            "group_offload_record_stream": (
                self.group_offload_record_stream
                if self.offload in {"group_block", "group_leaf"}
                else None
            ),
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "pipeline_parameters": dict(self.pipeline_parameters),
        }

    @property
    def pipeline_fingerprint(self) -> str:
        return stable_fingerprint(f"runtime:{self.family}", self.pipeline_payload)

    @property
    def lora_signature(self) -> str:
        payload = [
            {
                "slot": lora.slot,
                "resource_id": lora.resource_id,
                "resource": path_signature(lora.path),
                "strength": lora.strength,
            }
            for lora in self.loras
        ]
        return stable_fingerprint(f"loras:{self.family}", {"loras": payload})

    def component_path(self, name: str) -> Path:
        for component in self.components:
            if component.name == name:
                return component.path
        raise KeyError(f"Runtime plan has no component named {name!r}")

    def revalidate_components(self) -> None:
        """Recheck every captured component immediately around first load."""

        for component in self.components:
            component.revalidate()

    def assert_same_pipeline(self, other: ResolvedRuntimePlan) -> None:
        if self.pipeline_fingerprint != other.pipeline_fingerprint:
            raise RuntimeError(
                "Runtime plan does not match the loaded pipeline fingerprint; "
                "the RuntimeManager key is stale or incomplete."
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "variant_key": self.variant_key,
            "family": self.family,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "model": {
                "id": self.model_resource_id or self.model_id,
                "format": self.model_format,
                "precision": self.model_precision,
                "quantization": self.model_quantization,
                "override": self.model_resource_id is not None,
                "components": [component.provenance() for component in self.components],
            },
            "optimizations": {
                "attention": self.attention,
                "offload": self.offload,
                "quantization": self.quantization,
                "compile": self.compile,
                "compile_mode": self.compile_mode if self.compile else None,
                "compile_fullgraph": self.compile_fullgraph if self.compile else None,
                "compile_dynamic": self.compile_dynamic if self.compile else None,
                "vae_tiling": self.vae_tiling,
                "vae_slicing": self.vae_slicing,
                "cache": self.cache,
                "group_offload_blocks": (
                    self.group_offload_blocks if self.offload == "group_block" else None
                ),
                "group_offload_use_stream": (
                    self.group_offload_use_stream
                    if self.offload in {"group_block", "group_leaf"}
                    else None
                ),
                "group_offload_record_stream": (
                    self.group_offload_record_stream
                    if self.offload in {"group_block", "group_leaf"}
                    else None
                ),
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
                "keep_pipeline_loaded": self.keep_pipeline_loaded,
            },
            "loras": [
                {
                    "slot": lora.slot,
                    "resource_id": lora.resource_id,
                    "strength": lora.strength,
                }
                for lora in self.loras
            ],
            "pipeline_parameters": dict(self.pipeline_parameters),
            "runtime_parameters": dict(self.runtime_parameters),
        }


def _resolve_mode(requested: Any, inherited: str, *, label: str) -> str:
    mode = str(requested or "inherit")
    if mode == "inherit":
        return inherited
    if mode == "auto":
        raise ValueError(
            f"{label}='auto' is not resolved by the generic runtime kit; "
            "the family adapter must choose and advertise one exact mode"
        )
    return mode


def resolve_runtime_plan(
    execution: ExecutionPlan | None,
    defaults: RuntimeDefaults,
) -> ResolvedRuntimePlan:
    if execution is not None and execution.family != defaults.family:
        raise ValueError(
            f"Execution plan family {execution.family!r} does not match "
            f"runtime family {defaults.family!r}"
        )
    optimizations = execution.optimizations if execution and execution.optimizations else {}
    model_path = execution.model_path if execution and execution.model_path else defaults.model_path
    model_format = (
        execution.model_format if execution and execution.model_format else defaults.model_format
    )
    model_precision = (
        execution.model_precision
        if execution and execution.model_precision
        else defaults.artifact_precision
    )
    model_quantization = (
        execution.model_quantization
        if execution and execution.model_quantization
        else defaults.artifact_quantization
    )
    cache_mode = str(optimizations.get("cache", "inherit"))
    if cache_mode == "inherit":
        cache_mode = defaults.cache
    if cache_mode not in {"none", "prompt", "media", "both"}:
        raise ValueError(f"cache mode {cache_mode!r} is not implemented by the safe runtime kit")

    component_paths: list[tuple[str, Path]] = [("model", Path(model_path))]
    if execution is None or execution.model_path is None:
        component_paths.extend(defaults.component_paths)
    component_names = [name for name, _path in component_paths]
    duplicate_names = sorted(
        name for name in set(component_names) if component_names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError("Runtime component names must be unique: " + ", ".join(duplicate_names))
    components = tuple(
        RuntimeComponent.capture(name, component_path) for name, component_path in component_paths
    )

    compile_enabled = bool(optimizations.get("compile", False))
    group_offload_use_stream = bool(
        optimizations.get(
            "group_offload_use_stream",
            defaults.group_offload_use_stream,
        )
    )
    group_offload_record_stream = group_offload_use_stream and bool(
        optimizations.get(
            "group_offload_record_stream",
            defaults.group_offload_record_stream,
        )
    )
    resolved_quantization = _resolve_mode(
        optimizations.get("quantization"),
        defaults.quantization,
        label="quantization",
    )
    _assert_artifact_compatibility(
        quantization=resolved_quantization,
        precision=str(model_precision),
        artifact_quantization=str(model_quantization),
    )
    return ResolvedRuntimePlan(
        family=defaults.family,
        variant_key=execution.variant_key if execution else None,
        model_id=defaults.model_id,
        model_resource_id=execution.model_resource_id if execution else None,
        model_path=Path(model_path).resolve(),
        model_format=str(model_format),
        model_precision=str(model_precision),
        model_quantization=str(model_quantization),
        device=defaults.device,
        quantization=resolved_quantization,
        attention=_resolve_mode(
            optimizations.get("attention"),
            defaults.attention,
            label="attention",
        ),
        offload=_resolve_mode(
            optimizations.get("offload"),
            defaults.offload,
            label="offload",
        ),
        compile=compile_enabled,
        compile_mode=str(optimizations.get("compile_mode", "default")),
        compile_fullgraph=bool(optimizations.get("compile_fullgraph", False)),
        compile_dynamic=bool(optimizations.get("compile_dynamic", False)),
        vae_tiling=_resolve_mode(
            optimizations.get("vae_tiling"),
            defaults.vae_tiling,
            label="VAE tiling",
        ),
        vae_slicing=_resolve_mode(
            optimizations.get("vae_slicing"),
            defaults.vae_slicing,
            label="VAE slicing",
        ),
        cache=cache_mode,
        group_offload_blocks=int(
            optimizations.get("group_offload_blocks") or defaults.group_offload_blocks
        ),
        group_offload_use_stream=group_offload_use_stream,
        group_offload_record_stream=group_offload_record_stream,
        low_cpu_mem_usage=bool(optimizations.get("low_cpu_mem_usage", defaults.low_cpu_mem_usage)),
        keep_pipeline_loaded=bool(
            optimizations.get("keep_pipeline_loaded", defaults.keep_pipeline_loaded)
        ),
        components=components,
        loras=execution.loras if execution else (),
        pipeline_parameters=tuple(defaults.pipeline_parameters),
        runtime_parameters=(dict(execution.runtime_parameters or {}) if execution else {}),
    )


def _assert_artifact_compatibility(
    *,
    quantization: str,
    precision: str,
    artifact_quantization: str,
) -> None:
    """Runtime defence in depth: plans select artifacts; they never convert them."""

    if quantization in {"int8", "nvfp4", "gguf"}:
        if artifact_quantization != quantization:
            raise ValueError(
                f"quantization={quantization!r} requires a matching pre-quantized "
                f"artifact, found {artifact_quantization!r}"
            )
        return
    if quantization in {"bf16", "fp16", "fp8"}:
        if artifact_quantization != "native" or precision != quantization:
            raise ValueError(
                f"quantization={quantization!r} requires a native {quantization} "
                f"artifact, found precision={precision!r}, "
                f"quantization={artifact_quantization!r}"
            )
        return
    if quantization == "native" and artifact_quantization != "native":
        raise ValueError(
            "quantization='native' requires an unquantized artifact; "
            f"found {artifact_quantization!r}"
        )


def apply_attention_backend(pipeline: Any, mode: str) -> str:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("The pipeline has no transformer attention backend to configure")
    try:
        backend = _ATTENTION_BACKENDS[mode]
    except KeyError as exc:
        raise RuntimeError(
            f"Attention mode {mode!r} is not implemented by the runtime kit"
        ) from exc

    if mode == "native":
        reset = getattr(transformer, "reset_attention_backend", None)
        if callable(reset):
            reset()
        else:
            setter = getattr(transformer, "set_attention_backend", None)
            if not callable(setter):
                raise TypeError("The transformer does not support attention dispatch")
            setter("native")
        return backend

    setter = getattr(transformer, "set_attention_backend", None)
    if not callable(setter):
        raise TypeError("The transformer does not support attention dispatch")
    setter(backend)
    return backend


def _set_vae_toggle(pipeline: Any, feature: str, mode: str) -> None:
    if mode not in {"on", "off"}:
        raise RuntimeError(f"VAE {feature} mode {mode!r} is not implemented")
    action = "enable" if mode == "on" else "disable"
    pipeline_method = getattr(pipeline, f"{action}_vae_{feature}", None)
    if callable(pipeline_method):
        pipeline_method()
        return
    vae = getattr(pipeline, "vae", None)
    vae_method = getattr(vae, f"{action}_{feature}", None)
    if callable(vae_method):
        vae_method()
        return
    raise RuntimeError(f"The pipeline VAE does not support {feature}")


def apply_vae_policy(pipeline: Any, *, tiling: str, slicing: str) -> None:
    _set_vae_toggle(pipeline, "tiling", tiling)
    _set_vae_toggle(pipeline, "slicing", slicing)


def apply_compile_policy(
    pipeline: Any,
    *,
    enabled: bool,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> str | None:
    if not enabled:
        return None
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("The pipeline has no transformer to compile")
    kwargs = {"mode": mode, "fullgraph": fullgraph, "dynamic": dynamic}
    repeated = getattr(transformer, "_repeated_blocks", None)
    regional = getattr(transformer, "compile_repeated_blocks", None)
    if repeated and callable(regional):
        regional(**kwargs)
        return "regional"

    import torch

    pipeline.transformer = torch.compile(transformer, **kwargs)
    return "full"


def apply_offload_policy(pipeline: Any, plan: ResolvedRuntimePlan) -> None:
    import torch

    mode = plan.offload
    if mode == "none":
        pipeline.to(plan.device)
        return
    if mode == "model":
        pipeline.enable_model_cpu_offload(device=plan.device)
        return
    if mode == "sequential":
        pipeline.enable_sequential_cpu_offload(device=plan.device)
        return
    if mode not in {"group_block", "group_leaf"}:
        raise RuntimeError(f"Offload mode {mode!r} is not implemented by the runtime kit")

    kwargs: dict[str, Any] = {
        "onload_device": torch.device(plan.device),
        "offload_device": torch.device("cpu"),
        "offload_type": "block_level" if mode == "group_block" else "leaf_level",
        "use_stream": plan.group_offload_use_stream,
        "record_stream": (plan.group_offload_record_stream and plan.group_offload_use_stream),
        "low_cpu_mem_usage": plan.low_cpu_mem_usage,
    }
    if mode == "group_block":
        kwargs["num_blocks_per_group"] = plan.group_offload_blocks

    pipeline_method = getattr(pipeline, "enable_group_offload", None)
    if callable(pipeline_method):
        pipeline_method(**kwargs)
        return

    from diffusers.hooks import apply_group_offloading

    applied = False
    for component in getattr(pipeline, "components", {}).values():
        if not isinstance(component, torch.nn.Module):
            continue
        component_method = getattr(component, "enable_group_offload", None)
        if callable(component_method):
            component_method(**kwargs)
        else:
            apply_group_offloading(component, **kwargs)
        applied = True
    if not applied:
        raise RuntimeError("The pipeline has no modules that support group offloading")


def apply_pipeline_kit(pipeline: Any, plan: ResolvedRuntimePlan) -> dict[str, Any]:
    backend = apply_attention_backend(pipeline, plan.attention)
    apply_vae_policy(
        pipeline,
        tiling=plan.vae_tiling,
        slicing=plan.vae_slicing,
    )
    compile_scope = apply_compile_policy(
        pipeline,
        enabled=plan.compile,
        mode=plan.compile_mode,
        fullgraph=plan.compile_fullgraph,
        dynamic=plan.compile_dynamic,
    )
    apply_offload_policy(pipeline, plan)
    return {
        "attention_backend": backend,
        "compile_scope": compile_scope,
        "offload": plan.offload,
        "vae_tiling": plan.vae_tiling,
        "vae_slicing": plan.vae_slicing,
    }


def _adapter_name(resource_id: str) -> str:
    digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]
    return f"ls_{digest}"


def _lora_file_signature(lora: LoraExecution) -> str:
    return stable_fingerprint(
        "lora-file",
        {
            "resource_id": lora.resource_id,
            "resource": path_signature(lora.path),
        },
    )


@dataclass(slots=True)
class _LoadedLora:
    adapter_name: str
    resource_id: str
    signature: str


class LoraLifecycle:
    """Bounded, reusable Diffusers/PEFT adapter lifecycle for one pipeline."""

    def __init__(self, *, max_loaded: int = 8) -> None:
        self.max_loaded = max(1, int(max_loaded))
        self._loaded: OrderedDict[str, _LoadedLora] = OrderedDict()

    def apply(
        self,
        pipeline: Any,
        loras: tuple[LoraExecution, ...],
        *,
        low_cpu_mem_usage: bool,
    ) -> dict[str, Any]:
        if not all(hasattr(pipeline, name) for name in ("load_lora_weights", "set_adapters")):
            if loras:
                raise RuntimeError("This pipeline does not implement the Diffusers LoRA lifecycle")
            return {"active": [], "loaded": [], "reused": 0, "loaded_now": 0}

        resource_ids = [lora.resource_id for lora in loras]
        duplicate_ids = sorted(
            resource_id for resource_id in set(resource_ids) if resource_ids.count(resource_id) > 1
        )
        if duplicate_ids:
            raise ValueError(
                "The same LoRA resource cannot be selected more than once: "
                + ", ".join(duplicate_ids)
            )
        if len(resource_ids) > self.max_loaded:
            raise ValueError(
                f"This runtime allows at most {self.max_loaded} simultaneously active LoRAs"
            )

        desired: list[tuple[LoraExecution, str, str]] = []
        for lora in loras:
            desired.append((lora, _adapter_name(lora.resource_id), _lora_file_signature(lora)))
        desired_names = {adapter_name for _lora, adapter_name, _signature in desired}

        stale_names = [
            adapter_name
            for _lora, adapter_name, signature in desired
            if (existing := self._loaded.get(adapter_name)) is not None
            and existing.signature != signature
        ]
        missing_names = [
            adapter_name
            for _lora, adapter_name, signature in desired
            if (existing := self._loaded.get(adapter_name)) is None
            or existing.signature != signature
        ]
        inactive_names = [name for name in self._loaded if name not in desired_names]
        required_deletions = len(stale_names) + max(
            0,
            len(self._loaded) - len(stale_names) + len(missing_names) - self.max_loaded,
        )
        if required_deletions and not callable(getattr(pipeline, "delete_adapters", None)):
            raise RuntimeError(
                "The pipeline cannot enforce the LoRA memory bound because it does not "
                "implement delete_adapters"
            )

        for adapter_name in stale_names:
            self._delete(pipeline, adapter_name)
        for adapter_name in inactive_names:
            if len(self._loaded) + len(missing_names) <= self.max_loaded:
                break
            self._delete(pipeline, adapter_name)

        active_names: list[str] = []
        weights: list[float] = []
        reused = 0
        loaded_now = 0
        for lora, adapter_name, signature in desired:
            existing = self._loaded.get(adapter_name)
            if existing is None:
                while len(self._loaded) >= self.max_loaded:
                    victim = next(
                        (name for name in self._loaded if name not in desired_names),
                        None,
                    )
                    if victim is None:
                        raise RuntimeError(
                            "No inactive LoRA can be evicted before loading the requested stack"
                        )
                    self._delete(pipeline, victim)
                pipeline.load_lora_weights(
                    str(lora.path.parent),
                    weight_name=lora.path.name,
                    adapter_name=adapter_name,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self._loaded[adapter_name] = _LoadedLora(
                    adapter_name=adapter_name,
                    resource_id=lora.resource_id,
                    signature=signature,
                )
                loaded_now += 1
            else:
                reused += 1
                self._loaded.move_to_end(adapter_name)
            active_names.append(adapter_name)
            weights.append(float(lora.strength))

        if active_names:
            enable = getattr(pipeline, "enable_lora", None)
            if callable(enable):
                enable()
            pipeline.set_adapters(active_names, adapter_weights=weights)
        else:
            disable = getattr(pipeline, "disable_lora", None)
            if callable(disable) and self._loaded:
                disable()

        return {
            "active": [lora.resource_id for lora in loras],
            "weights": weights,
            "loaded": [entry.resource_id for entry in self._loaded.values()],
            "reused": reused,
            "loaded_now": loaded_now,
        }

    def clear(self, pipeline: Any | None = None) -> None:
        if pipeline is None:
            if self._loaded:
                raise RuntimeError(
                    "A live LoRA lifecycle cannot be cleared without its owning pipeline"
                )
            return
        if self._loaded:
            delete = getattr(pipeline, "delete_adapters", None)
            if not callable(delete):
                raise RuntimeError("The pipeline does not implement delete_adapters")
            delete(list(self._loaded))
            self._loaded.clear()

    def reset(self) -> None:
        # Safe only when the owning pipeline object is being discarded as a whole.
        self._loaded.clear()

    def status(self) -> dict[str, Any]:
        return {
            "loaded": [entry.resource_id for entry in self._loaded.values()],
            "max_loaded": self.max_loaded,
        }

    def _delete(self, pipeline: Any, adapter_name: str) -> None:
        delete = getattr(pipeline, "delete_adapters", None)
        if not callable(delete):
            raise TypeError(
                "The pipeline cannot release a LoRA adapter because delete_adapters is absent"
            )
        delete(adapter_name)
        self._loaded.pop(adapter_name)


def is_cuda_oom(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        type_name = type(current).__name__.lower()
        module_name = type(current).__module__.lower()
        message = str(current).lower()
        if (
            "cuda out of memory" in message
            or "cublas_status_alloc_failed" in message
            or ("outofmemoryerror" in type_name and module_name.startswith("torch"))
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def cleanup_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect: Callable[[], None] | None = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:  # noqa: BLE001 - cleanup must never mask the original failure
        return
