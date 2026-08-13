from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID

from ..config import Settings
from ..lora import ConfiguredLora, active_loras
from ..protocol import ToolDescriptor
from ..resources import ResourceDescriptor
from ..storage import Storage, StoredArtifact

ProgressCallback = Callable[[float, str | None], None]


@dataclass(frozen=True, slots=True)
class LoraExecution:
    slot: str
    resource_id: str
    path: Path
    strength: float
    expected_sha256: str | None = None
    expected_schema_sha256: str | None = None
    expected_architecture: str | None = None
    expected_rank: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    variant_key: str
    family: str
    model_resource_id: str | None = None
    model_path: Path | None = None
    model_format: str | None = None
    model_precision: str | None = None
    model_quantization: str | None = None
    loras: tuple[LoraExecution, ...] = ()
    configured_loras: tuple[ConfiguredLora, ...] = ()
    optimizations: dict[str, Any] | None = None
    runtime_parameters: dict[str, Any] | None = None
    recipe: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "loras", active_loras(self.loras))


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Resolved variant requirements used before a runtime is allowed to advertise a tool."""

    family: str
    model_override: bool = False
    model_formats: frozenset[str] = frozenset()
    loras: bool = False
    lora_formats: frozenset[str] = frozenset()
    optimizations: dict[str, Any] = field(default_factory=dict)
    runtime_parameters: bool = False
    recipe_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """Exact execution modes a family adapter can actually honor.

    Empty sets are intentionally conservative. An adapter must opt into each concrete
    mode; advertising INT8 never implies NVFP4/GGUF, native attention never implies
    Sage/Flash, and accepting one recipe type never implies arbitrary component graphs.
    Adapters may additionally override ``validate_execution_request`` for combination
    rules.
    """

    model_formats: frozenset[str] = frozenset()
    lora_formats: frozenset[str] = frozenset()
    recipe_types: frozenset[str] = frozenset()
    attention_modes: frozenset[str] = frozenset()
    offload_modes: frozenset[str] = frozenset()
    quantization_modes: frozenset[str] = frozenset()
    compile_modes: frozenset[str] = frozenset()
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    vae_tiling_modes: frozenset[str] = frozenset()
    vae_slicing_modes: frozenset[str] = frozenset()
    cache_modes: frozenset[str] = frozenset()
    load_policy: bool = False
    residency_policy: bool = False
    runtime_parameters: bool = False

    def validate(self, request: ExecutionRequest) -> list[str]:
        errors: list[str] = []
        if request.recipe_type is not None and request.recipe_type not in self.recipe_types:
            errors.append(f"recipe type {request.recipe_type!r} is not supported by this runtime")
        if request.model_override:
            if not self.model_formats:
                errors.append("model overrides are not supported by this runtime")
            else:
                unsupported = sorted(request.model_formats - self.model_formats)
                if unsupported:
                    errors.append(
                        "model override formats are not supported by this runtime: "
                        + ", ".join(unsupported)
                    )
        if request.loras:
            if not self.lora_formats:
                errors.append("LoRAs are not supported by this runtime")
            else:
                unsupported = sorted(request.lora_formats - self.lora_formats)
                if unsupported:
                    errors.append(
                        "LoRA formats are not supported by this runtime: " + ", ".join(unsupported)
                    )

        optimizations = request.optimizations
        self._validate_mode(
            errors,
            "attention",
            str(optimizations.get("attention", "inherit")),
            self.attention_modes,
        )
        self._validate_mode(
            errors,
            "offload",
            str(optimizations.get("offload", "inherit")),
            self.offload_modes,
        )
        self._validate_mode(
            errors,
            "quantization",
            str(optimizations.get("quantization", "inherit")),
            self.quantization_modes,
        )
        self._validate_mode(
            errors,
            "VAE tiling",
            str(optimizations.get("vae_tiling", "inherit")),
            self.vae_tiling_modes,
        )
        self._validate_mode(
            errors,
            "VAE slicing",
            str(optimizations.get("vae_slicing", "inherit")),
            self.vae_slicing_modes,
        )
        self._validate_mode(
            errors,
            "cache",
            str(optimizations.get("cache", "inherit")),
            self.cache_modes,
        )

        if bool(optimizations.get("compile", False)):
            mode = str(optimizations.get("compile_mode", "default"))
            if mode not in self.compile_modes:
                errors.append(f"compile mode {mode!r} is not supported by this runtime")
            if bool(optimizations.get("compile_fullgraph", False)) and not self.compile_fullgraph:
                errors.append("compile_fullgraph is not supported by this runtime")
            if bool(optimizations.get("compile_dynamic", False)) and not self.compile_dynamic:
                errors.append("compile_dynamic is not supported by this runtime")
        if not bool(optimizations.get("low_cpu_mem_usage", True)) and not self.load_policy:
            errors.append("low_cpu_mem_usage overrides are not supported by this runtime")
        if not bool(optimizations.get("keep_pipeline_loaded", True)) and not self.residency_policy:
            errors.append("pipeline residency overrides are not supported by this runtime")
        if request.runtime_parameters and not self.runtime_parameters:
            errors.append("extra runtime parameters are not supported by this runtime")
        return errors

    @staticmethod
    def _validate_mode(
        errors: list[str],
        label: str,
        mode: str,
        supported: frozenset[str],
    ) -> None:
        if mode != "inherit" and mode not in supported:
            errors.append(f"{label} mode {mode!r} is not supported by this runtime")


@dataclass(slots=True)
class ToolContext:
    job_id: UUID
    settings: Settings
    storage: Storage
    cancel_event: Event
    progress: ProgressCallback
    execution: ExecutionPlan | None = None
    runtime_provenance: dict[str, Any] = field(default_factory=dict)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise ToolCancelled("Generation canceled")

    def resolve_asset(self, asset_id: UUID) -> Path:
        return self.storage.resolve_asset(asset_id)

    def with_execution(self, execution: ExecutionPlan) -> ToolContext:
        # dataclasses.replace keeps the same provenance dictionary so the variant
        # wrapper and the underlying curated tool contribute to one job record.
        return replace(self, execution=execution)

    def record_provenance(self, **values: Any) -> None:
        self.runtime_provenance.update(values)


class ToolCancelled(RuntimeError):
    pass


class Tool(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> ToolDescriptor:
        raise NotImplementedError

    @abstractmethod
    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {}

    def model_family(self) -> str | None:
        """Return the exact Engine resource family this curated tool executes."""

        return None

    def variant_base_availability(self) -> tuple[bool, str | None]:
        """Return whether variants may build on this tool's runtime implementation."""

        descriptor = self.descriptor
        return descriptor.available, descriptor.unavailable_reason

    def variant_recipe_availability(
        self,
        recipe_type: str | None,
    ) -> tuple[bool, str | None]:
        """Return availability for one typed recipe execution path.

        Most tools have one runtime and delegate to the existing base check.
        Family adapters with distinct Reference and optimized runtimes override
        this so one path cannot inherit another path's dependencies.
        """

        del recipe_type
        return self.variant_base_availability()

    def variant_provenance(self, recipe_type: str | None) -> dict[str, Any]:
        """Return recipe-aware public runtime provenance for a variant."""

        del recipe_type
        return self.provenance()

    def variant_requirements(self, recipe_type: str | None):
        """Return bundle requirements appropriate to one variant execution path."""

        del recipe_type
        return [requirement.model_copy(deep=True) for requirement in self.descriptor.requirements]

    def execution_capabilities(self) -> ExecutionCapabilities:
        """Return exact runtime modes supported by data-defined variants."""

        return ExecutionCapabilities()

    def model_resource_components(self) -> frozenset[str]:
        """Component roles this tool can promote to a complete executable model.

        Components normally require an explicit recipe. A family adapter may opt
        into exact roles only when it supplies and binds the missing pipeline shell.
        """

        return frozenset()

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        """Validate exact modes and combinations before advertising a variant tool."""

        errors: list[str] = []
        family = self.model_family()
        if family is None:
            errors.append("base runtime does not declare a model family")
        elif request.family != family:
            errors.append(
                f"execution family {request.family!r} does not match base runtime family {family!r}"
            )
        errors.extend(self.execution_capabilities().validate(request))
        return errors

    def validate_model_resource(
        self,
        resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        """Validate a concrete selected resource during catalog compilation.

        Generic tools accept metadata-compatible resources. Family adapters with
        stricter executable layouts override this hook so broken resources never
        appear as selectable options.
        """

        return []

    def validate_lora_resource(
        self,
        resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        """Validate a selected LoRA before it is exposed by a variant."""

        return []
