from __future__ import annotations

import copy
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Settings
from .klein_recipe import (
    Klein4ComfyRecipe,
    Klein4RecipeComponent,
    build_klein4_comfy_runtime_request,
    validate_klein4_comfy_recipe,
)
from .model_store import MODEL_FAMILIES
from .protocol import ChoiceOption, InputType, InputUi, ToolDescriptor, ToolInput
from .resources import (
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceInventory,
    ResourceKind,
)
from .tools.base import (
    ExecutionCapabilities,
    ExecutionPlan,
    ExecutionRequest,
    LoraExecution,
    Tool,
    ToolContext,
)
from .wan22_recipe import (
    Wan22I2VRecipe,
    Wan22RecipeComponent,
    build_native_wan22_i2v_14b_runtime_request,
    validate_native_wan22_i2v_14b_recipe,
)

VARIANT_NAMESPACE = UUID("27b92258-6010-4d2f-8761-d19ab94a8f79")
_PARAMETER_PATTERN = r"^[a-z][a-z0-9_]*$"
_SLOT_PATTERN = r"^[a-z][a-z0-9_.-]*$"

AttentionMode = Literal[
    "inherit",
    "auto",
    "native",
    "flash",
    "flash_hub",
    "flash3_hub",
    "flash4_hub",
    "sage",
    "sage_hub",
    "xformers",
    "sol",
]
OffloadMode = Literal[
    "inherit",
    "auto",
    "none",
    "model",
    "sequential",
    "group_block",
    "group_leaf",
    "staged",
]
QuantizationMode = Literal[
    "inherit",
    "native",
    "bf16",
    "fp16",
    "fp8",
    "int8",
    "nvfp4",
    "gguf",
]
ToggleMode = Literal["inherit", "auto", "on", "off"]
CacheMode = Literal["inherit", "none", "prompt", "media", "first_block", "tea", "easy"]


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attention: AttentionMode = "inherit"
    offload: OffloadMode = "inherit"
    quantization: QuantizationMode = "inherit"
    compile: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    vae_tiling: ToggleMode = "inherit"
    vae_slicing: ToggleMode = "inherit"
    cache: CacheMode = "inherit"
    group_offload_blocks: int | None = Field(default=None, ge=1)
    group_offload_use_stream: bool = False
    group_offload_record_stream: bool = False
    low_cpu_mem_usage: bool = True
    keep_pipeline_loaded: bool = True

    @model_validator(mode="after")
    def validate_dependencies(self) -> OptimizationConfig:
        compile_options = {"compile_mode", "compile_fullgraph", "compile_dynamic"}
        if (
            not self.compile
            and self.model_fields_set.intersection(compile_options)
            and (self.compile_mode != "default" or self.compile_fullgraph or self.compile_dynamic)
        ):
            raise ValueError("compile options require optimizations.compile = true")

        group_options = {
            "group_offload_blocks",
            "group_offload_use_stream",
            "group_offload_record_stream",
        }
        if self.model_fields_set.intersection(group_options) and self.offload not in {
            "group_block",
            "group_leaf",
        }:
            raise ValueError("group offload options require a group offload mode")
        if self.group_offload_blocks is not None and self.offload != "group_block":
            raise ValueError("group_offload_blocks requires offload = 'group_block'")
        return self

    def requested_features(self) -> set[str]:
        requested: set[str] = set()
        if self.attention != "inherit":
            requested.add("attention_backend")
        if self.offload != "inherit":
            requested.add("offload_override")
        if self.quantization != "inherit":
            requested.add("quantization")
        if self.compile:
            requested.add("compile")
        if self.vae_tiling != "inherit" or self.vae_slicing != "inherit":
            requested.add("vae_options")
        if self.cache in {"none", "prompt", "media"}:
            requested.add("cache_policy")
        elif self.cache in {"first_block", "tea", "easy"}:
            requested.add("cross_step_cache")
        if not self.low_cpu_mem_usage:
            requested.add("load_policy")
        if not self.keep_pipeline_loaded:
            requested.add("residency_policy")
        return requested


class VariantModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource: str | None = None
    exposed: bool = False
    parameter_key: str = Field(default="model", pattern=_PARAMETER_PATTERN)
    label: str = Field(default="Model", min_length=1)
    allowed: list[str] = Field(default_factory=list)
    default: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> VariantModelConfig:
        if not self.resource and not self.exposed:
            raise ValueError("model must declare a fixed resource or an exposed selector")
        if not self.exposed and (self.default is not None or self.allowed):
            raise ValueError("model default/allowed constraints require exposed = true")
        return self


class Wan22I2VRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["wan22_i2v_14b"] = "wan22_i2v_14b"
    base_model: str = Field(min_length=1)
    pipeline_support: str = Field(min_length=1)
    transformer_high_noise: str = Field(min_length=1)
    transformer_low_noise: str = Field(min_length=1)
    text_encoder: str = Field(min_length=1)
    vae: str = Field(min_length=1)

    def resource_references(self) -> dict[str, str]:
        return {
            "pipeline_support": self.pipeline_support,
            "transformer_high_noise": self.transformer_high_noise,
            "transformer_low_noise": self.transformer_low_noise,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
        }


class Klein4ComfyRecipeConfig(BaseModel):
    """One exact Comfy Klein component set and immutable inference schedule."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["klein4_comfy"] = "klein4_comfy"
    mode: Literal["base", "distilled"]
    base_model: str = Field(min_length=1)
    steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0, allow_inf_nan=False)
    pipeline_support: str = Field(min_length=1)
    transformer: str = Field(min_length=1)
    text_encoder: str = Field(min_length=1)
    vae: str = Field(min_length=1)

    def resource_references(self) -> dict[str, str]:
        return {
            "pipeline_support": self.pipeline_support,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
        }


class VariantLoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str = Field(pattern=_SLOT_PATTERN)
    resource: str | None = None
    exposed: bool = False
    parameter_key: str | None = Field(default=None, pattern=_PARAMETER_PATTERN)
    label: str | None = None
    allowed: list[str] = Field(default_factory=list)
    default: str | None = None
    required: bool = False
    strength: float = 1.0
    strength_exposed: bool = False
    strength_key: str | None = Field(default=None, pattern=_PARAMETER_PATTERN)
    strength_label: str | None = None
    strength_min: float = -2.0
    strength_max: float = 2.0
    strength_step: float = 0.05

    @model_validator(mode="after")
    def validate_slot(self) -> VariantLoraConfig:
        for name, value in (
            ("strength", self.strength),
            ("strength_min", self.strength_min),
            ("strength_max", self.strength_max),
            ("strength_step", self.strength_step),
        ):
            if not math.isfinite(value):
                raise ValueError(f"LoRA {name} must be finite")
        if self.strength_min > self.strength_max:
            raise ValueError("LoRA strength_min cannot exceed strength_max")
        if self.strength_step <= 0:
            raise ValueError("LoRA strength_step must be positive")
        if not self.strength_min <= self.strength <= self.strength_max:
            raise ValueError("LoRA strength must fall within strength_min/strength_max")
        if not self.resource and not self.exposed:
            raise ValueError("LoRA must declare a fixed resource or an exposed selector")
        if not self.exposed and (
            self.parameter_key is not None
            or self.label is not None
            or self.default is not None
            or self.allowed
        ):
            raise ValueError("LoRA selector fields require exposed = true")
        if not self.strength_exposed and (
            self.strength_key is not None or self.strength_label is not None
        ):
            raise ValueError("LoRA strength fields require strength_exposed = true")
        if self.required and self.default == "none":
            raise ValueError("a required LoRA selector cannot default to 'none'")
        return self


class VariantInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    label: str | None = None
    description: str | None = None
    required: bool | None = None
    default: Any = None
    options: list[str] | None = None
    ui: InputUi | None = None


class VariantDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: UUID | None = None
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1)
    description: str | None = None
    enabled: bool = True
    family: str
    base_tool: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    tags: list[str] = Field(default_factory=list)
    model: VariantModelConfig | None = None
    recipe: Wan22I2VRecipeConfig | Klein4ComfyRecipeConfig | None = None
    inputs: dict[str, VariantInputConfig] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    loras: list[VariantLoraConfig] = Field(default_factory=list)
    optimizations: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @model_validator(mode="after")
    def validate_family_and_slots(self) -> VariantDefinition:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(f"unknown model family {self.family!r}")
        slots = [lora.slot for lora in self.loras]
        if len(slots) != len(set(slots)):
            raise ValueError("LoRA slot names must be unique")
        if self.model is not None and self.recipe is not None:
            raise ValueError("variant cannot declare both model and recipe")
        if isinstance(self.recipe, Wan22I2VRecipeConfig) and self.family != "wan22":
            raise ValueError("wan22_i2v_14b recipes require family = 'wan22'")
        if isinstance(self.recipe, Klein4ComfyRecipeConfig) and self.family != "klein4b":
            raise ValueError("klein4_comfy recipes require family = 'klein4b'")
        if self.optimizations.quantization == "gguf" and (
            self.model is None or self.model.resource is None or self.model.exposed
        ):
            raise ValueError("quantization 'gguf' requires one fixed GGUF model resource")
        return self


class VariantCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    key: str
    name: str
    family: str
    base_tool: str
    source_path: str
    enabled: bool
    available: bool
    unavailable_reason: str | None = None
    # Human presentation metadata. Keep the established structured catalog payload
    # stable; recipe files remain the authoritative source for the full tag set.
    tags: list[str] = Field(default_factory=list, exclude=True)
    model_resource: str | None = None
    lora_slots: list[str] = Field(default_factory=list)
    recipe_type: str | None = None
    recipe_resources: dict[str, str] = Field(default_factory=dict)
    optimizations: dict[str, Any] = Field(default_factory=dict)
    fixed_resources: list[str] = Field(default_factory=list)
    dynamic_resource_slots: list[str] = Field(default_factory=list)


class VariantCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variants: list[VariantCatalogEntry]
    errors: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class VariantLoadResult:
    tools: list[Tool]
    entries: list[VariantCatalogEntry]
    errors: list[str]


class VariantTool(Tool):
    def __init__(
        self,
        *,
        definition: VariantDefinition,
        source_path: Path,
        base_tool: Tool,
        inventory: ResourceInventory,
    ) -> None:
        self.definition = definition
        self.source_path = source_path
        self.base_tool = base_tool
        self.inventory = inventory
        self._variant_id = definition.id or uuid5(VARIANT_NAMESPACE, definition.key)
        self._input_bindings: dict[str, str] = {}
        self._model_input_key: str | None = None
        self._lora_input_keys: dict[str, str] = {}
        self._lora_strength_keys: dict[str, str] = {}
        self._descriptor = self._compile_descriptor()

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def model_family(self) -> str | None:
        return self.base_tool.model_family()

    def execution_capabilities(self) -> ExecutionCapabilities:
        return self.base_tool.execution_capabilities()

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        return self.base_tool.validate_execution_request(request)

    def run(self, context: ToolContext, inputs: dict[str, Any]):
        base_input_keys = {descriptor.key for descriptor in self.base_tool.descriptor.inputs}
        base_inputs = {
            key: copy.deepcopy(value)
            for key, value in self.definition.fixed.items()
            if key in base_input_keys
        }
        runtime_parameters = {
            key: copy.deepcopy(value)
            for key, value in self.definition.fixed.items()
            if key not in base_input_keys
        }
        for variant_key, base_key in self._input_bindings.items():
            if variant_key in inputs:
                base_inputs[base_key] = inputs[variant_key]

        model_resource = self._resolve_selected_model(inputs)
        loras = self._resolve_selected_loras(inputs)
        recipe_request = self._resolve_recipe_request()
        optimizations = self.definition.optimizations.model_dump(mode="json")
        if model_resource is not None:
            resolved_quantization = _resolve_resource_quantization(
                model_resource,
                self.definition.optimizations.quantization,
                self.base_tool.execution_capabilities(),
            )
            if resolved_quantization is None:
                raise ValueError(_inherit_resolution_error(model_resource))
            optimizations["quantization"] = resolved_quantization
        plan = ExecutionPlan(
            variant_key=self.definition.key,
            family=self.definition.family,
            model_resource_id=model_resource.id if model_resource else None,
            model_path=self.inventory.path_for(model_resource.id) if model_resource else None,
            model_format=model_resource.format.value if model_resource else None,
            model_precision=model_resource.precision.value if model_resource else None,
            model_quantization=(model_resource.quantization.value if model_resource else None),
            loras=tuple(loras),
            optimizations=optimizations,
            runtime_parameters=runtime_parameters,
            recipe=recipe_request,
        )
        if model_resource is not None and (
            error := _quantization_compatibility_error(
                model_resource,
                optimizations["quantization"],
            )
        ):
            raise ValueError(error)
        return self.base_tool.run(context.with_execution(plan), base_inputs)

    def provenance(self) -> dict[str, Any]:
        return {
            **self.base_tool.provenance(),
            "variant_key": self.definition.key,
            "variant_source": self.source_path.as_posix(),
            "variant_family": self.definition.family,
        }

    def catalog_entry(self) -> VariantCatalogEntry:
        return _catalog_entry(
            self.definition,
            self.source_path,
            variant_id=self._variant_id,
            available=self.descriptor.available,
            unavailable_reason=self.descriptor.unavailable_reason,
        )

    def _compile_descriptor(self) -> ToolDescriptor:
        base = self.base_tool.descriptor
        self._validate_fixed_base_inputs(base)
        inputs = self._compile_inputs(base)
        unavailable: list[str] = []
        base_available, base_unavailable_reason = self.base_tool.variant_base_availability()
        if not base_available:
            unavailable.append(base_unavailable_reason or "base tool is unavailable")

        unavailable.extend(self.base_tool.validate_execution_request(self._execution_request(base)))
        unavailable.extend(self._validate_resources())
        unavailable.extend(self._validate_recipe())
        required_base = {
            descriptor.key
            for descriptor in base.inputs
            if descriptor.required and descriptor.default is None
        }
        supplied_base = set(self._input_bindings.values()) | set(self.definition.fixed)
        missing = sorted(required_base - supplied_base)
        if missing:
            unavailable.append("variant omits required base inputs: " + ", ".join(missing))

        reason = "; ".join(unavailable) or None
        return ToolDescriptor(
            id=self._variant_id,
            key=self.definition.key,
            schema_revision=self.definition.schema_revision,
            name=self.definition.name,
            description=self.definition.description or base.description,
            workflow_kind=base.workflow_kind,
            output=base.output.model_copy(deep=True),
            inputs=inputs,
            requirements=[requirement.model_copy(deep=True) for requirement in base.requirements],
            available=reason is None,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _execution_request(self, base: ToolDescriptor) -> ExecutionRequest:
        model_formats: set[str] = set()
        if self.definition.model is not None:
            if self.definition.model.exposed:
                model_formats.update(
                    resource.format.value for resource in self._matching_model_resources()
                )
            elif self.definition.model.resource:
                try:
                    model_formats.add(
                        self._resolve_resource_reference(
                            self.definition.model.resource,
                            kind=ResourceKind.MODEL,
                        ).format.value
                    )
                except (KeyError, ValueError):
                    pass

        lora_formats: set[str] = set()
        for lora in self.definition.loras:
            if lora.exposed:
                lora_formats.update(
                    resource.format.value
                    for resource in self._matching_resources(ResourceKind.LORA, lora.allowed)
                )
            if lora.resource:
                try:
                    lora_formats.add(
                        self._resolve_resource_reference(
                            lora.resource,
                            kind=ResourceKind.LORA,
                        ).format.value
                    )
                except (KeyError, ValueError):
                    pass

        runtime_fixed = set(self.definition.fixed) - {descriptor.key for descriptor in base.inputs}
        return ExecutionRequest(
            family=self.definition.family,
            model_override=self.definition.model is not None,
            model_formats=frozenset(model_formats),
            loras=bool(self.definition.loras),
            lora_formats=frozenset(lora_formats),
            optimizations=self.definition.optimizations.model_dump(mode="json"),
            runtime_parameters=bool(runtime_fixed),
            recipe_type=self.definition.recipe.type if self.definition.recipe else None,
        )

    def _validate_fixed_base_inputs(self, base: ToolDescriptor) -> None:
        base_by_key = {descriptor.key: descriptor for descriptor in base.inputs}
        for key, value in self.definition.fixed.items():
            descriptor = base_by_key.get(key)
            if descriptor is not None:
                _validate_fixed_value(descriptor, value)

    def _compile_inputs(self, base: ToolDescriptor) -> list[ToolInput]:
        base_by_key = {descriptor.key: descriptor for descriptor in base.inputs}
        compiled: list[ToolInput] = []
        configured_inputs = self.definition.inputs
        if not configured_inputs:
            configured_inputs = {
                descriptor.key: VariantInputConfig(source=descriptor.key)
                for descriptor in base.inputs
                if descriptor.key not in self.definition.fixed
            }

        bound_sources: dict[str, str] = {}
        for variant_key, config in configured_inputs.items():
            source = (config.source or variant_key).removeprefix("tool.")
            try:
                base_input = base_by_key[source]
            except KeyError as exc:
                raise ValueError(
                    f"Variant {self.definition.key!r} input {variant_key!r} references "
                    f"unknown base input {source!r}"
                ) from exc
            if source in self.definition.fixed:
                raise ValueError(f"base input {source!r} cannot be both fixed and exposed")
            if source in bound_sources:
                raise ValueError(
                    f"base input {source!r} is exposed by both "
                    f"{bound_sources[source]!r} and {variant_key!r}"
                )

            update: dict[str, Any] = {"key": variant_key}
            for field_name in ("label", "description", "required", "ui"):
                value = getattr(config, field_name)
                if value is not None:
                    update[field_name] = value
            if "default" in config.model_fields_set:
                update["default"] = copy.deepcopy(config.default)
            if config.options is not None:
                if base_input.type != InputType.CHOICE:
                    raise ValueError(
                        f"Variant input {variant_key!r} can only override options on a choice"
                    )
                update["options"] = [
                    ChoiceOption(value=value, label=value) for value in config.options
                ]
            variant_payload = base_input.model_dump(mode="python")
            variant_payload.update(update)
            variant_input = ToolInput.model_validate(variant_payload)
            if (
                base_input.required
                and base_input.default is None
                and not variant_input.required
                and variant_input.default is None
            ):
                raise ValueError(
                    f"required base input {source!r} cannot become optional without a default"
                )
            compiled.append(variant_input)
            self._input_bindings[variant_key] = source
            bound_sources[source] = variant_key

        if self.definition.model and self.definition.model.exposed:
            resources = self._matching_model_resources()
            options = [ChoiceOption(value=item.id, label=item.name) for item in resources]
            if not options:
                options = [ChoiceOption(value="unavailable", label="No compatible models found")]
            default = self.definition.model.default or self.definition.model.resource
            if default:
                selected = self._resolve_resource_reference(default, kind=ResourceKind.MODEL)
                if selected.id not in {resource.id for resource in resources}:
                    raise ValueError(
                        f"model default {default!r} is excluded by the allowed resource patterns"
                    )
                default = selected.id
            elif resources:
                default = resources[0].id
            self._model_input_key = self.definition.model.parameter_key
            compiled.append(
                ToolInput(
                    key=self._model_input_key,
                    label=self.definition.model.label,
                    type=InputType.CHOICE,
                    required=True,
                    default=default or options[0].value,
                    options=options,
                    ui=InputUi(group="Model", advanced=True),
                )
            )

        for lora in self.definition.loras:
            if lora.exposed:
                resources = self._matching_resources(ResourceKind.LORA, lora.allowed)
                options = [ChoiceOption(value=item.id, label=item.name) for item in resources]
                if lora.required:
                    if not options:
                        options = [
                            ChoiceOption(value="unavailable", label="No compatible LoRAs found")
                        ]
                else:
                    options.insert(0, ChoiceOption(value="none", label="None"))

                default = lora.default or lora.resource
                if default is None:
                    if lora.required:
                        default = resources[0].id if resources else "unavailable"
                    else:
                        default = "none"
                if default not in {"none", "unavailable"}:
                    selected = self._resolve_resource_reference(default, kind=ResourceKind.LORA)
                    if selected.id not in {resource.id for resource in resources}:
                        raise ValueError(
                            f"LoRA default {default!r} in slot {lora.slot!r} is excluded "
                            "by the allowed resource patterns"
                        )
                    default = selected.id
                if lora.required and default == "none":
                    raise ValueError(f"required LoRA slot {lora.slot!r} cannot default to none")

                key = lora.parameter_key or f"{lora.slot}_lora"
                self._lora_input_keys[lora.slot] = key
                compiled.append(
                    ToolInput(
                        key=key,
                        label=lora.label or f"{lora.slot.replace('_', ' ').title()} LoRA",
                        type=InputType.CHOICE,
                        required=lora.required,
                        default=default,
                        options=options,
                        ui=InputUi(group="LoRAs", advanced=True),
                    )
                )
            if lora.strength_exposed:
                key = lora.strength_key or f"{lora.slot}_strength"
                self._lora_strength_keys[lora.slot] = key
                compiled.append(
                    ToolInput(
                        key=key,
                        label=lora.strength_label
                        or f"{lora.slot.replace('_', ' ').title()} Strength",
                        type=InputType.NUMBER,
                        required=True,
                        default=lora.strength,
                        ui=InputUi(
                            group="LoRAs",
                            advanced=True,
                            min=lora.strength_min,
                            max=lora.strength_max,
                            step=lora.strength_step,
                        ),
                    )
                )
        return compiled

    def _validate_resources(self) -> list[str]:
        errors: list[str] = []
        model_resource: ResourceDescriptor | None = None
        if self.definition.model and self.definition.model.resource:
            try:
                model_resource = self._resolve_resource_reference(
                    self.definition.model.resource,
                    kind=ResourceKind.MODEL,
                )
            except Exception as exc:  # noqa: BLE001 - catalog should explain bad variants
                errors.append(f"model resource: {exc}")
        if (
            self.definition.model
            and self.definition.model.exposed
            and not self._matching_model_resources()
        ):
            errors.append("no compatible model resources were discovered")

        quantization = self.definition.optimizations.quantization
        if model_resource is not None:
            if not model_resource.available:
                errors.append(
                    f"model resource {model_resource.id!r}: "
                    f"{model_resource.unavailable_reason or 'is not installed'}"
                )
            else:
                resolved = _resolve_resource_quantization(
                    model_resource,
                    quantization,
                    self.base_tool.execution_capabilities(),
                )
                if resolved is None:
                    errors.append(_inherit_resolution_error(model_resource))
                elif error := _quantization_compatibility_error(model_resource, resolved):
                    errors.append(error)
                else:
                    errors.extend(self._model_resource_errors(model_resource))

        for lora in self.definition.loras:
            if lora.resource:
                try:
                    self._resolve_resource_reference(lora.resource, kind=ResourceKind.LORA)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"LoRA slot {lora.slot}: {exc}")
            if (
                lora.exposed
                and lora.required
                and not self._matching_resources(ResourceKind.LORA, lora.allowed)
            ):
                errors.append(f"required LoRA slot {lora.slot!r} has no compatible resources")
        return errors

    def _validate_recipe(self) -> list[str]:
        if self.definition.recipe is None:
            return []
        try:
            recipe = self._resolve_recipe_definition()
            if isinstance(recipe, Klein4ComfyRecipe):
                validation = validate_klein4_comfy_recipe(
                    recipe,
                    self.inventory,
                    include_adapter_plans=False,
                )
            else:
                validation = validate_native_wan22_i2v_14b_recipe(
                    recipe,
                    self.inventory,
                    include_adapter_plans=False,
                )
        except Exception as exc:  # noqa: BLE001 - catalog must explain recipe failures
            return [f"recipe: {exc}"]
        return [f"recipe: {error}" for error in validation.errors]

    def _resolve_recipe_request(self):
        if self.definition.recipe is None:
            return None
        recipe = self._resolve_recipe_definition()
        if isinstance(recipe, Klein4ComfyRecipe):
            return build_klein4_comfy_runtime_request(recipe, self.inventory)
        return build_native_wan22_i2v_14b_runtime_request(recipe, self.inventory)

    def _resolve_recipe_definition(self) -> Wan22I2VRecipe | Klein4ComfyRecipe:
        config = self.definition.recipe
        if config is None:
            raise ValueError("variant does not declare a recipe")

        def resource_component(reference: str) -> ResourceDescriptor:
            resource = self._resolve_resource_reference(
                reference,
                kind=ResourceKind.MODEL,
                include_components=True,
            )
            return resource

        if isinstance(config, Klein4ComfyRecipeConfig):
            def klein_component(reference: str) -> Klein4RecipeComponent:
                resource = resource_component(reference)
                return Klein4RecipeComponent(resource, self.inventory.path_for(resource.id))

            return Klein4ComfyRecipe(
                mode=config.mode,
                base_model=config.base_model,
                steps=config.steps,
                guidance_scale=config.guidance_scale,
                pipeline_support=klein_component(config.pipeline_support),
                transformer=klein_component(config.transformer),
                text_encoder=klein_component(config.text_encoder),
                vae=klein_component(config.vae),
            )

        def wan_component(reference: str) -> Wan22RecipeComponent:
            resource = resource_component(reference)
            return Wan22RecipeComponent(resource, self.inventory.path_for(resource.id))

        return Wan22I2VRecipe(
            base_model=config.base_model,
            high_noise=wan_component(config.transformer_high_noise),
            low_noise=wan_component(config.transformer_low_noise),
            text_encoder=wan_component(config.text_encoder),
            vae=wan_component(config.vae),
            pipeline_support=wan_component(config.pipeline_support),
        )

    def _matching_model_resources(self) -> list[ResourceDescriptor]:
        if self.definition.model is None:
            return []
        compatible: list[ResourceDescriptor] = []
        capabilities = self.base_tool.execution_capabilities()
        for resource in self._matching_resources(
            ResourceKind.MODEL,
            self.definition.model.allowed,
        ):
            if resource.format.value == "gguf":
                continue
            resolved = _resolve_resource_quantization(
                resource,
                self.definition.optimizations.quantization,
                capabilities,
            )
            if (
                resolved is not None
                and _quantization_compatibility_error(resource, resolved) is None
                and not self._model_resource_errors(resource)
            ):
                compatible.append(resource)
        return compatible

    def _model_resource_errors(self, resource: ResourceDescriptor) -> list[str]:
        try:
            path = self.inventory.path_for(resource.id)
            return [
                f"model resource {resource.id!r}: {error}"
                for error in self.base_tool.validate_model_resource(resource, path)
            ]
        except Exception as exc:  # noqa: BLE001 - catalog must explain invalid resources
            return [f"model resource {resource.id!r}: {exc}"]

    def _matching_resources(
        self,
        kind: ResourceKind,
        allow: list[str],
    ) -> list[ResourceDescriptor]:
        allowed_components = self.base_tool.model_resource_components()
        resources = self.inventory.matching(
            kind=kind,
            family=self.definition.family,
            allow=allow or None,
            include_components=(kind == ResourceKind.MODEL and bool(allowed_components)),
        )
        if kind == ResourceKind.MODEL:
            resources = [
                resource
                for resource in resources
                if resource.component is None or resource.component in allowed_components
            ]
        return resources

    def _resolve_resource_reference(
        self,
        reference: str,
        *,
        kind: ResourceKind,
        include_components: bool = False,
    ) -> ResourceDescriptor:
        allowed_components = self.base_tool.model_resource_components()
        resource = self.inventory.resolve(
            reference,
            kind=kind,
            family=self.definition.family,
            include_components=(
                include_components or (kind == ResourceKind.MODEL and bool(allowed_components))
            ),
        )
        if (
            kind == ResourceKind.MODEL
            and not include_components
            and resource.component is not None
            and resource.component not in allowed_components
        ):
            raise ValueError(
                f"model component {resource.component!r} requires an explicit family recipe"
            )
        return resource

    def _resolve_selected_model(self, inputs: dict[str, Any]) -> ResourceDescriptor | None:
        config = self.definition.model
        if config is None:
            return None
        reference = (
            inputs.get(self._model_input_key)
            if self._model_input_key is not None
            else config.resource
        )
        if not reference:
            return None
        return self._resolve_resource_reference(str(reference), kind=ResourceKind.MODEL)

    def _resolve_selected_loras(self, inputs: dict[str, Any]) -> list[LoraExecution]:
        selections: list[LoraExecution] = []
        for config in self.definition.loras:
            reference = (
                inputs.get(self._lora_input_keys[config.slot])
                if config.slot in self._lora_input_keys
                else config.resource
            )
            if not reference or reference == "none":
                continue
            resource = self._resolve_resource_reference(str(reference), kind=ResourceKind.LORA)
            strength_key = self._lora_strength_keys.get(config.slot)
            strength = (
                float(inputs.get(strength_key, config.strength))
                if strength_key
                else config.strength
            )
            selections.append(
                LoraExecution(
                    slot=config.slot,
                    resource_id=resource.id,
                    path=self.inventory.path_for(resource.id),
                    strength=strength,
                )
            )
        return selections


def _recipe_files(
    settings: Settings,
    *,
    include_builtin_recipes: bool = True,
):
    seen_paths: set[Path] = set()
    for label, root in settings.recipe_catalog_roots():
        if label == "builtin" and not include_builtin_recipes:
            continue
        if not root.exists():
            continue
        resolved_root = root.resolve()
        for path in sorted(resolved_root.rglob("*.toml")):
            if path.name.startswith("."):
                continue
            resolved_path = path.resolve()
            if resolved_path in seen_paths:
                continue
            try:
                relative = resolved_path.relative_to(resolved_root)
            except ValueError:
                yield (
                    resolved_path,
                    None,
                    f"recipe file must stay within catalog root {resolved_root}",
                )
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            seen_paths.add(resolved_path)
            yield resolved_path, Path(label) / relative, None


def load_variant_tools(
    settings: Settings,
    base_tools: list[Tool],
    inventory: ResourceInventory,
    *,
    include_builtin_recipes: bool = True,
) -> VariantLoadResult:
    by_key = {tool.descriptor.key: tool for tool in base_tools}
    tools: list[Tool] = []
    entries: list[VariantCatalogEntry] = []
    errors: list[str] = []
    seen_ids = {tool.descriptor.id for tool in base_tools}
    seen_keys = set(by_key)

    for resolved_path, source_path, discovery_error in _recipe_files(
        settings,
        include_builtin_recipes=include_builtin_recipes,
    ):
        if discovery_error is not None:
            errors.append(f"{resolved_path}: {discovery_error}")
            continue
        assert source_path is not None
        try:
            raw = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
            definition_data = raw.get("runnable_recipe", raw.get("variant", raw))
            definition = VariantDefinition.model_validate(definition_data)
            variant_id = definition.id or uuid5(VARIANT_NAMESPACE, definition.key)
            if variant_id in seen_ids:
                raise ValueError(f"duplicate tool UUID {variant_id}")
            if definition.key in seen_keys:
                raise ValueError(f"duplicate tool key {definition.key!r}")
            seen_ids.add(variant_id)
            seen_keys.add(definition.key)

            try:
                base_tool = by_key[definition.base_tool]
            except KeyError as exc:
                raise ValueError(f"unknown base_tool {definition.base_tool!r}") from exc
            base_family = base_tool.model_family()
            if base_family is None:
                raise ValueError(
                    f"base_tool {definition.base_tool!r} does not declare a model family"
                )
            if definition.family != base_family:
                raise ValueError(
                    f"variant family {definition.family!r} does not match base_tool "
                    f"family {base_family!r}"
                )

            if not definition.enabled:
                entries.append(
                    _catalog_entry(
                        definition,
                        source_path,
                        variant_id=variant_id,
                        available=False,
                        unavailable_reason="variant is disabled",
                    )
                )
                continue

            tool = VariantTool(
                definition=definition,
                source_path=source_path,
                base_tool=base_tool,
                inventory=inventory,
            )
            tools.append(tool)
            entries.append(tool.catalog_entry())
        except Exception as exc:  # noqa: BLE001 - collect all iterative authoring errors
            errors.append(f"{resolved_path}: {exc}")

    entries.sort(key=lambda entry: (entry.family, entry.name.casefold(), entry.key))
    return VariantLoadResult(tools=tools, entries=entries, errors=errors)

def _catalog_entry(
    definition: VariantDefinition,
    source_path: Path,
    *,
    variant_id: UUID,
    available: bool,
    unavailable_reason: str | None,
) -> VariantCatalogEntry:
    return VariantCatalogEntry(
        id=variant_id,
        key=definition.key,
        name=definition.name,
        family=definition.family,
        base_tool=definition.base_tool,
        source_path=source_path.as_posix(),
        enabled=definition.enabled,
        available=available,
        unavailable_reason=unavailable_reason,
        tags=list(definition.tags),
        model_resource=definition.model.resource if definition.model else None,
        lora_slots=[lora.slot for lora in definition.loras],
        recipe_type=definition.recipe.type if definition.recipe else None,
        recipe_resources=(definition.recipe.resource_references() if definition.recipe else {}),
        optimizations=definition.optimizations.model_dump(mode="json"),
        fixed_resources=_fixed_resource_references(definition),
        dynamic_resource_slots=_dynamic_resource_slots(definition),
    )


def _fixed_resource_references(definition: VariantDefinition) -> list[str]:
    references: list[str] = []
    if definition.model is not None and definition.model.resource:
        references.append(definition.model.resource)
    if definition.recipe is not None:
        references.extend(definition.recipe.resource_references().values())
    references.extend(
        lora.resource
        for lora in definition.loras
        if lora.resource is not None and lora.resource != "none"
    )
    return list(dict.fromkeys(references))


def _dynamic_resource_slots(definition: VariantDefinition) -> list[str]:
    slots: list[str] = []
    if definition.model is not None and definition.model.exposed:
        slots.append(f"model:{definition.model.parameter_key}")
    slots.extend(
        f"lora:{lora.slot}"
        for lora in definition.loras
        if lora.exposed
    )
    return slots


def _quantization_compatibility_error(
    resource: ResourceDescriptor,
    requested: QuantizationMode,
) -> str | None:
    """Validate a variant's requested artifact against its selected dropped model.

    ``optimizations.quantization`` is deliberately a selection constraint, never a
    conversion instruction. Unknown resources are intentionally excluded whenever a
    concrete artifact type was requested.
    """

    if requested == "inherit":
        return "quantization='inherit' must resolve to a concrete artifact mode"
    actual_quantization = resource.quantization
    actual_precision = resource.precision
    if requested == "gguf":
        expected = ArtifactQuantization.GGUF
        if actual_quantization != expected:
            return (
                f"model resource {resource.id!r} is {actual_quantization.value!r}; "
                "quantization = 'gguf' requires a GGUF artifact"
            )
        return None
    if requested in {"int8", "nvfp4"}:
        expected = ArtifactQuantization(requested)
        if actual_quantization != expected:
            actual = actual_quantization.value
            return (
                f"model resource {resource.id!r} declares quantization={actual!r}; "
                f"quantization = {requested!r} requires an explicitly annotated "
                "matching pre-quantized artifact"
            )
        return None
    if requested in {"bf16", "fp16", "fp8"}:
        if actual_quantization != ArtifactQuantization.NATIVE:
            return (
                f"model resource {resource.id!r} declares quantization="
                f"{actual_quantization.value!r}; quantization = {requested!r} "
                "requires an unquantized artifact"
            )
        if actual_precision.value != requested:
            return (
                f"model resource {resource.id!r} declares precision="
                f"{actual_precision.value!r}; quantization = {requested!r} "
                "requires explicit matching artifact precision metadata"
            )
        return None
    if requested == "native":
        if actual_quantization != ArtifactQuantization.NATIVE:
            return (
                f"model resource {resource.id!r} declares quantization="
                f"{actual_quantization.value!r}; quantization = 'native' requires "
                "an explicitly annotated unquantized artifact"
            )
        if actual_precision.value != "fp32":
            return (
                f"model resource {resource.id!r} declares precision="
                f"{actual_precision.value!r}; quantization = 'native' currently "
                "accepts only explicitly annotated fp32 artifacts"
            )
        return None
    return f"unsupported quantization compatibility constraint {requested!r}"


def _resolve_resource_quantization(
    resource: ResourceDescriptor,
    requested: QuantizationMode,
    capabilities: ExecutionCapabilities,
) -> str | None:
    """Resolve ``inherit`` to one proven loader mode for this exact artifact."""

    if requested != "inherit":
        return requested
    if resource.quantization == ArtifactQuantization.NATIVE:
        candidate = resource.precision.value
        if candidate == "fp32":
            candidate = "native"
    else:
        candidate = resource.quantization.value
    if candidate not in capabilities.quantization_modes:
        return None
    if _quantization_compatibility_error(resource, candidate) is not None:
        return None
    return candidate


def _inherit_resolution_error(resource: ResourceDescriptor) -> str:
    return (
        f"model resource {resource.id!r} cannot resolve quantization='inherit': "
        f"stored precision={resource.precision.value!r}, "
        f"stored quantization={resource.quantization.value!r} has no proven loader"
    )


def _validate_fixed_value(descriptor: ToolInput, value: Any) -> None:
    values = value if descriptor.multiple else [value]
    if descriptor.multiple and not isinstance(value, list):
        raise ValueError(f"fixed input {descriptor.key!r} must be a list")
    for item in values:
        if item is None:
            if descriptor.required:
                raise ValueError(f"fixed input {descriptor.key!r} cannot be null")
            continue
        if descriptor.type == InputType.TEXT:
            if not isinstance(item, str):
                raise ValueError(f"fixed input {descriptor.key!r} must be text")
            if descriptor.required and not item.strip():
                raise ValueError(f"fixed input {descriptor.key!r} cannot be empty")
        elif descriptor.type == InputType.NUMBER:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"fixed input {descriptor.key!r} must be a number")
            if not math.isfinite(float(item)):
                raise ValueError(f"fixed input {descriptor.key!r} must be finite")
            _validate_fixed_bounds(descriptor, float(item))
        elif descriptor.type == InputType.INTEGER:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"fixed input {descriptor.key!r} must be an integer")
            _validate_fixed_bounds(descriptor, float(item))
        elif descriptor.type == InputType.BOOLEAN:
            if not isinstance(item, bool):
                raise ValueError(f"fixed input {descriptor.key!r} must be a boolean")
        elif descriptor.type == InputType.CHOICE:
            allowed = {option.value for option in descriptor.options}
            if not isinstance(item, str) or item not in allowed:
                raise ValueError(f"fixed input {descriptor.key!r} must be one of {sorted(allowed)}")
        elif descriptor.type in {InputType.IMAGE, InputType.VIDEO, InputType.AUDIO}:
            raise ValueError(
                f"media input {descriptor.key!r} cannot be fixed in a variant; "
                "expose it or omit it when optional"
            )
        elif descriptor.type == InputType.RESOURCE:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"fixed input {descriptor.key!r} must be a resource ID")


def _validate_fixed_bounds(descriptor: ToolInput, value: float) -> None:
    if descriptor.ui is None:
        return
    if descriptor.ui.min is not None and value < descriptor.ui.min:
        raise ValueError(f"fixed input {descriptor.key!r} is below its minimum {descriptor.ui.min}")
    if descriptor.ui.max is not None and value > descriptor.ui.max:
        raise ValueError(f"fixed input {descriptor.key!r} exceeds its maximum {descriptor.ui.max}")
