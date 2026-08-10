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
from .model_store import MODEL_FAMILIES
from .protocol import ChoiceOption, InputType, InputUi, ToolDescriptor, ToolInput
from .resources import ResourceDescriptor, ResourceFormat, ResourceInventory, ResourceKind
from .tools.base import ExecutionPlan, LoraExecution, Tool, ToolContext

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
    group_offload_use_stream: bool = True
    group_offload_record_stream: bool = True
    low_cpu_mem_usage: bool = True
    keep_pipeline_loaded: bool = True

    @model_validator(mode="after")
    def validate_dependencies(self) -> OptimizationConfig:
        compile_options = {"compile_mode", "compile_fullgraph", "compile_dynamic"}
        if (
            not self.compile
            and self.model_fields_set.intersection(compile_options)
            and (
                self.compile_mode != "default"
                or self.compile_fullgraph
                or self.compile_dynamic
            )
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
        if self.strength_min > self.strength_max:
            raise ValueError("LoRA strength_min cannot exceed strength_max")
        if self.strength_step <= 0:
            raise ValueError("LoRA strength_step must be positive")
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
    model_resource: str | None = None
    lora_slots: list[str] = Field(default_factory=list)
    optimizations: dict[str, Any] = Field(default_factory=dict)


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

    def execution_capabilities(self) -> set[str]:
        return self.base_tool.execution_capabilities()

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
        plan = ExecutionPlan(
            variant_key=self.definition.key,
            family=self.definition.family,
            model_resource_id=model_resource.id if model_resource else None,
            model_path=self.inventory.path_for(model_resource.id) if model_resource else None,
            model_format=model_resource.format.value if model_resource else None,
            loras=tuple(loras),
            optimizations=self.definition.optimizations.model_dump(mode="json"),
            runtime_parameters=runtime_parameters,
        )
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
        if not base.available:
            unavailable.append(base.unavailable_reason or "base tool is unavailable")

        requested = self.definition.optimizations.requested_features()
        if self.definition.model is not None:
            requested.add("model_override")
        if self.definition.loras:
            requested.add("loras")
        runtime_fixed = set(self.definition.fixed) - {descriptor.key for descriptor in base.inputs}
        if runtime_fixed:
            requested.add("runtime_parameters")
        unsupported = sorted(requested - self.base_tool.execution_capabilities())
        if unsupported:
            unavailable.append(
                "base runtime does not yet support variant features: " + ", ".join(unsupported)
            )

        unavailable.extend(self._validate_resources())
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
                update["options"] = [ChoiceOption(value=value, label=value) for value in config.options]
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
            resources = self._matching_resources(ResourceKind.MODEL, self.definition.model.allowed)
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
                        options = [ChoiceOption(value="unavailable", label="No compatible LoRAs found")]
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
            and not self._matching_resources(
                ResourceKind.MODEL,
                self.definition.model.allowed,
            )
        ):
            errors.append("no compatible model resources were discovered")

        quantization = self.definition.optimizations.quantization
        if model_resource is not None:
            if quantization == "gguf" and model_resource.format != ResourceFormat.GGUF:
                errors.append("quantization 'gguf' requires a GGUF model resource")
            elif model_resource.format == ResourceFormat.GGUF and quantization not in {
                "inherit",
                "gguf",
            }:
                errors.append(
                    f"GGUF model resource is incompatible with quantization {quantization!r}"
                )

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

    def _matching_resources(
        self,
        kind: ResourceKind,
        allow: list[str],
    ) -> list[ResourceDescriptor]:
        return self.inventory.matching(
            kind=kind,
            family=self.definition.family,
            allow=allow or None,
        )

    def _resolve_resource_reference(
        self,
        reference: str,
        *,
        kind: ResourceKind,
    ) -> ResourceDescriptor:
        return self.inventory.resolve(
            reference,
            kind=kind,
            family=self.definition.family,
        )

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


def load_variant_tools(
    settings: Settings,
    base_tools: list[Tool],
    inventory: ResourceInventory,
) -> VariantLoadResult:
    by_key = {tool.descriptor.key: tool for tool in base_tools}
    tools: list[Tool] = []
    entries: list[VariantCatalogEntry] = []
    errors: list[str] = []
    seen_ids = {tool.descriptor.id for tool in base_tools}
    seen_keys = set(by_key)

    settings.variants_root.mkdir(parents=True, exist_ok=True)
    variants_root = settings.variants_root.resolve()
    home = settings.home.resolve()
    for path in sorted(settings.variants_root.rglob("*.toml")):
        if path.name.startswith(".") or any(
            part.startswith(".") for part in path.relative_to(settings.variants_root).parts
        ):
            continue
        try:
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(variants_root)
            except ValueError as exc:
                raise ValueError(f"variant file must stay within {variants_root}") from exc
            raw = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
            definition_data = raw.get("variant", raw)
            definition = VariantDefinition.model_validate(definition_data)
            variant_id = definition.id or uuid5(VARIANT_NAMESPACE, definition.key)
            if variant_id in seen_ids:
                raise ValueError(f"duplicate tool UUID {variant_id}")
            if definition.key in seen_keys:
                raise ValueError(f"duplicate tool key {definition.key!r}")
            seen_ids.add(variant_id)
            seen_keys.add(definition.key)
            source_path = resolved_path.relative_to(home)

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

            try:
                base_tool = by_key[definition.base_tool]
            except KeyError as exc:
                raise ValueError(f"unknown base_tool {definition.base_tool!r}") from exc
            tool = VariantTool(
                definition=definition,
                source_path=source_path,
                base_tool=base_tool,
                inventory=inventory,
            )
            tools.append(tool)
            entries.append(tool.catalog_entry())
        except Exception as exc:  # noqa: BLE001 - collect all iterative authoring errors
            errors.append(f"{path}: {exc}")

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
        model_resource=definition.model.resource if definition.model else None,
        lora_slots=[lora.slot for lora in definition.loras],
        optimizations=definition.optimizations.model_dump(mode="json"),
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
                raise ValueError(
                    f"fixed input {descriptor.key!r} must be one of {sorted(allowed)}"
                )
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
        raise ValueError(
            f"fixed input {descriptor.key!r} is below its minimum {descriptor.ui.min}"
        )
    if descriptor.ui.max is not None and value > descriptor.ui.max:
        raise ValueError(
            f"fixed input {descriptor.key!r} exceeds its maximum {descriptor.ui.max}"
        )
