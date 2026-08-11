from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.protocol import (
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceFormat,
    ResourceKind,
    discover_resources,
)
from latentslate_engine.storage import Storage
from latentslate_engine.tools import _bind_canonical_availability, default_registry
from latentslate_engine.tools.base import (
    ExecutionCapabilities,
    Tool,
    ToolContext,
)
from latentslate_engine.tools.h3 import H3TextToVideoTool
from latentslate_engine.tools.klein import Klein4BTextToImageTool
from latentslate_engine.variants import load_variant_tools


class RecordingTool(Tool):
    def __init__(
        self,
        capabilities: ExecutionCapabilities | None = None,
        *,
        family: str = "custom",
    ) -> None:
        self.inputs: dict[str, Any] | None = None
        self.execution = None
        self._capabilities = capabilities or ExecutionCapabilities()
        self._family = family

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=UUID("e31ea040-cf48-43a5-a850-d86d1090e159"),
            key="test.base",
            schema_revision=1,
            name="Base",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(key="prompt", label="Prompt", type=InputType.TEXT, required=True),
                ToolInput(
                    key="steps",
                    label="Steps",
                    type=InputType.INTEGER,
                    required=True,
                    default=20,
                    ui=InputUi(min=1, max=50, step=1),
                ),
            ],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]):
        self.inputs = inputs
        self.execution = context.execution
        return []

    def model_family(self) -> str:
        return self._family

    def execution_capabilities(self) -> ExecutionCapabilities:
        return self._capabilities


def settings(tmp_path: Path) -> Settings:
    value = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    value.ensure_directories()
    return value


def test_missing_canonical_bundles_are_unavailable_but_variant_base_stays_usable(
    tmp_path: Path,
):
    value = settings(tmp_path)
    base = H3TextToVideoTool()
    runtime_available = base.variant_base_availability()

    bound = _bind_canonical_availability(value, base)

    assert not bound.descriptor.available
    assert "bundles install h3-basic" in (bound.descriptor.unavailable_reason or "")
    assert bound.variant_base_availability() == runtime_available


def test_default_registry_marks_missing_heavy_canonical_tools_unavailable(
    tmp_path: Path,
):
    value = settings(tmp_path)

    descriptors = {
        descriptor.key: descriptor
        for descriptor in default_registry(value, emit_warnings=False).descriptors()
    }

    expected_bundles = {
        "h3.text_to_video": "h3-basic",
        "h3.first_last_frame_video": "h3-basic",
        "ltx23.text_to_video": "ltx23-basic",
        "flux2_klein9b.text_to_image": "klein9b-basic",
        "flux2_klein9b.image_to_image": "klein9b-basic",
    }
    for key, bundle_id in expected_bundles.items():
        descriptor = descriptors[key]
        assert not descriptor.available
        assert f"bundles install {bundle_id}" in (descriptor.unavailable_reason or "")


def test_file_drop_resource_discovery(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "klein4b" / "local-klein"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"weights")
    lora = value.lora_root / "klein4b" / "cinematic.safetensors"
    lora.write_bytes(b"lora")
    lora.with_suffix(".toml").write_text(
        'name = "Cinematic"\ndefault_strength = 0.75\ntags = ["style"]\n',
        encoding="utf-8",
    )

    inventory = discover_resources(value)

    assert inventory.errors == []
    models = [resource for resource in inventory.resources if resource.kind == ResourceKind.MODEL]
    loras = [resource for resource in inventory.resources if resource.kind == ResourceKind.LORA]
    assert len(models) == 1
    assert models[0].id == "model:klein4b:local-klein"
    assert models[0].format == ResourceFormat.DIFFUSERS
    assert models[0].relative_path == "models/klein4b/local-klein"
    assert len(loras) == 1
    assert loras[0].id == "lora:klein4b:cinematic"
    assert loras[0].name == "Cinematic"
    assert loras[0].default_strength == 0.75
    assert inventory.path_for(loras[0].id) == lora.resolve()


def test_model_sidecar_records_stored_precision_and_quantization(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "klein4b" / "local-klein"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / ".latentslate-model.toml").write_text(
        'precision = "bf16"\nquantization = "native"\n',
        encoding="utf-8",
    )

    inventory = discover_resources(value)

    assert inventory.errors == []
    resource = inventory.resolve("model:klein4b:local-klein")
    assert resource.precision == ArtifactPrecision.BF16
    assert resource.quantization == ArtifactQuantization.NATIVE


def test_inherit_variant_resolves_canonical_bf16_resource_to_concrete_mode(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "custom" / "canonical"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / ".latentslate-model.toml").write_text(
        'precision = "bf16"\nquantization = "native"\n', encoding="utf-8"
    )
    (value.variants_root / "custom" / "inherit.toml").write_text(
        """
key = "test.inherit"
name = "Inherited artifact"
family = "custom"
base_tool = "test.base"

[model]
resource = "model:custom:canonical"

[inputs.prompt]
""",
        encoding="utf-8",
    )
    base = RecordingTool(
        ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            quantization_modes=frozenset({"bf16"}),
        )
    )
    result = load_variant_tools(value, [base], discover_resources(value))
    variant = result.tools[0]
    context = ToolContext(
        job_id=UUID(int=0),
        settings=value,
        storage=Storage(value),
        cancel_event=Event(),
        progress=lambda _value, _message: None,
    )
    variant.run(context, {"prompt": "test"})
    assert base.execution.optimizations["quantization"] == "bf16"


def test_inherit_variant_with_unknown_or_unproven_fp8_artifact_is_unavailable(tmp_path: Path):
    value = settings(tmp_path)
    for name, metadata in (
        ("unknown", ""),
        ("fp8", 'precision = "fp8"\nquantization = "native"\n'),
    ):
        model = value.model_root / "custom" / name
        model.mkdir()
        (model / "model_index.json").write_text("{}", encoding="utf-8")
        if metadata:
            (model / ".latentslate-model.toml").write_text(metadata, encoding="utf-8")
    (value.variants_root / "custom" / "inherit.toml").write_text(
        """
key = "test.inherit_unknown"
name = "Inherited unknown"
family = "custom"
base_tool = "test.base"

[model]
exposed = true
""",
        encoding="utf-8",
    )
    base = RecordingTool(
        ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            quantization_modes=frozenset({"bf16", "native"}),
        )
    )
    result = load_variant_tools(value, [base], discover_resources(value))
    assert result.tools[0].descriptor.available is False
    assert "no compatible model resources" in result.tools[0].descriptor.unavailable_reason


def test_exposed_model_selector_filters_family_invalid_resources(tmp_path: Path):
    value = settings(tmp_path)
    for name in ("valid", "broken"):
        model = value.model_root / "custom" / name
        model.mkdir()
        (model / "model_index.json").write_text("{}", encoding="utf-8")
        (model / ".latentslate-model.toml").write_text(
            'precision = "bf16"\nquantization = "native"\n', encoding="utf-8"
        )
    (value.variants_root / "custom" / "validated.toml").write_text(
        """
key = "test.validated_selector"
name = "Validated selector"
family = "custom"
base_tool = "test.base"

[model]
exposed = true

[inputs.prompt]
""",
        encoding="utf-8",
    )

    class ValidatingTool(RecordingTool):
        def validate_model_resource(self, _resource, path: Path) -> list[str]:
            return ["synthetic family contract failure"] if path.name == "broken" else []

    base = ValidatingTool(
        ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            quantization_modes=frozenset({"bf16"}),
        )
    )
    result = load_variant_tools(value, [base], discover_resources(value))

    assert result.errors == []
    variant = result.tools[0]
    model_input = next(item for item in variant.descriptor.inputs if item.key == "model")
    assert [option.value for option in model_input.options] == ["model:custom:valid"]
    assert variant.descriptor.available is True


def test_gguf_quantization_is_the_only_inferred_artifact_property(tmp_path: Path):
    value = settings(tmp_path)
    gguf = value.model_root / "custom" / "quantized.gguf"
    gguf.write_bytes(b"gguf")
    safetensors = value.model_root / "custom" / "model.safetensors"
    safetensors.write_bytes(b"weights")

    inventory = discover_resources(value)

    assert inventory.errors == []
    gguf_resource = inventory.resolve("model:custom:quantized")
    native_resource = next(
        resource
        for resource in inventory.resources
        if resource.relative_path.endswith("model.safetensors")
    )
    assert gguf_resource.quantization == ArtifactQuantization.GGUF
    assert gguf_resource.precision == ArtifactPrecision.UNKNOWN
    assert native_resource.quantization == ArtifactQuantization.UNKNOWN
    assert native_resource.precision == ArtifactPrecision.UNKNOWN


def test_resource_symlink_cannot_escape_owned_root(tmp_path: Path):
    value = settings(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
    outside.write_bytes(b"outside")
    link = value.lora_root / "klein4b" / "escape.safetensors"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")

    inventory = discover_resources(value)

    assert inventory.resources == []
    assert any("must stay within" in error for error in inventory.errors)


def test_variant_reshapes_base_schema_and_executes_fixed_inputs(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "simple.toml"
    variant_path.write_text(
        """
key = "test.simple"
name = "Simple"
family = "custom"
base_tool = "test.base"

[inputs.prompt]
label = "Direction"

[fixed]
steps = 4
""",
        encoding="utf-8",
    )
    base = RecordingTool()
    inventory = discover_resources(value)
    result = load_variant_tools(value, [base], inventory)

    assert result.errors == []
    assert len(result.tools) == 1
    variant = result.tools[0]
    assert variant.descriptor.available
    assert [item.key for item in variant.descriptor.inputs] == ["prompt"]
    assert variant.descriptor.inputs[0].label == "Direction"

    context = ToolContext(
        job_id=UUID(int=0),
        settings=value,
        storage=Storage(value),
        cancel_event=Event(),
        progress=lambda _value, _message: None,
    )
    variant.run(context, {"prompt": "A quiet room"})
    assert base.inputs == {"prompt": "A quiet room", "steps": 4}
    assert base.execution is not None
    assert base.execution.variant_key == "test.simple"


def test_variant_never_silently_ignores_unimplemented_runtime_features(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "custom" / "model.gguf"
    model.write_bytes(b"gguf")
    variant_path = value.variants_root / "custom" / "advanced.toml"
    variant_path.write_text(
        """
key = "test.advanced"
name = "Advanced"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[model]
resource = "model:custom:model"

[optimizations]
attention = "sage_hub"
quantization = "gguf"
cache = "prompt"
""",
        encoding="utf-8",
    )
    base = RecordingTool()
    result = load_variant_tools(value, [base], discover_resources(value))

    assert result.errors == []
    descriptor = result.tools[0].descriptor
    assert not descriptor.available
    reason = descriptor.unavailable_reason or ""
    assert "attention mode 'sage_hub'" in reason
    assert "cache mode 'prompt'" in reason
    assert "model overrides" in reason
    assert "quantization mode 'gguf'" in reason


def test_invalid_fixed_input_is_an_authoring_error(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "invalid-fixed.toml"
    variant_path.write_text(
        """
key = "test.invalid_fixed"
name = "Invalid fixed"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[fixed]
steps = "four"
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("must be an integer" in error for error in result.errors)


def test_base_input_cannot_be_both_fixed_and_exposed(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "conflict.toml"
    variant_path.write_text(
        """
key = "test.conflict"
name = "Conflict"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[inputs.steps]

[fixed]
steps = 4
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("both fixed and exposed" in error for error in result.errors)


def test_required_exposed_lora_without_resources_is_unavailable(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "required-lora.toml"
    variant_path.write_text(
        """
key = "test.required_lora"
name = "Required LoRA"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[[loras]]
slot = "style"
exposed = true
required = true
""",
        encoding="utf-8",
    )

    result = load_variant_tools(
        value,
        [
            RecordingTool(
                ExecutionCapabilities(lora_formats=frozenset({ResourceFormat.SAFETENSORS.value}))
            )
        ],
        discover_resources(value),
    )

    assert result.errors == []
    assert len(result.tools) == 1
    descriptor = result.tools[0].descriptor
    assert not descriptor.available
    assert "no compatible resources" in (descriptor.unavailable_reason or "")
    lora_input = next(item for item in descriptor.inputs if item.key == "style_lora")
    assert [option.value for option in lora_input.options] == ["unavailable"]


def test_disabled_variant_is_listed_but_not_executable(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "disabled.toml"
    variant_path.write_text(
        """
key = "test.disabled"
name = "Disabled"
enabled = false
family = "custom"
base_tool = "test.base"
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.errors == []
    assert result.tools == []
    assert len(result.entries) == 1
    assert not result.entries[0].enabled
    assert not result.entries[0].available
    assert result.entries[0].unavailable_reason == "variant is disabled"


def test_variant_symlink_cannot_escape_variants_root(tmp_path: Path):
    value = settings(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-variant.toml"
    outside.write_text(
        'key = "test.escape"\nname = "Escape"\nfamily = "custom"\nbase_tool = "test.base"\n',
        encoding="utf-8",
    )
    link = value.variants_root / "custom" / "escape.toml"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("must stay within" in error for error in result.errors)


def test_invalid_model_selector_definition_is_reported(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "bad-model.toml"
    variant_path.write_text(
        """
key = "test.bad_model"
name = "Bad model"
family = "custom"
base_tool = "test.base"

[model]
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("fixed resource or an exposed selector" in error for error in result.errors)


def test_variant_choice_override_is_revalidated(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "invalid-choice.toml"
    variant_path.write_text(
        """
key = "test.invalid_choice"
name = "Invalid choice"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[inputs.steps]
options = ["four"]
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("only override options on a choice" in error for error in result.errors)


def test_exposed_model_default_must_match_allowed_resources(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "custom" / "local.gguf"
    model.write_bytes(b"gguf")
    variant_path = value.variants_root / "custom" / "excluded-model.toml"
    variant_path.write_text(
        """
key = "test.excluded_model"
name = "Excluded model"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[model]
resource = "model:custom:local"
exposed = true
allowed = ["*other*"]
""",
        encoding="utf-8",
    )

    result = load_variant_tools(
        value,
        [
            RecordingTool(
                ExecutionCapabilities(model_formats=frozenset({ResourceFormat.GGUF.value}))
            )
        ],
        discover_resources(value),
    )

    assert result.tools == []
    assert any("excluded by the allowed resource patterns" in error for error in result.errors)


def test_variant_family_must_match_curated_base_tool(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "wan22" / "wrong-base.toml"
    variant_path.write_text(
        """
key = "wan22.wrong_base"
name = "Wrong base"
family = "wan22"
base_tool = "flux2_klein4b.text_to_image"
""",
        encoding="utf-8",
    )

    result = load_variant_tools(
        value,
        [Klein4BTextToImageTool()],
        discover_resources(value),
    )

    assert result.tools == []
    assert any(
        "variant family 'wan22' does not match base_tool family 'klein4b'" in error
        for error in result.errors
    )


@pytest.mark.parametrize(
    ("default", "message"),
    [
        ('"four"', "integer input defaults must be integers"),
        ("100", "input default exceeds its maximum 50.0"),
    ],
)
def test_overridden_defaults_are_validated_at_catalog_build(
    tmp_path: Path,
    default: str,
    message: str,
):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "invalid-default.toml"
    variant_path.write_text(
        f"""
key = "test.invalid_default"
name = "Invalid default"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[inputs.steps]
default = {default}
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any(message in error for error in result.errors)


@pytest.mark.parametrize(
    ("settings_text", "message"),
    [
        ("strength = nan", "LoRA strength must be finite"),
        ("strength_min = -inf", "LoRA strength_min must be finite"),
        ("strength_max = inf", "LoRA strength_max must be finite"),
        ("strength_step = nan", "LoRA strength_step must be finite"),
        ("strength = 3.0\nstrength_min = -2.0\nstrength_max = 2.0", "within"),
    ],
)
def test_lora_numeric_settings_are_finite_and_bounded(
    tmp_path: Path,
    settings_text: str,
    message: str,
):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "bad-lora-numbers.toml"
    variant_path.write_text(
        f"""
key = "test.bad_lora_numbers"
name = "Bad LoRA numbers"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[[loras]]
slot = "style"
resource = "missing.safetensors"
{settings_text}
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any(message in error for error in result.errors)


def test_sharded_component_repository_is_grouped_and_not_selectable_as_model(tmp_path: Path):
    value = settings(tmp_path)
    component = value.model_root / "klein9b" / "Qwen--Qwen3-8B-FP8"
    component.mkdir(parents=True)
    (component / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (component / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (component / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (component / "model-00002-of-00002.safetensors").write_bytes(b"two")
    pipeline = value.model_root / "klein9b" / "local-pipeline"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
    (pipeline / ".latentslate-model.toml").write_text(
        'precision = "bf16"\nquantization = "native"\n', encoding="utf-8"
    )

    inventory = discover_resources(value)

    assert inventory.errors == []
    component_resources = [resource for resource in inventory.resources if resource.component]
    assert len(component_resources) == 1
    assert component_resources[0].relative_path.endswith("Qwen--Qwen3-8B-FP8")
    assert component_resources[0].component == "repository"
    assert not any("model-0000" in resource.relative_path for resource in inventory.resources)
    selectable = inventory.matching(kind=ResourceKind.MODEL, family="klein9b")
    assert [resource.id for resource in selectable] == ["model:klein9b:local-pipeline"]

    variant_path = value.variants_root / "klein9b" / "model-selector.toml"
    variant_path.write_text(
        """
key = "test.klein_selector"
name = "Klein selector"
family = "klein9b"
base_tool = "test.base"

[inputs.prompt]

[model]
exposed = true
""",
        encoding="utf-8",
    )
    result = load_variant_tools(
        value,
        [
            RecordingTool(
                ExecutionCapabilities(
                    model_formats=frozenset({ResourceFormat.DIFFUSERS.value}),
                    quantization_modes=frozenset({"bf16"}),
                ),
                family="klein9b",
            )
        ],
        inventory,
    )
    assert result.errors == []
    model_input = next(
        descriptor for descriptor in result.tools[0].descriptor.inputs if descriptor.key == "model"
    )
    assert [option.value for option in model_input.options] == ["model:klein9b:local-pipeline"]


def test_family_adapter_can_promote_one_exact_component_role_to_model(tmp_path: Path):
    value = settings(tmp_path)
    artifact = value.model_root / "klein4b" / "stored-fp8.safetensors"
    artifact.write_bytes(b"stored")
    artifact.with_suffix(".toml").write_text(
        'format = "safetensors"\nprecision = "fp8"\nquantization = "native"\n'
        'component = "transformer"\n',
        encoding="utf-8",
    )
    variant_path = value.variants_root / "klein4b" / "stored.toml"
    variant_path.write_text(
        """
key = "test.klein_stored"
name = "Klein stored"
family = "klein4b"
base_tool = "test.base"

[inputs.prompt]

[model]
resource = "model:klein4b:stored-fp8"

[optimizations]
quantization = "fp8"
""",
        encoding="utf-8",
    )

    class ComponentAwareTool(RecordingTool):
        def model_resource_components(self) -> frozenset[str]:
            return frozenset({"transformer"})

    result = load_variant_tools(
        value,
        [
            ComponentAwareTool(
                ExecutionCapabilities(
                    model_formats=frozenset({"safetensors"}),
                    quantization_modes=frozenset({"fp8"}),
                ),
                family="klein4b",
            )
        ],
        discover_resources(value),
    )

    assert result.errors == []
    assert len(result.tools) == 1
    assert result.tools[0].descriptor.available


def test_gguf_quantization_requires_one_fixed_gguf_model(tmp_path: Path):
    value = settings(tmp_path)
    model = value.model_root / "custom" / "model.gguf"
    model.write_bytes(b"gguf")
    variant_path = value.variants_root / "custom" / "exposed-gguf.toml"
    variant_path.write_text(
        """
key = "test.exposed_gguf"
name = "Exposed GGUF"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[model]
exposed = true

[optimizations]
quantization = "gguf"
""",
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("requires one fixed GGUF model resource" in error for error in result.errors)


def test_exposed_model_selector_never_offers_gguf(tmp_path: Path):
    value = settings(tmp_path)
    (value.model_root / "custom" / "model.gguf").write_bytes(b"gguf")
    pipeline = value.model_root / "custom" / "pipeline"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
    (pipeline / ".latentslate-model.toml").write_text(
        'precision = "bf16"\nquantization = "native"\n', encoding="utf-8"
    )
    variant_path = value.variants_root / "custom" / "selector.toml"
    variant_path.write_text(
        """
key = "test.selector"
name = "Selector"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[model]
exposed = true
""",
        encoding="utf-8",
    )

    result = load_variant_tools(
        value,
        [
            RecordingTool(
                ExecutionCapabilities(
                    model_formats=frozenset(
                        {ResourceFormat.DIFFUSERS.value, ResourceFormat.GGUF.value}
                    ),
                    quantization_modes=frozenset({"bf16"}),
                )
            )
        ],
        discover_resources(value),
    )

    assert result.errors == []
    model_input = next(item for item in result.tools[0].descriptor.inputs if item.key == "model")
    assert [option.value for option in model_input.options] == ["model:custom:pipeline"]


def test_explicit_bf16_selector_excludes_mismatched_artifacts(tmp_path: Path):
    value = settings(tmp_path)
    for name, precision, quantization in (
        ("bf16", "bf16", "native"),
        ("fp8", "fp8", "native"),
        ("int8", "unknown", "int8"),
    ):
        pipeline = value.model_root / "custom" / name
        pipeline.mkdir()
        (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
        (pipeline / ".latentslate-model.toml").write_text(
            f'precision = "{precision}"\nquantization = "{quantization}"\n',
            encoding="utf-8",
        )
    variant_path = value.variants_root / "custom" / "bf16-selector.toml"
    variant_path.write_text(
        """
key = "test.bf16_selector"
name = "BF16 selector"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[model]
exposed = true

[optimizations]
quantization = "bf16"
""",
        encoding="utf-8",
    )

    result = load_variant_tools(
        value,
        [
            RecordingTool(
                ExecutionCapabilities(
                    model_formats=frozenset({ResourceFormat.DIFFUSERS.value}),
                    quantization_modes=frozenset({"bf16"}),
                )
            )
        ],
        discover_resources(value),
    )

    assert result.errors == []
    model_input = next(item for item in result.tools[0].descriptor.inputs if item.key == "model")
    assert [option.value for option in model_input.options] == ["model:custom:bf16"]


def test_allow_patterns_are_case_insensitive_and_platform_independent(tmp_path: Path):
    value = settings(tmp_path)
    lora = value.lora_root / "klein4b" / "CinematicStyle.safetensors"
    lora.write_bytes(b"lora")
    inventory = discover_resources(value)

    matched = inventory.matching(
        kind=ResourceKind.LORA,
        family="klein4b",
        allow=["*CINEMATICSTYLE*"],
    )

    assert [resource.id for resource in matched] == ["lora:klein4b:cinematicstyle"]


def test_execution_capabilities_are_exact_modes_not_feature_buckets(tmp_path: Path):
    value = settings(tmp_path)
    supported = value.variants_root / "custom" / "supported.toml"
    supported.write_text(
        """
key = "test.supported_modes"
name = "Supported modes"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[optimizations]
attention = "native"
quantization = "int8"
""",
        encoding="utf-8",
    )
    unsupported = value.variants_root / "custom" / "unsupported.toml"
    unsupported.write_text(
        """
key = "test.unsupported_modes"
name = "Unsupported modes"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[optimizations]
attention = "sage"
quantization = "nvfp4"
""",
        encoding="utf-8",
    )
    base = RecordingTool(
        ExecutionCapabilities(
            attention_modes=frozenset({"native"}),
            quantization_modes=frozenset({"int8"}),
        )
    )

    result = load_variant_tools(value, [base], discover_resources(value))

    assert result.errors == []
    by_key = {tool.descriptor.key: tool.descriptor for tool in result.tools}
    assert by_key["test.supported_modes"].available
    assert not by_key["test.unsupported_modes"].available
    reason = by_key["test.unsupported_modes"].unavailable_reason or ""
    assert "attention mode 'sage'" in reason
    assert "quantization mode 'nvfp4'" in reason
