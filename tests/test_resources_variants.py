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
from latentslate_engine.resources import ResourceFormat, ResourceKind, discover_resources
from latentslate_engine.storage import Storage
from latentslate_engine.tools.base import Tool, ToolContext
from latentslate_engine.variants import load_variant_tools


class RecordingTool(Tool):
    def __init__(self, capabilities: set[str] | None = None) -> None:
        self.inputs: dict[str, Any] | None = None
        self.execution = None
        self._capabilities = capabilities or set()

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

    def execution_capabilities(self) -> set[str]:
        return set(self._capabilities)


def settings(tmp_path: Path) -> Settings:
    value = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    value.ensure_directories()
    return value


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
        '''
key = "test.simple"
name = "Simple"
family = "custom"
base_tool = "test.base"

[inputs.prompt]
label = "Direction"

[fixed]
steps = 4
''',
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
        '''
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
''',
        encoding="utf-8",
    )
    base = RecordingTool()
    result = load_variant_tools(value, [base], discover_resources(value))

    assert result.errors == []
    descriptor = result.tools[0].descriptor
    assert not descriptor.available
    reason = descriptor.unavailable_reason or ""
    assert "attention_backend" in reason
    assert "cache_policy" in reason
    assert "model_override" in reason
    assert "quantization" in reason


def test_invalid_fixed_input_is_an_authoring_error(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "invalid-fixed.toml"
    variant_path.write_text(
        '''
key = "test.invalid_fixed"
name = "Invalid fixed"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[fixed]
steps = "four"
''',
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("must be an integer" in error for error in result.errors)


def test_base_input_cannot_be_both_fixed_and_exposed(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "conflict.toml"
    variant_path.write_text(
        '''
key = "test.conflict"
name = "Conflict"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[inputs.steps]

[fixed]
steps = 4
''',
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert result.tools == []
    assert any("both fixed and exposed" in error for error in result.errors)


def test_required_exposed_lora_without_resources_is_unavailable(tmp_path: Path):
    value = settings(tmp_path)
    variant_path = value.variants_root / "custom" / "required-lora.toml"
    variant_path.write_text(
        '''
key = "test.required_lora"
name = "Required LoRA"
family = "custom"
base_tool = "test.base"

[inputs.prompt]

[[loras]]
slot = "style"
exposed = true
required = true
''',
        encoding="utf-8",
    )

    result = load_variant_tools(value, [RecordingTool({"loras"})], discover_resources(value))

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
        '''
key = "test.disabled"
name = "Disabled"
enabled = false
family = "custom"
base_tool = "test.base"
''',
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
        '''
key = "test.bad_model"
name = "Bad model"
family = "custom"
base_tool = "test.base"

[model]
''',
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
        [RecordingTool({"model_override"})],
        discover_resources(value),
    )

    assert result.tools == []
    assert any("excluded by the allowed resource patterns" in error for error in result.errors)
