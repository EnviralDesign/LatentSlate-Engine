from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient

from latentslate_engine.app import create_app
from latentslate_engine.config import Settings
from latentslate_engine.protocol import (
    InputType,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.recipes import (
    build_deployment_lock,
    build_deployment_plan,
    deployment_profile_catalog,
)
from latentslate_engine.resources import ResourceSourceKind, discover_resources
from latentslate_engine.tools import ToolRegistry
from latentslate_engine.tools.base import ExecutionCapabilities, Tool, ToolContext
from latentslate_engine.variants import load_variant_tools


class RecordingTool(Tool):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=UUID("e8d0879f-9cc7-4e55-82d4-298cdf5d914e"),
            key="test.base",
            schema_revision=1,
            name="Base",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(
                    key="prompt",
                    label="Prompt",
                    type=InputType.TEXT,
                    required=True,
                )
            ],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]):
        del context, inputs
        return []

    def model_family(self) -> str:
        return "custom"

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset({"safetensors"}),
            quantization_modes=frozenset({"bf16"}),
        )


def settings(
    tmp_path: Path,
    *,
    recipe_paths: tuple[Path, ...] = (),
    profile_paths: tuple[Path, ...] = (),
) -> Settings:
    value = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
        recipe_paths=recipe_paths,
        deployment_profile_paths=profile_paths,
    )
    value.ensure_directories()
    return value


def _write_model_and_lora(value: Settings) -> tuple[Path, Path]:
    model = value.model_root / "custom" / "shared-model"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / ".latentslate-model.toml").write_text(
        '''
id = "model:custom:shared"
name = "Shared Model"
precision = "bf16"
quantization = "native"

[[sources]]
type = "huggingface"
repo_id = "example/shared-model"
revision = "0123456789abcdef"
requires_auth = true
''',
        encoding="utf-8",
    )
    lora = value.lora_root / "custom" / "style.safetensors"
    lora.write_bytes(b"style")
    lora.with_suffix(".toml").write_text(
        '''
id = "lora:custom:style"
name = "Style"

[[sources]]
type = "civitai"
model_version_id = 12345
file_id = 67890
requires_auth = true
''',
        encoding="utf-8",
    )
    return model, lora


def _write_recipe(path: Path, key: str, *, with_lora: bool) -> None:
    lora = '''
[[runnable_recipe.loras]]
slot = "style"
resource = "lora:custom:style"
strength = 0.7
''' if with_lora else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''
[runnable_recipe]
key = "{key}"
name = "{key}"
family = "custom"
base_tool = "test.base"

[runnable_recipe.model]
resource = "model:custom:shared"

[runnable_recipe.inputs.prompt]

[runnable_recipe.optimizations]
quantization = "bf16"
{lora}
''',
        encoding="utf-8",
    )


def _registry(value: Settings) -> ToolRegistry:
    base = RecordingTool()
    inventory = discover_resources(value)
    loaded = load_variant_tools(value, [base], inventory)
    assert loaded.errors == []
    return ToolRegistry(
        [base, *loaded.tools],
        resources=inventory,
        variants=loaded.entries,
        variant_errors=loaded.errors,
    )


def test_data_layout_and_private_catalog_search_paths(tmp_path: Path):
    private = tmp_path.parent / "private-recipes"
    profiles = tmp_path.parent / "private-profiles"
    value = settings(tmp_path, recipe_paths=(private,), profile_paths=(profiles,))

    assert value.recipes_root.is_dir()
    assert value.deployment_profiles_root.is_dir()
    assert (value.recipes_root / "wan22").is_dir()
    assert [label for label, _path in value.recipe_catalog_roots()] == [
        "builtin",
        "local",
        "private-1",
        "legacy",
    ]
    assert [label for label, _path in value.deployment_profile_roots()] == [
        "builtin",
        "local",
        "private-1",
    ]


def test_private_recipes_and_legacy_variants_share_one_catalog(tmp_path: Path):
    private = tmp_path.parent / "private-recipes"
    value = settings(tmp_path, recipe_paths=(private,))
    _write_recipe(private / "private.toml", "test.private", with_lora=False)
    _write_recipe(value.variants_root / "custom" / "legacy.toml", "test.legacy", with_lora=False)
    _write_model_and_lora(value)

    loaded = load_variant_tools(value, [RecordingTool()], discover_resources(value))

    assert loaded.errors == []
    assert {entry.key for entry in loaded.entries} == {"test.private", "test.legacy"}
    sources = {entry.key: entry.source_path for entry in loaded.entries}
    assert sources["test.private"] == "private-1/private.toml"
    assert sources["test.legacy"] == "legacy/custom/legacy.toml"


def test_resource_sources_are_typed_and_credentials_stay_external(tmp_path: Path):
    value = settings(tmp_path)
    _write_model_and_lora(value)

    inventory = discover_resources(value)
    model = inventory.resolve("model:custom:shared")
    lora = inventory.resolve("lora:custom:style")

    assert model.sources[0].type == ResourceSourceKind.HUGGINGFACE
    assert model.sources[0].required_secret() == "HF_TOKEN"
    assert lora.sources[0].type == ResourceSourceKind.CIVITAI
    assert lora.sources[0].required_secret() == "CIVITAI_TOKEN"
    dumped = model.sources[0].model_dump(mode="json")
    assert "token" not in dumped
    assert dumped["repo_id"] == "example/shared-model"


def test_deployment_profile_dedupes_exact_resource_closure(tmp_path: Path):
    value = settings(tmp_path)
    _write_model_and_lora(value)
    _write_recipe(value.recipes_root / "custom" / "one.toml", "test.one", with_lora=True)
    _write_recipe(value.recipes_root / "custom" / "two.toml", "test.two", with_lora=False)
    (value.deployment_profiles_root / "editor.toml").write_text(
        '''
[profile]
key = "editor"
name = "Editor"
recipes = ["test.one", "test.two"]
target = "local-5080"
''',
        encoding="utf-8",
    )
    registry = _registry(value)

    plan = build_deployment_plan(value, registry, "editor")
    lock = build_deployment_lock(
        value,
        registry,
        "editor",
        generated_at=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert [resource.id for resource in plan.resources] == [
        "lora:custom:style",
        "model:custom:shared",
    ]
    assert plan.total_bytes == sum(resource.size_bytes for resource in plan.resources)
    assert plan.incremental_bytes == 0
    assert plan.locally_runnable
    assert plan.remote_provisionable
    assert plan.required_secrets == ["CIVITAI_TOKEN", "HF_TOKEN"]
    assert lock.profile_key == "editor"
    assert lock.engine_version == plan.engine_version
    assert lock.required_secrets == plan.required_secrets
    assert len(lock.resources) == 2
    assert all("token" not in source.model_dump(mode="json") for item in lock.resources for source in item.sources)


def test_source_less_local_resource_is_runnable_but_not_remotely_provisionable(
    tmp_path: Path,
):
    value = settings(tmp_path)
    model = value.model_root / "custom" / "manual"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / ".latentslate-model.toml").write_text(
        'id = "model:custom:manual"\nprecision = "bf16"\nquantization = "native"\n',
        encoding="utf-8",
    )
    recipe = value.recipes_root / "custom" / "manual.toml"
    recipe.write_text(
        '''
[runnable_recipe]
key = "test.manual"
name = "Manual"
family = "custom"
base_tool = "test.base"

[runnable_recipe.model]
resource = "model:custom:manual"

[runnable_recipe.inputs.prompt]

[runnable_recipe.optimizations]
quantization = "bf16"
''',
        encoding="utf-8",
    )
    (value.deployment_profiles_root / "manual.toml").write_text(
        '[profile]\nkey = "manual"\nname = "Manual"\nrecipes = ["test.manual"]\n',
        encoding="utf-8",
    )

    plan = build_deployment_plan(value, _registry(value), "manual")

    assert plan.locally_runnable
    assert not plan.remote_provisionable
    assert "no remote acquisition source" in " ".join(plan.warnings)


def test_recipe_and_deployment_api_surfaces(tmp_path: Path):
    value = settings(tmp_path)
    _write_model_and_lora(value)
    _write_recipe(value.recipes_root / "custom" / "one.toml", "test.one", with_lora=True)
    (value.deployment_profiles_root / "editor.toml").write_text(
        '[profile]\nkey = "editor"\nname = "Editor"\nrecipes = ["test.one"]\n',
        encoding="utf-8",
    )
    app = create_app(value, _registry(value))

    with TestClient(app) as client:
        recipes = client.get("/v1/recipes")
        profiles = client.get("/v1/deployment/profiles")
        plan = client.get("/v1/deployment/plan/editor")
        lock = client.get("/v1/deployment/lock/editor")

    assert recipes.status_code == 200
    assert recipes.json()["recipes"][0]["key"] == "test.one"
    assert profiles.status_code == 200
    assert profiles.json()["profiles"][0]["key"] == "editor"
    assert plan.status_code == 200
    assert plan.json()["remote_provisionable"] is True
    assert lock.status_code == 200
    assert lock.json()["profile_key"] == "editor"


def test_profile_catalog_reports_private_source_labels(tmp_path: Path):
    private = tmp_path.parent / "private-profiles"
    private.mkdir(parents=True)
    (private / "remote.toml").write_text(
        '[profile]\nkey = "remote"\nname = "Remote"\nrecipes = ["test.none"]\n',
        encoding="utf-8",
    )
    value = settings(tmp_path, profile_paths=(private,))

    catalog = deployment_profile_catalog(value)

    assert catalog.errors == []
    assert catalog.profiles[0].source_path == "private-1/remote.toml"
