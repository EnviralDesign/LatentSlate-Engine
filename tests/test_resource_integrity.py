from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.protocol import (
    InputType,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.recipes import build_deployment_lock, build_deployment_plan
from latentslate_engine.resources import ResourceSource, discover_resources
from latentslate_engine.tools import ToolRegistry
from latentslate_engine.tools.base import ExecutionCapabilities, Tool, ToolContext
from latentslate_engine.variants import load_variant_tools

PINNED_HF_REVISION = "0123456789abcdef0123456789abcdef01234567"


class IntegrityTool(Tool):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=UUID("f76bb6a5-426c-4a3c-93d0-610f6d59fc31"),
            key="test.integrity-base",
            schema_revision=1,
            name="Integrity base",
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
            model_formats=frozenset({"diffusers", "safetensors"}),
            quantization_modes=frozenset({"bf16"}),
        )


def _settings(tmp_path: Path) -> Settings:
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


def _write_recipe_and_profile(value: Settings, resource_id: str, key: str) -> None:
    recipe = value.recipes_root / "custom" / f"{key}.toml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(
        f'''
[runnable_recipe]
key = "{key}"
name = "{key}"
family = "custom"
base_tool = "test.integrity-base"

[runnable_recipe.model]
resource = "{resource_id}"

[runnable_recipe.inputs.prompt]

[runnable_recipe.optimizations]
quantization = "bf16"
''',
        encoding="utf-8",
    )
    (value.deployment_profiles_root / f"{key}.toml").write_text(
        f'[profile]\nkey = "{key}"\nname = "{key}"\nrecipes = ["{key}"]\n',
        encoding="utf-8",
    )


def _write_file_declaration(
    value: Settings,
    *,
    resource_id: str,
    source: str,
) -> None:
    (value.resource_declarations_root / f"{resource_id.rsplit(':', 1)[-1]}.toml").write_text(
        f'''
[resource]
id = "{resource_id}"
kind = "model"
family = "custom"
name = "Integrity file model"
relative_path = "models/custom/file.safetensors"
format = "safetensors"
precision = "bf16"
quantization = "native"
size_bytes = 4

[[resource.sources]]
{source}
''',
        encoding="utf-8",
    )


def _registry(value: Settings) -> ToolRegistry:
    base = IntegrityTool()
    inventory = discover_resources(value)
    loaded = load_variant_tools(value, [base], inventory)
    assert loaded.errors == []
    return ToolRegistry(
        [base, *loaded.tools],
        resources=inventory,
        variants=loaded.entries,
        variant_errors=loaded.errors,
    )


def _write_diffusers_declaration(
    value: Settings,
    *,
    resource_id: str,
    relative_path: str,
    size_bytes: int,
) -> None:
    (value.resource_declarations_root / f"{resource_id.rsplit(':', 1)[-1]}.toml").write_text(
        f'''
[resource]
id = "{resource_id}"
kind = "model"
family = "custom"
name = "Integrity model"
relative_path = "{relative_path}"
format = "diffusers"
precision = "bf16"
quantization = "native"
size_bytes = {size_bytes}

[[resource.sources]]
type = "huggingface"
repo_id = "example/integrity-model"
revision = "{PINNED_HF_REVISION}"
''',
        encoding="utf-8",
    )


def _write_builtin_declaration(root: Path, *, resource_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "declared.toml").write_text(
        f'''
[resource]
id = "{resource_id}"
kind = "model"
family = "custom"
name = "Built-in unavailable model"
relative_path = "models/custom/builtin-unavailable"
format = "diffusers"
precision = "bf16"
quantization = "native"
size_bytes = 1

[[resource.sources]]
type = "huggingface"
repo_id = "example/builtin-unavailable"
revision = "{PINNED_HF_REVISION}"
''',
        encoding="utf-8",
    )


def test_builtin_declarations_load_and_keep_missing_artifacts_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = _settings(tmp_path)
    builtin_root = tmp_path / "package-data" / "builtin_resource_declarations"
    resource_id = "model:custom:builtin-unavailable"
    _write_builtin_declaration(builtin_root, resource_id=resource_id)
    monkeypatch.setattr(
        Settings,
        "builtin_resource_declarations_root",
        property(lambda _settings: builtin_root),
    )

    inventory = discover_resources(value)

    assert inventory.errors == []
    resource = inventory.resolve(resource_id)
    assert resource.name == "Built-in unavailable model"
    assert not resource.available
    assert resource.unavailable_reason == "resource artifact is not installed or incomplete"


def test_local_declaration_overrides_compatible_builtin_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = _settings(tmp_path)
    builtin_root = tmp_path / "package-data" / "builtin_resource_declarations"
    resource_id = "model:custom:builtin-unavailable"
    _write_builtin_declaration(builtin_root, resource_id=resource_id)
    monkeypatch.setattr(
        Settings,
        "builtin_resource_declarations_root",
        property(lambda _settings: builtin_root),
    )
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/builtin-unavailable",
        size_bytes=1,
    )

    inventory = discover_resources(value)

    assert inventory.errors == []
    resource = inventory.resolve(resource_id)
    assert resource.name == "Integrity model"
    assert not resource.available
    assert resource.unavailable_reason == "resource artifact is not installed or incomplete"


@pytest.mark.parametrize(
    "suffix",
    [
        "?authToken=SUPERSECRET",
        "?accessToken=SUPERSECRET",
        "?clientSecret=SUPERSECRET",
        "?privateKey=SUPERSECRET",
        "?accessKey=SUPERSECRET",
        "?download=1",
        "#token=SUPERSECRET",
    ],
)
def test_resource_source_urls_reject_all_queries_and_fragments(suffix: str):
    with pytest.raises(ValueError) as exc_info:
        ResourceSource(
            type="civitai",
            url=f"https://civitai.com/api/download/models/123{suffix}",
            sha256="a" * 64,
        )

    assert "SUPERSECRET" not in str(exc_info.value)
    assert "query strings" in str(exc_info.value) or "fragments" in str(exc_info.value)


def test_zero_size_missing_exact_resource_is_not_lockable(tmp_path: Path):
    value = _settings(tmp_path)
    resource_id = "model:custom:zero-size"
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/zero-size",
        size_bytes=0,
    )
    _write_recipe_and_profile(value, resource_id, "test.zero-size")

    registry = _registry(value)
    plan = build_deployment_plan(value, registry, "test.zero-size")

    assert registry.resources.errors == []
    assert not plan.resources[0].installed
    assert not plan.resources[0].provisionable
    assert not plan.remote_provisionable
    assert "positive size_bytes" in " ".join(plan.warnings)
    with pytest.raises(ValueError, match="resources without positive declared size"):
        build_deployment_lock(value, registry, "test.zero-size")


def test_truncated_declared_diffusers_resource_is_not_installed(tmp_path: Path):
    value = _settings(tmp_path)
    resource_id = "model:custom:truncated"
    model = value.model_root / "custom" / "truncated"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"x")
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/truncated",
        size_bytes=128,
    )
    _write_recipe_and_profile(value, resource_id, "test.truncated")

    registry = _registry(value)
    plan = build_deployment_plan(value, registry, "test.truncated")

    assert not registry.resources.is_installed(resource_id)
    assert not plan.resources[0].installed
    assert plan.resources[0].size_bytes == 128
    assert plan.incremental_bytes == 128
    assert plan.remote_provisionable


@pytest.mark.parametrize("second_shard", [None, b""])
def test_indexed_declared_resource_requires_every_nonempty_shard(
    tmp_path: Path,
    second_shard: bytes | None,
):
    value = _settings(tmp_path)
    resource_id = "model:custom:sharded"
    model = value.model_root / "custom" / "sharded"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    shard_one = "weights-00001-of-00002.safetensors"
    shard_two = "weights-00002-of-00002.safetensors"
    (model / shard_one).write_bytes(b"one")
    if second_shard is not None:
        (model / shard_two).write_bytes(second_shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.one": shard_one,
                    "layer.two": shard_two,
                }
            }
        ),
        encoding="utf-8",
    )
    declared_size = sum(path.stat().st_size for path in model.rglob("*") if path.is_file())
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/sharded",
        size_bytes=declared_size,
    )
    _write_recipe_and_profile(value, resource_id, "test.sharded")

    registry = _registry(value)
    plan = build_deployment_plan(value, registry, "test.sharded")

    assert not registry.resources.is_installed(resource_id)
    assert not plan.resources[0].installed
    assert plan.incremental_bytes == declared_size
    assert plan.remote_provisionable


def test_unindexed_numbered_shard_series_requires_every_nonempty_part(tmp_path: Path):
    value = _settings(tmp_path)
    resource_id = "model:custom:unindexed-sharded"
    model = value.model_root / "custom" / "unindexed-sharded"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "weights-00001-of-00002.safetensors").write_bytes(b"one")
    declared_size = sum(path.stat().st_size for path in model.rglob("*") if path.is_file())
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/unindexed-sharded",
        size_bytes=declared_size,
    )
    _write_recipe_and_profile(value, resource_id, "test.unindexed-sharded")

    registry = _registry(value)
    plan = build_deployment_plan(value, registry, "test.unindexed-sharded")

    assert not registry.resources.is_installed(resource_id)
    assert not plan.resources[0].installed


def test_shared_resource_is_verified_once_per_deployment_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = _settings(tmp_path)
    resource_id = "model:custom:shared"
    _write_diffusers_declaration(
        value,
        resource_id=resource_id,
        relative_path="models/custom/shared",
        size_bytes=1,
    )
    _write_recipe_and_profile(value, resource_id, "test.shared-one")
    _write_recipe_and_profile(value, resource_id, "test.shared-two")
    (value.deployment_profiles_root / "shared.toml").write_text(
        '''[profile]
key = "shared"
name = "shared"
recipes = ["test.shared-one", "test.shared-two"]
''',
        encoding="utf-8",
    )
    registry = _registry(value)
    calls = 0
    inventory_type = type(registry.resources)
    original = inventory_type.is_installed

    def counted(self: object, resource_id: str) -> bool:
        nonlocal calls
        calls += 1
        return original(self, resource_id)

    monkeypatch.setattr(inventory_type, "is_installed", counted)
    plan = build_deployment_plan(value, registry, "shared")

    assert [resource.id for resource in plan.resources] == [resource_id]
    assert calls == 1


def test_file_resource_rejects_snapshot_without_file_selector(tmp_path: Path):
    value = _settings(tmp_path)
    _write_file_declaration(
        value,
        resource_id="model:custom:file-source-shape",
        source=f'''type = "huggingface"
repo_id = "example/file"
revision = "{PINNED_HF_REVISION}"''',
    )

    inventory = discover_resources(value)

    assert len(inventory.errors) == 1
    assert "file resources require a file selector" in inventory.errors[0]


def test_directory_resource_rejects_single_file_hash_source(tmp_path: Path):
    value = _settings(tmp_path)
    _write_diffusers_declaration(
        value,
        resource_id="model:custom:directory-source-shape",
        relative_path="models/custom/directory-source-shape",
        size_bytes=1,
    )
    declaration = value.resource_declarations_root / "directory-source-shape.toml"
    declaration.write_text(
        declaration.read_text(encoding="utf-8")
        + f'''\nfilename = "model.safetensors"\nsha256 = "{'a' * 64}"\n''',
        encoding="utf-8",
    )

    inventory = discover_resources(value)

    assert len(inventory.errors) == 1
    assert "directory resources cannot declare a single-file sha256" in inventory.errors[0]


def test_declared_file_sha256_must_match_installed_bytes(tmp_path: Path):
    value = _settings(tmp_path)
    resource_id = "model:custom:hashed"
    model = value.model_root / "custom" / "hashed.safetensors"
    model.write_bytes(b"evil")
    expected_hash = hashlib.sha256(b"good").hexdigest()
    (value.resource_declarations_root / "hashed.toml").write_text(
        f'''
[resource]
id = "{resource_id}"
kind = "model"
family = "custom"
name = "Hashed model"
relative_path = "models/custom/hashed.safetensors"
format = "safetensors"
precision = "bf16"
quantization = "native"
size_bytes = 4

[[resource.sources]]
type = "huggingface"
repo_id = "example/hashed"
filename = "hashed.safetensors"
sha256 = "{expected_hash}"
''',
        encoding="utf-8",
    )
    _write_recipe_and_profile(value, resource_id, "test.hashed")

    registry = _registry(value)
    plan = build_deployment_plan(value, registry, "test.hashed")

    assert not registry.resources.is_installed(resource_id)
    assert not plan.resources[0].installed
    assert plan.incremental_bytes == 4
    assert plan.remote_provisionable
