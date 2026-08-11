from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.acquisition import deployment_install as installer
from latentslate_engine.cli_product import resource_detail_payload
from latentslate_engine.config import Settings
from latentslate_engine.recipes import (
    build_recipe_selection_lock,
    build_recipe_selection_plan,
    deployment_profile_catalog,
    recipe_catalog,
)
from latentslate_engine.tools import default_registry


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("recipe-cli")
    settings = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    return settings, default_registry(settings, emit_warnings=False)


def _wire_cli(monkeypatch: pytest.MonkeyPatch, settings: Settings, registry) -> None:
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        "latentslate_engine.tools.default_registry",
        lambda *_args, **_kwargs: registry,
    )


def test_catalog_json_is_backward_equivalent_and_human_by_default(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)

    expected_recipes = recipe_catalog(settings, registry).model_dump(mode="json")
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "list", "--json"])
    engine_cli.main()
    assert capsys.readouterr().out == json.dumps(expected_recipes, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "list"])
    engine_cli.main()
    recipes_human = capsys.readouterr().out
    assert recipes_human.startswith("Recipes (8):")
    assert "Inspect: uv run latentslate-engine recipes show <recipe-key>" in recipes_human
    assert not recipes_human.lstrip().startswith("{")

    expected_resources = {
        "resources": [
            resource.model_dump(mode="json") for resource in registry.resources.resources
        ],
        "errors": registry.resources.errors,
    }
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "list", "--json"])
    engine_cli.main()
    assert capsys.readouterr().out == json.dumps(expected_resources, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "list"])
    engine_cli.main()
    resources_human = capsys.readouterr().out
    assert resources_human.startswith("Resources (")
    assert "automatic install" in resources_human
    assert not resources_human.lstrip().startswith("{")

    expected_profiles = deployment_profile_catalog(settings).model_dump(mode="json")
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles", "--json"])
    engine_cli.main()
    assert capsys.readouterr().out == json.dumps(expected_profiles, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles"])
    engine_cli.main()
    profiles_human = capsys.readouterr().out
    assert profiles_human.startswith("Deployment profiles (5 saved recipe selections):")
    assert "Plan: uv run latentslate-engine deployments plan <profile-key>" in profiles_human


def test_recipe_validate_preserves_failure_exit_and_json_catalog(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    monkeypatch.setattr(registry, "variant_errors", ["broken local recipe"])

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "validate"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 1
    human = capsys.readouterr().out
    assert human.startswith("Recipe catalog invalid: 1 authoring error(s).")
    assert "Details: rerun with --json" in human

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "validate", "--json"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["broken local recipe"]


def test_show_commands_cover_existing_and_unknown_entries(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    recipe_key = "klein4b.comfy-fp8.text-to-image"
    resource_id = "model:klein4b:text_encoders/qwen_3_4b"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "show", recipe_key])
    engine_cli.main()
    recipe_human = capsys.readouterr().out
    assert "Source: builtin/klein4b/comfy-fp8-text-to-image.toml" in recipe_human
    assert "Execution:" in recipe_human
    assert "text_encoder:" in recipe_human
    assert "WinError" not in recipe_human

    monkeypatch.setattr(
        sys, "argv", ["latentslate-engine", "recipes", "show", recipe_key, "--json"]
    )
    monkeypatch.setenv("HF_TOKEN", "TOPSECRET")
    engine_cli.main()
    recipe_json = json.loads(capsys.readouterr().out)
    assert recipe_json["identity"]["source_path"].startswith("builtin/")
    assert recipe_json["required_resources"][0]["role"] == "pipeline_support"
    assert "TOPSECRET" not in json.dumps(recipe_json)

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "show", resource_id])
    engine_cli.main()
    resource_human = capsys.readouterr().out
    assert "Artifact path:" in resource_human
    assert "Required by recipes:" in resource_human

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "show", "no.such.recipe"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 2
    assert "Unknown recipe" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "show", "model:none"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 2
    assert "Unknown resource" in capsys.readouterr().err


def test_recipe_selection_dedupes_and_wan_reports_nonprovisionable(catalog):
    settings, registry = catalog
    text = "klein4b.comfy-fp8.text-to-image"
    image = "klein4b.comfy-fp8.image-to-image"
    text_plan = build_recipe_selection_plan(settings, registry, [text])
    image_plan = build_recipe_selection_plan(settings, registry, [image])
    combined = build_recipe_selection_plan(settings, registry, [image, text])

    assert combined.profile_key.startswith("recipes-")
    assert len(combined.profile_key) == len("recipes-") + 64
    assert (
        combined.profile_key
        == build_recipe_selection_plan(settings, registry, [text, image]).profile_key
    )
    assert len(combined.resources) < len(text_plan.resources) + len(image_plan.resources)
    assert combined.total_bytes < text_plan.total_bytes + image_plan.total_bytes
    assert combined.remote_provisionable
    lock = build_recipe_selection_lock(settings, registry, [image, text])
    assert lock.profile_key == combined.profile_key
    assert len(lock.recipes) == 2

    wan = build_recipe_selection_plan(settings, registry, ["wan22.comfy-org-14b-i2v-fp8"])
    assert not wan.remote_provisionable
    assert any(not resource.provisionable for resource in wan.resources)


def test_recipe_selection_human_summary_hides_internal_selection_digest(catalog):
    from latentslate_engine.deployment_summary import format_recipe_selection_plan

    settings, registry = catalog
    plan = build_recipe_selection_plan(
        settings,
        registry,
        [
            "klein4b.comfy-fp8.image-to-image",
            "klein4b.comfy-fp8.text-to-image",
        ],
    )

    output = format_recipe_selection_plan(plan)

    assert output.startswith(
        "Recipe plan: klein4b.comfy-fp8.image-to-image, "
        "klein4b.comfy-fp8.text-to-image"
    )
    assert plan.profile_key not in output
    assert "required resources are missing or incomplete" in output


def test_recipe_selection_identity_tracks_lock_relevant_catalog_changes(
    catalog, monkeypatch: pytest.MonkeyPatch
):
    settings, registry = catalog
    recipe_key = "klein4b.comfy-fp8.text-to-image"
    resource_id = "model:klein4b:text_encoders/qwen_3_4b"
    monkeypatch.setenv("HF_TOKEN", "TOPSECRET")
    baseline = build_recipe_selection_plan(settings, registry, [recipe_key])
    assert "TOPSECRET" not in baseline.profile_key

    entry = next(item for item in registry.variants if item.key == recipe_key)
    with monkeypatch.context() as patch:
        patch.setattr(
            registry,
            "variants",
            [
                item.model_copy(update={"id": UUID("5b73246d-34ea-4450-8184-62374b34cfee")})
                if item.key == recipe_key
                else item
                for item in registry.variants
            ],
        )
        assert (
            build_recipe_selection_plan(settings, registry, [recipe_key]).profile_key
            != baseline.profile_key
        )

    recipe_tool = next(
        tool for tool in registry._tools.values() if tool.descriptor.key == recipe_key
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            recipe_tool,
            "_descriptor",
            recipe_tool.descriptor.model_copy(update={"schema_hash": "sha256:" + "f" * 64}),
        )
        assert (
            build_recipe_selection_plan(settings, registry, [recipe_key]).profile_key
            != baseline.profile_key
        )

    resource = registry.resources.by_id()[resource_id]
    changed_source = resource.sources[0].model_copy(update={"revision": "f" * 40})
    with monkeypatch.context() as patch:
        patch.setattr(
            registry.resources,
            "resources",
            [
                item.model_copy(update={"sources": [changed_source, *item.sources[1:]]})
                if item.id == resource_id
                else item
                for item in registry.resources.resources
            ],
        )
        assert (
            build_recipe_selection_plan(settings, registry, [recipe_key]).profile_key
            != baseline.profile_key
        )

    assert entry.id != UUID("5b73246d-34ea-4450-8184-62374b34cfee")


@pytest.mark.parametrize("alias_field", ["id", "relative_path", "name"])
def test_resource_show_resolves_every_supported_recipe_resource_alias(
    catalog, monkeypatch: pytest.MonkeyPatch, alias_field: str
):
    _settings, registry = catalog
    recipe_key = "klein4b.comfy-fp8.text-to-image"
    resource = registry.resources.by_id()["model:klein4b:text_encoders/qwen_3_4b"]
    alias = getattr(resource, alias_field)
    original_entry = next(item for item in registry.variants if item.key == recipe_key)
    aliased_entry = original_entry.model_copy(
        update={"recipe_resources": {"text_encoder": alias}, "fixed_resources": [alias]}
    )
    monkeypatch.setattr(
        registry,
        "variants",
        [aliased_entry if item.key == recipe_key else item for item in registry.variants],
    )

    payload = resource_detail_payload(registry, resource.id)

    assert {"recipe_key": recipe_key, "roles": ["text_encoder"]} in payload["referenced_by"]


def test_unprovisionable_recipe_selection_refuses_before_network(
    catalog, monkeypatch: pytest.MonkeyPatch
):
    settings, registry = catalog
    monkeypatch.setattr(
        installer, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called")
    )
    with pytest.raises(installer.DeploymentInstallError, match="not remotely provisionable"):
        installer.install_recipe_selection(settings, registry, ["wan22.comfy-org-14b-i2v-fp8"])


def test_recipe_install_delegates_and_keeps_json_stdout_clean(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    recipe_key = "klein4b.comfy-fp8.text-to-image"
    plan = build_recipe_selection_plan(settings, registry, [recipe_key])
    result = installer.DeploymentInstallResult(
        profile_key=plan.profile_key,
        deployment_plan=plan,
        deployment_lock=build_recipe_selection_lock(settings, registry, [recipe_key]),
    )
    calls: list[tuple[str, ...]] = []

    def fake_install(_settings, _registry, keys):
        calls.append(tuple(keys))
        print("download progress")
        return result

    monkeypatch.setattr(installer, "install_recipe_selection", fake_install)
    monkeypatch.setattr(
        sys, "argv", ["latentslate-engine", "recipes", "install", recipe_key, "--json"]
    )
    engine_cli.main()
    captured = capsys.readouterr()
    assert calls == [(recipe_key,)]
    assert json.loads(captured.out) == result.model_dump(mode="json")
    assert "download progress" in captured.err

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "install", recipe_key])
    engine_cli.main()
    assert "Recipe installation:" in capsys.readouterr().out


def test_help_docs_and_product_view_stay_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import latentslate_engine.cli_product; "
                "assert not {'torch', 'diffusers', 'transformers'} & set(sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    readme = Path("README.md").read_text(encoding="utf-8")
    recipe_docs = Path("docs/RECIPES.md").read_text(encoding="utf-8")
    assert "recipes show klein4b.comfy-fp8.text-to-image" in readme
    assert "builtin_resource_declarations" in readme
    assert "[runnable_recipe]" in recipe_docs
    assert "saved reusable recipe selection" in recipe_docs
