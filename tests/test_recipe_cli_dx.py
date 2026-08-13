from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.acquisition import deployment_install as installer
from latentslate_engine.cli_presentation import engine_command, render_human
from latentslate_engine.cli_product import (
    _recipe_tier,
    format_recipe_detail,
    recipe_detail_payload,
    resource_detail_payload,
)
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


def test_recipe_tier_colors_distinguish_support_from_caution():
    assert _recipe_tier(["recommended"]).style == "status.ok"
    assert _recipe_tier(["fallback"]).style == "status.ok"
    assert _recipe_tier(["reference"]).style == "identifier"
    assert _recipe_tier(["quality-alternate"]).style == ""
    assert _recipe_tier(["experimental"]).style == "status.warn"


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
    assert recipes_human.startswith("Recipes · 29")
    assert "Family" in recipes_human
    assert "Tier" in recipes_human
    assert "RECOMME" in recipes_human
    assert "FALLBAC" in recipes_human
    assert "REFEREN" in recipes_human
    assert "MISSING" in recipes_human
    assert engine_command("recipes", "show", "<recipe-key>") in recipes_human
    assert engine_command("recipes", "install", "<recipe-key>...") in recipes_human
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
    assert resources_human.startswith("Resources ·")
    assert "Acquisition" in resources_human
    assert "MISSING" in resources_human
    assert not resources_human.lstrip().startswith("{")

    expected_profiles = deployment_profile_catalog(settings).model_dump(mode="json")
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles", "--json"])
    engine_cli.main()
    assert capsys.readouterr().out == json.dumps(expected_profiles, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles"])
    engine_cli.main()
    profiles_human = capsys.readouterr().out
    assert profiles_human.startswith("Deployment profiles · 9 saved recipe selections")
    assert engine_command("deployments", "plan", "<profile-key>") in profiles_human


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
    assert human.startswith("Recipe catalog · 1 authoring error(s)")
    assert "INVALID" in human
    assert "Rerun with --json" in human

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "validate", "--json"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 1
    expected = recipe_catalog(settings, registry).model_dump(mode="json")
    assert capsys.readouterr().out == json.dumps(expected, indent=2) + "\n"


def test_show_commands_cover_existing_and_unknown_entries(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    recipe_key = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
    resource_id = "model:klein4b:text_encoders/qwen_3_4b"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "show", recipe_key])
    engine_cli.main()
    recipe_human = capsys.readouterr().out
    assert "Source" in recipe_human
    assert "flux2-klein-4b.text-to-image.comfy-distilled-fp8" in recipe_human
    assert "Execution" in recipe_human
    assert "text_encoder" in recipe_human
    assert "WinError" not in recipe_human

    monkeypatch.setattr(
        sys, "argv", ["latentslate-engine", "recipes", "show", recipe_key, "--json"]
    )
    monkeypatch.setenv("HF_TOKEN", "TOPSECRET")
    engine_cli.main()
    recipe_output = capsys.readouterr().out
    recipe_json = json.loads(recipe_output)
    assert (
        recipe_output
        == json.dumps(recipe_detail_payload(settings, registry, recipe_key), indent=2) + "\n"
    )
    assert recipe_json["identity"]["source_path"].startswith("builtin/")
    assert recipe_json["required_resources"][0]["role"] == "pipeline_support"
    assert "TOPSECRET" not in json.dumps(recipe_json)

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "show", resource_id])
    engine_cli.main()
    resource_human = capsys.readouterr().out
    assert "Artifact path" in resource_human
    assert "Required by recipes" in resource_human

    expected_resource = resource_detail_payload(registry, resource_id)
    monkeypatch.setattr(
        sys, "argv", ["latentslate-engine", "resources", "show", resource_id, "--json"]
    )
    engine_cli.main()
    assert capsys.readouterr().out == json.dumps(expected_resource, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "show", "no.such.recipe"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 2
    assert "Unknown recipe" in capsys.readouterr().err

    legacy_recipe_key_parts = ["klein4b", "comfy-fp8", "text-to-image"]
    legacy_recipe_key = ".".join(legacy_recipe_key_parts)
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "show", legacy_recipe_key])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 2
    assert "Unknown recipe" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "resources", "show", "model:none"])
    with pytest.raises(SystemExit) as exit_info:
        engine_cli.main()
    assert exit_info.value.code == 2
    assert "Unknown resource" in capsys.readouterr().err


def test_recipe_selection_dedupes_and_wan_reports_remote_provisionable(catalog):
    settings, registry = catalog
    text = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
    image = "flux2-klein-4b.image-to-image.comfy-base-fp8"
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

    wan = build_recipe_selection_plan(
        settings, registry, ["wan-2-2-14b-i2v.image-to-video.comfy-org-fp8"]
    )
    assert wan.remote_provisionable
    assert all(resource.provisionable for resource in wan.resources)


def test_recipe_plan_json_remains_the_exact_structured_payload(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    recipe_key = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
    expected = build_recipe_selection_plan(settings, registry, [recipe_key]).model_dump(mode="json")

    monkeypatch.setattr(
        sys, "argv", ["latentslate-engine", "recipes", "plan", recipe_key, "--json"]
    )
    engine_cli.main()

    assert capsys.readouterr().out == json.dumps(expected, indent=2) + "\n"


def test_recipe_selection_human_summary_hides_internal_selection_digest(catalog):
    from latentslate_engine.deployment_summary import format_recipe_selection_plan

    settings, registry = catalog
    plan = build_recipe_selection_plan(
        settings,
        registry,
        [
            "flux2-klein-4b.image-to-image.comfy-base-fp8",
            "flux2-klein-4b.text-to-image.comfy-distilled-fp8",
        ],
    )

    output = render_human(format_recipe_selection_plan(plan), width=100)

    assert output.startswith("Recipe plan")
    assert "flux2-klein-4b.image-to-image.comfy-base-fp8" in output
    assert "flux2-klein-4b.text-to-image.comfy-distilled-fp8" in output
    assert plan.profile_key not in output
    assert "MISSING RESOURCES" in output


def test_human_recipe_view_wraps_at_narrow_width_without_color_codes(catalog):
    settings, registry = catalog
    payload = recipe_detail_payload(
        settings, registry, "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
    )

    output = render_human(format_recipe_detail(payload), width=48)
    assert "\x1b" not in output
    assert "Required resources" in output
    assert "text_e" in output
    assert "ncoder" in output
    assert "builtin/klein4b" in output


def test_recipe_selection_identity_tracks_lock_relevant_catalog_changes(
    catalog, monkeypatch: pytest.MonkeyPatch
):
    settings, registry = catalog
    recipe_key = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
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
    recipe_key = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
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


def test_wan_recipe_selection_plan_declares_exact_remote_support_closure(catalog):
    settings, registry = catalog
    plan = build_recipe_selection_plan(
        settings, registry, ["wan-2-2-14b-i2v.image-to-video.comfy-org-fp8"]
    )

    assert plan.remote_provisionable
    assert len(plan.resources) == 5
    support = next(
        resource
        for resource in plan.resources
        if resource.id == "model:wan22:wan22-14b-i2v-official-support"
    )
    assert support.provisionable
    assert support.size_bytes == 529_069_044
    assert len(support.sources) == 1
    source = support.sources[0]
    assert source.repo_id == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert source.revision == "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
    assert source.allow_patterns == (
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "transformer_2/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_recipe_install_delegates_and_keeps_json_stdout_clean(
    catalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings, registry = catalog
    _wire_cli(monkeypatch, settings, registry)
    recipe_key = "flux2-klein-4b.text-to-image.comfy-distilled-fp8"
    plan = build_recipe_selection_plan(settings, registry, [recipe_key])
    result = installer.DeploymentInstallResult(
        profile_key=plan.profile_key,
        deployment_plan=plan,
        deployment_lock=build_recipe_selection_lock(settings, registry, [recipe_key]),
    )
    calls: list[tuple[str, ...]] = []

    def fake_install(_settings, _registry, keys, **_kwargs):
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
    assert captured.out == json.dumps(result.model_dump(mode="json"), indent=2) + "\n"
    assert "download progress" in captured.err

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "recipes", "install", recipe_key])
    engine_cli.main()
    assert "Recipe installation ·" in capsys.readouterr().out


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
    assert "recipes show flux2-klein-4b.text-to-image.comfy-distilled-fp8" in readme
    assert "builtin_resource_declarations" in readme
    assert "## Workflow-derived Engine authority" in recipe_docs
    assert "A deployment profile is a saved recipe selection" in recipe_docs
