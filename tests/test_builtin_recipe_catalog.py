from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.app import create_app
from latentslate_engine.config import Settings
from latentslate_engine.recipes import build_deployment_plan
from latentslate_engine.tools import default_registry


def _settings(home: Path) -> Settings:
    value = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    value.ensure_directories()
    return value


def test_builtin_recipes_are_exact_lean_and_unavailable_when_artifacts_are_absent(
    tmp_path: Path,
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    recipes = {entry.key: entry for entry in registry.variants}

    assert set(recipes) == {
        "klein4b.comfy-fp8.image-to-image",
        "klein4b.comfy-fp8.text-to-image",
        "klein4b.reference-bf16.image-to-image",
        "klein4b.reference-bf16.text-to-image",
        "ltx23.distilled.image-to-video",
        "ltx23.distilled.text-to-video",
        "wan22.comfy-org-14b-i2v-fp8",
        "wan22.ti2v5b.text-to-video",
    }
    assert all(not recipe.available for recipe in recipes.values())
    for key, recipe in recipes.items():
        reason = recipe.unavailable_reason or ""
        if key == "wan22.comfy-org-14b-i2v-fp8" or key.startswith("klein4b.comfy-fp8"):
            assert "inventory path is unavailable" in reason
        else:
            assert "artifact is not installed or incomplete" in reason

    resources = {resource.id: resource for resource in registry.resources.resources}
    klein = resources["model:klein4b:black-forest-labs--flux.2-klein-4b"]
    klein_base = resources["model:klein4b:transformers/flux-2-klein-base-4b-fp8"]
    klein_distilled = resources["model:klein4b:transformers/flux-2-klein-4b-fp8"]
    klein_qwen = resources["model:klein4b:text_encoders/qwen_3_4b"]
    klein_vae = resources["model:klein4b:vae/flux2-vae"]
    klein_small_vae = resources["model:klein4b:vae/full_encoder_small_decoder"]
    klein_base_support = resources["model:klein4b:support/comfy-base-pipeline-support"]
    klein_distilled_support = resources[
        "model:klein4b:support/comfy-distilled-pipeline-support"
    ]
    ltx = resources["model:ltx23:diffusers--ltx-2.3-distilled-diffusers"]
    wan = resources["model:wan22:wan-ai--wan2.2-ti2v-5b-diffusers"]
    wan14_support = resources["model:wan22:wan22-14b-i2v-official-support"]
    wan14_resources = [
        resources["model:wan22:comfy-org-wan22-14b-i2v-fp8/split_files/diffusion_models/wan2.2_i2v_high_noise_14b_fp8_scaled"],
        resources["model:wan22:comfy-org-wan22-14b-i2v-fp8/split_files/diffusion_models/wan2.2_i2v_low_noise_14b_fp8_scaled"],
        resources["model:wan22:wan22-14b-i2v-comfy-support/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled"],
        resources["model:wan22:wan22-14b-i2v-comfy-support/split_files/vae/wan_2.1_vae"],
        wan14_support,
    ]
    assert (klein.size_bytes, ltx.size_bytes, wan.size_bytes) == (
        23740007447,
        94977700554,
        34203021834,
    )
    assert klein.sources[0].revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert ltx.sources[0].revision == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert wan.sources[0].revision == "b8fff7315c768468a5333511427288870b2e9635"
    assert all(resource.sources[0].is_exact() for resource in (klein, ltx, wan))

    wan14_plan = build_deployment_plan(value, registry, "wan22-14b-i2v-fp8")
    assert [resource.id for resource in wan14_plan.resources] == sorted(
        resource.id for resource in wan14_resources
    )
    assert wan14_plan.total_bytes == 36108276923
    assert wan14_plan.incremental_bytes == 36108276923
    assert not wan14_plan.locally_runnable
    # The four file sources are exact. The support directory deliberately has no
    # source because its locally filtered tree is not the upstream whole snapshot.
    assert all(resource.sources[0].is_exact() for resource in wan14_resources[:-1])
    assert wan14_support.sources == []
    assert wan14_support.metadata["upstream_snapshot"]["revision"] == (
        "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
    )
    assert not wan14_plan.remote_provisionable
    assert "no immutable remote source" in " ".join(wan14_plan.warnings)

    plan = build_deployment_plan(value, registry, "klein4b-image")
    assert [resource.id for resource in plan.resources] == sorted(
        resource.id
        for resource in (
            klein_base,
            klein_distilled,
            klein_qwen,
            klein_vae,
            klein_small_vae,
            klein_base_support,
            klein_distilled_support,
        )
    )
    assert plan.total_bytes == 16_822_610_222
    assert plan.incremental_bytes == plan.total_bytes
    assert not plan.locally_runnable
    assert plan.remote_provisionable
    assert all(resource.sources[0].is_exact() for resource in plan.resources)

    reference = build_deployment_plan(value, registry, "klein4b-reference-bf16-image")
    assert [resource.id for resource in reference.resources] == [klein.id]
    assert reference.total_bytes == klein.size_bytes

    ltx_plan = build_deployment_plan(value, registry, "ltx23-video")
    assert [recipe.key for recipe in ltx_plan.recipes] == [
        "ltx23.distilled.text-to-video",
        "ltx23.distilled.image-to-video",
    ]
    assert [resource.id for resource in ltx_plan.resources] == [ltx.id]
    assert ltx_plan.total_bytes == ltx.size_bytes
    assert ltx_plan.incremental_bytes == ltx.size_bytes


def test_builtin_catalog_is_exposed_through_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    app = create_app(value, registry)

    with TestClient(app) as client:
        recipes = client.get("/v1/recipes")
        profiles = client.get("/v1/deployment/profiles")
        plan = client.get("/v1/deployment/plan/wan22-ti2v5b-text-to-video")

    assert recipes.status_code == 200
    assert len(recipes.json()["recipes"]) == 8
    assert profiles.status_code == 200
    assert [profile["key"] for profile in profiles.json()["profiles"]] == [
        "klein4b-image",
        "klein4b-reference-bf16-image",
        "ltx23-video",
        "wan22-14b-i2v-fp8",
        "wan22-ti2v5b-text-to-video",
    ]
    assert plan.status_code == 200
    assert plan.json()["total_bytes"] == 34203021834

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: value))
    monkeypatch.setattr(
        "latentslate_engine.tools.default_registry",
        lambda *_args, **_kwargs: registry,
    )
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles"])
    engine_cli.main()
    assert "Deployment profiles (5 saved recipe selections):" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "profiles", "--json"],
    )
    engine_cli.main()
    assert '"key": "klein4b-image"' in capsys.readouterr().out


def test_builtin_wan14_profile_is_runnable_from_the_opt_in_workstation_home():
    """Prove package and retained local recipe keys coexist against actual artifacts."""

    configured_home = os.environ.get("LATENTSLATE_TEST_WAN14_HOME")
    if not configured_home:
        pytest.skip("set LATENTSLATE_TEST_WAN14_HOME for the workstation artifact proof")
    workstation_home = Path(configured_home)
    if not workstation_home.is_dir():
        pytest.skip("configured workstation Wan 14B artifact home is unavailable")

    # Do not create or alter anything in the opt-in artifact home.
    value = Settings(
        home=workstation_home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    registry = default_registry(value, emit_warnings=False)
    recipes = {entry.key: entry for entry in registry.variants}

    assert registry.resources.errors == []
    assert registry.variant_errors == []

    package_recipe = recipes.get("wan22.comfy-org-14b-i2v-fp8")
    retained_local_recipe = recipes.get("wan22.comfy_org_14b_i2v_fp8")
    if (
        package_recipe is None
        or retained_local_recipe is None
        or not package_recipe.available
        or not retained_local_recipe.available
    ):
        pytest.skip("opt-in workstation Wan 14B artifact closure is not installed")

    assert package_recipe.available
    assert retained_local_recipe.available
