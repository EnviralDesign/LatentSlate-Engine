from __future__ import annotations

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
        "klein4b.distilled.image-to-image",
        "klein4b.distilled.text-to-image",
        "ltx23.distilled.text-to-video",
        "wan22.ti2v5b.text-to-video",
    }
    assert all(not recipe.available for recipe in recipes.values())
    assert all("artifact is not installed or incomplete" in (recipe.unavailable_reason or "")
               for recipe in recipes.values())

    resources = {resource.id: resource for resource in registry.resources.resources}
    klein = resources["model:klein4b:black-forest-labs--flux.2-klein-4b"]
    ltx = resources["model:ltx23:diffusers--ltx-2.3-distilled-diffusers"]
    wan = resources["model:wan22:wan-ai--wan2.2-ti2v-5b-diffusers"]
    assert (klein.size_bytes, ltx.size_bytes, wan.size_bytes) == (
        23740007447,
        94977700554,
        34203021834,
    )
    assert klein.sources[0].revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert ltx.sources[0].revision == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert wan.sources[0].revision == "b8fff7315c768468a5333511427288870b2e9635"
    assert all(resource.sources[0].is_exact() for resource in (klein, ltx, wan))

    plan = build_deployment_plan(value, registry, "klein4b-image")
    assert [resource.id for resource in plan.resources] == [klein.id]
    assert plan.total_bytes == klein.size_bytes
    assert plan.incremental_bytes == klein.size_bytes
    assert not plan.locally_runnable
    assert plan.remote_provisionable


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
    assert len(recipes.json()["recipes"]) == 4
    assert profiles.status_code == 200
    assert [profile["key"] for profile in profiles.json()["profiles"]] == [
        "klein4b-image",
        "ltx23-text-to-video",
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
    assert '"key": "klein4b-image"' in capsys.readouterr().out
