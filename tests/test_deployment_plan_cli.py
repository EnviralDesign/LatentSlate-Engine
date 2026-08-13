from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.cli_presentation import engine_command, render_human
from latentslate_engine.config import Settings
from latentslate_engine.deployment_summary import format_deployment_plan, format_iec_bytes
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


def _wire_cli(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    registry,
) -> None:
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        "latentslate_engine.tools.default_registry",
        lambda *_args, **_kwargs: registry,
    )


def test_deployments_plan_defaults_to_human_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    _wire_cli(monkeypatch, value, registry)
    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "plan", "klein4b-image"],
    )

    engine_cli.main()

    output = capsys.readouterr().out
    assert output.startswith("Deployment plan")
    rendered = render_human(
        format_deployment_plan(build_deployment_plan(value, registry, "klein4b-image")), width=160
    )
    assert "Klein 4B image tools" in rendered
    assert "Recipes · 5" in rendered
    assert "flux2-klein-4b.text-to-image.bfl-distilled-nvfp4" in rendered
    assert "flux2-klein-4b.image-to-image.bfl-distilled-nvfp4" in rendered
    assert "flux2-klein-4b.text-to-image.comfy-distilled-fp8" in rendered
    assert "flux2-klein-4b.image-to-image.comfy-distilled-fp8" in rendered
    assert "Resources · 8 unique" in rendered
    assert "MISSING · AUTO INSTALL" in rendered
    assert "Total footprint:" in rendered
    assert "18.0 GiB" in rendered
    assert "Local runnable:" in rendered
    assert "Automatic provisioning:" in rendered
    assert "Required secrets:" in rendered
    assert engine_command("deployments", "install", "klein4b-image") in rendered
    assert not output.lstrip().startswith("{")


def test_deployments_plan_json_matches_the_existing_full_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    expected = build_deployment_plan(value, registry, "klein4b-image").model_dump(mode="json")
    _wire_cli(monkeypatch, value, registry)
    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "plan", "klein4b-image", "--json"],
    )

    engine_cli.main()

    output = capsys.readouterr().out
    assert json.loads(output) == expected
    assert output == json.dumps(expected, indent=2) + "\n"


def test_wan_style_remote_plan_has_concise_blockers_without_os_error_noise(
    tmp_path: Path,
):
    value = _settings(tmp_path)
    plan = build_deployment_plan(
        value,
        default_registry(value, emit_warnings=False),
        "wan22-14b-i2v-fp8",
    )
    noisy_recipe = plan.recipes[0].model_copy(
        update={
            "unavailable_reason": "artifact validation failed: [WinError 3] The system "
            "cannot find the path specified",
        }
    )

    output = render_human(
        format_deployment_plan(
            plan.model_copy(update={"recipes": [noisy_recipe]})
        ),
        width=160,
    )

    assert format_iec_bytes(plan.total_bytes) == "33.6 GiB"
    assert "Wan 2.2 14B I2V Stored FP8" in output
    assert "Recipes · 1" in output
    assert "required artifact could not be inspected; repair or reinstall it" in output
    assert "Resources · 5 unique" in output
    assert "MISSING · AUTO INSTALL" in output
    assert "MISSING · MANUAL STAGING" not in output
    assert "Total footprint:" in output
    assert "33.6 GiB" in output
    assert "Automatic provisioning:" in output
    assert "YES" in output
    assert "Required secrets:" in output
    assert "HF_TOKEN" not in output
    assert "Resources without an immutable automatic source:" not in output
    assert "Rerun with --json for the full structured diagnostics." not in output
    assert "WinError" not in output


def test_deployments_lock_remains_json_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    _wire_cli(monkeypatch, value, registry)
    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "lock", "klein4b-image"],
    )

    engine_cli.main()

    output = capsys.readouterr().out
    assert json.loads(output)["profile_key"] == "klein4b-image"
    assert "Deployment plan:" not in output
