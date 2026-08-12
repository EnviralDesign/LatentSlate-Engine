from __future__ import annotations

from latentslate_engine import cli_presentation


def test_engine_command_uses_the_bootstrap_safe_platform_prefix(monkeypatch):
    monkeypatch.setattr(cli_presentation.os, "name", "nt")
    assert (
        cli_presentation.engine_command("recipes", "list") == r".\scripts\engine.ps1 recipes list"
    )
    assert cli_presentation.bootstrap_command() == r".\scripts\bootstrap.ps1"

    monkeypatch.setattr(cli_presentation.os, "name", "posix")
    assert cli_presentation.engine_command("recipes", "list") == "./scripts/engine.sh recipes list"
    assert cli_presentation.bootstrap_command() == "./scripts/bootstrap.sh"
