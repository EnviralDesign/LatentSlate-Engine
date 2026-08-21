from __future__ import annotations

from latentslate_engine.config import Settings


def test_execution_device_uses_neutral_setting_with_exact_source(monkeypatch):
    monkeypatch.setenv("LATENTSLATE_EXECUTION_DEVICE", "cuda:2")
    monkeypatch.setenv("LATENTSLATE_WAN22_DEVICE", "cuda:1")

    settings = Settings.from_env()

    assert settings.execution_device == "cuda:2"
    assert settings.execution_device_source == "LATENTSLATE_EXECUTION_DEVICE"
    assert settings.wan22_device == "cuda:1"


def test_execution_device_accepts_wan_compatibility_alias(monkeypatch):
    monkeypatch.delenv("LATENTSLATE_EXECUTION_DEVICE", raising=False)
    monkeypatch.setenv("LATENTSLATE_WAN22_DEVICE", "cuda:3")

    settings = Settings.from_env()

    assert settings.execution_device == "cuda:3"
    assert settings.execution_device_source == "LATENTSLATE_WAN22_DEVICE (compatibility alias)"


def test_execution_device_reports_default_source(monkeypatch):
    monkeypatch.delenv("LATENTSLATE_EXECUTION_DEVICE", raising=False)
    monkeypatch.delenv("LATENTSLATE_WAN22_DEVICE", raising=False)

    settings = Settings.from_env()

    assert settings.execution_device == "cuda"
    assert settings.execution_device_source == "default"
