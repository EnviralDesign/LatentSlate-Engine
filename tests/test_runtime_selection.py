from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from latentslate_engine.runtime_selection import (
    GpuFact,
    HardwareFacts,
    RuntimeSelectionError,
    apply_validation_fallback,
    choose_runtime,
    probe_hardware,
    read_selection_state,
    supports_kitchen_capability,
    write_selection_state,
)


def facts(
    *,
    driver: tuple[int, ...] | None = (610, 47),
    devices: tuple[GpuFact, ...] = (GpuFact("NVIDIA GeForce RTX 5080", (12, 0)),),
    python_version: tuple[int, int] = (3, 12),
) -> HardwareFacts:
    return HardwareFacts("Windows", "AMD64", python_version, driver, devices)


def test_auto_prefers_cu130_for_current_blackwell_facts():
    selection = choose_runtime("auto", facts())

    assert selection.selected_tier == "nvidia-cu130"
    assert selection.preferred_tier == "nvidia-cu130"
    assert selection.fallback_reason is None


def test_auto_uses_cu128_for_older_driver():
    selection = choose_runtime("auto", facts(driver=(572, 61)))

    assert selection.selected_tier == "nvidia-cu128"
    assert "r580" in selection.fallback_reason


def test_auto_uses_protocol_when_default_gpu_is_older_than_kitchen_floor():
    selection = choose_runtime(
        "auto",
        facts(
            devices=(
                GpuFact("NVIDIA Older GPU", (7, 0)),
                GpuFact("NVIDIA RTX 5080", (12, 0)),
            )
        ),
    )

    assert selection.selected_tier == "protocol"
    assert "SM >= 7.5" in selection.fallback_reason


def test_explicit_cu128_rejects_default_gpu_below_kitchen_floor():
    with pytest.raises(RuntimeSelectionError, match="CUDA 12.8"):
        choose_runtime("cu128", facts(devices=(GpuFact("NVIDIA Older GPU", (7, 0)),)))


def test_kitchen_capability_floor_is_7_5_for_the_actual_default_device():
    assert supports_kitchen_capability((7, 5)) is True
    assert supports_kitchen_capability((7, 0)) is False


def test_auto_uses_protocol_without_supported_nvidia_gpu():
    selection = choose_runtime("auto", facts(driver=None, devices=()))

    assert selection.selected_tier == "protocol"
    assert "No NVIDIA GPU" in selection.fallback_reason


def test_unsupported_python_stops_before_any_runtime_selection():
    with pytest.raises(RuntimeSelectionError, match="Python >=3.11,<3.13"):
        choose_runtime("auto", facts(python_version=(3, 13)))


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("cu130", "nvidia-cu130"), ("cu128", "nvidia-cu128"), ("protocol", "protocol")),
)
def test_explicit_modes_select_requested_tier_when_supported(mode: str, expected: str):
    assert choose_runtime(mode, facts()).selected_tier == expected


def test_explicit_cu130_rejects_incompatible_facts():
    with pytest.raises(RuntimeSelectionError, match="r580"):
        choose_runtime("cu130", facts(driver=(572, 61)))


def test_auto_validation_fallback_is_limited_to_cu130_backend_failures():
    selected = choose_runtime("auto", facts())
    fallback = apply_validation_fallback(
        selected,
        {
            "ok": False,
            "error_code": "kitchen_cuda_backend_unavailable",
            "message": "extension did not load",
        },
    )

    assert fallback is not None
    assert fallback.selected_tier == "nvidia-cu128"
    assert fallback.fallback_reason == "extension did not load"


@pytest.mark.parametrize(
    "payload",
    (
        {"ok": False, "error_code": "network_error", "message": "connection reset"},
        {"ok": False, "error_code": "hash_mismatch", "message": "bad wheel"},
        {"ok": False, "error_code": "torch_import_failed", "message": "corrupt install"},
        {"ok": False, "error_code": "kitchen_import_failed", "message": "corrupt extension"},
        {"ok": True},
    ),
)
def test_general_or_install_failures_never_trigger_automatic_downgrade(payload):
    assert apply_validation_fallback(choose_runtime("auto", facts()), payload) is None


def test_default_device_below_kitchen_floor_is_not_a_cu130_to_cu128_fallback():
    assert (
        apply_validation_fallback(
            choose_runtime("auto", facts()),
            {
                "ok": False,
                "error_code": "torch_device_capability_unsupported",
                "message": "device 0 is too old",
            },
        )
        is None
    )


def test_hardware_probe_parses_mocked_nvidia_smi_output():
    def mocked_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args="nvidia-smi",
            returncode=0,
            stdout="610.47, NVIDIA GeForce RTX 5080, 12.0\n",
            stderr="",
        )

    probed = probe_hardware(mocked_run)

    assert probed.nvidia_driver == (610, 47)
    assert probed.devices == (GpuFact("NVIDIA GeForce RTX 5080", (12, 0)),)
    assert choose_runtime("auto", probed).selected_tier == "nvidia-cu130"


def test_state_round_trip_is_non_secret_and_doctor_safe(tmp_path):
    selection = choose_runtime("auto", facts())
    validation = {"ok": True, "torch": {"compiled_cuda": "13.0"}}
    path = write_selection_state(tmp_path, selection, facts(), validation)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "token" not in json.dumps(payload).lower()
    recorded = read_selection_state(tmp_path)
    assert recorded["state"] == "recorded"
    assert recorded["selected_tier"] == "nvidia-cu130"
    assert recorded["validation"] == validation


def test_state_reader_accepts_windows_utf8_bom(tmp_path):
    selection = choose_runtime("auto", facts())
    path = write_selection_state(tmp_path, selection, facts(), {"ok": True})
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload, encoding="utf-8-sig")

    assert read_selection_state(tmp_path)["selected_tier"] == "nvidia-cu130"


def test_selector_runs_in_an_isolated_fresh_interpreter_without_package_imports():
    script = Path(__file__).resolve().parents[1] / "scripts" / "runtime_bootstrap.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "select", "--mode", "protocol"],
        capture_output=True,
        check=False,
        cwd=script.parent.parent,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected_tier"] == "protocol"
