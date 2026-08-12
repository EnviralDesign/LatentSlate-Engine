import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from latentslate_engine import __main__ as engine_cli
from latentslate_engine import doctor
from latentslate_engine.cli_presentation import render_human
from latentslate_engine.config import Settings
from latentslate_engine.protocol import BundleDescriptor, BundleStatus
from latentslate_engine.runtime_selection import (
    GpuFact,
    HardwareFacts,
    choose_runtime,
    write_selection_state,
)

GIB = 1024**3


def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="MiniMaxAI/MiniMax-H3",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )


def installed_bundles(
    _model_root: Path | None = None,
    _settings: Settings | None = None,
):
    return [
        BundleDescriptor(
            id="h3-basic",
            name="H3",
            source="huggingface",
            repo_id="MiniMaxAI/MiniMax-H3",
            status=BundleStatus.INSTALLED,
        ),
        BundleDescriptor(
            id="ltx23-basic",
            name="LTX 2.3",
            source="huggingface",
            repo_id="diffusers/LTX-2.3-Distilled-Diffusers",
            status=BundleStatus.INSTALLED,
        ),
        BundleDescriptor(
            id="wan22-basic",
            name="Wan 2.2",
            source="huggingface",
            repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            status=BundleStatus.INSTALLED,
        ),
        BundleDescriptor(
            id="klein4b-basic",
            name="Klein 4B",
            source="huggingface",
            repo_id="black-forest-labs/FLUX.2-klein-4B",
            status=BundleStatus.INSTALLED,
        ),
        BundleDescriptor(
            id="klein9b-basic",
            name="Klein 9B",
            source="huggingface",
            repo_id="black-forest-labs/FLUX.2-klein-9B",
            status=BundleStatus.INSTALLED,
        ),
    ]


def all_packages_ready():
    return {name: {"available": True, "version": "test"} for name in doctor._PACKAGE_PROBES}


def cuda_report(capability=(12, 0), memory_gib=16):
    metadata = doctor.capability_metadata(capability)
    return {
        "available": True,
        "torch_version": "test",
        "compiled_cuda_version": "12.8",
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA Test GPU",
                "total_memory_bytes": memory_gib * GIB,
                "total_memory_gib": float(memory_gib),
                **metadata,
            }
        ],
        "error": None,
    }


def test_doctor_report_is_serializable_and_actionable(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    report_data = cuda_report()
    report_data["devices"][0]["name"] = "NVIDIA GeForce RTX 5080"
    monkeypatch.setattr(doctor, "_cuda_report", lambda: report_data)
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is True
    assert report["model_store"]["root"] == str(tmp_path / "models")
    for family in ("h3", "ltx23", "wan22", "klein4b", "klein9b"):
        assert report["families"][family]["dependencies_ready"] is True
        assert report["families"][family]["runtime_dependencies_ready"] is True
    assert report["profiles"]["ltx23"] == "bf16_sequential_offload"
    assert report["profiles"]["wan22"] == "bf16_sequential_offload"
    assert report["profiles"]["klein4b"] == "bf16_model_offload"
    assert any(check["code"] == "h3_system_memory" for check in report["checks"])
    assert any(check["code"] == "ltx23_vram_unvalidated" for check in report["checks"])
    assert any(check["code"] == "wan22_vram_unvalidated" for check in report["checks"])
    formatted = render_human(doctor.format_report(report), width=140)
    assert "RTX 5080" in formatted
    assert "SM120" in formatted
    assert str(tmp_path / "models") in formatted
    assert "Model families · runtime prerequisites only" in formatted
    assert "READY" in formatted
    assert "bf16_model_offload" in formatted
    assert "installed" in formatted
    assert "Not inspected. Run `latentslate-engine recipes list`" in formatted
    assert "profile=" not in formatted
    assert "Runtime selection" in formatted
    assert "Comfy Kitchen" in formatted
    json.dumps(report)


def test_doctor_reports_recorded_runtime_tier_and_actual_kitchen_capabilities(
    tmp_path: Path, monkeypatch
):
    selected = choose_runtime(
        "auto",
        HardwareFacts(
            "Windows",
            "AMD64",
            (3, 12),
            (610, 47),
            (GpuFact("NVIDIA GeForce RTX 5080", (12, 0)),),
        ),
    )
    write_selection_state(
        tmp_path,
        selected,
        HardwareFacts(
            "Windows",
            "AMD64",
            (3, 12),
            (610, 47),
            (GpuFact("NVIDIA GeForce RTX 5080", (12, 0)),),
        ),
        {"ok": True, "tier": "nvidia-cu130"},
    )
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    matching_cuda = cuda_report()
    matching_cuda["compiled_cuda_version"] = "13.0"
    monkeypatch.setattr(doctor, "_cuda_report", lambda: matching_cuda)
    monkeypatch.setattr(
        doctor,
        "_kitchen_report",
        lambda _cuda: {
            "available": True,
            "version": "test",
            "backends": ["cuda", "eager"],
            "cuda_backend": True,
            "capabilities": {"fp8": True, "nvfp4": True, "mxfp8": True},
            "error": None,
        },
    )
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))
    human = render_human(doctor.format_report(report), width=140)

    assert report["runtime"]["selected_tier"] == "nvidia-cu130"
    assert report["kitchen"]["capabilities"]["nvfp4"] is True
    assert "nvidia-cu130" in human
    assert "NVFP4" in human
    assert any(check["code"] == "runtime_selection" for check in report["checks"])


@pytest.mark.parametrize(
    ("tier_mode", "compiled_cuda"),
    (("cu130", "13.0"), ("cu128", "12.8")),
)
def test_doctor_requires_kitchen_for_recorded_nvidia_tier(
    tmp_path: Path, monkeypatch, tier_mode: str, compiled_cuda: str
):
    facts = HardwareFacts(
        "Windows",
        "AMD64",
        (3, 12),
        (610, 47),
        (GpuFact("NVIDIA Test GPU", (12, 0)),),
    )
    selected = choose_runtime(tier_mode, facts)
    write_selection_state(tmp_path, selected, facts, {"ok": True})
    report_data = cuda_report()
    report_data["compiled_cuda_version"] = compiled_cuda
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: report_data)
    monkeypatch.setattr(doctor, "_kitchen_report", lambda _cuda: {"available": False, "error": "absent"})
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is False
    assert any(check["code"] == "runtime_kitchen_missing" for check in report["checks"])


def test_doctor_requires_qualified_kitchen_cuda_backend_for_cu130(tmp_path: Path, monkeypatch):
    facts = HardwareFacts(
        "Windows",
        "AMD64",
        (3, 12),
        (610, 47),
        (GpuFact("NVIDIA Test GPU", (12, 0)),),
    )
    selected = choose_runtime("cu130", facts)
    write_selection_state(tmp_path, selected, facts, {"ok": True})
    report_data = cuda_report()
    report_data["compiled_cuda_version"] = "13.0"
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: report_data)
    monkeypatch.setattr(
        doctor,
        "_kitchen_report",
        lambda _cuda: {"available": True, "cuda_backend": False, "error": None},
    )
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is False
    assert any(
        check["code"] == "runtime_kitchen_cuda_backend_missing" for check in report["checks"]
    )


@pytest.mark.parametrize(
    ("tier_mode", "compiled_cuda"),
    (("cu130", "12.8"), ("cu128", "13.0")),
)
def test_doctor_marks_recorded_nvidia_tier_drift_as_error(
    tmp_path: Path, monkeypatch, tier_mode: str, compiled_cuda: str
):
    selected = choose_runtime(tier_mode, HardwareFacts("Windows", "AMD64", (3, 12), (610, 47), (GpuFact("NVIDIA Test GPU", (12, 0)),)))
    write_selection_state(tmp_path, selected, HardwareFacts("Windows", "AMD64", (3, 12), (610, 47), (GpuFact("NVIDIA Test GPU", (12, 0)),)), {"ok": True})
    report_data = cuda_report()
    report_data["compiled_cuda_version"] = compiled_cuda
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: report_data)
    monkeypatch.setattr(doctor, "_kitchen_report", lambda _cuda: {"available": False, "error": "absent"})
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is False
    assert any(check["code"] == "runtime_tier_drift" for check in report["checks"])


def test_doctor_marks_protocol_state_with_model_runtime_as_drift(tmp_path: Path, monkeypatch):
    facts = HardwareFacts("Windows", "AMD64", (3, 12), None, ())
    selected = choose_runtime("protocol", facts)
    write_selection_state(tmp_path, selected, facts, {"ok": True})
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: cuda_report())
    monkeypatch.setattr(doctor, "_kitchen_report", lambda _cuda: {"available": False, "error": "absent"})
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is False
    assert any(check["code"] == "runtime_tier_drift" for check in report["checks"])


def test_doctor_cli_keeps_json_exact_and_human_output_plain(tmp_path: Path, monkeypatch, capsys):
    report = doctor.collect_report(settings(tmp_path))
    report["ready_for_inference"] = True
    monkeypatch.setattr(doctor, "collect_report", lambda: report)

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "doctor", "--json"])
    with pytest.raises(SystemExit) as result:
        engine_cli.main()
    assert result.value.code == 0
    assert capsys.readouterr().out == json.dumps(report, indent=2) + "\n"

    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "doctor"])
    with pytest.raises(SystemExit):
        engine_cli.main()
    human = capsys.readouterr().out
    assert "LatentSlate Engine doctor" in human
    assert "Checks" in human
    assert "\x1b" not in human


def test_doctor_does_not_advertise_removed_blackwell_conversion_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: cuda_report((8, 9), 24))
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert not any("nvfp4" in check["code"] for check in report["checks"])


def test_doctor_reports_actionable_legacy_conversion_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: cuda_report())
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)
    configured = settings(tmp_path)
    configured = replace(configured, klein_profile="consumer_nvfp4")

    report = doctor.collect_report(configured)

    assert report["ready_for_inference"] is False
    errors = [check for check in report["checks"] if check["level"] == "error"]
    assert any(
        check["code"] == "klein9b_legacy_conversion_profile"
        and "bf16_model_offload" in check["message"]
        for check in errors
    )


def test_doctor_rejects_unknown_profile_typos(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: cuda_report())
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)
    configured = replace(settings(tmp_path), ltx23_profile="bf16_sequentail_offload")

    report = doctor.collect_report(configured)

    assert report["ready_for_inference"] is False
    assert report["families"]["ltx23"]["profile_valid"] is False
    assert report["families"]["ltx23"]["dependencies_ready"] is False
    assert report["families"]["ltx23"]["runtime_dependencies_ready"] is True
    assert any(check["code"] == "ltx23_invalid_profile" for check in report["checks"])
    assert "INVALID · bf16_sequentail_offload" in render_human(
        doctor.format_report(report), width=140
    )


def test_cpu_only_torch_message_is_actionable():
    message = doctor._cuda_unavailable_error(None)

    assert "CPU-only PyTorch" in message
    assert "bootstrap" in message
    assert "uv sync" not in message


def test_missing_runtime_selection_command_is_platform_aware(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(doctor, "_cuda_report", lambda: cuda_report())
    monkeypatch.setattr(doctor, "_kitchen_report", lambda _cuda: {"available": False, "error": "absent"})
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 96 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)
    monkeypatch.setattr(doctor.os, "name", "posix")

    report = doctor.collect_report(settings(tmp_path))

    missing = next(check for check in report["checks"] if check["code"] == "runtime_selection_missing")
    assert "./scripts/bootstrap.sh" in missing["message"]


def test_doctor_fails_cleanly_without_cuda_or_runtime_packages(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        doctor,
        "_package_report",
        lambda: {name: {"available": False, "version": None} for name in doctor._PACKAGE_PROBES},
    )
    monkeypatch.setattr(
        doctor,
        "_cuda_report",
        lambda: {
            "available": False,
            "torch_version": None,
            "compiled_cuda_version": None,
            "devices": [],
            "error": "PyTorch is not installed.",
        },
    )
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: None)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: None)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: False)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["status"] == "error"
    assert report["ready_for_inference"] is False
    assert any(check["code"] == "cuda_unavailable" for check in report["checks"])
    assert any(check["code"] == "runtime_dependencies" for check in report["checks"])
