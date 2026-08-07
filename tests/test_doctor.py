import json
from pathlib import Path

from latentslate_engine import doctor
from latentslate_engine.config import Settings
from latentslate_engine.protocol import BundleDescriptor, BundleStatus


GIB = 1024**3


def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="MiniMaxAI/MiniMax-H3",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )


def installed_bundles():
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
            repo_id="Lightricks/LTX-2.3",
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
    return {
        name: {"available": True, "version": "test"}
        for name in doctor._PACKAGE_PROBES
    }


def test_doctor_report_is_serializable_and_actionable(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(doctor, "_package_report", all_packages_ready)
    monkeypatch.setattr(
        doctor,
        "_cuda_report",
        lambda: {
            "available": True,
            "torch_version": "test",
            "compiled_cuda_version": "12.8",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 5080",
                    "capability": [12, 0],
                    "total_memory_bytes": 16 * GIB,
                    "total_memory_gib": 16.0,
                }
            ],
            "error": None,
        },
    )
    monkeypatch.setattr(doctor, "_system_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(doctor, "_disk_free_bytes", lambda _: 100 * GIB)
    monkeypatch.setattr(doctor, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(doctor, "bundle_descriptors", installed_bundles)

    report = doctor.collect_report(settings(tmp_path))

    assert report["ready_for_inference"] is True
    for family in ("h3", "ltx23", "wan22", "klein4b", "klein9b"):
        assert report["families"][family]["dependencies_ready"] is True
    assert report["profiles"]["ltx23"] == "bf16_sequential_offload"
    assert report["profiles"]["wan22"] == "bf16_sequential_offload"
    assert report["profiles"]["klein4b"] == "bf16_model_offload"
    assert any(check["code"] == "h3_system_memory" for check in report["checks"])
    assert any(
        check["code"] == "ltx23_vram_unvalidated" for check in report["checks"]
    )
    assert any(
        check["code"] == "wan22_vram_unvalidated" for check in report["checks"]
    )
    assert "RTX 5080" in doctor.format_report(report)
    json.dumps(report)


def test_doctor_fails_cleanly_without_cuda_or_runtime_packages(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        doctor,
        "_package_report",
        lambda: {
            name: {"available": False, "version": None}
            for name in doctor._PACKAGE_PROBES
        },
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
