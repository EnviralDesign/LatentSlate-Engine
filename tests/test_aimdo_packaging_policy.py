from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_engine_is_gpl3_or_later_and_preserves_dependency_notices() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["license"] == "GPL-3.0-or-later"
    assert project["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "either version 3 of the License, or (at your option) any later" in license_text
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "comfy-aimdo 0.4.15" in notices
    assert "Comfy Kitchen 0.2.31" in notices


def test_aimdo_is_pinned_only_in_supported_nvidia_extras() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    dependency = "comfy-aimdo==0.4.15"
    assert dependency in extras["nvidia-cu128"]
    assert dependency in extras["nvidia-cu130"]
    assert dependency not in extras["protocol"]


def test_aimdo_remains_lazy_and_absent_from_managed_parent() -> None:
    adapter = ROOT / "src/latentslate_engine/runtime/framework/residency/aimdo.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "comfy_aimdo" or name.startswith("comfy_aimdo.") for name in imported)
    managed = (
        ROOT / "src/latentslate_engine/runtime/ltx23_kitchen_managed.py"
    ).read_text(encoding="utf-8")
    assert "comfy_aimdo" not in managed
    bootstrap = (ROOT / "scripts/runtime_bootstrap.py").read_text(encoding="utf-8")
    assert "import comfy_aimdo" not in bootstrap
    assert 'importlib.metadata.version("comfy-aimdo")' in bootstrap


def test_comfy_policy_names_standalone_low_level_boundary() -> None:
    policy = (ROOT / "docs/COMFY_ENGINE_POLICY.md").read_text(encoding="utf-8")
    assert "standalone **comfy-aimdo**" in policy
    assert "does not use AIMDO's ComfyUI" in policy
    assert "HostBuffer primitives inside" in policy
    assert "four logical lanes" in policy
    assert "64 MiB base and 8 MiB patch prewarm" in policy
    assert "40%-of-system-RAM registration cap" in policy
    assert "model-local `VRAMBuffer`" in policy
    assert "authenticated device-directed file-slice DMA" in policy
    assert "API/managed parent never initializes or owns AIMDO" in policy
