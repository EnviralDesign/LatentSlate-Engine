from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_scenarios() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "wan5-generation-tests.py"
    spec = importlib.util.spec_from_file_location("latentslate_wan5_scenarios", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manual_scenarios_cover_required_public_hardware_lifecycle() -> None:
    scenarios = load_scenarios()
    assert set(scenarios.SCENARIOS) == {
        "t2v-single",
        "i2v-single",
        "t2v-warm",
        "i2v-warm",
        "switch",
        "cancel-recovery",
        "lora-control",
    }
    assert scenarios.SCENARIOS["t2v-warm"][0].repeat == 4
    assert scenarios.SCENARIOS["i2v-warm"][0].repeat == 4
    assert [item.recipe for item in scenarios.SCENARIOS["switch"]] == [
        scenarios.T2V,
        scenarios.I2V,
        scenarios.T2V,
    ]
    assert scenarios.SCENARIOS["cancel-recovery"][0].expect_cancel is True
    assert scenarios.SCENARIOS["lora-control"][1].lora == scenarios.CRUSH_LORA


def test_deterministic_default_source_is_byte_stable(tmp_path: Path) -> None:
    scenarios = load_scenarios()
    first = scenarios.deterministic_source(tmp_path / "first.png")
    second = scenarios.deterministic_source(tmp_path / "second.png")
    assert scenarios.file_sha256(first) == scenarios.file_sha256(second)
