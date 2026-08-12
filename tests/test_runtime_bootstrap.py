from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _bootstrap_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "runtime_bootstrap.py"
    spec = importlib.util.spec_from_file_location("test_runtime_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_actual_torch_device_capability_has_a_distinct_validation_error():
    bootstrap = _bootstrap_module()

    failure = bootstrap._device_capability_failure((7, 0))

    assert failure is not None
    assert failure["error_code"] == "torch_device_capability_unsupported"
    assert bootstrap._device_capability_failure((7, 5)) is None
