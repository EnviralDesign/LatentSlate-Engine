from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "wan14-generation-tests.py"
    spec = importlib.util.spec_from_file_location("latentslate_wan14_generation_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wan14_runner_keeps_the_exact_i2v_baseline_and_lifecycle_scenarios() -> None:
    runner = load_runner()

    assert (runner.WIDTH, runner.HEIGHT, runner.FRAMES, runner.FPS, runner.STEPS) == (
        832,
        480,
        81,
        16,
        20,
    )
    assert runner.SEED == 43301611940728
    assert "i2v-sequential" in runner.SCENARIOS
    assert "cancel-recovery" in runner.SCENARIOS
    assert "changed-image" in runner.SCENARIOS
    assert runner.SCENARIOS["cancel-recovery"][0].expect_cancel is True


def test_wan14_runner_does_not_mislabel_unavailable_execution_cache_as_warm() -> None:
    runner = load_runner()

    assert (
        runner.EXECUTION_CACHE_POLICY
        == "not supported; every native Wan job runs in a disposable worker"
    )
    assert runner.SCENARIOS["i2v-sequential"][0].repeat == 3


def test_wan14_runner_leaves_generic_cache_assertions_off() -> None:
    runner = load_runner()

    command = runner.run_wan_spec.__code__.co_consts
    assert "--assert-runtime-state" not in command


def test_wan14_runner_rejects_parent_memory_that_does_not_return_to_baseline() -> None:
    runner = load_runner()

    with pytest.raises(RuntimeError, match="did not return near"):
        runner._assert_parent_memory_returned_to_baseline(
            {"host_process": {"pid": 9, "private_bytes": 100}},
            {
                "host_process": {
                    "pid": 9,
                    "private_bytes": 100 + runner.PARENT_PRIVATE_MEMORY_LEEWAY_BYTES + 1,
                }
            },
        )


def test_wan14_runner_requires_pid_and_parent_memory_proof_for_cancellation(
    monkeypatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "_process_exists", lambda _pid: False)
    host = {"pid": 17, "private_bytes": 100, "working_set_bytes": 100}
    runner.validate_wan_cancellation(
        {
            "timeout": {"final_job": {"status": "canceled"}},
            "runtime_before": {"host_process": host},
            "runtime_after": {
                "host_process": host,
                "runtimes": [
                    {
                        "runtime": "native_wan_i2v_14b_disposable_worker",
                        "active_worker": False,
                        "last_worker": {"pid": 44, "terminated": True},
                    }
                ],
            },
        }
    )
