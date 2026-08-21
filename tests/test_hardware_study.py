from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_harness() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "hardware-study.py"
    spec = importlib.util.spec_from_file_location("latentslate_hardware_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_timing_separates_queue_execution_and_total() -> None:
    harness = load_harness()

    assert harness.server_timing(
        {
            "created_at": "2026-08-12T12:00:00Z",
            "started_at": "2026-08-12T12:00:01Z",
            "completed_at": "2026-08-12T12:00:04.5Z",
        }
    ) == {
        "server_queue_seconds": 1.0,
        "server_execution_seconds": 3.5,
        "server_total_seconds": 4.5,
    }


def test_observed_measurement_state_distinguishes_cold_and_cache_warm() -> None:
    harness = load_harness()

    cold = harness.observed_measurement_state(
        {
            "provenance": {
                "runtime_result": {
                    "pipeline_warm": False,
                    "cache": {
                        "prompt_hit": False,
                        "reference_hits": 0,
                        "reference_misses": 1,
                    },
                }
            }
        }
    )
    warm = harness.observed_measurement_state(
        {
            "provenance": {
                "runtime_result": {
                    "pipeline_warm": True,
                    "cache": {
                        "prompt_hit": True,
                        "reference_hits": 1,
                        "reference_misses": 0,
                    },
                }
            }
        }
    )

    assert cold["classification"] == "pipeline_cold"
    assert warm["classification"] == "pipeline_warm_cache_warm"


def test_runtime_state_assertion_accepts_runtime_without_optional_cache_signals() -> None:
    harness = load_harness()

    for expected_warm in (False, True):
        observed = {
            "pipeline_warm": expected_warm,
            "prompt_hit": None,
            "reference_hits": None,
            "reference_misses": None,
        }
        harness.assert_observed_runtime_state(
            observed,
            recipe_key="z-image-turbo.text-to-image.comfy-int8-convrot",
            repeat_index=2 if expected_warm else 1,
            expected_warm=expected_warm,
            has_media_inputs=False,
        )

    warm = harness.observed_measurement_state(
        {"provenance": {"runtime_result": {"pipeline_warm": True}}}
    )
    assert warm["classification"] == "pipeline_warm_cache_warm"


def test_runtime_state_assertion_respects_explicitly_unsupported_prompt_cache() -> None:
    harness = load_harness()
    support = {"prompt": False, "media": False}
    job = {
        "provenance": {
            "runtime_result": {
                "pipeline_warm": True,
                "cache": {"prompt_hit": False, "media_hit": False},
            }
        }
    }

    observed = harness.observed_measurement_state(job, support)
    harness.assert_observed_runtime_state(
        observed,
        recipe_key="ltx-2-3.text-to-video.kitchen-dev-fp8",
        repeat_index=2,
        expected_warm=True,
        has_media_inputs=False,
        cache_support=support,
    )

    assert observed["classification"] == "pipeline_warm_cache_warm"


def test_active_runtime_cache_support_selects_active_wrapper() -> None:
    harness = load_harness()

    assert harness.active_runtime_cache_support(
        {
            "runtimes": [
                {"active": False, "cache_support": {"prompt": True}},
                {"active": True, "cache_support": {"prompt": False, "media": False}},
            ]
        }
    ) == {"prompt": False, "media": False}


def test_measurement_summary_uses_expected_lifecycle_and_hashes() -> None:
    harness = load_harness()
    base_job = {"status": "succeeded"}
    runs = [
        {
            "recipe_key": "recipe.a",
            "expected_state": "runtime_cold",
            "job": base_job,
            "timing": {
                "server_execution_seconds": 12.0,
                "client_total_seconds": 13.0,
            },
            "artifacts": [{"download": {"sha256": "same"}}],
        },
        *[
            {
                "recipe_key": "recipe.a",
                "expected_state": "pipeline_warm_cache_warm",
                "job": base_job,
                "timing": {
                    "server_execution_seconds": value,
                    "client_total_seconds": value + 1,
                },
                "artifacts": [{"download": {"sha256": "same"}}],
            }
            for value in (4.0, 5.0, 6.0)
        ],
    ]

    summary = harness.measurement_summary(runs)[0]

    assert summary["runtime_cold"]["server_execution_seconds"]["mean"] == 12.0
    assert summary["pipeline_warm_cache_warm"]["server_execution_seconds"] == {
        "count": 3,
        "minimum": 4.0,
        "maximum": 6.0,
        "mean": 5.0,
        "median": 5.0,
        "sample_stdev": 1.0,
    }
    assert summary["byte_deterministic_within_recipe"] is True


def test_progress_trigger_waits_for_matching_running_message_before_cancel(tmp_path: Path) -> None:
    harness = load_harness()

    class Client:
        def __init__(self) -> None:
            self.gets = iter(
                (
                    {"status": "queued", "progress": 0.0, "message": "Queued"},
                    {"status": "running", "progress": 0.10, "message": "Loading model"},
                    {
                        "status": "running",
                        "progress": 0.225,
                        "message": "Generating synchronized video and audio (1/8)",
                    },
                    {"status": "canceled", "progress": 0.225, "message": "Canceled"},
                )
            )
            self.deletes = 0

        def request_json(self, method: str, _path: str):
            if method == "GET":
                return next(self.gets)
            assert method == "DELETE"
            self.deletes += 1
            return {"accepted": True}

    client = Client()
    with pytest.raises(harness.StudyTimeoutError) as error:
        harness.await_job(
            client,
            "job-1",
            timeout=60.0,
            poll_interval=0.0,
            cancellation_grace=1.0,
            events_path=tmp_path / "events.jsonl",
            cancel_after_message_prefix="Generating synchronized video and audio (",
        )

    assert client.deletes == 1
    assert error.value.final_job["status"] == "canceled"
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"trigger":"message_prefix"' in events


def test_progress_trigger_and_timeout_are_parser_mutually_exclusive() -> None:
    harness = load_harness()
    parser = harness.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--recipe",
                "example.recipe",
                "--timeout",
                "5",
                "--cancel-after-message-prefix",
                "Generating",
            ]
        )
