from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "ltx23-generation-tests.py"
    spec = importlib.util.spec_from_file_location("latentslate_ltx23_generation_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ltx23_runner_keeps_three_operations_and_lifecycle_cases() -> None:
    runner = load_runner()
    assert (runner.WIDTH, runner.HEIGHT, runner.FRAMES, runner.FPS, runner.STEPS) == (
        768, 512, 25, 24, 8,
    )
    assert set(runner.SCENARIOS) == {
        "t2v-single", "first-frame-single", "first-last-single", "t2v-sequential",
        "first-frame-sequential", "switch", "cancel-recovery",
    }
    assert [item.recipe for item in runner.SCENARIOS["switch"]] == [
        runner.T2V, runner.I2V, runner.FLF, runner.T2V,
    ]
    cancel_recovery = runner.SCENARIOS["cancel-recovery"]
    assert cancel_recovery[0].name == "first-first-last"
    assert cancel_recovery[1].expect_cancel is True
    assert cancel_recovery[2].name == "recovery-after-cancel"
    assert runner.SIGMAS == [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
    ]
    assert (runner.VOCODER_SAMPLE_RATE, runner.VOCODER_CHANNELS) == (48_000, 2)
    assert set(runner.OPTIMIZED_COMFY_SCENARIOS) == {
        "comfy-t2v-single", "comfy-i2v-single", "comfy-flf-single", "comfy-switch", "comfy-cancel-recovery",
    }
    assert [item.recipe for item in runner.OPTIMIZED_COMFY_SCENARIOS["comfy-switch"]] == [
        runner.COMFY_T2V, runner.COMFY_I2V, runner.COMFY_FLF, runner.COMFY_T2V,
    ]


def test_ltx23_runner_uses_ltx_prompt_cache_and_progress_trigger(tmp_path: Path) -> None:
    runner = load_runner()
    run_dir = tmp_path / runner.SCENARIOS["cancel-recovery"][1].name
    command = []

    class Completed:
        returncode = 0

    def fake_run(argv, **_kwargs):
        command.extend(argv)
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"runs": []}', encoding="utf-8")
        return Completed()

    original = runner.subprocess.run
    runner.subprocess.run = fake_run
    try:
        runner.run_spec(
            runner.SCENARIOS["cancel-recovery"][1],
            repository=tmp_path,
            run_root=tmp_path,
            source=tmp_path / "first.png",
            end=tmp_path / "last.png",
            base_url="http://example.invalid",
            default_timeout=1.0,
            preflight_only=True,
        )
    finally:
        runner.subprocess.run = original

    assert "--assert-runtime-state" not in command
    assert "--cancel-after-message-prefix" in command
    assert runner.DENOISE_FIRST_STEP_PREFIX in command
    assert "--cancellation-grace" in command


def test_ltx23_runner_uses_deterministic_endpoint_fixtures(tmp_path: Path) -> None:
    runner = load_runner()
    first_a = runner.deterministic_source(tmp_path / "first-a.png", endpoint="first")
    first_b = runner.deterministic_source(tmp_path / "first-b.png", endpoint="first")
    last = runner.deterministic_source(tmp_path / "last.png", endpoint="last")
    assert runner.file_sha256(first_a) == runner.file_sha256(first_b)
    assert runner.file_sha256(first_a) != runner.file_sha256(last)


def test_ltx23_runner_accepts_only_the_first_observed_denoise_step() -> None:
    runner = load_runner()

    assert runner.observed_denoise_step("Generating synchronized video and audio (1/8)") == (1, 8)
    assert runner.observed_denoise_step("Generating synchronized video and audio (8/8)") == (8, 8)
    assert runner.observed_denoise_step("Generating synchronized video and audio") is None
    runner.validate_cancellation_progress(
        [["running", 0.2, "Generating synchronized video and audio (1/8)"]]
    )
    try:
        runner.validate_cancellation_progress(
            [
                ["running", 0.2, "Generating synchronized video and audio (1/8)"],
                ["running", 0.9, "Generating synchronized video and audio (8/8)"],
            ]
        )
    except RuntimeError as exc:
        assert "final denoise step" in str(exc)
    else:
        raise AssertionError("final denoise step was accepted for cancellation")


def test_ltx23_runner_ffprobe_requires_counted_frames_and_native_av_timing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = load_runner()
    valid = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": runner.WIDTH,
                "height": runner.HEIGHT,
                "avg_frame_rate": "24/1",
                "nb_read_frames": "25",
                "duration": str(runner.FRAMES / runner.FPS),
            },
            {
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "duration": str(runner.FRAMES / runner.FPS),
            },
        ]
    }
    monkeypatch.setattr(runner, "_probe_mp4", lambda _path: valid)
    runner.validate_mp4_streams(tmp_path / "valid.mp4")

    invalid = {**valid, "streams": [*valid["streams"]]}
    invalid["streams"][0] = {**invalid["streams"][0], "nb_read_frames": "24"}
    monkeypatch.setattr(runner, "_probe_mp4", lambda _path: invalid)
    try:
        runner.validate_mp4_streams(tmp_path / "bad.mp4")
    except RuntimeError as exc:
        assert "counted video frames" in str(exc)
    else:
        raise AssertionError("ffprobe count-frame mismatch was accepted")


def test_ltx23_runner_requires_canceled_worker_tree_exit() -> None:
    runner = load_runner()
    record = {
        "job": {"provenance": {"runtime_plan": {"pipeline_fingerprint": "runtime:ltx23:sha256:test"}}},
        "runtime_after": {
            "runtimes": [
                {
                    "key": "ltx23_condition:first_last:runtime:ltx23:sha256:test",
                    "runtime": "ltx23_disposable_worker",
                    "last_worker": {
                        "outcome": "canceled",
                        "terminated": True,
                        "tree_empty": True,
                        "memory_boundary": "disposable_process_exit",
                    },
                }
            ]
        },
    }
    runner.validate_canceled_worker_terminal(record, recipe=runner.FLF)
    stale = {**record, "runtime_after": {"runtimes": [*record["runtime_after"]["runtimes"]]}}
    stale["runtime_after"]["runtimes"][0] = {
        **stale["runtime_after"]["runtimes"][0],
        "key": "ltx23_condition:first_frame:runtime:ltx23:sha256:test",
    }
    try:
        runner.validate_canceled_worker_terminal(stale, recipe=runner.FLF)
    except RuntimeError as exc:
        assert "terminal disposable-worker" in str(exc)
    else:
        raise AssertionError("stale operation wrapper was accepted for cancellation")
    missing_tree = {**record, "runtime_after": {"runtimes": [*record["runtime_after"]["runtimes"]]}}
    missing_tree["runtime_after"]["runtimes"][0] = {
        **missing_tree["runtime_after"]["runtimes"][0],
        "last_worker": {**record["runtime_after"]["runtimes"][0]["last_worker"], "tree_empty": False},
    }
    try:
        runner.validate_canceled_worker_terminal(missing_tree, recipe=runner.FLF)
    except RuntimeError as exc:
        assert "terminal disposable-worker" in str(exc)
    else:
        raise AssertionError("cancellation without tree-empty proof was accepted")


def test_ltx23_runner_routes_each_optimized_operation_and_cancel_trigger(tmp_path: Path) -> None:
    runner = load_runner()
    commands: dict[str, list[str]] = {}

    class Completed:
        returncode = 0

    original = runner.subprocess.run

    def fake_run(argv, **_kwargs):
        label = next(value for index, value in enumerate(argv) if argv[index - 1] == "--run-dir")
        run_dir = Path(label)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text('{"runs": []}', encoding="utf-8")
        commands[run_dir.name] = list(argv)
        return Completed()

    runner.subprocess.run = fake_run
    try:
        for scenario in runner.OPTIMIZED_COMFY_SCENARIOS.values():
            for spec in scenario:
                runner.run_spec(
                    spec,
                    repository=tmp_path,
                    run_root=tmp_path,
                    source=tmp_path / "first.png",
                    end=tmp_path / "last.png",
                    base_url="http://example.invalid",
                    default_timeout=1.0,
                    preflight_only=True,
                )
    finally:
        runner.subprocess.run = original

    assert set(commands) >= {"comfy-t2v", "comfy-i2v", "comfy-flf", "comfy-cancel"}
    assert "--asset" not in commands["comfy-t2v"]
    assert "start_image=" in " ".join(commands["comfy-i2v"])
    assert "end_image=" in " ".join(commands["comfy-flf"])
    assert commands["comfy-cancel"][commands["comfy-cancel"].index("--cancel-after-message-prefix") + 1] == runner.COMFY_QUEUE_RUNNING_PREFIX
    assert "ltx23-comfy-optimized-comfy-flf" in commands["comfy-flf"]


def test_ltx23_runner_main_all_includes_optimized_scenarios(tmp_path: Path, monkeypatch) -> None:
    runner = load_runner()
    invoked = []

    def fake_source(path: Path, *, endpoint: str) -> Path:
        path.write_bytes(endpoint.encode())
        return path

    def fake_run_spec(spec, **_kwargs):
        invoked.append(spec.recipe)
        return {"name": spec.name, "recipe": spec.recipe, "manifest": "fake"}

    monkeypatch.setattr(runner, "deterministic_source", fake_source)
    monkeypatch.setattr(runner, "run_spec", fake_run_spec)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        ["ltx23-generation-tests.py", "all", "--run-root", str(tmp_path / "all")],
    )
    assert runner.main() == 0
    assert set(invoked) >= {runner.COMFY_T2V, runner.COMFY_I2V, runner.COMFY_FLF}


def test_ltx23_runner_validates_optimized_success_and_cancelled_manager_key(tmp_path: Path, monkeypatch) -> None:
    runner = load_runner()
    probe = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": runner.WIDTH, "height": runner.HEIGHT, "avg_frame_rate": "24/1", "nb_read_frames": "25", "duration": str(runner.FRAMES / runner.FPS)},
            {"codec_type": "audio", "sample_rate": "48000", "channels": 2, "duration": str(runner.FRAMES / runner.FPS)},
        ],
    }
    monkeypatch.setattr(runner, "_probe_mp4", lambda _path: probe)
    component = "ltx23-comfy-components:sha256:test"
    worker = {"outcome": "succeeded", "tree_empty": True}
    runtime = {
        "backend": "comfyui/disposable-official-graph",
        "operation": "comfy_dev_i2v",
        "raw_template_sha256": "raw",
        "submitted_workflow_sha256": "submitted",
        "pipeline_warm": False,
        "cache": {"prompt_hit": False},
        "worker": worker,
        "sampling": {"main_sigmas": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0], "upscale_sigmas": [0.85, 0.725, 0.4219, 0.0], "cfg": 1, "fps": 24},
        "conditioning": {"mode": "first_frame", "ordered_indices": [0], "strength": 0.7},
        "audio_video": {"has_audio": True, "codec": "h264", "width": runner.WIDTH, "height": runner.HEIGHT, "fps": 24, "frame_count": 25, "sample_rate": 48000, "channels": 2, "video_duration_seconds": runner.FRAMES / runner.FPS, "audio_duration_seconds": runner.FRAMES / runner.FPS},
    }
    record = {
        "job": {"status": "succeeded", "provenance": {"runtime_result": runtime, "ltx23_comfy_recipe": {"component_fingerprint": component}}, "artifacts": [{"metadata": {"width": runner.WIDTH, "height": runner.HEIGHT, "fps": 24, "frame_count": 25, "has_audio": True, "duration_seconds": runner.FRAMES / runner.FPS}}]},
        "artifacts": [{"download": {"sha256": "hash", "path": str(tmp_path / "out.mp4")}}],
        "runtime_after": {"runtimes": [{"key": f"ltx23_comfy:comfy_dev_i2v:{component}", "runtime": "comfyui_disposable_worker", "cleanup_errors": [], "last_worker": {"outcome": "succeeded", "terminated": True, "tree_empty": True, "memory_boundary": "disposable_process_exit"}}]},
    }
    runner.validate_comfy_run(record, recipe=runner.COMFY_I2V)
    bad_sampling = {**record, "job": {**record["job"], "provenance": {**record["job"]["provenance"], "runtime_result": {**runtime, "sampling": {}}}}}
    try:
        runner.validate_comfy_run(bad_sampling, recipe=runner.COMFY_I2V)
    except RuntimeError as exc:
        assert "sampling schedule" in str(exc)
    else:
        raise AssertionError("optimized run accepted missing sampling provenance")
    canceled = {
        "job": {"provenance": {"ltx23_comfy_recipe": {"component_fingerprint": component}}},
        "runtime_after": {"runtimes": [{"key": f"ltx23_comfy:comfy_distilled_flf:{component}", "runtime": "comfyui_disposable_worker", "cleanup_errors": [], "last_worker": {"outcome": "canceled", "terminated": True, "tree_empty": True, "memory_boundary": "disposable_process_exit"}}]},
    }
    runner.validate_canceled_worker_terminal(canceled, recipe=runner.COMFY_FLF)
    dirty = {**record, "runtime_after": {"runtimes": [{**record["runtime_after"]["runtimes"][0], "cleanup_errors": ["workspace"]}]}}
    try:
        runner.validate_comfy_run(dirty, recipe=runner.COMFY_I2V)
    except RuntimeError as exc:
        assert "cleanup errors" in str(exc)
    else:
        raise AssertionError("optimized success accepted a retained private workspace")
