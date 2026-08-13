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
        "t2v-single", "first-frame-single", "first-last-single", "t2v-warm",
        "first-frame-warm", "switch", "cancel-recovery",
    }
    assert [item.recipe for item in runner.SCENARIOS["switch"]] == [
        runner.T2V, runner.I2V, runner.FLF, runner.T2V,
    ]
    cancel_recovery = runner.SCENARIOS["cancel-recovery"]
    assert cancel_recovery[0].name == "warm-first-last"
    assert cancel_recovery[1].expect_cancel is True
    assert cancel_recovery[2].name == "recovery-after-cancel"
    assert runner.SIGMAS == [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
    ]
    assert (runner.VOCODER_SAMPLE_RATE, runner.VOCODER_CHANNELS) == (48_000, 2)


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
        "streams": [
            {
                "codec_type": "video",
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
