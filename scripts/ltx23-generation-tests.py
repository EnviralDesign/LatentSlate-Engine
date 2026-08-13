#!/usr/bin/env python3
"""Run opt-in LTX 2.3 public-API acceptance scenarios.

These cases intentionally stay outside pytest and CI.  They exercise the exact
native Diffusers structural closure and the separate official optimized Comfy
graphs through the public API. They retain output hashes, full job provenance,
GPU samples, terminal-worker state, and MP4 stream facts below
``hardware-study-runs``. The native BF16 and Comfy Dev-FP8-plus-Distilled-LoRA /
Distilled-FP8 first+last operations remain distinct comparison boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED = 20_260_813
WIDTH = 768
HEIGHT = 512
DURATION_SECONDS = 1.0
FPS = 24
FRAMES = 25
STEPS = 8
GUIDANCE_SCALE = 1.0
SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
VOCODER_SAMPLE_RATE = 48_000
VOCODER_CHANNELS = 2
CANCELLATION_GRACE_SECONDS = 180.0
DENOISE_FIRST_STEP_PREFIX = "Generating synchronized video and audio (1/"
COMFY_QUEUE_RUNNING_PREFIX = "Comfy queue running official LTX 2.3 graph"
T2V = "ltx-2-3.text-to-video.native-distilled-bf16"
I2V = "ltx-2-3.image-to-video.native-distilled-bf16"
FLF = "ltx-2-3.first-last-frame-to-video.native-distilled-bf16"
COMFY_T2V = "ltx-2-3.text-to-video.comfy-dev-fp8"
COMFY_I2V = "ltx-2-3.image-to-video.comfy-dev-fp8"
COMFY_FLF = "ltx-2-3.first-last-frame-to-video.comfy-distilled-fp8"

NATIVE_RECIPES = frozenset({T2V, I2V, FLF})
COMFY_RECIPES = frozenset({COMFY_T2V, COMFY_I2V, COMFY_FLF})


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    recipe: str
    repeat: int = 1
    reset: bool = False
    timeout: float | None = None
    expect_cancel: bool = False
    expected_pipeline_warm: bool | None = None


SCENARIOS: dict[str, tuple[RunSpec, ...]] = {
    "t2v-single": (RunSpec("t2v", T2V),),
    "first-frame-single": (RunSpec("first-frame", I2V),),
    "first-last-single": (RunSpec("first-last", FLF),),
    "t2v-sequential": (RunSpec("t2v-four-disposable-workers", T2V, repeat=4, reset=True),),
    "first-frame-sequential": (RunSpec("first-frame-four-disposable-workers", I2V, repeat=4, reset=True),),
    "switch": (
        RunSpec("t2v-before", T2V, reset=True),
        RunSpec("first-frame-middle", I2V),
        RunSpec("first-last-middle", FLF),
        RunSpec("t2v-after", T2V),
    ),
    "cancel-recovery": (
        RunSpec("first-first-last", FLF, reset=True, expected_pipeline_warm=False),
        RunSpec("cancel-during-denoise", FLF, expect_cancel=True),
        RunSpec("recovery-after-cancel", FLF, expected_pipeline_warm=False),
    ),
}

# Separate from the BF16 reference suite: all optimized Comfy scenarios are
# cold disposable workers and their output facts are validated from ffprobe.
OPTIMIZED_COMFY_SCENARIOS: dict[str, tuple[RunSpec, ...]] = {
    "comfy-t2v-single": (RunSpec("comfy-t2v", COMFY_T2V),),
    "comfy-i2v-single": (RunSpec("comfy-i2v", COMFY_I2V),),
    "comfy-flf-single": (RunSpec("comfy-flf", COMFY_FLF),),
    "comfy-switch": (
        RunSpec("comfy-t2v-before", COMFY_T2V, reset=True),
        RunSpec("comfy-i2v-middle", COMFY_I2V),
        RunSpec("comfy-flf-middle", COMFY_FLF),
        RunSpec("comfy-t2v-after", COMFY_T2V),
    ),
    "comfy-cancel-recovery": (
        RunSpec("comfy-first", COMFY_FLF, reset=True, expected_pipeline_warm=False),
        RunSpec("comfy-cancel", COMFY_FLF, expect_cancel=True),
        RunSpec("comfy-recovery", COMFY_FLF, expected_pipeline_warm=False),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted((*SCENARIOS, *OPTIMIZED_COMFY_SCENARIOS, "all")))
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--end-image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def deterministic_source(path: Path, *, endpoint: str) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), (27, 45, 72))
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        tone = int(72 + 62 * y / HEIGHT)
        draw.line((0, y, WIDTH, y), fill=(tone // 2, tone, min(255, tone + 50)))
    x = 176 if endpoint == "first" else 510
    draw.ellipse((x, 170, x + 105, 325), fill=(210, 104, 42))
    draw.polygon(((x + 85, 210), (x + 220, 257), (x + 85, 300)), fill=(145, 67, 37))
    draw.rectangle((0, 387, WIDTH, HEIGHT), fill=(38, 44, 53))
    draw.text((32, 32), f"LTX fixed {endpoint} endpoint", fill=(236, 240, 246))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_mp4(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _ratio(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator or float(denominator) == 0:
        raise RuntimeError(f"invalid rational stream value {value!r}")
    return float(numerator) / float(denominator)


def validate_mp4_streams(path: Path) -> None:
    probe = _probe_mp4(path)
    container = probe.get("format") or {}
    if "mp4" not in str(container.get("format_name", "")).split(","):
        raise RuntimeError("expected an MP4-family output container")
    streams = probe.get("streams") or []
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError("expected exactly one video stream and one audio stream")
    video, audio = videos[0], audios[0]
    if video.get("codec_name") != "h264":
        raise RuntimeError(f"MP4 video codec diverged: {video}")
    if (video.get("width"), video.get("height")) != (WIDTH, HEIGHT):
        raise RuntimeError(f"MP4 video dimensions diverged: {video}")
    if not math.isclose(_ratio(str(video.get("avg_frame_rate"))), FPS, abs_tol=0.001):
        raise RuntimeError(f"MP4 video frame rate diverged: {video}")
    if int(video.get("nb_read_frames") or 0) != FRAMES:
        raise RuntimeError(f"MP4 counted video frames diverged: {video}")
    if (int(audio.get("sample_rate") or 0), int(audio.get("channels") or 0)) != (
        VOCODER_SAMPLE_RATE,
        VOCODER_CHANNELS,
    ):
        raise RuntimeError(f"MP4 audio is not native LTX 48k stereo: {audio}")
    video_duration = float(video.get("duration") or 0.0)
    audio_duration = float(audio.get("duration") or 0.0)
    target_duration = FRAMES / FPS
    tolerance = 1 / FPS
    if not math.isclose(video_duration, target_duration, abs_tol=tolerance):
        raise RuntimeError(f"MP4 video duration diverged: {video_duration}")
    if not math.isclose(audio_duration, target_duration, abs_tol=tolerance):
        raise RuntimeError(f"MP4 audio duration diverged: {audio_duration}")
    if abs(video_duration - audio_duration) > tolerance:
        raise RuntimeError(f"MP4 A/V durations drift beyond one frame: {video_duration}, {audio_duration}")


def observed_denoise_step(message: object) -> tuple[int, int] | None:
    match = re.fullmatch(r"Generating synchronized video and audio \((\d+)/(\d+)\)", str(message))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def validate_cancellation_progress(states: list[object]) -> None:
    observed_steps = [
        observed_denoise_step(state[2])
        for state in states
        if isinstance(state, list)
        and len(state) >= 3
        and state[0] == "running"
        and state[1] is not None
        and float(state[1]) > 0.10
    ]
    if (1, STEPS) not in observed_steps:
        raise RuntimeError("cancellation was not issued after observed first LTX denoise step")
    if any(step is not None and step[0] >= step[1] for step in observed_steps):
        raise RuntimeError("cancellation acceptance observed the final denoise step")


def validate_run(record: dict[str, Any], *, recipe: str) -> None:
    job = record.get("job") or {}
    if job.get("status") != "succeeded":
        raise RuntimeError(f"job did not succeed: {job.get('status')!r}")
    artifacts = record.get("artifacts") or []
    if len(artifacts) != 1 or not artifacts[0].get("download", {}).get("sha256"):
        raise RuntimeError("run did not retain one hashed MP4 artifact")
    validate_mp4_streams(Path(artifacts[0]["download"]["path"]))
    metadata = (job.get("artifacts") or [{}])[0].get("metadata") or {}
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAMES,
        "fps": FPS,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "has_audio": True,
        "sigmas": SIGMAS,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"artifact metadata diverged from the fixed contract: {metadata}")
    if metadata.get("duration_seconds") != FRAMES / FPS:
        raise RuntimeError("artifact metadata did not retain 24fps/8n+1 duration")
    conditioning = metadata.get("conditioning")
    if recipe == T2V:
        if conditioning is not None:
            raise RuntimeError("text-to-video unexpectedly reported endpoint conditioning")
    elif recipe == I2V:
        if conditioning != {
            "mode": "first_frame",
            "start_frame": True,
            "end_frame": False,
            "ordered_indices": [0],
        }:
            raise RuntimeError(f"first-frame conditioning provenance is incomplete: {conditioning}")
    elif recipe == FLF and conditioning != {
        "mode": "first_last_frame",
        "start_frame": True,
        "end_frame": True,
        "ordered_indices": [0, -1],
    }:
        raise RuntimeError(f"first+last conditioning provenance is incomplete: {conditioning}")
    runtime = (job.get("provenance") or {}).get("runtime_result") or {}
    cache = runtime.get("cache") or {}
    if not runtime.get("pipeline_fingerprint") or "pipeline_warm" not in cache:
        raise RuntimeError(f"runtime provenance/cache state is incomplete: {runtime}")
    if recipe in {I2V, FLF} and runtime.get("conditioning") != conditioning:
        raise RuntimeError("runtime provenance lost ordered endpoint conditioning")
    audio_video = runtime.get("audio_video") or {}
    if audio_video != {
        "fps": FPS,
        "frame_count": FRAMES,
        "duration_seconds": FRAMES / FPS,
        "has_audio": True,
    }:
        raise RuntimeError("runtime provenance lost synchronized audio/video facts")
    if runtime.get("sampling") != {
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "sigmas": SIGMAS,
    }:
        raise RuntimeError("runtime provenance lost the pinned Distilled sigma schedule")


def _comfy_operation(recipe: str) -> str:
    return {
        COMFY_T2V: "comfy_dev_t2v",
        COMFY_I2V: "comfy_dev_i2v",
        COMFY_FLF: "comfy_distilled_flf",
    }[recipe]


def _comfy_manager_entry(record: dict[str, Any], *, recipe: str) -> dict[str, Any]:
    recipe_provenance = ((record.get("job") or {}).get("provenance") or {}).get(
        "ltx23_comfy_recipe"
    ) or {}
    fingerprint = recipe_provenance.get("component_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("optimized Comfy job did not retain its component fingerprint")
    expected_key = ":".join(("ltx23_comfy", _comfy_operation(recipe), fingerprint))
    runtimes = (record.get("runtime_after") or {}).get("runtimes") or []
    entry = next(
        (
            candidate
            for candidate in runtimes
            if candidate.get("runtime") == "comfyui_disposable_worker"
            and candidate.get("key") == expected_key
        ),
        None,
    )
    if not isinstance(entry, dict):
        raise TypeError("optimized Comfy runtime status is missing its exact worker entry")
    return entry


def _validate_comfy_manager_terminal(record: dict[str, Any], *, recipe: str, outcome: str) -> None:
    entry = _comfy_manager_entry(record, recipe=recipe)
    if entry.get("cleanup_errors") != []:
        raise RuntimeError("optimized Comfy worker retained private workspace cleanup errors")
    worker = entry.get("last_worker") or {}
    if (
        worker.get("outcome") != outcome
        or worker.get("terminated") is not True
        or worker.get("tree_empty") is not True
        or worker.get("memory_boundary") != "disposable_process_exit"
    ):
        raise RuntimeError("optimized Comfy worker did not prove terminal disposable cleanup")


def validate_comfy_run(record: dict[str, Any], *, recipe: str) -> None:
    """Validate only observed/public facts from one optimized Comfy operation."""

    job = record.get("job") or {}
    if job.get("status") != "succeeded":
        raise RuntimeError(f"optimized Comfy job did not succeed: {job.get('status')!r}")
    artifacts = record.get("artifacts") or []
    if len(artifacts) != 1 or not artifacts[0].get("download", {}).get("sha256"):
        raise RuntimeError("optimized Comfy run did not retain one hashed MP4 artifact")
    validate_mp4_streams(Path(artifacts[0]["download"]["path"]))
    runtime = ((job.get("provenance") or {}).get("runtime_result") or {})
    if runtime.get("backend") != "comfyui/disposable-official-graph":
        raise RuntimeError("optimized Comfy runtime provenance has the wrong backend")
    if runtime.get("operation") != _comfy_operation(recipe):
        raise RuntimeError("optimized Comfy runtime provenance has the wrong operation")
    if not runtime.get("raw_template_sha256") or not runtime.get("submitted_workflow_sha256"):
        raise RuntimeError("optimized Comfy provenance lacks raw/submitted graph identities")
    if runtime.get("pipeline_warm") is not False or (runtime.get("cache") or {}).get("prompt_hit") is not False:
        raise RuntimeError("optimized Comfy run was not an explicit cold disposable worker")
    worker = runtime.get("worker") or {}
    if worker.get("outcome") != "succeeded" or worker.get("tree_empty") is not True:
        raise RuntimeError("optimized Comfy run did not prove terminal worker cleanup")
    audio_video = runtime.get("audio_video") or {}
    expected_av = {
        "has_audio": True,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": FRAMES,
        "sample_rate": VOCODER_SAMPLE_RATE,
        "channels": VOCODER_CHANNELS,
    }
    if any(audio_video.get(key) != value for key, value in expected_av.items()):
        raise RuntimeError(f"optimized Comfy provenance lost observed A/V facts: {audio_video}")
    expected_duration = FRAMES / FPS
    video_duration = float(audio_video.get("video_duration_seconds") or 0.0)
    audio_duration = float(audio_video.get("audio_duration_seconds") or 0.0)
    if (
        audio_video.get("codec") != "h264"
        or not math.isclose(video_duration, expected_duration, abs_tol=1 / FPS)
        or not math.isclose(audio_duration, expected_duration, abs_tol=1 / FPS)
        or abs(video_duration - audio_duration) > 1 / FPS
    ):
        raise RuntimeError("optimized Comfy provenance lost synchronized observed A/V timing")
    metadata = ((job.get("artifacts") or [{}])[0].get("metadata") or {})
    if any(metadata.get(key) != value for key, value in expected_av.items() if key in metadata):
        raise RuntimeError("optimized Comfy artifact metadata diverged from observed A/V facts")
    if not math.isclose(float(metadata.get("duration_seconds") or 0.0), expected_duration, abs_tol=1 / FPS):
        raise RuntimeError("optimized Comfy artifact metadata lost observed video duration")
    expected_sampling = {
        "main_sigmas": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
        "upscale_sigmas": None if recipe == COMFY_FLF else [0.85, 0.725, 0.4219, 0.0],
        "cfg": 1,
        "fps": 24,
    }
    if runtime.get("sampling") != expected_sampling:
        raise RuntimeError("optimized Comfy provenance lost the pinned sampling schedule")
    expected_conditioning = {
        COMFY_T2V: {"mode": "text"},
        COMFY_I2V: {"mode": "first_frame", "ordered_indices": [0], "strength": 0.7},
        COMFY_FLF: {"mode": "first_last_frame", "ordered_indices": [0, -1], "strength": 0.7},
    }[recipe]
    if runtime.get("conditioning") != expected_conditioning:
        raise RuntimeError("optimized Comfy provenance lost operation endpoint conditioning")
    _validate_comfy_manager_terminal(record, recipe=recipe, outcome="succeeded")


def validate_ltx_disposable_workers(records: list[dict[str, Any]]) -> None:
    """Every LTX job must be cold in a fresh, terminally exited worker tree."""

    for index, record in enumerate(records):
        runtime = ((record.get("job") or {}).get("provenance") or {}).get("runtime_result") or {}
        cache = runtime.get("cache") or {}
        if runtime.get("pipeline_warm") is not False:
            raise RuntimeError(
                f"LTX run {index + 1} expected disposable pipeline_warm=False, "
                f"found {runtime.get('pipeline_warm')!r}"
            )
        if cache.get("prompt_hit") is not False:
            raise RuntimeError(
                f"LTX run {index + 1} expected prompt_hit=False, "
                f"found {cache.get('prompt_hit')!r}"
            )
        worker = runtime.get("worker") or {}
        if (
            worker.get("memory_boundary") != "disposable_process_exit"
            or worker.get("terminated") is not True
            or worker.get("tree_empty") is not True
        ):
            raise RuntimeError(f"LTX run {index + 1} did not prove disposable worker exit")


def validate_canceled_worker_terminal(record: dict[str, Any], *, recipe: str) -> None:
    if recipe in COMFY_RECIPES:
        recipe_provenance = ((record.get("job") or {}).get("provenance") or {}).get(
            "ltx23_comfy_recipe"
        ) or {}
        fingerprint = recipe_provenance.get("component_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise RuntimeError(
                "canceled optimized Comfy job did not retain its component fingerprint"
            )
        expected_key = ":".join(("ltx23_comfy", _comfy_operation(recipe), fingerprint))
    else:
        operation = {
            T2V: ("ltx23", "t2v"),
            I2V: ("ltx23_condition", "first_frame"),
            FLF: ("ltx23_condition", "first_last"),
        }[recipe]
        runtime_plan = ((record.get("job") or {}).get("provenance") or {}).get(
            "runtime_plan"
        ) or {}
        fingerprint = runtime_plan.get("pipeline_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise RuntimeError("canceled LTX job did not retain its runtime-plan fingerprint")
        expected_key = ":".join((*operation, fingerprint))
    runtimes = (record.get("runtime_after") or {}).get("runtimes") or []
    workers = [
        entry.get("last_worker")
        for entry in runtimes
        if entry.get("runtime")
        == ("comfyui_disposable_worker" if recipe in COMFY_RECIPES else "ltx23_disposable_worker")
        and entry.get("key") == expected_key
    ]
    if not any(
        isinstance(worker, dict)
        and worker.get("outcome") == "canceled"
        and worker.get("terminated") is True
        and worker.get("tree_empty") is True
        and worker.get("memory_boundary") == "disposable_process_exit"
        for worker in workers
    ):
        raise RuntimeError("canceled LTX job did not prove terminal disposable-worker cleanup")
    if recipe in COMFY_RECIPES:
        _validate_comfy_manager_terminal(record, recipe=recipe, outcome="canceled")


def run_spec(
    spec: RunSpec,
    *,
    repository: Path,
    run_root: Path,
    source: Path,
    end: Path,
    base_url: str,
    default_timeout: float,
    preflight_only: bool,
) -> dict[str, Any]:
    run_dir = run_root / spec.name
    command = [
        sys.executable,
        str(repository / "scripts" / "hardware-study.py"),
        "--base-url", base_url,
        "--run-dir", str(run_dir),
        "--recipe", spec.recipe,
        "--repeat", str(spec.repeat),
        "--seed", str(SEED),
        "--prompt", "A small brass airship moves steadily over a quiet futuristic harbor. Sound: soft wind, distant water, and a low engine hum.",
        "--cancellation-grace", str(CANCELLATION_GRACE_SECONDS),
        "--study-label", (
            f"ltx23-{'comfy-optimized' if spec.recipe in COMFY_RECIPES else 'native-bf16'}-"
            f"{spec.name}"
        ),
        "--input", f"width={WIDTH}",
        "--input", f"height={HEIGHT}",
        "--input", f"duration_seconds={DURATION_SECONDS}",
    ]
    if spec.expect_cancel:
        prefix = COMFY_QUEUE_RUNNING_PREFIX if spec.recipe in COMFY_RECIPES else DENOISE_FIRST_STEP_PREFIX
        command.extend(("--cancel-after-message-prefix", prefix))
    else:
        command.extend(("--timeout", str(spec.timeout or default_timeout)))
    if spec.recipe in {I2V, FLF, COMFY_I2V, COMFY_FLF}:
        command.extend(("--asset", f"start_image={source}"))
    if spec.recipe in {FLF, COMFY_FLF}:
        command.extend(("--asset", f"end_image={end}"))
    if spec.reset:
        command.append("--reset-runtime-before-recipe")
    if spec.repeat > 1:
        command.append("--assert-deterministic")
    if preflight_only:
        command.append("--preflight-only")
    completed = subprocess.run(command, cwd=repository, check=False)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.expect_cancel and not preflight_only:
        final = ((manifest.get("runs") or [{}])[0].get("timeout") or {}).get("final_job") or {}
        if final.get("status") != "canceled":
            raise RuntimeError(f"expected confirmed cancellation, found {final.get('status')!r}")
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        requested = [event for event in events if event.get("kind") == "cancellation_requested"]
        if (
            len(requested) != 1
            or requested[0].get("trigger") != "message_prefix"
            or requested[0].get("message_prefix")
            != (COMFY_QUEUE_RUNNING_PREFIX if spec.recipe in COMFY_RECIPES else DENOISE_FIRST_STEP_PREFIX)
        ):
            raise RuntimeError(
                "cancellation did not use the exact native denoise or optimized queue-running trigger"
            )
        states = [event.get("state") for event in events if event.get("kind") == "job_state"]
        if spec.recipe in COMFY_RECIPES:
            if not any(
                isinstance(state, list)
                and len(state) >= 3
                and state[0] == "running"
                and str(state[2]).startswith(COMFY_QUEUE_RUNNING_PREFIX)
                for state in states
            ):
                raise RuntimeError(
                    "optimized Comfy cancellation did not observe prompt-bound queue-running progress"
                )
        else:
            validate_cancellation_progress(states)
        validate_canceled_worker_terminal((manifest.get("runs") or [{}])[0], recipe=spec.recipe)
    elif completed.returncode != 0:
        raise RuntimeError(f"hardware study failed with exit code {completed.returncode}")
    if not preflight_only and not spec.expect_cancel:
        records = manifest.get("runs", [])
        for record in records:
            if spec.recipe in COMFY_RECIPES:
                validate_comfy_run(record, recipe=spec.recipe)
            else:
                validate_run(record, recipe=spec.recipe)
        if spec.reset:
            validate_ltx_disposable_workers(records)
        if spec.expected_pipeline_warm is not None:
            runtime = ((records[-1].get("job") or {}).get("provenance") or {}).get(
                "runtime_result"
            ) or {}
            cache = runtime.get("cache") or {}
            if runtime.get("pipeline_warm") is not spec.expected_pipeline_warm or cache.get(
                "prompt_hit"
            ) is not spec.expected_pipeline_warm:
                raise RuntimeError(
                    "LTX disposable-worker cache state diverged: "
                    f"pipeline_warm={runtime.get('pipeline_warm')!r}, "
                    f"prompt_hit={cache.get('prompt_hit')!r}"
                )
    return {"name": spec.name, "recipe": spec.recipe, "manifest": str(manifest_path)}


def main() -> int:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or repository / "hardware-study-runs" / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-ltx23-{args.scenario}").resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    source = args.source_image.resolve() if args.source_image else deterministic_source(run_root / "fixed-first.png", endpoint="first")
    end = args.end_image.resolve() if args.end_image else deterministic_source(run_root / "fixed-last.png", endpoint="last")
    if not source.is_file() or not end.is_file():
        raise SystemExit("--source-image and --end-image must name existing image files")
    all_scenarios = {**SCENARIOS, **OPTIMIZED_COMFY_SCENARIOS}
    specs = (
        tuple(item for items in all_scenarios.values() for item in items)
        if args.scenario == "all"
        else all_scenarios[args.scenario]
    )
    summary = {"format": "latentslate-ltx23-acceptance-v2", "scenario": args.scenario, "fixed": {"seed": SEED, "width": WIDTH, "height": HEIGHT, "duration_seconds": DURATION_SECONDS, "frames": FRAMES, "fps": FPS, "steps": STEPS, "guidance_scale": GUIDANCE_SCALE, "sigmas": SIGMAS, "native_vocoder_sample_rate": VOCODER_SAMPLE_RATE, "native_vocoder_channels": VOCODER_CHANNELS, "source_sha256": file_sha256(source), "end_sha256": file_sha256(end)}, "runs": []}
    summary_path = run_root / "scenario.json"
    for spec in specs:
        summary["runs"].append(run_spec(spec, repository=repository, run_root=run_root, source=source, end=end, base_url=args.base_url, default_timeout=args.timeout, preflight_only=args.preflight_only))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Scenario record: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
