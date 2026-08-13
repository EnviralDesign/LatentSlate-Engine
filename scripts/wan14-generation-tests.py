#!/usr/bin/env python3
"""Run opt-in Wan 2.2 14B I2V public-API acceptance scenarios.

This runner uses an explicit 832x480 target-workstation acceptance override
(the built-in Comfy default is 640x640), with 81 frames at 16 fps, fixed
20-step split 10/10, Euler/simple shift-5 sampling, CFG 3.5 in both stages,
and a fixed seed. It delegates jobs to ``hardware-study.py`` so every run
retains a public API manifest, output hash, provenance, and device-wide GPU
samples. It is deliberately excluded from normal pytest and CI.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED = 43301611940728
WIDTH = 832
HEIGHT = 480
FRAMES = 81
STEPS = 20
FPS = 16
RECIPE = "wan-2-2-14b-i2v.image-to-video.comfy-org-fp8"
DEFAULT_PEER_RECIPE = "flux2-klein-4b.text-to-image.bfl-distilled-nvfp4"
EXECUTION_CACHE_POLICY = "not supported; every native Wan job runs in a disposable worker"
# A normal Engine can retain small Python/request allocations. A Wan worker must
# never leave its former tens-of-GiB materialization in the parent process.
PARENT_PRIVATE_MEMORY_LEEWAY_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    recipe: str = RECIPE
    repeat: int = 1
    timeout: float | None = None
    expect_cancel: bool = False
    alternate_source: bool = False


SCENARIOS: dict[str, tuple[RunSpec, ...]] = {
    "i2v-single": (RunSpec("i2v"),),
    # The recipe deliberately unloads after each job. These are repeated cold
    # executions, not falsely-labelled cache warm measurements.
    "i2v-sequential": (RunSpec("i2v-three-sequential", repeat=3),),
    "changed-image": (
        RunSpec("source-control"),
        RunSpec("source-changed", alternate_source=True),
    ),
    "cancel-recovery": (
        RunSpec("cancel-during-materialization", timeout=30.0, expect_cancel=True),
        RunSpec("recovery-after-cancel"),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted((*SCENARIOS, "switch", "all")))
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--peer-recipe",
        default=DEFAULT_PEER_RECIPE,
        help=(
            "Accepted non-Wan recipe used only for A-to-B-to-A lifecycle testing. "
            "The default is the installed Klein 4B Recommended T2I recipe."
        ),
    )
    return parser


def deterministic_source(path: Path, *, alternate: bool = False) -> Path:
    """Create a stable source image without relying on an external corpus."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), (217, 231, 246))
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        tone = int(232 - 40 * y / image.height)
        draw.line((0, y, image.width, y), fill=(tone, min(248, tone + 8), 255))
    fox = (72, 153, 145, 276) if alternate else (526, 153, 599, 276)
    draw.ellipse(fox, fill=(186, 67, 25))
    draw.polygon(
        ((fox[2] - 10, fox[1] + 32), (fox[2] + 112, fox[1] + 76), (fox[2] - 10, fox[1] + 116)),
        fill=(124, 51, 31),
    )
    draw.ellipse((fox[0] - 55, fox[1] + 18, fox[0] + 20, fox[1] + 72), fill=(201, 78, 26))
    draw.ellipse((fox[0] - 33, fox[1] + 40, fox[0] - 24, fox[1] + 49), fill=(8, 8, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wan_run(record: dict[str, Any]) -> None:
    job = record.get("job") or {}
    if job.get("status") != "succeeded":
        raise RuntimeError(f"Wan job did not succeed: {job.get('status')!r}")
    artifacts = record.get("artifacts") or []
    if len(artifacts) != 1 or not artifacts[0].get("download", {}).get("sha256"):
        raise RuntimeError("Wan run did not retain exactly one hashed video")
    _assert_exact_video_stream(Path(artifacts[0]["download"]["path"]))
    metadata = (job.get("artifacts") or [{}])[0].get("metadata") or {}
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAMES,
        "fps": FPS,
        "steps": STEPS,
        "seed": SEED,
        "stage_policy": "expert_split",
        "high_guidance": 3.5,
        "low_guidance": 3.5,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Wan artifact metadata diverged from the fixed contract: {metadata}")
    runtime = metadata.get("runtime_provenance") or {}
    if (
        runtime.get("stage_policy") != "expert_split"
        or runtime.get("steps") != STEPS
        or runtime.get("seed") != SEED
        or runtime.get("sampler") != "euler"
        or runtime.get("scheduler") != "simple"
        or runtime.get("shift") != 5.0
        or runtime.get("transformer_high_contract") != "comfy_legacy/scaled_fp8_e4m3fn"
        or runtime.get("transformer_low_contract") != "comfy_legacy/scaled_fp8_e4m3fn"
        or runtime.get("text_encoder_contract") != "comfy_legacy/scaled_fp8_e4m3fn"
        or not runtime.get("support_fingerprint")
    ):
        raise RuntimeError(f"Wan runtime provenance is incomplete or mismatched: {runtime}")
    runtime_after = record.get("runtime_after") or {}
    matching = [
        item
        for item in runtime_after.get("runtimes", [])
        if item.get("recipe_fingerprint") == metadata.get("recipe_fingerprint")
    ]
    if (
        len(matching) != 1
        or matching[0].get("loaded") is not False
        or matching[0].get("active_worker") is not False
        or (matching[0].get("last_worker") or {}).get("terminated") is not True
    ):
        raise RuntimeError("Wan recipe did not prove disposable-worker teardown")
    cache_support = matching[0].get("cache_support") or {}
    if cache_support != {"prompt": False, "media": False}:
        raise RuntimeError(f"Wan execution-cache contract changed unexpectedly: {cache_support}")
    execution = (job.get("provenance") or {}).get("runtime_result") or {}
    if (
        execution.get("pipeline_warm") is not False
        or (execution.get("execution_cache") or {}).get("hit") is not False
        or (execution.get("execution_cache") or {}).get("mode") != "fresh_disposable_process"
        or (execution.get("worker") or {}).get("terminated") is not True
    ):
        raise RuntimeError(f"Wan run was not a proven fresh native execution: {execution}")
    worker = execution.get("worker") or {}
    worker_pid = worker.get("pid")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
        raise RuntimeError(f"Wan runtime did not retain a valid worker PID: {worker}")
    if _process_exists(worker_pid):
        raise RuntimeError(f"Wan worker PID {worker_pid} still exists after terminal success")
    _assert_parent_memory_returned_to_baseline(record.get("runtime_before") or {}, runtime_after)


def validate_wan_cancellation(record: dict[str, Any]) -> None:
    """Cancellation is accepted only after the owned worker is really gone."""

    final = (record.get("timeout") or {}).get("final_job") or {}
    if final.get("status") != "canceled":
        raise RuntimeError(f"expected confirmed cancellation, found {final.get('status')!r}")
    runtime_after = record.get("runtime_after") or {}
    matching = [
        item
        for item in runtime_after.get("runtimes", [])
        if item.get("runtime") == "native_wan_i2v_14b_disposable_worker"
    ]
    if len(matching) != 1 or matching[0].get("active_worker") is not False:
        raise RuntimeError("Wan cancellation did not reach an idle disposable-worker runtime")
    worker = matching[0].get("last_worker") or {}
    worker_pid = worker.get("pid")
    if worker.get("terminated") is not True or not isinstance(worker_pid, int) or worker_pid <= 0:
        raise RuntimeError(f"Wan cancellation did not retain worker termination evidence: {worker}")
    if _process_exists(worker_pid):
        raise RuntimeError(f"Wan worker PID {worker_pid} still exists after cancellation")
    _assert_parent_memory_returned_to_baseline(record.get("runtime_before") or {}, runtime_after)


def _assert_parent_memory_returned_to_baseline(
    runtime_before: dict[str, Any], runtime_after: dict[str, Any]
) -> None:
    before = runtime_before.get("host_process") or {}
    after = runtime_after.get("host_process") or {}
    before_pid = before.get("pid")
    after_pid = after.get("pid")
    before_private = before.get("private_bytes")
    after_private = after.get("private_bytes")
    if (
        isinstance(before_pid, bool)
        or not isinstance(before_pid, int)
        or before_pid <= 0
        or before_pid != after_pid
    ):
        raise RuntimeError(f"Engine host PID changed during Wan acceptance: {before} -> {after}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (before_private, after_private)
    ):
        raise RuntimeError(
            f"Engine did not expose Windows private-byte evidence: {before} -> {after}"
        )
    if after_private > before_private + PARENT_PRIVATE_MEMORY_LEEWAY_BYTES:
        raise RuntimeError(
            "Wan parent private bytes did not return near its pre-job baseline: "
            f"before={before_private}, after={after_private}, "
            f"leeway={PARENT_PRIVATE_MEMORY_LEEWAY_BYTES}"
        )


def _process_exists(pid: int) -> bool:
    """Check the recorded worker PID without shelling out to tasklist."""

    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID is no longer present.
            return False
        raise ctypes.WinError(error)
    if not kernel.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    return True


def _assert_exact_video_stream(path: Path) -> None:
    """Verify the public artifact, not only Engine-declared metadata."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe could not inspect Wan output: {completed.stderr.strip()}")
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError("Wan output lacks exactly one video stream")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or stream.get("width") != WIDTH
        or stream.get("height") != HEIGHT
        or stream.get("avg_frame_rate") != "16/1"
        or int(stream.get("nb_read_frames", -1)) != FRAMES
        or abs(float(stream.get("duration", "nan")) - FRAMES / FPS) > 1e-6
    ):
        raise RuntimeError(f"Wan output stream diverged from acceptance contract: {stream}")


def run_wan_spec(
    spec: RunSpec,
    *,
    repository: Path,
    run_root: Path,
    source: Path,
    base_url: str,
    default_timeout: float,
    preflight_only: bool,
) -> dict[str, Any]:
    run_dir = run_root / spec.name
    command = [
        sys.executable,
        str(repository / "scripts" / "hardware-study.py"),
        "--base-url",
        base_url,
        "--run-dir",
        str(run_dir),
        "--recipe",
        spec.recipe,
        "--repeat",
        str(spec.repeat),
        "--cold-repeats",
        str(spec.repeat),
        "--seed",
        str(SEED),
        "--prompt",
        "A red fox walks naturally through clean snow, stable cinematic side view.",
        "--asset",
        f"source_image={source}",
        "--input",
        f"width={WIDTH}",
        "--input",
        f"height={HEIGHT}",
        "--input",
        f"num_frames={FRAMES}",
        "--timeout",
        str(spec.timeout or default_timeout),
        "--study-label",
        f"wan22-14b-i2v-{spec.name}",
        "--reset-runtime-before-recipe",
    ]
    if preflight_only:
        command.append("--preflight-only")
    completed = subprocess.run(command, cwd=repository, check=False)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.expect_cancel and not preflight_only:
        record = (manifest.get("runs") or [{}])[0]
        validate_wan_cancellation(record)
    elif completed.returncode != 0:
        raise RuntimeError(f"hardware study failed with exit code {completed.returncode}")
    if not preflight_only and not spec.expect_cancel:
        for record in manifest.get("runs", []):
            validate_wan_run(record)
    return {
        "name": spec.name,
        "recipe": spec.recipe,
        "manifest": str(manifest_path),
        "exit_code": completed.returncode,
        "expected_cancel": spec.expect_cancel,
        "source_sha256": file_sha256(source),
    }


def run_peer_recipe(
    *, repository: Path, run_root: Path, base_url: str, recipe: str, preflight_only: bool
) -> dict[str, Any]:
    """Run one accepted Klein T2I job between two Wan jobs for A-to-B-to-A."""

    run_dir = run_root / "peer-between-wan"
    command = [
        sys.executable,
        str(repository / "scripts" / "hardware-study.py"),
        "--base-url",
        base_url,
        "--run-dir",
        str(run_dir),
        "--recipe",
        recipe,
        "--prompt",
        "A small brass clock on a white studio table.",
        "--seed",
        str(SEED),
        "--input",
        "width=1024",
        "--input",
        "height=1024",
        "--timeout",
        "1200",
        "--study-label",
        "wan22-14b-i2v-peer-switch",
    ]
    if preflight_only:
        command.append("--preflight-only")
    completed = subprocess.run(command, cwd=repository, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"peer switch study failed with exit code {completed.returncode}")
    return {
        "name": "peer-between-wan",
        "recipe": recipe,
        "manifest": str(run_dir / "manifest.json"),
    }


def main() -> int:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    run_root = (
        args.run_root
        or repository
        / "hardware-study-runs"
        / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-wan14-{args.scenario}"
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    source = (
        args.source_image.resolve()
        if args.source_image
        else deterministic_source(run_root / "fixed-source.png")
    )
    if not source.is_file():
        raise SystemExit("--source-image must name an existing image")
    alternate = deterministic_source(run_root / "alternate-source.png", alternate=True)
    summary: dict[str, Any] = {
        "format": "latentslate-wan22-14b-i2v-acceptance-v1",
        "scenario": args.scenario,
        "fixed": {
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "frames": FRAMES,
            "fps": FPS,
            "steps": STEPS,
            "stage_policy": "expert_split",
            "high_guidance": 3.5,
            "low_guidance": 3.5,
            "source_sha256": file_sha256(source),
        },
        "execution_cache": EXECUTION_CACHE_POLICY,
        "runs": [],
    }
    summary_path = run_root / "scenario.json"
    specs = (
        tuple(spec for scenario in SCENARIOS.values() for spec in scenario)
        if args.scenario == "all"
        else SCENARIOS.get(args.scenario, ())
    )
    for spec in specs:
        selected_source = alternate if spec.alternate_source else source
        summary["runs"].append(
            run_wan_spec(
                spec,
                repository=repository,
                run_root=run_root,
                source=selected_source,
                base_url=args.base_url,
                default_timeout=args.timeout,
                preflight_only=args.preflight_only,
            )
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.scenario == "switch":
        for spec in (RunSpec("wan-before-peer"),):
            summary["runs"].append(
                run_wan_spec(
                    spec,
                    repository=repository,
                    run_root=run_root,
                    source=source,
                    base_url=args.base_url,
                    default_timeout=args.timeout,
                    preflight_only=args.preflight_only,
                )
            )
        summary["runs"].append(
            run_peer_recipe(
                repository=repository,
                run_root=run_root,
                base_url=args.base_url,
                recipe=args.peer_recipe,
                preflight_only=args.preflight_only,
            )
        )
        summary["runs"].append(
            run_wan_spec(
                RunSpec("wan-after-peer"),
                repository=repository,
                run_root=run_root,
                source=source,
                base_url=args.base_url,
                default_timeout=args.timeout,
                preflight_only=args.preflight_only,
            )
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.scenario == "changed-image" and not args.preflight_only:
        hashes = [
            json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))["runs"][0]["artifacts"][
                0
            ]["download"]["sha256"]
            for run in summary["runs"]
        ]
        if len(set(hashes)) != 2:
            raise RuntimeError(
                "changed source image unexpectedly produced the same video artifact hash"
            )
    print(f"Scenario record: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
