#!/usr/bin/env python3
"""Run opt-in Wan 2.2 TI2V 5B public-API hardware acceptance scenarios.

These manual GPU scenarios are intentionally outside pytest. They delegate normal
jobs to ``hardware-study.py`` and retain exact artifacts, public job provenance,
GPU samples, and scenario assertions below ``hardware-study-runs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED = 20_260_813
WIDTH = 320
HEIGHT = 192
FRAMES = 9
T2V = "wan-2-2-5b-ti2v.text-to-video.comfy-fp16"
I2V = "wan-2-2-5b-ti2v.image-to-video.comfy-fp16"
CRUSH_LORA = "lora:wan22:ostris/wan22_5b_i2v_crush_it_lora"
RUNTIME_REVISION = "eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f"
WORKFLOW_REVISION = "f9431bb000ce792094ff345446e22cac1ea6cef3"
WORKFLOW_HASHES = {
    "text_to_video": "e7913b6b2c8f7d82a6a6f9940289bf6e7513cc908bbf455e4553de9804c6f571",
    "image_to_video": "c9408303c6d57b60aa10585d26fc2e10c9c221d2f85a28048cbe2cdba2dc5e12",
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    recipe: str
    repeat: int = 1
    reset: bool = False
    lora: str = "none"
    timeout: float | None = None
    expect_cancel: bool = False


SCENARIOS: dict[str, tuple[RunSpec, ...]] = {
    "t2v-single": (RunSpec("t2v", T2V),),
    "i2v-single": (RunSpec("i2v", I2V),),
    "t2v-warm": (RunSpec("t2v-cold-three-warm", T2V, repeat=4, reset=True),),
    "i2v-warm": (RunSpec("i2v-cold-three-warm", I2V, repeat=4, reset=True),),
    "switch": (
        RunSpec("t2v-before", T2V, reset=True),
        RunSpec("i2v-middle", I2V),
        RunSpec("t2v-after", T2V),
    ),
    "cancel-recovery": (
        RunSpec("i2v-cancel", I2V, timeout=3.0, expect_cancel=True),
        RunSpec("i2v-recovery", I2V),
    ),
    "lora-control": (
        RunSpec("i2v-control", I2V, reset=True),
        RunSpec("i2v-crush-lora", I2V, lora=CRUSH_LORA),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted((*SCENARIOS, "all")))
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def deterministic_source(path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 704), (218, 232, 250))
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        tone = int(232 - 42 * y / image.height)
        draw.line((0, y, image.width, y), fill=(tone, min(248, tone + 8), 255))
    draw.ellipse((420, 310, 820, 510), fill=(184, 66, 24))
    draw.polygon(((780, 350), (970, 425), (790, 472)), fill=(128, 53, 32))
    draw.ellipse((330, 335, 500, 440), fill=(200, 77, 25))
    draw.polygon(((350, 350), (385, 265), (420, 350)), fill=(113, 43, 26))
    draw.ellipse((370, 365, 382, 377), fill=(10, 10, 12))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(record: dict[str, Any], *, operation: str, source_sha256: str) -> None:
    job = record.get("job") or {}
    if job.get("status") != "succeeded":
        raise RuntimeError(f"job did not succeed: {job.get('status')!r}")
    artifacts = record.get("artifacts") or []
    if len(artifacts) != 1 or not artifacts[0].get("download", {}).get("sha256"):
        raise RuntimeError("run did not retain one hashed artifact")
    metadata = (job.get("artifacts") or [{}])[0].get("metadata") or {}
    runtime = (job.get("provenance") or {}).get("runtime_result") or {}
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAMES,
        "fps": 24,
        "steps": 30,
        "sampler": "uni_pc",
        "scheduler": "simple",
        "shift": 8.0,
        "seed": SEED,
        "operation": operation,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"artifact metadata diverged from fixed contract: {metadata}")
    if (
        runtime.get("comfy_runtime_revision") != RUNTIME_REVISION
        or runtime.get("workflow_revision") != WORKFLOW_REVISION
        or runtime.get("workflow_sha256") != WORKFLOW_HASHES[operation]
        or not runtime.get("component_fingerprint")
        or not runtime.get("submitted_workflow_sha256")
    ):
        raise RuntimeError(f"runtime provenance diverged from pinned graph: {runtime}")
    if operation == "image_to_video":
        source = metadata.get("source_image") or {}
        if source.get("sha256") != source_sha256 or not source.get(
            "preprocessing", {}
        ).get("first_latent_anchor"):
            raise RuntimeError("I2V source/anchor provenance is incomplete")
    lora = runtime.get("lora")
    if lora is not None:
        dispatch = runtime.get("lora_dispatch") or {}
        if (
            lora.get("resource_id") != CRUSH_LORA
            or dispatch.get("expected_adapter_tensors") != 600
            or dispatch.get("expected_patch_targets") != 300
            or dispatch.get("unmapped_key_warnings") != 0
        ):
            raise RuntimeError("LoRA load/dispatch provenance is incomplete")


def run_spec(
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
    harness = repository / "scripts" / "hardware-study.py"
    operation = "image_to_video" if spec.recipe == I2V else "text_to_video"
    command = [
        sys.executable,
        str(harness),
        "--base-url",
        base_url,
        "--run-dir",
        str(run_dir),
        "--recipe",
        spec.recipe,
        "--repeat",
        str(spec.repeat),
        "--seed",
        str(SEED),
        "--prompt",
        (
            "crush it, a red fox presses into deep snow, stable cinematic side view"
            if spec.lora != "none"
            else "A red fox walks naturally through clean snow, stable cinematic side view"
        ),
        "--timeout",
        str(spec.timeout or default_timeout),
        "--study-label",
        f"wan22-ti2v5b-{spec.name}",
        "--input",
        f"width={WIDTH}",
        "--input",
        f"height={HEIGHT}",
        "--input",
        f"num_frames={FRAMES}",
    ]
    if operation == "image_to_video":
        command.extend(("--asset", f"source_image={source}"))
    if spec.lora != "none":
        command.extend(("--input", f"style_lora={json.dumps(spec.lora)}"))
        command.extend(("--input", "style_strength=1.0"))
    if spec.reset:
        command.extend(("--reset-runtime-before-recipe", "--assert-runtime-state"))
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
    elif completed.returncode != 0:
        raise RuntimeError(f"hardware study failed with exit code {completed.returncode}")
    if not preflight_only and not spec.expect_cancel:
        for record in manifest.get("runs", []):
            validate_run(record, operation=operation, source_sha256=file_sha256(source))
    return {
        "name": spec.name,
        "recipe": spec.recipe,
        "manifest": str(manifest_path),
        "exit_code": completed.returncode,
        "expected_cancel": spec.expect_cancel,
        "component_fingerprints": [
            ((record.get("job") or {}).get("provenance") or {})
            .get("runtime_result", {})
            .get("component_fingerprint")
            for record in manifest.get("runs", [])
            if (record.get("job") or {}).get("status") == "succeeded"
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    run_root = (
        args.run_root
        or repository
        / "hardware-study-runs"
        / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-wan5-{args.scenario}"
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    source = (
        args.source_image.resolve()
        if args.source_image
        else deterministic_source(run_root / "fixed-source.png")
    )
    if not source.is_file():
        raise SystemExit("--source-image must name an existing image")
    specs = (
        tuple(spec for values in SCENARIOS.values() for spec in values)
        if args.scenario == "all"
        else SCENARIOS[args.scenario]
    )
    summary: dict[str, Any] = {
        "format": "latentslate-wan22-ti2v5b-acceptance-v1",
        "scenario": args.scenario,
        "fixed": {
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "frames": FRAMES,
            "fps": 24,
            "steps": 30,
            "cfg": 5.0,
            "sampler": "uni_pc",
            "scheduler": "simple",
            "shift": 8.0,
            "source_image": str(source),
            "source_sha256": file_sha256(source),
        },
        "runs": [],
    }
    summary_path = run_root / "scenario.json"
    for spec in specs:
        record = run_spec(
            spec,
            repository=repository,
            run_root=run_root,
            source=source,
            base_url=args.base_url,
            default_timeout=args.timeout,
            preflight_only=args.preflight_only,
        )
        summary["runs"].append(record)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.scenario == "switch" and not args.preflight_only:
        fingerprints = [value for run in summary["runs"] for value in run["component_fingerprints"]]
        if not fingerprints or len(set(fingerprints)) != 1:
            raise RuntimeError("T2V/I2V switching did not retain one component closure")
    print(f"Scenario record: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
