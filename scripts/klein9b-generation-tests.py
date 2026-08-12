#!/usr/bin/env python3
"""Run fixed-parameter Klein 9B acceptance scenarios through the public API.

These are manual GPU tests, not pytest cases. Each scenario delegates to
``hardware-study.py`` so it exercises the same HTTP routes used by LatentSlate
and retains one exact manifest per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SEED = 43_301_611_940_728
WIDTH = 1024
HEIGHT = 1024
T2I_PROMPT = (
    "A cobalt-blue ceramic robot cat seated on a walnut desk, soft window light, "
    "fine material detail, clean editorial photograph."
)
I2I_PROMPT = (
    "Change the primary subject to cobalt blue while preserving composition, "
    "lighting, identity, and fine detail."
)

NVFP4_T2I = "flux2-klein-9b.text-to-image.bfl-distilled-nvfp4"
FP8_T2I = "flux2-klein-9b.text-to-image.bfl-distilled-fp8"
BF16_T2I = "flux2-klein-9b.text-to-image.native-distilled-bf16"
NVFP4_I2I = "flux2-klein-9b.image-to-image.bfl-distilled-nvfp4"
FP8_I2I = "flux2-klein-9b.image-to-image.bfl-distilled-fp8"
BF16_I2I = "flux2-klein-9b.image-to-image.native-distilled-bf16"


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    recipes: tuple[str, ...]
    operation: str
    repeat: int = 1


SCENARIOS: dict[str, tuple[RunSpec, ...]] = {
    "t2i-smoke": (RunSpec("recommended", (NVFP4_T2I,), "t2i"),),
    "t2i-warm": (RunSpec("recommended-warm", (NVFP4_T2I,), "t2i", repeat=3),),
    "t2i-switch": (
        RunSpec("recommended-before-switch", (NVFP4_T2I,), "t2i"),
        RunSpec("fallback-switch", (FP8_T2I,), "t2i"),
        RunSpec("recommended-after-switch", (NVFP4_T2I,), "t2i"),
    ),
    "t2i-family": (RunSpec("family", (NVFP4_T2I, FP8_T2I, BF16_T2I), "t2i"),),
    "i2i-smoke": (RunSpec("recommended", (NVFP4_I2I,), "i2i"),),
    "i2i-warm": (RunSpec("recommended-warm", (NVFP4_I2I,), "i2i", repeat=3),),
    "i2i-switch": (
        RunSpec("recommended-before-switch", (NVFP4_I2I,), "i2i"),
        RunSpec("fallback-switch", (FP8_I2I,), "i2i"),
        RunSpec("recommended-after-switch", (NVFP4_I2I,), "i2i"),
    ),
    "i2i-family": (RunSpec("family", (NVFP4_I2I, FP8_I2I, BF16_I2I), "i2i"),),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-seed Klein 9B generation acceptance scenarios."
    )
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Required by image-to-image scenarios; uploaded through /v1/assets.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failure; useful for best-effort reference sweeps.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = SCENARIOS[args.scenario]
    requires_source = any(spec.operation == "i2i" for spec in specs)
    source_image = args.source_image.resolve() if args.source_image else None
    if requires_source and (source_image is None or not source_image.is_file()):
        raise SystemExit("--source-image must name an existing file for an I2I scenario")

    repository = Path(__file__).resolve().parents[1]
    harness = repository / "scripts" / "hardware-study.py"
    run_root = (
        args.run_root
        or repository
        / "hardware-study-runs"
        / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-klein9b-{args.scenario}"
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "latentslate-klein9b-acceptance-v1",
        "scenario": args.scenario,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "source_image": str(source_image) if source_image else None,
        "runs": [],
    }
    summary_path = run_root / "scenario.json"

    failed = False
    for index, spec in enumerate(specs, start=1):
        run_dir = run_root / f"{index:02d}-{spec.name}"
        command = [
            sys.executable,
            str(harness),
            "--base-url",
            args.base_url,
            "--run-dir",
            str(run_dir),
            "--timeout",
            str(args.timeout),
            "--repeat",
            str(spec.repeat),
            "--seed",
            str(SEED),
            "--prompt",
            T2I_PROMPT if spec.operation == "t2i" else I2I_PROMPT,
            "--input",
            f"width={WIDTH}",
            "--input",
            f"height={HEIGHT}",
        ]
        for recipe in spec.recipes:
            command.extend(("--recipe", recipe))
        if spec.operation == "i2i":
            command.extend(("--asset", f"source_image={source_image}"))
        if args.preflight_only:
            command.append("--preflight-only")

        print(f"\n[{index}/{len(specs)}] {spec.name}")
        completed = subprocess.run(command, cwd=repository, check=False)
        summary["runs"].append(
            {
                "name": spec.name,
                "recipes": list(spec.recipes),
                "repeat": spec.repeat,
                "manifest": str(run_dir / "manifest.json"),
                "exit_code": completed.returncode,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            failed = True
            if not args.keep_going:
                break

    print(f"\nScenario record: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
