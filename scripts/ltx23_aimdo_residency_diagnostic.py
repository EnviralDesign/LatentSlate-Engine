"""Opt-in real-hardware LTX 2.3 Gemma AIMDO residency diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from latentslate_engine.runtime.ltx23_aimdo_diagnostic import (
    run_ltx23_aimdo_residency_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-artifact", type=Path, required=True)
    parser.add_argument(
        "--text-support-root",
        type=Path,
        required=True,
        help="Installed pipeline-support text_encoder directory containing config.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--i-understand-this-loads-gemma",
        action="store_true",
        help="Required guard: materializes the roughly 10 GiB CPU text model",
    )
    args = parser.parse_args()
    if not args.i_understand_this_loads_gemma:
        parser.error("--i-understand-this-loads-gemma is required")
    result = run_ltx23_aimdo_residency_diagnostic(
        args.text_artifact,
        args.text_support_root,
        device=args.device,
        progress=lambda message: print(f"[ltx23-aimdo] {message}", file=sys.stderr, flush=True),
        # This script is a dedicated one-shot GPU child. A poisoned VBAR must
        # bypass Python/native finalizers exactly like the production worker.
        hard_exit=os._exit,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
