from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import Klein9BIdentity, Klein9BRuntime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical FLUX.2 Klein 9B distilled T2I"
    )
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = Klein9BIdentity.from_paths(
        args.diffusion, args.text_encoder, args.vae, args.tokenizer
    )
    with Klein9BRuntime() as runtime:
        for index, seed in enumerate(args.seed):
            output = args.output.with_stem(f"{args.output.stem}-{index:02d}-{seed}")
            result = runtime.generate(identity, args.prompt, seed, output)
            print(
                json.dumps(
                    {
                        "output": str(result.output),
                        "seconds": result.elapsed_seconds,
                        "conditioning_reused": result.conditioning_reused,
                        "models_reused": result.models_reused,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
