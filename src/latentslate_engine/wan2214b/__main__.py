from __future__ import annotations

import argparse

from .pipeline import WanSession, result_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Wan 2.2 14B LightX2V T2V")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=923510416338945)
    parser.add_argument("--capture-seams", action="store_true")
    args = parser.parse_args()
    session = WanSession()
    print(
        result_json(
            session.generate(
                args.output, seed=args.seed, capture_seams=args.capture_seams
            )
        )
    )


if __name__ == "__main__":
    main()
