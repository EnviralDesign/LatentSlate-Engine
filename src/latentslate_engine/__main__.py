from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="latentslate-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the LatentSlate Engine API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")

    bundles = subparsers.add_parser("bundles", help="Inspect or install model bundles")
    bundle_commands = bundles.add_subparsers(dest="bundle_command", required=True)
    bundle_commands.add_parser("list", help="List canonical model bundles")
    install = bundle_commands.add_parser("install", help="Download one canonical bundle")
    install.add_argument("bundle_id")

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "latentslate_engine.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    from . import bundles as bundle_registry

    if args.bundle_command == "list":
        print(
            json.dumps(
                [descriptor.model_dump(mode="json") for descriptor in bundle_registry.descriptors()],
                indent=2,
            )
        )
    elif args.bundle_command == "install":
        print(bundle_registry.install(args.bundle_id))


if __name__ == "__main__":
    main()
