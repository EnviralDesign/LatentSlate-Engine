from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="latentslate-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the LatentSlate Engine API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect packages, hardware, profiles, authentication, and bundle state",
    )
    doctor.add_argument("--json", action="store_true")

    bundles = subparsers.add_parser("bundles", help="Inspect or install model bundles")
    bundle_commands = bundles.add_subparsers(dest="bundle_command", required=True)
    bundle_commands.add_parser("list", help="List canonical model bundles")
    install = bundle_commands.add_parser("install", help="Download one canonical bundle")
    install.add_argument("bundle_id")

    resources = subparsers.add_parser("resources", help="Inspect file-drop models and LoRAs")
    resource_commands = resources.add_subparsers(dest="resource_command", required=True)
    resource_commands.add_parser("list", help="List discovered resources")

    variants = subparsers.add_parser("variants", help="Inspect legacy variant aliases")
    variant_commands = variants.add_subparsers(dest="variant_command", required=True)
    variant_commands.add_parser("list", help="List loaded variants and authoring errors")
    variant_commands.add_parser("validate", help="Validate all variant/recipe files")

    recipes = subparsers.add_parser("recipes", help="Inspect runnable recipes")
    recipe_commands = recipes.add_subparsers(dest="recipe_command", required=True)
    recipe_commands.add_parser("list", help="List runnable recipes and catalog errors")
    recipe_commands.add_parser("validate", help="Validate all recipe catalogs")

    deployments = subparsers.add_parser(
        "deployments",
        help="Inspect recipe selection profiles and exact resource closure",
    )
    deployment_commands = deployments.add_subparsers(
        dest="deployment_command",
        required=True,
    )
    deployment_commands.add_parser("profiles", help="List deployment profiles")
    deployment_plan = deployment_commands.add_parser("plan", help="Plan one profile")
    deployment_plan.add_argument("profile_key")
    deployment_lock = deployment_commands.add_parser("lock", help="Generate one lock")
    deployment_lock.add_argument("profile_key")
    deployment_install = deployment_commands.add_parser(
        "install", help="Install the exact missing resource closure for one profile"
    )
    deployment_install.add_argument("profile_key")

    data = subparsers.add_parser("data", help="Inspect or initialize Engine data storage")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("path", help="Print the configured Engine data root")
    data_commands.add_parser("init", help="Create the complete Engine data layout")

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

    if args.command == "doctor":
        from .doctor import collect_report, format_report

        report = collect_report()
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        raise SystemExit(0 if report["ready_for_inference"] else 1)

    from .config import Settings

    settings = Settings.from_env()
    settings.ensure_directories()
    if args.command == "data":
        print(settings.home)
        return

    if args.command in {"resources", "variants", "recipes", "deployments"}:
        from .tools import default_registry

        registry = default_registry(settings, emit_warnings=False)
        if args.command == "resources":
            print(
                json.dumps(
                    {
                        "resources": [
                            resource.model_dump(mode="json")
                            for resource in registry.resources.resources
                        ],
                        "errors": registry.resources.errors,
                    },
                    indent=2,
                )
            )
            return
        if args.command == "recipes":
            from .recipes import recipe_catalog

            payload = recipe_catalog(settings, registry)
            print(json.dumps(payload.model_dump(mode="json"), indent=2))
            if args.recipe_command == "validate" and payload.errors:
                raise SystemExit(1)
            return
        if args.command == "deployments":
            from .recipes import (
                build_deployment_lock,
                build_deployment_plan,
                deployment_profile_catalog,
            )

            try:
                if args.deployment_command == "profiles":
                    payload = deployment_profile_catalog(settings)
                elif args.deployment_command == "plan":
                    payload = build_deployment_plan(settings, registry, args.profile_key)
                elif args.deployment_command == "lock":
                    payload = build_deployment_lock(settings, registry, args.profile_key)
                else:
                    from .acquisition.deployment_install import install_deployment_profile

                    payload = install_deployment_profile(settings, registry, args.profile_key)
            except (KeyError, ValueError) as exc:
                deployments.error(str(exc))
            print(json.dumps(payload.model_dump(mode="json"), indent=2))
            return
        payload = {
            "variants": [variant.model_dump(mode="json") for variant in registry.variants],
            "errors": registry.variant_errors,
        }
        print(json.dumps(payload, indent=2))
        if args.variant_command == "validate" and registry.variant_errors:
            raise SystemExit(1)
        return

    from . import bundles as bundle_registry

    if args.bundle_command == "list":
        print(
            json.dumps(
                [
                    descriptor.model_dump(mode="json")
                    for descriptor in bundle_registry.descriptors(settings.model_root, settings)
                ],
                indent=2,
            )
        )
    elif args.bundle_command == "install":
        print(
            f"Preparing bundle installation for {args.bundle_id}...",
            file=sys.stderr,
            flush=True,
        )
        print(bundle_registry.install(args.bundle_id, settings.model_root, settings))


if __name__ == "__main__":
    main()
