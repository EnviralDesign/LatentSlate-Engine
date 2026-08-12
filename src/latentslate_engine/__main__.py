from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from contextlib import redirect_stdout
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 8765


class _NoRedirectHandler(HTTPRedirectHandler):
    """Turn every redirect into an HTTP error instead of following it."""

    def http_error_302(self, request, response, code, message, headers):  # type: ignore[no-untyped-def]
        raise HTTPError(request.full_url, code, "Redirects are not allowed", headers, response)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def default_authoring_url() -> str:
    """Return the resource authoring page for the default local Engine server."""

    return f"http://{DEFAULT_SERVE_HOST}:{DEFAULT_SERVE_PORT}/authoring/"


def normalize_authoring_url(url: str) -> str:
    """Accept only a local Engine origin and its authoring page path."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Author URL must be a loopback HTTP(S) URL at /authoring/") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Author URL must be a loopback HTTP(S) URL at /authoring/")

    if parsed.path not in {"", "/", "/authoring", "/authoring/"}:
        raise ValueError("Author URL must be a loopback HTTP(S) URL at /authoring/")
    del port  # Accessing it above validates a malformed port before opening anything.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, "/authoring/", "", ""))


def open_authoring_page(url: str) -> None:
    """Verify an already-running Engine page before asking the OS to open it."""

    url = normalize_authoring_url(url)
    try:
        opener = build_opener(_NoRedirectHandler())
        with opener.open(Request(url, method="GET"), timeout=2) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Engine is not reachable at {url}") from exc
    if not webbrowser.open(url):
        raise RuntimeError(f"Could not open the default browser for {url}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="latentslate-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the LatentSlate Engine API")
    serve.add_argument("--host", default=DEFAULT_SERVE_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_SERVE_PORT)
    serve.add_argument("--reload", action="store_true")

    author = subparsers.add_parser("author", help="Open the local Resource Editor in a browser")
    author.add_argument(
        "--url", default=default_authoring_url(), help="Existing Engine authoring page URL"
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect runtime prerequisites, hardware, authentication, and legacy bundle state",
    )
    doctor.add_argument("--json", action="store_true")

    bundles = subparsers.add_parser("bundles", help="Inspect or install model bundles")
    bundle_commands = bundles.add_subparsers(dest="bundle_command", required=True)
    bundle_commands.add_parser("list", help="List canonical model bundles")
    install = bundle_commands.add_parser("install", help="Download one canonical bundle")
    install.add_argument("bundle_id")

    resources = subparsers.add_parser(
        "resources",
        help="Inspect, author, validate, and fetch model or LoRA resources",
    )
    resource_commands = resources.add_subparsers(dest="resource_command", required=True)
    resource_list = resource_commands.add_parser(
        "list", help="List installed resources and declared acquisition targets"
    )
    resource_list.add_argument(
        "--json", action="store_true", help="Emit the structured resource catalog"
    )
    resource_show = resource_commands.add_parser("show", help="Inspect one resource")
    resource_show.add_argument("resource_id")
    resource_show.add_argument(
        "--json", action="store_true", help="Emit the structured resource detail"
    )
    from .authoring.cli import configure_resource_authoring_cli

    configure_resource_authoring_cli(resource_commands)

    variants = subparsers.add_parser("variants", help="Inspect legacy variant aliases")
    variant_commands = variants.add_subparsers(dest="variant_command", required=True)
    variant_commands.add_parser("list", help="List loaded variants and authoring errors")
    variant_commands.add_parser("validate", help="Validate all variant/recipe files")

    recipes = subparsers.add_parser("recipes", help="Inspect and author runnable recipes")
    recipe_commands = recipes.add_subparsers(dest="recipe_command", required=True)
    recipe_list = recipe_commands.add_parser(
        "list", help="List runnable recipes and catalog errors"
    )
    recipe_list.add_argument(
        "--json", action="store_true", help="Emit the structured recipe catalog"
    )
    recipe_validate = recipe_commands.add_parser("validate", help="Validate recipe catalogs or a draft file")
    recipe_validate.add_argument(
        "--json", action="store_true", help="Emit the structured validation result"
    )
    recipe_show = recipe_commands.add_parser(
        "show", help="Inspect one recipe and its resource closure"
    )
    recipe_show.add_argument("recipe_key")
    recipe_show.add_argument(
        "--json", action="store_true", help="Emit the structured recipe detail"
    )
    recipe_plan = recipe_commands.add_parser(
        "plan", help="Plan one or more recipes (human summary by default)"
    )
    recipe_plan.add_argument("recipe_keys", nargs="+")
    recipe_plan.add_argument("--json", action="store_true", help="Emit the structured recipe plan")
    recipe_install = recipe_commands.add_parser(
        "install", help="Install one or more recipe closures through the safe deployment pipeline"
    )
    recipe_install.add_argument("recipe_keys", nargs="+")
    recipe_install.add_argument(
        "--json", action="store_true", help="Emit the structured install result"
    )
    from .authoring.cli import configure_recipe_authoring_cli

    configure_recipe_authoring_cli(recipe_commands, recipe_validate)

    deployments = subparsers.add_parser(
        "deployments",
        help="Inspect recipe selection profiles and exact resource closure",
    )
    deployment_commands = deployments.add_subparsers(
        dest="deployment_command",
        required=True,
    )
    deployment_profiles = deployment_commands.add_parser(
        "profiles", help="List saved reusable recipe selections"
    )
    deployment_profiles.add_argument(
        "--json", action="store_true", help="Emit the structured profile catalog"
    )
    deployment_plan = deployment_commands.add_parser(
        "plan", help="Plan one profile (human summary by default)"
    )
    deployment_plan.add_argument("profile_key")
    deployment_plan.add_argument(
        "--json",
        action="store_true",
        help="Emit the full structured deployment plan for automation",
    )
    deployment_lock = deployment_commands.add_parser("lock", help="Generate one lock")
    deployment_lock.add_argument("profile_key")
    deployment_install = deployment_commands.add_parser(
        "install", help="Install the exact missing resource closure for one saved profile"
    )
    deployment_install.add_argument("profile_key")
    deployment_install.add_argument(
        "--json", action="store_true", help="Emit the structured install result"
    )

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

    if args.command == "author":
        try:
            open_authoring_page(args.url)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.command == "doctor":
        from .cli_presentation import print_human
        from .doctor import collect_report, format_report

        report = collect_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_human(format_report(report))
        raise SystemExit(0 if report["ready_for_inference"] else 1)

    from .config import Settings

    settings = Settings.from_env()
    settings.ensure_directories()
    if args.command == "data":
        print(settings.home)
        return

    if args.command in {"resources", "variants", "recipes", "deployments"}:
        from .cli_presentation import print_human
        from .tools import default_registry

        registry = default_registry(settings, emit_warnings=False)
        if args.command == "resources":
            from .authoring.cli import handle_resource_authoring_cli
            from .cli_product import (
                format_resource_catalog,
                format_resource_detail,
                resource_detail_payload,
            )

            if handle_resource_authoring_cli(args, settings, registry, resources):
                return
            if args.resource_command == "show":
                try:
                    payload = resource_detail_payload(registry, args.resource_id)
                except KeyError as exc:
                    resources.error(str(exc))
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    print_human(format_resource_detail(payload))
                return
            payload = {
                "resources": [
                    resource.model_dump(mode="json") for resource in registry.resources.resources
                ],
                "errors": registry.resources.errors,
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print_human(
                    format_resource_catalog(registry.resources.resources, registry.resources.errors)
                )
            return
        if args.command == "recipes":
            from .authoring.cli import handle_recipe_authoring_cli
            from .cli_product import (
                concise_cli_error,
                format_recipe_catalog,
                format_recipe_detail,
                format_recipe_install,
                format_recipe_validation,
                recipe_detail_payload,
            )
            from .deployment_summary import format_recipe_selection_plan
            from .recipes import build_recipe_selection_plan, recipe_catalog

            if handle_recipe_authoring_cli(args, settings, registry, recipes):
                return
            payload = recipe_catalog(settings, registry)
            if args.recipe_command == "list":
                if args.json:
                    print(json.dumps(payload.model_dump(mode="json"), indent=2))
                else:
                    print_human(format_recipe_catalog(payload))
                return
            if args.recipe_command == "validate":
                if args.json:
                    print(json.dumps(payload.model_dump(mode="json"), indent=2))
                else:
                    print_human(format_recipe_validation(payload))
                if payload.errors:
                    raise SystemExit(1)
                return
            try:
                if args.recipe_command == "show":
                    detail = recipe_detail_payload(settings, registry, args.recipe_key)
                    if args.json:
                        print(json.dumps(detail, indent=2))
                    else:
                        print_human(format_recipe_detail(detail))
                    return
                if args.recipe_command == "plan":
                    selection_plan = build_recipe_selection_plan(
                        settings, registry, args.recipe_keys
                    )
                    if args.json:
                        print(json.dumps(selection_plan.model_dump(mode="json"), indent=2))
                    else:
                        print_human(format_recipe_selection_plan(selection_plan))
                    return
                from .acquisition.deployment_install import install_recipe_selection

                if args.json:
                    # Third-party download helpers may emit progress on stdout.
                    # Machine consumers receive one JSON document on stdout only.
                    with redirect_stdout(sys.stderr):
                        install_result = install_recipe_selection(
                            settings, registry, args.recipe_keys
                        )
                else:
                    from .cli_install_progress import HumanInstallProgress

                    with HumanInstallProgress() as progress:
                        install_result = install_recipe_selection(
                            settings, registry, args.recipe_keys, progress=progress
                        )
            except (KeyError, ValueError) as exc:
                recipes.error(concise_cli_error(str(exc)))
            if args.json:
                print(json.dumps(install_result.model_dump(mode="json"), indent=2))
            else:
                print_human(format_recipe_install(install_result, args.recipe_keys))
            return
        if args.command == "deployments":
            from .cli_product import (
                concise_cli_error,
                format_deployment_install,
                format_deployment_profiles,
            )
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

                    if args.json:
                        # Keep downloader progress separate from structured stdout.
                        with redirect_stdout(sys.stderr):
                            payload = install_deployment_profile(
                                settings, registry, args.profile_key
                            )
                    else:
                        from .cli_install_progress import HumanInstallProgress

                        with HumanInstallProgress() as progress:
                            payload = install_deployment_profile(
                                settings, registry, args.profile_key, progress=progress
                            )
            except (KeyError, ValueError) as exc:
                deployments.error(concise_cli_error(str(exc)))
            if args.deployment_command == "profiles" and not args.json:
                print_human(format_deployment_profiles(payload))
            elif args.deployment_command == "plan" and not args.json:
                from .deployment_summary import format_deployment_plan

                print_human(format_deployment_plan(payload))
            elif args.deployment_command == "install" and not args.json:
                print_human(format_deployment_install(payload, args.profile_key))
            else:
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
