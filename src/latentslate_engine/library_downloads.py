"""Download selection across saved recipes and official built-in dependencies."""

from copy import deepcopy
from pathlib import Path

from .authoring import artifact_dependencies
from .authoring_store import StoreError
from .bootstrap import install, selected_assets, target_path
from .bootstrap import plan as bootstrap_plan


def plan_downloads(materializer, selections):
    home = materializer.cache.root.parent
    dependencies = {}
    recipes = []
    for document, builtin in selections:
        base = materializer.plan([document], verify_cache=False)
        recipes.extend(base["recipes"])
        candidates = (
            selected_assets([document["operation"].split(".")[0]]) if builtin else []
        )
        slots = {item["path"]: item for item in artifact_dependencies(document)}
        for entry in base["dependencies"]:
            matches = []
            if candidates and entry["reference"]["source"] == "local":
                local = Path(entry["reference"]["path"])
                directory = (
                    slots[entry["consumers"][0]["path"]]["requirements"]["kind"]
                    == "directory"
                )
                for asset in candidates:
                    target = target_path(home, asset["path"])
                    if target == local or directory and target.is_relative_to(local):
                        matches.append(asset)
                # Klein's tokenizer also reads its sibling text-encoder config.
                if (
                    matches
                    and document["operation"].startswith("flux2_klein9b.")
                    and directory
                ):
                    matches.extend(
                        asset
                        for asset in candidates
                        if target_path(home, asset["path"])
                        == local.parent / "text_encoder" / "config.json"
                    )
            if matches:
                report = bootstrap_plan(home, assets=matches)
                entries = []
                for asset in report["assets"]:
                    state = asset["status"]
                    entries.append(
                        {
                            "id": "sha256:" + asset["reference"]["sha256"],
                            "reference": asset["reference"],
                            "consumers": entry["consumers"],
                            "size_bytes": asset["size_bytes"],
                            "status": "unresolved"
                            if state == "conflict"
                            else "needs_install"
                            if state == "cached"
                            else "needs_download"
                            if state == "missing"
                            else "resolved_local",
                            "message": "Existing built-in file conflicts with the official source; it will not be overwritten."
                            if state == "conflict"
                            else None,
                            "bootstrap_assets": [asset],
                        }
                    )
            else:
                entries = [entry]
            for item in entries:
                previous = dependencies.get(item["id"])
                if previous is None:
                    dependencies[item["id"]] = deepcopy(item)
                    continue
                previous["consumers"].extend(item["consumers"])
                previous["size_bytes"] = previous["size_bytes"] or item["size_bytes"]
                assets = previous.setdefault("bootstrap_assets", [])
                assets.extend(
                    a
                    for a in item.get("bootstrap_assets", [])
                    if a["path"] not in {b["path"] for b in assets}
                )
                states = {previous["status"], item["status"]}
                if "unresolved" in states:
                    previous.update(
                        status="unresolved",
                        message=previous["message"] or item["message"],
                    )
                elif "needs_download" in states:
                    previous["status"] = (
                        "needs_install"
                        if states & {"cached", "resolved_local", "needs_install"}
                        else "needs_download"
                    )
                elif "needs_install" in states:
                    previous["status"] = "needs_install"
    entries = list(dependencies.values())
    missing = [e for e in entries if e["status"] == "needs_download"]
    return {
        "recipes": recipes,
        "dependencies": entries,
        "summary": {
            "recipes": len(recipes),
            "unique_artifacts": len(entries),
            "missing": len(missing),
            "install": sum(e["status"] == "needs_install" for e in entries),
            "available": sum(
                e["status"] in {"cached", "resolved_local"} for e in entries
            ),
            "unresolved": sum(e["status"] == "unresolved" for e in entries),
            "download_bytes_known": sum(e["size_bytes"] or 0 for e in missing),
            "unknown_sizes": sum(e["size_bytes"] is None for e in missing),
        },
    }


def start_downloads(materializer, selections):
    # Snapshot selected saved definitions before starting the existing task owner.
    selections = deepcopy(selections)

    def action(cancel, progress, update):
        report = plan_downloads(materializer, selections)
        pending = [
            e
            for e in report["dependencies"]
            if e["status"] in {"needs_download", "needs_install"}
        ]
        completed_bytes = 0
        update(
            plan=deepcopy(report),
            completed_files=0,
            file_count=len(pending),
            overall_bytes=0,
        )
        for completed_files, entry in enumerate(pending, start=1):
            cancel()
            received_bytes = 0

            def received(count, total, previous_bytes=completed_bytes):
                nonlocal received_bytes
                received_bytes = count
                progress(count, total)
                update(overall_bytes=previous_bytes + count)

            ref = entry["reference"]
            update(
                stage=ref.get("file")
                or ref.get("url")
                or "Civitai file " + str(ref.get("file_id")),
                bytes_downloaded=0,
                total_bytes=entry["size_bytes"],
            )
            entry["status"] = "downloading"
            update(plan=deepcopy(report))
            try:
                if entry.get("bootstrap_assets"):
                    install(
                        materializer.cache.root.parent,
                        assets=entry["bootstrap_assets"],
                        source=materializer.sources["huggingface"],
                        cancel=cancel,
                        download_progress=received,
                        progress=lambda message: update(stage=message),
                    )
                else:
                    source = materializer.sources[ref["source"]]
                    remote = source.describe(
                        {k: v for k, v in ref.items() if k not in {"source", "sha256"}}
                    )
                    cancel()
                    if remote.sha256 is not None and remote.sha256 != ref["sha256"]:
                        raise ValueError("Source SHA-256 differs from the saved recipe")
                    materializer._sizes[ref["sha256"]] = remote.size
                    materializer.cache.acquire(
                        ref["sha256"],
                        lambda write, cancel, source=source, remote=remote: (
                            source.download(remote, write, cancel)
                        ),
                        cancel,
                        received,
                        size=remote.size,
                    )
                entry.update(status="cached", message=None)
            except Exception as error:
                entry.update(status="failed", message=str(error) or "Download canceled")
                update(plan=deepcopy(report))
                raise
            completed_bytes += received_bytes
            update(
                plan=deepcopy(report),
                completed_files=completed_files,
                overall_bytes=completed_bytes,
            )
        final = plan_downloads(materializer, selections)
        blocked = {
            consumer["id"]
            for entry in final["dependencies"]
            if entry["status"] not in {"cached", "resolved_local"}
            for consumer in entry["consumers"]
        }
        update(plan=final)
        return {
            "ready": len(final["recipes"]) - len(blocked),
            "needs_attention": len(blocked),
        }

    return materializer._start("library_downloads", action)


def selected_recipes(value, builtins, store):
    if set(value) != {"builtin_keys", "recipe_ids"}:
        raise StoreError(422, "Select builtin_keys and recipe_ids")
    keys, ids = value["builtin_keys"], value["recipe_ids"]
    if (
        not all(
            isinstance(items, list) and all(isinstance(item, str) for item in items)
            for items in (keys, ids)
        )
        or len(keys) + len(ids) > 256
    ):
        raise StoreError(422, "Select at most 256 recipes")
    if any(key not in builtins for key in keys):
        raise StoreError(404, "Selected built-in recipe no longer exists")
    return [(builtins[key], True) for key in dict.fromkeys(keys)] + [
        (store.read(key)["document"], False) for key in dict.fromkeys(ids)
    ]
