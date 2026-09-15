"""Explicit, repeatable installation of the official built-in dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .artifact_materialization import ArtifactCache
from .artifact_sources import (
    COMMIT,
    SHA256,
    HuggingFaceSource,
    validate_file,
    validate_reference,
)
from .authoring_store import _filesystem_writer_lock


def manifest() -> list[dict]:
    return json.loads(
        Path(__file__).with_name("builtin-assets.json").read_text(encoding="utf-8")
    )["assets"]


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def target_path(home: Path, relative: str) -> Path:
    validate_file(relative)
    target = home / relative
    if not target.resolve().is_relative_to(home.resolve()):
        raise ValueError("Bootstrap destination escapes the Engine home")
    return target


def selected_assets(families=(), assets=None) -> list[dict]:
    assets = manifest() if assets is None else assets
    known = {family for entry in assets for family in entry["families"]}
    if set(families) - known:
        raise ValueError(
            "Unknown built-in family: " + ", ".join(sorted(set(families) - known))
        )
    selected = [
        entry
        for entry in assets
        if not families or set(families) & set(entry["families"])
    ]
    paths = set()
    for entry in selected:
        validate_file(entry["path"])
        if entry["path"] in paths:
            raise ValueError("Duplicate bootstrap destination")
        paths.add(entry["path"])
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ValueError("Invalid bootstrap byte length")
        ref = entry["reference"]
        if ref["source"] == "builtin_support":
            url = urlsplit(ref["url"])
            parts = url.path.split("/")
            if (
                url.scheme != "https"
                or url.netloc != "raw.githubusercontent.com"
                or len(parts) != 8
                or parts[1:3] != ["Comfy-Org", "ComfyUI"]
                or not COMMIT.fullmatch(parts[3])
                or parts[4:7] != ["comfy", "text_encoders", "qwen25_tokenizer"]
                or parts[7] not in {"vocab.json", "merges.txt", "tokenizer_config.json"}
                or url.query
                or url.fragment
                or not SHA256.fullmatch(ref["sha256"])
            ):
                raise ValueError("Invalid pinned built-in tokenizer source")
        else:
            validate_reference(ref)
            if ref["source"] != "huggingface":
                raise ValueError(
                    "Built-in weights must use pinned Hugging Face sources"
                )
    return selected


def _cache(home: Path, assets=None) -> ArtifactCache:
    destinations = {}
    for entry in selected_assets(assets=assets):
        destinations.setdefault(
            entry["reference"]["sha256"], target_path(home, entry["path"])
        )
    return ArtifactCache(home / "artifacts", destinations=destinations)


def plan(home: Path, families=(), *, verify=False, assets=None) -> dict:
    entries = selected_assets(families, assets)
    cache = _cache(home, assets)
    result = []
    available = set()
    sizes = {}
    for entry in entries:
        ref = entry["reference"]
        digest = ref["sha256"]
        if digest in sizes and sizes[digest] != entry["size_bytes"]:
            raise ValueError("Conflicting lengths for one artifact digest")
        sizes[digest] = entry["size_bytes"]
        path = target_path(home, entry["path"])
        status = "missing"
        if path.exists():
            if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
                status = "conflict"
            elif verify:
                status = "installed" if checksum(path) == digest else "conflict"
            else:
                status = "present_unverified"
        if status == "missing":
            blob = cache.path(digest)
            if blob.is_file() and blob.stat().st_size == entry["size_bytes"]:
                try:
                    status = (
                        "cached" if not verify or cache.verified(digest) else "missing"
                    )
                except ValueError:
                    status = "missing"
        if status in {"installed", "present_unverified", "cached"}:
            available.add(digest)
        result.append({**entry, "status": status})
    return {
        "home": str(home),
        "verified": verify,
        "assets": result,
        "unique_artifacts": len(sizes),
        "required_bytes": sum(sizes.values()),
        "download_bytes": sum(
            size for digest, size in sizes.items() if digest not in available
        ),
        "conflicts": sum(entry["status"] == "conflict" for entry in result),
    }


def _download(entry, source, write, cancel):
    ref = entry["reference"]
    if ref["source"] == "huggingface":
        remote = source.describe(
            {key: ref[key] for key in ("repo", "revision", "file")}
        )
        if remote.sha256 is not None and remote.sha256 != ref["sha256"]:
            raise ValueError("Source metadata differs from the pinned bootstrap digest")
        source.download(remote, write, cancel)
    else:
        # Only the fixed official tokenizer sources above; never recipe URLs.
        with httpx.stream(
            "GET", ref["url"], timeout=30, headers={"Accept-Encoding": "identity"}
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                cancel()
                write(chunk)


def install(
    home: Path,
    families=(),
    *,
    assets=None,
    source=None,
    progress=None,
    cancel=None,
    download_progress=None,
) -> dict:
    home = home.resolve()
    home.mkdir(parents=True, exist_ok=True)
    cache = _cache(home, assets)
    cache.root.mkdir(parents=True, exist_ok=True)
    source = source or HuggingFaceSource()
    progress = progress or (lambda message: None)
    cancel = cancel or (lambda: None)
    download_progress = download_progress or (lambda received, total: None)
    with _filesystem_writer_lock(home / "artifacts"):
        progress("Verifying existing built-in files against the pinned checksums...")
        report = plan(home, families, verify=True, assets=assets)
        if report["conflicts"]:
            raise ValueError(
                "Existing files differ from built-in pins; no files were replaced. Inspect the verified plan."
            )
        if shutil.disk_usage(home).free < report["download_bytes"]:
            raise ValueError("Insufficient free space for the missing built-in assets")
        # Each digest has one canonical destination. Tokenizer directory
        # contracts may require small, independent copies of support files.
        blobs = {}
        for entry in report["assets"]:
            cancel()
            digest = entry["reference"]["sha256"]
            progress(entry["path"])
            try:
                blob = blobs.get(digest) or cache.verified(digest)
            except ValueError:
                blob = None
            if blob is None:
                existing = next(
                    (
                        target_path(home, candidate["path"])
                        for candidate in report["assets"]
                        if candidate["reference"]["sha256"] == digest
                        and candidate["status"] == "installed"
                    ),
                    None,
                )

                def produce(write, cancel, entry=entry, existing=existing):
                    if existing is None:
                        _download(entry, source, write, cancel)
                    else:
                        with existing.open("rb") as handle:
                            while chunk := handle.read(1024 * 1024):
                                cancel()
                                write(chunk)

                downloaded = cache.acquire(
                    digest,
                    produce,
                    cancel,
                    download_progress,
                    size=entry["size_bytes"],
                )
                blob = Path(downloaded["path"])
            blobs[digest] = blob
            target = target_path(home, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as output, blob.open("rb") as source_file:
                    shutil.copyfileobj(source_file, output)
            if target.stat().st_size != entry["size_bytes"] or (
                not os.path.samefile(blob, target) and checksum(target) != digest
            ):
                raise ValueError("Bootstrap destination changed during installation")
        return {
            "home": str(home),
            "installed_paths": len(report["assets"]),
            "unique_artifacts": report["unique_artifacts"],
            "complete": True,
        }


def main():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "install"))
    parser.add_argument(
        "--home",
        type=Path,
        default=Path(
            os.environ.get("LATENTSLATE_ENGINE_HOME", "LatentSlateEngineData")
        ),
    )
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Hash existing files when planning (install always verifies)",
    )
    args = parser.parse_args()
    try:
        result = (
            install(
                args.home,
                args.family,
                progress=lambda text: print(text, file=sys.stderr, flush=True),
            )
            if args.action == "install"
            else plan(args.home.resolve(), args.family, verify=args.verify)
        )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, httpx.HTTPError) as error:
        print(f"Bootstrap failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
