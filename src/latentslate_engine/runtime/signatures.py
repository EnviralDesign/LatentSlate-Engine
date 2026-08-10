from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SAMPLE_BYTES = 64 * 1024
_FULL_HASH_LIMIT = 4 * _SAMPLE_BYTES
_IGNORED_DIRECTORY_NAMES = {".cache", ".git", "__pycache__"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sampled_file_digest(path: Path, size: int) -> tuple[str, str]:
    """Hash small files fully and large weights at deterministic offsets.

    Model checkpoints can be tens of gigabytes, so a full hash during every
    runtime-plan resolution is not practical. Large files are sampled at the
    beginning, midpoint, and end while also committing their size and offsets
    to the digest. This detects same-size/same-mtime replacements in the areas
    most likely to contain headers and changed payload while keeping startup
    work bounded to roughly 192 KiB per shard.
    """

    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    if size <= _FULL_HASH_LIMIT:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "full"

    offsets = (0, max(0, size // 2 - _SAMPLE_BYTES // 2), size - _SAMPLE_BYTES)
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(_SAMPLE_BYTES)
            digest.update(offset.to_bytes(8, "big", signed=False))
            digest.update(len(chunk).to_bytes(8, "big", signed=False))
            digest.update(chunk)
    return digest.hexdigest(), "sampled"


def _stable_file_record(path: Path, *, relative_path: str) -> dict[str, Any]:
    """Read a stable file record, retrying once if a writer races the scan."""

    for attempt in range(2):
        before = path.stat()
        digest, digest_mode = _sampled_file_digest(path, before.st_size)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return {
                "relative_path": relative_path,
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "digest": digest,
                "digest_mode": digest_mode,
            }
        if attempt == 1:
            break
    raise RuntimeError(f"Model component changed while it was being fingerprinted: {path}")


def _iter_component_files(root: Path):
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.is_file():
            yield path, relative.as_posix()


def path_signature(path: Path) -> dict[str, Any]:
    """Fingerprint a loaded file or model repository deterministically.

    Directory signatures commit every nested file's relative path, size,
    modification time, and content digest. They intentionally ignore local Hub
    cache bookkeeping because those files are not loaded by Diffusers.
    """

    resolved = path.resolve(strict=True)
    if resolved.is_file():
        record = _stable_file_record(resolved, relative_path=resolved.name)
        return {
            "path": resolved.as_posix(),
            "kind": "file",
            **record,
        }
    if not resolved.is_dir():
        raise ValueError(f"Runtime component is neither a file nor a directory: {resolved}")

    records = [
        _stable_file_record(file_path, relative_path=relative_path)
        for file_path, relative_path in _iter_component_files(resolved)
    ]
    if not records:
        raise ValueError(f"Runtime component directory contains no model files: {resolved}")
    return {
        "path": resolved.as_posix(),
        "kind": "directory",
        "files": len(records),
        "bytes": sum(int(record["size"]) for record in records),
        "manifest_digest": hashlib.sha256(_canonical_json(records)).hexdigest(),
    }
