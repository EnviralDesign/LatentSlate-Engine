"""Bounded and atomic JSON files for private worker IPC."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .auth import canonical_json


class WorkerJsonFileError(ValueError):
    """A worker JSON file is absent, unstable, empty, or outside its bound."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WorkerJsonFileError("worker JSON file contains a duplicate object key")
        value[key] = item
    return value


def cleanup_atomic_write_siblings(path: Path) -> None:
    """Remove only temp siblings produced by :func:`atomic_write_json`."""

    target = Path(path)
    prefix = f".{target.name}."
    suffix = ".tmp"
    try:
        entries = target.parent.iterdir()
    except FileNotFoundError:
        return
    for candidate in entries:
        name = candidate.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        pid = name[len(prefix) : -len(suffix)]
        if not pid.isascii() or not pid.isdecimal():
            continue
        candidate.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON through fsync plus same-directory replacement."""

    target = Path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        encoded = canonical_json(value)
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_bounded_json(path: Path, *, maximum_bytes: int) -> Any:
    """Read one present, non-empty JSON file within an explicit byte bound."""

    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
        raise TypeError("worker JSON byte bound must be an integer")
    if maximum_bytes <= 0:
        raise ValueError("worker JSON byte bound must be positive")
    source = Path(path)
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise WorkerJsonFileError("worker JSON path is not a regular file")
            if before.st_size <= 0:
                raise WorkerJsonFileError("worker JSON file is empty")
            if before.st_size > maximum_bytes:
                raise WorkerJsonFileError("worker JSON file exceeds its byte bound")
            raw = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
        current = source.stat()
    except FileNotFoundError as exc:
        raise WorkerJsonFileError("worker JSON file is missing") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or any(
        getattr(after, field) != getattr(current, field) for field in stable_fields
    ):
        raise WorkerJsonFileError("worker JSON file changed during its bounded read")
    if len(raw) != before.st_size or len(raw) > maximum_bytes:
        raise WorkerJsonFileError("worker JSON file changed or exceeds its byte bound")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
