"""Bounded and atomic JSON files for private worker IPC."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .auth import canonical_json


class WorkerJsonFileError(ValueError):
    """A worker JSON file is absent, unstable, empty, or outside its bound."""


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
    if not source.is_file():
        raise WorkerJsonFileError("worker JSON file is missing")
    size = source.stat().st_size
    if size <= 0:
        raise WorkerJsonFileError("worker JSON file is empty")
    if size > maximum_bytes:
        raise WorkerJsonFileError("worker JSON file exceeds its byte bound")
    with source.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) != size or len(raw) > maximum_bytes:
        raise WorkerJsonFileError("worker JSON file changed or exceeds its byte bound")
    return json.loads(raw.decode("utf-8"))
