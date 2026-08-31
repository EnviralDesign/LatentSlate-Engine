"""Content identities shared by proven request-derived caches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FileContentIdentity:
    """Path-independent identity for one request-owned file payload."""

    path: Path = field(compare=False)
    size: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> FileContentIdentity:
        resolved = Path(path).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return cls(path=resolved, size=size, sha256=digest.hexdigest())
