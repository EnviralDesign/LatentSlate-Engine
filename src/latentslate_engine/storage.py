from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import UUID, uuid4, uuid5

from .config import Settings

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_ASSET_NAMESPACE = UUID("706d259b-8fcf-54a1-b7ca-8b8a6d7ed245")


def safe_filename(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    sanitized = _SAFE_FILENAME.sub("_", name).strip("._")
    return sanitized or fallback


@dataclass(frozen=True, slots=True)
class StoredAsset:
    id: UUID
    filename: str
    content_type: str | None
    size_bytes: int
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    id: UUID
    filename: str
    content_type: str
    path: Path
    role: str
    media_type: str
    metadata: dict[str, object]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._asset_lock = RLock()
        settings.ensure_directories()

    def store_asset(
        self,
        stream: BinaryIO,
        filename: str | None,
        content_type: str | None,
        max_bytes: int,
    ) -> StoredAsset:
        incoming_root = self.settings.assets_dir / ".incoming"
        incoming_root.mkdir(parents=True, exist_ok=True)
        incoming_id = uuid4()
        safe_name = safe_filename(filename, f"asset-{incoming_id}")
        incoming_path = incoming_root / f"{incoming_id}.upload"
        total = 0
        digest = hashlib.sha256()
        try:
            with incoming_path.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    output.write(chunk)

            sha256 = digest.hexdigest()
            asset_id = uuid5(_ASSET_NAMESPACE, sha256)
            folder = self.settings.assets_dir / str(asset_id)
            with self._asset_lock:
                if folder.is_dir():
                    files = [path for path in folder.iterdir() if path.is_file()]
                    if len(files) != 1:
                        raise FileNotFoundError(f"Asset {asset_id} is incomplete")
                    incoming_path.unlink(missing_ok=True)
                    existing = files[0]
                    return StoredAsset(
                        asset_id,
                        existing.name,
                        content_type,
                        existing.stat().st_size,
                        existing,
                        sha256,
                    )

                folder.mkdir(parents=True, exist_ok=False)
                path = folder / safe_name
                os.replace(incoming_path, path)
                return StoredAsset(
                    asset_id,
                    safe_name,
                    content_type,
                    total,
                    path,
                    sha256,
                )
        except Exception:
            incoming_path.unlink(missing_ok=True)
            raise

    def resolve_asset(self, asset_id: UUID) -> Path:
        folder = self.settings.assets_dir / str(asset_id)
        if not folder.is_dir():
            raise FileNotFoundError(f"Asset {asset_id} does not exist")
        files = [path for path in folder.iterdir() if path.is_file()]
        if len(files) != 1:
            raise FileNotFoundError(f"Asset {asset_id} is incomplete")
        return files[0]

    def job_folder(self, job_id: UUID) -> Path:
        folder = self.settings.jobs_dir / str(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def artifact_path(self, job_id: UUID, filename: str) -> Path:
        return self.job_folder(job_id) / safe_filename(filename, "artifact.bin")

    def remove_job_folder(self, job_id: UUID) -> None:
        folder = self.settings.jobs_dir / str(job_id)
        if not folder.exists():
            return
        for root, directories, files in os.walk(folder, topdown=False):
            for filename in files:
                Path(root, filename).unlink(missing_ok=True)
            for directory in directories:
                Path(root, directory).rmdir()
        folder.rmdir()
