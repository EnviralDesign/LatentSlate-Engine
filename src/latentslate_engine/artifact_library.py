"""Host-local model roots and a disposable, contextual artifact search index."""

from __future__ import annotations

import heapq
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .authoring import OPERATION_OWNERSHIP, OPERATIONS, local_reference
from .authoring_store import StoreError, _atomic_json, _filesystem_writer_lock


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path))


def _score(query: str, name: str, relative: str) -> float | None:
    if not query:
        return 0
    if query == name:
        return 400
    if name.startswith(query):
        return 300
    if query in name:
        return 250 - name.index(query) / max(1, len(name))
    if all(token in relative for token in query.split()):
        return 200
    # Subsequence matching supports abbreviated filenames/paths without a
    # quadratic edit-distance calculation for every indexed path on each key.
    compact = "".join(query.split())
    offset, first = 0, None
    for character in compact:
        found = relative.find(character, offset)
        if found < 0:
            return None
        first = found if first is None else first
        offset = found + 1
    return 100 * len(compact) / max(1, offset - (first or 0))


class ArtifactLibrary:
    """Persist roots, not scans; family metadata owns slot selection semantics."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._lock = threading.RLock()
        self._entries: list[dict] | None = None
        self._signature: tuple = ()
        self._index_info: dict = {}

    def _configured(self) -> list[dict]:
        if not self.config_path.exists():
            return []
        return json.loads(self.config_path.read_bytes())["roots"]

    def roots(self) -> list[dict]:
        return [
            {**root, "available": Path(root["path"]).is_dir()}
            for root in self._configured()
        ]

    def add_root(self, path: object, name: object = None) -> dict:
        if not isinstance(path, str) or not path or not Path(path).is_absolute():
            raise StoreError(
                422, "A model root must be an absolute local directory path"
            )
        if name is not None and not isinstance(name, str):
            raise StoreError(422, "Root name must be a string")
        try:
            local = Path(path).resolve(strict=True)
            if not local.is_dir():
                raise ValueError
        except (OSError, ValueError):
            raise StoreError(422, "The model root directory does not exist") from None
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, _filesystem_writer_lock(self.config_path.parent):
            roots = self._configured()
            if any(_path_key(Path(root["path"])) == _path_key(local) for root in roots):
                raise StoreError(409, "This model root is already registered")
            root = {
                "id": str(uuid.uuid4()),
                "name": name.strip() or None if name else None,
                "path": str(local),
            }
            _atomic_json(
                self.config_path, {"format_version": 1, "roots": [*roots, root]}
            )
            self._entries = None
        return {**root, "available": True}

    def remove_root(self, root_id: str) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, _filesystem_writer_lock(self.config_path.parent):
            roots = self._configured()
            remaining = [root for root in roots if root["id"] != root_id]
            if len(remaining) == len(roots):
                raise StoreError(404, "Model root not found")
            _atomic_json(self.config_path, {"format_version": 1, "roots": remaining})
            self._entries = None

    def refresh(self) -> dict:
        with self._lock:
            return self._scan(self.roots())

    def _scan(self, roots: list[dict]) -> dict:
        started = time.monotonic()
        entries, issues, seen = [], [], set()
        # Prefer the most specific registered root as the display owner when
        # roots overlap. Paths, including aliases, appear only once in results.
        for root in sorted(
            roots, key=lambda item: (-len(Path(item["path"]).parts), item["path"])
        ):
            if not root["available"]:
                issues.append(
                    {
                        "root_id": root["id"],
                        "message": "Registered directory is unavailable",
                    }
                )
                continue
            base = Path(root["path"])

            def add(path: Path, kind: str, *, root=root, base=base):
                try:
                    absolute = path.resolve()
                except OSError as error:
                    issues.append({"root_id": root["id"], "message": str(error)})
                    return
                identity = _path_key(absolute)
                if identity in seen:
                    return
                seen.add(identity)
                relative = str(path.relative_to(base))
                entries.append(
                    {
                        "root_id": root["id"],
                        "relative_path": relative,
                        "absolute_path": str(absolute),
                        "kind": kind,
                        "_name": path.name.casefold(),
                        "_relative": relative.casefold(),
                    }
                )

            def scan_error(error, *, root=root):
                issues.append({"root_id": root["id"], "message": str(error)})

            add(base, "directory")
            for directory, directories, filenames in os.walk(
                base, followlinks=False, onerror=scan_error
            ):
                parent = Path(directory)
                for name in directories:
                    add(parent / name, "directory")
                for name in filenames:
                    add(parent / name, "file")
        self._entries = entries
        self._signature = tuple(
            (root["id"], root["path"], root["available"]) for root in roots
        )
        self._index_info = {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "files": sum(entry["kind"] == "file" for entry in entries),
            "directories": sum(entry["kind"] == "directory" for entry in entries),
            "scan_ms": round((time.monotonic() - started) * 1000, 2),
            "issues": issues,
        }
        return {**self._index_info, "roots": roots}

    def search(
        self, operation: str, field: str, query: str = "", limit: int = 50
    ) -> dict:
        if (
            operation not in OPERATIONS
            or OPERATION_OWNERSHIP[operation].get(field) != "artifact"
        ):
            raise StoreError(
                422, "Search requires a known operation and artifact field"
            )
        if (
            not isinstance(query, str)
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise StoreError(422, "Search requires text and a limit between 1 and 100")
        family, _ = OPERATIONS[operation]
        requirements = family.ARTIFACT_SLOTS[field]
        with self._lock:
            roots = self.roots()
            signature = tuple(
                (root["id"], root["path"], root["available"]) for root in roots
            )
            if self._entries is None or self._signature != signature:
                self._scan(roots)
            by_id = {root["id"]: root for root in roots}
            scored = []
            normalized_query = query.strip().casefold()
            for index, entry in enumerate(self._entries):
                if entry["kind"] != requirements["kind"]:
                    continue
                score = _score(normalized_query, entry["_name"], entry["_relative"])
                if score is not None:
                    scored.append((-score, entry["_relative"], index))
            results = []
            for _, _, index in heapq.nsmallest(limit, scored):
                entry = self._entries[index]
                path = Path(entry["absolute_path"])
                root = by_id[entry["root_id"]]
                exists = (
                    path.is_dir() if entry["kind"] == "directory" else path.is_file()
                )
                checks = [{"path": str(path), "present": exists, "kind": entry["kind"]}]
                checks.extend(
                    {"path": name, "present": (path / name).is_file(), "kind": "file"}
                    for name in requirements.get("required_files", ())
                )
                results.append(
                    {
                        key: value
                        for key, value in entry.items()
                        if not key.startswith("_")
                    }
                    | {
                        "root_name": root["name"]
                        or Path(root["path"]).name
                        or root["path"],
                        "artifact": local_reference(entry["absolute_path"]),
                        "structural_status": "resolved"
                        if all(check["present"] for check in checks)
                        else "unresolved",
                        "checks": checks,
                    }
                )
            return {
                "results": results,
                "total": len(scored),
                "index": dict(self._index_info),
                "model_architecture_checked": False,
            }
