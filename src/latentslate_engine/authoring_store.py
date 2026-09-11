"""Immutable JSON revisions with an atomically replaced head per recipe."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .authoring import (
    canonical_bytes,
    definition_hash,
    parse_document,
    validate_document,
)
from .catalog import user_request_schema_hash


class StoreError(ValueError):
    def __init__(self, status: int, message: str, validation: dict | None = None):
        super().__init__(message)
        self.status = status
        self.validation = validation


@contextmanager
def _filesystem_writer_lock(root: Path):
    """Non-blocking OS locks release on process exit, including a crashed writer."""
    with (root / ".writer.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise StoreError(
                409, "Another authoring write is in progress; reload and retry"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RecipeStore:
    """A local filesystem store; built-in IDs are reserved and never writable."""

    def __init__(self, root: Path, builtin_ids=()):
        self.root = root
        self.builtin_ids = frozenset(builtin_ids)
        self._lock = threading.Lock()

    def _directory(self, recipe_id: str) -> Path:
        try:
            if str(uuid.UUID(recipe_id)) != recipe_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise StoreError(422, "Expected a canonical recipe UUID") from None
        if recipe_id in self.builtin_ids:
            raise StoreError(
                409, "Built-in definitions are immutable; duplicate one to edit"
            )
        return self.root / recipe_id

    def _published_history(self, recipe_id: str):
        directory = self._directory(recipe_id)
        try:
            head = json.loads((directory / "head.json").read_bytes())
            number = head["revision"]
            while number is not None:
                record = json.loads(
                    (directory / "revisions" / f"{number}.json").read_bytes()
                )
                yield record
                number = record["parent_revision"]
        except FileNotFoundError:
            raise StoreError(404, "Recipe or revision not found") from None

    def read(self, recipe_id: str, revision: int | None = None) -> dict:
        for record in self._published_history(recipe_id):
            if (
                revision is None
                or type(revision) is int
                and record["revision"] == revision
            ):
                return record
        raise StoreError(404, "Recipe or revision not found")

    def publication(self, recipe_id: str) -> dict:
        """Read publication state and its exact immutable head together."""
        directory = self._directory(recipe_id)
        try:
            head = json.loads((directory / "head.json").read_bytes())
            record = json.loads(
                (directory / "revisions" / f"{head['revision']}.json").read_bytes()
            )
        except FileNotFoundError:
            raise StoreError(404, "Recipe not found") from None
        schema = head.get("schema")
        if schema is None:
            schema = {
                "revision": 1,
                "hash": user_request_schema_hash(record["document"]),
            }
        return {
            "enabled": head.get("enabled", False),
            "schema": schema,
            "record": record,
        }

    def set_enabled(self, recipe_id: str, enabled: bool) -> dict:
        """Change host publication without rewriting the recipe or its revision."""
        if type(enabled) is not bool:
            raise StoreError(422, "enabled must be a boolean")
        directory = self._directory(recipe_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, _filesystem_writer_lock(self.root):
            try:
                head = json.loads((directory / "head.json").read_bytes())
            except FileNotFoundError:
                raise StoreError(404, "Recipe not found") from None
            if enabled and "schema" not in head:
                head["schema"] = self.publication(recipe_id)["schema"]
            _atomic_json(directory / "head.json", {**head, "enabled": enabled})
            return self.publication(recipe_id)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        return [
            self.read(path.parent.name)
            for path in sorted(self.root.glob("*/head.json"))
        ]

    def revisions(self, recipe_id: str) -> list[dict]:
        return list(reversed(list(self._published_history(recipe_id))))

    def preview_import(self, value: object) -> dict:
        """Validate and classify a definition without changing the local store."""
        validation = validate_document(value)
        document = parse_document(value) if validation["document_valid"] else None
        result = {"document": document, "validation": validation, "status": "invalid"}
        if not validation["recipe_compiles"]:
            return result
        if document["id"] in self.builtin_ids:
            return {**result, "status": "builtin"}
        try:
            current = self.read(document["id"])
        except StoreError as error:
            if error.status != 404:
                raise
            return {**result, "status": "new"}
        return {
            **result,
            "status": "identical"
            if canonical_bytes(current["document"]) == canonical_bytes(document)
            else "conflict",
            "current_revision": current["revision"],
        }

    def import_document(self, value: object, *, as_copy: bool = False) -> dict:
        """Import only a new identity; collisions never append to existing history."""
        preview = self.preview_import(value)
        status = preview["status"]
        if status == "invalid":
            raise StoreError(422, "Recipe policy is invalid", preview["validation"])
        if status == "identical":
            return {"status": "already_present", "preview": preview}
        if status in {"conflict", "builtin"} and not as_copy:
            raise StoreError(409, "Recipe UUID already exists; import as a copy")
        document = preview["document"]
        if as_copy:
            document["id"] = str(uuid.uuid4())
        # save's base=None comparison is the atomic guard if another writer
        # creates this UUID after preview. Imports never use a foreign revision.
        record = self.save(document, base_revision=None)
        return {"status": "copied" if as_copy else "imported", "record": record}

    def save(self, value: object, *, base_revision: int | None) -> dict:
        validation = validate_document(value)
        if not validation["document_valid"] or not validation["recipe_compiles"]:
            raise StoreError(422, "Recipe policy is invalid", validation)
        document = parse_document(value)
        directory = self._directory(document["id"])
        if base_revision is not None and (
            type(base_revision) is not int or base_revision < 1
        ):
            raise StoreError(422, "base_revision must be a positive integer")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, _filesystem_writer_lock(self.root):
            try:
                previous = self.read(document["id"])
            except StoreError as error:
                if error.status != 404:
                    raise
                previous = None
            actual = previous["revision"] if previous else None
            if actual != base_revision:
                raise StoreError(409, "Recipe head changed; reload before saving")
            revisions = directory / "revisions"
            revisions.mkdir(parents=True, exist_ok=True)
            # A crash before publishing head may leave an orphan. Never overwrite
            # its bytes; reserve the next monotonic number under the writer lock.
            number = (
                max((int(path.stem) for path in revisions.glob("*.json")), default=0)
                + 1
            )
            record = {
                "revision": number,
                "parent_revision": actual,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "definition_hash": definition_hash(document),
                "document": document,
            }
            # Host schema lineage is separate from canonical definition identity.
            # Public inputs, canvas and timing determine request policy;
            # hidden artifacts and display names cannot change that policy.
            previous_head = (
                json.loads((directory / "head.json").read_bytes()) if previous else {}
            )
            head = {
                "revision": number,
                "definition_hash": record["definition_hash"],
                "enabled": previous_head.get("enabled", False),
            }
            if previous_schema := previous_head.get("schema"):
                schema_hash = user_request_schema_hash(document)
                head["schema"] = {
                    "hash": schema_hash,
                    "revision": previous_schema["revision"]
                    + (previous_schema["hash"] != schema_hash),
                }
            _atomic_json(revisions / f"{number}.json", record)
            _atomic_json(directory / "head.json", head)
        return record
