"""Explicit recipe dependency acquisition, above local-only inference recipes."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from pathlib import Path

from .artifact_sources import SHA256, HuggingFaceSource, hf_locator, validate_reference
from .authoring import (
    artifact_dependencies,
    definition_hash,
    parse_document,
    validate_document,
)
from .authoring_store import StoreError, _filesystem_writer_lock
from .civitai_source import CivitaiSource, civitai_locator


class MaterializationCanceled(Exception):
    pass


def _stamp(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ArtifactCache:
    """Source-neutral, digest-addressed blobs published only after verification."""

    def __init__(self, root: Path):
        self.root = root
        self._verified: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def path(self, digest: str) -> Path:
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("Invalid SHA-256 cache identity")
        return self.root / "sha256" / digest[:2] / digest / "blob"

    def verified(self, digest: str, *, force: bool = False) -> Path | None:
        path = self.path(digest)
        if not path.is_file():
            return None
        # Verify on first use after restart and after any file identity/stat
        # change. Catalog reads need not repeatedly hash multi-GB unchanged files.
        with path.open("rb") as handle:
            before = _stamp(os.fstat(handle.fileno()))
            with self._lock:
                if not force and self._verified.get(digest) == before:
                    return path
            checksum = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                checksum.update(chunk)
            if (
                before != _stamp(os.fstat(handle.fileno()))
                or checksum.hexdigest() != digest
            ):
                raise ValueError(
                    "Cached artifact failed SHA-256 verification; materialize it again"
                )
        with self._lock:
            self._verified[digest] = before
        return path

    def acquire(
        self,
        expected: str | None,
        produce,
        cancel,
        progress,
        *,
        size: int | None = None,
    ) -> dict:
        if expected is not None:
            try:
                path = self.verified(expected, force=True)
            except ValueError:
                path = None
            if path is not None:
                return {
                    "sha256": expected,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "cache_hit": True,
                }
        staging = self.root / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=staging, suffix=".partial")
        digest, received = hashlib.sha256(), 0
        try:
            with os.fdopen(descriptor, "wb") as output:

                def write(chunk):
                    nonlocal received
                    cancel()
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    progress(received, size)

                cancel()
                produce(write, cancel)
                cancel()
                if size is not None and received != size:
                    raise ValueError(
                        "Downloaded artifact size does not match source metadata"
                    )
                actual = digest.hexdigest()
                if expected is not None and actual != expected:
                    raise ValueError("Downloaded artifact failed SHA-256 verification")
                output.flush()
                os.fsync(output.fileno())
            destination = self.path(actual)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _filesystem_writer_lock(destination.parent):
                cancel()
                os.replace(temporary, destination)
                with self._lock:
                    self._verified[actual] = _stamp(destination.stat())
            return {
                "sha256": actual,
                "path": str(destination),
                "size": received,
                "cache_hit": False,
            }
        finally:
            Path(temporary).unlink(missing_ok=True)


class ArtifactMaterializer:
    """One bounded authoring download at a time; only verified blobs are durable."""

    def __init__(self, root: Path, source: HuggingFaceSource | None = None):
        self.cache = ArtifactCache(root)
        self.sources = {
            "huggingface": source or HuggingFaceSource(),
            "civitai": CivitaiSource(),
        }
        self._sizes: dict[str, int | None] = {}
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    def resolve(self, reference: dict, *, verify: bool = False) -> Path:
        validate_reference(reference)
        if reference["source"] == "local":
            return Path(reference["path"])
        path = self.cache.verified(reference["sha256"], force=verify)
        if path is None:
            raise ValueError("Remote dependency is not materialized on this host")
        return path

    def plan(self, values: object) -> dict:
        if not isinstance(values, list) or not 1 <= len(values) <= 32:
            raise StoreError(
                422, "Plan requires between 1 and 32 exact recipe documents"
            )
        dependencies, recipes = {}, []
        for value in values:
            validation = validate_document(value, resolve_artifact=self.resolve)
            if not validation["document_valid"] or not validation["recipe_compiles"]:
                raise StoreError(
                    422, "Cannot materialize an invalid recipe policy", validation
                )
            document = parse_document(value)
            identity = {
                "id": document["id"],
                "name": document["name"],
                "definition_hash": definition_hash(document),
            }
            recipes.append(identity)
            slots = {
                item["path"]: item
                for item in validation["artifact_resolution"]["slots"]
            }
            for dependency in artifact_dependencies(document):
                reference = validate_reference(dependency["reference"])
                remote = reference["source"] != "local"
                key = (
                    "sha256:" + reference["sha256"]
                    if remote
                    else "local:" + reference["path"]
                )
                slot = slots[dependency["path"]]
                status = "cached" if remote else "resolved_local"
                message = None
                if slot["status"] != "resolved":
                    status = (
                        "needs_download"
                        if remote and dependency["requirements"]["kind"] == "file"
                        else "unresolved"
                    )
                    message = "; ".join(item["message"] for item in slot["issues"])
                consumer = {
                    **identity,
                    "field": dependency["key"],
                    "path": dependency["path"],
                }
                if key not in dependencies:
                    dependencies[key] = {
                        "id": key,
                        "reference": deepcopy(reference),
                        "consumers": [],
                        "status": status,
                        "message": message,
                        "size_bytes": self._sizes.get(reference.get("sha256"))
                        if remote
                        else None,
                    }
                entry = dependencies[key]
                entry["consumers"].append(consumer)
                # A shared reference may be used by differently constrained slots.
                if status == "unresolved":
                    entry.update(status=status, message=message)
        entries = list(dependencies.values())
        if len(entries) > 256:
            raise StoreError(
                422, "A materialization plan may contain at most 256 unique artifacts"
            )
        missing = [entry for entry in entries if entry["status"] == "needs_download"]
        return {
            "recipes": recipes,
            "dependencies": entries,
            "summary": {
                "recipes": len(recipes),
                "unique_artifacts": len(entries),
                "cached": sum(entry["status"] == "cached" for entry in entries),
                "resolved_local": sum(
                    entry["status"] == "resolved_local" for entry in entries
                ),
                "missing": len(missing),
                "unresolved": sum(entry["status"] == "unresolved" for entry in entries),
                "download_bytes_known": sum(
                    entry["size_bytes"] or 0 for entry in missing
                ),
                "unknown_sizes": sum(entry["size_bytes"] is None for entry in missing),
            },
        }

    def _start(self, kind, action) -> dict:
        with self._lock:
            if self._closed:
                raise StoreError(503, "Materialization is shutting down")
            if self._thread is not None and self._thread.is_alive():
                raise StoreError(
                    409, "Another artifact task is active; wait or cancel it"
                )
            while len(self._tasks) >= 32:
                del self._tasks[next(iter(self._tasks))]
            task_id = str(uuid.uuid4())
            self._cancel = threading.Event()
            self._tasks[task_id] = {
                "id": task_id,
                "kind": kind,
                "status": "running",
                "bytes_downloaded": 0,
                "total_bytes": None,
                "stage": "Resolving source",
                "result": None,
                "error": None,
            }

            def run():
                def cancel():
                    if self._cancel.is_set():
                        raise MaterializationCanceled()

                def update(**changes):
                    with self._lock:
                        self._tasks[task_id].update(changes)

                def progress(received, total):
                    update(bytes_downloaded=received, total_bytes=total)

                try:
                    result = action(cancel, progress, update)
                    cancel()
                    update(status="succeeded", stage="Complete", result=result)
                except MaterializationCanceled:
                    update(status="canceled", stage="Canceled")
                except (ValueError, StoreError) as error:
                    update(status="failed", stage="Failed", error=str(error))
                except Exception as error:  # noqa: BLE001 - an ephemeral task must terminate on any failure
                    update(
                        status="failed",
                        stage="Failed",
                        error=f"Artifact task failed ({type(error).__name__})",
                    )

            self._thread = threading.Thread(
                target=run, name="artifact-materialization", daemon=True
            )
            self._thread.start()
            return deepcopy(self._tasks[task_id])

    def pin(self, value: object, source_name: str = "huggingface") -> dict:
        try:
            locator = (
                hf_locator(value)
                if source_name == "huggingface"
                else civitai_locator(value, pin=True)
            )
        except (TypeError, ValueError) as error:
            raise StoreError(422, str(error)) from None
        source = self.sources[source_name]

        def action(cancel, progress, update):
            file = source.describe(locator)
            cancel()
            digest = file.sha256
            acquired = None
            if digest is None:
                update(stage="Downloading to establish SHA-256")
                acquired = self.cache.acquire(
                    None,
                    lambda write, cancel: source.download(file, write, cancel),
                    cancel,
                    progress,
                    size=file.size,
                )
                digest = acquired["sha256"]
            self._sizes[digest] = file.size
            return {
                "reference": file.reference(digest),
                "size_bytes": file.size,
                "downloaded_for_hash": acquired is not None,
                "cache": acquired,
            }

        return self._start("pin", action)

    def materialize(self, values: object) -> dict:
        plan = self.plan(values)

        def action(cancel, progress, update):
            results = []
            update(plan=deepcopy(plan))
            for entry in plan["dependencies"]:
                cancel()
                if entry["status"] != "needs_download":
                    results.append(deepcopy(entry))
                    continue
                reference = entry["reference"]
                entry.update(status="downloading")
                update(
                    stage=f"Downloading {entry['id']}",
                    plan=deepcopy(plan),
                    bytes_downloaded=0,
                    total_bytes=None,
                )
                try:
                    source = self.sources[reference["source"]]
                    file = source.describe(
                        {
                            key: value
                            for key, value in reference.items()
                            if key not in {"source", "sha256"}
                        }
                    )
                    if file.sha256 is not None and file.sha256 != reference["sha256"]:
                        raise ValueError(
                            "Source SHA-256 differs from the canonical recipe"
                        )
                    self._sizes[reference["sha256"]] = file.size
                    result = self.cache.acquire(
                        reference["sha256"],
                        lambda write, cancel, file=file, source=source: source.download(
                            file, write, cancel
                        ),
                        cancel,
                        progress,
                        size=file.size,
                    )
                    entry.update(
                        status="cached",
                        message=None,
                        size_bytes=result["size"],
                        cache=result,
                    )
                    results.append(deepcopy(entry))
                    update(plan=deepcopy(plan))
                except Exception:
                    entry.update(status="failed")
                    update(plan=deepcopy(plan))
                    raise
            return {
                "dependencies": results,
                "resolved": all(
                    entry["status"] in {"cached", "resolved_local"} for entry in results
                ),
            }

        return self._start("materialize", action)

    def status(self, task_id: str) -> dict:
        with self._lock:
            if task_id not in self._tasks:
                raise StoreError(
                    404,
                    "Artifact task not found (tasks are ephemeral across Engine restarts)",
                )
            return deepcopy(self._tasks[task_id])

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            if task_id not in self._tasks:
                raise StoreError(404, "Artifact task not found")
            if self._tasks[task_id]["status"] == "running":
                self._cancel.set()
                self._tasks[task_id]["stage"] = "Cancellation requested"
            return deepcopy(self._tasks[task_id])

    def close(self):
        with self._lock:
            self._closed = True
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=15)
