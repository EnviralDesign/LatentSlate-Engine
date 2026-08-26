"""Bounded JSON-lines transport for worker progress and liveness records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth import canonical_json

_READ_CHUNK_BYTES = 64 * 1024


class WorkerJsonlFileError(ValueError):
    """A worker JSONL stream violates its file or record contract."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class JsonlCursor:
    """Incremental state for one append-only worker JSONL stream."""

    offset: int = 0
    pending: bytes = b""
    records: int = 0


def append_bounded_jsonl(
    path: Path,
    value: Mapping[str, Any],
    *,
    maximum_bytes: int,
    maximum_record_bytes: int = 4096,
) -> None:
    """Append one canonical object record without exceeding explicit bounds."""

    _validate_bounds(maximum_bytes, maximum_record_bytes)
    if not isinstance(value, Mapping):
        raise TypeError("worker JSONL record must be an object")
    encoded = canonical_json(value) + b"\n"
    if len(encoded) > maximum_record_bytes:
        raise WorkerJsonlFileError(
            "record_bound", "worker JSONL record exceeds its byte bound"
        )
    current_size = path.stat().st_size if path.exists() else 0
    if current_size + len(encoded) > maximum_bytes:
        raise WorkerJsonlFileError(
            "stream_bound", "worker JSONL stream exceeds its byte bound"
        )
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()


def drain_bounded_jsonl(
    path: Path,
    cursor: JsonlCursor,
    *,
    maximum_bytes: int,
    maximum_records: int,
    maximum_record_bytes: int = 4096,
) -> tuple[JsonlCursor, tuple[dict[str, Any], ...]]:
    """Drain complete object records while retaining one partial trailing record."""

    _validate_bounds(maximum_bytes, maximum_record_bytes)
    if isinstance(maximum_records, bool) or not isinstance(maximum_records, int):
        raise TypeError("worker JSONL record bound must be an integer")
    if maximum_records <= 0:
        raise ValueError("worker JSONL record bound must be positive")
    if (
        isinstance(cursor.offset, bool)
        or not isinstance(cursor.offset, int)
        or cursor.offset < 0
        or not isinstance(cursor.pending, bytes)
        or isinstance(cursor.records, bool)
        or not isinstance(cursor.records, int)
        or cursor.records < 0
    ):
        raise TypeError("worker JSONL cursor is invalid")
    if not path.exists():
        return cursor, ()
    size = path.stat().st_size
    if size > maximum_bytes:
        raise WorkerJsonlFileError(
            "stream_bound", "worker JSONL stream exceeds its byte bound"
        )
    if size < cursor.offset:
        raise WorkerJsonlFileError(
            "stream_replaced", "worker JSONL stream was replaced or truncated"
        )
    offset = cursor.offset
    pending = cursor.pending
    records = cursor.records
    values: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        stream.seek(cursor.offset)
        while True:
            chunk = stream.read(
                min(_READ_CHUNK_BYTES, maximum_bytes - offset + 1)
            )
            offset = stream.tell()
            if offset > maximum_bytes:
                raise WorkerJsonlFileError(
                    "stream_bound", "worker JSONL stream exceeds its byte bound"
                )
            if not chunk:
                break

            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            if len(pending) >= maximum_record_bytes:
                raise WorkerJsonlFileError(
                    "partial_record_bound",
                    "worker JSONL partial record exceeds its byte bound",
                )
            for raw in lines:
                if not raw:
                    raise WorkerJsonlFileError(
                        "empty_record", "worker JSONL contains an empty record"
                    )
                if len(raw) + 1 > maximum_record_bytes:
                    raise WorkerJsonlFileError(
                        "record_bound", "worker JSONL record exceeds its byte bound"
                    )
                records += 1
                if records > maximum_records:
                    raise WorkerJsonlFileError(
                        "record_count", "worker JSONL exceeds its record bound"
                    )
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise WorkerJsonlFileError(
                        "invalid_record", "worker JSONL record is invalid"
                    ) from exc
                if not isinstance(value, dict):
                    raise WorkerJsonlFileError(
                        "record_type", "worker JSONL record must be an object"
                    )
                values.append(value)
    return JsonlCursor(offset=offset, pending=pending, records=records), tuple(values)


def _validate_bounds(maximum_bytes: int, maximum_record_bytes: int) -> None:
    for value, label in (
        (maximum_bytes, "stream"),
        (maximum_record_bytes, "record"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"worker JSONL {label} byte bound must be an integer")
        if value <= 0:
            raise ValueError(f"worker JSONL {label} byte bound must be positive")
