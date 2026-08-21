from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from latentslate_engine.runtime.framework.worker import (
    JsonlCursor,
    PersistentChildContext,
    PersistentChildPaths,
    WorkerJsonlFileError,
    append_bounded_jsonl,
    drain_bounded_jsonl,
)


def test_bounded_jsonl_append_is_canonical_and_drains_in_order(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    append_bounded_jsonl(path, {"message": "one", "progress": 0.25}, maximum_bytes=256)
    append_bounded_jsonl(path, {"progress": 0.5, "message": "two"}, maximum_bytes=256)

    assert path.read_bytes() == (
        b'{"message":"one","progress":0.25}\n'
        b'{"message":"two","progress":0.5}\n'
    )
    cursor, values = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=256,
        maximum_records=2,
    )
    assert values == (
        {"message": "one", "progress": 0.25},
        {"message": "two", "progress": 0.5},
    )
    assert cursor == JsonlCursor(offset=path.stat().st_size, records=2)


def test_bounded_jsonl_retains_partial_record_until_new_bytes_arrive(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    path.write_bytes(b'{"progress":0.25}\n{"progress":')

    cursor, values = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=256,
        maximum_records=4,
    )
    assert values == ({"progress": 0.25},)
    assert cursor.pending == b'{"progress":'
    with path.open("ab") as stream:
        stream.write(b"0.5}\n")

    cursor, values = drain_bounded_jsonl(
        path,
        cursor,
        maximum_bytes=256,
        maximum_records=4,
    )
    assert values == ({"progress": 0.5},)
    assert cursor.pending == b""
    assert cursor.records == 2


@pytest.mark.parametrize(
    ("content", "maximum_bytes", "maximum_records", "maximum_record_bytes", "message"),
    [
        (b'{"x":1}\n', 4, 4, 64, "stream exceeds"),
        (b'{"x":1}\n{"x":2}\n', 64, 1, 64, "record bound"),
        (b'{"long":"value"}\n', 64, 4, 4, "record exceeds"),
        (b"\n", 64, 4, 64, "empty record"),
        (b"not-json\n", 64, 4, 64, "record is invalid"),
        (b"[]\n", 64, 4, 64, "must be an object"),
        (b'{"unfinished":true', 64, 4, 4, "partial record exceeds"),
    ],
)
def test_bounded_jsonl_drain_fails_closed(
    tmp_path: Path,
    content: bytes,
    maximum_bytes: int,
    maximum_records: int,
    maximum_record_bytes: int,
    message: str,
):
    path = tmp_path / "progress.jsonl"
    path.write_bytes(content)
    with pytest.raises(WorkerJsonlFileError, match=message):
        drain_bounded_jsonl(
            path,
            JsonlCursor(),
            maximum_bytes=maximum_bytes,
            maximum_records=maximum_records,
            maximum_record_bytes=maximum_record_bytes,
        )


def test_bounded_jsonl_rejects_truncated_stream_and_append_overflow(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    append_bounded_jsonl(path, {"x": 1}, maximum_bytes=16)
    cursor, _ = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=16,
        maximum_records=4,
    )
    path.write_bytes(b"")
    with pytest.raises(WorkerJsonlFileError, match="replaced or truncated"):
        drain_bounded_jsonl(path, cursor, maximum_bytes=16, maximum_records=4)

    with pytest.raises(WorkerJsonlFileError, match="stream exceeds"):
        append_bounded_jsonl(path, {"message": "too large"}, maximum_bytes=8)


def test_record_bound_counts_the_complete_physical_line(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    # Canonical {"x":1} is seven bytes; its newline is the eighth byte.
    append_bounded_jsonl(
        path,
        {"x": 1},
        maximum_bytes=32,
        maximum_record_bytes=8,
    )
    cursor, values = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=32,
        maximum_records=1,
        maximum_record_bytes=8,
    )
    assert values == ({"x": 1},)
    assert cursor.pending == b""

    rejected = tmp_path / "rejected.jsonl"
    with pytest.raises(WorkerJsonlFileError, match="record exceeds"):
        append_bounded_jsonl(
            rejected,
            {"x": 1},
            maximum_bytes=32,
            maximum_record_bytes=7,
        )
    rejected.write_bytes(b'{"x":1}\n')
    with pytest.raises(WorkerJsonlFileError, match="record exceeds"):
        drain_bounded_jsonl(
            rejected,
            JsonlCursor(),
            maximum_bytes=32,
            maximum_records=1,
            maximum_record_bytes=7,
        )


@pytest.mark.parametrize(
    ("module_name", "bound_name", "expected_type", "expected_message"),
    [
        (
            "latentslate_engine.runtime.ltx23_kitchen_worker",
            "_MAX_PROGRESS_BYTES",
            ValueError,
            "LTX 2.3 Kitchen worker progress exceeds its bound",
        ),
        (
            "latentslate_engine.runtime.wan22_native_worker",
            "_MAX_PROGRESS_BYTES",
            ValueError,
            "native Wan worker progress exceeds its aggregate bound",
        ),
        (
            "latentslate_engine.runtime.ltx23_worker",
            "_MAX_JSON_BYTES",
            ValueError,
            "LTX worker progress exceeds its bound",
        ),
    ],
)
def test_worker_progress_adapters_preserve_family_overflow_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    bound_name: str,
    expected_type: type[Exception],
    expected_message: str,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, bound_name, 8)

    with pytest.raises(expected_type, match=expected_message) as raised:
        module._append_progress(
            tmp_path / "progress.jsonl",
            {"progress": 0.5, "message": "bounded"},
        )
    assert type(raised.value) is expected_type


def test_z_progress_adapter_preserves_closed_runtime_error(tmp_path: Path):
    module = importlib.import_module("latentslate_engine.runtime.z_image_turbo_worker")
    path = tmp_path / "progress.jsonl"
    context = PersistentChildContext(
        paths=PersistentChildPaths(
            request=tmp_path / "request.json",
            result=tmp_path / "result.json",
            progress=path,
            heartbeat=tmp_path / "heartbeat.jsonl",
            start_gate=tmp_path / "gate",
            command=tmp_path / "command.json",
            cancel=tmp_path / "cancel",
        ),
        maximum_bytes=8,
        heartbeat_seconds=1,
        protocol_error=module._ZImageHandler("").protocol_error,
    )
    with pytest.raises(RuntimeError, match="Z-Image worker progress exceeds its bound") as raised:
        context.publish_progress(0.5, "bounded")
    assert type(raised.value) is RuntimeError


def test_native_wan_progress_adapter_distinguishes_record_overflow(tmp_path: Path):
    module = importlib.import_module("latentslate_engine.runtime.wan22_native_worker")
    with pytest.raises(
        ValueError,
        match="native Wan worker progress record exceeds its bound",
    ) as raised:
        module._append_progress(
            tmp_path / "progress.jsonl",
            {"completed": 1, "total": 1, "stage": "x" * 4096},
        )
    assert type(raised.value) is ValueError
