from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.runtime.framework.worker import (
    DisposableChildContext,
    DisposableChildPaths,
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


def test_bounded_jsonl_reads_at_most_64k_and_drains_newline_tail_to_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large-progress.jsonl"
    message = "x" * 70_000
    path.write_bytes((f'{{"message":"{message}"}}\n').encode())
    maximum_bytes = path.stat().st_size + 1
    read_sizes: list[int] = []
    original_open = Path.open

    class _TrackingReader:
        def __init__(self, stream) -> None:
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def seek(self, *args):
            return self._stream.seek(*args)

        def tell(self):
            return self._stream.tell()

        def read(self, size: int = -1):
            read_sizes.append(size)
            return self._stream.read(size)

    def tracked_open(self: Path, mode: str = "r", *args, **kwargs):
        stream = original_open(self, mode, *args, **kwargs)
        if self == path and mode == "rb":
            return _TrackingReader(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    cursor, values = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=maximum_bytes,
        maximum_records=1,
        maximum_record_bytes=80_000,
    )

    assert values == ({"message": message},)
    assert cursor.pending == b""
    assert cursor.offset == path.stat().st_size
    assert read_sizes[0] == 64 * 1024
    assert all(0 < size <= 64 * 1024 for size in read_sizes)


def test_bounded_jsonl_drains_mid_record_tail_to_actual_eof(tmp_path: Path) -> None:
    path = tmp_path / "truncated-progress.jsonl"
    content = b'{"message":"' + b"x" * 70_000
    path.write_bytes(content)

    cursor, values = drain_bounded_jsonl(
        path,
        JsonlCursor(),
        maximum_bytes=len(content) + 1,
        maximum_records=1,
        maximum_record_bytes=80_000,
    )

    assert values == ()
    assert cursor.offset == len(content)
    assert cursor.pending == content


def test_bounded_jsonl_rejects_growth_past_total_bound_during_chunked_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "growing-progress.jsonl"
    maximum_bytes = 70_000
    path.write_bytes(b"x" * (maximum_bytes + 1))
    original_stat = Path.stat

    def bounded_stat(self: Path, *args, **kwargs):
        observed = original_stat(self, *args, **kwargs)
        if self != path:
            return observed
        return SimpleNamespace(st_size=maximum_bytes)

    monkeypatch.setattr(Path, "stat", bounded_stat)

    with pytest.raises(WorkerJsonlFileError, match="stream exceeds") as raised:
        drain_bounded_jsonl(
            path,
            JsonlCursor(),
            maximum_bytes=maximum_bytes,
            maximum_records=1,
            maximum_record_bytes=maximum_bytes + 2,
        )
    assert raised.value.reason == "stream_bound"


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
    ("module_name", "handler_name", "handler_args", "expected_message"),
    [
        (
            "latentslate_engine.runtime.ltx23_kitchen_worker",
            "_LTX23KitchenHandler",
            (b"x" * 32,),
            "LTX 2.3 Kitchen worker protocol violation: progress_bound",
        ),
        (
            "latentslate_engine.runtime.wan22_native_worker",
            "_WanWorkerHandler",
            (b"x" * 32,),
            "native Wan worker protocol violation: progress_bound",
        ),
    ],
)
def test_persistent_worker_progress_uses_shared_transport_and_family_error(
    tmp_path: Path,
    module_name: str,
    handler_name: str,
    handler_args: tuple[object, ...],
    expected_message: str,
):
    module = importlib.import_module(module_name)
    handler = getattr(module, handler_name)(*handler_args)
    context = PersistentChildContext(
        paths=PersistentChildPaths(
            request=tmp_path / "request.json",
            result=tmp_path / "result.json",
            progress=tmp_path / "progress.jsonl",
            heartbeat=tmp_path / "heartbeat.jsonl",
            start_gate=tmp_path / "gate",
            command=tmp_path / "command.json",
            cancel=tmp_path / "cancel",
        ),
        maximum_bytes=8,
        heartbeat_seconds=1,
        protocol_error=handler.protocol_error,
    )

    with pytest.raises(ValueError, match=expected_message) as raised:
        context.publish_progress_record({"progress": 0.5, "message": "bounded"})
    assert type(raised.value) is ValueError


def test_disposable_ltx_progress_uses_shared_transport_and_family_error(tmp_path: Path):
    module = importlib.import_module("latentslate_engine.runtime.ltx23_worker")
    handler = module._LTX23Handler()
    context = DisposableChildContext(
        paths=DisposableChildPaths(
            request=tmp_path / "request.json",
            result=tmp_path / "result.json",
            progress=tmp_path / "progress.jsonl",
            start_gate=tmp_path / "gate",
        ),
        maximum_progress_bytes=8,
        stage_for_progress=handler.stage_for_progress,
        protocol_error=handler.protocol_error,
    )

    with pytest.raises(ValueError, match="LTX worker progress exceeds its bound") as raised:
        context.publish_progress(0.5, "bounded")
    assert type(raised.value) is ValueError


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
