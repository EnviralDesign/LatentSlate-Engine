from __future__ import annotations

from contextlib import nullcontext

import pytest

from latentslate_engine.runtime import memory_telemetry as telemetry_module
from latentslate_engine.runtime.memory_telemetry import PhaseMemoryTelemetry


def _patch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry_module.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        telemetry_module.torch.cuda, "mem_get_info", lambda _device: (6_000, 10_000)
    )
    monkeypatch.setattr(
        telemetry_module.torch.cuda, "memory_allocated", lambda _device: 2_000
    )
    monkeypatch.setattr(
        telemetry_module.torch.cuda, "memory_reserved", lambda _device: 3_000
    )


def test_phase_memory_telemetry_collects_exact_ordered_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cuda(monkeypatch)
    timestamps = iter((100, 101))
    elapsed = iter((1_000, 1_010, 1_020))
    collector = PhaseMemoryTelemetry(
        ("first", "second"),
        "cuda:0",
        process_reader=lambda: {
            "pid": 7,
            "private_bytes": 11,
            "working_set_bytes": 9,
        },
        system_reader=lambda: {
            "total_physical_bytes": 100,
            "free_physical_bytes": 40,
            "used_physical_bytes": 60,
        },
        timestamp_ns=lambda: next(timestamps),
        elapsed_ns=lambda: next(elapsed),
    )

    collector.capture("first")
    collector.capture("second")
    metadata = collector.metadata()

    assert metadata["schema_version"] == 1
    assert [sample["phase"] for sample in metadata["samples"]] == ["first", "second"]
    assert [sample["elapsed_ns"] for sample in metadata["samples"]] == [10, 20]
    assert metadata["samples"][0]["process"] == {
        "status": "ok",
        "error": None,
        "pid": 7,
        "private_bytes": 11,
        "working_set_bytes": 9,
    }
    assert metadata["samples"][0]["system"]["used_physical_bytes"] == 60
    assert metadata["samples"][0]["cuda"] == {
        "status": "ok",
        "error": None,
        "device": "cuda:0",
        "allocated_bytes": 2_000,
        "reserved_bytes": 3_000,
        "free_bytes": 6_000,
        "total_bytes": 10_000,
    }


def test_phase_memory_telemetry_isolates_every_source_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cuda(monkeypatch)
    monkeypatch.setattr(
        telemetry_module.torch.cuda,
        "mem_get_info",
        lambda _device: (_ for _ in ()).throw(RuntimeError("private cuda detail")),
    )

    collector = PhaseMemoryTelemetry(
        ("only",),
        "cuda:0",
        process_reader=lambda: (_ for _ in ()).throw(OSError("private process detail")),
        system_reader=lambda: (_ for _ in ()).throw(ValueError("private system detail")),
    )
    collector.capture("only")
    sample = collector.metadata()["samples"][0]

    assert sample["process"] == {
        "status": "error",
        "error": "OSError",
        "pid": None,
        "private_bytes": None,
        "working_set_bytes": None,
    }
    assert sample["system"] == {
        "status": "error",
        "error": "ValueError",
        "total_physical_bytes": None,
        "free_physical_bytes": None,
        "used_physical_bytes": None,
    }
    assert sample["cuda"] == {
        "status": "error",
        "error": "RuntimeError",
        "device": "cuda:0",
        "allocated_bytes": None,
        "reserved_bytes": None,
        "free_bytes": None,
        "total_bytes": None,
    }
    assert "private process detail" not in repr(sample)
    assert "private system detail" not in repr(sample)
    assert "private cuda detail" not in repr(sample)


def test_phase_memory_telemetry_rejects_wrong_order_and_incomplete_metadata() -> None:
    collector = PhaseMemoryTelemetry(("first", "second"), "cpu")

    with pytest.raises(RuntimeError, match="phase order"):
        collector.capture("second")
    with pytest.raises(RuntimeError, match="incomplete"):
        collector.metadata()


def test_bool_numeric_sources_become_explicit_errors_not_integer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cuda(monkeypatch)
    monkeypatch.setattr(
        telemetry_module.torch.cuda, "memory_allocated", lambda _device: True
    )
    collector = PhaseMemoryTelemetry(
        ("only",),
        "cuda:0",
        process_reader=lambda: {
            "pid": True,
            "private_bytes": 11,
            "working_set_bytes": 9,
        },
        system_reader=lambda: {
            "total_physical_bytes": 100,
            "free_physical_bytes": True,
            "used_physical_bytes": 60,
        },
    )

    collector.capture("only")
    sample = collector.metadata()["samples"][0]

    assert sample["process"]["status"] == "error"
    assert sample["process"]["error"] == "TypeError"
    assert sample["process"]["pid"] is None
    assert sample["system"]["status"] == "error"
    assert sample["system"]["error"] == "TypeError"
    assert sample["system"]["free_physical_bytes"] is None
    assert sample["cuda"]["status"] == "error"
    assert sample["cuda"]["error"] == "TypeError"
    assert sample["cuda"]["allocated_bytes"] is None


class _CoercibleInteger:
    def __int__(self) -> int:
        return 1


@pytest.mark.parametrize("invalid", [True, 1.0, "1", _CoercibleInteger()])
def test_non_integer_clocks_are_rejected_before_conversion(invalid: object) -> None:
    with pytest.raises(TypeError, match="elapsed clock must be an integer"):
        PhaseMemoryTelemetry(("only",), "cpu", elapsed_ns=lambda: invalid)  # type: ignore[arg-type]

    timestamp = PhaseMemoryTelemetry(
        ("only",), "cpu", timestamp_ns=lambda: invalid  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="timestamp clock must be an integer"):
        timestamp.capture("only")

    elapsed_values = iter((1, invalid))
    elapsed = PhaseMemoryTelemetry(
        ("only",), "cpu", elapsed_ns=lambda: next(elapsed_values)  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="elapsed clock must be an integer"):
        elapsed.capture("only")


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("process", "pid"),
        ("process", "private_bytes"),
        ("process", "working_set_bytes"),
        ("system", "total_physical_bytes"),
        ("system", "free_physical_bytes"),
        ("system", "used_physical_bytes"),
        ("cuda", "allocated_bytes"),
        ("cuda", "reserved_bytes"),
        ("cuda", "free_bytes"),
        ("cuda", "total_bytes"),
    ],
)
@pytest.mark.parametrize("invalid", [True, 1.0, "1", _CoercibleInteger()])
def test_coercible_numeric_sources_become_isolated_errors(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    invalid: object,
) -> None:
    _patch_cuda(monkeypatch)
    process: dict[str, object] = {
        "pid": 7,
        "private_bytes": 11,
        "working_set_bytes": 9,
    }
    system: dict[str, object] = {
        "total_physical_bytes": 100,
        "free_physical_bytes": 40,
        "used_physical_bytes": 60,
    }
    if section == "process":
        process[field] = invalid
    elif section == "system":
        system[field] = invalid
    elif field == "allocated_bytes":
        monkeypatch.setattr(
            telemetry_module.torch.cuda, "memory_allocated", lambda _device: invalid
        )
    elif field == "reserved_bytes":
        monkeypatch.setattr(
            telemetry_module.torch.cuda, "memory_reserved", lambda _device: invalid
        )
    elif field == "free_bytes":
        monkeypatch.setattr(
            telemetry_module.torch.cuda,
            "mem_get_info",
            lambda _device: (invalid, 10_000),
        )
    else:
        monkeypatch.setattr(
            telemetry_module.torch.cuda,
            "mem_get_info",
            lambda _device: (6_000, invalid),
        )
    collector = PhaseMemoryTelemetry(
        ("only",),
        "cuda:0",
        process_reader=lambda: process,  # type: ignore[arg-type]
        system_reader=lambda: system,  # type: ignore[arg-type]
    )

    collector.capture("only")
    observation = collector.metadata()["samples"][0][section]

    assert observation["status"] == "error"
    assert observation["error"] == "TypeError"
    assert all(
        value is None
        for key, value in observation.items()
        if key not in {"status", "error", "device"}
    )
