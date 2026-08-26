"""Model-neutral, best-effort phase memory observations for runtime metadata."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .process_memory import read_current_process_memory, read_system_physical_memory

MEMORY_TELEMETRY_SCHEMA_VERSION = 1
_SAFE_ERROR = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class PhaseMemoryTelemetry:
    """Collect an exact ordered set of non-fatal host and CUDA observations."""

    def __init__(
        self,
        phases: Sequence[str],
        device: torch.device | str,
        *,
        process_reader: Callable[[], Mapping[str, int]] = read_current_process_memory,
        system_reader: Callable[[], Mapping[str, int]] = read_system_physical_memory,
        timestamp_ns: Callable[[], int] = time.time_ns,
        elapsed_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        expected = tuple(phases)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("memory telemetry phases must be nonempty and unique")
        if any(not isinstance(phase, str) or not phase for phase in expected):
            raise ValueError("memory telemetry phase names must be nonempty strings")
        self._phases = expected
        self._device = torch.device(device)
        self._process_reader = process_reader
        self._system_reader = system_reader
        self._timestamp_ns = timestamp_ns
        self._elapsed_ns = elapsed_ns
        self._started_ns = _strict_int(elapsed_ns(), "elapsed clock")
        self._samples: list[dict[str, Any]] = []

    def capture(self, phase: str) -> None:
        """Capture one required phase; source failures become explicit evidence."""

        sequence = len(self._samples)
        if sequence >= len(self._phases) or phase != self._phases[sequence]:
            raise RuntimeError("memory telemetry phase order changed")
        timestamp = _strict_int(self._timestamp_ns(), "timestamp clock")
        elapsed = max(
            0,
            _strict_int(self._elapsed_ns(), "elapsed clock") - self._started_ns,
        )
        self._samples.append(
            {
                "sequence": sequence,
                "phase": phase,
                "timestamp_unix_ns": timestamp,
                "elapsed_ns": elapsed,
                "process": self._capture_process(),
                "system": self._capture_system(),
                "cuda": self._capture_cuda(),
            }
        )

    def metadata(self) -> dict[str, Any]:
        """Return a detached exact-schema snapshot after all phases were captured."""

        if len(self._samples) != len(self._phases):
            raise RuntimeError("memory telemetry is incomplete")
        return {
            "schema_version": MEMORY_TELEMETRY_SCHEMA_VERSION,
            "timestamp_clock": "time.time_ns",
            "elapsed_clock": "time.perf_counter_ns",
            "samples": [
                {
                    **sample,
                    "process": dict(sample["process"]),
                    "system": dict(sample["system"]),
                    "cuda": dict(sample["cuda"]),
                }
                for sample in self._samples
            ],
        }

    def _capture_process(self) -> dict[str, Any]:
        try:
            values = self._process_reader()
            pid = _strict_int(values["pid"], "process pid")
            private = _strict_int(values["private_bytes"], "process private bytes")
            working_set = _strict_int(
                values["working_set_bytes"], "process working-set bytes"
            )
            if pid <= 0 or private <= 0 or working_set <= 0:
                raise ValueError("invalid_process_memory")
            return {
                "status": "ok",
                "error": None,
                "pid": pid,
                "private_bytes": private,
                "working_set_bytes": working_set,
            }
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break inference
            return {
                "status": "error",
                "error": _error_category(exc),
                "pid": None,
                "private_bytes": None,
                "working_set_bytes": None,
            }

    def _capture_system(self) -> dict[str, Any]:
        try:
            values = self._system_reader()
            total = _strict_int(
                values["total_physical_bytes"], "system total physical bytes"
            )
            free = _strict_int(
                values["free_physical_bytes"], "system free physical bytes"
            )
            used = _strict_int(
                values["used_physical_bytes"], "system used physical bytes"
            )
            if total <= 0 or not 0 <= free <= total or used != total - free:
                raise ValueError("invalid_system_memory")
            return {
                "status": "ok",
                "error": None,
                "total_physical_bytes": total,
                "free_physical_bytes": free,
                "used_physical_bytes": used,
            }
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break inference
            return {
                "status": "error",
                "error": _error_category(exc),
                "total_physical_bytes": None,
                "free_physical_bytes": None,
                "used_physical_bytes": None,
            }

    def _capture_cuda(self) -> dict[str, Any]:
        device = str(self._device)
        try:
            canonical = self._device
            if canonical.type != "cuda":
                raise ValueError("unsupported_device")
            if canonical.index is None:
                canonical = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(canonical):
                free, total = torch.cuda.mem_get_info(canonical)
                allocated = torch.cuda.memory_allocated(canonical)
                reserved = torch.cuda.memory_reserved(canonical)
            device = str(canonical)
            allocated = _strict_int(allocated, "CUDA allocated bytes")
            reserved = _strict_int(reserved, "CUDA reserved bytes")
            free = _strict_int(free, "CUDA free bytes")
            total = _strict_int(total, "CUDA total bytes")
            if (
                total <= 0
                or not 0 <= allocated <= reserved <= total
                or not 0 <= free <= total
            ):
                raise ValueError("invalid_cuda_memory")
            return {
                "status": "ok",
                "error": None,
                "device": device,
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "free_bytes": free,
                "total_bytes": total,
            }
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break inference
            return {
                "status": "error",
                "error": _error_category(exc),
                "device": device,
                "allocated_bytes": None,
                "reserved_bytes": None,
                "free_bytes": None,
                "total_bytes": None,
            }


def _error_category(exc: Exception) -> str:
    name = type(exc).__name__
    return name if _SAFE_ERROR.fullmatch(name) else "TelemetryError"


def _strict_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value
