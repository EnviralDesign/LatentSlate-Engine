"""Best-effort in-place CUDA host registration for authoritative CPU tensors."""

from __future__ import annotations

import ctypes
import os
from typing import Any

import torch

_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
_ALLOWED_TYPES = frozenset({"Tensor", "Parameter", "QuantizedTensor"})


def _windows_memory_status() -> tuple[int, int] | None:
    """Return ``(total, available)`` physical bytes from GlobalMemoryStatusEx."""

    if os.name != "nt":
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return None


def system_memory_bytes() -> int | None:
    """Return installed physical memory without importing optional packages."""

    windows = _windows_memory_status()
    if windows is not None:
        return windows[0]
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def available_physical_memory_bytes() -> int | None:
    """Return currently available physical RAM through an OS read-only query."""

    windows = _windows_memory_status()
    if windows is not None:
        return windows[1]
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def default_host_registration_budget_bytes() -> int:
    """Use Comfy's Windows 40%-of-RAM budget and a portable safe fallback."""

    total = system_memory_bytes()
    return 4 * 1024**3 if total is None else max(0, int(total * 0.40))


class BestEffortHostRegistrationLedger:
    """Track only registrations created by one synchronized residency stage."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        self.seen: set[tuple[int, int]] = set()
        self.owned: dict[tuple[int, int], torch.Tensor] = {}
        self.counts = {
            "candidates": 0,
            "candidate_bytes": 0,
            "deduplicated_aliases": 0,
            "already_registered": 0,
            "already_registered_bytes": 0,
            "attempts": 0,
            "attempt_bytes": 0,
            "successes": 0,
            "registered_bytes": 0,
            "failures": 0,
            "failure_bytes": 0,
            "ineligible": 0,
            "ineligible_bytes": 0,
            "unregistered": 0,
            "unregistered_bytes": 0,
            "unregister_failures": 0,
            "unregister_failure_bytes": 0,
        }
        self.categories = {
            "unsupported_type": 0,
            "non_cpu": 0,
            "noncontiguous": 0,
            "zero_pointer": 0,
            "budget_exceeded": 0,
            "eligibility_error": 0,
            "register_error": 0,
            "unregister_error": 0,
        }

    def consider(self, value: torch.Tensor) -> bool:
        """Attempt registration and return whether this source is pinned."""

        size = int(getattr(value, "nbytes", 0))
        self.counts["candidates"] += 1
        self.counts["candidate_bytes"] += max(0, size)
        try:
            ptr = int(value.data_ptr())
        except (RuntimeError, TypeError, ValueError):
            self._ineligible("eligibility_error", max(0, size))
            return False
        key = (ptr, max(0, size))
        if key in self.seen:
            self.counts["deduplicated_aliases"] += 1
            return key in self.owned or bool(value.is_pinned())
        self.seen.add(key)
        if type(value).__name__ not in _ALLOWED_TYPES:
            self._ineligible("unsupported_type", max(0, size))
            return False
        if value.device.type != "cpu":
            self._ineligible("non_cpu", max(0, size))
            return False
        if not value.is_contiguous():
            self._ineligible("noncontiguous", max(0, size))
            return False
        if ptr == 0 or size <= 0:
            self._ineligible("zero_pointer", max(0, size))
            return False
        if value.is_pinned():
            self.counts["already_registered"] += 1
            self.counts["already_registered_bytes"] += size
            return True
        if self.counts["registered_bytes"] + size > self.budget_bytes:
            self._ineligible("budget_exceeded", size)
            return False
        self.counts["attempts"] += 1
        self.counts["attempt_bytes"] += size
        try:
            result = torch.cuda.cudart().cudaHostRegister(ptr, size, 1)
        except (RuntimeError, TypeError, ValueError):
            result = -1
        if result == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            self._discard_cuda_error()
            self.counts["attempts"] -= 1
            self.counts["attempt_bytes"] -= size
            self.counts["already_registered"] += 1
            self.counts["already_registered_bytes"] += size
            return True
        if result != 0:
            self._discard_cuda_error()
            self.counts["failures"] += 1
            self.counts["failure_bytes"] += size
            self.categories["register_error"] += 1
            return False
        self.owned[key] = value
        self.counts["successes"] += 1
        self.counts["registered_bytes"] += size
        return True

    def unregister_owned(self) -> list[BaseException]:
        errors: list[BaseException] = []
        for (ptr, size), _value in tuple(self.owned.items()):
            try:
                result = torch.cuda.cudart().cudaHostUnregister(ptr)
            except (RuntimeError, TypeError, ValueError) as exc:
                result = -1
                errors.append(exc)
            if result != 0:
                self.counts["unregister_failures"] += 1
                self.counts["unregister_failure_bytes"] += size
                self.categories["unregister_error"] += 1
                if not errors or not isinstance(errors[-1], RuntimeError):
                    errors.append(RuntimeError("CUDA host unregistration failed"))
                continue
            self.owned.pop((ptr, size), None)
            self.counts["unregistered"] += 1
            self.counts["unregistered_bytes"] += size
        return errors

    def provenance(self) -> dict[str, Any]:
        return {
            "policy": "comfy_best_effort_in_place_cuda_host_register",
            "lifecycle": "residency_stage_through_synchronized_close",
            "budget_bytes": self.budget_bytes,
            **self.counts,
            "owned_active": len(self.owned),
            "owned_active_bytes": sum(size for _, size in self.owned),
            "categories": dict(self.categories),
        }

    def _ineligible(self, category: str, size: int) -> None:
        self.counts["ineligible"] += 1
        self.counts["ineligible_bytes"] += size
        self.categories[category] += 1

    @staticmethod
    def _discard_cuda_error() -> None:
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
