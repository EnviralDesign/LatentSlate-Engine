"""Small, dependency-free host-process memory observation for acceptance runners."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def read_current_process_memory() -> dict[str, int]:
    """Read exact Windows process counters or raise when unavailable."""

    if os.name != "nt":
        raise OSError("unsupported_platform")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "pid": os.getpid(),
        "private_bytes": int(counters.PrivateUsage),
        "working_set_bytes": int(counters.WorkingSetSize),
    }


def read_system_physical_memory() -> dict[str, int]:
    """Read exact Windows physical-memory totals or raise when unavailable."""

    if os.name != "nt":
        raise OSError("unsupported_platform")
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    total = int(status.ullTotalPhys)
    available = int(status.ullAvailPhys)
    if total <= 0 or available < 0 or available > total:
        raise OSError("invalid_physical_memory_counters")
    return {
        "total_physical_bytes": total,
        "free_physical_bytes": available,
        "used_physical_bytes": total - available,
    }


def current_process_memory() -> dict[str, int | None]:
    """Return a stable parent PID and Windows private/working bytes when available."""

    values: dict[str, int | None] = {
        "pid": os.getpid(),
        "private_bytes": None,
        "working_set_bytes": None,
    }
    try:
        observed = read_current_process_memory()
        values.update(observed)
    except OSError:
        # Runtime observation must never make an otherwise healthy Engine API
        # unavailable. Hardware acceptance runners fail closed on nulls.
        pass
    return values
