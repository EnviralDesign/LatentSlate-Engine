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


def current_process_memory() -> dict[str, int | None]:
    """Return a stable parent PID and Windows private/working bytes when available."""

    values: dict[str, int | None] = {
        "pid": os.getpid(),
        "private_bytes": None,
        "working_set_bytes": None,
    }
    if os.name != "nt":
        return values
    try:
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
        values["private_bytes"] = int(counters.PrivateUsage)
        values["working_set_bytes"] = int(counters.WorkingSetSize)
    except OSError:
        # Runtime observation must never make an otherwise healthy Engine API
        # unavailable. The opt-in Wan acceptance runner fails closed on nulls.
        pass
    return values
