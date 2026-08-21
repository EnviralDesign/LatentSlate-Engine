"""Small Windows process-tree boundary used by disposable heavyweight workers."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from subprocess import Popen

_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _LargeInteger(ctypes.Structure):
    _fields_ = [("QuadPart", ctypes.c_longlong)]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", _LargeInteger),
        ("PerJobUserTimeLimit", _LargeInteger),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", _LargeInteger),
        ("TotalKernelTime", _LargeInteger),
        ("ThisPeriodTotalUserTime", _LargeInteger),
        ("ThisPeriodTotalKernelTime", _LargeInteger),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


def _kernel32():
    if os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


class DisposableProcessTree:
    """Own one worker and guarantee its descendants die with the job object."""

    def __init__(self, process: Popen[object]) -> None:
        self.process = process
        self._handle: wintypes.HANDLE | None = None
        if os.name == "nt":
            self._handle = _create_kill_on_close_job()
            try:
                _assign_process(self._handle, process.pid)
            except BaseException:
                _close_handle(self._handle)
                self._handle = None
                raise

    def active_processes(self) -> int:
        """Return the Job Object's live process count (zero off Windows)."""

        if self._handle is None:
            return 0 if self.process.poll() is not None else 1
        return _active_processes(self._handle)

    def wait_for_empty(self, timeout: float = 15.0) -> None:
        """Prove all worker-tree processes are gone before parent success/cancel."""

        deadline = time.monotonic() + timeout
        while self.active_processes() != 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("worker Job Object still has active processes")
            time.sleep(0.05)

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle is not None:
            _terminate_job(self._handle, exit_code)
        elif self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        if self._handle is not None:
            _close_handle(self._handle)
            self._handle = None


def _create_kill_on_close_job() -> wintypes.HANDLE:
    kernel = _kernel32()
    assert kernel is not None
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(
        handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        _close_handle(handle)
        raise ctypes.WinError(error)
    return handle


def _assign_process(job_handle: wintypes.HANDLE, pid: int) -> None:
    kernel = _kernel32()
    assert kernel is not None
    process_handle = kernel.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
    if not process_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.AssignProcessToJobObject(job_handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _close_handle(process_handle)


def _active_processes(job_handle: wintypes.HANDLE) -> int:
    kernel = _kernel32()
    assert kernel is not None
    info = _BasicAccountingInformation()
    if not kernel.QueryInformationJobObject(
        job_handle,
        _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(info.ActiveProcesses)


def _terminate_job(job_handle: wintypes.HANDLE, exit_code: int) -> None:
    kernel = _kernel32()
    assert kernel is not None
    if not kernel.TerminateJobObject(job_handle, wintypes.UINT(exit_code)):
        raise ctypes.WinError(ctypes.get_last_error())


def _close_handle(handle: wintypes.HANDLE) -> None:
    if os.name != "nt":
        return
    kernel = _kernel32()
    assert kernel is not None
    if not kernel.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())
