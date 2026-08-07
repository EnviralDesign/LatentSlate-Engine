from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import get_token

from . import __version__
from .bundles import descriptors as bundle_descriptors
from .config import Settings


_GIB = 1024**3
_PACKAGE_PROBES: dict[str, tuple[str, str]] = {
    "accelerate": ("accelerate", "accelerate"),
    "av": ("av", "av"),
    "diffusers": ("diffusers", "diffusers"),
    "ftfy": ("ftfy", "ftfy"),
    "modelopt": ("modelopt", "nvidia-modelopt"),
    "numpy": ("numpy", "numpy"),
    "pillow": ("PIL", "pillow"),
    "sentencepiece": ("sentencepiece", "sentencepiece"),
    "torch": ("torch", "torch"),
    "torchao": ("torchao", "torchao"),
    "transformers": ("transformers", "transformers"),
}


def collect_report(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    packages = _package_report()
    cuda = _cuda_report()
    memory_bytes = _system_memory_bytes()
    disk_free_bytes = _disk_free_bytes(settings.home)
    token_configured = _hf_token_configured()
    bundles = [
        descriptor.model_dump(mode="json") for descriptor in bundle_descriptors()
    ]

    video_base = {
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "av",
        "numpy",
        "pillow",
    }
    h3_required = {
        *video_base,
        *(["torchao"] if settings.h3_profile == "consumer_int8" else []),
    }
    ltx23_required = {*video_base, "sentencepiece"}
    wan22_required = {*video_base, "ftfy", "sentencepiece"}
    klein4b_required = {
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "pillow",
    }
    klein9b_required = {
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "pillow",
        *(["modelopt"] if settings.klein_profile == "consumer_nvfp4" else []),
        *(["torchao"] if settings.klein_profile == "consumer_int8" else []),
    }

    families = {
        "h3": _family_report(
            profile=settings.h3_profile,
            required_packages=h3_required,
            packages=packages,
            bundle_id="h3-basic",
            bundles=bundles,
        ),
        "ltx23": _family_report(
            profile=settings.ltx23_profile,
            required_packages=ltx23_required,
            packages=packages,
            bundle_id="ltx23-basic",
            bundles=bundles,
        ),
        "wan22": _family_report(
            profile=settings.wan22_profile,
            required_packages=wan22_required,
            packages=packages,
            bundle_id="wan22-basic",
            bundles=bundles,
        ),
        "klein4b": _family_report(
            profile=settings.klein4b_profile,
            required_packages=klein4b_required,
            packages=packages,
            bundle_id="klein4b-basic",
            bundles=bundles,
        ),
        "klein9b": _family_report(
            profile=settings.klein_profile,
            required_packages=klein9b_required,
            packages=packages,
            bundle_id="klein9b-basic",
            bundles=bundles,
        ),
    }

    checks: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        checks.append({"level": level, "code": code, "message": message})

    if cuda["available"]:
        device_names = ", ".join(device["name"] for device in cuda["devices"])
        add("ok", "cuda_available", f"CUDA is available: {device_names}")
    else:
        message = cuda.get("error") or "Torch did not report a CUDA device."
        add("error", "cuda_unavailable", message)

    ready_families = [
        name for name, family in families.items() if family["dependencies_ready"]
    ]
    if ready_families:
        add(
            "ok",
            "runtime_dependencies",
            f"Runtime dependencies are ready for: {', '.join(ready_families)}",
        )
    else:
        add(
            "error",
            "runtime_dependencies",
            "No model family has a complete runtime dependency set.",
        )

    for family_name, family in families.items():
        missing = family["missing_packages"]
        if missing:
            add(
                "warning",
                f"{family_name}_packages_missing",
                f"{family_name} is missing packages: {', '.join(missing)}",
            )
        if family["bundle_status"] == "missing":
            add(
                "warning",
                f"{family_name}_bundle_missing",
                f"{family_name} bundle is not installed; run "
                f"`latentslate-engine bundles install {family['bundle_id']}`.",
            )

    if memory_bytes is not None and memory_bytes < 72 * _GIB:
        add(
            "warning",
            "h3_system_memory",
            (
                f"System RAM is {_format_gib(memory_bytes)} GiB. The upstream H3 "
                "consumer path may require about 75 GiB before OS/application headroom."
            ),
        )

    max_vram = _max_cuda_memory_bytes(cuda)
    if max_vram is not None and max_vram < 24 * _GIB:
        add(
            "warning",
            "ltx23_vram_unvalidated",
            (
                f"The configured LTX 2.3 full-model path has not been validated on "
                f"{_format_gib(max_vram)} GiB VRAM and will rely heavily on CPU offload."
            ),
        )
        add(
            "warning",
            "wan22_vram_unvalidated",
            (
                f"Wan 2.2 TI2V-5B is documented for a 24 GB-class consumer GPU. The "
                f"configured {_format_gib(max_vram)} GiB offload path is best effort."
            ),
        )

    if disk_free_bytes is not None and disk_free_bytes < 30 * _GIB:
        add(
            "warning",
            "disk_space",
            (
                f"Only {_format_gib(disk_free_bytes)} GiB is free near the Engine home. "
                "Model bundles and generated media can consume tens of gigabytes."
            ),
        )

    if settings.klein_profile == "consumer_nvfp4" and platform.system() == "Windows":
        add(
            "warning",
            "klein9b_windows_modelopt",
            (
                "Klein 9B consumer_nvfp4 uses NVIDIA ModelOpt. Native Windows behavior "
                "is not yet validated; Klein 4B or WSL2/Linux is the safer first test."
            ),
        )

    if not token_configured and any(
        bundle["id"] == "klein9b-basic" and bundle["status"] != "installed"
        for bundle in bundles
    ):
        add(
            "warning",
            "huggingface_auth",
            (
                "No Hugging Face token was found. Klein 9B repositories are gated, so "
                "accept their terms and authenticate before installing the bundle."
            ),
        )

    status = "error" if any(item["level"] == "error" for item in checks) else (
        "warning" if any(item["level"] == "warning" for item in checks) else "ok"
    )
    ready_for_inference = bool(cuda["available"] and ready_families)

    return {
        "status": status,
        "ready_for_inference": ready_for_inference,
        "engine_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "engine_home": str(settings.home),
        "system": {
            "memory_bytes": memory_bytes,
            "memory_gib": _format_gib(memory_bytes) if memory_bytes is not None else None,
            "disk_free_bytes": disk_free_bytes,
            "disk_free_gib": (
                _format_gib(disk_free_bytes) if disk_free_bytes is not None else None
            ),
        },
        "cuda": cuda,
        "packages": packages,
        "huggingface": {"token_configured": token_configured},
        "profiles": {
            "h3": settings.h3_profile,
            "ltx23": settings.ltx23_profile,
            "wan22": settings.wan22_profile,
            "klein4b": settings.klein4b_profile,
            "klein9b": settings.klein_profile,
        },
        "families": families,
        "bundles": bundles,
        "checks": checks,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "LatentSlate Engine doctor",
        f"Status: {str(report['status']).upper()}",
        f"Engine: {report['engine_version']}",
        (
            f"Python: {report['python']['version']} "
            f"({report['python']['executable']})"
        ),
        (
            f"Platform: {report['platform']['system']} "
            f"{report['platform']['release']} ({report['platform']['machine']})"
        ),
    ]

    memory = report["system"]["memory_gib"]
    disk = report["system"]["disk_free_gib"]
    lines.append(f"System RAM: {memory if memory is not None else 'unknown'} GiB")
    lines.append(f"Disk free: {disk if disk is not None else 'unknown'} GiB")

    cuda = report["cuda"]
    if cuda["available"]:
        for device in cuda["devices"]:
            capability = ".".join(str(part) for part in device["capability"])
            lines.append(
                f"CUDA {device['index']}: {device['name']} · "
                f"{device['total_memory_gib']} GiB · compute {capability}"
            )
    else:
        lines.append(f"CUDA: unavailable ({cuda.get('error') or 'not reported'})")

    lines.append("Model families:")
    for name, family in report["families"].items():
        readiness = "ready" if family["dependencies_ready"] else "missing dependencies"
        lines.append(
            f"  {name}: {readiness} · profile={family['profile']} · "
            f"bundle={family['bundle_status']}"
        )

    lines.append("Checks:")
    labels = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
    for check in report["checks"]:
        lines.append(
            f"  [{labels.get(check['level'], check['level'].upper())}] "
            f"{check['message']}"
        )
    return "\n".join(lines)


def _family_report(
    *,
    profile: str,
    required_packages: set[str],
    packages: dict[str, dict[str, Any]],
    bundle_id: str,
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = sorted(
        package
        for package in required_packages
        if not packages.get(package, {}).get("available", False)
    )
    bundle_status = next(
        (
            bundle["status"]
            for bundle in bundles
            if bundle["id"] == bundle_id
        ),
        "unknown",
    )
    return {
        "profile": profile,
        "required_packages": sorted(required_packages),
        "missing_packages": missing,
        "dependencies_ready": not missing,
        "bundle_id": bundle_id,
        "bundle_status": bundle_status,
    }


def _package_report() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for label, (module_name, distribution_name) in _PACKAGE_PROBES.items():
        available = importlib.util.find_spec(module_name) is not None
        version = None
        if available:
            try:
                version = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                pass
        report[label] = {"available": available, "version": version}
    return report


def _cuda_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": False,
        "torch_version": None,
        "compiled_cuda_version": None,
        "devices": [],
        "error": None,
    }
    if importlib.util.find_spec("torch") is None:
        report["error"] = "PyTorch is not installed."
        return report
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["compiled_cuda_version"] = getattr(torch.version, "cuda", None)
        report["available"] = bool(torch.cuda.is_available())
        if not report["available"]:
            report["error"] = "PyTorch is installed but CUDA is unavailable."
            return report

        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            total_memory = int(properties.total_memory)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(capability),
                    "total_memory_bytes": total_memory,
                    "total_memory_gib": _format_gib(total_memory),
                }
            )
        report["devices"] = devices
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive broken installs
        report["available"] = False
        report["error"] = f"Failed to inspect PyTorch/CUDA: {exc}"
    return report


def _max_cuda_memory_bytes(cuda: dict[str, Any]) -> int | None:
    devices = cuda.get("devices") or []
    values = [
        int(device["total_memory_bytes"])
        for device in devices
        if device.get("total_memory_bytes") is not None
    ]
    return max(values) if values else None


def _system_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
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

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _disk_free_bytes(path: Path) -> int | None:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return int(shutil.disk_usage(candidate).free)
    except OSError:
        return None


def _hf_token_configured() -> bool:
    try:
        return bool(get_token())
    except Exception:
        return False


def _format_gib(value: int) -> float:
    return round(value / _GIB, 1)
