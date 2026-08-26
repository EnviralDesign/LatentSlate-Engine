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
from rich.console import RenderableType
from rich.text import Text

from . import __version__
from .bundles import descriptors as bundle_descriptors
from .cli_presentation import (
    bootstrap_command,
    data_table,
    engine_command,
    key_values,
    next_action,
    page,
    panel,
    status,
)
from .config import Settings
from .hardware import capability_metadata, supports_fp8, supports_nvfp4
from .runtime_selection import read_selection_state

_GIB = 1024**3
_SUPPORTED_PROFILES: dict[str, tuple[str, ...]] = {
    "h3": ("bf16_auto_offload",),
    "ltx23": ("bf16_sequential_offload", "bf16_model_offload", "bf16_cuda"),
    "wan22": (
        "bf16_sequential_offload",
        "bf16_group_leaf",
        "bf16_model_offload",
        "bf16_cuda",
    ),
    "klein4b": ("bf16_model_offload", "bf16_cuda"),
    "klein9b": ("bf16_model_offload", "bf16_cuda"),
}
_PROFILE_ENV = {
    "h3": "LATENTSLATE_H3_PROFILE",
    "ltx23": "LATENTSLATE_LTX23_PROFILE",
    "wan22": "LATENTSLATE_WAN22_PROFILE",
    "klein4b": "LATENTSLATE_KLEIN4B_PROFILE",
    "klein9b": "LATENTSLATE_KLEIN_PROFILE",
}
_PACKAGE_PROBES: dict[str, tuple[str, str]] = {
    "accelerate": ("accelerate", "accelerate"),
    "av": ("av", "av"),
    "comfy_aimdo": ("comfy_aimdo", "comfy-aimdo"),
    "diffusers": ("diffusers", "diffusers"),
    "ftfy": ("ftfy", "ftfy"),
    "numpy": ("numpy", "numpy"),
    "pillow": ("PIL", "pillow"),
    "sentencepiece": ("sentencepiece", "sentencepiece"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
}


def collect_report(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    packages = _package_report()
    cuda = _cuda_report()
    kitchen = _kitchen_report(cuda)
    runtime_selection = read_selection_state(settings.home)
    memory_bytes = _system_memory_bytes()
    engine_disk_free_bytes = _disk_free_bytes(settings.home)
    model_disk_free_bytes = _disk_free_bytes(settings.model_root)
    token_configured = _hf_token_configured()
    bundles = [
        descriptor.model_dump(mode="json")
        for descriptor in bundle_descriptors(settings.model_root, settings)
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
    h3_required = set(video_base)
    # AIMDO is reported by the package probes, but it is not a hard LTX
    # prerequisite: the default auto policy may select the bounded Engine-hook
    # fallback before any dynamic allocation starts.
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

    legacy_profiles = {
        ("h3", "consumer_int8"): "Set LATENTSLATE_H3_PROFILE=bf16_auto_offload.",
        ("wan22", "int8_model_offload"): (
            "Install a supported pre-quantized artifact when its loader lands, or set "
            "LATENTSLATE_WAN22_PROFILE=bf16_sequential_offload."
        ),
        ("klein4b", "consumer_int8"): ("Set LATENTSLATE_KLEIN4B_PROFILE=bf16_model_offload."),
        ("klein9b", "consumer_int8"): ("Set LATENTSLATE_KLEIN_PROFILE=bf16_model_offload."),
        ("klein9b", "consumer_nvfp4"): (
            "Set LATENTSLATE_KLEIN_PROFILE=bf16_model_offload; the partial NVFP4 "
            "conversion recipe was removed."
        ),
    }
    for family, profile in (
        ("h3", settings.h3_profile),
        ("ltx23", settings.ltx23_profile),
        ("wan22", settings.wan22_profile),
        ("klein4b", settings.klein4b_profile),
        ("klein9b", settings.klein_profile),
    ):
        profile_valid = profile in _SUPPORTED_PROFILES[family]
        families[family]["profile_valid"] = profile_valid
        families[family]["dependencies_ready"] = bool(
            families[family]["dependencies_ready"] and profile_valid
        )
        if profile_valid:
            continue
        if migration := legacy_profiles.get((family, profile)):
            add(
                "error",
                f"{family}_legacy_conversion_profile",
                f"{family} profile {profile!r} is no longer valid. {migration}",
            )
        else:
            supported = ", ".join(_SUPPORTED_PROFILES[family])
            add(
                "error",
                f"{family}_invalid_profile",
                f"{_PROFILE_ENV[family]}={profile!r} is invalid; expected one of: {supported}.",
            )

    if cuda["available"]:
        device_names = ", ".join(device["name"] for device in cuda["devices"])
        add("ok", "cuda_available", f"CUDA is available: {device_names}")
    else:
        message = cuda.get("error") or "Torch did not report a CUDA device."
        add("error", "cuda_unavailable", message)

    if runtime_selection["state"] == "recorded":
        selected_tier = runtime_selection.get("selected_tier") or "unknown"
        fallback_reason = runtime_selection.get("fallback_reason")
        message = f"Runtime tier: {selected_tier} (mode={runtime_selection.get('selection_mode')})."
        if fallback_reason:
            message += f" Fallback: {fallback_reason}"
        add("ok", "runtime_selection", message)
        _add_runtime_drift_checks(runtime_selection, cuda, packages, kitchen, add)
    else:
        add(
            "warning",
            "runtime_selection_missing",
            f"No bootstrap runtime selection is recorded; run `{bootstrap_command()}`.",
        )

    if kitchen["available"]:
        if kitchen.get("probe_mode") == "metadata_only":
            add(
                "ok",
                "kitchen_probe",
                "Comfy Kitchen is installed; backend dispatch is verified only inside disposable workers.",
            )
        else:
            backend = "ready" if kitchen["cuda_backend"] else "eager/triton only"
            add("ok", "kitchen_probe", f"Comfy Kitchen is available ({backend}).")
    elif runtime_selection.get("selected_tier") == "protocol":
        add(
            "ok",
            "kitchen_probe",
            "Comfy Kitchen is intentionally absent in the protocol-only tier.",
        )
    else:
        add("warning", "kitchen_probe", f"Comfy Kitchen is unavailable: {kitchen['error']}")

    ready_families = [name for name, family in families.items() if family["dependencies_ready"]]
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
                f"The configured distilled LTX 2.3 path has not been validated on "
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

    if model_disk_free_bytes is not None and model_disk_free_bytes < 30 * _GIB:
        add(
            "warning",
            "model_disk_space",
            (
                f"Only {_format_gib(model_disk_free_bytes)} GiB is free at the model root. "
                "Model bundles can consume tens of gigabytes."
            ),
        )

    if not token_configured and any(bundle["status"] != "installed" for bundle in bundles):
        add(
            "warning",
            "huggingface_auth",
            (
                "No Hugging Face token was found. Public repositories may still work, "
                "but gated models require HF_TOKEN in the process environment or local .env."
            ),
        )

    status = (
        "error"
        if any(item["level"] == "error" for item in checks)
        else ("warning" if any(item["level"] == "warning" for item in checks) else "ok")
    )
    ready_for_inference = bool(
        cuda["available"]
        and ready_families
        and not any(item["level"] == "error" for item in checks)
    )

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
        "model_store": {
            "root": str(settings.model_root),
            "exists": settings.model_root.is_dir(),
            "disk_free_bytes": model_disk_free_bytes,
            "disk_free_gib": (
                _format_gib(model_disk_free_bytes) if model_disk_free_bytes is not None else None
            ),
        },
        "system": {
            "memory_bytes": memory_bytes,
            "memory_gib": _format_gib(memory_bytes) if memory_bytes is not None else None,
            "disk_free_bytes": engine_disk_free_bytes,
            "disk_free_gib": (
                _format_gib(engine_disk_free_bytes) if engine_disk_free_bytes is not None else None
            ),
        },
        "cuda": cuda,
        "kitchen": kitchen,
        "runtime": runtime_selection,
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


def format_report(report: dict[str, Any]) -> RenderableType:
    """Render the human doctor view; callers retain the raw report for JSON."""

    status_kind = {"ok": "ok", "warning": "warn", "error": "bad"}.get(report["status"], "warn")
    system_rows: list[tuple[str, str | Text]] = [
        ("Engine", report["engine_version"]),
        ("Python", f"{report['python']['version']} ({report['python']['executable']})"),
        (
            "Platform",
            f"{report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})",
        ),
    ]
    memory = report["system"]["memory_gib"]
    disk = report["system"]["disk_free_gib"]
    model_disk = report["model_store"]["disk_free_gib"]
    system_rows.extend(
        (
            ("System RAM", f"{memory if memory is not None else 'unknown'} GiB"),
            ("Engine home", report["engine_home"]),
            ("Engine disk free", f"{disk if disk is not None else 'unknown'} GiB"),
            ("Model root", report["model_store"]["root"]),
            ("Model disk free", f"{model_disk if model_disk is not None else 'unknown'} GiB"),
        )
    )

    cuda = report["cuda"]
    torch_version = cuda.get("torch_version") or "not installed"
    compiled_cuda = cuda.get("compiled_cuda_version") or "CPU-only"
    cuda_rows: list[tuple[str, str | Text]] = [
        ("PyTorch", f"{torch_version} · build={compiled_cuda}")
    ]
    if cuda["available"]:
        for device in cuda["devices"]:
            capability = ".".join(str(part) for part in device["capability"])
            sm = str(device.get("sm") or "").upper()
            architecture = device.get("architecture") or "unknown architecture"
            capabilities = device.get("capabilities") or {}
            enabled = [name.upper() for name, value in capabilities.items() if value]
            suffix = f" · {', '.join(enabled)}" if enabled else ""
            cuda_rows.append(
                (
                    f"CUDA {device['index']}",
                    f"{device['name']} · {device['total_memory_gib']} GiB · compute {capability} · {sm} {architecture}{suffix}",
                )
            )
    else:
        cuda_rows.append(
            ("CUDA", status(f"UNAVAILABLE · {cuda.get('error') or 'not reported'}", "warn"))
        )

    runtime = report.get("runtime", {})
    runtime_rows: list[tuple[str, str | Text]] = [
        ("Selection state", str(runtime.get("state") or "unknown")),
        ("Selection mode", str(runtime.get("selection_mode") or "not recorded")),
        ("Preferred tier", str(runtime.get("preferred_tier") or "not recorded")),
        ("Selected tier", str(runtime.get("selected_tier") or "not recorded")),
    ]
    if runtime.get("fallback_reason"):
        runtime_rows.append(("Fallback reason", str(runtime["fallback_reason"])))
    if runtime.get("impact"):
        runtime_rows.append(("Impact", str(runtime["impact"])))

    kitchen = report.get("kitchen", {})
    kitchen_rows: list[tuple[str, str | Text]] = [
        ("State", "READY" if kitchen.get("available") else "UNAVAILABLE"),
        ("Probe", str(kitchen.get("probe_mode") or "runtime")),
        ("Backends", ", ".join(kitchen.get("backends") or []) or "none"),
        (
            "CUDA backend",
            "READY"
            if kitchen.get("cuda_backend")
            else (
                "reported, but not qualified by the selected Torch CUDA build"
                if kitchen.get("cuda_backend_reported")
                else (
                    "deferred to disposable worker"
                    if kitchen.get("probe_mode") == "metadata_only"
                    else "not active"
                )
            ),
        ),
    ]
    capabilities = kitchen.get("capabilities") or {}
    kitchen_rows.append(
        (
            "Validated capabilities",
            ", ".join(name.upper() for name, value in capabilities.items() if value) or "none",
        )
    )
    if kitchen.get("error"):
        kitchen_rows.append(("Detail", str(kitchen["error"])))

    token_label = "configured" if report["huggingface"]["token_configured"] else "not configured"
    system_rows.append(
        (
            "Hugging Face token",
            status(token_label.upper(), "ok" if token_label == "configured" else "warn"),
        )
    )

    family_table = data_table(
        "Family", "Runtime", "Default mode", "Legacy bundle", ratio=(1, 1, 2, 1)
    )
    for name, family in report["families"].items():
        runtime_dependencies = "READY" if family["runtime_dependencies_ready"] else "MISSING"
        if not family.get("profile_valid", True):
            execution_mode = f"INVALID · {family['profile']}"
        else:
            execution_mode = family["profile"]
        family_table.add_row(
            name,
            status(runtime_dependencies, "ok" if family["runtime_dependencies_ready"] else "bad"),
            execution_mode,
            family["bundle_status"],
        )
    checks = data_table("Status", "Check", ratio=(1, 5))
    for check in report["checks"]:
        level = check["level"]
        label = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(level, level.upper())
        checks.add_row(
            status(label, {"ok": "ok", "warning": "warn", "error": "bad"}.get(level, "warn")),
            check["message"],
        )
    return page(
        "LatentSlate Engine doctor",
        panel(
            "Status",
            status(str(report["status"]).upper(), status_kind),
            style={"ok": "green", "warn": "yellow", "bad": "red"}[status_kind],
        ),
        panel("System", key_values(system_rows)),
        panel("Runtime selection", key_values(runtime_rows)),
        panel("CUDA", key_values(cuda_rows)),
        panel("Comfy Kitchen", key_values(kitchen_rows)),
        panel("Model families · runtime prerequisites only", family_table),
        panel(
            "Recipes",
            Text(
                "Not inspected. Run `latentslate-engine recipes list` for runnable recipe availability."
            ),
            style="yellow",
        ),
        panel("Checks", checks),
        next_action(engine_command("recipes", "list"), label="Recipe availability"),
    )


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
        (bundle["status"] for bundle in bundles if bundle["id"] == bundle_id),
        "unknown",
    )
    return {
        "profile": profile,
        "required_packages": sorted(required_packages),
        "missing_packages": missing,
        "runtime_dependencies_ready": not missing,
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
        compiled_cuda = getattr(torch.version, "cuda", None)
        report["compiled_cuda_version"] = compiled_cuda
        report["available"] = bool(torch.cuda.is_available())
        if not report["available"]:
            report["error"] = _cuda_unavailable_error(compiled_cuda)
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
                    "total_memory_bytes": total_memory,
                    "total_memory_gib": _format_gib(total_memory),
                    **capability_metadata(capability),
                }
            )
        report["devices"] = devices
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive broken installs
        report["available"] = False
        report["error"] = f"Failed to inspect PyTorch/CUDA: {exc}"
    return report


def _kitchen_report(cuda: dict[str, Any]) -> dict[str, Any]:
    """Inspect Kitchen metadata without importing kernels into the Engine parent."""

    report: dict[str, Any] = {
        "available": False,
        "version": None,
        "probe_mode": "metadata_only",
        "backends": [],
        "cuda_backend_reported": None,
        "cuda_backend": None,
        "capabilities": {"fp8": False, "nvfp4": False, "mxfp8": False},
        "error": None,
    }
    if importlib.util.find_spec("comfy_kitchen") is None:
        report["error"] = "Package is not installed."
        return report
    report["available"] = True
    try:
        report["version"] = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        report["error"] = "Package metadata is unavailable."
    return report


def _add_runtime_drift_checks(
    runtime: dict[str, Any],
    cuda: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    kitchen: dict[str, Any],
    add: Any,
) -> None:
    """Fail closed when recorded bootstrap state disagrees with installed reality."""

    tier = runtime.get("selected_tier")
    compiled_cuda = str(cuda.get("compiled_cuda_version") or "")
    torch_available = bool(packages.get("torch", {}).get("available"))
    kitchen_available = bool(kitchen.get("available"))
    if tier == "nvidia-cu130" and not compiled_cuda.startswith("13."):
        add(
            "error",
            "runtime_tier_drift",
            "Recorded nvidia-cu130 tier does not match the installed PyTorch CUDA build "
            f"({compiled_cuda or 'CPU-only'}). Re-run the bootstrap for this tier.",
        )
    elif tier == "nvidia-cu128" and not compiled_cuda.startswith("12.8"):
        add(
            "error",
            "runtime_tier_drift",
            "Recorded nvidia-cu128 tier does not match the installed PyTorch CUDA build "
            f"({compiled_cuda or 'CPU-only'}). Re-run the bootstrap for this tier.",
        )
    elif tier == "protocol" and (torch_available or kitchen_available):
        add(
            "error",
            "runtime_tier_drift",
            "Recorded protocol tier has model runtime packages installed. Re-run "
            "the protocol bootstrap for a clean catalog-only environment or bootstrap an NVIDIA tier.",
        )
    elif tier in {"nvidia-cu130", "nvidia-cu128"} and not kitchen_available:
        add(
            "error",
            "runtime_kitchen_missing",
            f"Recorded {tier} tier is missing Comfy Kitchen. Re-run the bootstrap for this tier.",
        )
    elif (
        tier == "nvidia-cu130"
        and kitchen.get("probe_mode") != "metadata_only"
        and not kitchen.get("cuda_backend")
    ):
        add(
            "error",
            "runtime_kitchen_cuda_backend_missing",
            "Recorded nvidia-cu130 tier does not have a qualified Comfy Kitchen CUDA backend. "
            "Re-run bootstrap or select the compatibility tier.",
        )


def _cuda_unavailable_error(compiled_cuda_version: str | None) -> str:
    if compiled_cuda_version is None:
        return (
            "A CPU-only PyTorch build is installed. Re-run the adaptive bootstrap "
            f"(`{bootstrap_command()}`) or use the recorded Engine wrapper."
        )
    return (
        f"PyTorch was built for CUDA {compiled_cuda_version}, but no CUDA device is "
        "available. Check the NVIDIA driver and whether the GPU is visible to this process."
    )


def _cuda_has_capability(cuda: dict[str, Any], name: str) -> bool:
    for device in cuda.get("devices") or []:
        capabilities = device.get("capabilities") or {}
        if capabilities.get(name):
            return True
        raw = device.get("capability")
        if not raw:
            continue
        if name == "fp8" and supports_fp8(raw):
            return True
        if name == "nvfp4" and supports_nvfp4(raw):
            return True
    return False


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
    except Exception:  # noqa: BLE001 - optional local token sources may fail
        return False


def _format_gib(value: int) -> float:
    return round(value / _GIB, 1)
