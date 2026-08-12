"""Deterministic local selection and reporting for Engine accelerator tiers.

This module deliberately has no third-party imports.  The checked-in
PowerShell bootstrap uses the same policy that tests exercise, while doctor
only reads the small non-secret state record it leaves under the Engine home.
"""

from __future__ import annotations

import json
import platform as host_platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

RuntimeTier = Literal["nvidia-cu130", "nvidia-cu128", "protocol"]
SelectionMode = Literal["auto", "cu130", "cu128", "protocol"]

_MIN_PYTHON = (3, 11)
_MAX_PYTHON_EXCLUSIVE = (3, 13)
_MIN_CU130_DRIVER = (580,)
_MIN_CU128_DRIVER_WINDOWS = (570, 65)
_MIN_CU128_DRIVER_LINUX = (570, 26)
_MIN_KITCHEN_SM = (7, 5)
_STATE_FILENAME = "runtime-selection.json"


@dataclass(frozen=True, slots=True)
class GpuFact:
    name: str
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class HardwareFacts:
    system: str
    machine: str
    python_version: tuple[int, int]
    nvidia_driver: tuple[int, ...] | None
    devices: tuple[GpuFact, ...]
    probe_error: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    mode: SelectionMode
    preferred_tier: RuntimeTier
    selected_tier: RuntimeTier
    reason: str
    impact: str
    fallback_reason: str | None = None

    @property
    def uses_runtime_group(self) -> bool:
        return self.selected_tier != "protocol"

    @property
    def extra(self) -> str:
        return self.selected_tier


class RuntimeSelectionError(ValueError):
    """The requested tier cannot be selected from deterministic local facts."""


def parse_version(value: str | None) -> tuple[int, ...] | None:
    """Parse a dotted driver version without treating arbitrary text as a version."""

    if not value or not re.fullmatch(r"\d+(?:\.\d+)*", value.strip()):
        return None
    return tuple(int(part) for part in value.strip().split("."))


def parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if not value or not re.fullmatch(r"\d+\.\d+", value.strip()):
        return None
    major, minor = value.strip().split(".", maxsplit=1)
    return int(major), int(minor)


def version_at_least(actual: tuple[int, ...] | None, minimum: tuple[int, ...]) -> bool:
    if actual is None:
        return False
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (width - len(minimum))


def python_is_supported(version: tuple[int, int]) -> bool:
    return _MIN_PYTHON <= version < _MAX_PYTHON_EXCLUSIVE


def supported_platform(facts: HardwareFacts) -> bool:
    return facts.system in {"Windows", "Linux"} and facts.machine.lower() in {
        "amd64",
        "x86_64",
    }


def has_kitchen_architecture(facts: HardwareFacts) -> bool:
    """Check the default CUDA device, which runtime validation also probes.

    A mixed-GPU host must not select cu130 merely because a secondary adapter
    qualifies while PyTorch's device 0 does not. Users can make a qualifying
    adapter the default with their normal CUDA visibility/device selection.
    """

    if not facts.devices:
        return False
    return supports_kitchen_capability(facts.devices[0].compute_capability)


def supports_kitchen_capability(capability: tuple[int, int] | None) -> bool:
    """Return whether a CUDA device meets Kitchen's tested architecture floor."""

    return capability is not None and capability >= _MIN_KITCHEN_SM


def can_use_cu130(facts: HardwareFacts) -> bool:
    return (
        supported_platform(facts)
        and python_is_supported(facts.python_version)
        and bool(facts.devices)
        and has_kitchen_architecture(facts)
        and version_at_least(facts.nvidia_driver, _MIN_CU130_DRIVER)
    )


def can_use_cu128(facts: HardwareFacts) -> bool:
    if not (
        supported_platform(facts)
        and python_is_supported(facts.python_version)
        and bool(facts.devices)
        and has_kitchen_architecture(facts)
    ):
        return False
    minimum = (
        _MIN_CU128_DRIVER_WINDOWS if facts.system == "Windows" else _MIN_CU128_DRIVER_LINUX
    )
    return version_at_least(facts.nvidia_driver, minimum)


def choose_runtime(mode: SelectionMode, facts: HardwareFacts) -> RuntimeSelection:
    """Select one locked tier; auto never guesses around failed prerequisites."""

    if not python_is_supported(facts.python_version):
        raise RuntimeSelectionError(
            "LatentSlate Engine requires Python >=3.11,<3.13; "
            f"detected {facts.python_version[0]}.{facts.python_version[1]}."
        )

    if mode == "protocol":
        return RuntimeSelection(
            mode=mode,
            preferred_tier="protocol",
            selected_tier="protocol",
            reason="Protocol-only tier was explicitly requested.",
            impact="Inference dependencies and CUDA recipes are not installed.",
        )

    if mode == "cu130":
        if not can_use_cu130(facts):
            raise RuntimeSelectionError(_unavailable_message("nvidia-cu130", facts))
        return RuntimeSelection(
            mode=mode,
            preferred_tier="nvidia-cu130",
            selected_tier="nvidia-cu130",
            reason="CUDA 13.0 was explicitly requested and local prerequisites match.",
            impact="Installs the recommended CUDA 13.0 runtime and Comfy Kitchen CUBLAS support.",
        )

    if mode == "cu128":
        if not can_use_cu128(facts):
            raise RuntimeSelectionError(_unavailable_message("nvidia-cu128", facts))
        return RuntimeSelection(
            mode=mode,
            preferred_tier="nvidia-cu128",
            selected_tier="nvidia-cu128",
            reason="CUDA 12.8 compatibility tier was explicitly requested.",
            impact="CUDA 13-only Kitchen acceleration is not selected.",
        )

    if can_use_cu130(facts):
        return RuntimeSelection(
            mode="auto",
            preferred_tier="nvidia-cu130",
            selected_tier="nvidia-cu130",
            reason="Detected a supported NVIDIA GPU, CUDA 13-capable driver, and Python runtime.",
            impact="Maximum currently-tested Engine recipe compatibility is selected.",
        )
    if can_use_cu128(facts):
        return RuntimeSelection(
            mode="auto",
            preferred_tier="nvidia-cu130",
            selected_tier="nvidia-cu128",
            reason=_auto_cu128_reason(facts),
            impact="CUDA 13-only Kitchen acceleration is unavailable; CUDA 12.8 compatibility recipes remain available.",
            fallback_reason=_auto_cu128_reason(facts),
        )
    return RuntimeSelection(
        mode="auto",
        preferred_tier="nvidia-cu130",
        selected_tier="protocol",
        reason=_auto_protocol_reason(facts),
        impact="No compatible NVIDIA runtime was detected; protocol/catalog tooling remains available.",
        fallback_reason=_auto_protocol_reason(facts),
    )


def classify_validation_failure(payload: dict[str, Any]) -> str | None:
    """Return a narrow fallback class, or ``None`` for failures that must stop."""

    if payload.get("ok"):
        return None
    code = payload.get("error_code")
    if code in {
        "torch_cuda_unavailable",
        "torch_cuda_mismatch",
        "kitchen_cuda_backend_unavailable",
    }:
        return "cuda_or_kitchen_incompatibility"
    # A default device below Kitchen's SM floor would fail the cu128 validation
    # too, so retrying a smaller CUDA runtime would be misleading.
    if code == "torch_device_capability_unsupported":
        return None
    return None


def apply_validation_fallback(
    selection: RuntimeSelection,
    validation: dict[str, Any],
) -> RuntimeSelection | None:
    """Return the one permitted auto fallback, never hiding general install failures."""

    if (
        selection.mode != "auto"
        or selection.selected_tier != "nvidia-cu130"
        or classify_validation_failure(validation) != "cuda_or_kitchen_incompatibility"
    ):
        return None
    detail = str(validation.get("message") or validation.get("error_code"))
    return RuntimeSelection(
        mode="auto",
        preferred_tier="nvidia-cu130",
        selected_tier="nvidia-cu128",
        reason="CUDA 13 validation failed with a classified CUDA/Kitchen compatibility error.",
        impact="CUDA 13 Kitchen acceleration is unavailable; CUDA 12.8 compatibility recipes remain available.",
        fallback_reason=detail,
    )


def probe_hardware(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> HardwareFacts:
    """Collect the small fact set needed before any accelerator sync occurs."""

    system = host_platform.system()
    machine = host_platform.machine()
    python_version = (sys.version_info.major, sys.version_info.minor)
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return HardwareFacts(system, machine, python_version, None, (), str(exc))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        return HardwareFacts(system, machine, python_version, None, (), detail)

    driver: tuple[int, ...] | None = None
    devices: list[GpuFact] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        parsed_driver = parse_version(values[0])
        if parsed_driver is None:
            continue
        driver = driver or parsed_driver
        devices.append(GpuFact(name=values[1], compute_capability=parse_compute_capability(values[2])))
    return HardwareFacts(system, machine, python_version, driver, tuple(devices))


def selection_payload(selection: RuntimeSelection, facts: HardwareFacts) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_mode": selection.mode,
        "preferred_tier": selection.preferred_tier,
        "selected_tier": selection.selected_tier,
        "fallback_reason": selection.fallback_reason,
        "reason": selection.reason,
        "impact": selection.impact,
        "hardware": {
            "system": facts.system,
            "machine": facts.machine,
            "python_version": ".".join(str(part) for part in facts.python_version),
            "nvidia_driver": ".".join(str(part) for part in facts.nvidia_driver)
            if facts.nvidia_driver
            else None,
            "devices": [
                {
                    "name": device.name,
                    "compute_capability": (
                        ".".join(str(part) for part in device.compute_capability)
                        if device.compute_capability
                        else None
                    ),
                }
                for device in facts.devices
            ],
            "probe_error": facts.probe_error,
        },
    }


def runtime_selection_path(engine_home: Path) -> Path:
    return engine_home / "runtime" / _STATE_FILENAME


def write_selection_state(
    engine_home: Path,
    selection: RuntimeSelection,
    facts: HardwareFacts,
    validation: dict[str, Any],
) -> Path:
    path = runtime_selection_path(engine_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = selection_payload(selection, facts)
    payload["validation"] = validation
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_selection_state(engine_home: Path) -> dict[str, Any]:
    """Read a bootstrap record defensively; malformed local state is non-fatal."""

    path = runtime_selection_path(engine_home)
    baseline: dict[str, Any] = {
        "state": "not_bootstrapped",
        "selection_mode": None,
        "preferred_tier": None,
        "selected_tier": None,
        "fallback_reason": None,
        "reason": None,
        "impact": None,
        "validation": None,
    }
    try:
        # Windows PowerShell 5.1's UTF-8 writer emits a BOM. Accept it as
        # local state so an otherwise valid bootstrap record never disappears.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return baseline
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return baseline
    for key in baseline:
        if key == "state":
            continue
        value = payload.get(key)
        if value is None or isinstance(value, (str, dict)):
            baseline[key] = value
    baseline["state"] = "recorded"
    return baseline


def _unavailable_message(tier: RuntimeTier, facts: HardwareFacts) -> str:
    if not supported_platform(facts):
        return f"{tier} requires Windows or Linux x86_64."
    if not facts.devices:
        return f"{tier} requires an NVIDIA GPU detected by nvidia-smi."
    if tier == "nvidia-cu130":
        return (
            "nvidia-cu130 requires an NVIDIA r580+ driver and a Comfy Kitchen-supported "
            "GPU architecture (SM >= 7.5)."
        )
    return (
        "nvidia-cu128 requires the CUDA 12.8 minimum NVIDIA driver and a Comfy Kitchen-supported "
        "default GPU architecture (SM >= 7.5)."
    )


def _auto_cu128_reason(facts: HardwareFacts) -> str:
    if not has_kitchen_architecture(facts):
        return "GPU is below the tested Comfy Kitchen CUDA architecture floor for CUDA 13."
    return "NVIDIA driver is below the CUDA 13 / r580 minimum."


def _auto_protocol_reason(facts: HardwareFacts) -> str:
    if not supported_platform(facts):
        return "Platform is outside the Windows/Linux x86_64 NVIDIA runtime tiers."
    if not facts.devices:
        return "No NVIDIA GPU was detected by nvidia-smi."
    if not has_kitchen_architecture(facts):
        return "Default NVIDIA GPU is below the tested Comfy Kitchen architecture floor (SM >= 7.5)."
    return "NVIDIA driver is below the CUDA 12.8 compatibility minimum."


def facts_as_dict(facts: HardwareFacts) -> dict[str, Any]:
    payload = asdict(facts)
    payload["python_version"] = list(facts.python_version)
    payload["nvidia_driver"] = list(facts.nvidia_driver) if facts.nvidia_driver else None
    for device in payload["devices"]:
        capability = device["compute_capability"]
        device["compute_capability"] = list(capability) if capability else None
    return payload
