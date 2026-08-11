from __future__ import annotations

from collections.abc import Sequence
from typing import Any

CudaCapability = tuple[int, int]


def normalize_capability(capability: Sequence[int]) -> CudaCapability:
    if len(capability) < 2:
        raise ValueError("CUDA capability must contain major and minor values")
    return int(capability[0]), int(capability[1])


def sm_number(capability: Sequence[int]) -> int:
    major, minor = normalize_capability(capability)
    return major * 10 + minor


def architecture_name(capability: Sequence[int]) -> str:
    sm = sm_number(capability)
    if sm >= 100:
        return "Blackwell or newer"
    if sm >= 90:
        return "Hopper"
    if sm >= 89:
        return "Ada Lovelace"
    if sm >= 80:
        return "Ampere"
    if sm >= 75:
        return "Turing"
    if sm >= 70:
        return "Volta"
    return "pre-Volta or unknown"


def supports_fp8(capability: Sequence[int]) -> bool:
    return sm_number(capability) >= 89


def supports_nvfp4(capability: Sequence[int]) -> bool:
    return sm_number(capability) >= 100


def capability_metadata(capability: Sequence[int]) -> dict[str, Any]:
    normalized = normalize_capability(capability)
    sm = sm_number(normalized)
    return {
        "capability": list(normalized),
        "sm": f"sm{sm}",
        "architecture": architecture_name(normalized),
        "capabilities": {
            "fp8": supports_fp8(normalized),
            "nvfp4": supports_nvfp4(normalized),
        },
    }
