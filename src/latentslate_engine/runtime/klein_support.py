from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..hardware import capability_metadata, supports_nvfp4

_CORE_MODULES = ("torch", "diffusers", "transformers", "accelerate", "PIL")
_VERSION_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "kernels",
    "peft",
    "torchao",
    "nvidia-modelopt",
)


@dataclass(frozen=True, slots=True)
class KleinRuntimeSupport:
    core_available: bool
    core_reason: str | None
    kernels_available: bool
    kernels_reason: str | None
    peft_available: bool
    peft_reason: str | None
    torchao_available: bool
    torchao_reason: str | None
    modelopt_available: bool
    modelopt_reason: str | None
    cuda_available: bool
    cuda_capability: tuple[int, int] | None
    nvfp4_available: bool
    nvfp4_reason: str | None
    versions: tuple[tuple[str, str], ...]

    def version_summary(self) -> str:
        return ", ".join(f"{name}={version}" for name, version in self.versions)


def _installed_versions() -> tuple[tuple[str, str], ...]:
    versions: list[tuple[str, str]] = []
    for package in _VERSION_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        versions.append((package, version))
    return tuple(versions)


def _import_check(module_name: str, attributes: tuple[str, ...] = ()) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(module_name)
        missing = [attribute for attribute in attributes if not hasattr(module, attribute)]
        if missing:
            return False, f"{module_name} is missing: {', '.join(missing)}"
        return True, None
    except Exception as exc:  # noqa: BLE001 - availability must surface import-time failures
        return False, f"{module_name} import failed: {type(exc).__name__}: {exc}"


def nvfp4_host_status(torch_module: Any) -> tuple[bool, tuple[int, int] | None, str | None]:
    try:
        if not bool(torch_module.cuda.is_available()):
            return False, None, "NVFP4 requires a visible CUDA GPU"
        device = torch_module.cuda.current_device()
        capability = tuple(int(value) for value in torch_module.cuda.get_device_capability(device))
    except Exception as exc:  # noqa: BLE001 - host probing must become an unavailable reason
        return False, None, f"CUDA capability probe failed: {type(exc).__name__}: {exc}"

    if not supports_nvfp4(capability):
        metadata = capability_metadata(capability)
        reason = (
            "NVFP4 requires Blackwell-class SM100+ hardware; "
            f"detected {str(metadata['sm']).upper()} {metadata['architecture']}"
        )
        return False, capability, reason
    return True, capability, None


@lru_cache(maxsize=1)
def klein_runtime_support() -> KleinRuntimeSupport:
    versions = _installed_versions()
    missing = [module for module in _CORE_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        reason = "Run `uv sync`; missing Klein runtime packages: " + ", ".join(missing)
        return KleinRuntimeSupport(
            core_available=False,
            core_reason=reason,
            kernels_available=False,
            kernels_reason="Klein core runtime is unavailable",
            peft_available=False,
            peft_reason="Klein core runtime is unavailable",
            torchao_available=False,
            torchao_reason="Klein core runtime is unavailable",
            modelopt_available=False,
            modelopt_reason="Klein core runtime is unavailable",
            cuda_available=False,
            cuda_capability=None,
            nvfp4_available=False,
            nvfp4_reason="Klein core runtime is unavailable",
            versions=versions,
        )

    try:
        diffusers = importlib.import_module("diffusers")
        transformers = importlib.import_module("transformers")
        for symbol in ("Flux2KleinPipeline", "Flux2Transformer2DModel"):
            getattr(diffusers, symbol)
        for symbol in ("Qwen3ForCausalLM", "Qwen2TokenizerFast"):
            getattr(transformers, symbol)
        torch_module = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - report the actual dependency failure in catalog
        version_text = ", ".join(f"{name}={version}" for name, version in versions)
        reason = (
            "FLUX.2 Klein runtime import failed: "
            f"{type(exc).__name__}: {exc}. Installed stack: {version_text}"
        )
        return KleinRuntimeSupport(
            core_available=False,
            core_reason=reason,
            kernels_available=False,
            kernels_reason=reason,
            peft_available=False,
            peft_reason=reason,
            torchao_available=False,
            torchao_reason=reason,
            modelopt_available=False,
            modelopt_reason=reason,
            cuda_available=False,
            cuda_capability=None,
            nvfp4_available=False,
            nvfp4_reason=reason,
            versions=versions,
        )

    kernels_available, kernels_reason = _import_check("kernels")
    peft_available, peft_reason = _import_check("peft")
    torchao_available, torchao_reason = _import_check(
        "torchao.quantization", ("Int8WeightOnlyConfig",)
    )
    modelopt_available, modelopt_reason = _import_check(
        "modelopt.torch.opt", ("enable_huggingface_checkpointing",)
    )
    host_nvfp4, capability, host_reason = nvfp4_host_status(torch_module)
    nvfp4_available = modelopt_available and host_nvfp4
    nvfp4_reason = None
    if not modelopt_available:
        nvfp4_reason = modelopt_reason or "NVIDIA ModelOpt is unavailable"
    elif not host_nvfp4:
        nvfp4_reason = host_reason

    return KleinRuntimeSupport(
        core_available=True,
        core_reason=None,
        kernels_available=kernels_available,
        kernels_reason=kernels_reason,
        peft_available=peft_available,
        peft_reason=peft_reason,
        torchao_available=torchao_available,
        torchao_reason=torchao_reason,
        modelopt_available=modelopt_available,
        modelopt_reason=modelopt_reason,
        cuda_available=capability is not None,
        cuda_capability=capability,
        nvfp4_available=nvfp4_available,
        nvfp4_reason=nvfp4_reason,
        versions=versions,
    )


def clear_klein_runtime_support_cache() -> None:
    klein_runtime_support.cache_clear()
