from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from functools import lru_cache

_CORE_MODULES = ("torch", "diffusers", "transformers", "accelerate", "PIL")
_VERSION_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "kernels",
    "peft",
)


def _import_check(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - availability must report import failures
        return False, f"{module_name} import failed: {type(exc).__name__}: {exc}"
    return True, None


@dataclass(frozen=True, slots=True)
class KleinRuntimeSupport:
    core_available: bool
    core_reason: str | None
    kernels_available: bool
    kernels_reason: str | None
    peft_available: bool
    peft_reason: str | None
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
            versions=versions,
        )

    try:
        diffusers = importlib.import_module("diffusers")
        transformers = importlib.import_module("transformers")
        for symbol in ("Flux2KleinPipeline", "Flux2Transformer2DModel"):
            getattr(diffusers, symbol)
        for symbol in ("Qwen3ForCausalLM", "Qwen2TokenizerFast"):
            getattr(transformers, symbol)
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
            versions=versions,
        )

    kernels_available, kernels_reason = _import_check("kernels")
    peft_available, peft_reason = _import_check("peft")
    return KleinRuntimeSupport(
        core_available=True,
        core_reason=None,
        kernels_available=kernels_available,
        kernels_reason=kernels_reason,
        peft_available=peft_available,
        peft_reason=peft_reason,
        versions=versions,
    )


def clear_klein_runtime_support_cache() -> None:
    klein_runtime_support.cache_clear()
