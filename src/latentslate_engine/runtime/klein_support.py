from __future__ import annotations

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


def _module_check(module_name: str) -> tuple[bool, str | None]:
    if importlib.util.find_spec(module_name) is None:
        return False, f"{module_name} is not installed"
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
    """Report catalog-time package presence without importing the GPU stack.

    Bootstrap and Doctor perform the expensive import/backend validation. Catalog
    construction runs in every CLI process and must not import Torch, Diffusers, and
    Transformers merely to list recipes.
    """

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

    kernels_available, kernels_reason = _module_check("kernels")
    peft_available, peft_reason = _module_check("peft")
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
