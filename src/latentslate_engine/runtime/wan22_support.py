from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from functools import lru_cache

_CORE_MODULES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "av",
    "ftfy",
)
_VERSION_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
)


@dataclass(frozen=True, slots=True)
class Wan22RuntimeSupport:
    core_available: bool
    core_reason: str | None
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
def wan22_runtime_support() -> Wan22RuntimeSupport:
    """Report catalog-time package presence without importing the GPU stack.

    Bootstrap and Doctor own full import/backend validation. Keeping catalog discovery
    metadata-only prevents every CLI command and API startup from importing Torch,
    Diffusers, Transformers, and their operator registrations.
    """

    versions = _installed_versions()
    missing = [module for module in _CORE_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        reason = "Run `uv sync`; missing Wan 2.2 runtime packages: " + ", ".join(missing)
        return Wan22RuntimeSupport(
            core_available=False,
            core_reason=reason,
            versions=versions,
        )

    return Wan22RuntimeSupport(
        core_available=True,
        core_reason=None,
        versions=versions,
    )


def clear_wan22_runtime_support_cache() -> None:
    wan22_runtime_support.cache_clear()
