from __future__ import annotations

import importlib
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


def _import_check(
    module_name: str,
    attributes: tuple[str, ...] = (),
) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(module_name)
        missing = [attribute for attribute in attributes if not hasattr(module, attribute)]
        if missing:
            return False, f"{module_name} is missing: {', '.join(missing)}"
        return True, None
    except Exception as exc:  # noqa: BLE001 - catalog must expose import-time failures
        return False, f"{module_name} import failed: {type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def wan22_runtime_support() -> Wan22RuntimeSupport:
    versions = _installed_versions()
    missing = [module for module in _CORE_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        reason = "Run `uv sync`; missing Wan 2.2 runtime packages: " + ", ".join(missing)
        return Wan22RuntimeSupport(
            core_available=False,
            core_reason=reason,
            versions=versions,
        )

    try:
        diffusers = importlib.import_module("diffusers")
        transformers = importlib.import_module("transformers")
        for symbol in (
            "AutoencoderKLWan",
            "UniPCMultistepScheduler",
            "WanPipeline",
            "WanTransformer3DModel",
        ):
            getattr(diffusers, symbol)
        for symbol in ("T5TokenizerFast", "UMT5EncoderModel"):
            getattr(transformers, symbol)
        pipeline_wan = importlib.import_module("diffusers.pipelines.wan.pipeline_wan")
        if not callable(pipeline_wan.prompt_clean):
            raise TypeError("diffusers Wan prompt_clean is not callable")
        safetensors_torch = importlib.import_module("safetensors.torch")
        for symbol in ("load_file", "save_file"):
            getattr(safetensors_torch, symbol)
    except Exception as exc:  # noqa: BLE001 - report exact stack failure in the catalog
        version_text = ", ".join(f"{name}={version}" for name, version in versions)
        reason = (
            "Wan 2.2 runtime import failed: "
            f"{type(exc).__name__}: {exc}. Installed stack: {version_text}"
        )
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
