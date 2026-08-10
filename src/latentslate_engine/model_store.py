from __future__ import annotations

import os
import re
from pathlib import Path

ENGINE_HOME_ENV = "LATENTSLATE_ENGINE_HOME"
DEFAULT_DATA_DIRECTORY = "LatentSlateEngineData"
MODEL_FAMILIES = ("h3", "ltx23", "wan22", "klein4b", "klein9b", "custom")
_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9._-]+")


class ModelNotInstalledError(RuntimeError):
    """Raised when a runtime requests a repository absent from the model store."""


def repository_root() -> Path:
    """Return the source checkout root that owns the default model directory."""

    starts = (Path.cwd().resolve(), Path(__file__).resolve().parent)
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "src" / "latentslate_engine"
            ).is_dir():
                return candidate
    # A wheel has no source checkout to own the store. Keep the fallback visible
    # and local to the directory from which the Engine was launched.
    return Path.cwd().resolve()


def configured_engine_home() -> Path:
    configured = os.getenv(ENGINE_HOME_ENV)
    if configured and configured.strip():
        path = Path(configured.strip()).expanduser()
        if not path.is_absolute():
            path = repository_root() / path
        return path.resolve()
    return (repository_root() / DEFAULT_DATA_DIRECTORY).resolve()


def configured_model_root() -> Path:
    return configured_engine_home() / "models"


def engine_data_directories(engine_home: Path) -> tuple[Path, ...]:
    model_root = engine_home / "models"
    lora_root = engine_home / "loras"
    cache_root = engine_home / "cache"
    return (
        engine_home,
        *(model_root / family for family in MODEL_FAMILIES),
        *(lora_root / family for family in MODEL_FAMILIES),
        cache_root / "huggingface" / "hub",
        cache_root / "huggingface" / "assets",
        cache_root / "huggingface" / "xet",
        cache_root / "torch",
        engine_home / "assets",
        engine_home / "jobs",
        engine_home / "logs",
        engine_home / "temp",
    )


def initialize_engine_data(engine_home: Path | None = None) -> Path:
    engine_home = engine_home or configured_engine_home()
    for directory in engine_data_directories(engine_home):
        directory.mkdir(parents=True, exist_ok=True)
    return engine_home


def configure_library_cache_environment() -> Path:
    """Force third-party model caches beneath LatentSlate's single data root."""

    cache_root = configured_engine_home() / "cache"
    huggingface_root = cache_root / "huggingface"
    forced_paths = {
        "HF_HOME": huggingface_root,
        "HF_HUB_CACHE": huggingface_root / "hub",
        "HF_ASSETS_CACHE": huggingface_root / "assets",
        "HUGGINGFACE_HUB_CACHE": huggingface_root / "hub",
        "HUGGINGFACE_ASSETS_CACHE": huggingface_root / "assets",
        "HF_XET_CACHE": huggingface_root / "xet",
        "DIFFUSERS_CACHE": huggingface_root / "hub",
        "TRANSFORMERS_CACHE": huggingface_root / "hub",
        "TORCH_HOME": cache_root / "torch",
    }
    for name, path in forced_paths.items():
        os.environ[name] = str(path)
    return cache_root


def family_name(bundle_id: str) -> str:
    name = bundle_id.removesuffix("-basic")
    sanitized = _SAFE_PATH_PART.sub("-", name).strip(".-")
    if not sanitized:
        raise ValueError(f"Bundle ID {bundle_id!r} cannot produce a model family path")
    return sanitized


def repository_directory_name(repo_id: str) -> str:
    parts = [_SAFE_PATH_PART.sub("-", part).strip(".-") for part in repo_id.strip().split("/")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid model repository ID {repo_id!r}")
    return "--".join(parts)


def family_directory(model_root: Path, bundle_id: str) -> Path:
    return model_root / family_name(bundle_id)


def repository_directory(model_root: Path, bundle_id: str, repo_id: str) -> Path:
    return family_directory(model_root, bundle_id) / repository_directory_name(repo_id)


def _require_owned_path(model_root: Path, path: Path, label: str) -> Path:
    resolved_root = model_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within model root {resolved_root}") from exc
    return resolved_path


def owned_repository_directory(model_root: Path, bundle_id: str, repo_id: str) -> Path:
    """Resolve a repository directory while enforcing model-root containment."""

    path = repository_directory(model_root, bundle_id, repo_id)
    return _require_owned_path(model_root, path, f"Model repository {repo_id!r}")


def installed_manifest_path(model_root: Path, bundle_id: str) -> Path:
    family = _require_owned_path(
        model_root,
        family_directory(model_root, bundle_id),
        f"Model family {family_name(bundle_id)!r}",
    )
    return family / ".latentslate-installed.json"


def require_repository(model_root: Path, bundle_id: str, repo_id: str) -> Path:
    path = owned_repository_directory(model_root, bundle_id, repo_id)
    if path.is_dir():
        return path
    raise ModelNotInstalledError(
        f"Model repository {repo_id!r} is not installed at {path}. "
        f"Run `latentslate-engine bundles install {bundle_id}`."
    )


def owned_model_file_path(repository: Path, filename: str) -> Path:
    """Resolve a model filename while enforcing repository containment."""

    resolved_repository = repository.resolve()
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Model filename must stay within its repository: {filename!r}")
    path = (resolved_repository / relative_path).resolve()
    try:
        path.relative_to(resolved_repository)
    except ValueError as exc:
        raise ValueError(f"Model filename must stay within its repository: {filename!r}") from exc
    return path


def require_model_file(
    model_root: Path,
    bundle_id: str,
    repo_id: str,
    filename: str,
) -> Path:
    repository = require_repository(model_root, bundle_id, repo_id)
    path = owned_model_file_path(repository, filename)
    if path.is_file():
        return path
    raise ModelNotInstalledError(
        f"Model file {filename!r} from {repo_id!r} is not installed at {path}. "
        f"Run `latentslate-engine bundles install {bundle_id}`."
    )
