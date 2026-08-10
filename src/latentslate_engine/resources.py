from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .model_store import MODEL_FAMILIES


class ResourceKind(StrEnum):
    MODEL = "model"
    LORA = "lora"


class ResourceFormat(StrEnum):
    DIFFUSERS = "diffusers"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    CHECKPOINT = "checkpoint"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


class ArtifactPrecision(StrEnum):
    """Stored numeric precision, never a request to transform a resource."""

    UNKNOWN = "unknown"
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"


class ArtifactQuantization(StrEnum):
    """Quantization encoded by a dropped model artifact."""

    UNKNOWN = "unknown"
    NATIVE = "native"
    INT8 = "int8"
    NVFP4 = "nvfp4"
    GGUF = "gguf"


class ResourceDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_.:/-]*$")
    kind: ResourceKind
    family: str
    name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    format: ResourceFormat
    precision: ArtifactPrecision = ArtifactPrecision.UNKNOWN
    quantization: ArtifactQuantization = ArtifactQuantization.UNKNOWN
    size_bytes: int = Field(ge=0)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    trigger_words: list[str] = Field(default_factory=list)
    default_strength: float | None = Field(default=None, allow_inf_nan=False)
    base_model: str | None = None
    component: str | None = None
    config: str | None = None
    available: bool = True
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources: list[ResourceDescriptor]
    errors: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class ResourceInventory:
    resources: list[ResourceDescriptor] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def by_id(self) -> dict[str, ResourceDescriptor]:
        return {resource.id: resource for resource in self.resources}

    def resolve(
        self,
        reference: str,
        *,
        kind: ResourceKind | None = None,
        family: str | None = None,
        include_components: bool = False,
    ) -> ResourceDescriptor:
        candidates = [
            resource
            for resource in self.resources
            if (kind is None or resource.kind == kind)
            and (family is None or resource.family == family)
            and (
                include_components
                or kind != ResourceKind.MODEL
                or resource.component is None
            )
            and reference in {resource.id, resource.relative_path, resource.name}
        ]
        if not candidates:
            raise KeyError(reference)
        if len(candidates) > 1:
            raise ValueError(f"Resource reference {reference!r} is ambiguous")
        return candidates[0]

    def path_for(self, resource_id: str) -> Path:
        try:
            return self.paths[resource_id]
        except KeyError as exc:
            raise KeyError(f"Unknown resource {resource_id!r}") from exc

    def matching(
        self,
        *,
        kind: ResourceKind,
        family: str,
        allow: list[str] | None = None,
        include_components: bool = False,
    ) -> list[ResourceDescriptor]:
        resources = [
            resource
            for resource in self.resources
            if resource.kind == kind
            and resource.family == family
            and resource.available
            and (
                include_components
                or kind != ResourceKind.MODEL
                or resource.component is None
            )
        ]
        if allow:
            resources = [
                resource
                for resource in resources
                if any(_matches_allow_pattern(resource, pattern) for pattern in allow)
            ]
        return sorted(resources, key=lambda item: (item.name.casefold(), item.id))


_MODEL_EXTENSIONS = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin"}
_LORA_EXTENSIONS = {".safetensors", ".pt", ".pth", ".bin"}
_ID_PART = re.compile(r"[^a-z0-9._/-]+")
_WEIGHT_SHARD = re.compile(r".+-\d{5}-of-\d{5}\.(?:safetensors|bin)$", re.IGNORECASE)
_WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")
_KNOWN_METADATA = {
    "id",
    "kind",
    "family",
    "name",
    "description",
    "format",
    "precision",
    "quantization",
    "tags",
    "trigger_words",
    "default_strength",
    "base_model",
    "component",
    "config",
}


def _normalize_match_value(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _matches_allow_pattern(resource: ResourceDescriptor, pattern: str) -> bool:
    normalized_pattern = _normalize_match_value(pattern)
    return any(
        fnmatchcase(_normalize_match_value(candidate), normalized_pattern)
        for candidate in (resource.id, resource.relative_path, resource.name)
    )


def _is_weight_shard(filename: str) -> bool:
    return bool(_WEIGHT_SHARD.fullmatch(filename))


def _looks_like_component_repository(file_names: set[str]) -> bool:
    if any(name.lower().endswith(_WEIGHT_INDEX_SUFFIXES) for name in file_names):
        return True
    return "config.json" in file_names and any(
        Path(name).suffix.lower() in _MODEL_EXTENSIONS for name in file_names
    )


def discover_resources(settings: Settings) -> ResourceInventory:
    inventory = ResourceInventory()
    _discover_kind(settings, inventory, ResourceKind.MODEL, settings.model_root)
    _discover_kind(settings, inventory, ResourceKind.LORA, settings.lora_root)
    inventory.resources.sort(
        key=lambda resource: (
            resource.kind.value,
            resource.family,
            resource.name.casefold(),
            resource.id,
        )
    )
    return inventory


def _discover_kind(
    settings: Settings,
    inventory: ResourceInventory,
    kind: ResourceKind,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for declared_family in MODEL_FAMILIES:
        family_root = root / declared_family
        family_root.mkdir(parents=True, exist_ok=True)
        if kind == ResourceKind.MODEL:
            _discover_models(settings, inventory, family_root, declared_family)
        else:
            _discover_loras(settings, inventory, family_root, declared_family)


def _discover_models(
    settings: Settings,
    inventory: ResourceInventory,
    family_root: Path,
    declared_family: str,
) -> None:
    for current, directories, files in os.walk(family_root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            directory
            for directory in directories
            if not directory.startswith(".")
            and directory != "__pycache__"
            and not (current_path / directory).is_symlink()
        ]
        file_names = set(files)
        metadata_path = _directory_metadata_path(current_path)
        is_component = current_path != family_root and _looks_like_component_repository(file_names)
        if current_path != family_root and (
            "model_index.json" in file_names or metadata_path is not None or is_component
        ):
            _add_resource(
                settings,
                inventory,
                ResourceKind.MODEL,
                current_path,
                declared_family,
                metadata_path,
                inferred_component="repository" if is_component else None,
            )
            directories[:] = []
            continue

        for filename in files:
            path = current_path / filename
            if (
                filename.startswith(".")
                or _is_weight_shard(filename)
                or path.suffix.lower() not in _MODEL_EXTENSIONS
            ):
                continue
            _add_resource(
                settings,
                inventory,
                ResourceKind.MODEL,
                path,
                declared_family,
                _file_metadata_path(path),
            )


def _discover_loras(
    settings: Settings,
    inventory: ResourceInventory,
    family_root: Path,
    declared_family: str,
) -> None:
    for path in family_root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if any(part.startswith(".") for part in path.relative_to(family_root).parts):
            continue
        if path.suffix.lower() not in _LORA_EXTENSIONS:
            continue
        _add_resource(
            settings,
            inventory,
            ResourceKind.LORA,
            path,
            declared_family,
            _file_metadata_path(path),
        )


def _directory_metadata_path(path: Path) -> Path | None:
    for name in (".latentslate-model.toml", ".latentslate-resource.toml"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _file_metadata_path(path: Path) -> Path | None:
    for candidate in (
        path.with_suffix(".toml"),
        path.with_name(f"{path.name}.toml"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_metadata(path: Path | None, *, root: Path) -> dict[str, Any]:
    if path is None:
        return {}
    owned = _require_within(root, path, "Resource metadata")
    data = tomllib.loads(owned.read_text(encoding="utf-8"))
    resource = data.get("resource", data)
    if not isinstance(resource, dict):
        raise TypeError("resource metadata must be a TOML table")
    return dict(resource)


def _add_resource(
    settings: Settings,
    inventory: ResourceInventory,
    kind: ResourceKind,
    path: Path,
    declared_family: str,
    metadata_path: Path | None,
    *,
    inferred_component: str | None = None,
) -> None:
    try:
        kind_root = settings.model_root if kind == ResourceKind.MODEL else settings.lora_root
        family_root = kind_root / declared_family
        owned_path = _require_within(kind_root, path, f"{kind.value.title()} resource")
        metadata = _read_metadata(metadata_path, root=kind_root)
        family = str(metadata.get("family", declared_family)).strip().lower()
        if family not in MODEL_FAMILIES:
            raise ValueError(f"unknown model family {family!r}")
        metadata_kind = metadata.get("kind")
        if metadata_kind is not None and ResourceKind(str(metadata_kind)) != kind:
            raise ValueError(
                f"metadata kind {metadata_kind!r} does not match {kind.value!r} folder"
            )

        relative = owned_path.relative_to(settings.home.resolve()).as_posix()
        resource_key = owned_path.relative_to(family_root.resolve()).as_posix()
        resource_id = str(metadata.get("id") or _resource_id(kind, family, resource_key)).strip()
        if resource_id in inventory.paths:
            raise ValueError(f"duplicate resource ID {resource_id!r}")
        descriptor = ResourceDescriptor(
            id=resource_id,
            kind=kind,
            family=family,
            name=str(metadata.get("name") or _default_name(owned_path)).strip(),
            relative_path=relative,
            format=_resource_format(owned_path, metadata.get("format")),
            precision=_artifact_precision(metadata.get("precision")),
            quantization=_artifact_quantization(
                metadata.get("quantization"),
                path=owned_path,
            ),
            size_bytes=_path_size(owned_path, root=kind_root),
            description=_optional_string(metadata.get("description")),
            tags=_string_list(metadata.get("tags")),
            trigger_words=_string_list(metadata.get("trigger_words")),
            default_strength=_optional_float(metadata.get("default_strength")),
            base_model=_optional_string(metadata.get("base_model")),
            component=_optional_string(metadata.get("component")) or inferred_component,
            config=_optional_string(metadata.get("config")),
            metadata={key: value for key, value in metadata.items() if key not in _KNOWN_METADATA},
        )
        inventory.resources.append(descriptor)
        inventory.paths[resource_id] = owned_path
    except Exception as exc:  # noqa: BLE001 - discovery should report all bad drops
        inventory.errors.append(f"{path}: {exc}")


def _require_within(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within {resolved_root}") from exc
    return resolved_path


def _resource_id(kind: ResourceKind, family: str, resource_key: str) -> str:
    normalized = resource_key.lower().replace("\\", "/")
    for suffix in (".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".bin"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    normalized = _ID_PART.sub("-", normalized).strip("-./")
    if not normalized:
        raise ValueError("resource path cannot produce an empty stable ID")
    return f"{kind.value}:{family}:{normalized}"


def _default_name(path: Path) -> str:
    name = path.name if path.is_dir() else path.stem
    return name.replace("--", "/").replace("_", " ")


def _resource_format(path: Path, configured: Any) -> ResourceFormat:
    if configured is not None:
        return ResourceFormat(str(configured).lower())
    if path.is_dir():
        if (path / "model_index.json").is_file():
            return ResourceFormat.DIFFUSERS
        return ResourceFormat.DIRECTORY
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return ResourceFormat.SAFETENSORS
    if suffix == ".gguf":
        return ResourceFormat.GGUF
    if suffix in {".ckpt", ".pt", ".pth", ".bin"}:
        return ResourceFormat.CHECKPOINT
    return ResourceFormat.UNKNOWN


def _artifact_precision(configured: Any) -> ArtifactPrecision:
    if configured is None:
        return ArtifactPrecision.UNKNOWN
    return ArtifactPrecision(str(configured).strip().lower())


def _artifact_quantization(
    configured: Any,
    *,
    path: Path,
) -> ArtifactQuantization:
    """Only infer GGUF; all other quantization claims are author supplied."""

    if configured is None:
        return (
            ArtifactQuantization.GGUF
            if path.suffix.lower() == ".gguf"
            else ArtifactQuantization.UNKNOWN
        )
    return ArtifactQuantization(str(configured).strip().lower())


def _path_size(path: Path, *, root: Path) -> int:
    if path.is_file():
        return _require_within(root, path, "Resource file").stat().st_size
    total = 0
    for child in path.rglob("*"):
        relative = child.relative_to(path)
        if ".cache" in relative.parts:
            continue
        owned = _require_within(root, child, "Resource content")
        if owned.is_file():
            total += owned.stat().st_size
    return total


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a list of strings")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string")
    value = value.strip()
    return value or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number
