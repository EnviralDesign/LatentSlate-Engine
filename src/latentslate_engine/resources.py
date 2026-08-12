from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    FP4 = "fp4"


class ArtifactQuantization(StrEnum):
    """Quantization encoded by a dropped model artifact."""

    UNKNOWN = "unknown"
    NATIVE = "native"
    INT8 = "int8"
    NVFP4 = "nvfp4"
    GGUF = "gguf"


class ResourceSourceKind(StrEnum):
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    MANUAL = "manual"


_IMMUTABLE_HF_REVISION = re.compile(r"^[a-fA-F0-9]{40}$")


def _validate_snapshot_glob(pattern: str) -> str:
    """Accept only a relative, portable Hugging Face snapshot glob.

    Hugging Face applies these patterns inside a repository checkout. Keep the
    declaration syntax deliberately narrower than a filesystem path: POSIX
    separators only, no empty/dot/traversal segments, and no whitespace or
    control characters. Standard glob metacharacters remain valid because they
    are interpreted by the Hub, not by a local shell.
    """

    if not pattern or pattern != pattern.strip():
        raise ValueError("snapshot patterns must be non-empty relative POSIX globs")
    if "\\" in pattern or "\x00" in pattern or any(character.isspace() for character in pattern):
        raise ValueError("snapshot patterns must be safe relative POSIX globs")
    if pattern.startswith("/") or "//" in pattern:
        raise ValueError("snapshot patterns must be safe relative POSIX globs")
    if any(segment in {"", ".", ".."} for segment in pattern.split("/")):
        raise ValueError("snapshot patterns must be safe relative POSIX globs")
    return pattern


class ResourceSource(BaseModel):
    """A declarative acquisition source; credentials remain in environment variables."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: ResourceSourceKind
    repo_id: str | None = None
    revision: str | None = None
    filename: str | None = None
    url: str | None = None
    model_version_id: int | None = Field(default=None, ge=1)
    file_id: int | None = Field(default=None, ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    requires_auth: bool = False
    label: str | None = None

    @model_validator(mode="after")
    def validate_locator(self) -> ResourceSource:
        for pattern in (*self.allow_patterns, *self.ignore_patterns):
            _validate_snapshot_glob(pattern)

        if self.url:
            parsed = urlsplit(self.url)
            if parsed.scheme.lower() != "https" or not parsed.hostname:
                raise ValueError("resource source URLs must use HTTPS")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("resource source URLs cannot contain userinfo")
            if parsed.query:
                raise ValueError(
                    "resource source URLs cannot contain secret-bearing query parameters "
                    "or other query strings"
                )
            if parsed.fragment:
                raise ValueError("resource source URLs cannot contain fragments")

        if self.type == ResourceSourceKind.HUGGINGFACE:
            if not self.repo_id:
                raise ValueError("Hugging Face sources require repo_id")
            if self.url is not None:
                raise ValueError("Hugging Face sources must use repo_id instead of url")
            if self.model_version_id is not None or self.file_id is not None:
                raise ValueError("Hugging Face sources cannot declare Civitai identifiers")
            if (self.allow_patterns or self.ignore_patterns) and not (
                self.revision and _IMMUTABLE_HF_REVISION.fullmatch(self.revision)
            ):
                raise ValueError("filtered Hugging Face snapshots require an immutable revision")
        elif self.type == ResourceSourceKind.CIVITAI:
            if self.allow_patterns or self.ignore_patterns:
                raise ValueError("snapshot patterns are only supported for Hugging Face directory snapshots")
            if not (self.url or self.model_version_id):
                raise ValueError("Civitai sources require url or model_version_id")
            if self.url and (self.model_version_id is not None or self.file_id is not None):
                raise ValueError(
                    "Civitai sources must use either an exact URL/hash or model_version_id/file_id, not both"
                )
            if self.url and (self.requires_auth or self.token_env):
                parsed = urlsplit(self.url)
                if parsed.hostname != "civitai.com" or parsed.port not in {None, 443}:
                    raise ValueError(
                        "authenticated Civitai exact URL sources must start at the trusted civitai.com origin"
                    )
            if any(
                value is not None
                for value in (self.repo_id, self.revision, self.filename)
            ):
                raise ValueError("Civitai sources cannot declare Hugging Face identifiers")
        elif self.type == ResourceSourceKind.MANUAL and any(
            value is not None
            for value in (
                self.repo_id,
                self.revision,
                self.filename,
                self.url,
                self.model_version_id,
                self.file_id,
                self.token_env,
            )
        ):
            raise ValueError("manual sources cannot declare a network locator or secret")
        elif self.type == ResourceSourceKind.MANUAL and (self.allow_patterns or self.ignore_patterns):
            raise ValueError("snapshot patterns are only supported for Hugging Face directory snapshots")
        return self

    def required_secret(self) -> str | None:
        if self.token_env:
            return self.token_env
        if self.requires_auth and self.type == ResourceSourceKind.HUGGINGFACE:
            return "HF_TOKEN"
        if self.requires_auth and self.type == ResourceSourceKind.CIVITAI:
            return "CIVITAI_TOKEN"
        return None

    def is_exact(self) -> bool:
        """Return whether this source identifies immutable acquisition bytes."""

        if self.type == ResourceSourceKind.HUGGINGFACE:
            pinned_revision = bool(
                self.revision and _IMMUTABLE_HF_REVISION.fullmatch(self.revision)
            )
            pinned_file_hash = bool(self.filename and self.sha256)
            return bool(self.repo_id and (pinned_revision or pinned_file_hash))
        if self.type == ResourceSourceKind.CIVITAI:
            pinned_ids = bool(self.model_version_id and self.file_id)
            pinned_url_hash = bool(self.url and self.sha256)
            return pinned_ids or pinned_url_hash
        return False


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
    sources: list[ResourceSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_shapes(self) -> ResourceDescriptor:
        """Require lockable sources to describe the artifact they acquire.

        A revision identifies a repository snapshot, while a filename/file ID
        identifies one downloaded file.  Treating those interchangeably makes a
        deployment lock claim guarantees that the installer cannot actually
        enforce (and, for directories, would make a source file hash unused).
        """

        is_directory = (
            self.kind != ResourceKind.LORA
            and self.format in {ResourceFormat.DIFFUSERS, ResourceFormat.DIRECTORY}
        )
        for source in self.sources:
            if (source.allow_patterns or source.ignore_patterns) and (
                not is_directory or source.type != ResourceSourceKind.HUGGINGFACE
            ):
                raise ValueError(
                    "snapshot patterns are only supported for Hugging Face directory snapshots"
                )
            if is_directory:
                if source.sha256 is not None:
                    raise ValueError(
                        "directory resources cannot declare a single-file sha256; "
                        "use an exact snapshot source instead"
                    )
                if source.filename is not None:
                    raise ValueError(
                        "directory resources cannot declare a single-file filename; "
                        "use an exact snapshot source instead"
                    )
                if source.type == ResourceSourceKind.CIVITAI and source.is_exact():
                    raise ValueError(
                        "directory resources require an exact snapshot source; "
                        "Civitai file selectors cannot lock a directory"
                    )
            elif source.type == ResourceSourceKind.HUGGINGFACE:
                if not source.filename:
                    raise ValueError(
                        "file resources require a file selector; "
                        "Hugging Face sources must declare filename"
                    )
        return self


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

    def is_installed(self, resource_id: str) -> bool:
        try:
            resource = self.by_id()[resource_id]
            path = self.paths[resource_id]
        except KeyError:
            return False
        return _artifact_complete(path, resource)

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
_WEIGHT_SHARD_PARTS = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.(?P<ext>safetensors|bin)$",
    re.IGNORECASE,
)
_WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")
_WAN22_PIPELINE_SUPPORT_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/spiece.model",
    "transformer/config.json",
    "transformer_2/config.json",
    "text_encoder/config.json",
    "vae/config.json",
)
_KLEIN_PIPELINE_SUPPORT_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "vae/config.json",
)
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
    "source",
    "sources",
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


def _contains_model_weights(path: Path) -> bool:
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            directory
            for directory in directories
            if not (current_path / directory).is_symlink()
        ]
        for filename in files:
            candidate = current_path / filename
            if Path(filename.lower()).suffix not in _MODEL_EXTENSIONS:
                continue
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _indexed_weight_shards_complete(path: Path) -> bool:
    root = path.resolve()
    indexes = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.name.lower().endswith(_WEIGHT_INDEX_SUFFIXES)
    ]
    for index in indexes:
        try:
            if index.stat().st_size <= 0:
                return False
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        raw_shard_names = list(weight_map.values())
        if not raw_shard_names or not all(
            isinstance(name, str) and name.strip() for name in raw_shard_names
        ):
            return False
        shard_names = set(raw_shard_names)

        series: dict[tuple[Path, str, int, str], set[int]] = {}
        for name in shard_names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            shard = (index.parent / relative).resolve()
            try:
                shard.relative_to(root)
            except ValueError:
                return False
            try:
                if not shard.is_file() or shard.stat().st_size <= 0:
                    return False
            except OSError:
                return False

            match = _WEIGHT_SHARD_PARTS.fullmatch(shard.name)
            if match is None:
                continue
            total = int(match.group("total"))
            part = int(match.group("part"))
            if total <= 0 or part <= 0 or part > total:
                return False
            key = (
                shard.parent,
                match.group("prefix"),
                total,
                match.group("ext"),
            )
            series.setdefault(key, set()).add(part)

        for (directory, prefix, total, extension), referenced_parts in series.items():
            expected_parts = set(range(1, total + 1))
            if referenced_parts != expected_parts:
                return False
            for part in expected_parts:
                shard = directory / f"{prefix}-{part:05d}-of-{total:05d}.{extension}"
                try:
                    if not shard.is_file() or shard.stat().st_size <= 0:
                        return False
                except OSError:
                    return False
    return True


def _numbered_weight_shards_complete(path: Path) -> bool:
    """Validate every numbered shard series present on disk, indexed or not."""

    series: dict[tuple[Path, str, int, str], set[int]] = {}
    for current, directories, files in os.walk(path, followlinks=False):
        directory = Path(current)
        directories[:] = [
            child for child in directories if not (directory / child).is_symlink()
        ]
        for filename in files:
            match = _WEIGHT_SHARD_PARTS.fullmatch(filename)
            if match is None:
                continue
            total = int(match.group("total"))
            part = int(match.group("part"))
            if total <= 0 or part <= 0 or part > total:
                return False
            key = (directory, match.group("prefix"), total, match.group("ext"))
            series.setdefault(key, set()).add(part)

    for (directory, prefix, total, extension), present_parts in series.items():
        expected_parts = set(range(1, total + 1))
        if present_parts != expected_parts:
            return False
        for part in expected_parts:
            shard = directory / f"{prefix}-{part:05d}-of-{total:05d}.{extension}"
            try:
                if not shard.is_file() or shard.stat().st_size <= 0:
                    return False
            except OSError:
                return False
    return True


def _has_wan22_pipeline_support_files(path: Path) -> bool:
    return all((path / relative).is_file() for relative in _WAN22_PIPELINE_SUPPORT_FILES)


def _has_pipeline_support_files(path: Path, family: str) -> bool:
    required = (
        _KLEIN_PIPELINE_SUPPORT_FILES if family == "klein4b" else _WAN22_PIPELINE_SUPPORT_FILES
    )
    return all((path / relative).is_file() for relative in required)


def _looks_like_wan22_pipeline_support(path: Path) -> bool:
    # Only infer config-only support trees. A weight-bearing Diffusers directory is
    # a normal model unless the user explicitly marks it as pipeline_support.
    return _has_wan22_pipeline_support_files(path) and not _contains_model_weights(path)


def _artifact_complete(path: Path, resource: ResourceDescriptor) -> bool:
    if resource.kind == ResourceKind.LORA or path.is_file():
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        return _artifact_size_matches(path, resource) and _artifact_hash_matches(
            path, resource
        )

    if not path.is_dir():
        return False
    if resource.component == "pipeline_support":
        structurally_complete = _has_pipeline_support_files(path, resource.family)
    elif resource.format == ResourceFormat.DIFFUSERS:
        structurally_complete = (
            (path / "model_index.json").is_file()
            and _contains_model_weights(path)
            and _indexed_weight_shards_complete(path)
            and _numbered_weight_shards_complete(path)
        )
    else:
        structurally_complete = (
            _contains_model_weights(path)
            and _indexed_weight_shards_complete(path)
            and _numbered_weight_shards_complete(path)
        )
    return structurally_complete and _artifact_size_matches(path, resource)


def _artifact_size_matches(path: Path, resource: ResourceDescriptor) -> bool:
    if resource.size_bytes <= 0:
        return False
    try:
        root = path if path.is_dir() else path.parent
        return _path_size(path, root=root) == resource.size_bytes
    except (OSError, ValueError):
        return False


def _artifact_hash_matches(path: Path, resource: ResourceDescriptor) -> bool:
    expected = {
        source.sha256.casefold()
        for source in resource.sources
        if source.sha256 is not None
    }
    if not expected:
        return True
    if not path.is_file() or len(expected) != 1:
        return False
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest().casefold() in expected


def _with_artifact_availability(
    resource: ResourceDescriptor,
    path: Path,
) -> ResourceDescriptor:
    if _artifact_complete(path, resource):
        return resource.model_copy(update={"available": True, "unavailable_reason": None})
    return resource.model_copy(
        update={
            "available": False,
            "unavailable_reason": "resource artifact is not installed or incomplete",
        }
    )


def discover_resources(settings: Settings) -> ResourceInventory:
    inventory = ResourceInventory()
    _discover_kind(settings, inventory, ResourceKind.MODEL, settings.model_root)
    _discover_kind(settings, inventory, ResourceKind.LORA, settings.lora_root)
    _discover_declarations(settings, inventory)
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
        is_pipeline_support = (
            declared_family == "wan22"
            and current_path != family_root
            and _looks_like_wan22_pipeline_support(current_path)
        )
        if current_path != family_root and (
            "model_index.json" in file_names
            or metadata_path is not None
            or is_component
            or is_pipeline_support
        ):
            inferred_component = (
                "pipeline_support"
                if is_pipeline_support
                else ("repository" if is_component else None)
            )
            _add_resource(
                settings,
                inventory,
                ResourceKind.MODEL,
                current_path,
                declared_family,
                metadata_path,
                inferred_component=inferred_component,
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


def _discover_declarations(settings: Settings, inventory: ResourceInventory) -> None:
    for label, root in settings.resource_declaration_roots():
        if label == "local":
            root.mkdir(parents=True, exist_ok=True)
        if root.is_dir():
            _discover_declarations_from_root(settings, inventory, root)


def _discover_declarations_from_root(
    settings: Settings,
    inventory: ResourceInventory,
    root: Path,
) -> None:
    for path in sorted(root.rglob("*.toml")):
        if path.name.startswith("."):
            continue
        relative_declaration = path.relative_to(root)
        if any(part.startswith(".") for part in relative_declaration.parts):
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            raw = data.get("resource", data)
            if not isinstance(raw, dict):
                raise TypeError("resource declaration must be a TOML table")
            if "available" in raw or "unavailable_reason" in raw:
                raise ValueError("resource availability is derived from artifact completeness")
            descriptor = ResourceDescriptor.model_validate(raw)
            if descriptor.family not in MODEL_FAMILIES:
                raise ValueError(f"unknown model family {descriptor.family!r}")
            target = _declared_resource_path(settings, descriptor)
            descriptor = _with_artifact_availability(descriptor, target)
            _merge_declared_resource(inventory, descriptor, target)
        except Exception as exc:  # noqa: BLE001 - report all authoring errors
            inventory.errors.append(f"{path}: {exc}")


def _declared_resource_path(settings: Settings, resource: ResourceDescriptor) -> Path:
    relative = Path(resource.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("resource relative_path must stay within the Engine data root")
    target = (settings.home.resolve() / relative).resolve()
    kind_root = settings.model_root if resource.kind == ResourceKind.MODEL else settings.lora_root
    family_root = kind_root / resource.family
    _require_within(family_root, target, "Declared resource")
    return target


def _merge_declared_resource(
    inventory: ResourceInventory,
    resource: ResourceDescriptor,
    path: Path,
) -> None:
    by_id = inventory.by_id()
    existing = by_id.get(resource.id)
    if existing is None:
        duplicate_path = next(
            (
                existing_id
                for existing_id, existing_path in inventory.paths.items()
                if existing_id != resource.id and existing_path.resolve() == path.resolve()
            ),
            None,
        )
        if duplicate_path is not None:
            raise ValueError(
                f"declared resource path is already discovered as {duplicate_path!r}; "
                "use the same resource id"
            )
        inventory.resources.append(resource)
        inventory.paths[resource.id] = path
        return

    if (
        existing.kind != resource.kind
        or existing.family != resource.family
        or existing.relative_path != resource.relative_path
    ):
        raise ValueError(f"resource declaration conflicts with discovered resource {resource.id!r}")
    index = next(
        index for index, candidate in enumerate(inventory.resources) if candidate.id == resource.id
    )
    inventory.resources[index] = resource
    inventory.paths[resource.id] = path


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
        declared_component = _optional_string(metadata.get("component"))
        component = declared_component or inferred_component
        if component == "pipeline_support":
            if not owned_path.is_dir():
                raise ValueError("pipeline_support must be a directory resource")
            if not _has_pipeline_support_files(owned_path, family):
                raise ValueError(
                    "pipeline_support is missing required scheduler/tokenizer/component configs"
                )
            if declared_component is None and _contains_model_weights(owned_path):
                raise ValueError(
                    "weight-bearing pipeline support must be explicitly tagged as "
                    "component='pipeline_support'"
                )
        resource_format = (
            ResourceFormat.DIRECTORY
            if component == "pipeline_support"
            else _resource_format(owned_path, metadata.get("format"))
        )
        descriptor = ResourceDescriptor(
            id=resource_id,
            kind=kind,
            family=family,
            name=str(metadata.get("name") or _default_name(owned_path)).strip(),
            relative_path=relative,
            format=resource_format,
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
            component=component,
            config=_optional_string(metadata.get("config")),
            metadata={key: value for key, value in metadata.items() if key not in _KNOWN_METADATA},
            sources=_resource_sources(metadata),
        )
        inventory.resources.append(descriptor)
        inventory.paths[resource_id] = owned_path
    except Exception as exc:  # noqa: BLE001 - discovery should report all bad drops
        inventory.errors.append(f"{path}: {exc}")


def _resource_sources(metadata: dict[str, Any]) -> list[ResourceSource]:
    raw = metadata.get("sources")
    if raw is None and "source" in metadata:
        raw = metadata["source"]
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(value, dict) for value in values):
        raise TypeError("resource source metadata must be a table or array of tables")
    return [ResourceSource.model_validate(value) for value in values]


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
        # The bundle installer writes this Engine-owned sidecar after acquiring
        # the upstream snapshot.  Its newline representation differs between
        # Windows and Unix, so it cannot be part of a portable source-artifact
        # byte contract.  Hugging Face cache files are similarly local-only.
        if ".cache" in relative.parts or relative == Path(".latentslate-model.toml"):
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
