from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from .config import Settings
from .model_store import (
    configured_model_root,
    installed_manifest_path,
    owned_model_file_path,
    owned_repository_directory,
)
from .protocol import BundleDescriptor, BundleStatus


def snapshot_download(**kwargs):
    """Load the optional network client only when a download actually starts."""

    from huggingface_hub import snapshot_download as download

    return download(**kwargs)


def hf_hub_download(**kwargs):
    """Load the optional network client only when a file download actually starts."""

    from huggingface_hub import hf_hub_download as download

    return download(**kwargs)


@dataclass(frozen=True, slots=True)
class BundleFile:
    repo_id: str
    filename: str
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class BundleRepository:
    repo_id: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BundleDefinition:
    id: str
    name: str
    description: str
    repo_id: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    additional_repositories: tuple[BundleRepository, ...] = ()
    files: tuple[BundleFile, ...] = ()
    artifact_precision: str | None = None
    artifact_quantization: str | None = None

    def descriptor(self, model_root: Path | None = None) -> BundleDescriptor:
        return BundleDescriptor(
            id=self.id,
            name=self.name,
            description=self.description,
            source="huggingface",
            repo_id=self.repo_id,
            revision=self.revision,
            status=self.status(model_root),
            install_command=f"latentslate-engine bundles install {self.id}",
        )

    def required_repo_ids(self) -> set[str]:
        return {
            self.repo_id,
            *(repository.repo_id for repository in self.additional_repositories),
            *(file.repo_id for file in self.files),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "bundle_id": self.id,
            "repositories": [
                {
                    "repo_id": self.repo_id,
                    "revision": self.revision,
                    "allow_patterns": list(self.allow_patterns),
                    "ignore_patterns": list(self.ignore_patterns),
                },
                *(
                    {
                        "repo_id": repository.repo_id,
                        "revision": repository.revision,
                        "allow_patterns": list(repository.allow_patterns),
                        "ignore_patterns": list(repository.ignore_patterns),
                    }
                    for repository in self.additional_repositories
                ),
            ],
            "files": [
                {
                    "repo_id": file.repo_id,
                    "filename": file.filename,
                    "revision": file.revision,
                }
                for file in self.files
            ],
            "artifact": (
                {
                    "precision": self.artifact_precision,
                    "quantization": self.artifact_quantization,
                }
                if self.artifact_precision and self.artifact_quantization
                else None
            ),
        }

    def status(self, model_root: Path | None = None) -> BundleStatus:
        model_root = model_root or configured_model_root()
        try:
            manifest_path = installed_manifest_path(model_root, self.id)
            if not manifest_path.is_file():
                return BundleStatus.MISSING
            installed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(installed, dict):
                return BundleStatus.MISSING
            inventory = installed.pop("inventory", None)
            expected = self.manifest()
            legacy_expected = dict(expected)
            legacy_expected.pop("artifact", None)
            legacy_manifest = installed == legacy_expected
            if installed != expected and not legacy_manifest:
                return BundleStatus.MISSING
            if not isinstance(inventory, list) or not inventory:
                return BundleStatus.MISSING

            resolved_root = model_root.resolve()
            inventoried_paths: list[Path] = []
            for item in inventory:
                if not isinstance(item, dict):
                    return BundleStatus.MISSING
                path = (model_root / item["path"]).resolve()
                path.relative_to(resolved_root)
                if not path.is_file() or path.stat().st_size != item["size"]:
                    return BundleStatus.MISSING
                inventoried_paths.append(path)

            for repo_id in self.required_repo_ids():
                repository = owned_repository_directory(model_root, self.id, repo_id)
                if not repository.is_dir() or not any(
                    path.is_relative_to(repository) for path in inventoried_paths
                ):
                    return BundleStatus.MISSING
            if (
                legacy_manifest
                and self.artifact_precision
                and self.artifact_quantization
                and not _artifact_sidecar_matches(
                    owned_repository_directory(model_root, self.id, self.repo_id),
                    precision=self.artifact_precision,
                    quantization=self.artifact_quantization,
                )
            ):
                return BundleStatus.MISSING
            if any(
                not owned_model_file_path(
                    owned_repository_directory(model_root, self.id, file.repo_id),
                    file.filename,
                ).is_file()
                for file in self.files
            ):
                return BundleStatus.MISSING
            return BundleStatus.INSTALLED
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return BundleStatus.UNKNOWN

    def install(self, model_root: Path | None = None) -> str:
        model_root = model_root or configured_model_root()
        model_root.mkdir(parents=True, exist_ok=True)
        primary_dir = owned_repository_directory(model_root, self.id, self.repo_id)
        primary_dir.mkdir(parents=True, exist_ok=True)
        primary_path = snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            allow_patterns=list(self.allow_patterns) or None,
            ignore_patterns=list(self.ignore_patterns) or None,
            local_dir=primary_dir,
        )
        for repository in self.additional_repositories:
            local_dir = owned_repository_directory(
                model_root,
                self.id,
                repository.repo_id,
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repository.repo_id,
                revision=repository.revision,
                allow_patterns=list(repository.allow_patterns) or None,
                ignore_patterns=list(repository.ignore_patterns) or None,
                local_dir=local_dir,
            )
        for file in self.files:
            local_dir = owned_repository_directory(model_root, self.id, file.repo_id)
            local_dir.mkdir(parents=True, exist_ok=True)
            owned_model_file_path(local_dir, file.filename)
            hf_hub_download(
                repo_id=file.repo_id,
                filename=file.filename,
                revision=file.revision,
                local_dir=local_dir,
            )
        manifest_path = installed_manifest_path(model_root, self.id)
        if self.artifact_precision and self.artifact_quantization:
            _write_artifact_sidecar(
                primary_dir,
                precision=self.artifact_precision,
                quantization=self.artifact_quantization,
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pending_manifest = manifest_path.with_suffix(".tmp")
        manifest = self.manifest()
        manifest["inventory"] = _download_inventory(model_root, self)
        pending_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending_manifest.replace(manifest_path)
        return primary_path


def _write_artifact_sidecar(
    directory: Path,
    *,
    precision: str,
    quantization: str,
) -> None:
    """Write Engine-owned metadata for a canonical, complete bundle download."""

    (directory / ".latentslate-model.toml").write_text(
        "# Generated by LatentSlate Engine for this complete canonical bundle.\n"
        'format = "diffusers"\n'
        f'precision = "{precision}"\n'
        f'quantization = "{quantization}"\n',
        encoding="utf-8",
    )


def _artifact_sidecar_matches(
    directory: Path,
    *,
    precision: str,
    quantization: str,
) -> bool:
    sidecar = directory / ".latentslate-model.toml"
    if not sidecar.is_file():
        return False
    try:
        metadata = tomllib.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return metadata.get("precision") == precision and metadata.get("quantization") == quantization


BUNDLES: dict[str, BundleDefinition] = {
    "h3-basic": BundleDefinition(
        id="h3-basic",
        name="MiniMax-H3 Basic",
        description=(
            "Canonical upstream MiniMax-H3 components used by the first-party "
            "text-to-video and first/last-frame tools."
        ),
        repo_id="MiniMaxAI/MiniMax-H3",
        revision="9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6",
        ignore_patterns=("transformer_ref/**",),
        artifact_precision="bf16",
        artifact_quantization="native",
    ),
    "ltx23-basic": BundleDefinition(
        id="ltx23-basic",
        name="LTX 2.3 Distilled",
        description=(
            "The Diffusers-converted distilled LTX 2.3 checkpoint used by the "
            "eight-step synchronized-audio Text to Video tool."
        ),
        repo_id="diffusers/LTX-2.3-Distilled-Diffusers",
        revision="432e0d3c2d1769aaa4d295f9243f7062bf6b47ee",
        artifact_precision="bf16",
        artifact_quantization="native",
    ),
    "wan22-basic": BundleDefinition(
        id="wan22-basic",
        name="Wan 2.2 TI2V 5B",
        description=(
            "The complete official Wan 2.2 dense TI2V-5B Diffusers repository, "
            "initially used in text-only mode."
        ),
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        # A recipe lock must describe the same bytes as the compatibility bundle.
        # This is the resolved snapshot currently installed by the canonical
        # bundle, rather than the repository's moving default branch.
        revision="b8fff7315c768468a5333511427288870b2e9635",
        artifact_precision="bf16",
        artifact_quantization="native",
    ),
    "klein4b-basic": BundleDefinition(
        id="klein4b-basic",
        name="FLUX.2 Klein 4B",
        description=(
            "The complete official FLUX.2 Klein 4B Diffusers pipeline used by the "
            "consumer-GPU text-to-image and one-to-three-reference editing tools."
        ),
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        artifact_precision="bf16",
        artifact_quantization="native",
    ),
    "klein9b-basic": BundleDefinition(
        id="klein9b-basic",
        name="FLUX.2 Klein 9B",
        description=(
            "The complete official FLUX.2 Klein 9B Diffusers pipeline. LatentSlate "
            "does not assemble or convert component artifacts at runtime."
        ),
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        artifact_precision="bf16",
        artifact_quantization="native",
    ),
}


def configured_bundles(settings: Settings | None = None) -> dict[str, BundleDefinition]:
    """Return bundle definitions aligned with the active runtime model IDs."""

    settings = settings or Settings.from_env()
    configured = dict(BUNDLES)
    configured["h3-basic"] = _with_configured_repo(BUNDLES["h3-basic"], settings.h3_model_id)
    configured["ltx23-basic"] = _with_configured_repo(
        BUNDLES["ltx23-basic"], settings.ltx23_model_id
    )
    configured["wan22-basic"] = _with_configured_repo(
        BUNDLES["wan22-basic"], settings.wan22_model_id
    )
    configured["klein4b-basic"] = _with_configured_repo(
        BUNDLES["klein4b-basic"], settings.klein4b_model_id
    )
    configured["klein9b-basic"] = _with_configured_repo(
        BUNDLES["klein9b-basic"], settings.klein_model_id
    )
    return configured


def _with_configured_repo(bundle: BundleDefinition, repo_id: str) -> BundleDefinition:
    """Never transfer canonical artifact claims to an arbitrary repository override."""

    if repo_id == bundle.repo_id:
        return bundle
    return replace(
        bundle,
        repo_id=repo_id,
        revision=None,
        artifact_precision=None,
        artifact_quantization=None,
    )


def descriptors(
    model_root: Path | None = None,
    settings: Settings | None = None,
) -> list[BundleDescriptor]:
    model_root = model_root or configured_model_root()
    return [bundle.descriptor(model_root) for bundle in configured_bundles(settings).values()]


def install(
    bundle_id: str,
    model_root: Path | None = None,
    settings: Settings | None = None,
) -> str:
    try:
        bundle = configured_bundles(settings)[bundle_id]
    except KeyError as exc:
        raise ValueError(f"Unknown bundle {bundle_id!r}") from exc
    return bundle.install(model_root)


def _download_inventory(
    model_root: Path,
    bundle: BundleDefinition,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for repo_id in sorted(bundle.required_repo_ids()):
        repository = owned_repository_directory(model_root, bundle.id, repo_id)
        for path in sorted(repository.rglob("*")):
            relative_to_repository = path.relative_to(repository)
            if path.is_file() and ".cache" not in relative_to_repository.parts:
                inventory.append(
                    {
                        "path": path.relative_to(model_root).as_posix(),
                        "size": path.stat().st_size,
                    }
                )
    if not inventory:
        raise RuntimeError(f"Bundle {bundle.id!r} downloaded no model files")
    return inventory
