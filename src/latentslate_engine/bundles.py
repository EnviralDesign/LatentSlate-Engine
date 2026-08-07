from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download

from .protocol import BundleDescriptor, BundleStatus


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

    def descriptor(self) -> BundleDescriptor:
        return BundleDescriptor(
            id=self.id,
            name=self.name,
            description=self.description,
            source="huggingface",
            repo_id=self.repo_id,
            revision=self.revision,
            status=self.status(),
            install_command=f"latentslate-engine bundles install {self.id}",
        )

    def required_repo_ids(self) -> set[str]:
        return {
            self.repo_id,
            *(repository.repo_id for repository in self.additional_repositories),
            *(file.repo_id for file in self.files),
        }

    def status(self) -> BundleStatus:
        try:
            cache = scan_cache_dir()
            cached_repo_ids = {repo.repo_id for repo in cache.repos}
            if self.required_repo_ids().issubset(cached_repo_ids):
                return BundleStatus.INSTALLED
            return BundleStatus.MISSING
        except Exception:
            return BundleStatus.UNKNOWN

    def install(self) -> str:
        primary_path = snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            allow_patterns=list(self.allow_patterns) or None,
            ignore_patterns=list(self.ignore_patterns) or None,
        )
        for repository in self.additional_repositories:
            snapshot_download(
                repo_id=repository.repo_id,
                revision=repository.revision,
                allow_patterns=list(repository.allow_patterns) or None,
                ignore_patterns=list(repository.ignore_patterns) or None,
            )
        for file in self.files:
            hf_hub_download(
                repo_id=file.repo_id,
                filename=file.filename,
                revision=file.revision,
            )
        return primary_path


BUNDLES: dict[str, BundleDefinition] = {
    "h3-basic": BundleDefinition(
        id="h3-basic",
        name="MiniMax-H3 Basic",
        description=(
            "Canonical upstream MiniMax-H3 components used by the first-party "
            "text-to-video and first/last-frame tools."
        ),
        repo_id="MiniMaxAI/MiniMax-H3",
        ignore_patterns=("transformer_ref/**",),
    ),
    "ltx23-basic": BundleDefinition(
        id="ltx23-basic",
        name="LTX 2.3 Distilled",
        description=(
            "The Diffusers-converted distilled LTX 2.3 checkpoint used by the "
            "eight-step synchronized-audio Text to Video tool."
        ),
        repo_id="diffusers/LTX-2.3-Distilled-Diffusers",
    ),
    "wan22-basic": BundleDefinition(
        id="wan22-basic",
        name="Wan 2.2 TI2V 5B",
        description=(
            "The complete official Wan 2.2 dense TI2V-5B Diffusers repository, "
            "initially used in text-only mode."
        ),
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
    "klein4b-basic": BundleDefinition(
        id="klein4b-basic",
        name="FLUX.2 Klein 4B",
        description=(
            "The complete official FLUX.2 Klein 4B Diffusers pipeline used by the "
            "consumer-GPU text-to-image and one-to-three-reference editing tools."
        ),
        repo_id="black-forest-labs/FLUX.2-klein-4B",
    ),
    "klein9b-basic": BundleDefinition(
        id="klein9b-basic",
        name="FLUX.2 Klein 9B Consumer",
        description=(
            "The canonical Klein 9B pipeline metadata and VAE, official BFL NVFP4 "
            "transformer, and official Qwen3-8B FP8 text encoder."
        ),
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        allow_patterns=(
            "model_index.json",
            "scheduler/**",
            "transformer/config.json",
            "vae/**",
            "LICENSE.md",
            "README.md",
        ),
        additional_repositories=(
            BundleRepository(repo_id="Qwen/Qwen3-8B-FP8"),
        ),
        files=(
            BundleFile(
                repo_id="black-forest-labs/FLUX.2-klein-9b-nvfp4",
                filename="flux-2-klein-9b-nvfp4.safetensors",
            ),
        ),
    ),
}


def descriptors() -> list[BundleDescriptor]:
    return [bundle.descriptor() for bundle in BUNDLES.values()]


def install(bundle_id: str) -> str:
    try:
        bundle = BUNDLES[bundle_id]
    except KeyError as exc:
        raise ValueError(f"Unknown bundle {bundle_id!r}") from exc
    return bundle.install()
