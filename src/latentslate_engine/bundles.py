from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import scan_cache_dir, snapshot_download

from .protocol import BundleDescriptor, BundleStatus


@dataclass(frozen=True, slots=True)
class BundleDefinition:
    id: str
    name: str
    description: str
    repo_id: str
    revision: str | None = None
    ignore_patterns: tuple[str, ...] = ()

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

    def status(self) -> BundleStatus:
        try:
            cache = scan_cache_dir()
            if any(repo.repo_id == self.repo_id for repo in cache.repos):
                return BundleStatus.INSTALLED
            return BundleStatus.MISSING
        except Exception:
            return BundleStatus.UNKNOWN

    def install(self) -> str:
        return snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            ignore_patterns=list(self.ignore_patterns) or None,
        )


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
    "klein9b-basic": BundleDefinition(
        id="klein9b-basic",
        name="FLUX.2 Klein 9B",
        description=(
            "Canonical distilled FLUX.2 Klein 9B components used by the first-party "
            "text-to-image and image-to-image tools."
        ),
        repo_id="black-forest-labs/FLUX.2-klein-9B",
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
