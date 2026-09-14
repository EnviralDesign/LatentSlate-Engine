"""Native-free identities for the curated Qwen Image Edit 2511 core."""

from dataclasses import dataclass
from pathlib import Path

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def validate_sampling(steps, cfg, shift):
    """Keep execution inside the curated base and Lightning sampling domain."""
    if (steps, cfg, shift) not in ((40, 4.0, 3.1), (4, 1.0, 3.1)):
        raise ValueError("Qwen requires 40 steps/CFG 4 or 4 steps/CFG 1, with shift 3.1")


def validate_adapters(adapters):
    """Bound this certification to an empty list or one strength-one adapter."""
    if len(adapters) > 1 or any(
        isinstance(strength, bool) or strength != 1.0 for _, strength in adapters
    ):
        raise ValueError("Qwen currently supports one transformer adapter at strength 1")


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path):
        """Snapshot one concrete local model or tokenizer file."""
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Qwen artifact must be a file")
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class Qwen2511Identity:
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]
    adapters: tuple[tuple[ArtifactIdentity, float], ...] = ()

    @classmethod
    def from_paths(cls, diffusion, text_encoder, vae, tokenizer, adapters=()):
        """Resolve every state-bearing artifact before loading the native core."""
        tokenizer = Path(tokenizer).resolve(strict=True)
        adapters = tuple(adapters)
        validate_adapters(adapters)
        return cls(
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer,
            tuple(ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES),
            tuple((ArtifactIdentity.from_path(path), float(strength)) for path, strength in adapters),
        )
