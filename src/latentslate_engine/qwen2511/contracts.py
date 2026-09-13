"""Native-free identities for the curated Qwen Image Edit 2511 core."""

from dataclasses import dataclass
from pathlib import Path

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def validate_sampling(steps, cfg, shift):
    """Keep this certification inside the proven non-Lightning sampling domain."""
    if (steps, cfg, shift) != (40, 4.0, 3.1):
        raise ValueError("Qwen curated sampling requires 40 steps, CFG 4, and shift 3.1")


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

    @classmethod
    def from_paths(cls, diffusion, text_encoder, vae, tokenizer):
        """Resolve every state-bearing artifact before loading the native core."""
        tokenizer = Path(tokenizer).resolve(strict=True)
        return cls(
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer,
            tuple(ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES),
        )
