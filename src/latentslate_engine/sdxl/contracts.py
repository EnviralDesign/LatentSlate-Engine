"""Native-free identity and request bounds for SDXL T2I."""

from dataclasses import dataclass
from pathlib import Path

from latentslate_engine.validation import validate_u64

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")
ALIGNMENT = 8
MIN_SIDE = 256
MAX_SIDE = 8 * 1024
MAX_PIXELS = 4 * 1024 * 1024
MAX_ASPECT = MAX_SIDE / MIN_SIDE


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path: Path):
        """Capture the concrete local file used by a native session."""
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("SDXL artifact must be a file")
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class SDXLIdentity:
    checkpoint: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]
    vae: ArtifactIdentity | None = None

    @classmethod
    def from_paths(cls, checkpoint, tokenizer, vae=None):
        """Resolve embedded components and an optional explicit decoder override."""
        tokenizer = Path(tokenizer).resolve(strict=True)
        return cls(
            ArtifactIdentity.from_path(checkpoint),
            tokenizer,
            tuple(
                ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES
            ),
            None if vae is None else ArtifactIdentity.from_path(vae),
        )


def validate_request(width: int, height: int, seed: int) -> None:
    """Validate the canvas domain for square and landscape generation."""
    validate_u64(seed, label="seed")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height)):
        raise TypeError("width and height must be integers")
    if width % ALIGNMENT or height % ALIGNMENT:
        raise ValueError("SDXL width and height must be multiples of 8 pixels")
    if min(width, height) < MIN_SIDE:
        raise ValueError("SDXL width and height must each be at least 256 pixels")
    if max(width, height) > MAX_SIDE:
        raise ValueError(f"SDXL width and height must each be at most {MAX_SIDE} pixels")
    if width * height > MAX_PIXELS:
        raise ValueError(f"SDXL width * height must not exceed {MAX_PIXELS} pixels")
    if max(width, height) > min(width, height) * MAX_ASPECT:
        raise ValueError(f"SDXL aspect ratio must not exceed {MAX_ASPECT:g}:1")
