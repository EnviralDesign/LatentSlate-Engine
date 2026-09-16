"""Native-free identity and request bounds for Ideogram v4 T2I."""

from dataclasses import dataclass
from pathlib import Path

from latentslate_engine.validation import validate_u64

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")
ALIGNMENT = 16
MIN_SIDE = 256
# The reference's 1 MP widescreen preset rounds up to this 16-pixel canvas.
MAX_PIXELS = 1376 * 768


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
            raise ValueError("Ideogram v4 artifact must be a file")
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class Ideogram4Identity:
    negative_diffusion: ArtifactIdentity
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]

    @classmethod
    def from_paths(cls, diffusion, negative_diffusion, text_encoder, vae, tokenizer):
        """Resolve all state-bearing artifacts before native loading."""
        tokenizer = Path(tokenizer).resolve(strict=True)
        return cls(
            ArtifactIdentity.from_path(negative_diffusion),
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer,
            tuple(
                ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES
            ),
        )


def validate_request(width: int, height: int, seed: int) -> None:
    """Validate the canvas domain for square and landscape generation."""
    validate_u64(seed, label="seed")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height)):
        raise TypeError("width and height must be integers")
    if width % ALIGNMENT or height % ALIGNMENT:
        raise ValueError("Ideogram v4 width and height must be multiples of 16 pixels")
    if min(width, height) < MIN_SIDE:
        raise ValueError(
            "Ideogram v4 width and height must each be at least 256 pixels"
        )
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Ideogram v4 width * height must not exceed {MAX_PIXELS} pixels"
        )
    if max(width, height) > min(width, height) * 4:
        raise ValueError("Ideogram v4 aspect ratio must not exceed 4:1")
