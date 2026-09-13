"""Native-free identity and request bounds for certified Krea Turbo T2I."""

from dataclasses import dataclass
import math
from pathlib import Path

from latentslate_engine.validation import validate_u64

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")
ALIGNMENT = 8
MIN_SIDE = 256
MAX_PIXELS = 840 * 1256


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
            raise ValueError("Krea artifact must be a file")
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class Krea2Identity:
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]
    adapters: tuple[tuple[ArtifactIdentity, float], ...] = ()

    @classmethod
    def from_paths(cls, diffusion, text_encoder, vae, tokenizer, adapters=()):
        """Resolve all state-bearing artifacts before native loading."""
        tokenizer = Path(tokenizer).resolve(strict=True)
        adapters = tuple(adapters)
        validate_adapters(adapters)
        return cls(
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer,
            tuple(
                ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES
            ),
            tuple(
                (ArtifactIdentity.from_path(path), float(strength))
                for path, strength in adapters
            ),
        )


def validate_adapters(adapters) -> None:
    """Validate the measured ordered transformer-adapter domain."""
    if len(adapters) > 2:
        raise ValueError("Krea supports at most two ordered transformer adapters")
    for _, strength in adapters:
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise TypeError("Krea adapter strength must be numeric")
        if not math.isfinite(strength) or not -2.0 <= strength <= 2.0:
            raise ValueError("Krea adapter strength must be finite and within -2 to 2")


def validate_request(width: int, height: int, seed: int) -> None:
    """Validate the canvas domain containing the certified square and landscape."""
    validate_u64(seed, label="seed")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height)):
        raise TypeError("width and height must be integers")
    if width % ALIGNMENT or height % ALIGNMENT:
        raise ValueError("Krea width and height must be multiples of 8 pixels")
    if min(width, height) < MIN_SIDE:
        raise ValueError("Krea width and height must each be at least 256 pixels")
    if width * height > MAX_PIXELS:
        raise ValueError(f"Krea width * height must not exceed {MAX_PIXELS} pixels")
    if max(width, height) > min(width, height) * 4:
        raise ValueError("Krea aspect ratio must not exceed 4:1")
