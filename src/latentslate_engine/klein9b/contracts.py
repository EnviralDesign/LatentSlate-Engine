"""Torch-free identity and request contract for the proven Klein operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from latentslate_engine.validation import MAX_U64, validate_u64

RECIPE_ID = "flux2-klein-9b-distilled-t2i-768-v1"
KLEIN_ALIGNMENT = 16
KLEIN_MIN_SIDE = 256
KLEIN_MAX_PIXELS = 1024 * 1024
KLEIN_MAX_ASPECT = 4.0
KLEIN_MAX_SEED = MAX_U64
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path: Path) -> ArtifactIdentity:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        return cls(resolved, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class Klein9BIdentity:
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]
    text_encoder_config: ArtifactIdentity
    loras: tuple[ArtifactIdentity, ...] = ()
    recipe: str = RECIPE_ID

    @classmethod
    def from_paths(
        cls,
        diffusion: Path,
        text_encoder: Path,
        vae: Path,
        tokenizer: Path,
        *,
        loras: tuple[Path, ...] = (),
    ) -> Klein9BIdentity:
        tokenizer_path = tokenizer.resolve(strict=True)
        config_path = tokenizer_path.parent / "text_encoder" / "config.json"
        return cls(
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer_path,
            tuple(
                ArtifactIdentity.from_path(tokenizer_path / name)
                for name in TOKENIZER_FILES
            ),
            ArtifactIdentity.from_path(config_path),
            tuple(ArtifactIdentity.from_path(lora) for lora in loras),
        )


def validate_klein_seed(seed: int) -> None:
    validate_u64(seed, label="seed")


def validate_klein_dimensions(width: int, height: int) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise TypeError("width and height must be integers")
    if width % KLEIN_ALIGNMENT != 0 or height % KLEIN_ALIGNMENT != 0:
        raise ValueError(
            f"width and height must be multiples of {KLEIN_ALIGNMENT} pixels"
        )
    if width < KLEIN_MIN_SIDE or height < KLEIN_MIN_SIDE:
        raise ValueError(
            f"width and height must each be at least {KLEIN_MIN_SIDE} pixels"
        )
    if width * height > KLEIN_MAX_PIXELS:
        raise ValueError(f"width * height must not exceed {KLEIN_MAX_PIXELS} pixels")
    if max(width, height) > min(width, height) * KLEIN_MAX_ASPECT:
        raise ValueError(f"aspect ratio must not exceed {KLEIN_MAX_ASPECT:g}:1")


def validate_klein_request(width: int, height: int, seed: int) -> None:
    validate_klein_dimensions(width, height)
    validate_klein_seed(seed)
