"""Torch-free H3 request constraints shared by recipes and execution."""

import math
from dataclasses import dataclass
from pathlib import Path

from latentslate_engine.validation import validate_u64

ALIGNMENT = 32
MIN_SIDE = 32
FRAME_RATE = 24


@dataclass(frozen=True)
class H3ModelPaths:
    fl2va: Path
    ref2va: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    tokenizer: Path

    @classmethod
    def from_root(cls, root: Path):
        """Canonical ordinary-file destinations used by built-in bootstrap."""
        return cls(
            root
            / "diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            root
            / "diffusion_models/h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            root / "text_encoders/h3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            root / "vae/h3/minimax_h3_video_vae_fp16.safetensors",
            root / "vae/h3/minimax_h3_audio_vae_fp32.safetensors",
            root / "text_encoders/h3/tokenizer",
        )

    def bindings(self, operation):
        """Select the operation's backbone and shared companion artifacts."""
        if operation not in ("t2v", "i2v", "r2v"):
            raise ValueError("Unknown H3 operation")
        return {
            "diffusion": self.ref2va if operation == "r2v" else self.fl2va,
            **{
                key: getattr(self, key)
                for key in ("text_encoder", "video_vae", "audio_vae", "tokenizer")
            },
        }

    def available(self, operation):
        paths = self.bindings(operation)
        return all(
            path.is_file() for key, path in paths.items() if key != "tokenizer"
        ) and all(
            (self.tokenizer / name).is_file()
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json")
        )


def validate_canvas(width: int, height: int) -> None:
    """Require explicit patch-aligned output dimensions without rounding."""
    if type(width) is not int or type(height) is not int:
        raise TypeError("H3 width and height must be integers")
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ValueError(f"H3 width and height must each be at least {MIN_SIDE} pixels")
    if width % ALIGNMENT or height % ALIGNMENT:
        raise ValueError(f"H3 width and height must be multiples of {ALIGNMENT} pixels")


def validate_request(width, height, frame_count, seed, fps=FRAME_RATE):
    """Reject unsupported requests before any native model preparation."""
    validate_canvas(width, height)
    if type(frame_count) is not int or frame_count < 5 or frame_count % 17 != 5:
        raise ValueError("H3 frame count must use the 17n+5 grid, starting at 5")
    if type(fps) is not int or fps != FRAME_RATE:
        raise ValueError(f"H3 requires {FRAME_RATE} integer FPS")
    validate_u64(seed, label="H3 seed")


def validate_adapters(adapters):
    """Require finite strengths for the ordered model-only factor updates."""
    for _, strength in adapters:
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
        ):
            raise ValueError("H3 adapter strength must be a finite number")


@dataclass(frozen=True)
class H3Identity:
    diffusion: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    tokenizer: str
    artifact_versions: tuple[tuple[str, int, int], ...] = ()
    device_index: int = 0
    adapters: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_paths(cls, *, adapters=(), **paths):
        """Snapshot artifact files and tokenizer companions for worker identity."""
        adapters = tuple(adapters)
        validate_adapters(adapters)
        resolved = {}
        versions = []
        for key, value in sorted(paths.items()):
            path = Path(value).resolve(strict=True)
            resolved[key] = str(path)
            files = (
                [
                    path / name
                    for name in ("vocab.json", "merges.txt", "tokenizer_config.json")
                ]
                if key == "tokenizer"
                else [path]
            )
            for file in files:
                if not file.is_file():
                    raise ValueError("H3 artifacts must resolve to files")
                stat = file.stat()
                versions.append((str(file), stat.st_size, stat.st_mtime_ns))
        resolved_adapters = []
        for value, strength in adapters:
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError("H3 adapter must resolve to a file")
            stat = path.stat()
            versions.append((str(path), stat.st_size, stat.st_mtime_ns))
            resolved_adapters.append((str(path), float(strength)))
        return cls(
            **resolved,
            artifact_versions=tuple(versions),
            adapters=tuple(resolved_adapters),
        )
