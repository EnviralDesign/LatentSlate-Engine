"""Native-free MetaView artifact identity and request constraints."""

from dataclasses import dataclass
import math
from pathlib import Path
from latentslate_engine.qwen2511.contracts import ArtifactIdentity, TOKENIZER_FILES
from latentslate_engine.validation import validate_u64

ALIGNMENT = 16
MIN_SIDE = 256
MAX_PIXELS = 960 * 528


@dataclass(frozen=True)
class MetaViewModelPaths:
    diffusion: Path
    text_encoder: Path
    vae: Path
    tokenizer: Path
    geometry_model: Path
    depth_model: Path

    @classmethod
    def from_root(cls, root: Path):
        return cls(
            root / "diffusion_models/qwen/metaview_dit_fp8scaled.safetensors",
            root / "text_encoders/qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            root / "vae/qwen/qwen_image_vae.safetensors",
            root / "text_encoders/qwen/tokenizer",
            root / "depth_anything_3/da3_giant.safetensors",
            root / "depth_anything_3/da3nested_giant_large.safetensors",
        )

    def available(self):
        return all(path.is_file() for path in (
            self.diffusion, self.text_encoder, self.vae, self.geometry_model, self.depth_model,
        )) and all((self.tokenizer / name).is_file() for name in TOKENIZER_FILES)


def validate_request(width, height, seed, yaw, pitch, radius):
    validate_u64(seed, label="seed")
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value < MIN_SIDE or value % ALIGNMENT:
            raise ValueError(f"MetaView {name} must be an integer multiple of {ALIGNMENT}, at least {MIN_SIDE}")
    if width * height > MAX_PIXELS:
        raise ValueError(f"MetaView canvas exceeds its {MAX_PIXELS}-pixel budget")
    for name, value, low, high in (("yaw", yaw, -180, 180), ("pitch", pitch, -90, 90)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"MetaView {name} must be finite and between {low} and {high} degrees")
    if radius is not None and (isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0):
        raise ValueError("MetaView radius must be finite and nonnegative; zero or omitted selects automatic radius")


@dataclass(frozen=True)
class MetaViewIdentity:
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    geometry_model: ArtifactIdentity
    depth_model: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]

    @classmethod
    def from_paths(cls, *, diffusion, text_encoder, vae, geometry_model, depth_model, tokenizer):
        tokenizer = Path(tokenizer).resolve(strict=True)
        return cls(*(ArtifactIdentity.from_path(p) for p in (
            diffusion, text_encoder, vae, geometry_model, depth_model,
        )), tokenizer, tuple(ArtifactIdentity.from_path(tokenizer / name) for name in TOKENIZER_FILES))
