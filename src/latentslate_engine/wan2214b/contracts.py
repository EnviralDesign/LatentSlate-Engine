"""Torch-free identity, recipe, and request contract for proven Wan T2V."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from latentslate_engine.validation import MAX_U64, validate_u64

POSITIVE_PROMPT = (
    "a robot walks through the interior of a house, scanning ordinary objects such as a "
    "coffee table, a kitchen table, and a plant."
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
)
FRAME_RATE = 16
MIN_SIDE = 480
MAX_PIXELS = 1280 * 720
MAX_ASPECT_NUMERATOR = 16
MAX_ASPECT_DENOMINATOR = 9
MIN_FRAME_COUNT = 17
MAX_FRAME_COUNT = 81
MAX_SEED = MAX_U64


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: str | Path) -> ArtifactIdentity:
        resolved = Path(path).resolve(strict=True)
        stat = resolved.stat()
        return cls(str(resolved), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class WanRecipe:
    high_checkpoint: str = r"M:\ComfyUI\models\diffusion_models\wan22\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    high_lora: str | None = (
        r"M:\ComfyUI\models\loras\wan\wan2.2_t2v_lightx2v_4steps_lora_v1_1_high_noise.safetensors"
    )
    low_checkpoint: str = r"M:\ComfyUI\models\diffusion_models\wan22\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    low_lora: str | None = (
        r"M:\ComfyUI\models\loras\wan\wan2.2_t2v_lightx2v_4steps_lora_v1_1_low_noise.safetensors"
    )
    text_encoder: str = (
        r"M:\ComfyUI\models\text_encoders\wan\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    )
    vae: str = r"M:\ComfyUI\models\vae\wan\wan_2.1_vae.safetensors"
    high_secondary_lora: str | None = None
    low_secondary_lora: str | None = None
    high_lora_strength: float = 1.0000000000000002
    low_lora_strength: float = 1.0000000000000002
    high_secondary_lora_strength: float = 1.0
    low_secondary_lora_strength: float = 1.0
    shift: float = 5.000000000000001
    steps: int = 4
    split_step: int = 2
    cfg: float = 1.0
    width: int = 512
    height: int = 512
    frame_count: int = 81
    positive: str = POSITIVE_PROMPT
    negative: str = NEGATIVE_PROMPT

    @property
    def identity(self) -> tuple[object, ...]:
        artifacts = tuple(
            ArtifactIdentity.from_path(path) if path is not None else None
            for path in (
                self.high_checkpoint,
                self.high_lora,
                self.low_checkpoint,
                self.low_lora,
                self.text_encoder,
                self.vae,
                self.high_secondary_lora,
                self.low_secondary_lora,
            )
        )
        return artifacts + (
            self.high_lora_strength,
            self.low_lora_strength,
            self.high_secondary_lora_strength,
            self.low_secondary_lora_strength,
            self.shift,
            self.steps,
            self.split_step,
            self.cfg,
        )

    def validate(self) -> None:
        expected = WanRecipe()
        fixed = ("shift", "steps", "split_step", "cfg")
        mismatches = [
            name for name in fixed if getattr(self, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ValueError(
                f"Wan T2V turbo runtime does not support changed settings: {mismatches}"
            )
        validate_request(self.width, self.height, self.frame_count, 0)


def validate_request(width: int, height: int, frame_count: int, seed: int) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise TypeError("Wan width and height must be integers")
    if width % 16 or height % 16:
        raise ValueError("Wan width and height must be multiples of 16 pixels")
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ValueError(
            f"Wan width and height must each be at least {MIN_SIDE} pixels"
        )
    if width * height > MAX_PIXELS:
        raise ValueError(f"Wan width * height must not exceed {MAX_PIXELS} pixels")
    short_side = min(width, height)
    long_side = max(width, height)
    if long_side * MAX_ASPECT_DENOMINATOR > short_side * MAX_ASPECT_NUMERATOR:
        raise ValueError("Wan aspect ratio must not exceed 16:9")

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("Wan frame_count must be an integer")
    if not MIN_FRAME_COUNT <= frame_count <= MAX_FRAME_COUNT:
        raise ValueError(
            f"Wan frame_count must be between {MIN_FRAME_COUNT} and {MAX_FRAME_COUNT}"
        )
    if frame_count % 4 != 1:
        raise ValueError("Wan frame_count must use the 4n+1 lattice")
    validate_u64(seed, label="Wan seed")
