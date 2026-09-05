"""Torch-free identity and request contracts for the proven LTX operations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from latentslate_engine.validation import MAX_U64, validate_u64

MIN_SIDE = 64
MAX_PIXELS = 942_080
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 10.0
MAX_SEED = MAX_U64


@dataclass(frozen=True)
class Ltx23T2VIdentity:
    """The complete, concrete model identity of the proven T2V operation."""

    checkpoint_path: str
    text_checkpoint_path: str
    transformer_lora_path: str | None
    upsampler_path: str
    lora_strength: float = 0.5
    device_index: int = 0
    transformer_loras: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Ltx23I2VIdentity:
    """The complete, concrete model identity of the LTX I2V operation."""

    checkpoint_path: str
    text_checkpoint_path: str
    transformer_lora_path: str | None
    upsampler_path: str
    lora_strength: float = 0.5
    device_index: int = 0
    transformer_loras: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Ltx23FlfIdentity:
    """The complete model identity of the LTX FLF operation."""

    checkpoint_path: str
    text_checkpoint_path: str
    device_index: int = 0


def validate_ltx_request(
    width: int,
    height: int,
    duration_seconds: float,
    seed: int,
    *,
    alignment: int,
) -> None:
    """Validate the recovered LTX product domain without changing inputs."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise TypeError("LTX width and height must be integers")
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ValueError(f"LTX width and height must each be at least {MIN_SIDE}")
    if width % alignment or height % alignment:
        raise ValueError(f"LTX width and height must each be divisible by {alignment}")
    if width * height > MAX_PIXELS:
        raise ValueError(f"LTX width * height must not exceed {MAX_PIXELS}")

    if isinstance(duration_seconds, bool):
        raise TypeError("LTX duration_seconds must be numeric")
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("LTX duration_seconds must be numeric") from error
    if not math.isfinite(duration):
        raise ValueError("LTX duration_seconds must be finite")
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"LTX duration_seconds must be between {MIN_DURATION_SECONDS} and "
            f"{MAX_DURATION_SECONDS}"
        )
    if not math.isclose(duration * 2.0, round(duration * 2.0), abs_tol=1e-9):
        raise ValueError("LTX duration_seconds must use 0.5-second increments")

    validate_u64(seed, label="LTX seed")
