"""Torch-free Wan product timing rules shared with the service catalog."""

from __future__ import annotations

import math

FRAME_RATE = 16
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 5.0
DURATION_STEP_SECONDS = 0.25


def validate_duration_seconds(duration_seconds: float) -> float:
    if isinstance(duration_seconds, bool) or not isinstance(
        duration_seconds, (int, float)
    ):
        raise TypeError("Wan duration_seconds must be numeric")
    duration = float(duration_seconds)
    if not math.isfinite(duration):
        raise ValueError("Wan duration_seconds must be finite")
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError("Wan duration_seconds must be between 1.0 and 5.0")
    if not math.isclose(duration * 4.0, round(duration * 4.0), abs_tol=1e-9):
        raise ValueError("Wan duration_seconds must use 0.25-second increments")
    return duration


def native_frame_count(duration_seconds: float) -> int:
    duration = validate_duration_seconds(duration_seconds)
    return round(duration * FRAME_RATE) + 1


def delivery_frame_count(duration_seconds: float) -> int:
    return native_frame_count(duration_seconds) - 1
