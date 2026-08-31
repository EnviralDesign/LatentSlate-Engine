"""Small request invariants shared by every proven model family."""

MAX_U64 = (1 << 64) - 1


def validate_u64(value: int, *, label: str) -> None:
    """Reject non-integer and out-of-range unsigned 64-bit values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value <= MAX_U64:
        raise ValueError(f"{label} must be between 0 and {MAX_U64}")
