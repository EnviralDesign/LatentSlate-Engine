"""Shared LoRA selection semantics independent of any runtime package."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfiguredLora:
    """A declared LoRA slot, including disabled slots that never reach a runtime.

    ``resource_reference`` intentionally remains an identifier rather than a resolved
    path.  A zero-strength slot must not cause a resource path lookup, artifact probe,
    or adapter load merely so it can be recorded in provenance.
    """

    slot: str
    resource_reference: str | None
    strength: float
    active: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError("Configured LoRA strength must be finite")
        expected_active = self.resource_reference is not None and self.strength != 0.0
        if self.active != expected_active:
            raise ValueError("Configured LoRA active state must match its reference and strength")


def active_loras(loras: Iterable[Any]) -> tuple[Any, ...]:
    """Return the ordered active adapter stack.

    Strength zero is a semantic bypass, not a loaded adapter with a zero weight.
    This stays duck-typed so it can protect family-specific lifecycle calls without
    coupling their runtime packages back to ``tools``.
    """

    return tuple(lora for lora in loras if float(lora.strength) != 0.0)
