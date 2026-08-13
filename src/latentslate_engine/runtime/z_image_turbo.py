"""CPU-safe lifecycle guard for the bounded Z-Image Turbo native path.

It intentionally does not implement a generic diffusion fallback.  This module
owns the pieces that can be proven without a GPU: immutable request identity,
ordered residency intent, cancellation checkpoints, and the requirement that
the stored ConvRot layers retain their exact source layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event

from ..z_image_turbo_recipe import (
    ZImageTurboRuntimeRequest,
    revalidate_z_image_turbo_runtime_request,
)


class ZImagePhase(StrEnum):
    PLANNING = "planning"
    TEXT_ENCODER = "text_encoder"
    TRANSFORMER = "transformer"
    VAE = "vae"
    COMPLETE = "complete"
    EJECTED = "ejected"


class ZImageTurboCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class ZImageTurboLifecycle:
    """One-job lifecycle.  Cancellation always ejects the planned runtime."""

    request: ZImageTurboRuntimeRequest
    phase: ZImagePhase = ZImagePhase.PLANNING
    events: list[ZImagePhase] = field(default_factory=lambda: [ZImagePhase.PLANNING])
    ejected: bool = False

    def checkpoint(self, phase: ZImagePhase, cancelled: Callable[[], bool] | Event) -> None:
        if self.ejected:
            raise RuntimeError("Z-Image runtime was ejected")
        expected = {
            ZImagePhase.PLANNING: ZImagePhase.TEXT_ENCODER,
            ZImagePhase.TEXT_ENCODER: ZImagePhase.TRANSFORMER,
            ZImagePhase.TRANSFORMER: ZImagePhase.VAE,
        }.get(self.phase)
        if phase is not expected:
            self.eject()
            raise ValueError(
                f"Z-Image lifecycle phase {phase.value!r} is invalid after {self.phase.value!r}"
            )
        if not revalidate_z_image_turbo_runtime_request(self.request):
            self.eject()
            raise ValueError("Z-Image runtime request changed or lost native dispatch proof")
        is_cancelled = cancelled.is_set if isinstance(cancelled, Event) else cancelled
        if is_cancelled():
            self.eject()
            raise ZImageTurboCancelled(f"Z-Image generation canceled during {phase.value}")
        self.phase = phase
        self.events.append(phase)

    def require_stored_transformer_plan(self) -> int:
        plan = self.request.plans["transformer"]
        # The recipe module deliberately owns the concrete type so this remains
        # a narrow runtime seam rather than another ad-hoc loader.
        require_stored_layout = getattr(plan, "require_stored_layout", None)
        if not callable(require_stored_layout):
            self.eject()
            raise TypeError("Z-Image transformer plan has no stored-layout contract")
        require_stored_layout()
        return int(getattr(plan, "stored_layer_count", 0))

    def complete(self) -> None:
        if self.ejected:
            raise RuntimeError("Z-Image runtime was ejected")
        if self.phase is not ZImagePhase.VAE:
            self.eject()
            raise ValueError("Z-Image lifecycle cannot complete before VAE decode")
        self.phase = ZImagePhase.COMPLETE
        self.events.append(ZImagePhase.COMPLETE)

    def eject(self) -> None:
        self.ejected = True
        self.phase = ZImagePhase.EJECTED
        if not self.events or self.events[-1] != ZImagePhase.EJECTED:
            self.events.append(ZImagePhase.EJECTED)

    def public_provenance(self) -> dict[str, object]:
        return {
            "runtime": "ZImageTurboNative",
            "request_fingerprint": self.request.fingerprint,
            "components": self.request.public_component_manifest(),
            "pipeline_warm": False,
            "execution_cache": {"supported": False, "hit": False, "mode": "fresh_contract"},
            "staging_order": ["text_encoder", "transformer", "vae"],
            "phases": [phase.value for phase in self.events],
            "ejected": self.ejected,
            "stored_transformer_layers_planned": self.require_stored_transformer_plan()
            if not self.ejected
            else 0,
            "native_transformer_dispatch": {
                "proven": False,
                "count": 0,
                "reason": "GPU execution has not been accepted",
            },
        }
