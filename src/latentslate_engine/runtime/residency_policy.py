"""Small reusable policy seam for stored-weight runtime residency.

Model adapters remain responsible for identifying safe movement groups and for
moving their physical tensor storage.  This module only decides whether the
current CUDA budget can retain the whole model while preserving inference
headroom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ResidencyDecision:
    mode: str
    free_bytes: int
    total_bytes: int
    stored_bytes: int
    reserved_headroom_bytes: int
    stream_buffer_bytes: int
    resident_weight_budget_bytes: int
    reason: str

    def provenance(self) -> dict[str, int | str]:
        return asdict(self)


def choose_cuda_residency(
    *,
    free_bytes: int,
    total_bytes: int,
    stored_bytes: int,
    largest_group_bytes: int = 0,
    headroom_fraction: float = 0.60,
    minimum_headroom_bytes: int = 2 * 1024**3,
) -> ResidencyDecision:
    """Choose full or grouped residency from live capacity, not a GPU table.

    Generative transformer activations can dominate stored weights.  Sixty
    percent of physical capacity is retained by default for activations and
    already-live pipeline components; adapters can apply a stricter value when
    they have measured family-specific requirements.
    """

    if min(free_bytes, total_bytes, stored_bytes, largest_group_bytes) < 0 or free_bytes > total_bytes:
        raise ValueError("invalid CUDA residency capacity")
    if not 0.0 < headroom_fraction < 1.0 or minimum_headroom_bytes < 0:
        raise ValueError("invalid CUDA residency headroom policy")
    headroom = max(minimum_headroom_bytes, int(total_bytes * headroom_fraction))
    full_fits = stored_bytes + headroom <= free_bytes
    mode = "full" if full_fits else "grouped"
    stream_buffer = 0 if full_fits else largest_group_bytes
    resident_budget = min(
        stored_bytes,
        max(0, free_bytes - headroom - stream_buffer),
    )
    reason = (
        "live CUDA free memory retains activation headroom after full onload"
        if full_fits
        else "full onload would consume reserved activation headroom"
    )
    return ResidencyDecision(
        mode=mode,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        stored_bytes=stored_bytes,
        reserved_headroom_bytes=headroom,
        stream_buffer_bytes=stream_buffer,
        resident_weight_budget_bytes=resident_budget,
        reason=reason,
    )


def require_grouped_residency(
    decision: ResidencyDecision,
    *,
    largest_group_bytes: int,
    reason: str,
) -> ResidencyDecision:
    """Fail closed to a partial plan for a workload lacking a safe full estimate."""

    if largest_group_bytes <= 0 or largest_group_bytes > decision.stored_bytes:
        raise ValueError("invalid required grouped-residency buffer")
    budget = min(
        decision.stored_bytes - largest_group_bytes,
        max(
            0,
            decision.free_bytes
            - decision.reserved_headroom_bytes
            - largest_group_bytes,
        ),
    )
    return ResidencyDecision(
        mode="grouped",
        free_bytes=decision.free_bytes,
        total_bytes=decision.total_bytes,
        stored_bytes=decision.stored_bytes,
        reserved_headroom_bytes=decision.reserved_headroom_bytes,
        stream_buffer_bytes=largest_group_bytes,
        resident_weight_budget_bytes=budget,
        reason=reason,
    )
