"""Identity-bound official support components for native Wan 2.2 T2V."""

from __future__ import annotations

from pathlib import Path

from .wan22_i2v_support import (
    WanI2VSupportPlan,
    _plan_wan_support,
    revalidate_wan_i2v_support,
)
from .wan22_stored_adapter import WAN22_14B_T2V_CONFIG

# The support-file identity and tokenizer behavior are structurally identical to
# I2V.  The operation-specific planner below is the only permitted constructor.
WanT2VSupportPlan = WanI2VSupportPlan


def plan_wan_t2v_support(root: Path) -> WanT2VSupportPlan:
    """Plan only the official T2V support topology (WanPipeline, boundary .875)."""

    return _plan_wan_support(
        root,
        pipeline_class="WanPipeline",
        boundary_ratio=0.875,
        transformer_config=WAN22_14B_T2V_CONFIG,
    )


def revalidate_wan_t2v_support(plan: WanT2VSupportPlan) -> bool:
    if not revalidate_wan_i2v_support(plan):
        return False
    try:
        current = plan_wan_t2v_support(plan.root)
    except (OSError, TypeError, ValueError):
        return False
    return (
        current.files == plan.files
        and current.fingerprint == plan.fingerprint
        and current.tokenizer_sha256 == plan.tokenizer_sha256
        and current.boundary_ratio == plan.boundary_ratio
    )
