"""Wan authoring keeps ordered high/low adapter phases independently owned."""

from .recipes import WAN2214B_FLF_POLICY, WAN2214B_I2V_POLICY, WAN2214B_T2V_POLICY

POLICIES = (WAN2214B_T2V_POLICY, WAN2214B_I2V_POLICY, WAN2214B_FLF_POLICY)
CALLER_INPUTS = frozenset({"prompt", "start_image", "end_image"})
RECIPE_FIELDS = frozenset(
    {
        "negative_prompt",
        "shift",
        "steps",
        "split_step",
        "cfg",
        "width",
        "height",
        "duration_seconds",
        "seed",
    }
)
HOST_BINDINGS = {}
FIELD_PRESENTATION = {
    "steps": {
        "certified_value": 4,
        "advanced_warning": (
            "Certified baseline: 4 steps. Other values 3–8 are mechanically "
            "supported / advanced; output quality is not certified."
        ),
    }
}
ARTIFACT_SLOTS = {
    "high_checkpoint": {"kind": "file"},
    "high_adapters": {"kind": "file"},
    "low_checkpoint": {"kind": "file"},
    "low_adapters": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
}
