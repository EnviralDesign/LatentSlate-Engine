"""Recipe ownership for the three native LTX 2.5 operations."""

from .recipes import POLICIES as OPERATION_POLICIES

POLICIES = tuple(OPERATION_POLICIES.values())
CALLER_INPUTS = frozenset({"prompt", "start_image", "end_image"})
RECIPE_FIELDS = frozenset(
    {
        "width",
        "height",
        "duration_seconds",
        "fps",
        "seed",
        "prompt_enhancement",
        "transformer_adapter_strengths",
    }
)
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {
    key: {"kind": "file"}
    for key in (
        "diffusion",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "upsampler",
        "prompt_enhancer",
        "transformer_adapter_artifacts",
    )
}
FIELD_PRESENTATION = {
    "fps": {"label": "FPS"},
    "prompt_enhancement": {"label": "Prompt enhancement"},
    "prompt_enhancer": {"label": "Prompt enhancer"},
    "transformer_adapter_artifacts": {"label": "LoRA adapters", "item_label": "LoRA"},
    "transformer_adapter_strengths": {
        "label": "LoRA strengths",
        "item_label": "Strength",
    },
}
FIELD_GROUPS = (
    {
        "layout": "collection",
        "fields": ("transformer_adapter_artifacts", "transformer_adapter_strengths"),
    },
)
