"""Ideogram v4 ownership for the existing descriptor-driven Recipe Studio."""

from .contracts import TOKENIZER_FILES
from .recipes import IDEOGRAM4_T2I_POLICY

POLICIES = (IDEOGRAM4_T2I_POLICY,)
CALLER_INPUTS = frozenset({"prompt"})
RECIPE_FIELDS = frozenset(
    {"background", "width", "height", "seed", "steps", "mu", "std", "sampler"}
)
HOST_BINDINGS = {}
FIELD_PRESENTATION = {
    "negative_diffusion": {"label": "Negative diffusion (omit for CFG 1)"},
    "background": {"label": "Background"},
    "steps": {"label": "Steps"},
    "mu": {"label": "Mu"},
    "std": {"label": "Std"},
    "sampler": {"label": "Sampler"},
}
ARTIFACT_SLOTS = {
    "diffusion": {"kind": "file"},
    "negative_diffusion": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {"kind": "directory", "required_files": TOKENIZER_FILES},
    "adapters": {"kind": "file"},
}
PRESET_TEMPLATES = (
    {
        "key": "quality",
        "label": "Quality",
        "mode": "exposed",
        "value": "default",
        "driven": ["steps", "mu", "std"],
        "choices": [
            {"key": "quality", "label": "Quality", "values": {"steps": 48, "mu": 0.0, "std": 1.5}},
            {"key": "default", "label": "Default", "values": {"steps": 20, "mu": 0.0, "std": 1.75}},
            {"key": "turbo", "label": "Turbo", "values": {"steps": 12, "mu": 0.5, "std": 1.75}},
        ],
    },
)
