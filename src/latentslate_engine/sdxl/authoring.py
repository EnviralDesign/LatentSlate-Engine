"""SDXL fields for descriptor-driven Recipe Studio."""

from .contracts import TOKENIZER_FILES
from .recipes import SDXL_T2I_POLICY

POLICIES = (SDXL_T2I_POLICY,)
CALLER_INPUTS = frozenset({"prompt"})
RECIPE_FIELDS = frozenset(
    {
        "negative_prompt",
        "width",
        "height",
        "seed",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
    }
)
HOST_BINDINGS = {}
FIELD_PRESENTATION = {
    "vae": {"label": "VAE override (omit to use checkpoint VAE)"},
    "cfg": {"label": "CFG"},
}
ARTIFACT_SLOTS = {
    "checkpoint": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {"kind": "directory", "required_files": TOKENIZER_FILES},
}
