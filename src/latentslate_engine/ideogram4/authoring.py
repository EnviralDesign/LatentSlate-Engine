"""Ideogram v4 ownership for the existing descriptor-driven Recipe Studio."""

from .contracts import TOKENIZER_FILES
from .recipes import IDEOGRAM4_T2I_POLICY

POLICIES = (IDEOGRAM4_T2I_POLICY,)
CALLER_INPUTS = frozenset({"prompt"})
RECIPE_FIELDS = frozenset({"width", "height", "seed"})
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {
    "diffusion": {"kind": "file"},
    "negative_diffusion": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {"kind": "directory", "required_files": TOKENIZER_FILES},
}
