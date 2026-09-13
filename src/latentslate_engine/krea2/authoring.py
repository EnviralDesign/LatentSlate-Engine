"""Krea T2I ownership for ordinary descriptor-driven Recipe Studio."""

from .contracts import TOKENIZER_FILES
from .recipes import KREA2_T2I_POLICY

POLICIES = (KREA2_T2I_POLICY,)
CALLER_INPUTS = frozenset({"prompt"})
RECIPE_FIELDS = frozenset({"width", "height", "seed", "prompt_suffix"})
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {
    "diffusion": {"kind": "file"},
    "adapters": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {"kind": "directory", "required_files": TOKENIZER_FILES},
}
