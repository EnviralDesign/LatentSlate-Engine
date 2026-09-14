"""Qwen ownership for immutable service Recipe compilation."""

from .contracts import TOKENIZER_FILES
from .recipes import QWEN2511_EDIT_POLICY

POLICIES = (QWEN2511_EDIT_POLICY,)
CALLER_INPUTS = frozenset({"prompt", "image_1", "image_2", "image_3"})
RECIPE_FIELDS = frozenset({"seed", "steps", "cfg", "shift"})
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {
    "diffusion": {"kind": "file"},
    "adapters": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {"kind": "directory", "required_files": TOKENIZER_FILES},
}
