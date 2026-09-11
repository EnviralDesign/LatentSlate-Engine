"""Klein authoring preserves artifact-only LoRAs and paired optional geometry."""

from .contracts import TOKENIZER_FILES
from .recipes import KLEIN9B_T2I_POLICY, KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY

POLICIES = (KLEIN9B_T2I_POLICY, KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY)
CALLER_INPUTS = frozenset({"prompt", "image_1", "image_2"})
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {
    "diffusion": {"kind": "file"},
    "text_encoder": {"kind": "file"},
    "vae": {"kind": "file"},
    "tokenizer": {
        "kind": "directory",
        "required_files": (*TOKENIZER_FILES, "../text_encoder/config.json"),
    },
    "loras": {"kind": "file"},
}
