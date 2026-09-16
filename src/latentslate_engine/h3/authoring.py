"""Portable authoring ownership for native H3 recipes."""

from .recipes import ARTIFACT_KEYS, REFERENCE_KEYS
from .recipes import POLICIES as OPERATION_POLICIES

POLICIES = tuple(OPERATION_POLICIES.values())
CALLER_INPUTS = frozenset(
    {
        "prompt",
        "start_image",
        *(key for keys in REFERENCE_KEYS.values() for key in keys),
    }
)
RECIPE_FIELDS = frozenset(
    {"width", "height", "duration_seconds", "fps", "seed", "reference_image_size"}
)
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {key: {"kind": "file"} for key in ARTIFACT_KEYS if key != "tokenizer"}
ARTIFACT_SLOTS["adapters"] = {"kind": "file"}
ARTIFACT_SLOTS["tokenizer"] = {
    "kind": "directory",
    "required_files": ("vocab.json", "merges.txt", "tokenizer_config.json"),
}
FIELD_PRESENTATION = {
    "fps": {"label": "FPS"},
    "reference_image_size": {"label": "Reference image size"},
}
