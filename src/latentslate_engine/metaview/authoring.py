"""Native-free authoring ownership for MetaView custom recipes."""

from .contracts import TOKENIZER_FILES
from .recipes import ARTIFACTS, METAVIEW_POLICY

POLICIES = (METAVIEW_POLICY,)
CALLER_INPUTS = frozenset({"image"})
RECIPE_FIELDS = frozenset({"width", "height", "seed", "yaw", "pitch", "radius"})
HOST_BINDINGS = {}
ARTIFACT_SLOTS = {key: {"kind": "file"} for key in ARTIFACTS}
ARTIFACT_SLOTS["tokenizer"] = {"kind": "directory", "required_files": TOKENIZER_FILES}
