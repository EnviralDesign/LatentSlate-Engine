"""LTX authoring ownership; execution device selection stays with the host."""

from .recipes import LTX23_FLF_POLICY, LTX23_I2V_POLICY, LTX23_T2V_POLICY

POLICIES = (LTX23_T2V_POLICY, LTX23_I2V_POLICY, LTX23_FLF_POLICY)
CALLER_INPUTS = frozenset({"prompt", "start_image", "end_image"})
HOST_BINDINGS = {"device_index": 0}
ARTIFACT_SLOTS = {
    "checkpoint": {"kind": "file"},
    "text_checkpoint": {"kind": "file"},
    "upsampler": {"kind": "file"},
    "transformer_adapter_artifacts": {"kind": "file"},
}
