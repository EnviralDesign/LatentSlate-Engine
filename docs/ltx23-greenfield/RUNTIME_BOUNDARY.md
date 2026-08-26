# LTX 2.3 runtime boundary

The rebuilt path must remain Engine-native:

- No broad `comfy.*` runtime dependency.
- Do not recreate Comfy's graph executor or global model manager.
- Use `comfy-aimdo` and `comfy-kitchen` directly at their intended
  abstraction level.
- Preserve useful warm state for the same model identity.
- Strictly replace prior model context for a different model identity.
- Implement and prove T2V first.
- Do not create a general model-neutral framework until working paths
  demonstrate real reuse.
- Prefer existing primitives and direct source-conformant code over wrappers.
- Treat the old implementation as Git history, not architecture authority.

The internal mechanism for meeting these boundaries is deliberately unspecified.
