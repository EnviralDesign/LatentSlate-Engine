# LatentSlate Engine greenfield reset

## Scope

`greenfield-reset` is a clean documentation baseline. All former implementation
code, recipes, resource declarations, profiles, tests, automation, build and CI
configuration, generated artifacts, hardware-study output, and historical
architecture documents have been removed from the tracked repository.

The Engine must be designed and rebuilt deliberately. No pre-reset module,
process boundary, storage shape, recipe schema, or inference abstraction is
assumed to survive.

## Preserved truth

Only the facts needed to begin the rebuild are carried forward:

- `ENGINE_CONTRACT.md` records the external LatentSlate-facing service/tool
  contract and stable LTX public identities.
- `COMFY_REFERENCE.md` records how pinned ComfyUI, comfy-aimdo, and comfy-kitchen
  should be used as source authorities without recreating ComfyUI.
- `LTX23_TARGET.md` records the first bounded implementation target and fixed
  parity benchmark.

These documents are constraints and reference points, not a preselected internal
architecture.

## Recovery checkpoint

The complete pre-reset working tree is preserved in Git:

- Commit: `86419a7b943a2dcd9a172c817aafb3f05728331d`
- Annotated tag: `ltx23-pre-greenfield-reset-2026-08-26`

The checkpoint is recoverable for audit or selective factual comparison. It is
not the starting architecture for the new Engine.
