# LatentSlate Engine greenfield reset

## Scope

`greenfield-reset` began as a clean documentation baseline. All former implementation
code, recipes, resource declarations, profiles, tests, automation, build and CI
configuration, generated artifacts, hardware-study output, and historical architecture
documents were removed from the tracked repository before the rebuild began.

New implementation is now being added from that clean state. The reset remains an
architectural boundary: no pre-reset module, process boundary, storage shape, recipe
schema, or inference abstraction is assumed to survive merely because it existed before.

The active implementation must continue to earn structure from measured working model
paths according to `AGENTS.md`; this document is not a frozen claim that the branch is
still empty.

## Preserved truth

Only the facts needed to begin and constrain the rebuild were carried forward:

- `ENGINE_CONTRACT.md` records the external LatentSlate-facing service/tool contract
  and stable LTX public identities.
- `COMFY_REFERENCE.md` records how pinned ComfyUI, comfy-aimdo, and comfy-kitchen
  should be used as source authorities without recreating ComfyUI.
- `LTX23_TARGET.md` records the first bounded implementation target and parity process.
- `../reference/comfy/ltx23/` contains canonical executable Comfy parity fixtures.

These are constraints and reference points, not a preselected internal architecture.
Current implementation code added after the reset is allowed to become authority only
through successful measured execution and the evidence progression in `AGENTS.md`.

## Recovery checkpoint

The complete pre-reset working tree is preserved in Git:

- Commit: `86419a7b943a2dcd9a172c817aafb3f05728331d`
- Annotated tag: `ltx23-pre-greenfield-reset-2026-08-26`

The checkpoint is recoverable for audit or selective factual comparison. It is not the
starting architecture for the new Engine and should not be browsed for implementation
inspiration unless a bounded archaeology task is explicitly authorized.
