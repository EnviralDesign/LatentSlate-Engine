# LatentSlate Engine greenfield reset

## Scope

`greenfield-reset` is a clean documentation baseline. All former implementation
code, recipes, resource declarations, profiles, tests, automation, build and CI
configuration, generated artifacts, hardware-study output, and historical
architecture documents have been removed from the tracked repository.

The Engine must be designed and rebuilt from scratch. No pre-reset module,
process boundary, storage shape, or recipe schema is assumed to survive.

## Preserved product facts

The LTX 2.3 product requirements that must remain stable are intentionally
recorded as contracts in `docs/ltx23-greenfield/`. They establish capabilities,
identifiers, reference pins, runtime boundaries, and parity gates; they do not
prescribe an implementation.

## Recovery checkpoint

The complete pre-reset working tree is preserved in Git:

- Commit: `86419a7b943a2dcd9a172c817aafb3f05728331d`
- Annotated tag: `ltx23-pre-greenfield-reset-2026-08-26`

The checkpoint is recoverable for audit or selective comparison. It is not the
starting architecture for the new Engine.
