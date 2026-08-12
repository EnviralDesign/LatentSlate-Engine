# Documentation map

Keep this directory small and current.

## Active product documentation

- [Recipes, resources, and deployment profiles](./RECIPES.md) — catalog schema,
  acquisition, planning, installation, and lock behavior.
- [Engine diagnostics](./DIAGNOSTICS.md) — runtime-tier and hardware preflight.
- [Opt-in hardware studies](./HARDWARE_STUDIES.md) — deterministic one-off and
  small A/B generation runs through the public Engine API, outside routine CI.
- [Model roadmaps](./model-roadmaps/README.md) — qualification matrices and ordered
  implementation targets for the model families Engine cares about.

## Historical material

[Archived documentation](./archive/README.md) preserves useful implementation and
research context that is no longer an active contract. Do not use archived files as
setup instructions.

When an active document becomes stale, prefer one of three outcomes:

1. Update it if it remains a real source of truth.
2. Move still-useful history to `archive/` and point readers to the replacement.
3. Delete it when Git history is sufficient and the content has no continuing value.

Do not add session transcripts, speculative implementation diaries, or duplicated
README instructions. Model research belongs in a structured roadmap only when it
changes priorities, qualification criteria, or explicit non-goals.
