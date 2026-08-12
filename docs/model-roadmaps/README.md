# Model roadmaps

This directory records which exact model artifacts and execution paths
LatentSlate Engine should qualify, recommend, defer, or reject. It is the bridge
between upstream experimentation and an opinionated Engine recipe.

These are living decision documents, not a promise to implement every format
that exists. A roadmap should reduce work by making the next qualification target
obvious and by recording why tempting alternatives are not currently worth the
maintenance cost.

## Status vocabulary

- **Reference** — exact high-precision source of truth used for quality comparison.
- **Recommended** — the current opinionated default based on measured creator value.
- **Experimental** — an exact artifact and plausible runtime exist, but qualification
  is incomplete.
- **Fallback** — useful for compatibility or constrained hardware, but not preferred.
- **Deferred** — real or plausible, but blocked by missing artifacts, loaders, kernels,
  evidence, or sufficient product value.
- **Rejected** — deliberately outside the current product direction.

`Recommended` is intentionally a product judgment. It may change as Engine gathers
better output, speed, memory, stability, and hardware evidence.

## Required structure

Each roadmap should contain:

1. Scope: exact upstream lineage, operations, and target hardware.
2. Current Engine truth: implemented recipes, proof level, and known gaps.
3. Availability matrix: published artifacts, runtime/kernel support, evidence,
   Engine support, and disposition.
4. A short qualification ladder, normally one reference, one incumbent, and one
   challenger—not a combinatorial format menu.
5. Reproducible acceptance methodology and a material-win threshold.
6. Ordered next actions and explicit non-goals.
7. Primary upstream sources, exact versions/revisions where decisions depend on them,
   and a last-reviewed date.

## Qualification rules

- Compare like with like: same model lineage, operation, prompt/input, seed,
  dimensions, scheduler, step count, guidance, encoder, and VAE.
- Artifact precision and quantization are recipe/resource facts, not runtime toggles.
- Engine never quantizes or converts weights during normal execution.
- Record the actual dispatched kernel/backend. A low-bit result that fell back to
  eager execution is not evidence for the intended acceleration path.
- Measure cold time, warm time, peak VRAM, peak host RAM, load/compile overhead,
  output quality, and teardown/reuse behavior.
- Do not add a production loader merely because a kernel or checkpoint exists.
  Require an exact artifact contract and a creator-visible benefit.
- Prefer first-party and Comfy-supported artifacts. Community artifacts may enter
  Experimental status when their provenance and layout can be pinned exactly.

## Roadmap research branches

A research-only roadmap task should modify files under `docs/model-roadmaps/` only.
It must not change recipes, resource declarations, runtime code, the root README, or
active schema documentation. That keeps long-running model research independently
reviewable and mergeable while Engine implementation continues on `main`.

Research additions should distinguish verified primary-source facts from inference,
link directly to the supporting source, and leave uncertain measurements explicitly
unqualified. Implementation decisions remain a separate reviewed change after a
roadmap establishes the target.

The first roadmap is [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md).

## Target roadmap queue

Research should cover these families or materially distinct model lines. Keep
separate roadmaps where artifact lineage, runtime, or qualification criteria differ:

- FLUX.2 Klein 4B
- FLUX.2 Klein 9B
- FLUX.2 Krea 2
- Stable Diffusion XL
- Qwen Image Edit 2511
- Ideogram 4
- Wan 2.2 TI2V 5B
- Wan 2.2 14B
- LTX 2.3
- LTX 2.5
- MiniMax H3

The queue is a research boundary, not a commitment to implement every published
precision or community checkpoint. Each roadmap should collapse the available
surface into a small qualification ladder and explicitly reject or defer low-value
branches.
