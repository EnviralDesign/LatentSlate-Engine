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
  When an upstream vendor publishes no dense checkpoint, the roadmap must say so
  and may name a first-party public baseline without pretending it is lossless.
- **Recommended** — the current opinionated default based on measured creator value.
- **Experimental** — an exact artifact and plausible runtime exist, but qualification
  is incomplete.
- **Fallback** — useful for compatibility or constrained hardware, but not preferred.
- **Deferred** — real or plausible, but blocked by missing artifacts, loaders, kernels,
  evidence, license clearance, or sufficient product value.
- **Rejected** — deliberately outside the current product direction.

`Recommended` is intentionally a product judgment. It may change as Engine gathers
better output, speed, memory, stability, and hardware evidence. A roadmap may
legitimately have no Recommended path yet.

## Engine proof vocabulary

Roadmaps use these proof levels so “code exists” is not confused with creator-ready
support:

- **Hardware-proven** — an end-to-end job completed on the target workstation class
  and the intended backend was observed.
- **Runtime-proven** — an end-to-end Engine job completed, but the target-class
  acceptance matrix is incomplete.
- **Structurally tested** — schemas, loaders, component contracts, or mocked runtime
  behavior are tested; accepted generated output is not established.
- **Cataloged** — a recipe/resource declaration exists, but execution proof is absent
  or incomplete.
- **Direct tool only** — callable Engine code exists outside the opinionated recipe
  and acquisition system.
- **Not implemented** — no Engine family/runtime/recipe path exists.

## Portfolio decision surface

Last reviewed: **2026-08-13**. The workstation lens is Windows 11, RTX 5080 16 GB
(SM120), Python 3.12, and the Engine runtime selected by its adaptive bootstrap.
“None” is deliberate: it means the evidence does not yet justify a product default.

| Target | Reference | Recommended | Next Experimental challenger | Engine proof level | Highest-priority gap |
| --- | --- | --- | --- | --- | --- |
| [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md) | Matching BFL BF16 Distilled or Base with the same Comfy component closure | First-party Distilled NVFP4 on qualified Blackwell; first-party Distilled FP8 fallback elsewhere | Base only after a matching Base BF16 edit reference; one Distilled ConvRot experiment later if justified | Controlled 3-cold/3-warm 1024² T2I/I2I baseline, deterministic outputs, switching, and native dispatch accepted | Add cancellation and two/three-reference lifecycle coverage; keep cross-format quality comparison separate |
| [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md) | Authenticated first-party Distilled BF16 for ordinary T2I/edit | First-party Distilled NVFP4 on qualified Blackwell; first-party Distilled FP8 fallback elsewhere | Base and KV are separate backburner lines | Controlled 3-cold/3-warm 1024² NVFP4/FP8 T2I/I2I baseline and deterministic output accepted; BF16 honestly OOMs on 16 GB | Add cancellation and two/three-reference lifecycle coverage; retry BF16 only on larger hardware |
| [Krea 2](./KREA_2.md) | Turbo BF16 for product T2I; Raw BF16 only for training/foundation comparisons | None | Official Turbo BF16 exact-checkpoint path | Not implemented | Build a bounded T2I loader and resolve the Community License revenue/content-filter obligations |
| [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md) | Official FP16 Base; Base+Refiner is a separate reference operation | None | Base-only FP16 recipe | Not implemented | Demonstrate creator value versus newer image families before adding a legacy family |
| [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md) | Official BF16 40-step edit | None | BF16 + official 4-step Lightning LoRA, then its fused scaled-FP8 sibling | Not implemented | Design a bounded multi-image/offload path for a 40.9 GB transformer and prove edit fidelity |
| [Ideogram 4](./IDEOGRAM_4.md) | Official NF4 Diffusers public baseline; no dense public source of truth | None | Exact Comfy FP8 topology, followed by complete NVFP4 topology on Blackwell | Not implemented | Establish a local JSON-prompt pipeline, license posture, and a truthful reference despite no public BF16 |
| [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md) | Official dense BF16 Diffusers T2V | None | None | Dense BF16 reference is cataloged; split T2V and required-image I2V are accepted fallback paths with fixed-seed public-API acceptance on RTX 5080 | Broaden creator-quality coverage; preserve the separately qualified T2V/I2V contracts and fixed split closure |
| [Wan 2.2 14B](./WAN22_14B.md) | Matching official dense BF16 T2V or I2V expert pair | None; active Comfy FP8 I2V, T2V, and FLF baselines are **Fallback** | LightX2V I2V and T2V v1.1 are separately lifecycle-accepted as **Experimental**; FLF LightX remains future Experimental work; Winnougan INT8 ConvRot resources are cataloged but not runnable | Exact I2V, T2V, and FLF baseline success, cancellation, fresh recovery, native LoRA dispatch, disposable-worker teardown, and ConvRot catalog metadata accepted; exact ConvRot header planning/native dispatch remain unproven | Complete corpus, source-invalidation, peer-switch, endpoint-pair diversity, and BF16-reference quality evidence; qualify each LightX operation separately, pin a clean ConvRot operation, and prove its exact-header planner/dispatch path before recipe exposure |
| [LTX 2.3](./LTX_2_3.md) | Official Distilled BF16 for the matching T2V/I2V condition path | None | Official Distilled FP8 stored checkpoint | Cataloged BF16 T2V/I2V; hardware output acceptance pending | Replace the 95.0 GB full-folder substitution with exact components and verify synchronized audio plus conditioning |
| [LTX 2.5](./LTX_2_5.md) | Matching official Distilled or Dev BF16 component set | None | Distilled two-stage exact-component path with the Windows-viable VAE | Not implemented | Bound the roughly 66 GiB component closure, offload behavior, Windows decoder backend, and license gate |
| [MiniMax H3](./MINIMAX_H3.md) | Official BF16 FL2VA or Ref2VA matching the 768p operation | None | Re-pinned current-release BF16 FL2VA closure | Direct older-pinned FL2VA tools only; no package recipe; output acceptance pending | Reconcile release drift, prove single-5080 feasibility, keep Ref2VA separate, and preserve the hosted Context-IR/2K boundary |

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
  dimensions, scheduler, step count, guidance, encoder, VAE, frame count, and fps.
- Do not collapse Base and Distilled, T2I and edit, T2V and I2V, or single-stage and
  two-stage pipelines into one benchmark result.
- Artifact precision and quantization are recipe/resource facts, not runtime toggles.
- Engine never quantizes or converts weights during normal execution.
- Record the actual dispatched kernel/backend. A low-bit result that fell back to
  eager execution is not evidence for the intended acceleration path.
- Measure cold time, warm time, peak VRAM, peak host RAM, load/compile overhead,
  output quality, cancellation, reuse, and teardown behavior.
- Measure component churn: prompt-cache hit/miss, encoder residency, stage swaps,
  VAE encode/decode, mux/export, and whether a failed or cancelled job poisons reuse.
- Do not add a production loader merely because a kernel or checkpoint exists.
  Require an exact artifact contract and a creator-visible benefit.
- Prefer first-party and Comfy-supported artifacts. Community artifacts may enter
  Experimental status when their provenance and layout can be pinned exactly.
- Treat license gates and distribution obligations as recipe blockers, not a footnote.
- Record source conflicts explicitly. Do not reconcile different vendor VRAM or speed
  claims by guessing; reproduce them under one harness.

## Shared acceptance harness

Every model-specific ladder should reuse one harness and add only operation-specific
cases:

1. Pin exact artifact revisions, file identities, runtime commit, driver, Torch/CUDA,
   Comfy Kitchen version, and detected GPU capability.
2. Save the full effective request: prompt, negative prompt, media inputs, dimensions,
   steps, sampler/scheduler, guidance, seed, duration/frames/fps, LoRAs, and stage split.
3. Run one cold job from a fresh process and one to three warm jobs without changing
   inputs. A first structural pass may use one cold plus one warm; promotion evidence
   should include at least three stable warm observations.
4. Record wall-clock load, encode, denoise/stage, decode, audio, mux/export, and total
   times; peak VRAM; peak process RAM; and disk reads when offload is used.
5. Capture backend dispatch for every claimed low-bit linear path and attention backend.
6. Compare outputs with a fixed creator-reviewed corpus. Automated similarity metrics
   are supporting evidence, not the acceptance decision.
7. Cancel during loading, text encoding, denoising/stage transition, decode, and export;
   then run a clean follow-up job. Verify no stale model/session or leaked temp output.
8. Exercise reuse across identical requests, changed prompts, changed dimensions, and
   changed operations; then verify explicit teardown returns memory to the expected
   baseline.

A second production loader normally needs either a **20–25% material end-to-end warm
win**, a creator-relevant workload that the incumbent cannot run within the target
memory envelope, or a similarly clear cold-load/stability benefit. Single-digit kernel
wins, storage savings alone, or an unobserved fallback path do not justify the burden.

### Minimum opt-in API qualification record

The first hardware pass should be a small manual, non-CI API exercise—not a benchmark
framework.

The request manifest needs only:

- a `comparison_id`;
- one recipe key or an ordered A/B recipe sequence;
- lineage and operation;
- prompt/negative prompt;
- ordered asset IDs, roles, and content hashes;
- seed, effective dimensions, steps, sampler/scheduler, guidance, and cache policy;
- requested run count from one through three; and
- an optional cancellation phase.

The harness submits through the public Engine API, polls each job to a terminal state,
and writes outputs plus one structured record per run containing:

- job/recipe/resource identities and exact revisions/hashes;
- output path and content hash;
- cold/warm/cache state;
- load, encode, denoise/stage, decode/save/export, and total timing;
- peak allocated/reserved VRAM, process RSS, and system/Windows commit where relevant;
- GPU, capability, driver, Torch, CUDA, Kitchen, available backends, selected
  quantized layouts, backend-dispatch counts, and fallback events;
- terminal status, error/failure log, cancellation result, recovery result, and
  teardown memory baseline.

Stop the sequence immediately on a lineage/settings mismatch, unknown header layout,
eager/dequantized fallback in a claimed native path, OOM, NaN/black/corrupt output,
cancellation poison, or obvious creator-visible regression. Do not expand beyond the
small fixed corpus until structural correctness and native dispatch pass.

## Roadmap research branches

A research-only roadmap task should modify files under `docs/model-roadmaps/` only.
It must not change recipes, resource declarations, runtime code, the root README, or
active schema documentation. That keeps long-running model research independently
reviewable and mergeable while Engine implementation continues on `main`.

Research additions should distinguish verified primary-source facts from inference,
link directly to the supporting source, and leave uncertain measurements explicitly
unqualified. Implementation decisions remain a separate reviewed change after a
roadmap establishes the target.

## Roadmaps

- [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md)
- [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md)
- [Krea 2](./KREA_2.md) — Krea 2 is trained from scratch and is **not** a FLUX.2 lineage.
- [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md)
- [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md)
- [Ideogram 4](./IDEOGRAM_4.md)
- [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md)
- [Wan 2.2 14B](./WAN22_14B.md)
- [LTX 2.3](./LTX_2_3.md)
- [LTX 2.5](./LTX_2_5.md)
- [MiniMax H3](./MINIMAX_H3.md)

The portfolio is a research boundary, not a commitment to implement every published
precision or community checkpoint. Each roadmap collapses the available surface into
a small qualification ladder and explicitly rejects or defers low-value branches.
