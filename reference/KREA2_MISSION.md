# Native Krea 2 mission

## Objective and boundaries

Deliver certified Krea 2 Turbo text-to-image through the native Engine family,
ordinary Recipe/authoring/catalog/jobs contracts, and generic LatentSlate desktop
generation/version/provenance. Compare the frozen current curated Comfy workflow
on the RTX 5080 with the same native artifacts and inputs, then test meaningful
alternate weight formats and at least three credible LoRAs where available.
Style/image-reference and other new conditioning modes are follow-ups.

Preserve existing checkouts and unrelated work. Push feature branches only;
Home Lab ChatGPT reviews consequential findings and the final candidate before
any main-branch canonization. No generic inference framework is presumed.

## Starting state (fetched 2026-09-13 UTC)

| Repository | Verified origin/main | Feature branch | Isolated checkout |
|---|---|---|---|
| EnviralDesign/LatentSlate-Engine | `8c6e0b6f175074dd96fac75bc60e75546bf9f785` | `codex/krea2-native` | `C:/repos/LatentSlate-Engine-Krea2` |
| EnviralDesign/LatentSlate | `2e312d804694d745687fdcf86f669298be7a73ea` | `codex/krea2-native` | `C:/repos/LatentSlate-Krea2` |

The original Engine checkout contains uncommitted authoring/Klein work. It is
preserved and is not silently incorporated into this main-based mission.

## Current phase: 2–3 — measured boundary, native family tracer

Oracle frozen in `reference/comfy/krea2/`: current curated upstream template,
actual frontend API export, exact artifacts/hashes, enhanced text, output hashes,
coarse tensor shapes/dtypes/hashes and initial cold/five-warm measurements.
Square 1024×1024 and widescreen 1368×768 succeed. Fresh square repetition is
PNG-byte/pixel-identical. The README records the CLI subgraph-export discrepancy,
FP32 conditioning versus BF16 enhancement, and unresolved warm timing variance.
Initial median warm 24.101 s; final paired performance must be freshly measured.
Ignored full evidence is under `reference/local/krea2/oracle/`.

Native family bring-up is in progress (uncommitted `src/latentslate_engine/krea2`).
Post-enhancer token IDs and all 12 FP32 conditioning taps match exactly; CPU noise
and the full sigma vector match exactly. Text attention requires the pinned
native-GQA availability decision and repeated K/V fallback to select the same
FP32 attention kernel. Text FP8 weights use full-precision multiplication.
Full native runtime now matches the original prompt enhancement and both square
and landscape RGB outputs exactly. All 146 enhancer hidden/logit boundaries,
all 12 conditioning taps, noise, sigmas, first transformer output, final latent,
and decoded pixels match. Remaining dispatch differences were absent FP8 input
scale (Comfy uses 1.0), small-query attention dispatch plus BF16 math reduction,
and contiguous VAE attention inputs. The Krea decoder stays single-frame and
family-local; the existing Wan decoder's frozen layout remains unchanged.

First integrated diagnostic: square 61.817s (enhancement 17.427s, conditioning
0.122s, model load 0.535s, sampling 29.949s, decode 13.409s); same-session
landscape 28.513s (sampling 27.832s, decode 0.581s), model/conditioning reused.
These are correctness bring-up observations, NOT a paired performance gate.
Fresh cold/five-warm reference/native timing and RAM/VRAM remain required.
Native Recipe/ownership definitions exist, but service/catalog registration and
public authoring/desktop integration remain pending. Eight targeted regressions
pass. All temporary Comfy probes are removed and the baseline is stopped.

## Acceptance ledger

- [x] Untouched curated oracle executes; canonical API fixture and primary-source revisions frozen.
- [x] Official artifacts in the discovered M-drive hierarchy, exact sources/sizes/SHA256 recorded.
- [x] Conditioning/noise/schedule/transformer/latent/decode boundaries measured and reproduced.
- [ ] Native immutable identity/request, capability/policy/recipe, isolated lifecycle and built-in implemented.
- [ ] Same-artifact output parity and equivalent cold/five-warm timing/RAM/VRAM accepted (roughly 10% maximum regression; explained numerical residuals).
- [ ] Existing built-in identifiers/schemas/contracts preserved; ordinary catalog/jobs and exact provenance verified.
- [ ] Recipe duplicate, local and pinned portable refs/materialization, fixed/exposed values, revisions/schema lineage, enable/disable, export/import verified.
- [ ] Generic desktop provider-to-image-version/preview/persisted-provenance flow and relevant normal/narrow layouts verified.
- [ ] Meaningful alternate-weight matrix and at least three credible LoRAs where available; ordering/strength/reuse/identity transitions measured.
- [ ] Full Windows tests, native-free cross-platform CI and desktop fmt/check/tests/release-stage gates pass.
- [ ] Evidence packet includes comparison images, screenshots, exact measurements, compatibility matrices, limitations and test/CI results.
- [ ] Clean feature branches pushed at proven checkpoints; independent final ChatGPT review complete.

## Review partner and next action

User-designated existing Home Lab conversation:
`https://chatgpt.com/g/g-p-6776fc5af4b4819194c3b58779fb0a7c/c/6a971736-5f74-83ea-8dc6-b6d013796058`.
Initial bounded research review completed: no blocker, continue. Credible
representation sequence FP8 → INT8 ConvRot → NVFP4 → W4A8; MXFP8 lower priority.
W4A8 candidate `realrebelai/Rebels_w4a8s` revision
`f82f0f2edcc85e5b8c4437ed7b60104bd0605212`, SHA256
`468bf97902559bfab6566b71fe10601a6098b221b27a9cd1aaaef49c14cef056` must be
verified locally. Official darkbrush, retroanime, rainywindow are the three
contrasting LoRA candidates. RAW and style-reference change semantics and remain
outside baseline. Treat research as candidate evidence until local verification.

User-requested Klein noise detour completed: FP8 adapter base-scale correction
applied to the running original checkout, preserving its unrelated changes, and
committed/pushed as `fd886a9029cad28608d18002167d9e7233835d37` on this branch.
Real desktop seed-42 LoRA output is now coherent; the no-LoRA control is
pixel-identical before/after; original recipe unchanged; 64 Klein tests and
`cargo check` passed. Local detailed evidence is in the original Engine checkout
under `reference/local/klein-lora-noise/`. Krea mission resumed.

Next: register the proven native family through ordinary service/catalog/authoring,
then establish the fresh paired performance/resource gate and desktop flow.
