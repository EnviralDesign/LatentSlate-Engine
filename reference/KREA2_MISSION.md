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

## Current phase: 3–4 — integrated family, performance diagnosis

Oracle frozen in `reference/comfy/krea2/`: current curated upstream template,
actual frontend API export, exact artifacts/hashes, enhanced text, output hashes,
coarse tensor shapes/dtypes/hashes and initial cold/five-warm measurements.
Square 1024×1024 and widescreen 1368×768 succeed. Fresh square repetition is
PNG-byte/pixel-identical. The README records the CLI subgraph-export discrepancy,
FP32 conditioning versus BF16 enhancement, and unresolved warm timing variance.
Initial median warm 24.101 s; final paired performance must be freshly measured.
Ignored full evidence is under `reference/local/krea2/oracle/`.

Native family and ordinary service/authoring integration are committed through
`0b166d5` on `codex/krea2-native`.
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
Native service/catalog/Recipe authoring registration now exists on the feature
branch. Thirteen geometries match exactly, including every curated aspect pair
and five freeform boundaries. The candidate freeform domain is eight-aligned,
256–2048 per side, at most 1,055,040 pixels, aspect at most 4:1. Portable contract
checks pass (225 tests; 20 native checks deselected). Live UI/desktop remain pending. All temporary Comfy probes are removed. A fresh uninstrumented reference
performance block is in progress.

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

Next: resolve the measured warm sampling regression, then live Recipe Studio and
desktop acceptance; alternate encodings and LoRAs remain required.

## Paired performance gate in progress

The first six-seed pair had failed telemetry and remains timing/correctness
only. The second pair has valid telemetry and exact pixels, but its standalone
native driver omitted the ordinary service's `cudaMallocAsync` allocator policy.
It is retained as direct-runtime diagnostic evidence, not the service timing or
memory gate. Home Lab review explicitly accepted this reclassification.

The corrected unchanged-source cold + five-warm native run verifies the same
allocator backend as the service and Comfy. Cold 51.697s; warm median 16.2174s
versus the retained Comfy 12.4890s: 29.85% slower, still outside tolerance.
Peak host working set 15,762,104,320 bytes; device-wide WDDM VRAM
16,343,769,088 bytes. All six RGB outputs and prompt enhancement remain exact.
`service-allocator-performance.json` retains every sample result and classification.

Corrected GPU profiling still isolates host transfers: native second-step HtoD
5.192 GB / 737ms versus Comfy 4.293 GB / 278ms, with equal GEMM operation counts.
CUDA-local scale, allocation sorting, and bundled-copy experiments did not help.
A naive cross-stream experiment produced invalid output and is rejected; none
of these experiments changed product source. The memory census proves Comfy registers 13,139,055,616 diffusion bytes,
within 1KB of the existing native cache (alignment). Total reference registered
memory is 17,693,575,168 bytes. Two small staging buffers preserve pixels but
regress to approximately 18 seconds and are rejected.

The narrow fix now follows the existing Klein lifetime: lazily register each
cached layer, retain unpinned operation if registration fails, synchronize and
unregister on model close. No scalar, sorting, stream, or allocator-policy change
is included. Fresh native cold 48.9854s; five-warm median 12.0598s; all six RGB
outputs remain exact. Peak host working set 15,748,165,632 bytes and VRAM
16,343,687,168 bytes. This passes against the retained reference; a fresh reference
and reversed-order sensitivity block remain pending. The existing ten Krea
checks plus a real-CUDA cache-registration/release regression pass (11 total).
`pinning-performance.json` retains measurements and the reference pin census.

Next: finish the fresh equivalent reference and order-sensitivity performance gate.
Then proceed to live authoring/desktop, alternate weights and LoRAs. Keep
allocator policy in the service; do not move process policy into family code.
