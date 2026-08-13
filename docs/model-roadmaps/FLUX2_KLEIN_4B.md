# FLUX.2 Klein 4B roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, lineage, license | Mutable first-party BFL model cards for discovery/access; exact Engine resource declarations and authenticated immutable artifact identities own accepted files |
| Saved operation topology/defaults | accepted workflow baseline [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb); current research may use `2b7f...` only after a diff |
| Node and quantized dispatch schema | accepted ComfyUI [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) and Kitchen `0.2.28` at [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) |
| Acceptance and tier | Engine public-API artifacts, native dispatch counters, deterministic repeats, switching, and creator review retained by the Klein hardware studies |

The accepted pins are historical provenance, not stale links to replace with a newer portfolio research pin.

## Product decision

Klein 4B remains the house example for an exact stored-weight image family:

- matching first-party Distilled BF16 is **Reference**;
- first-party Distilled NVFP4 is **Recommended** on qualified Blackwell;
- first-party Distilled FP8 is **Fallback**;
- Base FP8 edit is an **Alternate**, not a precision variant of Distilled.

The practical paths are already implemented and hardware-proven for 1024-square T2I and one-reference edit. The next work is cancellation/recovery and official two-reference lifecycle, not another quantization format.

## Operation boundaries

| Line/operation | Saved authority | Engine boundary |
| --- | --- | --- |
| Distilled T2I | four steps, CFG 1, Euler/Flux2 schedule | BF16, FP8, and NVFP4 recipes |
| Distilled edit | one active reference; a disabled two-reference example; nearest-exact scaling toward one megapixel | Engine accepts one to three ordered references; third is an Engine extension |
| Base edit | 20 steps, CFG 5, separate Base transformer/support and small decoder | FP8 Alternate; matching Base BF16 comparison remains a gap |
| inpaint/control/custom graphs | no bounded package-owned contract in this roadmap | generic Comfy |

Do not use Distilled output as a Base reference. Do not call the third reference official parity.

## Accepted component ladder

| Path | Stored closure | Product status |
| --- | ---: | --- |
| Distilled BF16 | package reference closure | Reference |
| Distilled NVFP4 | 10,857,495,368 bytes | Recommended on Blackwell |
| Distilled FP8 | 12,467,706,400 bytes | Fallback |
| Base FP8 edit | 12,399,885,870 bytes | Alternate |
| Base NVFP4 | 10,797,932,158 bytes | blocked until matching Base BF16 evidence |

Exact artifact revisions, hashes, header/schema fingerprints, and typed roles live in the built-in resource/recipe declarations at the audited Engine commit. Mutable BFL repository pages remain discovery/license sources and must be authenticated before new declarations.

## Engine proof preserved

The accepted study used seed `43301611940728`, 1024-square output, three runtime-reset jobs and three verified warm jobs per recipe. NVFP4 and FP8 T2I/one-reference edit completed deterministically with positive native Kitchen dispatch and no accepted dense/eager fallback. Recommended-to-Fallback-to-Recommended switching passed.

Those results prove operational repeatability and native dispatch, not perceptual equivalence across BF16, FP8, and NVFP4. Base FP8 has runtime evidence but lacks a matching Base BF16 creator comparison.

Proof level: **Hardware-proven** for ordinary Distilled T2I and one-reference edit; remaining lifecycle cells are incomplete.

## Runtime contract to preserve

- immutable file and header/schema revalidation before load;
- complete source-to-target maps, fused projections, dense exceptions, sidecars, and aliases/tied weights;
- stored quantized tensors through module assignment with no hidden dense transformer copy;
- staged Qwen encoder, transformer, and VAE residency;
- bounded prompt/reference caches keyed by exact resources, operation, ordered media, preprocessing, and canvas;
- runtime fingerprints including LoRA identity and optimization policy;
- positive native dispatch counters and zero fallback;
- poison/ejection on cancellation, materialization failure, CUDA error, NaN, or dispatch-integrity failure.

## Next bounded slices

1. **Cancellation and recovery:** cancel during encoder load, materialization, denoise, and decode; observe cleanup and a clean recovery job.
2. **Official two-reference edit:** compile the disabled upstream example into the normalized graph, preserve order/preprocessing, and test cache invalidation.
3. **Three-reference Engine extension:** separate recipe/provenance flag and corpus after two-reference parity.
4. **Base BF16 comparison:** operation-matched Base edit before any Base NVFP4 judgment.
5. **Stop:** no ConvRot/GGUF/W4/Nunchaku branch without a measured creator requirement.

Stop on graph drift, hidden fallback, stale reference cache, false availability, or poisoned recovery.
