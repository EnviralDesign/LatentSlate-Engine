# FLUX.2 Klein 4B implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Verified Comfy source set:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [T2I workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_text_to_image.json)
- [Distilled edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_4b_distilled.json)
- [Base edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_4b_base.json)
- [ComfyUI `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [Comfy Kitchen `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4)
- [ConvRot conversion source `1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341bb8a4e46f161a978b5faa2412d8c39c768/quant_int8_convrot.py)

## Decision

Klein 4B remains the house reference for an exact stored-weight image family. The ordinary Distilled line is implementation- and target-hardware-proven:

- matching BF16 is Reference;
- first-party NVFP4 is Recommended on qualified Blackwell;
- first-party FP8 is Fallback;
- Base FP8 editing is an Alternate, not a precision variant of Distilled.

The next work is lifecycle coverage and official reference multiplicity, not another loader.

## Product and operation boundary

| Line/operation | Exact official boundary | Engine boundary | Disposition |
| --- | --- | --- | --- |
| Distilled T2I | four steps, CFG 1, Euler, Flux2 scheduler, Qwen3 4B, full Flux2 VAE | matching BF16/FP8/NVFP4 keys | Accepted ladder |
| Distilled edit | one active reference; disabled two-reference example; references scaled toward one megapixel | one to three ordered references | one accepted; two must match official example; three is Engine extension |
| Base T2I/edit | 20 steps, CFG 5, separate Base transformer/support and small decoder | Base FP8 edit exists | Alternate line; matching Base BF16 gap remains |
| inpaint/control/custom graphs | separate operation topology | absent | Generic Comfy |

Do not compare Base and Distilled as precision variants. Do not call Engine’s third reference official Comfy parity.

## Exact component closures

A valid A/B changes only the transformer within one lineage/operation.

| Closure | Transformer | Shared components | Declared bytes | Disposition |
| --- | --- | --- | ---: | --- |
| Distilled BF16 | first-party BF16 transformer | exact Qwen3 4B, full Flux2 VAE, Distilled support shell | package reference closure | Reference |
| Distilled FP8 | first-party stored FP8 | same shared components | 12,467,706,400 | Fallback |
| Distilled NVFP4 | first-party stored NVFP4 | same shared components | 10,857,495,368 | Recommended on Blackwell |
| Base FP8 edit | first-party Base FP8 | same Qwen3 4B, small decoder, Base support shell | 12,399,885,870 | Alternate |
| Base NVFP4 | first-party Base NVFP4 | same Base shared components | 10,797,932,158 | blocked until matching Base BF16 reference |

The exact package declarations at the audited Engine commit are the implementation source of truth:

- [built-in 4B recipes](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_recipes/klein4b)
- [resource declarations](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_resource_declarations)
- [runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/klein.py)
- [stored adapter](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/klein_stored_adapter.py)

Mutable BFL model pages are discovery/license sources, not new locks: [4B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [4B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B-fp8), and [4B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4).

## Current Engine proof

Current keys:

| Operation/key family | Tier | Accepted evidence |
| --- | --- | --- |
| Distilled BF16 T2I/I2I | Reference | controlled 1024-square cold/warm baseline |
| Distilled NVFP4 T2I/I2I | Recommended | native SM120 dispatch, deterministic output, cold/warm, family switching |
| Distilled FP8 T2I/I2I | Fallback | deterministic output and switching |
| Base FP8 I2I | Alternate | public-API generation, but no matching Base BF16 quality comparison |

The controlled benchmark used seed `43301611940728`, 1024-square, three runtime-reset jobs and three verified warm/cache-hit jobs per recipe. Warm NVFP4 T2I averaged about 2.78 seconds versus 3.72 seconds FP8 and 4.76 seconds BF16. Warm one-reference I2I averaged about 5.37 seconds NVFP4, 6.31 seconds FP8, and 7.25 seconds BF16. Device-wide VRAM sampling is approximate; retained manifests are the provenance source.

This proves deterministic operability and native dispatch within each recipe, not perceptual equivalence across precisions.

## Recipe ladder

| Path | Tier | Contract |
| --- | --- | --- |
| matching first-party BF16 | Reference | same lineage, operation, encoder, VAE, support, schedule, dimensions, references |
| first-party Distilled NVFP4 | Recommended | exact stored artifact, staged residency, native Kitchen dispatch, no runtime conversion |
| first-party Distilled FP8 | Fallback | exact stored artifact and same shared closure |
| Base FP8 edit | Alternate | separate 20-step/CFG-5 line |
| community ConvRot | Deferred/Experimental only | one bounded experiment after licensing and only if accepted NVFP4 leaves a gap |
| other format zoo | Rejected | no creator requirement |

## Loader and runtime implementation packet

The loader exists. Preserve:

- exact SafeTensors schema fingerprints and file identity revalidation;
- source-to-target mapping, fused projections, dense exceptions, sidecars, aliases/tied weights;
- quantized tensors through assignment, with no hidden dense transformer copy;
- staged encoder, transformer, and VAE residency;
- byte-bounded prompt/reference caches keyed by resource and ordered media identity;
- runtime fingerprint including every fixed component and optimization;
- native dispatch counters and zero fallback events;
- poison/ejection after materialization, CUDA, cancellation, or integrity failure.

Do not widen `stored_quant.py` into a permissive format switch merely to add a community artifact.

## Hardware and scientific acceptance

Remaining cases:

- cancellation during encoder load, transformer materialization, denoise, and VAE decode, followed by a clean job;
- official two-reference topology with exact order/preprocessing;
- Engine third-reference extension only after two-reference acceptance;
- changed-reference cache invalidation;
- A-to-B-to-A switching across Recommended/Fallback/Reference;
- explicit teardown and Windows memory return;
- creator review of held-input BF16/FP8/NVFP4 output sets.

Provenance must distinguish runtime-cold from process-cold, cache replay from independent generation, official reference counts from Engine extensions, and exact backend dispatch from startup capability banners.

## Ordered bounded slices

1. **Next: cancellation and clean recovery.** Existing Recommended/Fallback/Reference keys; no new resources or formats.
2. **Official two-reference edit.** Activate the pinned disabled example; exact ordered hashes and cache behavior.
3. **Third-reference Engine extension.** Separate corpus and explicit provenance after slice 2.
4. **Matching Base BF16 edit Reference.** Required before Base NVFP4 quality claims.
5. **Stop.** No additional format or lineage without a measured creator requirement.

Stop on fallback, dense duplication, stale reference cache, poisoned recovery, or an unrecorded schedule/preprocessing change.
