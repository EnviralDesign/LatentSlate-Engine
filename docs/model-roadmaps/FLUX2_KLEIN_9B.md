# FLUX.2 Klein 9B implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Verified Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [9B T2I workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_text_to_image_9b.json)
- [Distilled edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_9b_distilled.json)
- [Base edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_9b_base.json)
- [KV edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_9b_kv_image_edit.json)
- [ComfyUI `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [Comfy Kitchen `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4)

## Decision

Klein 9B now has a complete ordinary-Distilled Engine ladder:

- first-party BF16 Reference;
- first-party NVFP4 Recommended on qualified Blackwell;
- first-party FP8 Fallback.

The quantized ordinary T2I and one-reference edit paths are target-hardware accepted. The exact BF16 Reference remains scientifically valid but exceeds the 15.9 GiB workstation. Base and KV are separate lineages/operations and remain backburnered.

The next work is cancellation and multi-reference lifecycle, not Base, KV, or another format.

## Product and operation boundary

| Line | Operations | Official boundary | Product treatment |
| --- | --- | --- | --- |
| Distilled ordinary | T2I and ordered-reference edit | four steps, CFG 1 for Distilled edit; one active reference and disabled two-reference example | accepted NVFP4/FP8 ladder |
| Base | T2I/edit/foundation | 20 steps, CFG 5 in pinned Comfy graphs | Deferred separate line |
| 9B-KV | repeated-reference edit | two active references; `FluxKVCache` state after first applicable call | separate experiment after ordinary edit |

The pinned 9B T2I graph selects Base FP8, not Distilled. Engine’s Distilled T2I graph is a derived operation-preserving graph and must be labeled as such. No checked-in official 9B NVFP4 graph was identified.

## Exact ordinary component closure

| Closure | Transformer | Shared components | Minimum declared bytes | Disposition |
| --- | --- | --- | ---: | --- |
| Distilled BF16 | complete first-party Distilled BF16 repository/transformer | operation-matched Qwen3 8B and decoder/support | larger than local envelope | Reference |
| Distilled FP8 | first-party FP8 transformer | exact mixed Qwen3 8B, small decoder, Distilled support | 18,347,429,362 | Fallback |
| Distilled NVFP4 | first-party NVFP4 transformer | same mixed Qwen3 8B, small decoder, support | 14,675,327,882 | Recommended on Blackwell |
| Base FP8 | first-party Base FP8 | same mixed Qwen, small decoder, Base support | 18,481,646,306 | Deferred Base line |
| KV FP8 | first-party KV FP8 | mixed Qwen plus full Flux2 VAE | 18,819,998,282 | separate KV experiment |

The exact package resources and schemas at the Engine commit are authoritative:

- [9B recipes](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_recipes/klein9b)
- [resource declarations](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_resource_declarations)
- [runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/klein.py)
- [stored adapter](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/klein_stored_adapter.py)
- [quantized text loader](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/klein_quantized_text.py)

Mutable BFL discovery/license pages: [Distilled 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [Distilled FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8), [Distilled NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4), and [KV FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8). Gated acquisition and NCL/filter/manual-review obligations remain product gates.

## Current Engine proof

Current ordinary keys expose Distilled T2I and one-to-three-reference edit for first-party NVFP4, FP8, and complete BF16. The mixed Qwen loader reconstructs exact FP8/NVFP4 layers without a converted copy and requires positive dispatch deltas. Partial/persistent transformer residency, encoder offload, VAE staging, and exact resource fingerprints are implemented.

Controlled RTX 5080 baseline, seed `43301611940728`, 1024-square, three runtime-reset jobs plus three verified warm/cache-hit jobs:

| Operation/path | Runtime-cold mean | Warm mean | Approximate sampled peak VRAM |
| --- | ---: | ---: | ---: |
| T2I NVFP4 | 19.99 s | 5.71 s | 13,850 MiB |
| T2I FP8 | 32.01 s | 9.46 s | 13,713 MiB |
| one-reference I2I NVFP4 | 28.60 s | 9.57 s | 13,893 MiB |
| one-reference I2I FP8 | 36.59 s | 13.33 s | 12,817 MiB |

Each recipe was deterministic within six repeats. Hashes differ between NVFP4 and FP8, so the record proves repeatability, not perceptual equivalence. NVFP4-to-FP8-to-NVFP4 switching passed.

The complete BF16 Reference was installed but OOMed on a 1.16 GiB allocation on the 15.9 GiB GPU. The correct response is larger reference hardware, not weakening the artifact or schedule.

A compatible 82.9 MB Comfy-format LoRA was applied over first-party NVFP4 without dequantizing the base; cold/warm deterministic output and base/LoRA dispatch were recorded. That is a narrow accepted seam, not arbitrary LoRA compatibility.

## Recipe ladder

| Path | Tier | Contract |
| --- | --- | --- |
| authenticated matching BF16 | Reference | complete first-party Distilled closure; adequate hardware |
| first-party Distilled NVFP4 | Recommended | exact stored transformer, mixed Qwen, small decoder/support, native dispatch |
| first-party Distilled FP8 | Fallback | same operation/shared closure, broader hardware tier |
| compatible exact LoRA on NVFP4 | Supported narrow seam | header-proven targets and additive branch; no base dequantization |
| Base BF16/FP8/NVFP4 | Deferred separate line | no current product requirement |
| KV BF16/FP8 | Separate Experimental | explicit repeated-reference cache lifecycle |
| community KV NVFP4/ConvRot/GGUF/W4/Nunchaku | Backburner | no need before ordinary lifecycle completes |

## Loader and runtime implementation packet

The loader exists. Preserve:

- exact gated artifact identity and schema revalidation;
- complete source-to-target mapping, fused QKV, dense exceptions, aliases/tied weights, sidecars/scales;
- mixed Qwen FP8/NVFP4 materialization and dispatch deltas;
- stored transformer partial residency without dense duplicate;
- ordered prompt/reference cache keys, dimensions, preprocessing, and resource identities;
- LoRA target mapping and additive execution without base dequantization;
- poison/ejection after fallback, cancellation, materialization, or CUDA failure.

KV, when implemented, requires a distinct cache key over transformer identity, ordered reference hashes, preprocessing, dimensions, and model configuration. First-generation and cache-reuse timing must be reported separately.

## Hardware and scientific acceptance

Remaining ordinary cases:

- cancellation during Qwen load, transformer materialization, denoise, reference encode, VAE decode, and save;
- official two-reference topology and cache invalidation;
- Engine third-reference extension only after official two-reference acceptance;
- changed prompt, dimensions, and reference ordering;
- explicit teardown and Windows memory return;
- creator review of held-input FP8/NVFP4 output sets;
- BF16 reference rerun only on larger hardware.

Record native dispatch counts for every quantized module, LoRA dispatch where selected, fallback counts, exact cache state, output hash, allocator/device memory, process RSS, Windows commit, and phase timing. A Comfy/Kitchen availability banner is not execution proof.

## Ordered bounded slices

1. **Next: cancellation and clean recovery.** Existing NVFP4/FP8 ordinary keys only.
2. **Official two-reference edit.** Activate the pinned disabled topology; exact order and cache invalidation.
3. **Third-reference Engine extension.** Separate label and corpus after slice 2.
4. **Creator quality review.** Held-input NVFP4/FP8 outputs; no claim from hash determinism alone.
5. **BF16 Reference on larger hardware.** Preserve exact closure/schedule.
6. **Stop.** Base and KV remain backburnered until an explicit product requirement.

Stop on gated identity drift, fallback, dense duplication, stale ordered-reference cache, poisoned recovery, or unrecorded Base/KV semantics.
