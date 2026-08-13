# LTX 2.3 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [first/last-frame workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json)

## Decision

LTX 2.3 is Engine’s first synchronized audio-video family, but it is now a legacy line relative to LTX 2.5. Preserve and finish the implemented Distilled path; do not broaden it into a format or feature program.

Engine already exposes:

- Distilled T2V;
- required-first-frame I2V;
- required-first plus optional-last anchored video;
- synchronized video and audio from one complete BF16 Diffusers repository.

The next work is target-hardware acceptance. Official Distilled FP8 is the only later loader worth considering, and only after exact component ownership is separated from the current 94.98 GB whole-folder resource.

## Product and operation boundary

| Operation | Inputs/order | Shared runtime | Distinct acceptance |
| --- | --- | --- | --- |
| T2V | prompt | complete Distilled pipeline | text-only generation and A/V sync |
| first-frame I2V | prompt plus start image at index 0 | same pipeline | image preprocessing, identity, motion onset |
| first+last anchored | prompt plus start at 0 and end at -1 | same pipeline | endpoint fidelity and middle motion |
| Dev/V2V/audio-conditioned/IC-LoRA/upscale | different lineage, artifacts, or request | none | Deferred; prefer LTX 2.5 for new work |

The pinned FLF template is exact evidence for its graph only. Specialized 2.3 workflows such as IC/ID LoRA, outpaint, style transition, subtitle, and watermark remain generic Comfy.

## Current Engine contract

The authoritative implementation is [`runtime/ltx23.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/ltx23.py).

Fixed behavior:

- 24 fps;
- 8 steps;
- guidance 1.0;
- Diffusers `DISTILLED_SIGMA_VALUES` and default negative prompt;
- frame count `8n+1`, 25 through 241 frames, one through ten seconds;
- dimensions aligned to 32, maximum canvas area 942,080 pixels;
- native attention;
- VAE tiling on, slicing off;
- sequential/model/no-offload profiles;
- optional prompt cache;
- no LoRA support;
- synchronized video/audio MP4 output.

Start/end images are converted to `LTX2VideoCondition` objects at indices 0 and -1. A generic unordered image list is not equivalent.

## Exact artifacts

| Path | Identity | Disposition |
| --- | --- | --- |
| Current complete BF16 repository | revision `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee`; 94,977,700,554 bytes | package Reference/Experimental incumbent |
| Distilled BF16 transformer | `ltx-2.3-22b-distilled-1.1.safetensors`; 46.1 GB; SHA-256 `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc` | first-party Reference transformer |
| Distilled FP8 transformer | `ltx-2.3-22b-distilled-fp8.safetensors`; 29.5 GB; SHA-256 `d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2` | only later Experimental challenger |
| Dev FP8 | 29.1 GB; SHA-256 `28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450` | Deferred different lineage |
| Distilled NVFP4 | no exact published Distilled artifact verified | Deferred; do not invent |

The mutable discovery pages are [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) and [LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8). A future componentized recipe must resolve one coherent immutable revision for tokenizer, text encoder, connectors, video/audio VAEs, vocoder, scheduler, and support files. Replacing the transformer does not eliminate the rest of the closure.

Current package evidence:

- [T2V recipe](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_recipes/ltx23/ltx-2-3-text-to-video-native-distilled-bf16.toml)
- [I2V recipe](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_recipes/ltx23/ltx-2-3-image-to-video-native-distilled-bf16.toml)
- [resource declaration](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/builtin_resource_declarations/ltx23-distilled-bf16.toml)
- [tools](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/tools/ltx23.py)

## Recipe ladder

| Key/path | Tier | Contract |
| --- | --- | --- |
| existing T2V and I2V native BF16 keys | Reference/Experimental | complete immutable BF16 repository, fixed eight-step Distilled schedule |
| future componentized Distilled BF16 | Reference migration | exact same bytes and behavior with explicit component roles |
| future official Distilled FP8 | Experimental challenger | exact stored FP8 transformer plus identical remaining components; no runtime conversion |
| Dev/IC-LoRA/V2V/upscale | Deferred | separate products and corpora |

No Recommended path exists until synchronized-A/V, cancellation, and target-memory acceptance pass.

## Loader and runtime implementation packet

The first packet is acceptance, not code. Reuse current family tools/runtime, complete-repository validator, prompt cache, video output, manager, and public hardware harness.

Before FP8 work:

- inventory the 94.98 GB repository into exact component roles and fingerprints;
- prove the current complete folder contains the operation-required closure and no accidental Dev substitution;
- inspect the FP8 SafeTensors header/layout and map every transformer tensor;
- retain text encoder, connectors, VAEs, vocoder, scheduler, and tokenizer identities;
- fail closed on runtime conversion, mixed Dev/Distilled components, unsupported attention/offload, or partial folder substitution.

A later FP8 loader must record actual backend dispatch and preserve the same operation semantics. Cancellation during materialization or any generation phase ejects the runtime.

## Hardware and scientific acceptance

Fixed T2V case: 960 by 544, 121 frames, 24 fps, 8 steps, guidance 1, seed `43301611940728`, synchronized video/audio. Fixed conditioned cases reuse the same request and add one start image, then start plus end image.

Add 25- and 241-frame diagnostics where practical. Corpus covers dialogue, music, ambience, foley, silence, speech lip timing, impacts, footsteps, instrument performance, camera motion, identity, temporal coherence, first-frame fidelity, end-frame approach, and audio drift.

Required scenarios:

- runtime-cold and three changed-prompt/seed warm runs;
- T2V to first-frame to first+last to T2V reuse;
- prompt-cache hit/miss and changed-image invalidation;
- cancellation during load, text encode, image encode, denoise, VAE video decode, audio decode, mux, and save;
- malformed repository/config cases;
- explicit teardown.

Record exact repository identity, phase timing, prompt cache, condition order, frames/fps/duration, audio sample rate/channels, VRAM/RAM/Windows commit, disk traffic, output hashes, and creator review. An MP4 with an audio stream is not by itself synchronized-A/V acceptance.

## Ordered bounded slices

1. **Next: BF16 T2V acceptance.** Existing key only; fixed synchronized-A/V corpus, cold/warm, cancellation, teardown.
2. **Conditioned reuse acceptance.** First-frame and first+last in one process; exact condition indices and cache invalidation.
3. **Component inventory.** Convert the whole-folder understanding into exact role/file identities without changing runtime behavior.
4. **Official Distilled FP8 only if 2.3 still has product value.** Header map, stored loader, native dispatch, same corpus.
5. **Stop.** Direct new operation investment to LTX 2.5 unless a 2.3 compatibility requirement is documented.

Stop on A/V drift, condition-order ambiguity, repository mismatch, cancellation poison, or a hidden schedule change.
