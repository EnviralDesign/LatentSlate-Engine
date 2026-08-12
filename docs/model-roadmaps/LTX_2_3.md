# LTX 2.3 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official Comfy evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)

## Decision and next slice

LTX 2.3 is an implemented legacy line relative to LTX 2.5. Preserve and finish it; do not broaden it. Engine already supports synchronized-audio T2V and a condition operation with a required first image and optional last image through one complete BF16 Diffusers repository. The next slice is **target-hardware BF16 acceptance** across T2V, first-frame, and first+last. Official Distilled FP8 is the only later loader worth considering.

## Product/operation boundary

| Operation | Inputs/order | Shared runtime | Distinct acceptance |
| --- | --- | --- | --- |
| T2V | prompt | same complete Distilled pipeline | text-only generation/audio sync |
| first-frame I2V | prompt + start image at index 0 | same pipeline | image preprocessing/encode, identity/motion onset |
| first+last anchored | prompt + start at 0 + end at -1 | same pipeline | endpoint fidelity, middle motion, exact order |
| Dev/V2V/audio-conditioned/IC-LoRA/upscale | different line/artifacts/contracts | none | Deferred; direct new work to LTX 2.5 unless compatibility need exists |

Pinned workflow: [LTX 2.3 first/last-frame](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json). Other checked-in 2.3 workflows (IC/ID LoRA, outpaint, style transition, subtitle/watermark operations) are specialized graphs and remain generic Comfy.

Current Engine runtime is the authoritative product truth at this audit:

- 24 fps, fixed 8 steps, guidance 1.0;
- `num_frames % 8 == 1`, 25–241 frames, 1–10 seconds;
- dimensions aligned to 32, maximum canvas area 942,080 pixels;
- Diffusers `DISTILLED_SIGMA_VALUES` and default negative prompt;
- native attention, VAE tiling on, VAE slicing off;
- sequential/model/no offload profiles; optional prompt cache; no LoRA;
- synchronized video/audio encoded to MP4.

Source: [`runtime/ltx23.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/ltx23.py).

## Exact artifacts

| Tier/path | Role | Immutable identity | Disposition |
| --- | --- | --- | --- |
| Current Reference/Experimental | complete official Diffusers folder | revision `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee`; **94,977,700,554 bytes** | package resource, gated LTX-2 Community License |
| Reference transformer | `ltx-2.3-22b-distilled-1.1.safetensors` | **46.1 GB**; SHA-256 `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc` | first-party Distilled BF16 |
| Experimental challenger | `ltx-2.3-22b-distilled-fp8.safetensors` | **29.5 GB**; SHA-256 `d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2` | first-party stored FP8 |
| Deferred | Dev FP8 | 29.1 GB; SHA-256 `28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450` | different Dev lineage |
| Deferred | Distilled NVFP4 | publisher repository says coming soon at audit; no loadable exact artifact | do not invent |

Before a componentized FP8 recipe, inventory the complete current folder into exact text encoder/tokenizer, video/audio VAEs, vocoder, scheduler/config, and support files. Transformer replacement alone is not a closure.

## Recipe ladder

Existing keys under `builtin_recipes/ltx23` are the source of truth:

| Key/operation | Tier | Contract |
| --- | --- | --- |
| `ltx-2-3.text-to-video.native-distilled-bf16` | Reference/Experimental | complete BF16 repository; 8 steps/CFG 1; exact runtime policies above |
| `ltx-2-3.image-to-video.native-distilled-bf16` | Reference/Experimental | same repository; required first image, optional last; exact index semantics |
| `ltx-2-3.text-to-video.native-distilled-fp8` | Experimental future | exact FP8 transformer + exact shared component closure; no conversion; only after BF16 acceptance |

Current complete-resource recipe needs no schema extension. A split FP8 closure needs a typed LTX component contract or carefully filtered complete-repository resource with exact transformer override; do not fake component roles in arbitrary metadata.

## Loader/runtime implementation packet

Reuse the current `LTX23Runtime`, `LTX23ConditionRuntime`, repository contract, `runtime/kit.py`, cache, manager, and tools. The current implementation already validates before/after load, keeps one lazy pipeline behind a lock, caches CPU prompt conditioning, decodes references before model allocation, removes hooks on unload, and clears CUDA cache best-effort.

Acceptance/defect work must preserve:

- start image at index 0; end image at index -1; strength 1.0;
- prompt-cache keys including default negative prompt and max sequence length 1024;
- identical pipeline fingerprint across T2V/condition only where runtime class/operation semantics permit;
- exact audio sample rate from the vocoder config;
- cancellation checks before load, after conditioning, after generation, and before/after output; a cancellation during third-party pipeline execution may only be cooperative between stages and must still poison/eject uncertain state.

For FP8 later: validate stored layout/header, exact source-to-target map and dense exceptions, prove native dispatch, and retain component staging. Fail closed on runtime cast or unsupported fallback.

## Hardware/scientific acceptance packet

Fixed T2V: prompt corpus case, seed `43301611940728`, 1280×736 effective aligned canvas (from requested 1280×720), 121 frames, 24 fps, 8 steps, guidance 1. Fixed I2V/FLF uses pinned start/end images and exact preprocessing hashes. If the Reference cannot fit, run smaller diagnostic dimensions/durations **and record the deviation**, then retain parity run for cloud hardware.

Scenarios: cold, three warm, T2V→I2V→T2V, first→first+last→first, cancellation during load/encode/generation/decode/mux, malformed repository, changed prompt and changed endpoint invalidation, explicit teardown. Assertions: repository revision/files, operation, conditions/indices, sigma schedule, cache state, offload profile, VAE tiling, frame count/fps, audio sample rate/channels/duration, output hash.

Review dialogue/music/ambience/foley/silence, lip/action sync, drift, clipping, identity, camera motion, endpoint approach, temporal warping, and audio continuity. An MP4 containing audio is not synchronized-A/V acceptance.

## Ordered bounded slices

1. **Next — BF16 T2V target-hardware acceptance.** Existing recipe/runtime only. Tests: cold/warm/cancel/recovery/audio metadata/teardown. Out of scope: FP8/new operations.
2. **First-frame and first+last acceptance.** Same repository; exact input indices; cross-operation warm reuse and endpoint corpus.
3. **Component inventory.** Documentation/catalog preparatory task: exact file roles/bytes/hashes from the 94.98 GB snapshot; no loader change.
4. **Official Distilled FP8 challenger.** Only if preserving 2.3 is valuable after LTX 2.5 comparison. Add stored loader/dispatch proof and compare per operation.
5. **Stop.** Direct new feature work to LTX 2.5 unless a 2.3 compatibility requirement is explicit.

## Primary sources

- [LTX 2.3](https://huggingface.co/Lightricks/LTX-2.3)
- [LTX 2.3 FP8](https://huggingface.co/Lightricks/LTX-2.3-fp8)
- [Current official FLF workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json)
- [Engine audited runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/ltx23.py)
