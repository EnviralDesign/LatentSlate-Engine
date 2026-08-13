# MiniMax H3 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official source audited: [`MiniMax-AI/MiniMax-H3@fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [T2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_t2v.json)
- [I2V/endpoint workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_i2v.json)
- [R2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_r2v.json)
- [ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)

## Decision

MiniMax H3 has two open local checkpoint families and two hosted product layers:

1. FL2VA Base for text-to-audio-video and zero/one/two endpoint images;
2. Ref2VA Base for multimodal reference-to-audio-video;
3. hosted Context-IR preprocessing;
4. hosted Regenerate-2K.

Engine currently implements only a dense BF16 FL2VA-style path: T2VA plus required first and optional last frame. It excludes `transformer_ref/**`, has no package-owned recipe/resource, and targets an older pinned repository snapshot. The next work is to re-pin the current exact FL2VA closure and qualify it, not add a low-bit format.

The open release uses full-attention inference. Architecture references to sparse attention are not proof of a released sparse runtime.

## Product and operation boundary

| Operation | Official boundary | Engine state | Disposition |
| --- | --- | --- | --- |
| FL2VA T2VA | prompt, no image; 24 fps stereo audio-video | direct runtime exists | First acceptance line |
| FL2VA one endpoint | first-only or last-only semantics are officially distinct | Engine exposes first-frame semantics when one image is supplied | Schema gap; qualify explicitly |
| FL2VA first+last | two ordered endpoint images | direct runtime exists | Separate endpoint corpus |
| Ref2VA | images, video, audio, or mixed references within published limits | absent; separate transformer branch | Deferred separate checkpoint/schema |
| Context-IR | hosted multimodal prompt/context preparation | absent | Hosted Fallback, separate provenance |
| Regenerate-2K | hosted 2K regeneration | absent | Hosted Fallback, not local Base capability |

Do not market local 768p H3 Base as the complete hosted 2K product. Raw prompt and Context-IR-expanded prompt are different operations.

## Official and current Comfy closure

The current Comfy T2V/I2V graph uses four model artifacts:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`;
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- `minimax_h3_video_vae_fp16.safetensors`;
- `minimax_h3_audio_vae_fp32.safetensors`.

The R2V graph uses the reference-specific transformer branch and must not be mixed with FL2VA.

These filenames are exact workflow evidence, but the Hugging Face repository is gated and the anonymous audit did not resolve a coherent immutable four-file snapshot with exact byte counts and hashes. The mutable discovery pages are [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). A built-in resource packet must authenticate and lock every selected file before implementation.

The current Comfy graph is a low-bit/pruned execution path, not a BF16 reference. Engine’s existing runtime instead loads a complete BF16 Diffusers repository. Keep these paths scientifically separate.

## Current Engine truth

The authoritative runtime is [`runtime/h3.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/h3.py), with tools in [`tools/h3.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/tools/h3.py).

Current behavior:

- complete BF16 Diffusers directory only;
- Modular Diffusers workflows `t2va` and `fl2va`;
- 24 fps;
- 124 through 345 frames, mapped to Engine’s `17k+5` grid;
- default 960 by 544;
- 20 default steps, allowed 1 through 30;
- auto CPU offload through `ComponentsManager`;
- native attention;
- VAE tiling/slicing off;
- no LoRA and no conditioning cache;
- synchronized video/audio MP4 output;
- first image required in FL2VA; last image optional.

The compatibility bundle pins an older HF revision and excludes `transformer_ref/**`. It is FL2VA-only, not Ref2VA support. No package recipe, resource declaration, profile, or target-output acceptance exists.

Proof level: direct runtime only; output acceptance pending.

## Recipe ladder

| Path | Tier | Boundary |
| --- | --- | --- |
| current official FL2VA BF16 | Reference | exact current first-party complete repository, matching operation and settings |
| existing older-pinned Engine BF16 | Experimental incumbent | compare only after exact closure/config differences are known |
| re-pinned current BF16 FL2VA | Experimental migration | same direct runtime after repository validator and source drift are updated |
| current Comfy four-file low-bit FL2VA | Deferred challenger | exact gated identities and layout/native-dispatch proof required; not a substitute for BF16 re-pin |
| official Ref2VA BF16 | Reference for Ref2VA only | separate transformer and multimodal ingress |
| Context-IR/Regenerate-2K | Fallback services | hosted, separate privacy/cost/version contract |
| sparse attention | Deferred unavailable | no released exact implementation proven in this audit |

No Recommended local path exists.

## Loader and runtime implementation packet

### Re-pin packet

1. authenticate and inventory the current first-party FL2VA BF16 repository;
2. compare model index, component classes, configs, tokenizer/processor files, schemas, and weight identities against Engine’s old pinned closure;
3. update a future implementation only after the difference report distinguishes byte-identical, compatible, and changed components;
4. retain `transformer_ref/**` exclusion for FL2VA and reject a mixed FL2VA/Ref2VA directory;
5. preserve complete-repository validation and post-plan revalidation.

### Lifecycle packet

Prompt/text/vision encoding, transformer, video VAE, audio VAE, and mux ownership must be explicit. Full attention must be reported unless a released sparse backend actually dispatches. Cancellation during load, encoder, endpoint preprocessing, denoise, video decode, audio decode, or mux ejects the pipeline and clears partial outputs.

### Future low-bit packet

Do not infer compatibility from filename or Kitchen kernel availability. Require exact headers, layouts, tensor maps, text-encoder AWQ/NVFP4 semantics, pruned-transformer provenance, native dispatch counts, and zero dense/eager fallback.

## Hardware and scientific acceptance

Reference settings remain exact official 768p-class output, 24 fps, and operation-matched duration/frame rules on adequate hardware. The local RTX 5080 diagnostic may use a smaller canvas/duration only when labeled diagnostic; it must not replace the cloud/reference result.

Corpus:

- T2VA at short, medium, and long durations;
- first-only, last-only, and first+last endpoint cases;
- dialogue, singing, music, ambience, foley, impacts, silence, stereo placement;
- identity, endpoint composition, motion onset/arrival, camera movement, lip timing, action-to-sound timing, and A/V drift;
- raw prompt and Context-IR prompt as separate corpora.

Required scenarios: runtime-cold plus meaningful warm repeats; T2VA to FL2VA to T2VA switching; cancellation in every phase; malformed repository/component; explicit teardown. Record exact repository identity, workflow, full/sparse attention state, frame mapping, endpoint order, audio sample rate/channels, phase timing, VRAM/RAM/Windows commit, output hashes, and creator review.

## Ordered bounded slices

1. **Next: current FL2VA BF16 closure and drift report.** Authentication/metadata only; no quant work. Stop if the exact release cannot be locked.
2. **Re-pinned BF16 direct runtime.** Update repository contract/config expectations while preserving operation behavior and fail-closed FL2VA-only scope.
3. **Target hardware/reference acceptance.** Local diagnostic plus exact adequate-hardware parity, synchronized A/V, cancellation, reuse, teardown.
4. **Last-frame-only semantics.** Add only after FL2VA base acceptance and an explicit request schema.
5. **Ref2VA separate packet.** New checkpoint, ordered multimodal ingress, closure, memory plan, and corpus.
6. **Low-bit/sparse only after first-party evidence exists.** No community format zoo or runtime conversion.

Stop on an unresolved gated identity, FL2VA/Ref2VA mixing, hidden hosted preprocessing, unreported full-attention fallback, or poisoned cancellation recovery.
