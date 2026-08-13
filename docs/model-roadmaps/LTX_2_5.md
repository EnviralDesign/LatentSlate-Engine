# LTX 2.5 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official code audited: [`Lightricks/LTX-2@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [current first-and-last-frame workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_flf2v.json)
- [ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)

## Decision

LTX 2.5 is the forward-looking LTX family, but two materially different upstream paths must remain separate:

1. the **current official Comfy first-and-last-frame graph**, which selects five model files and uses stored INT8 ConvRot for the main transformer and custom text encoder;
2. the **publisher BF16 Distilled two-stage path**, which performs half-resolution generation, x2 latent spatial upscaling, and short full-resolution refinement.

The checked-in Comfy graph is not evidence for the publisher BF16 two-stage T2V recipe. Conversely, the publisher quick-start is not an exact description of the current Comfy FLF graph.

The LTX repository is gated. Exact byte counts, hashes, and immutable Hugging Face revisions for the 2.5 artifacts were not available anonymously in this audit. They remain implementation blockers. No guessed revision or closure total is permitted.

## Product and operation boundary

| Operation/path | Exact boundary | Disposition |
| --- | --- | --- |
| Current Comfy first+last-frame A/V | exactly two endpoint images, prompt, prompt-enhance mode, duration, width/height, fps, five selected model files | Experimental after gated closure capture |
| Publisher Distilled two-stage T2V | prompt-only, synchronized A/V, half-resolution stage 1, x2 latent spatial upscaler, short full-resolution refinement | Separate Experimental path |
| Publisher Distilled first-frame I2V | same Distilled components plus first-frame conditioning | Follow-on after T2V lifecycle |
| Dev guided two-stage | Dev transformer, official Distilled LoRA in stage 1, different guidance topology | Deferred separate lineage |
| DFR, DubIt, IC-LoRA, temporal upscale, duration head | dedicated artifacts and schemas | Deferred separate products |

Do not model image input as an optional T2V field. T2V, first-frame I2V, and first+last interpolation require distinct request schemas and quality corpora.

## Closure boundaries

### Current official Comfy first-and-last-frame graph

The pinned graph actively selects these five model files:

| Role | Exact filename selected by graph | Stored/runtime role |
| --- | --- | --- |
| Main transformer | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | stored INT8 ConvRot Distilled transformer |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | diffusion-decoder video VAE |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | synchronized-audio VAE |
| Custom LTX text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | custom Gemma 4 12B plus LTX projection, stored INT8 ConvRot |
| Prompt enhancer | `gemma4_e2b_it_bf16.safetensors` | separate Gemma prompt-enhancement model |

The top-level graph also wires two sample PNG input fixtures. Those files are workflow examples, not fixed model resources. A product recipe accepts user-provided endpoints and records their hashes.

Prompt enhancement is enabled in the saved graph. A four-model “prompt enhance disabled” path would be a deliberate alternate execution mode and must not be described as full saved-workflow parity.

The mutable model discovery page is [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5). Before implementation, authenticated metadata must resolve each filename to an immutable revision, exact byte count, SHA-256/LFS or Xet identity, gate behavior, and license. Until then this is an exact filename/graph closure, not an acquisition-ready TOML closure.

### Publisher BF16 Distilled two-stage path

The publisher path requires, at minimum:

- `ltx-2.5-22b-distilled-transformer-bf16.safetensors`;
- `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`;
- one selected video VAE, initially the convolutional BF16 VAE for Windows feasibility;
- `ltx-2.5-audio-vae-bf16.safetensors`;
- `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`;
- tokenizer/config/support files required by the official code.

The BF16 path is a two-stage generation closure. Omitting the spatial upscaler and still calling it official two-stage parity is incorrect. The diffusion-decoder VAE is a controlled quality alternate; the convolutional VAE is the Windows-first feasibility candidate.

The official code pins Transformers below 5.15 at the audited commit. Environment compatibility is an implementation gate, not a model-weight property.

## Current official Comfy first-and-last-frame graph

Preserve the pinned graph’s operation semantics:

1. exactly one first and one last frame;
2. prompt and prompt-enhance Boolean;
3. duration in seconds, dimensions, seed, and frame rate;
4. five selected model files above;
5. synchronized video/audio output through `SaveVideo`;
6. exact internal condition, sampler, guider, VAE decode, and audio path represented by the subgraph.

The default example uses 1280 by 720, 24 fps, and five seconds. These are workflow defaults, not proof that the target workstation can complete them.

## Recipe ladder and candidate contracts

| Candidate key | Tier | Fixed contract |
| --- | --- | --- |
| `ltx-2-5.first-last-frame-to-video.comfy-int8-prompt-enhanced` | Experimental, blocked | exact five-model Comfy closure; two required endpoints; prompt enhancer enabled; synchronized A/V; no runtime conversion |
| `ltx-2-5.text-to-video.native-distilled-bf16-two-stage` | Reference/Experimental, blocked | exact BF16 Distilled transformer, custom encoder, selected VAE, x2 spatial upscaler, official two-stage schedule |
| `ltx-2-5.image-to-video.native-distilled-bf16-two-stage` | Deferred follow-on | reuse accepted BF16 T2V components plus required first-frame conditioning |
| Dev/DFR/DubIt keys | Deferred | separate lineages and artifacts |

Both candidate keys are blocked until the gated artifact manifest is exact. Do not author size-zero or mutable-`main` resource declarations as placeholders.

## Loader and runtime implementation packet

### Shared requirements

Reuse immutable resource acquisition, typed component recipes, runtime fingerprints, byte-bounded caches, manager poison/ejection, video/audio output, and LTX 2.3 lifecycle lessons.

Each path needs a distinct typed recipe. The Comfy FLF contract needs roles for main transformer, video VAE, audio VAE, custom text encoder, and prompt enhancer. The publisher two-stage contract additionally needs the latent spatial upscaler and support/config closure.

### Comfy FLF path

A Comfy-first worker may be justified if it pins the exact executable checkout, hardlinks only validated files, disables custom nodes, records the submitted graph hash, and unloads on cancellation/failure. Header validation must prove both INT8 ConvRot artifacts and the two VAE schemas before the worker starts.

### Publisher BF16 two-stage path

A native path must reproduce official stage ownership:

1. prompt/custom encoder;
2. half-resolution stage 1;
3. x2 latent spatial upscale;
4. short full-resolution refinement;
5. video and audio decode;
6. mux/export.

Do not use upstream runtime `fp8-cast` in a production recipe. Wait for stored artifacts if a low-bit native path is later needed.

Cancellation during either stage or stage transition must eject poisoned state. Runtime keys include every component, selected VAE, stage schedule, dimensions/frames/fps, attention backend, offload, compile policy, and prompt-enhance mode.

## Hardware and scientific acceptance

### Comfy FLF packet

Use two fixed endpoint images with content hashes, prompt, prompt enhancer enabled, seed, 24 fps, and a smaller diagnostic canvas/duration first. Retain the exact 1280 by 720, five-second workflow setting for adequate-hardware parity rather than silently shrinking it.

Test endpoint fidelity, middle-frame motion, identity, shot continuity, lip/action sync, audio duration/channels, cold/warm, FLF to another family to FLF switching, cancellation during prompt enhancement/model load/denoise/decode/mux, malformed each-resource case, and teardown.

### Publisher two-stage packet

Use a fixed T2V prompt, 121 frames, 24 fps, official two-stage schedule, and convolutional video VAE. Record stage 1, upscaler, refinement, video/audio decode, mux, peak VRAM/RAM, Windows commit, disk/PCIe traffic, and exact backend. Add first-frame I2V only after T2V lifecycle passes.

A convolutional-versus-diffusion-decoder VAE comparison changes only the decoder artifact and backend. Windows must record whether any neighborhood-attention path actually dispatched; Linux NATTEN claims are not Windows measurements.

## Ordered bounded slices

1. **Next: authenticated immutable closure capture.** Resolve all five Comfy files and all BF16 two-stage files separately. Record bytes, hashes, licenses, gating, and support files. Stop if any selected file is unavailable.
2. **Comfy FLF structural packet.** Typed five-role recipe, exact graph hash, prompt-enhance mode, endpoint schema, header tests, isolated-worker lifecycle. No hardware promotion yet.
3. **Comfy FLF diagnostic acceptance.** Smaller diagnostic settings plus retained exact parity case for adequate hardware.
4. **Publisher BF16 two-stage T2V.** Separate environment and recipe; convolutional VAE first; exact x2 upscaler mandatory.
5. **First-frame I2V reuse.** Only after BF16 T2V passes.
6. **Stop.** Dev, DFR, DubIt, duration head, temporal upscale, and runtime casting remain outside these slices.
