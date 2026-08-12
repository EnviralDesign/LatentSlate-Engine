# LTX 2.5 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official code audited: [`LTX-2@fd4ded7`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca)  
Official Comfy evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) and [Kitchen `9816d220021ab526e2cc1700a68b68d1b72d961c`](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)

## Decision and next slice

LTX 2.5 is the forward-looking LTX family and is not compatible with Engine's LTX 2.3 runtime. The first native slice should be the exact official **Distilled BF16 two-stage T2V** pipeline with the convolutional video VAE on Windows: half-resolution stage 1, x2 latent spatial upscaler, then a three-step full-resolution refinement. First-frame I2V reuses the same component seam after T2V. Dev, one-stage, DFR, IC-LoRA, DubIt, temporal upscaling, and duration prediction are separate products.

There is no stored first-party low-bit production artifact accepted in this audit. Upstream runtime `fp8-cast` is rejected for normal Engine recipes.

## Product/operation boundary

| Line/operation | Exact topology | Disposition |
| --- | --- | --- |
| Distilled two-stage T2V | Distilled transformer + custom Gemma4 encoder + video/audio VAEs + x2 latent spatial upscaler; half-res stage 1; three-step full-res refinement | First slice |
| Distilled first-frame I2V | same pipeline plus one image condition/preprocessing | Second slice |
| Distilled FLF | current official Comfy workflow has first/last inputs | Follow after first-frame only if demanded |
| Dev guided two-stage | Dev transformer + official distilled LoRA in stage 1 + upscaler + guided stage 2 | Separate Deferred line |
| Dev one-stage | full model, no Distilled two-stage topology | Deferred prototype line |
| DFR/detailing/temporal upscale/DubIt | extra model patches/LoRAs/condition media | Generic Comfy/Deferred |

Pinned current workflow: [LTX 2.5 FLF2V](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_flf2v.json). The implementation agent must extract exact node/subgraph settings from this JSON and cross-check the official Python pipeline; workflow and publisher code can differ, and the chosen recipe must name its parity source.

Publisher defaults/evidence: 24 fps; dimensions divisible by 32; documented baseline 121 frames at 1280×720; Distilled two-stage uses tuned sigmas, half-resolution generation, x2 latent upscaling, and short three-step full-resolution refinement. The diffusion-decoder VAE's fastest NATTEN backend is Linux/CUDA-only; Windows uses Triton or eager fallback. Start with the convolutional VAE.

## Exact component closure

Repository: [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5), gated under the LTX-2 Community License. The exact minimum Distilled closure must pin one immutable repository revision and these exact files:

| Role | Exact filename | Disposition/contract |
| --- | --- | --- |
| Transformer | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | Reference/first implementation target |
| Text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | exact trained `gemma4-12b-ltx-v1`; stock Gemma4 is not a substitute |
| Video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | initial Windows candidate |
| Audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | required for synchronized A/V |
| Spatial upscaler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | required by two-stage pipeline |
| Support | tokenizer/config/scheduler/vocoder/pipeline files selected by official code | exact allow-pattern closure required |

Optional quality challenger: `ltx-2.5-video-vae-bf16.safetensors` diffusion decoder. Deferred Dev resources include `ltx-2.5-22b-dev-transformer-bf16.safetensors` and `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`. Temporal/upscale/detailing/duration artifacts remain outside first closure.

The official quick-start Distilled closure is approximately 66 GiB, but **exact bytes and LFS SHA-256/OIDs must be resolved via the gated HF API before any declaration**. Missing identities are a publication blocker; do not infer sizes from parameter count or use mutable `main` as a pin.

## Recipe ladder and candidate contract

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `ltx-2-5.text-to-video.native-distilled-bf16` | Reference/Experimental | exact component closure above; two-stage Distilled T2V; 24 fps; explicit frames/duration; convolutional VAE; staged offload; no compile/runtime cast; synchronized A/V |
| `ltx-2-5.image-to-video.native-distilled-bf16` | Experimental follow-on | same closure + exactly one first image and exact preprocessing/index semantics |
| `ltx-2-5.first-last-frame-to-video.native-distilled-bf16` | Deferred | same line + two endpoint images; current official workflow parity |
| `ltx-2-5.text-to-video.native-dev-bf16` | Deferred Alternate | distinct Dev two-stage closure/schedule/LoRA |

Current Engine needs a new typed multi-component LTX 2.5 recipe. Roles: `transformer`, `text_encoder`, `video_vae`, `audio_vae`, `spatial_upscaler`, `support`, with optional future fixed `stage1_lora`/`video_decoder`. Stage topology and decoder choice are immutable recipe parameters, not runtime toggles. This is a schema extension.

## Loader/runtime implementation packet

Likely reuse: resource acquisition, `runtime/kit.py`, `runtime/cache.py`, manager/residency policy, output mux/provenance patterns from LTX 2.3/H3. Likely new `ltx25_recipe.py`, `runtime/ltx25.py`, `tools/ltx25.py`, request types, built-in declarations, tests. Do not subclass/alias `LTX23Runtime` as if components were compatible.

Implementation gates:

1. resolve package compatibility in isolation: official LTX code currently constrains Transformers below 5.15 because 5.15 breaks required Gemma4 construction; do not destabilize existing families;
2. validate every component/config and exact source-to-target map; custom Gemma projection and tokenizer must match;
3. represent stage 1, latent upscaler, and stage 2 as explicit state transitions with exact sigmas/steps/guidance;
4. validate video/audio VAE and vocoder sample-rate/channel contracts;
5. record actual Windows decoder/attention backend; fail if a recipe claims NATTEN on unsupported Windows;
6. no runtime FP8 casting/conversion.

Lifecycle: validate closure → stage Gemma and cache prompt conditioning on CPU → release encoder device residency → load stage-1 transformer and generate half-res joint A/V latents → release unnecessary residency → run latent spatial upscaler → run exact three-step full-res refinement → release transformer/upscaler → decode video/audio → mux/export. Pipeline fingerprint includes every component, stage schedule, decoder, frame/fps/canvas, conditioning, and package/runtime versions. Cancellation at a stage transition invalidates all downstream latent/cache state and ejects uncertain modules.

## Hardware/scientific acceptance packet

Parity case: 1280×720 requested, exact effective aligned canvas from official code, 121 frames, 24 fps, fixed prompt/seed, Distilled two-stage schedule, convolutional VAE. Because the BF16 closure may not fit, first run a clearly labeled diagnostic bucket (for example 768×448 and 49/73 frames) without claiming parity, then retain the full parity case for larger hardware/Vast.

Scenarios: cold/warm, repeated prompt, T2V→I2V→T2V later, cancellation during Gemma/stage1/upscale/stage2/video decode/audio decode/mux, malformed/missing component, decoder swap, and teardown. Assertions: exact component identities, stage timings/sigmas/steps, half/full canvas, upscaler invocation, decoder/backend, offload/residency, audio sample rate/channels, output hash. External peaks are approximate.

Corpus: dialogue/singing/music/ambience/foley/silence/stereo placement, lip/action/instrument sync, camera motion, identity/texture/temporal coherence, first-frame fidelity for I2V, and conv-VAE versus diffusion-decoder quality only with identical upstream latents where possible.

## Ordered bounded slices

1. **Next — exact Distilled T2V closure and package/environment proof.** Resolve gated file identities, support allow-list, and isolated import compatibility. Stop on unresolved license/gate or dependency incompatibility that would destabilize Engine.
2. **Distilled two-stage BF16 T2V runtime.** One recipe, convolutional VAE, explicit stage machine. Tests: component/schema/stage transitions/cancellation/provenance. Out of scope: I2V, Dev, decoder challenger, quantization.
3. **Diagnostic + cloud parity acceptance.** Smaller local case labeled non-parity; full 1280×720/121 reference on adequate hardware; synchronized A/V review.
4. **First-frame I2V reuse.** One image only, exact preprocessing/condition index, separate request/tool/tests.
5. **Diffusion-decoder VAE challenger.** Controlled decoder swap only; record actual Windows backend and require creator-visible quality win.
6. **Stop before Dev/DFR/DubIt/quantization** unless Distilled earns product value and a separate brief is approved.

## Primary sources

- [LTX-2 source `fd4ded7`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca)
- [LTX 2.5 gated components](https://huggingface.co/Lightricks/LTX-2.5)
- [Official current FLF workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_flf2v.json)
- [Comfy Kitchen LTX patchifier/kernel source](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)
