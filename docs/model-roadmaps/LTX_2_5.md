# LTX 2.5 roadmap

Last reviewed: **2026-08-13**

Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

LTX 2.5 is the forward-looking LTX audio-video family. Lightricks now calls 2.5
its recommended model and labels LTX 2.3 legacy. Engine should therefore avoid
adding more 2.3-only features until it has qualified one bounded 2.5 path.

Do not treat LTX 2.5 as one interchangeable checkpoint. Keep these lines separate:

1. **Distilled 22B, distilled two-stage** — the recommended fast pipeline: half-resolution generation, x2 latent upscaling, then a short full-resolution refinement with the same distilled model.
2. **Dev 22B, guided two-stage** — the full model, normally combined with the official distilled LoRA in stage 1, latent spatial upscaler, and guided full-resolution stage 2.
3. **Dev one-stage** — a separate full-model quick-prototyping path, not the Distilled pipeline.
4. **Refinement and specialized pipelines** — DFR, detailing IC-LoRA, temporal upscaling, DubIt, and other conditioning modes are separate products, not flags on the Distilled baseline.
5. **T2V, first-frame I2V, and other audio/video conditioning operations** — each
   requires an operation-matched reference and corpus.

There is no Recommended Engine path yet. The first bounded implementation packet is
the current official Comfy **T2V** graph—not its first+last-frame sibling—with its
exact six-resource closure. It is a stored low-bit Comfy path and must remain
Experimental until it passes lifecycle and output acceptance. The publisher BF16
Distilled two-stage path is a separate Reference/Experimental comparison path, not
an interchangeable replacement. Its convolutional VAE is the Windows-first
feasibility candidate; the diffusion-decoder VAE may be a quality challenger later,
but its fastest NATTEN backend is Linux/CUDA-only and Windows falls back to Triton or
eager execution.

Upstream suggests runtime `fp8-cast` for memory-constrained systems. Engine should
not adopt that as a recipe: normal execution never converts model weights. Wait for
an immutable, already-quantized first-party artifact before adding a low-bit loader.

## Evidence labels

- **Verified** — stated by Lightricks or the Engine source/catalog at the audited
  commit.
- **Publisher measurement** — an upstream performance, capacity, or quality claim;
  not an Engine result.
- **Inference** — a roadmap product judgment requiring target-workstation proof.

## Scope and lineage boundaries

| Line / operation | Canonical topology | Engine state | Comparison boundary |
| --- | --- | --- | --- |
| Official Comfy T2V | Stored INT8 ConvRot transformer and projected encoder, BF16 prompt enhancer, BF16 x2 latent upscaler, video/audio VAEs | Not implemented | Primary bounded product-entry candidate; fixed six-resource graph |
| Publisher Distilled two-stage T2V | BF16 Distilled transformer, custom Gemma 4 encoder, video/audio VAEs, x2 latent spatial upscaler; half-resolution stage 1 plus short full-resolution stage 2 | Not implemented | Separate Reference/Experimental comparison path |
| Distilled first-frame I2V | Same Distilled two-stage line with image conditioning | Not implemented | Same source image, preprocessing, strength, frame count, and audio settings |
| Dev guided two-stage T2V/I2V | Dev transformer, distilled LoRA in stage 1, x2 latent spatial upscaler, guided second stage | Not implemented | Never compare its output/time directly with Distilled two-stage |
| Dev one-stage T2V/I2V | Dev transformer without the two-stage upscaling topology | Not implemented | Separate prototype/full-model operation |
| DFR / detailing IC-LoRA | Spatial and optional temporal refinement with dedicated artifacts | Not implemented | Separate refinement product and acceptance corpus |
| DubIt / audio replacement and broader condition pipelines | Specialized official pipelines | Not implemented | Separate media-ingress and synchronization contracts |
| LTX 2.3 | Earlier, explicitly legacy family with different components | Implemented separately | Components and results are not interchangeable with 2.5 |

The reviewed official code defaults to **24 fps** and uses dimensions divisible by 32. The documented baseline uses 121 frames (about five seconds) at 1280x720. The Distilled pipeline itself is two-stage: stage 1 generates at half spatial resolution using the tuned distilled sigma schedule, then the official x2 latent spatial upscaler feeds a short three-step full-resolution refinement. The guided Dev two-stage path has a different guider/sampler topology.
Keep exact stage counts, guidance, strengths, and conditioning fixed during any A/B.

## Official component surface

The official repository is gated under the
[LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.5).
Lightricks publishes one file per component so a recipe can acquire only the closure
its operation needs. The official quick-start Distilled closure is approximately
**66 GiB**; this is download footprint, not peak VRAM.

| Artifact | Role / format | Exact verified behavior | Disposition |
| --- | --- | --- | --- |
| [`ltx-2.5-22b-distilled-transformer-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors) | Distilled 22B BF16 transformer | Expected by `DistilledPipeline`, `ICLoraPipeline`, and `DubItPipeline` | **Reference / first implementation target** |
| [`ltx-2.5-22b-dev-transformer-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors) | Full Dev 22B BF16 transformer | Used by guided two-stage pipelines | **Deferred line**, after Distilled |
| [`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors) | Required text encoder plus projection | Trained checkpoint version is `gemma4-12b-ltx-v1`; stock Gemma 4 is explicitly not a substitute | **Required exact component** |
| [`ltx-2.5-video-vae-conv-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) | Convolutional video VAE | Lighter decoder; no optional NATTEN dependency | **Initial Windows candidate** |
| [`ltx-2.5-video-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-bf16.safetensors) | Diffusion-decoder video VAE | Better upstream quality claim, but slower and more memory-intensive; NATTEN fastest path is Linux/CUDA-only | **Optional quality challenger** |
| [`ltx-2.5-audio-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-audio-vae-bf16.safetensors) | Audio VAE | Required by pipelines that generate or decode audio | **Required for synchronized A/V** |
| [`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) | Latent x2 spatial upscaler | Required by Distilled and guided Dev two-stage implementations | **Required exact component** |
| [`ltx-2.5-22b-distilled-lora-450-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors) | Official distilled LoRA | Required by two-stage pipelines that run Dev in stage 1 | **Deferred with Dev line** |
| [`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors) | Temporal latent upscaler | Used by DFR temporal refinement rounds | **Deferred** |
| [`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/blob/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors) | Optional detailing IC-LoRA | Separate repository and refinement stage | **Deferred** |
| [`ltx-2.5-duration-head-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/model_patches/ltx-2.5-duration-head-bf16.safetensors) | Optional duration predictor | Lets the pipeline infer frame count from the prompt | **Deferred**; Engine should begin with explicit duration |

Artifact names are verified from Lightricks' exact repository documentation. File
sizes and hashes must be captured from a pinned Hugging Face revision before any
resource declaration is authored; this roadmap intentionally does not invent them.

### Pinned official Comfy T2V closure

The current official
[`video_ltx2_5_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_t2v.json)
template (workflow-templates commit `2b7f823136606344f0bccce249898d771b809aa1`,
blob `60683c9f3cd9c708581e1fb2e2030d987d540634`) actively selects this **six-file**
T2V closure:

| Graph role | Exact selected file | Source |
| --- | --- | --- |
| Main transformer | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 `diffusion_models/` |
| Projected text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 `text_encoders/` |
| Prompt enhancer | `gemma4_e2b_it_bf16.safetensors` | Comfy-Org/gemma-4 `text_encoders/` |
| Latent x2 upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | Lightricks/LTX-2.5 `latent_upscale_models/` |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | Lightricks/LTX-2.5 `vae/` |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | Lightricks/LTX-2.5 `vae/` |

The saved T2V graph enables prompt enhancement and defaults to 1280x720, five
seconds, and 24 fps. Those values are graph evidence only; they are not target-machine
feasibility evidence. A vocoder name appears only in stale `extra.prompt` metadata,
not as an active loader selection. The six files above are the active model closure;
ordinary package/runtime support must still be pinned separately from model weights.

This is the first T2V implementation slice. Do not substitute the related
[`video_ltx2_5_flf2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_flf2v.json)
(blob `6d3d732038151d46946538f11cc3213520980cde`) for it: FLF has a different
two-endpoint request schema and a distinct five-model graph that does not select the
latent x2 upscaler. It is a later, separate operation. Likewise, the publisher BF16
two-stage path remains a separate six-role topology whose BF16 transformer and
projected encoder are not the Comfy INT8 ConvRot artifacts above.

## Runtime and optimization surface

| Path | Reality | Status |
| --- | --- | --- |
| Exact official Comfy T2V six-resource closure | Stored INT8 ConvRot main transformer and projected encoder with BF16 enhancer/upscaler/VAEs | **Experimental entry path** after immutable acquisition and native-dispatch proof |
| Exact Distilled BF16 two-stage components + convolutional VAE | First-party, bounded topology, Windows-compatible decoder path | **Experimental target** |
| Exact Distilled BF16 + diffusion-decoder VAE | First-party quality-oriented decoder; Windows cannot use the preferred NATTEN path | **Optional experiment** |
| Dev BF16 guided two-stage + distilled LoRA + x2 upscaler | First-party higher-quality/flexible line, materially larger and more complex | **Deferred** until Distilled proof |
| Upstream runtime `fp8-cast` | A runtime conversion option, not a published recipe artifact | **Rejected** for normal Engine execution |
| First-party stored FP8/NVFP4 | No exact published LTX 2.5 production artifact was verified in this review | **Deferred / unverified** |
| Kitchen low-bit kernels | Kernel availability alone does not prove a compatible LTX 2.5 artifact or end-to-end path | **Deferred** |
| Community GGUF, INT8, FP4, Nunchaku, mixed quantizations | May exist, but no need precedes the first exact BF16 path | **Rejected** from the first ladder |
| User-owned official Comfy/LTX workflow | Valid escape hatch for operations Engine has not productized | **Fallback** |

The upstream Python stack currently pins `transformers>=5.8.0,<5.15` because
Transformers 5.15 breaks construction of the required Gemma 4 encoder. This is a real
compatibility requirement to capture in an Engine environment decision, not a model
weight issue.

## Current Engine truth at `2ba5709`

- **Not implemented or cataloged.** LTX 2.5 is absent from Engine's family registry,
  tools, runtime modules, built-in recipes, resources, bundles, and deployment profiles.
- Engine's existing `ltx23` runtime cannot be renamed or pointed at 2.5. The text
  encoder, model line, VAEs, pipeline classes, stage topology, and dependencies differ.
- Engine currently pins Torch/CUDA/Diffusers/Transformers for its existing families.
  Adding LTX 2.5 would require a deliberate compatibility decision around the official
  LTX packages and their `<5.15` Transformers constraint.
- No target-workstation load, output, cancellation, reuse, decoder-backend, or audio
  proof exists.
- No immutable resource closure, gate handling, license review, or artifact identity
  contract exists.

**Proof level: Not implemented.**

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Matching official Distilled BF16 components | **Reference** | Exact few-step source for the operation under test |
| Exact Distilled BF16 two-stage + convolutional video VAE | **Experimental challenger / entry path** | Smallest truthful Windows-first product slice |
| Distilled BF16 + diffusion-decoder VAE | **Experimental optional** | Tests creator-visible decode quality against real Windows cost |
| Dev BF16 guided two-stage | **Deferred** | Separate, larger product with different guidance and artifact complexity |
| Dev BF16 one-stage | **Deferred** | Separate full-model prototyping path; not the Distilled reference |
| DFR, temporal refinement, detailing IC-LoRA | **Deferred** | Separate refinement products |
| Runtime FP8 casting | **Rejected** | Violates no-runtime-conversion recipe policy |
| Unverified stored low-bit formats | **Deferred** | No pinned first-party artifact and backend proof |
| Community quantization zoo | **Rejected** | No creator value before BF16 feasibility is established |
| User-owned official workflow | **Fallback** | Appropriate for unsupported LTX 2.5 operations |
| Recommended native path | **None** | A 16 GB Windows path has not been demonstrated |

## Small qualification ladder

Run separate ladders for official Comfy T2V, publisher Distilled two-stage T2V, and
only later their respective image-conditioned operations:

1. **Comfy entry:** exact official T2V six-resource graph with prompt enhancement,
   graph/support fingerprints, and stored-format native-dispatch proof. It is an
   Experimental path, not a BF16 reference.
2. **Reference:** exact publisher BF16 Distilled transformer, custom Gemma encoder,
   audio VAE, selected video VAE, and x2 upscaler at upstream two-stage settings.
3. **Native candidate:** the same exact BF16 two-stage component closure in Engine with
   the convolutional video VAE, explicit staged offload, and no runtime conversion.
4. **Optional decoder experiment:** swap only to the official diffusion-decoder VAE;
   keep every generation input and latent identical where the pipeline permits.

Stop there. Do not add Dev or any additional low-bit format beyond the exact pinned
Comfy INT8 entry until the first slice or publisher Distilled comparison can complete
the fixed corpus reliably and its creator value versus LTX 2.3 is clear.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- an official-Comfy T2V packet at the saved prompt-enhance setting: cold and meaningful
  warm repeats; graph/fingerprint capture; cancellation during prompt enhancement,
  model load, denoise, decode, and mux; and a clean follow-up job after each cancel;
- separate T2V and first-frame I2V corpora at the reviewed official 24 fps;
- fixed 1280x720 / 121-frame baseline plus smaller diagnostic buckets, all aligned to
  the model's dimension and frame requirements;
- dialogue, singing, music, ambience, foley, impacts, silence, and stereo placement;
- lip timing, instrument/action synchronization, camera motion, identity, texture,
  temporal coherence, and prompt adherence;
- first-frame identity/composition retention and motion onset for I2V;
- explicit comparison of convolutional versus diffusion-decoder VAE quality using the
  same upstream latents when feasible;
- separate Distilled two-stage, Dev guided two-stage, and Dev one-stage corpora if the Dev line is later admitted.

Record package import/compile, text encoding, prompt cache, image encode, stage load and
offload, denoise, spatial/temporal upscale where applicable, video decode, audio decode,
mux/export, peak VRAM/RAM, disk traffic, and actual decoder/attention backend. On
Windows, explicitly record whether neighborhood attention dispatched to Triton or
fell back to eager; never report the Linux NATTEN expectation as a Windows result.

Cancel during model load, Gemma encoding, image encoding, denoising, stage transition,
video decode, audio decode, and mux. Follow each cancellation with a clean job and
verify model/session reuse, output cleanup, and memory return.

## Material-win rule

The first Distilled two-stage path may be justified by new creator capability—LTX 2.5 rather
than a second precision of an existing model—but it still must complete reliably and
produce accepted synchronized A/V on the target class. A second VAE or low-bit loader
needs either a 20–25% end-to-end warm win, a workload that otherwise cannot fit, or a
clear creator-visible quality improvement worth its lifecycle cost.

## Hard gaps and source conflicts

1. **16 GB feasibility is unproved.** The official quick-start closure is roughly
   66 GiB and upstream examples assume substantial offload or larger hardware.
2. **No immutable Engine closure.** Exact file sizes, hashes, revisions, gate behavior,
   and component ownership have not been captured.
3. **Windows decoder asymmetry.** The publisher's fastest diffusion-VAE decoder backend
   is NATTEN on Linux/CUDA; Windows uses Triton or eager fallback.
4. **Environment compatibility.** Official LTX code pins Transformers below 5.15,
   while Engine has its own pinned runtime. Resolve this without silently destabilizing
   existing families.
5. **Runtime-cast conflict.** Lightricks recommends `fp8-cast` as one memory option;
   Engine rejects runtime conversion for production recipes. A stored artifact is
   required instead.
6. **Pipeline multiplication.** Distilled, Dev/two-stage, DFR, IC-LoRA, DubIt, and
   duration prediction are not one acceptance result.
7. **License/gate review.** Acquisition and redistribution under the LTX-2 Community
   License must be resolved before a built-in recipe.
8. **No target output proof.** Cold/warm/cancel/reuse, synchronized audio, and actual
   backend dispatch are all unmeasured on the RTX 5080 class.

## Ordered next actions

1. Capture authenticated immutable identities, sizes, hashes, gate behavior, license,
   and support files for the official Comfy T2V six-resource graph. Do not write
   placeholder declarations against mutable `main`.
2. Build and validate the exact Comfy T2V worker/recipe contract: typed six roles,
   prompt-enhance mode, submitted graph hash, component fingerprints, and strict
   cancellation/ejection. Prove stored-format native dispatch before any tier claim.
3. Run the Comfy T2V diagnostic and adequate-hardware parity cases with 24 fps
   synchronized-A/V acceptance, cancellation, reuse, and teardown.
4. Pin one official LTX 2.5 repository revision and inventory the separate publisher
   Distilled BF16 two-stage T2V/I2V closure, including exact identities and license
   gate behavior.
5. Resolve package compatibility in an isolated qualification environment; do not
   disturb existing Engine families before the import/runtime contract is known.
6. Prototype the exact Distilled BF16 two-stage path with the convolutional VAE and
   explicit staged offload—without downloading through normal recipe code yet.
7. Run the fixed 24 fps synchronized-A/V corpus, cancellation, reuse, and teardown on
   the target workstation class.
8. Test the diffusion-decoder VAE only as a controlled component swap and record the
   actual Windows backend.
9. Compare accepted LTX 2.5 outputs and lifecycle cost with Engine's LTX 2.3 path.
10. Only after that decision, add immutable resources/recipes or explicitly defer the
   family. Consider Dev guided two-stage and Dev one-stage as separate follow-on roadmap gates.
11. Wait for a first-party already-quantized artifact before considering a low-bit
   production loader.

## Explicit non-goals

- Do not point the LTX 2.3 runtime at 2.5 assets.
- Do not compare Distilled two-stage, Dev guided two-stage, and Dev one-stage as precision variants.
- Do not implement DFR, DubIt, temporal upscaling, duration prediction, and IC-LoRA in
  the first slice.
- Do not use runtime FP8 casting in a production recipe.
- Do not adopt community quantizations before the exact BF16 two-stage path is viable.
- Do not claim Windows NATTEN performance.
- Do not call an MP4 with an audio track synchronized-audio acceptance.

## Primary sources

- Official LTX-2 code and exact reviewed commit:
  <https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca>
- Official LTX 2.5 gated component repository:
  <https://huggingface.co/Lightricks/LTX-2.5>
- Official LTX 2.5 IC-LoRA repository:
  <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler>
- Official Comfy LTX 2.5 T2V template at the reviewed immutable commit:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_t2v.json>
- Official Comfy LTX 2.5 FLF template at the reviewed immutable commit:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_flf2v.json>
- LTX technical report:
  <https://arxiv.org/abs/2601.03233>
- Engine family/runtime catalog at the audited commit:
  <https://github.com/EnviralDesign/LatentSlate-Engine/tree/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine>
