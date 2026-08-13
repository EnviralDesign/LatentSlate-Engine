# MiniMax H3 roadmap

Last reviewed: **2026-08-13**

Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

MiniMax H3 is now a real open-weight audio-video family, but the open local model is
not the complete commercial H3 product. The truthful boundary is:

1. **H3-Base FL2VA BF16** — local 768p text-to-audio-video and zero/one/two endpoint
   image conditioning.
2. **H3-Base Ref2VA BF16** — a separate local omni-reference checkpoint accepting
   images, video, and audio references.
3. **H3-Context-IR** — hosted preprocessing/orchestration, not open-sourced.
4. **H3-Regenerate-2K** — hosted 2K regeneration, not open-sourced.

Engine currently exposes only a direct dense-BF16 FL2VA-style path: T2VA plus first,
last, or both endpoint frames. It has no package recipe, excludes the Ref2VA branch
from its compatibility bundle, and has no accepted target-workstation output matrix.
Its direct closure and compatibility bundle are pinned to the current audited official
H3 revision, but the runtime still has no target-hardware acceptance.

There is no Recommended path. The exact official FL2VA BF16 direct closure and the
matching Engine direct implementation are landed as CPU/source truth: immutable
`42ed227ee7df40d41602854ae760620d6eb651fe`, an explicit 61-file allowlist, and
144,051,143,011 bytes before filesystem overhead. The remaining gates are MiniMax H3
Community License/install review and target-workstation output acceptance. The initial
open release runs full attention; MiniMax says sparse-attention inference will arrive
later. Wait for that implementation or a first-party stored low-bit artifact rather
than inventing a conversion program.

## Evidence labels

- **Verified** — stated by MiniMax or the Engine source/catalog at the audited commit.
- **Publisher measurement** — a MiniMax capacity, quality, or serving claim; not an
  Engine result.
- **Inference** — a roadmap product judgment requiring target-workstation proof.

## Product-system and lineage boundaries

| Line / operation | Official behavior | Engine state | Comparison boundary |
| --- | --- | --- | --- |
| H3-Base FL2VA: T2VA | Zero input images, generated video plus stereo audio at 24 fps | Direct tool exists | Local BF16 768p baseline; raw prompt and Context-IR prompt are different pipelines |
| H3-Base FL2VA: first-frame or last-frame | One image may occupy either endpoint | Direct tool accepts exactly one first or last endpoint | Test endpoint semantics separately; do not assume them equivalent |
| H3-Base FL2VA: first+last | Two endpoint images | Direct tool accepts both ordered endpoints | Separate endpoint-fidelity corpus |
| H3-Base Ref2VA | Text plus up to 9 images, 3 videos, 3 audio clips, or 12 mixed files within documented duration limits | Excluded from Engine bundle; no tool | Separate checkpoint, ingress schema, memory plan, and acceptance |
| H3-Context-IR | Hosted multimodal understanding and prompt serialization | Not implemented; API-only upstream | A generated IR prompt is not apples-to-apples with a raw user prompt |
| H3-Regenerate-2K | Hosted regeneration of local/hosted 768p output to 2K | Not implemented; API-only upstream | Separate hybrid service and privacy/cost contract |

MiniMax publishes H3 for 4–15 second output, 24 fps, 32 kHz stereo audio, broad
aspect ratios, and a default 768-pixel short side. The open Base model produces 768p;
2K output requires the non-open Regenerate-2K service. Keep “open H3-Base” and “full
H3 2K product” distinct in every capability statement.

## Official local checkpoints and components

At the reviewed official GitHub commit
[`fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf),
MiniMax documents two task-specific BF16 checkpoints. Each is a self-contained
Hugging Face-style closure with processor, tokenizer, text encoder, transformer,
visual VAE, and audio VAE. The checkpoints are CFG-distilled.

| Artifact / service | Role / format | Verified scope | Disposition |
| --- | --- | --- | --- |
| [`MiniMaxAI/MiniMax-H3` FL2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Official BF16 local checkpoint family | T2VA; first-frame, last-frame, or first+last audio-video generation | **Reference for matching FL2VA operation** |
| [`MiniMaxAI/MiniMax-H3` Ref2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Official BF16 local reference checkpoint family | Multimodal reference-to-audio-video | **Reference for Ref2VA only; Deferred in Engine** |
| H3-Context-IR API | Hosted preprocessing/orchestration | Text, images, video, and audio interpreted into H3's structured context | **Fallback / separate hosted dependency** |
| H3-Regenerate-2K API | Hosted 2K regeneration | Uses 768p base output and original context | **Fallback / separate hosted dependency** |
| Official T2V/R2V workflow templates | Official workflow topology | Pinned source evidence for H3 Base operations | **Research evidence**, not Engine runtime proof |
| First-party stored FP8/NVFP4/INT8 | No exact H3 artifact verified in this review | None | **Deferred / unavailable** |
| Community quantized H3 files | Provenance and complete-path evidence not established | None admitted | **Rejected** from first ladder |

### Current FL2VA source/closure audit

The current official Hugging Face revision is
`42ed227ee7df40d41602854ae760620d6eb651fe`. It contains 280 repository files.
Its direct Diffusers FL2VA closure is the root `MiniMaxH3ModularPipeline` plus the
normal `transformer/` partition—not `transformer_ref/`; `t2va` and `fl2va` select
that normal partition, while `ref2va` selects the separate reference partition.
The original `FL2VA/` directory is a distinct SGLang/vLLM-style checkpoint topology,
not a drop-in replacement for the root Diffusers ModularPipeline closure.

The historical Engine revision `9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6` was
compared against that revision through the official immutable revision API. The two
trees have the same 280 paths; all direct-closure artifacts have the same size and
LFS SHA256. The only observed file change is `README.md` (38,479 to 38,406 bytes).
This records that the landed closure has no observed direct-weight/layout drift, but
it is not a GPU result or a license-acceptance record.

The direct Diffusers closure has an explicit install allowlist of 61 files and
144,051,143,011 bytes (134.16 GiB)
before filesystem overhead: 66,714,912,872 bytes of Qwen3-VL text encoder shards,
66,280,504,216 bytes of FL2VA transformer shards, 10,415,558,888 bytes of visual
VAE shards, and 605,429,340 bytes of audio VAE weights. Engine's CPU validator now
pins the complete tokenizer/processor/text-encoder support surface, component config
semantics, shard map/schema/dtypes, and local artifact stability before loading. It
does not read tensor payloads or claim payload SHA verification after a local copy.

Do not infer total storage from the 33B transformer parameter statement or invent
shard sizes. A package-owned resource proposal must still record the authoritative
revision, license/gate result, and all file identities in its declaration.

### Pinned official Comfy FL2VA observation packet

The official
[`video_minimax_h3_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_t2v.json)
template (workflow-templates commit `2b7f823136606344f0bccce249898d771b809aa1`,
blob `2502a910c45e08c55b37dd5d422efef6e1877304`) selects these four files for its
FL2VA text-to-video graph:

| Role | Exact graph-selected file | Format signal from filename |
| --- | --- | --- |
| FL2VA transformer | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | pruned INT8 ConvRot |
| Text/vision encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | NVFP4 AWQ |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | FP16 |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | FP32 |

This is graph and filename evidence only. The files are discovered through a gated
Comfy-Org repository and this audit has not authenticated an immutable four-file
snapshot, headers, tensor layouts, or a complete runtime-support closure. It must
therefore remain deferred low-bit research—not a substitute for the official BF16
FL2VA reference or a resource declaration candidate.

## Architecture facts that affect qualification

- **H3-Encoder:** uses the full Qwen3-VL-32B weights and hidden states from layer 50;
  H3's tokenizer and special-token configuration are required.
- **H3-Omni-Transformer:** 33B dense single-stream transformer; MiniMax says about 13B
  parameters are in AdaLN-related branches whose modulation outputs can be cached and
  need not stay loaded for inference-only deployment.
- **Visual VAE:** temporal-causal `f16t4d24`; subsequent `1 x 2 x 2` patchification
  yields effective 32x spatial and 4x temporal token downsampling.
- **Audio VAE:** independent left/right 32 kHz stereo channels represented at 40 latent
  tokens per second per channel.
- **Attention:** the model was trained with native sparse attention, but the first open
  release provides **full-attention inference only**. Sparse implementation is a
  future upstream release.

These are architectural opportunities, not an Engine memory result. The official
SGLang examples use four GPUs with Ulysses degree four. That is publisher serving
topology and gives no evidence that full H3 completes on one 16 GB RTX 5080.

## Current Engine truth

- [`tools/h3.py`](../../src/latentslate_engine/tools/h3.py)
  exposes text-to-video-with-audio and first-only, last-only, or first+last-frame
  video-with-audio.
- [`runtime/h3.py`](../../src/latentslate_engine/runtime/h3.py)
  loads one complete BF16 directory, uses model offload/native attention, and returns
  synchronized video/audio for muxing.
- Fixed Engine contract: 24 fps; 124–345 frames (about 5.17–14.38 seconds); chunk-
  aligned frame validation; default 960x544; 20 steps; no LoRA or prompt-cache path.
- The direct runtime is bound to the immutable current-closure validator revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`; the model-store acquisition-pin
  handoff is separate. It deliberately selects `t2va`/`fl2va`, never `ref2va`.
- Its FL2VA surface is exact: T2VA accepts no image; FL2VA accepts a first image, a
  last image, or both, but requires at least one endpoint. It retains 24 fps,
  `17*n+5` frame alignment, native 32 kHz channel-major stereo validation, and
  bounded cancellation checks around pipeline load, dispatch, and encoding.
- The compatibility bundle has an immutable 61-file FL2VA allowlist and will not
  acquire the original `FL2VA/`, `Ref2VA/`, or `transformer_ref/` trees. It remains
  a direct-tool install only: no package-owned recipe, resource declaration, or
  deployment profile exists.
- No Context-IR or Regenerate-2K API adapter exists.
- The direct path has no checked-in creator-accepted target-workstation result. Engine's
  project status explicitly leaves H3 target-hardware output acceptance pending.

**Proof level: Direct tool only / CPU-source contract hardened; output acceptance pending.**

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Official current FL2VA BF16 for matching T2VA/endpoint operation | **Reference** | First-party open local source at 768p |
| Official current Ref2VA BF16 | **Reference for Ref2VA** | Different checkpoint and ingress contract; not an FL2VA comparison |
| Engine direct FL2VA BF16 implementation and exact `42ed...` closure | **Experimental** | CPU/source contract, immutable revision, and 61-file/144,051,143,011-byte bundle are landed; license/install review and target-hardware output acceptance remain |
| Pinned Comfy four-file FL2VA low-bit graph | **Deferred research** | Exact gated identities, headers/tensor layouts, complete support closure, and native-dispatch proof are not yet captured |
| Context-IR + Regenerate-2K hosted workflow | **Fallback / separate service** | Needed for official full-product behavior but breaks fully local operation |
| Ref2VA Engine path | **Deferred** | Bundle excludes it; ingress, memory, and creator acceptance are unbuilt |
| Sparse-attention runtime | **Deferred / unpublished** | Upstream says it will be released later |
| First-party stored low-bit path | **Deferred / unavailable** | No exact artifact was verified |
| Community quantization zoo or runtime casts | **Rejected** | No value before current BF16 closure and feasibility are established |
| Recommended native path | **None** | No accepted single-5080 run or reproducible recipe exists |

## Small qualification ladder

Run separate ladders for T2VA, one-endpoint conditioning, and first+last conditioning:

1. **Source/CPU contract:** Engine's direct FL2VA BF16 implementation against the
   exact `42ed...` official 61-file closure. Immutable source, closure, and runtime
   validation are landed.
2. **Hardware acceptance:** target-workstation T2VA, one-endpoint, and both-endpoint
   runs at matched 768p/24 fps settings, with phase, cancellation, reuse, and
   synchronized-audio measurements.

The remaining ladder answers whether the 16 GB staged/offloaded path is viable and
valuable to creators. Do not insert a low-bit format or Ref2VA into it. Qualify Ref2VA
separately only after FL2VA earns product value.

### Deferred Comfy low-bit qualification packet

Only after the current BF16 FL2VA closure and lifecycle pass, authenticate the selected
Comfy repository revision and capture every artifact/support file. Inspect SafeTensors
headers and config layouts, map transformer and encoder tensors to the loader, and
prove actual ConvRot/NVFP4-AWQ dispatch with no silent dense/eager substitute. Keep
the pruned FL2VA transformer separate from `transformer_ref/**`, preserve the ordered
endpoint schema, and reject a mixed FL2VA/Ref2VA directory. This packet needs its own
separate baseline-versus-low-bit acceptance corpus and may not borrow a BF16 tier.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- T2VA at 4, 10, and 15 seconds; first-frame, last-frame, and first+last endpoint cases;
- 24 fps and official 768p short-side/aspect buckets, with exact frame/chunk rules;
- 32 kHz stereo validation, dialogue, singing, music, ambience, foley, impacts,
  silence, channel placement, and audio-video sync/drift;
- identity, endpoint composition, motion onset/arrival, camera motion, temporal
  coherence, prompt adherence, lip timing, and action-to-sound timing;
- raw prompt and official Context-IR output as separate corpora. Never attribute
  Context-IR gains to the local Base model;
- if Ref2VA is later evaluated: image counts 1/9, video/audio references at boundary
  durations, mixed-input maximum 12, reference identity/style/motion/audio adherence,
  and invalid-input rejection.

Record processor/tokenizer setup, Qwen text/vision encoding, cached AdaLN modulation if
implemented, full-attention backend and memory, transformer stage, visual/audio VAE,
decode, mux/export, peak VRAM/RAM, host offload and disk traffic. The log must say
**full attention** unless the released sparse implementation actually dispatched.

Cancel during closure load, encoder processing, endpoint/reference preprocessing,
denoising, visual decode, audio decode, and mux. Then prove a clean follow-up job,
correct temporary-output cleanup, no poisoned model session, and expected memory
return. Exercise T2VA to endpoint-conditioned reuse and explicit teardown.

## Material-win rule

The landed BF16 direct path is the sole implemented local baseline, not a production
recommendation. A later sparse or low-bit loader must provide at least a 20–25%
end-to-end warm win, enable an otherwise impossible creator workload on the target
machine, or materially improve cold/load stability while preserving accepted
synchronized A/V. A kernel microbenchmark or storage reduction alone is insufficient.

## Hard gaps and source conflicts

1. **Source-to-runtime compatibility:** the current direct closure is immutably
   inventoried, but its current Diffusers modular implementation remains experimental
   and needs target-hardware lifecycle proof. The audited MiniMax GitHub source is
   `fa6891f...`; its exact HF-to-GitHub release relationship is not asserted.
2. **License/install gate:** the bundle has an immutable revision and exact download
   allowlist, but user acceptance of the MiniMax H3 Community License and actual gate
   behavior still need to be documented before a normal install is approved.
3. **Single-5080 feasibility:** official serving examples use four GPUs. Full-attention
   load, host RAM, transfer cost, and runtime on 16 GB remain unknown.
4. **Open versus hosted boundary:** Context-IR and Regenerate-2K are not open. Local
   H3-Base parity with the full 2K product must not be claimed.
5. **Ref2VA omission:** Engine intentionally excludes the Ref2VA transformer branch and
   has no multimodal reference ingress contract.
6. **One-image acceptance:** Engine now exposes either first or last frame, but
   target-hardware endpoint fidelity remains unaccepted and must stay separate from
   first+last claims.
7. **Sparse-attention gap:** upstream architecture supports it, but the released
   inference path is full attention only.
8. **Settings mismatch:** Engine defaults 960x544 and 124–345 frames, while upstream
   describes 768p short-side output over 4–15 seconds. Match settings before quality
   comparison; do not assume either default is canonical for all paths.
9. **Output acceptance:** no target-workstation cold/warm/cancel/reuse or creator review
   exists for Engine's current H3 path.

## Ordered next actions

1. Review the MiniMax H3 Community License and gate behavior before approving the
   exact `42ed...` 61-file FL2VA bundle for its first normal download. The historical
   source comparison is retained above as audit evidence.
2. Build a documentation-only acquisition/VRAM plan for one exact 768p T2VA operation;
   do not download through normal Engine recipe code until the plan is approved.
3. Run the landed direct FL2VA BF16 closure on a target workstation with full
   backend/phase instrumentation, synchronized-audio review, cancellation, and reuse.
4. Decide whether FL2VA creator value and 16 GB lifecycle justify package-owned
   resources and recipes beyond the direct-tool bundle.
5. Treat Ref2VA as a separate follow-on with its own closure, ingress schema, corpus,
   memory plan, and license review.
6. Keep hosted Context-IR/2K integration separate from the local model recipe.
7. Wait for MiniMax's sparse-attention implementation or a first-party stored low-bit
   artifact before optimizing precision.
8. Treat the current Comfy low-bit graph as a separately gated research packet only:
    authenticate and pin its full closure, inspect its stored formats, then decide
    whether its lifecycle can justify a separate low-bit implementation.

## Explicit non-goals

- Do not market local H3-Base as the complete 2K H3 product.
- Do not collapse FL2VA and Ref2VA checkpoints or acceptance.
- Do not compare raw prompts with Context-IR-expanded prompts as model-only results.
- Do not invent sparse-attention support from the architecture description.
- Do not quantize at runtime or admit community formats before BF16 feasibility.
- Do not infer artifact sizes from parameter counts.
- Do not implement hosted Context-IR/Regenerate-2K inside the local loader.
- Do not call an MP4 with stereo audio synchronized-A/V acceptance.

## Primary sources

- Official H3 repository and exact reviewed commit:
  <https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf>
- Exact official H3 direct-closure revision:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe>
- Official H3 model repository (current branch):
  <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Official H3 license at the exact installed-closure revision:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE>
- Official MiniMax H3 release article:
  <https://www.minimax.io/news/minimax-h3>
- Official Diffusers H3 documentation branch linked by MiniMax:
  <https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md>
- Official Comfy H3 tutorial:
  <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- Official Comfy H3 T2V template at the reviewed immutable commit:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_t2v.json>
- Engine H3 tools/runtime and bundle: repository-relative source links above are the
  current implementation truth; use the Git commit history for an immutable snapshot.
