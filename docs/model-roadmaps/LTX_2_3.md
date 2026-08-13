# LTX 2.3 roadmap

Last reviewed: **2026-08-13**

Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

LTX 2.3 is Engine's first native synchronized audio-video family, but it is now a
**legacy upstream line** relative to LTX 2.5. Preserve and finish its existing path;
do not expand it into a broad optimization program.

Keep two model lines and three conditioning operations distinct:

1. **Distilled 22B** — eight steps, CFG 1; this is Engine's current line.
2. **Dev 22B** — flexible/full model for guided or training-oriented workflows; not a
   substitute for Distilled.
3. Distilled **T2V**, **first-frame I2V**, and **first+last-frame conditioned video**
   each need operation-matched references and acceptance.

Engine currently installs a 95.0 GB complete BF16 Diffusers repository and exposes
T2V plus a condition pipeline requiring a first frame and optionally accepting a last
frame. The exact runtime contract is strong, but target-hardware output acceptance is
unfinished. The only worthwhile challenger is Lightricks' official **Distilled FP8**
transformer. Distilled NVFP4 is still advertised as “coming soon”; do not build a
loader around an unpublished artifact.

This roadmap has no Recommended path yet. Finish BF16 acceptance, replace the coarse
complete-folder substitution with exact component ownership, then test official FP8.

## Evidence labels

- **Verified** — stated by Lightricks or the Engine source/catalog at the audited
  commit.
- **Publisher measurement** — an upstream performance/memory claim, not an Engine
  result.
- **Inference** — a roadmap product judgment requiring target-workstation validation.

## Scope and lineage boundaries

| Line / operation | Canonical behavior | Engine state | Comparison boundary |
| --- | --- | --- | --- |
| Distilled T2V | 8 steps, CFG 1, synchronized video and stereo audio | Recipe/runtime exists | Primary acceptance line |
| Distilled first-frame I2V | First image conditions video/audio generation | Recipe/runtime exists | Compare only with same first image and preprocessing |
| Distilled first+last | First image required, final image optional; anchored condition pipeline | Runtime exists through I2V tool | Separate from ordinary first-frame I2V |
| Dev 22B | Full/flexible model, guided and trainable | Not implemented | Separate future line; never use as a precision reference for Distilled |
| Video-to-video, audio-conditioned, IC-LoRA, upscaler, and other LTX operations | Additional official ecosystem capabilities | Not implemented | Deferred; separate schemas and acceptance corpora |

Lightricks describes LTX 2.3 as a 22B DiT-based joint audio-video foundation model.
Its official cards require dimensions divisible by 32 and frame counts satisfying
`8n + 1`. The weights are gated under the
[LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.3).
License and redistribution review are recipe gates.

## Published artifacts and topology

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) | Official Distilled/Dev BF16 repository and components | First-party, gated, LTX-2 Community License | **Reference source** |
| [`ltx-2.3-22b-distilled-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors) | Official Distilled BF16 transformer | **46.1 GB**; SHA-256 `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc` | **Reference transformer** |
| Engine complete BF16 resource | Complete official Diffusers directory | **94,977,700,554 bytes**, revision `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` | **Experimental incumbent** |
| [`ltx-2.3-22b-distilled-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-distilled-fp8.safetensors) | Official Distilled FP8 transformer | **29.5 GB**; SHA-256 `d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2` | **Experimental challenger** |
| [`ltx-2.3-22b-dev-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-dev-fp8.safetensors) | Official Dev FP8 transformer | **29.1 GB**; SHA-256 `28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450` | **Deferred**, different line |
| [`Lightricks/LTX-2.3-nvfp4`](https://huggingface.co/Lightricks/LTX-2.3-nvfp4) | QAD NVFP4 repository | Dev NVFP4 published; **Distilled NVFP4 listed as coming soon** | **Deferred** for current Distilled product line |
| Community GGUF, INT8, Nunchaku, mixed W4, custom FP8 casts | Various | Artifacts may exist, but provenance/layout and creator value do not beat first-party FP8 | **Rejected** from the initial ladder |

The transformer size is not the complete pipeline size. Qualification must inventory
the exact text encoder, video/audio VAE components, vocoder/audio decoder, scheduler,
tokenizer, and condition-pipeline files from one coherent revision. Replacing only the
transformer does not make the remaining 95 GB closure disappear.

### Pinned official Comfy FLF investigation packet

The official
[`video_ltx2_3_flf2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json)
template (workflow-templates commit `2b7f823136606344f0bccce249898d771b809aa1`,
blob `377ebd149ed9659a11d3828e922127ab10d55b04`) is useful topology evidence for a
later first+last-frame packet: it selects the official Distilled FP8 transformer and
the Comfy Gemma text encoder, with one image at frame zero and one at the final frame.
It is **not** a complete immutable acquisition closure and is not evidence that the
existing Engine BF16 runtime should change its component set or schedule.

If the FP8 challenger is admitted, first capture authenticated immutable identities
for the transformer, text encoder, video/audio VAE, vocoder, tokenizer, scheduler, and
the exact condition support files. Then represent the endpoint pair as an ordered,
typed request—not an unordered image list—and prove stored-FP8 native dispatch,
cancellation/recovery, and synchronized A/V separately from T2V and first-frame I2V.

## Current Engine truth at `2ba5709`

- **Package-owned recipes exist:**
  [`T2V`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/builtin_recipes/ltx23/ltx-2-3-text-to-video-native-distilled-bf16.toml)
  and
  [`I2V`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/builtin_recipes/ltx23/ltx-2-3-image-to-video-native-distilled-bf16.toml)
  bind the same complete BF16 resource.
- **Acquisition is immutable but coarse:**
  [`ltx23-distilled-bf16.toml`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/builtin_resource_declarations/ltx23-distilled-bf16.toml)
  pins the 94.98 GB official repository revision.
- **Runtime operations are explicit:**
  [`tools/ltx23.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/ltx23.py)
  and
  [`runtime/ltx23.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/runtime/ltx23.py)
  provide synchronized-audio T2V and a condition pipeline with required first and
  optional last images.
- **Fixed Engine behavior:** 24 fps; 8 steps; CFG 1; 1–10 seconds; 25–241 frames;
  `(frames - 1) % 8 == 0`; dimensions aligned to 32; maximum pixel area 942,080;
  native attention; VAE tiling; prompt caching; sequential/model/no offload policies.
- **No low-bit loader or LoRA path.** The current runtime accepts the complete BF16
  folder only and does not convert weights at runtime.
- **Proof level: cataloged / runtime-structured.** Repository status says H3/LTX and
  native Wan target-hardware output acceptance still needs the planned hands-on pass.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Matching Distilled BF16 operation | **Reference** | First-party eight-step source for T2V or the same conditioning path |
| Current complete-folder BF16 Engine path | **Experimental incumbent** | Exact recipe/runtime exists, but footprint and target output acceptance remain |
| Official Distilled FP8 transformer with exact shared components | **Experimental challenger** | Only first-party low-bit candidate worth adding to this legacy line |
| Dev BF16/FP8 | **Deferred** | Different full-model line and no current product requirement |
| Distilled NVFP4 | **Deferred / unavailable** | Official card says coming soon; no artifact contract to implement |
| Dev NVFP4 | **Deferred** | Wrong lineage for current Distilled recipe |
| V2V, audio-conditioned, upscaling, IC-LoRA | **Deferred** | Real capabilities, separate operation contracts; LTX 2.5 is the forward-looking family |
| Community format zoo | **Rejected** | No maintenance value before official FP8 qualification |
| User-owned Comfy/LTX workflow | **Fallback** | Appropriate for unsupported LTX operations |
| Recommended native path | **None** | Output and target-hardware lifecycle acceptance are incomplete |

## Small qualification ladder

For each operation—T2V, first-frame I2V, and first+last conditioning—run a separate
ladder:

1. **Reference:** exact Distilled BF16 transformer and coherent official components,
   8 steps, CFG 1, identical conditioning and audio settings.
2. **Incumbent:** current complete-folder BF16 Engine path made settings/component-
   equivalent to the reference.
3. **Challenger:** official Distilled FP8 transformer with the same remaining components
   and no runtime conversion.

Do not add NVFP4 while the Distilled artifact is unpublished. Do not expand the family
beyond these operations unless a concrete creator workflow justifies preserving LTX
2.3 instead of implementing it on LTX 2.5.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- T2V and I2V at 24 fps with 25, 121, and 241 frames where practical;
- source dimensions/aspect buckets aligned to 32;
- dialogue, music, ambience, foley, silence, and audio-video synchronization cases;
- human speech lip timing, impacts, footsteps, instrument performance, camera motion,
  subject identity, temporal coherence, and audio continuity;
- first-frame fidelity and motion onset for I2V;
- final-frame approach, endpoint fidelity, temporal warping, and middle-frame motion for
  first+last conditioning.

Record text encoding/cache, first/last image preprocessing and VAE encode, transformer
load/offload, denoising, video decode, audio decode, mux/export, peak VRAM/RAM, disk
traffic, and backend/attention dispatch. Cancel during each phase and prove the next
job succeeds. Exercise cross-operation reuse: T2V to I2V, I2V to first+last, changed
prompt, changed duration, and explicit teardown.

For synchronized audio, inspect waveform duration, channel count, sample rate, A/V
start/end alignment, drift across long clips, clipping, silence handling, and container
metadata—not just whether an MP4 file exists.

## Hard gaps and source conflicts

1. **Complete-folder substitution:** Engine's 94.98 GB directory is reproducible, but
   it obscures exact component ownership and makes FP8 closure planning difficult.
2. **16 GB feasibility:** even the 29.5 GB FP8 transformer needs staged execution. Peak
   VRAM/RAM, transfer overhead, and warm reuse are unproved.
3. **Output acceptance:** no checked-in creator review or complete target-workstation
   cold/warm/cancel/reuse matrix exists.
4. **Distilled NVFP4 is not published:** an official repository page is not a loadable
   artifact. Wait for exact files.
5. **Official card staleness:** some LTX 2.3 cards still say Diffusers support is
   “coming soon,” while an official Diffusers repository and Engine-pinned path exist.
   Treat model-card prose as stale where contradicted by actual first-party artifacts.
6. **Upstream priority changed:** Lightricks now describes LTX 2.5 as recommended and
   LTX 2.3 as legacy. New feature investment needs a reason specific to 2.3.
7. **Operation equivalence:** first-frame and first+last conditioning cannot share one
   output-quality score.

## Ordered next actions

1. Finish target-workstation BF16 acceptance for T2V, first-frame I2V, and first+last
   conditioning, including synchronized audio and cancellation.
2. Inventory the exact current 94.98 GB repository into component roles and identities;
   retain the immutable complete-folder reference while designing a smaller closure.
3. Pin the official Distilled BF16 and FP8 transformer revisions and all shared
   components from one coherent upstream release.
4. Build a stored-FP8 adapter only after header/config inspection proves the exact
   layout; perform no runtime conversion.
5. Run BF16 versus FP8 per operation with the fixed corpus and actual backend dispatch.
6. Promote only if FP8 provides a material full-pipeline win with accepted video/audio.
7. Direct new operation work toward LTX 2.5 unless a 2.3-specific compatibility need is
   documented.

## Explicit non-goals

- Do not mix Dev and Distilled results.
- Do not collapse T2V, first-frame I2V, and first+last conditioning.
- Do not implement Distilled NVFP4 before an official artifact exists.
- Do not add community GGUF/INT8/W4/FP8 casts before first-party FP8 is qualified.
- Do not call a successful video write synchronized-audio acceptance.
- Do not expand a legacy line simply because additional upstream features exist.

## Primary sources

- Official LTX 2.3 model repository:
  <https://huggingface.co/Lightricks/LTX-2.3>
- Official FP8 repository:
  <https://huggingface.co/Lightricks/LTX-2.3-fp8>
- Official NVFP4 status:
  <https://huggingface.co/Lightricks/LTX-2.3-nvfp4>
- Official LTX-2 codebase:
  <https://github.com/Lightricks/LTX-2>
- Official Comfy LTX 2.3 first+last template at the reviewed immutable commit:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json>
- Engine LTX 2.3 recipes and runtime at the audited commit:
  <https://github.com/EnviralDesign/LatentSlate-Engine/tree/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine>
