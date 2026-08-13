# LTX 2.3 roadmap

Last reviewed: **2026-08-13**

Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

LTX 2.3 is Engine's first native synchronized audio-video family, but it is now a
**legacy upstream line** relative to LTX 2.5. Preserve and finish its existing path;
do not expand it into a broad optimization program.

Keep the native reference and the three official Comfy operations distinct:

1. **Native Distilled BF16** is a structural/reference closure only, not a local 16 GB
   product path.
2. Official Comfy **T2V** and **first-frame I2V** use **Dev FP8 plus the Distilled
   LoRA**.
3. Official Comfy **first+last-frame** uses **Distilled FP8**. It is a different graph,
   not a replacement transformer for the Dev-FP8-plus-LoRA operations.

Engine now declares a 50-file, immutable **94,977,693,482-byte** BF16 Diffusers
component closure and exposes distinct T2V, first-frame I2V, and first+last-frame
operations. This exact closure is only 7,072 bytes smaller than the former complete
folder because it removes repository documentation and attributes, not model
components. The exact runtime contract is strong, but target-hardware output
acceptance is unfinished. The next product paths are Lightricks' official Comfy
topologies: **Dev FP8 plus the Distilled LoRA** for T2V/first-frame I2V, and
**Distilled FP8** for first+last-frame video. Distilled NVFP4 is still advertised as
“coming soon”; do not build a loader around an unpublished artifact.

This roadmap has no Recommended path yet. The native BF16 path is a structural
reference only: it loaded coherently but OOMed on the 16 GB RTX 5080 at first execution.
It must continue to fail safely through the disposable-worker boundary, but is not a
candidate for another local GPU retry. The next product slice is an exact official
Comfy optimized closure: Dev FP8 plus the Distilled LoRA for T2V/first-frame I2V, and
Distilled FP8 for first+last conditioning.

## Evidence labels

- **Verified** — stated by Lightricks or the Engine source/catalog at the audited
  commit.
- **Publisher measurement** — an upstream performance/memory claim, not an Engine
  result.
- **Inference** — a roadmap product judgment requiring target-workstation validation.

## Scope and lineage boundaries

| Line / operation | Canonical behavior | Engine state | Comparison boundary |
| --- | --- | --- | --- |
| Native Distilled BF16 T2V/I2V/FLF | Exact Diffusers structural reference | Recipe/runtime exists; local 16 GB OOM | Defer dense acceptance to a batched Vast campaign |
| Official Comfy Dev FP8 + Distilled LoRA T2V/I2V | Official optimized template topology | Not implemented | Next product acceptance line |
| Official Comfy Distilled FP8 FLF | Official first+last template topology | Not implemented | Separate optimized product operation |
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
| Engine native BF16 closure | Exact 50-file official Diffusers component closure | **94,977,693,482 bytes**, revision `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee`; documentation-only 7,072 B reduction | **Structural reference; 16 GB OOM evidence** |
| [`ltx-2.3-22b-distilled-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-distilled-fp8.safetensors) | Official Distilled FP8 transformer | **29.5 GB**; SHA-256 `d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2` | **Target for official Comfy FLF** |
| [`ltx-2.3-22b-dev-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-dev-fp8.safetensors) | Official Dev FP8 transformer | **29.1 GB**; SHA-256 `28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450` | **Target for official Comfy T2V/I2V with Distilled LoRA** |
| [`Lightricks/LTX-2.3-nvfp4`](https://huggingface.co/Lightricks/LTX-2.3-nvfp4) | QAD NVFP4 repository | Dev NVFP4 published; **Distilled NVFP4 listed as coming soon** | **Deferred** for current Distilled product line |
| Community GGUF, INT8, Nunchaku, mixed W4, custom FP8 casts | Various | Artifacts may exist, but provenance/layout and creator value do not beat first-party FP8 | **Rejected** from the initial ladder |

The transformer size is not the complete pipeline size. Qualification must inventory
the exact text encoder, video/audio VAE components, vocoder/audio decoder, scheduler,
tokenizer, and condition-pipeline files from one coherent revision. Replacing only the
transformer does not make the remaining 95 GB closure disappear.

### Pinned official Comfy optimized-template investigation packet

The official [T2V](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_t2v.json),
[first-frame I2V](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_i2v.json), and
[first+last-frame](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json)
templates at workflow-templates commit `2b7f823136606344f0bccce249898d771b809aa1`
are the topology sources for the optimized product slice. T2V and first-frame I2V
select Dev FP8 with the Distilled LoRA. FLF selects Distilled FP8, the Comfy Gemma text
encoder, and ordered endpoints at frame zero/final frame. These templates are **not**
complete immutable acquisition closures and do not change the exact native BF16
component set or schedule.

Before the optimized paths are implemented, first capture authenticated immutable identities
for the transformer, text encoder, video/audio VAE, vocoder, tokenizer, scheduler, and
the exact condition support files. Then represent the endpoint pair as an ordered,
typed request—not an unordered image list—and prove stored-FP8 native dispatch,
cancellation/recovery, and synchronized A/V separately from T2V and first-frame I2V.

## Current Engine truth

- **Package-owned recipes are distinct:** T2V, first-frame I2V, and first+last-frame
  video all bind one exact native BF16 closure. The latter requires both ordered
  endpoints, rather than treating an optional last image as an implementation detail.
- **Acquisition is immutable and explicit:**
  `ltx23-distilled-bf16.toml` pins the exact upstream revision and a 50-path
  allowlist, including every audio/video VAE, connectors, text encoder, tokenizer,
  transformer, scheduler, and vocoder payload/config required by the Diffusers
  runtime. It is not a material footprint reduction.
- **Runtime operations are explicit:** `tools/ltx23.py` and `runtime/ltx23.py`
  provide synchronized-audio T2V plus separately described first-frame and
  first+last condition operations. Native Diffusers denoising now checks job
  cancellation after every step; condition provenance records `first_frame` versus
  `first_last_frame` and endpoint order `[0]` or `[0, -1]`.
- **Fixed Engine behavior:** 24 fps; 8 steps; CFG 1; 1–10 seconds; 25–241 frames;
  `(frames - 1) % 8 == 0`; dimensions aligned to 32; maximum pixel area 942,080;
  native attention; VAE tiling; sequential/model/no offload policies. Each generation
  runs in a disposable worker, so no cross-job tensor prompt cache is retained.
- **Native audio contract:** the pinned upstream `vocoder/config.json` declares
  `output_sampling_rate = 48000` and `out_channels = 2`. Acceptance verifies one
  48 kHz stereo audio stream alongside one 24 fps video stream; Engine does not
  resample this output.
- **No low-bit loader or LoRA path.** The current runtime accepts the exact native
  BF16 closure only and does not convert weights at runtime.
- **Proof level: cataloged / runtime-structured.** Repository status says H3/LTX and
  native Wan target-hardware output acceptance still needs the planned hands-on pass.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Native Distilled BF16 operation | **Structural reference** | Exact closure/operations exist; the 16 GB local run OOMed. Defer dense reference hardware work to a batched Vast campaign. |
| Official Comfy Dev FP8 + Distilled LoRA T2V / first-frame I2V | **Pending optimized product candidate** | Must match the two official template topologies and get its own immutable closure and acceptance. |
| Official Comfy Distilled FP8 first+last-frame | **Pending optimized product candidate** | A distinct conditioning graph; must not be substituted for the Dev-FP8-plus-LoRA operations. |
| Distilled NVFP4 | **Deferred / unavailable** | Official card says coming soon; no artifact contract to implement |
| Dev NVFP4 | **Deferred** | Wrong lineage for current Distilled recipe |
| V2V, audio-conditioned, upscaling, IC-LoRA | **Deferred** | Real capabilities, separate operation contracts; LTX 2.5 is the forward-looking family |
| Community format zoo | **Rejected** | No maintenance value before official FP8 qualification |
| User-owned Comfy/LTX workflow | **Fallback** | Appropriate for unsupported LTX operations |
| Recommended native path | **None** | Output and target-hardware lifecycle acceptance are incomplete |

## Small qualification ladder

Keep qualification operation-specific:

1. T2V and first-frame I2V use the official Comfy Dev-FP8-plus-Distilled-LoRA topology.
2. First+last-frame uses the official Comfy Distilled-FP8 topology and its ordered
   endpoint condition contract.
3. The exact native BF16 line remains a structural/reference comparison only. It has no
   further local 16 GB retry; dense reference evidence belongs to a batched Vast campaign.

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

Record text encoding, first/last image preprocessing and VAE encode, transformer
load/offload, denoising, video decode, audio decode, mux/export, peak VRAM/RAM, disk
traffic, and backend/attention dispatch. Every LTX request is intentionally a cold,
disposable worker: record its terminal Job Object tree-empty proof rather than a warm
cache claim. Cancel during loading and first-step denoising, prove the worker tree has
exited, then prove the next cold job succeeds. Exercise cross-operation switching:
T2V to first-frame I2V, first-frame I2V to first+last, changed prompt, changed
duration, and explicit teardown.

For synchronized audio, inspect waveform duration, channel count, sample rate, A/V
start/end alignment, drift across long clips, clipping, silence handling, and container
metadata—not just whether an MP4 file exists.

## Hard gaps and source conflicts

1. **Gated-source verification:** the 50-file manifest makes component ownership
   explicit, but Engine must still compare every immutable tuple against the gated
   upstream source before installation and before it can serve as an FP8 closure
   baseline.
2. **Optimized-path feasibility:** the official FP8 topologies still need exact
   component closures and target-hardware acceptance. Peak VRAM/RAM, transfer overhead,
   and their operation-specific execution plans are unproved.
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

1. Preserve the native BF16 closure as a structural reference with disposable-worker
   cancel/failure cleanup; do not schedule another local BF16 GPU retry after the
   observed 16 GB OOM. Defer dense BF16 hardware reference work to a batched Vast
   campaign.
2. Build the exact official Comfy optimized closure: Dev FP8 plus Distilled LoRA for
   T2V/first-frame I2V, and Distilled FP8 for first+last conditioning; compare each
   implementation directly against its pinned template topology.
3. Verify the catalog's 50-file closure against the gated source before installing;
   no smaller closure has been proven because all retained components are runtime-owned.
4. Pin the official Distilled BF16 and FP8 transformer revisions and all shared
   components from one coherent upstream release.
5. Build a stored-FP8 adapter only after header/config inspection proves the exact
   layout; perform no runtime conversion.
6. Run optimized Comfy paths per operation with the fixed corpus and actual backend dispatch.
7. Promote only if an optimized path provides a material full-pipeline win with accepted video/audio.
8. Direct new operation work toward LTX 2.5 unless a 2.3-specific compatibility need is
   documented.

## Explicit non-goals

- Do not apply the Dev-FP8-plus-Distilled-LoRA T2V/I2V topology to FLF, or treat the
  FLF Distilled-FP8 topology as a T2V/I2V substitute.
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
- Official Comfy LTX 2.3 T2V template:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_t2v.json>
- Official Comfy LTX 2.3 first-frame I2V template:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_i2v.json>
- Official Comfy LTX 2.3 first+last template:
  <https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_3_flf2v.json>
- Engine implementation: `src/latentslate_engine/builtin_recipes/ltx23/`,
  `src/latentslate_engine/builtin_resource_declarations/ltx23-distilled-bf16.toml`,
  `src/latentslate_engine/runtime/ltx23.py`, and
  `src/latentslate_engine/tools/ltx23.py`. Pin a repository commit only when the
  acceptance seam is committed.
