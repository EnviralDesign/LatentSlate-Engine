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
acceptance is unfinished. The implemented product paths are Lightricks' official Comfy
topologies: **Dev FP8 plus the Distilled LoRA** for T2V/first-frame I2V, and
**Distilled FP8** for first+last-frame video. Distilled NVFP4 is still advertised as
“coming soon”; do not build a loader around an unpublished artifact.

This roadmap has no Recommended path yet. The native BF16 path is a structural
reference only: it loaded coherently but OOMed on the 16 GB RTX 5080 at first execution.
It must continue to fail safely through the disposable-worker boundary, but is not a
candidate for another local GPU retry. The exact official Comfy optimized closures are
implemented and CPU-contract reviewed; their next gate is per-operation hardware,
output, cancellation, and recovery acceptance.

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
| Official Comfy Dev FP8 + Distilled LoRA T2V/I2V | Official optimized template topology | Cataloged / CPU contract implemented | Requires local Comfy and hardware acceptance |
| Official Comfy Distilled FP8 FLF | Official first+last template topology | Cataloged / CPU contract implemented | Separate optimized product operation; requires hardware acceptance |
| Video-to-video, audio-conditioned, IC-LoRA, upscaler, and other LTX operations | Additional official ecosystem capabilities | Not implemented | Deferred; separate schemas and acceptance corpora |

Lightricks describes LTX 2.3 as a 22B DiT-based joint audio-video foundation model.
Its official cards require dimensions divisible by 32 and frame counts satisfying
`8n + 1`. The public source is not Hugging Face gated, but use remains subject to
the [LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.3).
License and redistribution review are recipe gates.

## Published artifacts and topology

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) | Official Distilled/Dev BF16 repository and components | First-party public source; LTX-2 Community License | **Reference source** |
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

The official [T2V](https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_t2v.json),
[first-frame I2V](https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_i2v.json), and
[first+last-frame](https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_flf2v.json)
templates at workflow-templates commit `8b2c08f297c63ffc73ce93f938b0f5139c0ed73f`
are the topology sources for the optimized product slice. Their raw UTF-8 blob
SHA-256 values are respectively `75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f`,
`91dd8e44926fd37f6d9307789484370fa333582b14e53ed771d63ed805379ee4`, and
`168bc2584ef117133e76341f04e001aab2641b72b75d81b66b5c0b66e56c24a5`.
T2V and first-frame I2V select Dev FP8 with the Distilled LoRA. FLF selects
Distilled FP8, the Comfy Gemma text encoder, and ordered endpoints at frame
zero/final frame. These templates are **not** complete immutable acquisition
closures and do not change the exact native BF16 component set or schedule.
The Dev packet uses Dev FP8 (29,145,431,166 bytes), fixed Distilled model LoRA
(2,741,024,390), Gemma FP4 text encoder (9,447,702,218), fixed Gemma text LoRA
(628,203,616), and spatial x2 upscaler (995,743,560). FLF uses Distilled FP8
(29,531,884,062) and the same Gemma text encoder. Each FP8 checkpoint header contains
the joint transformer plus video VAE, audio VAE, and vocoder tensors; no fictional
separate VAE/vocoder acquisition is declared. All resources have immutable Hugging Face
revision, byte size, LFS SHA-256, and header-schema pins in their declarations.

The Engine optimized operations now bind the pinned artifact declarations and build
separate T2V, first-frame I2V, and ordered first+last API graphs. Their submitted
graph fingerprints are derived provenance, distinct from the raw template blob
identities above. Stored-FP8 dispatch, cancellation/recovery, and synchronized A/V
still require local-Comfy and hardware acceptance separately for each operation.

## Current Engine truth

- **Package-owned recipes are distinct:** native T2V, first-frame I2V, and
  first+last-frame reference recipes bind one exact BF16 closure; the optimized
  Comfy recipes bind their own separate FP8 closures. Both first+last operations
  require ordered endpoints rather than an optional-last-image convention.
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
- **Native-reference boundary:** the native Diffusers runtime accepts the exact
  BF16 closure only and does not convert weights at runtime. The separately
  cataloged Comfy operations use their pinned FP8/LoRA topologies.
- **Proof level: cataloged / runtime-structured.** Repository status says H3/LTX and
  native Wan target-hardware output acceptance still needs the planned hands-on pass.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Native Distilled BF16 operation | **Structural reference** | Exact closure/operations exist; the 16 GB local run OOMed. Defer dense reference hardware work to a batched Vast campaign. |
| Official Comfy Dev FP8 + Distilled LoRA T2V / first-frame I2V | **Experimental / CPU-contract implemented** | Exact immutable closure and source-derived graph contract exist; local-Comfy and output acceptance remain. |
| Official Comfy Distilled FP8 first+last-frame | **Experimental / CPU-contract implemented** | Distinct ordered-conditioning contract; local-Comfy and output acceptance remain. |
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
cache claim. For the native reference, cancel during loading and first-step denoising.
For optimized Comfy operations, current acceptance cancels only after prompt-bound
active-queue/execution-start evidence (not a claimed denoise step); first-node/denoise
cancellation is a later hardware acceptance extension. In both cases prove the worker
tree has exited, then prove the next cold job succeeds. Exercise cross-operation switching:
T2V to first-frame I2V, first-frame I2V to first+last, changed prompt, changed
duration, and explicit teardown.

For synchronized audio, inspect waveform duration, channel count, sample rate, A/V
start/end alignment, drift across long clips, clipping, silence handling, and container
metadata—not just whether an MP4 file exists.

## Hard gaps and source conflicts

1. **Source verification:** the 50-file native manifest and optimized immutable
   component declarations are explicit; installation and target-machine source checks
   must still revalidate every pinned immutable tuple.
2. **Optimized-path feasibility:** the official FP8 topologies and operation-specific
   execution plans are implemented, but peak VRAM/RAM, transfer overhead, and actual
   target-hardware output remain unproved.
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
2. Run the implemented official Comfy optimized closures—Dev FP8 plus Distilled LoRA
   for T2V/first-frame I2V and Distilled FP8 for first+last conditioning—against each
   pinned template and acceptance corpus.
3. Verify the catalog's native 50-file closure and optimized component tuples against
   the public source before installing;
   no smaller closure has been proven because all retained components are runtime-owned.
4. Maintain the pinned official Distilled BF16/FP8 and optimized shared-component
   revisions as upstream releases evolve; perform no runtime conversion.
5. Run optimized Comfy paths per operation with the fixed corpus and actual backend dispatch.
6. Promote only if an optimized path provides a material full-pipeline win with accepted video/audio.
7. Direct new operation work toward LTX 2.5 unless a 2.3-specific compatibility need is
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
  <https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_t2v.json>
- Official Comfy LTX 2.3 first-frame I2V template:
  <https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_i2v.json>
- Official Comfy LTX 2.3 first+last template:
  <https://github.com/Comfy-Org/workflow_templates/blob/8b2c08f297c63ffc73ce93f938b0f5139c0ed73f/templates/video_ltx2_3_flf2v.json>
- Engine implementation: `src/latentslate_engine/builtin_recipes/ltx23/`,
  `src/latentslate_engine/builtin_resource_declarations/ltx23-distilled-bf16.toml`,
  `src/latentslate_engine/runtime/ltx23.py`, and
  `src/latentslate_engine/tools/ltx23.py`. Pin a repository commit only when the
  acceptance seam is committed.
