# Wan 2.2 TI2V 5B roadmap

Last reviewed: **2026-08-11**  
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Wan 2.2 TI2V 5B is the credible **consumer-class Wan 2.2 line**, but its two
operations must remain separate:

1. **Text-to-video** — currently implemented in Engine through a complete BF16
   Diffusers repository.
2. **Image-to-video** — supported by the upstream 5B model and official Comfy
   workflow, but not implemented as a native Engine tool.

The current Engine recipe is useful as a structural/reference path, not a product
default. It installs a 34.2 GB complete BF16 repository, uses 50 steps and CFG 5, and
has not completed target-hardware output acceptance. The official Comfy topology is
much tighter: a 10.0 GB FP16 transformer, 1.41 GB Wan 2.2 VAE, and staged scaled-FP8
UMT5 encoder; Comfy says native offloading can fit the 5B workflow in 8 GB VRAM.

Therefore the next useful work is **not quantization research**. It is migrating to an
exact, reproducible official component closure and deciding which official schedule
Engine should own. Keep BF16 as the T2V source of truth, qualify split FP16 as the
challenger, and defer I2V until T2V lifecycle and output quality are accepted.

## Evidence labels

- **Verified** — stated by Wan, Comfy, or the Engine source at the audited commit.
- **Publisher measurement** — an upstream speed/memory claim, not an Engine result.
- **Inference** — a roadmap product judgment requiring target-workstation validation.

## Scope and operation boundaries

Wan 2.2 TI2V 5B is a dense 5B hybrid text/image-to-video model using the Wan 2.2 VAE
with temporal/spatial compression. The official model card states:

- native **T2V and I2V** in one model;
- 720p output at **24 fps**;
- Apache-2.0 weights;
- a five-second 720p video in under nine minutes on a consumer GPU without specific
  optimization — a publisher measurement, not an Engine result.

Verified source: [official model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
and [official Wan2.2 code snapshot](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc).

| Operation | Inputs | Official/Engine state | Comparison boundary |
| --- | --- | --- | --- |
| T2V | Prompt and negative prompt | Upstream and Comfy supported; Engine recipe/tool exists | First qualification line |
| I2V | Prompt, negative prompt, source image | Upstream and official Comfy workflow supported; Engine absent | Separate future operation with image-encode and anchor-fidelity acceptance |

Do not use a T2V result to qualify I2V. Image preprocessing, latent conditioning,
source preservation, and cancellation points differ materially.

## Canonical topology and settings

### Official Comfy path

The official [Comfy Wan 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)
uses:

- `wan2.2_ti2v_5B_fp16.safetensors`;
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`;
- `wan2.2_vae.safetensors`;
- optional image input through `Wan22ImageToVideoLatent` for I2V.

The pinned official example workflow
[`text_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/master/wan22/text_to_video_wan22_5B.json)
currently encodes 30 steps, CFG 5, `uni_pc` sampling, `simple` scheduler, denoise 1,
and an SD3 model-sampling shift of 8. The repository `master` branch is mutable; pin
the exact workflow blob during implementation.

### Current Engine path

Engine's native T2V runtime at `2ba5709` uses:

- 24 fps;
- default 1280×704 and 5 seconds;
- 50 steps and CFG 5;
- 25–121 frames, with `(frames - 1) % 4 == 0`;
- one to five seconds;
- an isolated CPU UMT5 subprocess on prompt-cache misses;
- explicit pipeline unload before text encoding when memory recovery is needed.

This is not settings-equivalent to the current Comfy example. A migration must select
and freeze one schedule rather than attributing output differences to file layout.

## Published artifacts and exact roles

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Complete first-party dense BF16 Diffusers repository | Apache-2.0; shared T2V/I2V lineage | **Reference** per operation |
| Engine resource `model:wan22:Wan-AI--Wan2.2-TI2V-5B-Diffusers` | Complete BF16 directory | **34,203,021,834 bytes**, pinned upstream revision `b8fff7315c768468a5333511427288870b2e9635` | **Experimental incumbent** |
| [`wan2.2_ti2v_5B_fp16.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/49f4d34972b94c6079febaf2a8bbba3452f3f2a9/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) | Official Comfy FP16 transformer | **9,999,658,848 bytes**; SHA-256 `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | **Experimental challenger** |
| [`wan2.2_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/9c311dda91b13fb3c970f9f72971d4df87c9eb00/split_files/vae/wan2.2_vae.safetensors) | Wan 2.2 high-compression VAE | **1.41 GB**; SHA-256 `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | Required component |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | Scaled-FP8 UMT5-XXL text encoder | Official Comfy workflow role; Engine's equivalent exact resource is **6,735,906,897 bytes** | Required staged component; pin one authoritative source/revision |
| Community Turbo, Lightning, GGUF, FP8 transformer, INT8, and W4 variants | Distilled or repackaged descendants | Real ecosystem work exists, but no first-party product need is established | **Rejected or Deferred** from the first ladder |

The component closure is a topology change, not merely “BF16 versus FP16.” It changes
artifact ownership, load boundaries, text-encoder handling, and acquisition footprint.

## Current Engine truth at `2ba5709`

- **Cataloged T2V recipe.** The built-in recipe
  [`wan-2-2-5b-ti2v-text-to-video-native-bf16.toml`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/builtin_recipes/wan22/wan-2-2-5b-ti2v-text-to-video-native-bf16.toml)
  binds the complete BF16 repository.
- **Acquisition is pinned.** The resource declaration records the exact official
  Diffusers revision and 34.2 GB directory size:
  [`wan22-ti2v-5b-bf16.toml`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/builtin_resource_declarations/wan22-ti2v-5b-bf16.toml).
- **T2V runtime exists.** The direct tool and runtime are in
  [`tools/wan22.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/wan22.py)
  and
  [`runtime/wan22.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/runtime/wan22.py).
- **I2V is not implemented for 5B.** The generic upstream capability does not imply an
  Engine schema, recipe, or accepted output.
- **Proof level: cataloged / structurally exercised.** Repository status says target-
  hardware output acceptance remains pending. Do not call it Recommended.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Official dense BF16 T2V | **Reference** | Matching first-party source of truth |
| Current complete-folder BF16 Engine T2V | **Experimental** | Exact catalog and runtime exist, but footprint, settings divergence, and output acceptance remain |
| Official split FP16 Comfy T2V | **Experimental challenger** | Much smaller exact closure and documented offload path; closest likely product candidate |
| Official split FP16 I2V | **Deferred** | Same model supports it, but it is a separate Engine operation and acceptance corpus |
| Community Turbo/Lightning | **Deferred** | Different schedule/lineage; consider only after standard T2V is accepted |
| Transformer FP8/NVFP4/INT8/GGUF zoo | **Rejected** | No creator-value case before the official FP16 topology is proven |
| User-owned Comfy workflow | **Fallback** | Existing immediate path for T2V/I2V experimentation |
| Recommended native path | **None** | No accepted target-workstation outputs or lifecycle evidence |

## Small qualification ladder

### T2V

1. **Reference:** exact complete BF16 Diffusers repository at one frozen schedule.
2. **Incumbent:** current Engine complete-folder BF16 path, made settings-equivalent to
   the reference.
3. **Challenger:** official split FP16 transformer + scaled-FP8 UMT5 + Wan 2.2 VAE,
   using the same schedule and output contract.

The challenger earns a production loader only if it materially improves cold/warm
time, RAM/VRAM, disk footprint, or stability while preserving accepted video quality.

### I2V

Defer implementation until T2V is stable. Then use the same BF16-versus-split-FP16
ladder with a separate source-image corpus and anchor-fidelity metrics.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- T2V at 1280×704 / 121 frames / 24 fps and at least one portrait/landscape bucket;
- five-second clips with static, slow, fast, articulated-human, animal, vehicle, camera
  motion, scene transition, and text/signage cases;
- negative-prompt effectiveness and prompt-cache hit/miss;
- temporal coherence, subject identity, anatomy, object persistence, camera intent,
  first/last-frame corruption, loop-like artifacts, and VAE decode quality.

Record text-encoder subprocess startup, prompt-cache behavior, model load/offload,
denoising, VAE decode, video encoding, peak VRAM/RAM, host-device traffic, and disk
reads. Cancel during encoder startup, diffusion load, denoising, decode, and export;
then prove a clean subsequent generation.

For future I2V, add crop/resize provenance, first-frame reconstruction, prompt-versus-
image control balance, source identity, and motion without frame freezing.

## Hard gaps and source conflicts

1. **Schedule mismatch:** current Engine uses 50 steps; the current official Comfy
   example uses 30. No format comparison is valid until one schedule is frozen.
2. **Topology mismatch:** Engine's complete 34.2 GB BF16 folder is not the official
   Comfy split-component closure.
3. **Operation mismatch:** upstream and Comfy support I2V, but Engine currently exposes
   only T2V for the 5B family.
4. **Publisher memory claim:** Comfy says the 5B path fits 8 GB with native offload;
   Engine has not reproduced that claim on Windows/5080.
5. **Mutable Comfy template:** the workflow and repository defaults can change; pin an
   exact blob and coherent component revision.
6. **No accepted output set:** structural code and a catalog do not establish creator-
   ready quality, cancellation, reuse, or teardown.

## Ordered next actions

1. Freeze the T2V reference schedule after comparing the official Wan and Comfy
   defaults; record why Engine chooses 30, 50, or another exact setting.
2. Run the current BF16 recipe on the fixed T2V corpus and complete cancellation,
   warm reuse, and teardown acceptance.
3. Declare one coherent split-component acquisition closure with exact revisions,
   hashes, roles, and licenses.
4. Add a header-only adapter/manifest for the FP16 transformer, scaled-FP8 UMT5, and
   Wan 2.2 VAE; perform no runtime conversion.
5. Implement the split FP16 T2V challenger and measure the actual offload boundary.
6. Promote only after a material creator-visible win and output acceptance.
7. Add 5B I2V as a separate recipe only after the T2V component lifecycle is stable.

## Explicit non-goals

- Do not equate the current 34.2 GB Diffusers folder with the 11.4 GB Comfy diffusion+
  VAE closure or attribute all differences to precision.
- Do not add I2V by silently making the source image optional in the T2V schema.
- Do not introduce Turbo, Lightning, GGUF, FP8, INT8, or W4 variants before the
  official split FP16 path proves insufficient.
- Do not change fps, frames, steps, scheduler, or shift while calling a comparison
  apples-to-apples.
- Do not claim Comfy's 8 GB statement as an Engine measurement.

## Primary sources

- Official Wan 2.2 code snapshot:
  <https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc>
- Official TI2V 5B model card:
  <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>
- Official Diffusers repository:
  <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>
- Official Comfy tutorial:
  <https://docs.comfy.org/tutorials/video/wan/wan2_2>
- Official Comfy artifact repository:
  <https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged>
- Official Comfy examples:
  <https://github.com/comfyanonymous/ComfyUI_examples/tree/master/wan22>
