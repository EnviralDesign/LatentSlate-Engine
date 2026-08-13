# Wan 2.2 TI2V 5B roadmap

Last reviewed: **2026-08-13**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Wan 2.2 TI2V 5B is the credible **consumer-class Wan 2.2 line**, but its two
operations must remain separate:

1. **Text-to-video** — the dense native BF16 recipe is the **REFERENCE** path; the
   pinned official Comfy split-component graph is the accepted **FALLBACK** path on
   the target RTX 5080.
2. **Image-to-video** — accepted as a distinct required-image **FALLBACK** operation
   using the same exact components and pinned Comfy worker as T2V.

The current complete-folder Engine recipe remains useful as a structural/reference
path, not a product default. It installs a 34.2 GB BF16 Diffusers repository, uses
50 steps and CFG 5, and has not completed target-hardware output acceptance. The
accepted practical topology is a 10.0 GB FP16 transformer, 1.41 GB Wan 2.2 VAE, and
6.74 GB scaled-FP8 UMT5 encoder, loaded by an isolated pinned Comfy process.

Engine freezes the exact Comfy example schedule: 30 steps, CFG 5, `uni_pc` / `simple`,
SD3 shift 8, denoise 1, and 24 fps. The accepted result is practical rather than an
apples-to-apples BF16 quality comparison: the 34.2 GB reference was deliberately not
exercised during this qualification because the official split path did not require it.
It is now installed as the cataloged reference, but this install is not output or
target-hardware acceptance evidence for the dense runtime.

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
| T2V | Prompt and negative prompt | Engine public API accepted at 1280×704 / 121 / 24 | Practical split path |
| I2V | Prompt, negative prompt, required source image | Engine public API accepted at 1280×704 / 121 / 24 | Separate schema; same exact component closure |

T2V does not accept an image field. I2V records source SHA-256, source dimensions,
the exact bilinear center crop, VAE encoding, and first-latent anchor semantics.

## Canonical topology and settings

### Official Comfy path

The official [Comfy Wan 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)
uses:

- `wan2.2_ti2v_5B_fp16.safetensors`;
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`;
- `wan2.2_vae.safetensors`;
- optional image input through `Wan22ImageToVideoLatent` for I2V.

The pinned official example workflow at examples revision
`f9431bb000ce792094ff345446e22cac1ea6cef3` (workflow SHA-256
`e7913b6b2c8f7d82a6a6f9940289bf6e7513cc908bbf455e4553de9804c6f571`)
[`text_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/master/wan22/text_to_video_wan22_5B.json)
encodes 30 steps, CFG 5, `uni_pc` sampling, `simple` scheduler, denoise 1,
and an SD3 model-sampling shift of 8. The repository `master` branch is mutable; pin
the exact workflow blob during implementation. Engine also pins the executable
Comfy-Org checkout to `eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f` and records the
audited upstream source snapshot `725e6ec60621c6f001af04769173e7dbb3c53541`.

### Current Engine path

Engine's native T2V runtime at the reviewed `8ddf831` baseline uses:

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
| Engine resource `model:wan22:wan-ai--wan2.2-ti2v-5b-diffusers` | Complete BF16 directory | **34,203,021,834 bytes**, pinned upstream revision `b8fff7315c768468a5333511427288870b2e9635` | **Reference** |
| [`wan2.2_ti2v_5B_fp16.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/fb1388adc906ab39ffc26ee40e96b22886b56bc4/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) | Official Comfy FP16 transformer | revision `fb1388a…`; **9,999,658,848 bytes**; SHA-256 `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | **Fallback component** |
| [`wan2.2_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/fb1388adc906ab39ffc26ee40e96b22886b56bc4/split_files/vae/wan2.2_vae.safetensors) | Wan 2.2 high-compression VAE | revision `fb1388a…`; **1,409,400,960 bytes**; SHA-256 `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | Required fallback component |
| [`umt5_xxl_fp8_e4m3fn_scaled.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/06e001fc51048fb03433a6fb25334de7836704a5/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | Scaled-FP8 UMT5-XXL text encoder | revision `06e001f…`; **6,735,906,897 bytes**; SHA-256 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | Required fallback staged component |
| Community Turbo, Lightning, GGUF, FP8 transformer, INT8, and W4 variants | Distilled or repackaged descendants | Real ecosystem work exists, but no first-party product need is established | **Rejected or Deferred** from the first ladder |

The component closure is a topology change, not merely “BF16 versus FP16.” It changes
artifact ownership, load boundaries, text-encoder handling, and acquisition footprint.

## Current Engine truth

- **Cataloged dense T2V reference.** The built-in recipe
  [`wan-2-2-5b-ti2v-text-to-video-native-bf16.toml`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/8ddf831/src/latentslate_engine/builtin_recipes/wan22/wan-2-2-5b-ti2v-text-to-video-native-bf16.toml)
  binds the complete BF16 repository.
- **Acquisition is pinned.** The resource declaration records the exact official
  Diffusers revision and 34.2 GB directory size:
  [`wan22-ti2v-5b-bf16.toml`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/8ddf831/src/latentslate_engine/builtin_resource_declarations/wan22-ti2v-5b-bf16.toml).
- **T2V runtime exists.** The direct tool and runtime are in
  [`tools/wan22.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/8ddf831/src/latentslate_engine/tools/wan22.py)
  and
  [`runtime/wan22.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/8ddf831/src/latentslate_engine/runtime/wan22.py).
- **Distinct fallback operations are accepted.** The split Comfy closure backs a T2V
  schema with no image input and an I2V schema with a required source image. Both have
  fixed-seed public-API output and lifecycle acceptance on the target RTX 5080.
- **The dense path remains a reference.** Its catalog/runtime existence does not inherit
  the split path's hardware proof and it is not labeled Recommended.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Official dense BF16 T2V | **Reference** | Matching first-party source of truth |
| Current complete-folder BF16 Engine T2V | **Reference** | Exact first-party dense source-of-truth catalog/runtime; no cross-topology quality claim |
| Official split FP16 Comfy T2V | **Fallback** | Exact closure, lifecycle, public API, and 1280×704 / 121-frame output accepted on RTX 5080 |
| Official split FP16 I2V | **Fallback** | Distinct required-image schema, anchor fidelity, switching, lifecycle, and official-size output accepted |
| Community Turbo/Lightning | **Deferred** | Different schedule/lineage; consider only after standard T2V is accepted |
| Transformer FP8/NVFP4/INT8/GGUF zoo | **Rejected** | No creator-value case before the official FP16 topology is proven |
| User-owned Comfy workflow | **Fallback** | Existing immediate path for T2V/I2V experimentation |
| Recommended native path | **None** | Accepted path deliberately remains Comfy-first rather than being relabeled native |

## Small qualification ladder

### T2V

1. **Reference:** exact complete BF16 Diffusers repository at one frozen schedule.
2. **Fallback:** official split FP16 transformer + scaled-FP8 UMT5 + Wan 2.2 VAE,
   with its separately accepted Comfy schedule and output contract.

Any future alternative earns a separately qualified loader only if it materially
improves cold/warm time, RAM/VRAM, disk footprint, or stability while preserving
accepted video quality.

### I2V

The accepted I2V operation uses the same exact split component fingerprint as T2V,
but has its own recipe fingerprint, required-image schema, official conditioning
node, source preprocessing provenance, and anchor-fidelity checks. A future dense
comparison still requires a separate source-image corpus.

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

I2V acceptance records crop/resize provenance and first-frame reconstruction.
Broader creator-quality work should expand prompt-versus-image control balance,
source identity, and motion-without-freezing coverage.

## T2V target-hardware acceptance

Acceptance ran through the public Engine catalog/job/artifact API on an RTX 5080
16,303 MiB card. The frozen 1280×704 / 121-frame / 24 fps request used seed
`20260812`, the exact schedule above, and the exact component closure in this file.

| Case | Result | Engine/runtime timing | Peak sampled memory | Artifact |
| --- | --- | --- | --- | --- |
| Bounded diagnostic, 128×96 / 5 | Succeeded | 68.6 s first study | 15,472 MiB GPU | Valid VP9 WebM |
| Cancel during generation | Canceled, then recovered | 4.06 s cancel latency; recovery 25.11 s | GPU released after clear | Recovery output succeeded |
| Warm same-request reuse | Succeeded | 0.52 s API elapsed (Comfy execution-cache hit) | Same retained worker | Byte-identical to recovery output |
| Official contract, 1280×704 / 121 | Succeeded | 372.69 s API; 11.98 s server, 357.08 s generation/export | 15,413 MiB GPU; 51.55 GB sampled system RAM in use | 904,992-byte VP9 WebM, SHA-256 `ce16cb827d4bfcd9ced2eb3fcc4e55a80055f9d5456b0e425fbde1c877ca4341` |
| Post-pin diagnostic, 128×96 / 5 | Succeeded | 33.03 s API; 18.78 s server, 14.03 s generation/export | Worker cleared after run | Provenance records executable revision `eb4a7b4…` |

`ffprobe` confirmed 1280×704, VP9/yuv420p, 24 fps, and 5.042 seconds. Visual
inspection of first, middle, and final frames found a coherent fox traversal, stable
snow scene, and no endpoint corruption. This one prompt establishes operational
acceptance, not a broad creator-quality benchmark. Warm byte equality above is an
execution-cache result and is not represented as an independent stochastic rerun.

The Comfy worker uses isolated input/output/temp/model roots, hardlinks only the three
validated artifacts, disables custom nodes, binds loopback only, checks cancellation
while starting and polling, interrupts and evicts on cancellation/failure, atomically
publishes downloaded output, and releases the GPU on shutdown. Comfy logs measured
6,419 MB staged text-encoder state, 9,535 MB staged transformer state, and 1,344 MB
staged VAE state; these are loader reports, distinct from sampled peak VRAM.

## I2V target-hardware acceptance

The stable source is the middle frame of the accepted T2V fox clip: 1280×704 PNG,
SHA-256 `2c6ccaf32f958bb963962a3889f6ff47b500e3975b0af72dedbbcf592b7a4229`.

| Case | Result | Engine/runtime timing | Peak sampled memory | Artifact / assertion |
| --- | --- | --- | --- | --- |
| T2V→I2V→T2V, 128×96 / 5 | All succeeded in one worker | 25.95 s cold T2V; 9.31 s warm I2V; 6.55 s warm T2V | Released after sequence | One component fingerprint across both recipe fingerprints |
| I2V first-frame diagnostic | Succeeded | Included above | Included above | MAE 2.53/255, PSNR 33.59 dB versus exact recorded center-crop anchor |
| Cancel during official-size I2V | Canceled, then recovered | 2.33 s cancellation; 28.64 s recovery | GPU released | Windows log-handle cleanup bug found and fixed before acceptance |
| Official contract, 1280×704 / 121 | Succeeded | 442.73 s public API | 15,613 MiB GPU; 50.91 GB sampled system RAM in use | 786,547-byte VP9 WebM, SHA-256 `75fd03c57710a69b0accc82cd9ea47e016c1bf38850c47416d79159fc90c6d22` |

`ffprobe` confirmed 1280×704, VP9, 24 fps, and 5.042 seconds. Its first frame
reconstructed the exact source with MAE 1.32/255 and PSNR 44.04 dB; middle/final
inspection retained the fox and landscape while adding motion. This establishes one
operational I2V case, not exhaustive image/prompt-control quality coverage.

## Exact TI2V 5B LoRA closure

Engine exposes one optional model-only LoRA slot on each Comfy recipe. The fixed base
closure is exactly the lowercase transformer resource ID
`model:wan22:comfy-org-wan22-ti2v-5b/split_files/diffusion_models/wan2.2_ti2v_5b_fp16`,
the scaled-FP8 UMT5, and the Wan 2.2 VAE; all three must reside on one volume so the
worker can stage them by zero-copy hardlink. The base closure is therefore exact and
installable. Selection is
fail-closed: the resource must identify base `Wan-AI/Wan2.2-TI2V-5B`, carry immutable
SHA/schema/rank metadata, and probe as exactly 600 adapter tensors across all 30
blocks and the expected ten attention/FFN modules at 3072/14336 dimensions. Comfy's
pinned `LoraLoaderModelOnly` is inserted before `ModelSamplingSD3`; completed jobs
fail if Comfy logs any unmapped LoRA key. The exposed slot permits an arbitrary
user-selected exact adapter, so the complete profile cannot be remotely locked in
advance; this does not weaken the fixed base closure or the adapter validation gate.

| Adapter | Exact evidence | Status |
| --- | --- | --- |
| [`ostris/wan22_5b_i2v_crush_it_lora`](https://huggingface.co/ostris/wan22_5b_i2v_crush_it_lora/tree/e4b85be20d75c2ca2ee1b901ba2cf49d9416e233) | Apache-2.0; explicit TI2V-5B base; 161,293,208-byte BF16 file; observed SafeTensors rank **32**; SHA-256 `00a3ed72d8e257b416e1232cce07acf76cfb3ad7538f8ba995b6818f0b560f23`; trigger `crush it` | Installed and public-API accepted |
| [`AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets`](https://huggingface.co/AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets/tree/fb47fbdfb7fa391ed6d29f1d1b06f78bc815d7c0) | Apache-2.0; explicit TI2V-5B base; 322,511,512-byte FP16 final file; observed SafeTensors rank **64**; SHA-256 `5c2fc21b1e74d5088318fea72c676181650a0f771cc521151edfc43f6ea9ec77` | Exact catalog option, not installed locally |

Matched 320×192 / 9-frame / seed `10101` public-API control and Crush-It runs
succeeded in 29.17 s cold and 7.55 s warm. Their submitted graph hashes differ;
runtime provenance records adapter file/schema hashes, rank, strength, loader, 600
expected tensors, 300 patch targets, and zero unmapped-key warnings. A later user-
supplied CivitAI URL should be imported as an exact resource, then tested with the
same `lora-control` scenario; CivitAI naming alone must not bypass this header gate.

The opt-in manual runner `scripts/wan5-generation-tests.py` retains manifests and
artifacts for fixed single T2V/I2V, three warm repeats, T2V→I2V→T2V switching,
cancellation/recovery, and LoRA/control scenarios. It calls only the public HTTP API
through `scripts/hardware-study.py` and is intentionally excluded from routine CI.

## Hard gaps and source conflicts

1. **Schedule mismatch retained by design:** the BF16 reference remains at 50 steps;
   the accepted Comfy operation freezes the exact 30-step example schedule. No
   output-quality comparison between those paths is claimed.
2. **Topology mismatch:** Engine's complete 34.2 GB BF16 folder is not the official
   Comfy split-component closure.
3. **Operation boundary:** I2V is a distinct required-image schema with its own accepted
   output; generic upstream support alone is still not counted as Engine parity.
4. **Publisher memory claim not reproduced:** this run peaked at 15,413 MiB sampled
   VRAM; Engine does not repeat the publisher's 8 GB claim as a workstation result.
5. **Pinned template/runtime:** updates to Comfy or its example require a deliberate
   revision/hash bump and requalification.
6. **Quality breadth:** one official-size motion case plus lifecycle diagnostics proves
   operability, not every creator-quality category in the corpus above.

## Ordered next actions

1. Keep the complete BF16 recipe as the honest dense reference; any cross-topology
   quality comparison must first make the schedules/settings explicitly comparable.
2. Expand T2V creator-quality coverage beyond the accepted fox-motion case when a
   broad product-quality promotion is needed.
3. Expand the accepted I2V source-image and motion corpus if product-quality promotion
   beyond the current operational proof is needed.

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
