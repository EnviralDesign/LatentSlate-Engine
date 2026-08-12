# Wan 2.2 TI2V 5B implementation roadmap

Last audited: **2026-08-12**  
Portfolio architecture baseline: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Post-dispatch Wan 5B implementation audited: [`f59c3970d7ca72d63533f9eb37d8f0dcc91b2810`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810)  
Official workflow evidence: [Comfy examples `f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3), [audited ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), and executable Comfy pin `eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`

## Decision

Wan 2.2 TI2V 5B is no longer an unimplemented packet. Current `main` contains distinct package-owned **T2V** and required-image **I2V** recipes over the same exact official Comfy component closure, an isolated pinned Comfy worker, one optional fail-closed model-only LoRA slot, and target-RTX-5080 operational acceptance for both operations.

The tiering remains conservative:

- **Reference:** complete first-party BF16 Diffusers repository, qualified per operation on adequate hardware and at a settings-equivalent schedule.
- **Recommended candidate:** the exact Comfy FP16-transformer/scaled-FP8-UMT5/FP16-VAE recipes. They are hardware accepted, but creator-quality breadth and a matching BF16 comparison are incomplete.
- **Fallback:** user-owned generic Comfy workflows for unsupported Wan graphs.
- **Deferred/rejected:** Turbo, Lightning, transformer FP8/NVFP4/INT8/GGUF/W4 variants until the accepted official split path leaves a measured creator-visible gap.

Do not dispatch another Wan 5B loader. The next work is quality breadth and, only if worth the 34.2 GB acquisition, a settings-equivalent BF16 reference study.

## Product and operation boundary

| Operation | Exact request boundary | Shared closure/runtime | Distinct semantics |
| --- | --- | --- | --- |
| T2V | prompt, negative prompt, dimensions, `4k+1` frames, seed; **no image field** | same transformer, UMT5, VAE and warm worker as I2V | text-only latent construction; separate recipe/workflow fingerprint |
| I2V | all T2V controls plus **exactly one required source image** | same component fingerprint; worker can warm-switch T2V→I2V→T2V | bilinear center-crop provenance, VAE encode, first-latent anchor, source-fidelity acceptance |
| Model-only LoRA | zero or one exact TI2V-5B SafeTensors adapter | inserted before `ModelSamplingSD3` with `LoraLoaderModelOnly` | immutable adapter hash/schema/rank, strength, 600 tensors/300 patch targets, zero unmapped keys |
| FLF/control/reference-video/community accelerators | distinct inputs, nodes, artifacts, or schedules | not part of this accepted closure | generic Comfy unless separately productized |

The model supports T2V and I2V in one lineage, but Engine correctly exposes two non-ambiguous operations. I2V does not arise by making the T2V image optional.

## Exact official/default Comfy topology

Current Engine pins:

- source audit revision `725e6ec60621c6f001af04769173e7dbb3c53541`;
- executable checkout `eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`;
- workflow examples revision `f9431bb000ce792094ff345446e22cac1ea6cef3`;
- T2V workflow SHA-256 `e7913b6b2c8f7d82a6a6f9940289bf6e7513cc908bbf455e4553de9804c6f571`;
- I2V workflow SHA-256 `c9408303c6d57b60aa10585d26fc2e10c9c221d2f85a28048cbe2cdba2dc5e12`.

The frozen graph is:

1. `UNETLoader` with `wan2.2_ti2v_5B_fp16.safetensors`, stored FP16 and no conversion.
2. `CLIPLoader(type="wan")` with scaled-FP8 UMT5-XXL.
3. `VAELoader` with the 48-channel Wan 2.2 VAE.
4. Optional `LoraLoaderModelOnly`, then `ModelSamplingSD3(shift=8)`.
5. Positive and negative `CLIPTextEncode`.
6. `Wan22ImageToVideoLatent` at the requested width, height, frame count, and batch one. I2V alone connects `start_image` from `LoadImage`.
7. `KSampler`: 30 steps, CFG 5, `uni_pc`, `simple`, denoise 1.
8. `VAEDecode` and `SaveWEBM`: VP9, CRF 18, 24 fps.

Dimensions must be positive multiples of 32 and no larger than the 1280×704 pixel budget. Frames must be `4k+1` and at most 121. The complete BF16 Engine path remains a separate 50-step/CFG-5 topology; its output is not a precision-only comparison with this 30-step Comfy graph.

## Exact resource closure

| Tier/path | Role | Immutable identity | Contract |
| --- | --- | --- | --- |
| Reference | complete `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | revision `b8fff7315c768468a5333511427288870b2e9635`; **34,203,021,834 bytes** | first-party Apache-2.0 complete BF16 repository; not downloaded for the accepted split-path study |
| Recommended candidate | transformer `wan2.2_ti2v_5B_fp16.safetensors` | Comfy-Org revision `fb1388adc906ab39ffc26ee40e96b22886b56bc4`; file commit `5ca2dfecf59320b1d4605b5802e64f77a8676afe`; **9,999,658,848 bytes**; SHA-256 `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | native FP16, architecture `wan22_ti2v_5b_48ch_30block` |
| Shared | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | Comfy-Org revision `06e001fc51048fb03433a6fb25334de7836704a5`; file commit `dfcea77bcf258496e20c69cd84e8e8e41909bb3b`; **6,735,906,897 bytes**; SHA-256 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | stored legacy scaled FP8 E4M3 UMT5-XXL |
| Shared | `wan2.2_vae.safetensors` | Comfy-Org revision `fb1388adc906ab39ffc26ee40e96b22886b56bc4`; file commit `8441d066add15eae8d84f42aa6d9c45417973ce6`; **1,409,400,960 bytes**; SHA-256 `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | native FP16 48-channel Wan 2.2 VAE |

Exact fixed split closure: **18,144,966,705 bytes**. The complete BF16 directory and split path are compatible model topologies, not byte-identical or numerically equivalent representations.

### Exact accepted LoRA evidence

| Adapter | Immutable identity | Disposition |
| --- | --- | --- |
| `ostris/wan22_5b_i2v_crush_it_lora` | revision `e4b85be20d75c2ca2ee1b901ba2cf49d9416e233`; 161,293,208-byte rank-32 BF16 file; SHA-256 `00a3ed72d8e257b416e1232cce07acf76cfb3ad7538f8ba995b6818f0b560f23`; trigger `crush it` | installed and public-API accepted |
| `AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets` | revision `fb47fbdfb7fa391ed6d29f1d1b06f78bc815d7c0`; 322,511,512-byte rank-64 FP16 file; SHA-256 `5c2fc21b1e74d5088318fea72c676181650a0f771cc521151edfc43f6ea9ec77` | exact catalog option; not locally installed/qualified |

Community provenance is explicit. A Civitai name or claimed base is never enough to bypass header/schema/rank validation.

## Current recipe contract

These are implemented current keys, not proposals:

| Key | Tier/proof | Fixed resources and runtime |
| --- | --- | --- |
| `wan-2-2-5b-ti2v.text-to-video.comfy-fp16` | Experimental, **hardware accepted** | exact three components; recipe type `wan22_ti2v5b_comfy_t2v`; fixed 30-step graph; one optional model-only LoRA; keep worker loaded |
| `wan-2-2-5b-ti2v.image-to-video.comfy-fp16` | Experimental, **hardware accepted** | same component closure; recipe type `wan22_ti2v5b_comfy_i2v`; exactly one required source image; same LoRA slot and warm worker |
| complete-folder native BF16 T2V key | Reference/Experimental | 34.2 GB complete repository; 50-step native runtime; structural reference, not settings-equivalent acceptance |

No further schema extension is required for these two operations. Current `wan22_ti2v5b_recipe.py` provides explicit transformer/text-encoder/VAE roles, operation-specific workflow hashes, component and recipe fingerprints, exact schema fingerprints, post-catalog revalidation, and fail-closed role/architecture checks.

## Loader/runtime implementation packet at current `main`

Implementation truth lives in:

- `wan22_ti2v5b_recipe.py` — exact three-role recipe, source/runtime/workflow pins, schema hashes, component fingerprint shared across operations, operation-specific recipe fingerprint;
- `runtime/wan5_comfy.py` — one isolated loopback worker, exact checkout verification, hardlinked validated components, no custom nodes, deterministic/classic cache settings, bounded startup/job polling, atomic output publication, interruption and unload on any failure;
- `tools/wan5_comfy.py` — distinct T2V/I2V public tools and request validation, source-image hash/crop/VAE-anchor provenance, exact optional LoRA validation, runtime-manager warm reuse, poison/ejection semantics;
- package resource declarations, two built-in recipes, one profile, unit/structural tests, and `scripts/wan5-generation-tests.py` for opt-in public-API hardware scenarios.

The runtime may switch operations only when the component fingerprint is identical; the operation-specific recipe fingerprint and workflow hash remain distinct. It does not materialize or reimplement Comfy tensors in Engine, so “native Kitchen dispatch counts” are not the proof boundary here. The proof is exact artifact/header identity, exact Comfy checkout and submitted graph, logged staged component loads, zero unmapped LoRA keys, isolated execution, and fail-closed lifecycle.

Lifecycle: validate and revalidate all three files → start/verify isolated Comfy worker → hardlink only validated components → optionally upload one I2V source and one exact LoRA → submit exact graph → poll with cancellation → inspect LoRA log delta → atomically download output → keep warm only on success. Cancellation or any exception interrupts and unloads the worker, clears its execution cache, and releases accelerator memory.

## Hardware/scientific acceptance packet

Target acceptance used the public catalog/job/artifact API on the RTX 5080. The frozen official case was 1280×704, 121 frames, 24 fps, seed `20260812`, 30 steps, CFG 5, `uni_pc`/`simple`, shift 8.

| Case | Result | Timing/peak | Proven artifact assertion |
| --- | --- | --- | --- |
| T2V official contract | succeeded | 372.69 s public API; 15,413 MiB sampled GPU; 51.55 GB sampled system RAM | VP9/yuv420p, 1280×704, 24 fps, 5.042 s; SHA-256 `ce16cb827d4bfcd9ced2eb3fcc4e55a80055f9d5456b0e425fbde1c877ca4341` |
| T2V cancellation/recovery | canceled and recovered | 4.06 s cancel latency; 25.11 s diagnostic recovery | GPU released; clean subsequent output |
| T2V warm identical request | succeeded via Comfy execution cache | 0.52 s API | byte-identical cache hit; **not** an independent stochastic rerun |
| T2V→I2V→T2V diagnostic | all succeeded in one worker | 25.95 s cold T2V; 9.31 s warm I2V; 6.55 s warm T2V | one component fingerprint, distinct recipe/workflow fingerprints |
| I2V official contract | succeeded | 442.73 s public API; 15,613 MiB sampled GPU; 50.91 GB sampled system RAM | VP9 1280×704/24 fps/5.042 s; SHA-256 `75fd03c57710a69b0accc82cd9ea47e016c1bf38850c47416d79159fc90c6d22` |
| I2V first-frame fidelity | succeeded | included above | exact recorded center-crop anchor; first-frame MAE 1.32/255, PSNR 44.04 dB |
| I2V cancellation/recovery | canceled and recovered | 2.33 s cancellation; 28.64 s recovery | worker/GPU released; Windows log-handle cleanup defect fixed before acceptance |
| Crush-It LoRA/control | both succeeded | 29.17 s cold control; 7.55 s warm LoRA diagnostic | different submitted graph hashes; 600 adapter tensors, 300 targets, zero unmapped warnings |

These are Engine measurements, not publisher memory claims. They prove operational T2V/I2V, switching, cancellation recovery, and one LoRA. They do **not** prove broad creator-quality superiority, a BF16-equivalent quality comparison, or independent warm-run variance.

The next quality corpus should cover static/slow/fast motion, articulated people, animals, vehicles/products, camera moves, scene transitions, text/signage, negative-prompt sensitivity, source identity, prompt-versus-image balance, temporal texture, and first-frame freeze/corruption. Keep allocator peaks, device sampling, Windows commit, worker logs, stage timings, exact submitted graph, and output hashes separate.

## Ordered bounded slices

1. **Next — broaden T2V and I2V creator-quality acceptance.** Existing recipes/runtime only. Run three non-cache warm requests per operation with changed seeds/prompts, portrait/landscape buckets, source-image variety, cancellation/recovery, and retained provenance. Likely files: only opt-in hardware-study script/docs unless a defect surfaces. Out of scope: new formats, new operations, runtime rewrite. Stop on broad quality failure, stale source reuse, poison, or output corruption.
2. **Settings-equivalent BF16 Reference on adequate hardware.** Create a separate qualification graph/recipe at the same 30-step schedule rather than mutating the accepted 50-step path. Compare quality, lifecycle, and resource cost per operation. Out of scope: weakening BF16 or calling a 50-step result precision-only evidence. Human gate: whether the 34.2 GB download/cloud run is worth it.
3. **Optional LoRA breadth.** Qualify another exact adapter only when creator demand exists; reuse the 600-tensor/300-target header and log gate. Out of scope: arbitrary adapters or stacked LoRAs.
4. **Stop.** Do not implement another Wan 5B loader or quantization family until the accepted split path demonstrates a concrete shortfall.

## Primary sources

- [Wan 2.2 source `42bf4cf`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc)
- [TI2V 5B model](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Complete Diffusers repository](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [Official Comfy artifact repository](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Current Engine Wan 5B implementation](https://github.com/EnviralDesign/LatentSlate-Engine/tree/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine)
