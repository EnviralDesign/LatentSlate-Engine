# Wan 2.2 TI2V 5B implementation roadmap

Last corrected: **2026-08-12**

Portfolio architecture baseline: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Post-dispatch implementation audited without merge/rebase: [`f59c3970d7ca72d63533f9eb37d8f0dcc91b2810`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810)

Official evidence:

- [Wan source `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc)
- [ComfyUI examples `f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3)
- [audited ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [executable Comfy checkout `eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f)

## Decision

Wan 2.2 TI2V 5B is already implemented on current `main`. It has distinct package-owned T2V and required-image I2V recipes over one exact three-resource Comfy closure, an isolated pinned Comfy worker, one optional fail-closed model-only LoRA slot, and target-RTX-5080 operational acceptance for both operations.

Do not dispatch another Wan 5B loader. Remaining work is broader creator-quality evidence and, only if worth the cost, a settings-equivalent BF16 reference study.

Tiering remains conservative:

- **Reference**: complete first-party BF16 Diffusers repository, qualified per operation at an equivalent schedule.
- **Recommended candidate**: exact split Comfy recipes, hardware accepted but not broadly creator-reviewed.
- **Fallback**: user-owned generic Comfy workflows for unsupported Wan graphs.

## Product and operation boundary

| Operation | Exact request boundary | Shared closure/runtime | Distinct semantics |
| --- | --- | --- | --- |
| T2V | prompt, negative prompt, dimensions, `4k+1` frames, seed; no image field | same transformer, UMT5, VAE and warm worker as I2V | text-only latent construction and separate workflow fingerprint |
| I2V | all T2V controls plus exactly one required source image | same component fingerprint; warm T2V-to-I2V-to-T2V switching | center-crop provenance, VAE encode, first-latent anchor, source-fidelity acceptance |
| model-only LoRA | zero or one exact TI2V-5B SafeTensors adapter | `LoraLoaderModelOnly` before `ModelSamplingSD3` | exact hash/schema/rank, strength, 600 tensors/300 targets, zero unmapped keys |
| FLF/control/reference-video/community accelerators | distinct inputs, nodes, artifacts, or schedules | not part of accepted closure | generic Comfy unless separately productized |

T2V and I2V are separate non-ambiguous operations. I2V is not created by making a T2V image optional.

## Exact official Comfy topology

Current Engine pins:

- source audit revision `725e6ec60621c6f001af04769173e7dbb3c53541`;
- executable checkout `eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`;
- workflow examples revision `f9431bb000ce792094ff345446e22cac1ea6cef3`;
- T2V workflow SHA-256 `e7913b6b2c8f7d82a6a6f9940289bf6e7513cc908bbf455e4553de9804c6f571`;
- I2V workflow SHA-256 `c9408303c6d57b60aa10585d26fc2e10c9c221d2f85a28048cbe2cdba2dc5e12`.

The submitted graph is:

1. `UNETLoader` with the FP16 5B transformer.
2. `CLIPLoader(type="wan")` with scaled-FP8 UMT5-XXL.
3. `VAELoader` with the 48-channel Wan 2.2 VAE.
4. optional `LoraLoaderModelOnly`, then `ModelSamplingSD3(shift=8)`.
5. positive and negative `CLIPTextEncode`.
6. `Wan22ImageToVideoLatent`; I2V alone connects `start_image` from `LoadImage`.
7. `KSampler`: 30 steps, CFG 5, `uni_pc`, `simple`, denoise 1.
8. `VAEDecode` and `SaveWEBM`: VP9, CRF 18, 24 fps.

Dimensions are multiples of 32 within the 1280-by-704 pixel budget. Frames are `4k+1`, maximum 121. The complete-folder BF16 Engine path uses 50 steps and is not a precision-only comparison with this 30-step graph.

## Exact resource closure

| Tier/path | Role | Immutable identity | Contract |
| --- | --- | --- | --- |
| Reference | complete `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | revision `b8fff7315c768468a5333511427288870b2e9635`; 34,203,021,834 bytes | first-party Apache-2.0 BF16 repository; not downloaded for accepted split study |
| Recommended candidate | transformer `wan2.2_ti2v_5B_fp16.safetensors` | Comfy revision `fb1388adc906ab39ffc26ee40e96b22886b56bc4`; file commit `5ca2dfecf59320b1d4605b5802e64f77a8676afe`; 9,999,658,848 bytes; SHA-256 `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | native FP16, exact 30-block TI2V architecture |
| Shared | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | revision `06e001fc51048fb03433a6fb25334de7836704a5`; file commit `dfcea77bcf258496e20c69cd84e8e8e41909bb3b`; 6,735,906,897 bytes; SHA-256 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | stored legacy scaled FP8 UMT5-XXL |
| Shared | `wan2.2_vae.safetensors` | revision `fb1388adc906ab39ffc26ee40e96b22886b56bc4`; file commit `8441d066add15eae8d84f42aa6d9c45417973ce6`; 1,409,400,960 bytes; SHA-256 `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | native FP16 48-channel VAE |

Exact split closure: `18,144,966,705` bytes.

Current package resources and recipes at `f59c397` are authoritative:

- [`wan22_ti2v5b_recipe.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine/wan22_ti2v5b_recipe.py)
- [managed Comfy runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine/runtime/wan5_comfy.py)
- [public tools](https://github.com/EnviralDesign/LatentSlate-Engine/blob/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine/tools/wan5_comfy.py)
- [T2V recipe](https://github.com/EnviralDesign/LatentSlate-Engine/blob/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine/builtin_recipes/wan22/wan-2-2-5b-ti2v-text-to-video-comfy-fp16.toml)
- [I2V recipe](https://github.com/EnviralDesign/LatentSlate-Engine/blob/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810/src/latentslate_engine/builtin_recipes/wan22/wan-2-2-5b-ti2v-image-to-video-comfy-fp16.toml)

## Exact accepted LoRA evidence

| Adapter | Immutable identity | Disposition |
| --- | --- | --- |
| `ostris/wan22_5b_i2v_crush_it_lora` | revision `e4b85be20d75c2ca2ee1b901ba2cf49d9416e233`; 161,293,208-byte rank-32 BF16; SHA-256 `00a3ed72d8e257b416e1232cce07acf76cfb3ad7538f8ba995b6818f0b560f23`; trigger `crush it` | installed and public-API accepted |
| `AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets` | revision `fb47fbdfb7fa391ed6d29f1d1b06f78bc815d7c0`; 322,511,512-byte rank-64 FP16; SHA-256 `5c2fc21b1e74d5088318fea72c676181650a0f771cc521151edfc43f6ea9ec77` | exact catalog option; not locally qualified |

A claimed base model or Civitai name never bypasses header/schema/rank validation.

## Loader and runtime truth

The runtime verifies the exact Comfy checkout, creates isolated input/output/temp/model roots, hardlinks only validated artifacts, disables custom nodes, binds loopback, records workflow and submitted-graph hashes, polls with cancellation, atomically publishes output, and unloads on any failure.

T2V and I2V share a component fingerprint but retain operation-specific recipe/workflow fingerprints. The proof boundary is exact artifact/header identity, exact Comfy revision and graph, staged component logs, LoRA unmapped-key checks, isolation, and fail-closed lifecycle. Engine does not claim an in-process Kitchen dispatch counter for this Comfy worker.

## Hardware and scientific acceptance

Target acceptance used the public Engine catalog/job/artifact API on the RTX 5080. Frozen official case: 1280 by 704, 121 frames, 24 fps, seed `20260812`, 30 steps, CFG 5, `uni_pc`/`simple`, shift 8.

| Case | Result | Timing/peak | Proven assertion |
| --- | --- | --- | --- |
| T2V official contract | succeeded | 372.69 s API; 15,413 MiB sampled GPU; 51.55 GB sampled system RAM | VP9 1280-by-704, 24 fps, 5.042 s; SHA-256 `ce16cb827d4bfcd9ced2eb3fcc4e55a80055f9d5456b0e425fbde1c877ca4341` |
| T2V cancellation/recovery | canceled and recovered | 4.06 s cancel latency; 25.11 s diagnostic recovery | GPU released; clean next output |
| T2V identical cache replay | succeeded | 0.52 s API | byte-identical Comfy execution-cache hit; not an independent stochastic run |
| T2V-to-I2V-to-T2V diagnostic | all succeeded in one worker | 25.95 s cold T2V; 9.31 s warm I2V; 6.55 s warm T2V | one component fingerprint, distinct operation fingerprints |
| I2V official contract | succeeded | 442.73 s API; 15,613 MiB sampled GPU; 50.91 GB sampled system RAM | VP9 1280-by-704/24 fps/5.042 s; SHA-256 `75fd03c57710a69b0accc82cd9ea47e016c1bf38850c47416d79159fc90c6d22` |
| I2V first-frame fidelity | succeeded | included above | first-frame MAE 1.32/255, PSNR 44.04 dB versus recorded crop anchor |
| I2V cancellation/recovery | canceled and recovered | 2.33 s cancellation; 28.64 s recovery | worker/GPU released; Windows log-handle defect fixed before acceptance |
| Crush-It LoRA/control | both succeeded | 29.17 s cold control; 7.55 s warm LoRA diagnostic | different submitted graph hashes; 600 tensors, 300 targets, zero unmapped warnings |

These are Engine measurements, not publisher claims. They prove operational T2V/I2V, switching, cancellation recovery, and one LoRA. They do not prove broad creator quality, BF16 equivalence, or independent warm-run variance.

The next corpus covers static/slow/fast motion, articulated people, animals, products/vehicles, camera moves, transitions, text/signage, negative-prompt sensitivity, source identity, prompt-image balance, temporal texture, and first-frame freeze/corruption.

## Reference boundary

The complete BF16 reference must use a separate 30-step qualification graph to compare precision/topology fairly. The existing 50-step complete-folder runtime remains a structural Reference but is not settings-equivalent. Do not overwrite it or call a 50-versus-30 result a precision comparison.

## Ordered bounded slices

1. **Next: broaden T2V and I2V creator-quality acceptance.** Existing recipes/runtime only. Run changed prompts/seeds and portrait/landscape cases; preserve provenance. Out of scope: new formats or runtime rewrite.
2. **Settings-equivalent BF16 Reference.** Separate 30-step qualification graph/recipe on adequate hardware. Human gate: 34.2 GB acquisition/cloud cost.
3. **Optional LoRA breadth.** Qualify another exact adapter only on creator demand; reuse the exact header and log gate.
4. **Stop.** No additional Wan 5B loader or quantization format until the accepted path demonstrates a concrete shortfall.

This roadmap intentionally preserves the branch’s reconciliation against `f59c397`; it does not merge or overwrite current `main`.
