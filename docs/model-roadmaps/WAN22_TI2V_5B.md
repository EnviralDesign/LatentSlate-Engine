# Wan 2.2 TI2V 5B implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official workflow evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c)

## Decision and coordination boundary

Wan 2.2 TI2V 5B is the consumer-class Wan line and one model supports both T2V and I2V, but the operations have distinct request/preprocessing/acceptance contracts. At the audited commit Engine has a complete-repository BF16 **T2V** recipe/runtime; I2V is absent. Another agent is actively implementing this family, so this packet is the implementation/acceptance source, not permission to duplicate or overwrite that work.

The product ladder is:

- **Reference:** exact complete first-party BF16 Diffusers repository.
- **Recommended candidate:** official Comfy split FP16 transformer + scaled-FP8 UMT5 + Wan 2.2 VAE after current implementation/acceptance.
- **Fallback:** current complete BF16 recipe when hardware permits.
- **Deferred:** I2V until T2V lifecycle is accepted; community Turbo/quant zoo.

## Product/operation boundary

| Operation | Inputs | Shared artifacts | Distinct work |
| --- | --- | --- | --- |
| T2V | prompt, negative prompt | transformer, UMT5, VAE, scheduler/tokenizer support | no image preprocessing/encode; first acceptance slice |
| I2V | prompt, negative prompt, exactly one source image | same model lineage and core components | `Wan22ImageToVideoLatent`, resize/crop provenance, image/VAE encode, anchor fidelity, separate request/tool/tests |
| Fun inpaint/control | media/masks/control inputs and community artifacts | some Wan components | generic Comfy; not native first-party TI2V contract |

Pinned workflows:

- [official current TI2V 5B workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_wan2_2_5B_ti2v.json)
- [Fun inpaint](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_wan2_2_5B_fun_inpaint.json)
- [Fun control](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_wan2_2_5B_fun_control.json)

The official Comfy topology loads `wan2.2_ti2v_5B_fp16.safetensors`, scaled-FP8 UMT5-XXL, and `wan2.2_vae.safetensors`; I2V constructs a conditioned video latent from one image. Earlier Comfy example evidence used 30 steps, CFG 5, `uni_pc`, `simple`, denoise 1, and model-sampling shift 8. Engine's audited runtime uses 50 steps/CFG 5. Freeze one schedule per recipe and label it; do not attribute schedule changes to artifact layout.

Comfy source to follow: [Wan model implementation](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/ldm/wan), [Wan nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_wan.py), [model management](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/model_management.py).

## Exact resource closure

| Tier/path | Role | Immutable identity | Notes |
| --- | --- | --- | --- |
| Reference/current Engine | complete `Wan-AI/Wan2.2-TI2V-5B-Diffusers` directory | revision `b8fff7315c768468a5333511427288870b2e9635`; **34,203,021,834 bytes** | first-party Apache-2.0, complete BF16 folder |
| Recommended candidate | `wan2.2_ti2v_5B_fp16.safetensors` | Comfy-Org revision `49f4d34972b94c6079febaf2a8bbba3452f3f2a9`; **9,999,658,848 bytes**; SHA-256 `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | first-party/Comfy FP16 transformer |
| Shared | `wan2.2_vae.safetensors` | revision `9c311dda91b13fb3c970f9f72971d4df87c9eb00`; **1.41 GB**; SHA-256 `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | Wan 2.2 VAE |
| Shared | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | use one coherent official Comfy source; Engine's equivalent exact resource is **6,735,906,897 bytes**, SHA-256 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | stored scaled FP8 encoder |
| Shared | support/config/tokenizer/scheduler | exact filtered allow-pattern closure from first-party Diffusers snapshot | exact file list/bytes must be declared; no mutable or oversized snapshot |

The complete folder and split component path are compatible topologies, not byte-identical closures. A/B quality must hold schedule and remaining components fixed as far as the upstream representations permit.

## Recipe candidates

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `wan-2-2-5b-ti2v.text-to-video.native-bf16` | Reference/Fallback | existing complete BF16 resource; native attention; explicit UMT5 subprocess/cache lifecycle; exact 24 fps/frame grid; selected 50-step Engine schedule |
| `wan-2-2-5b-ti2v.text-to-video.comfy-fp16` | Recommended candidate | split FP16 transformer + scaled-FP8 UMT5 + VAE + filtered support; official Comfy schedule frozen; staged offload; no conversion |
| `wan-2-2-5b-ti2v.image-to-video.comfy-fp16` | Deferred next operation | same exact component closure plus one source-image slot and exact Comfy latent construction |

The split recipe requires a typed component contract if the active Wan 5B implementation does not already add one. Roles: `transformer`, `text_encoder`, `vae`, `support`. Do not make image optional in a single ambiguous operation request; register T2V and I2V separately while sharing the runtime seam.

## Loader/runtime packet at audited Engine commit

Current truth: `runtime/wan22.py` and `tools/wan22.py` implement T2V; built-in recipe/resource declarations bind the complete BF16 directory. Runtime defaults include 24 fps, default 1280×704/5 seconds, frame grid `(frames-1)%4==0`, 25–121 frames, and isolated CPU UMT5 subprocess/cache behavior.

Reuse `runtime/kit.py`, `runtime/cache.py`, `runtime/diffusers_repository.py`, manager/residency policy, resource/recipe acquisition, and current Wan runtime. For split closure add exact header/config schema validation; source-to-target tensor mapping; scaled-FP8 encoder validation; support file allow-list; and a component-aware pipeline fingerprint. Lifecycle: validate all files → UMT5 encode/cache → release encoder process/device memory → transformer denoise → release transformer → VAE decode → video encode. Cancellation at any phase must kill/eject the subprocess/runtime and remove partial output.

Fail closed on missing support, transformer architecture mismatch, dtype/layout mismatch, hidden runtime casts, unsupported attention/offload, or an I2V request routed to a T2V-only pipeline.

## Hardware/scientific acceptance packet

Target workstation: Windows 11, RTX 5080 15.9 GiB, 63.8 GiB RAM, CUDA 13.

Reference case: 1280×704, 121 frames, 24 fps, five seconds, seed `43301611940728`, fixed prompt/negative prompt, and separately frozen 50-step Engine and official-Comfy schedules. Compare artifact paths only within the same schedule; compare schedules as product alternatives.

Required scenarios: cold, three warm repeats, prompt-cache miss/hit, T2V BF16→split→BF16, cancellation during UMT5 startup/encode/load/denoise/decode/export, malformed component/support, and explicit teardown. I2V later adds one source image, preprocessing hash/dimensions, first-frame reconstruction, and prompt/image balance. Provenance: all exact resources, schedule/shift/sampler, fps/frame count, UMT5 process/cache state, residency, peak allocator/Windows commit, output hash. Comfy's low-VRAM statements are publisher measurements until reproduced.

## Ordered bounded slices

1. **Active/next — exact split FP16 T2V implementation and closure.** Coordinate with the active Wan 5B agent; do not duplicate. Exact workflow/resources above. Tests: component schema/support allow-list, schedule, cancellation, public API output. Out of scope: I2V, community distillation, transformer quantization.
2. **T2V target-hardware acceptance.** BF16 and split path under frozen schedules; cold/warm/cancel/recovery/provenance/creator review. Stop on paging-thrash, wrong fps, corrupted frames, or stale UMT5 process.
3. **I2V reuse/extension.** One image only; exact Comfy latent node/preprocessing; separate tool/request. Tests: malformed image, cache invalidation, anchor fidelity, cancellation.
4. **Stop.** Keep Fun/control and community acceleration in generic Comfy until a separate product decision.

## Primary sources

- [Wan 2.2 code `42bf4cf`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc)
- [TI2V 5B model](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Official current Comfy workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_wan2_2_5B_ti2v.json)
- [Comfy repackaged artifacts](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Engine audited Wan runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/wan22.py)
