# Krea 2 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Current official Comfy baseline: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) and [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c)

## Decision and next slice

Krea 2 is a from-scratch image family, not FLUX.2. **Turbo** is the creator-facing inference line; **Raw** is the training/post-training line. The old roadmap statement that no immutable official Comfy graph existed is stale: current Comfy ships exact Turbo BF16 and INT8/ConvRot T2I templates plus an INT8 style-reference template.

The next bounded implementation slice should be **Turbo T2I using the exact official Comfy INT8 ConvRot closure**, after the Krea Community License gate is approved. It is locally plausible and removes the need to first make a 26.3 GB BF16 transformer product-viable. BF16 remains the Reference and should be qualified on larger hardware/offload, not weakened.

## Product and operation boundary

| Operation | Official evidence | Native Engine disposition |
| --- | --- | --- |
| Turbo T2I | Exact BF16 and INT8 workflow templates; 8-step distilled schedule | First native slice |
| Turbo style reference | Exact Comfy INT8 template; one style image input | Second slice only after T2I; distinct vision-conditioning input and cache |
| Raw T2I/training | First-party Raw checkpoint; upstream says train on Raw, run on Turbo | Reference/training only; no normal inference recipe |
| I2I/edit/inpaint/control | No canonical first-party native Krea 2 editing contract established here | Keep generic Comfy/provider until exact official graph and product need exist |
| Hosted Krea API | Separate hosted product | Generic provider Fallback, not local parity |

Pinned workflows:

- [Turbo BF16 T2I](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i.json)
- [Turbo INT8 T2I](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i_int8.json)
- [Turbo INT8 style reference](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_int8_image_style_reference.json)

The INT8 T2I graph loads `krea2_turbo_int8_convrot.safetensors`, `qwen3vl_4b_fp8_scaled.safetensors`, and `qwen_image_vae.safetensors`; enables prompt enhancement by default; configures the enhancer for 512 maximum tokens with thinking disabled; emits 1024×1024 by default; and runs eight denoise steps. Krea's publisher defaults are Turbo 8 steps, CFG 0, `mu=1.15`; Raw approximately 52 steps/CFG 3.5 in the canonical repository example. Any Comfy graph value that differs must be recorded as a workflow choice rather than silently reconciled.

Comfy source to follow, not reimplement from screenshots:

- [Krea model](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/ldm/krea2/model.py)
- [Krea text encoder](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/text_encoders/krea2.py)
- [family detection/config](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/supported_models.py)
- [quantized loading](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/quant_ops.py)

## Exact artifact closure

| Tier | Component | Publisher/revision/identity | Status |
| --- | --- | --- | --- |
| Reference | `turbo.safetensors` | `krea/Krea-2-Turbo`, revision `68e6019eebd5040b612105903a0c366c35cac757`; 26.3 GB; SHA-256 `78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3` | First-party BF16/mixed source; gated Community License |
| Alternate/reference for training | `raw.safetensors` | `krea/Krea-2-Raw`, immutable source commit `db3984fbc6e13b34c0064990fc2d95ac64d00058`; 26.3 GB; SHA-256 `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7` | Do not include in normal deployment closure |
| Recommended candidate | `krea2_turbo_int8_convrot.safetensors` | Comfy-Org Krea 2 repository; exact file revision, byte count, and SHA must be resolved from the workflow-selected repository before catalog publication | Official Comfy/Kitchen stored INT8 ConvRot; not runtime-generated |
| Shared | `qwen3vl_4b_fp8_scaled.safetensors` | Comfy-Org workflow-selected encoder; exact immutable file identity required before publication | Stored scaled FP8; role is prompt/vision-language encoder |
| Shared | `qwen_image_vae.safetensors` | Comfy-Org Qwen image VAE; exact revision and SHA must be coherent with the selected workflow snapshot | Shared VAE, compatible by Comfy graph; byte identity must be pinned |

Unknown exact bytes are a **publication stop**, not permission to use mutable `main`. Use HF API/LFS pointer metadata and header ranges; do not download the payload merely to discover its identity.

### License/gate

Krea 2 uses the Krea 2 Community License. At the audited first-party commit it limits community-license commercial use to organizations under US$1M trailing-twelve-month company-wide revenue, requires content filtering, and has redistribution/notice obligations. Product/legal approval is required before a built-in downloader or Recommended label.

## Recipe ladder and TOML-ready candidates

| Candidate key | Tier | Fixed resources and runtime |
| --- | --- | --- |
| `krea2.text-to-image.native-turbo-bf16` | Reference | complete exact Turbo source closure; BF16/mixed stored weights; native attention; sequential/model offload; no compile; prompt cache; VAE staged; 8 steps, CFG 0, `mu=1.15` |
| `krea2.text-to-image.comfy-turbo-int8-convrot` | Recommended candidate | exact three-file Comfy closure; stored INT8 ConvRot transformer, scaled-FP8 Qwen3-VL, Qwen image VAE; prompt enhancement behavior fixed and provenance-recorded; native Kitchen dispatch required |
| `krea2.style-reference.comfy-turbo-int8-convrot` | Experimental follow-on | same closure plus exactly one ordered style image and exact workflow preprocessing/conditioning |

Current Engine cannot express this component set through a generic complete-model resource alone. Add a typed Krea recipe/component contract (or a reusable typed image-component contract) in `variants.py`/a family recipe module. Proposed component roles are `transformer`, `text_encoder`, `vae`, and optional `support`; these are **schema extensions**, not valid current TOML until implemented. Do not expose runtime quantization, arbitrary component mixing, or Raw/Turbo swapping.

## Loader/runtime implementation packet

Likely reuse: `resources.py`, `recipes.py`, `variants.py`, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, `runtime/manager.py`, `runtime/residency_policy.py`, plus Klein's stored-component planner/materializer patterns. Likely new files: `krea2_recipe.py`, `runtime/krea2.py`, `tools/krea2.py`, family tests, built-in resource/recipe declarations.

Required checks before allocation:

1. exact file size/hash/revision and SafeTensors schema fingerprint;
2. all transformer quant descriptors are `int8_tensorwise` with `convrot=true`, supported persisted group sizes, int8 weights, F32 row scales, and complete global layer mapping;
3. Qwen scaled-FP8 layout, fused projections, dense exceptions, tied aliases, and sidecars match Comfy source;
4. no runtime conversion/repacking and no dense duplicate;
5. Kitchen `int8_linear` dispatch occurs for every eligible transformer linear and the intended FP8 backend for the encoder, with zero eager/dequant fallback.

Lifecycle: validate and preprocess prompt → optionally run prompt enhancer and retain both raw and enhanced text in provenance → stage Qwen, cache conditioning on CPU → release encoder device residency → run eight transformer steps → release transformer → VAE decode/save. Warm key includes raw prompt, enhancement mode/model/config, enhanced prompt, exact resources, schedule, canvas, LoRAs, and runtime policy. Cancellation or fallback poisons the runtime and invalidates partial prompt/style caches.

## Hardware/scientific acceptance packet

Fixed T2I case: 1024×1024, seed `43301611940728`, eight steps, CFG 0, `mu=1.15`; run once with prompt enhancement disabled and once with the workflow default enabled. Use editorial photography, fashion/material, typography, illustration, and long art-direction prompts.

Scenarios: cold, three warm repeats, malformed header, cancellation during enhancement/encoder/materialization/denoise/decode, A→B→A between INT8 and BF16 when larger hardware is available, and one style-reference case later. Assert exact artifacts, enhanced-prompt text/hash, native dispatch counts, stage residency, cache hit/miss, effective dimensions, and no fallback. External GPU sampling is approximate; capture Torch allocator peaks and Windows commit separately.

Quality failures: aesthetic flattening, prompt-enhancer drift, text corruption, facial/anatomy errors, reduced diversity, style-reference overtake, and 2K texture collapse. Publisher measurements and Engine results remain separate.

## Ordered bounded slices

1. **Next — Turbo INT8 T2I closure and loader.** Exact three resources + pinned T2I workflow. Likely files above. Tests: catalog/header/schema, tensor mapping, fail-closed dispatch, mocked lifecycle. Hardware: one 1024² cold/warm/cancel/recovery case. Out of scope: style reference, Raw, LoRA, 2K promotion. Stop if license gate is not approved or exact identities/native dispatch cannot be proven.
2. **Target-hardware qualification.** Same recipe only; fixed corpus, prompt-enhancement A/B, three warm repeats, teardown, provenance. Stop on creator-visible regression or paging-thrash.
3. **One-image style reference.** Reuse closure; add explicit image role/preprocessing/cache. Out of scope: multiple references and generic edit.
4. **BF16 cloud Reference.** Run exact Turbo BF16 on larger hardware for scientific comparison; do not distort it to fit locally.

## Primary sources

- [Krea 2 source and license at `db3984f`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058)
- [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)
- [Current official Comfy workflows](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1/templates)
- [Comfy quantized operations](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/quant_ops.py)
- [Kitchen INT8 layouts](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)
