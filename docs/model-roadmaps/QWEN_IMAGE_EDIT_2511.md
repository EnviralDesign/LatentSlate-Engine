# Qwen Image Edit 2511 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official Comfy baseline: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) and [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c)

## Decision and next slice

Qwen Image Edit 2511 is a multi-image **editing** family. Do not expose it as ordinary T2I. The first native slice should implement the exact current Comfy **stored INT8 ConvRot + four-step Lightning** path for one and two ordered inputs, while retaining first-party BF16/40-step as the scientific Reference on larger hardware. The current official Comfy template exposes three image sockets, but its shipped top-level example connects one; the publisher model card demonstrates two. Engine must not call three-input behavior publisher parity until separately qualified.

## Product/operation boundary

| Operation | Input multiplicity/order | Lineage/schedule | Disposition |
| --- | --- | --- | --- |
| Standard edit | Ordered list; publisher example demonstrates two | first-party BF16, 40 steps, `true_cfg_scale=4.0`, guidance 1.0, negative prompt `" "`, seed 0 | Reference |
| Lightning edit | Same edit contract | official LightX2V four-step LoRA over standard model | Reference for distilled line |
| Fused/INT8 Lightning | Comfy template has image1 plus optional image2/image3 sockets | stored quantized transformer + optional Lightning mode/LoRA; current template note contrasts 40-step Qwen with 20-step Comfy standard settings | First product candidate; freeze exact chosen mode |
| Inpaint/control | Family/ecosystem possibilities, no bounded native contract here | separate graphs | Generic Comfy |

Pinned graphs:

- [standard edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511.json)
- [INT8 edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json)
- [Lightning/inflation example](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image-qwen_image_edit_2511_lora_inflation.json)

Current INT8 graph closure is `qwen_image_edit_2511_int8_convrot.safetensors`, `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors`, and optionally `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`. It saves PNG and exposes ordered image1/image2/image3 inputs. Follow core source: [Qwen model](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/ldm/qwen_image/model.py), [Qwen text/vision encoder](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/text_encoders/qwen_image.py), and [Qwen extra nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_qwen.py).

## Exact artifact closure

| Tier/path | Role | Immutable identity | Status |
| --- | --- | --- | --- |
| Reference | complete first-party BF16 Diffusers repository | `Qwen/Qwen-Image-Edit-2511`, snapshot `9736056621eb5ea6af201b65a74862a2f2b666f3`; approximately 57.7 GB complete | Apache-2.0, first-party |
| Reference component | Comfy BF16 transformer `qwen_image_edit_2511_bf16.safetensors` | revision `b6a0794717d3f5600f85c5edcdcd0c0eb93d7446`; 40.9 GB; SHA-256 `ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c` | First-party/Comfy repack |
| Shared | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | revision `bbdcab645099df455488b29f48957efbd91f996b`; 9.38 GB; SHA-256 `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4` | Official Comfy shared encoder |
| Shared | `qwen_image_vae.safetensors` | 253,806,246 bytes; SHA-256 `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f`; resolve exact immutable repository revision before declaration | Official Comfy shared VAE |
| Distilled reference | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | LightX2V official repository; exact revision/bytes/SHA must be resolved before catalog publication | Apache-2.0 LoRA; separate schedule |
| Experimental | `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors` | revision `b89b55b986fd724a8dd6db59cb4b9005d338941a`; 20.4 GB; SHA-256 `16d03fd9bab36b0a588e77db23825629df62ab99fc3e62f3a36887b201dcc36f` | LightX2V fused scaled FP8 Lightning |
| Recommended candidate | `qwen_image_edit_2511_int8_convrot.safetensors` | Official Comfy repository/workflow; exact immutable revision, byte count, SHA, and header schema remain a publication blocker | Stored Comfy/Kitchen INT8 ConvRot |

The standard Comfy `fp8mixed` transformer is 20.5 GB, SHA-256 `c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e`, but is Deferred: it adds a standard-line quant path before the operation contract is stable.

## Candidate recipe contract

| Key | Tier | Fixed resources/settings |
| --- | --- | --- |
| `qwen-image-edit-2511.image-to-image.native-bf16` | Reference | complete pinned BF16 repository; standard 40-step settings; ordered one/two inputs; aggressive staged offload; no quantization |
| `qwen-image-edit-2511.image-to-image.comfy-int8-lightning` | Recommended candidate | exact INT8 transformer, scaled-FP8 encoder, VAE, exact four-step Lightning LoRA or proven fused equivalent; ordered one/two inputs; fixed preprocessing and schedule; native Kitchen dispatch |
| `qwen-image-edit-2511.image-to-image.lightx2v-fp8-lightning` | Experimental alternate | exact fused scaled-FP8 Lightning transformer + same encoder/VAE; four steps only |

Current Engine lacks a typed multi-image Qwen component recipe. Proposed schema extension: explicit `transformer`, `text_vision_encoder`, `vae`, optional fixed `lora`, and an ordered asset list with stable role/index. Do not encode three arbitrary resources in generic metadata or overload a single `image` field. The request contract must record every input content hash and preprocessing result in order.

## Loader/runtime implementation packet

Likely reuse `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, manager/residency policy, resources/recipes, and the Klein stored-loader patterns. Likely new `qwen_image_recipe.py`, `runtime/qwen_image_edit.py`, `tools/qwen_image_edit.py`, request models, built-in declarations, tests.

Header/loader gates:

- exact source/target tensor map, fused QKV/projection exceptions, dense tensors, scale sidecars, aliases/tied weights;
- INT8 ConvRot descriptors/group geometry and native `int8_linear`; encoder scaled-FP8 layout and dispatch;
- Lightning lineage proof: fixed LoRA identity or fused file metadata must match the selected four-step schedule;
- reject partial closures, mismatched VAE/encoder, image count > accepted contract, runtime conversion, or any dense/eager fallback.

Lifecycle: decode/validate all input assets before allocating model → preprocess in exact order → stage Qwen2.5-VL and cache conditioning keyed by ordered input hashes + prompt + preprocessing → release encoder device residency → stage transformer → denoise → release transformer → VAE decode/save. Warm keys include image order, standard-versus-Lightning lineage, fixed LoRA/fused identity, schedule, and all runtime policy. Cancellation invalidates partial image conditioning and ejects model state.

## Hardware/scientific acceptance packet

Fixed standard Reference: publisher settings above, two pinned source images, 1024-class output, seed 0. Fixed Lightning case: same sources/prompt/canvas with exact four-step schedule. Do not compare a four-step output as a pure quantization of the 40-step teacher.

Corpus: no-op/surgical edits, identity-preserving one/two-person edits, exact text replacement, geometric/product changes, material/style transfer, object insertion/removal, relighting, and untouched-region stability. Scenarios: cold, warm repeat, one→two→one input, A→B→A recipe switch, cancellation at input decode/encoder/materialization/denoise/decode, malformed header, reversed input order, and a three-input Experimental case only after one/two acceptance.

Provenance assertions: exact ordered hashes and scaled sizes, raw/effective prompts, standard/Lightning lineage, fixed LoRA or fused artifact, native dispatch counts, cache hits, residency stage, output hash. Peak GPU/RAM measurements are approximate externally; retain allocator and Windows commit data.

## Ordered bounded slices

1. **Next — exact one/two-input INT8 Lightning edit.** Implement exact Comfy closure and pinned workflow. Tests: request ordering, header/schema, fixed LoRA/fused-lineage validation, dispatch fail-closed, cancellation. Out of scope: third input, standard FP8, inpaint/control.
2. **Target-hardware acceptance.** Fixed creator corpus, cold/warm, one→two→one, cancellation/recovery, provenance and quality review. Stop on untouched-region regression, order ambiguity, paging-thrash, or fallback.
3. **Three-input Engine extension.** Only after slice 2; preserve ordered list and mark non-publisher-parity.
4. **Cloud BF16 Reference.** Run exact 40-step source on larger hardware; do not weaken it for 16 GB.
5. **Optional fused FP8 Lightning comparison.** Only if INT8 leaves a measured quality/lifecycle gap; compare against BF16+same Lightning LoRA, not standard teacher alone.

## Primary sources

- [Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Current official Comfy workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json)
- [Comfy Qwen artifacts](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI)
- [LightX2V Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- [Comfy quantized loading](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/quant_ops.py)
