# Ideogram 4 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official source: [`ideogram4@990fe1c`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)  
Official Comfy baseline: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) and [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c)

## Decision and hard gate

Ideogram 4 has no published dense BF16/FP16 teacher. The first-party public baseline is gated NF4 Diffusers; the official local Comfy graph requires **conditional model + unconditional model + Qwen3-VL-8B + Flux2 VAE**. No single diffusion file is a complete path.

Native support is blocked first by the Ideogram 4 Non-Commercial Model Agreement. After legal/product approval, the smallest truthful slice is **complete Comfy scaled-FP8 T2I with structured JSON prompting**. NVFP4 is the Blackwell challenger, not an automatic default; its component sum already exceeds 16 GB before VAE/buffers and requires staged execution.

## Product and operation boundary

| Operation | Contract | Disposition |
| --- | --- | --- |
| Local T2I | structured JSON caption; optional bounding boxes/palette; dual conditional/unconditional branch; Qwen3-VL encoding; VAE decode | First native slice after gate |
| Plain-text prompt enhancement | Hosted Ideogram magic prompt or local Qwen expansion are different preprocessing products | Record provider/model/version and expanded JSON; never hide it |
| Hosted Ideogram API | Separate hosted quality/product and magic-prompt behavior | Generic provider Fallback |
| Editing/control | No bounded first-party local native contract in this roadmap | Generic provider/Comfy |

Pinned workflows:

- [local T2I](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i.json)
- [local INT8/NVFP4-oriented T2I](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i_int8.json)
- [hosted API blueprint](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/blueprints/text_to_image_ideogram_v4.json) — API evidence only.

Canonical publisher presets:

| Preset | Steps | Guidance schedule | Distribution |
| --- | ---: | --- | --- |
| `V4_QUALITY_48` | 48 | 45 steps at 7, then 3 polish steps at 3 | `mu=0.0`, `std=1.5` |
| `V4_DEFAULT_20` | 20 | 18 at 7, then 2 at 3 | `mu=0.0`, `std=1.75` |
| `V4_TURBO_12` | 12 | 11 at 7, then 1 at 3 | `mu=0.5`, `std=1.75` |

Dimensions are 256–2048, multiples of 16, up to 6:1. Follow exact source: [Ideogram model](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/ldm/ideogram4/model.py), [encoder](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/text_encoders/ideogram4.py), [nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_ideogram4.py), [JSON prompt nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_json_prompt.py), and [bounding boxes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_bounding_boxes.py).

## Exact resource closures

| Path | Component | Immutable identity | Notes |
| --- | --- | --- | --- |
| Reference baseline | `ideogram-ai/ideogram-4-nf4-diffusers` | pin one complete gated snapshot before implementation; no dense teacher exists | First-party NF4, not lossless |
| FP8 candidate | conditional `ideogram4_fp8_scaled.safetensors` | revision `b6c440e84da24539b8457c865e7994d4d87447f5`; 9.28 GB; SHA-256 `49a946f1b0f8bcf5eab7d3b1ecc7b453c104e034cb1b592032745692724bd306` | required branch |
| FP8 candidate | unconditional `ideogram4_unconditional_fp8_scaled.safetensors` | revision `f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9`; 9.28 GB; SHA-256 `9b359007dae162cca7591d00868feea733eb7c56e56e3a214a4d5a9a2a07cd60` | required branch |
| FP8 candidate | `qwen3vl_8b_fp8_scaled.safetensors` | revision `b6c440e84da24539b8457c865e7994d4d87447f5`; 10.6 GB; SHA-256 `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` | encoder |
| NVFP4 challenger | conditional `ideogram4_nvfp4_mixed.safetensors` | revision `f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9`; 5.49 GB; SHA-256 `e7923b4b0a1129ae5afcc09e63046185688c8b09eb9a1a748cccdbde5d381609` | Blackwell candidate |
| NVFP4 challenger | unconditional `ideogram4_unconditional_nvfp4_mixed.safetensors` | same revision; 5.49 GB; SHA-256 `639e37bd1dd7ee35e23c7cfccf93a518ddc7f4587818956ec42b31e659fd6ac0` | required branch |
| NVFP4 challenger | `qwen3vl_8b_nvfp4.safetensors` | same revision; 6.31 GB; SHA-256 `e462e9e0c3b9313ae17f82040d7c77beb92d7aef3e40692d7803228dab7c3b98` | encoder |
| Shared | `flux2-vae.safetensors` | exact immutable identity must be resolved from the selected template/repository before declaration | no mutable `main` |

FP8 closure is approximately 29.2 GB before VAE; NVFP4 approximately 17.3 GB before VAE. Files are compatible only as their complete matching branch/encoder sets; do not mix FP8 and NVFP4 branches or call them numerically equivalent.

## Recipe ladder and candidates

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `ideogram4.text-to-image.native-nf4` | Reference baseline | complete first-party NF4 Diffusers snapshot; exact structured JSON; `V4_QUALITY_48`; staged offload; no claim of dense truth |
| `ideogram4.text-to-image.comfy-fp8` | Experimental first local path | exact two FP8 branches + FP8 Qwen3-VL + exact VAE; fixed preset; structured JSON; native dispatch required |
| `ideogram4.text-to-image.comfy-nvfp4` | Experimental Blackwell challenger | complete matching NVFP4 branch/encoder closure; no component mixing; native SM120 dispatch |

A new typed Ideogram component recipe is mandatory. Roles: `conditional_transformer`, `unconditional_transformer`, `text_vision_encoder`, `vae`, optional `support`. A structured prompt schema is also a **new request/schema extension**: raw text, exact expanded JSON, normalized boxes, palette, expansion provenance. Current generic model-resource recipes cannot truthfully express dual branches.

## Loader/runtime implementation packet

Reuse Engine resource acquisition, exact component signatures, `stored_quant.py`, `runtime/kit.py`, caches, manager/residency policy, and Klein materializer patterns. New family recipe/runtime/tool/request modules and tests are likely.

Fail-closed gates:

- complete matching four-role closure; reject one branch or mixed quant families;
- exact branch tensor maps, dense exceptions, fused projections, Qwen hidden-state selection, aliases/tied weights, scales/sidecars;
- JSON schema/bounding-box normalization and deterministic serialization before load;
- actual Kitchen FP8/NVFP4 dispatch with zero dense/eager fallback;
- no hidden hosted prompt expansion in a local recipe.

Lifecycle: validate/normalize JSON → optional expansion (separate provider operation) → stage Qwen and cache exact hidden states → release encoder device residency → stage/swap conditional and unconditional branches according to the official sampling implementation → release branches → VAE decode. Runtime fingerprint includes all four resources, preset, JSON, expansion identity, attention/offload, and quant layout. Cancellation during either branch poisons both model states and discards partial prompt caches.

## Hardware/scientific acceptance packet

Fixed corpus: multilingual posters/signage/menus/logos; exact spelling/line breaks; box placement; palette hex values; object counts/relations; 1024², portrait, landscape, long banner, and 2048² where feasible. Use fixed JSON and seed; separately test hosted/local/no expansion.

Scenarios: cold/warm, FP8→NVFP4→FP8, cancellation during expansion/Qwen/each branch/denoise/decode, malformed/mixed closure, missing branch, invalid box/palette, and explicit teardown. Assertions: exact JSON and expansion provenance, all resource identities, conditional/unconditional dispatch counts, branch residency order, cache state, preset/schedule, output hash. Local NF4/FP8/NVFP4 outputs can be compared for creator value, but no result can quantify loss against an unpublished dense teacher.

## Ordered bounded slices

1. **Human gate — license/product decision.** Stop if native gated non-commercial weights do not belong in Engine.
2. **Structured JSON request and complete FP8 loader.** One operation: T2I at `V4_QUALITY_48`; exact four-role closure. Tests: schema/box/palette, branch completeness, header maps, dispatch fail-closed, cancellation. Out of scope: NVFP4, hosted magic prompt, editing.
3. **Target-hardware FP8 acceptance.** Fixed typography/design corpus, cold/warm, branch staging, teardown.
4. **NVFP4 challenger.** Same operation/JSON/preset; complete matching closure only. Promote only with material creator/memory/latency benefit.
5. **Prompt-expansion provider seam.** Separate request/provenance operation after local inference is stable.

## Primary sources

- [Ideogram 4 source](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)
- [Architecture](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/model_architecture.md)
- [Prompting](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/prompting.md)
- [Inference presets](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/inference.md)
- [Comfy Ideogram artifacts](https://huggingface.co/Comfy-Org/Ideogram-4)
