# Z-Image / Z-Image Turbo implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official Comfy evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c), [Kitchen `9816d220021ab526e2cc1700a68b68d1b72d961c`](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)

## Decision and next slice

Z-Image Turbo is the strongest newly discovered local image slice in this audit. It has an exact official Comfy **three-file INT8 ConvRot closure**, a simple eight-step T2I graph, Apache-2.0 first-party lineage, and a 6B-class target intended for 16 GB hardware. Implement Turbo T2I before Base. Base remains a separate ~50-step CFG line and scientific/quality Alternate; no official Z-Image Edit release was verified, so do not invent an editing operation.

## Product and operation boundary

| Line/operation | Exact boundary | Disposition |
| --- | --- | --- |
| Turbo T2I | 8 steps, CFG 1 in current Comfy graph, AuraFlow sampling shift 3, `res_multistep` + `simple`, empty SD3 latent, Qwen text encoder, VAE decode | **Next native candidate** |
| Base T2I | separate Base weights, approximately 50-step guided line | Reference/Alternate after Turbo |
| ControlNet/pose/canny/depth/upscale | official/community blueprint ecosystem, including Fun Union paths | Keep generic Comfy until one exact first-party closure warrants native support |
| Edit/I2I | no official first-party Z-Image Edit artifact verified | Unsupported; fail closed |

Pinned workflows:

- [Turbo BF16/standard](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo.json)
- [Base standard](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image.json)
- [Turbo INT8](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json)
- [Base INT8](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_int8.json)

Exact Turbo INT8 graph:

1. `UNETLoader` loads `z_image_turbo_int8_convrot.safetensors`.
2. `CLIPLoader` loads `qwen_3_4b_fp8_mixed.safetensors` with type `lumina2`.
3. `VAELoader` loads `ae.safetensors`.
4. Positive `CLIPTextEncode`; the same conditioning passes through `ConditioningZeroOut` as negative.
5. `EmptySD3LatentImage`, default 1024×1024 batch 1.
6. `ModelSamplingAuraFlow`, shift 3.
7. `KSampler`: 8 steps, CFG 1, sampler `res_multistep`, scheduler `simple`, denoise 1.
8. `VAEDecode` and `SaveImage`.

Comfy source: [Z-Image nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_zimage.py), [Lumina model](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/ldm/lumina/model.py), [Z-Image encoder](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/text_encoders/z_image.py), [latent formats](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/latent_formats.py).

## Exact Turbo INT8 resource closure

| Role | Publisher/repository/file | Immutable identity | Format/provenance |
| --- | --- | --- | --- |
| Transformer | `Comfy-Org/z_image_turbo/.../z_image_turbo_int8_convrot.safetensors` | file commit `d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e`; **6,201,001,296 bytes**; SHA-256 `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` | official Comfy stored INT8 tensorwise ConvRot; not runtime-generated |
| Text encoder | `.../qwen_3_4b_fp8_mixed.safetensors` | file commit `2f862278568d3f0a83167a16e5f11094da6dee72`; **5,631,994,051 bytes**; SHA-256 `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` | official Comfy mixed stored FP8 |
| VAE | `.../ae.safetensors` | file commit `93fae7d7f6189cc408fdd7cec36c91447b8506a2`; **335,304,388 bytes**; SHA-256 `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` | official Comfy VAE |

Total fixed closure: **12,168,299,735 bytes**. All three resources are first-party/Comfy-Org and publicly accessible at audit; family license is Apache-2.0. Byte-identical reuse is only claimed for the exact file identities above. The official Base snapshot `Tongyi-MAI/Z-Image` at `f8ea85d935e411494f6191b771030dbe12e02504` is approximately 20.5 GB complete and is not a substitute for Turbo.

## Recipe ladder and TOML-ready candidate

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `z-image-turbo.text-to-image.comfy-int8-convrot` | Recommended candidate | exact three resources above; native attention; Engine staged residency; no compile; prompt cache; VAE staged; 8 steps; CFG 1; AuraFlow shift 3; `res_multistep`/`simple`; denoise 1; 1024² default |
| `z-image-turbo.text-to-image.native-bf16` | Reference | exact first-party Turbo BF16 complete repository/split closure; same operation/schedule; qualified on larger hardware/offload |
| `z-image.text-to-image.native-bf16` | Alternate/Reference for Base | exact Base repository; Base schedule/CFG; separate operation line |

Current Engine needs a typed split image recipe with roles `transformer`, `text_encoder`, `vae` (or a Z-specific contract). That is a schema extension. The runtime must not accept arbitrary Lumina-family components or silently load Base into Turbo semantics.

## Loader/runtime implementation packet

Reuse `resources.py`, `recipes.py`, `variants.py`, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, manager/residency policy, and Klein's stored materializer/native-dispatch proof. Likely new `z_image_recipe.py`, `runtime/z_image.py`, `tools/z_image.py`, built-in declarations, and tests.

Header/schema gates:

- exact INT8 ConvRot global layer mapping; `I8` weights; F32 row scales; marker JSON; valid group geometry; complete target tensor map;
- mixed Qwen encoder mapping, fused projections, dense exceptions, sidecars/scales, aliases/tied weights;
- VAE key/config match;
- reject Base/Turbo lineage mismatch, missing metadata, unsupported transpose, duplicate dense copies, or runtime conversion.

Native proof: Kitchen `int8_linear` with `convrot=true` for every eligible transformer linear and the intended FP8/NVFP4 primitives for the mixed encoder; zero eager/dequantized fallback. A successful image is not proof.

Lifecycle: validate all three headers → stage encoder and cache conditioning on CPU → release encoder device residency → materialize/stage transformer and run eight steps → release transformer → VAE decode/save. Fingerprint includes all resources, schedule, shift, sampler/scheduler, attention/offload, canvas, LoRAs, and Kitchen/runtime versions. Cancellation or fallback poisons/ejects the runtime.

## Hardware/scientific acceptance packet

Fixed case: prompt `A cinematic editorial portrait of a clockmaker in a sunlit workshop, intricate brass tools, natural skin texture`, seed `43301611940728`, 1024×1024, 8 steps, CFG 1, shift 3, `res_multistep`/`simple`. Add typography, illustration, architecture, material, hands/faces, and long-prompt cases.

Required scenarios: cold + three warm repeats; same-recipe changed prompt/dimensions; Z Turbo → Klein 4B → Z Turbo warm-switch; cancellation during encoder/materialization/denoise/decode; malformed transformer/encoder/VAE; 1024² plus portrait/landscape aligned buckets; explicit teardown. Assertions: exact SHA/revision/schema fingerprints, layout counts, native dispatch counts, cache state, stage residency, effective schedule/canvas, output hash. External GPU/RAM peaks are approximate; record allocator and Windows commit.

Quality failure modes: oversharpening, text corruption, facial/hands errors, prompt truncation, aesthetic collapse relative to BF16, mixed-encoder drift, NaN/black output, and non-determinism.

## Ordered bounded slices

1. **Next — Turbo INT8 T2I exact closure and loader.** One operation/recipe, exact graph/resources above. Tests: TOML/schema, header maps, Kitchen dispatch fail-closed, lifecycle mocks, malformed closure. Hardware: one cold/warm/cancel/recovery 1024² case. Out of scope: Base, ControlNet, edit, LoRA. Stop on any unknown header or fallback.
2. **Target-workstation acceptance.** Fixed creator corpus, three warm repeats, Z→Klein→Z switching, dimensions, cancellation, teardown, provenance.
3. **Turbo BF16 Reference on larger hardware.** Exact same graph/settings; no weakened quant/cast surrogate.
4. **Base T2I only if creator review justifies it.** Separate closure/schedule and recipe; never a runtime flag.
5. **Keep ControlNet/edit generic Comfy** until exact first-party operations and user value are demonstrated.

## Primary sources

- [Z-Image project](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Official Turbo INT8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json)
- [Comfy quantized loading](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/quant_ops.py)
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)
- [Official ConvRot conversion source](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341001c27e8fe7e0450e8ce7fd3333d97c34c/quant_int8_convrot.py)
