# Z-Image and Z-Image Turbo implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [Turbo INT8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json)
- [Turbo standard workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo.json)
- [Base standard workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image.json)
- [ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [Comfy Kitchen `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4)
- [ConvRot conversion source `1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341bb8a4e46f161a978b5faa2412d8c39c768/quant_int8_convrot.py)

## Decision

Z-Image Turbo remains the strongest bounded new image family. Its pinned official Comfy INT8 graph has exactly three active model artifacts, a simple eight-step T2I topology, and no optional image input. Base is a separate longer guided line. No official first-party Z-Image Edit artifact was verified, so native I2I/edit must fail closed.

## Product and operation boundary

| Line/operation | Exact boundary | Disposition |
| --- | --- | --- |
| Turbo T2I | prompt, dimensions, seed; 8 steps, CFG 1, AuraFlow shift 3, `res_multistep`/`simple` | Next native candidate |
| Base T2I | separate Base weights and long guided schedule | Alternate/reference after Turbo |
| ControlNet/pose/canny/depth/upscale | separate official/community graphs | Generic Comfy until a precise product slice earns native support |
| I2I/edit | no official first-party edit artifact found | Unsupported; fail closed |

## Exact official topology

The pinned Turbo INT8 graph is:

1. `UNETLoader` with `z_image_turbo_int8_convrot.safetensors`.
2. `CLIPLoader` with `qwen_3_4b_fp8_mixed.safetensors`, type `lumina2`.
3. `VAELoader` with `ae.safetensors`.
4. positive `CLIPTextEncode`; the same conditioning passes through `ConditioningZeroOut` as negative.
5. `EmptySD3LatentImage`, default 1024 by 1024, batch one.
6. `ModelSamplingAuraFlow`, shift `3`.
7. `KSampler`: 8 steps, CFG 1, `res_multistep`, `simple`, denoise 1.
8. `VAEDecode` and `SaveImage`.

Relevant Comfy source at the verified commit:

- [Z-Image nodes](https://github.com/Comfy-Org/ComfyUI/blob/725e6ec60621c6f001af04769173e7dbb3c53541/comfy_extras/nodes_zimage.py)
- [Lumina model](https://github.com/Comfy-Org/ComfyUI/blob/725e6ec60621c6f001af04769173e7dbb3c53541/comfy/ldm/lumina/model.py)
- [Z-Image encoder](https://github.com/Comfy-Org/ComfyUI/blob/725e6ec60621c6f001af04769173e7dbb3c53541/comfy/text_encoders/z_image.py)
- [latent formats](https://github.com/Comfy-Org/ComfyUI/blob/725e6ec60621c6f001af04769173e7dbb3c53541/comfy/latent_formats.py)

## Exact Turbo INT8 resource closure

| Role | Repository/file and revision | Bytes | SHA-256 | Format/provenance |
| --- | --- | ---: | --- | --- |
| Transformer | `Comfy-Org/z_image_turbo`, `split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors`, `d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e` | 6,201,001,296 | `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` | official Comfy stored INT8 ConvRot |
| Text encoder | [`qwen_3_4b_fp8_mixed.safetensors`, `2f862278568d3f0a83167a16e5f11094da6dee72`](https://huggingface.co/Comfy-Org/z_image_turboblob/2f862278568d3f0a83167a16e5f11094da6dee72/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors) | 5,631,994,051 | `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` | official Comfy mixed stored FP8 |
| VAE | `Comfy-Org/z_image_turbo`, `split_files/vae/ae.safetensors`, path revision `93fae7d7f6189cc408fdd7cec36c91447b8506a2` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38g | official Comfy VAE |

Total fixed closure: `6,201,001,296 + 5,631,994,051 + 335,304,388 = 12,168,299,735` bytes.

The mutable discovery pages are [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) and [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). The exact transformer and VAE identities above were resolved from their file histories and pointer metadata; mutable `main` is not an acquisition lock.

The three artifacts are compatible components selected by the exact Comfy graph. Byte identity is claimed only for the filenames/revisions/hashes above. Community files with the same basename and different hashes are not substitutes.

## Recipe ladder and candidate contract

| Candidate key | Tier | Fixed contract |
| --- | --- | --- |
| `z-image-turbo.text-to-image.comfy-int8-convrot` | Recommended candidate | exact three resources; native attention; Engine staged residency; no compile; prompt cache; 8 steps; CFG 1; shift 3; `res_multistep`/`simple` 1024-square default |
| `z-image-turbo.text-to-image.native-bf16` | Reference | exact first-party Turbo BF16 closure; identical operation/schedule; adequate hardware/offload |
| `z-image.text-to-image.native-bf16` | Alternate/reference for Base | separate Base closure and schedule |

Engine needs a typed split image recipe with roles `transformer`, `text_encoder`, and `vae`, or a narrower Z-specific equivalent. It must reject arbitrary Lumina components and Base/Turbo interchange.

## Loader and runtime implementation packet

Reuse `resources.py`, `recipes.py`, `variants.py`, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, manager/residency policy, and Klein’s stored materializer/native-dispatch proof.

Likely new files: `z_image_recipe.py`, `runtime/z_image.py`, `tools/z_image.py`, built-in declarations, and tests.

Header/schema gates:

- exact INT8 ConvRot layer map, I8 weights, F32 row scales, marker JSON, group geometry, and complete target tensor map;
- mixed Qwen mapping, fused projections, dense exceptions, sidecars/scales, aliases, and tied weights;
- exact VAE key/config match;
- reject Base/Turbo mismatch, missing metadata, unsupported transpose, duplicate dense copies, arbitrary same-name community files, or runtime conversion.

Native proof requires Kitchen `int8_linear` with `convrot=true` for every eligible transformer linear and intended FP8/NVFP4 primitives for the mixed encoder, with zero eager/dequantized fallback.

Lifecycle: validate all three headers; stage encoder and cache CPU-frozen conditioning; release encoder device residency; materialize/stage transformer; run eight steps; release transformer; VAE decode/save. Fingerprint every resource and fixed schedule value. Cancellation or fallback poisons/ejects the runtime.

## Hardware and scientific acceptance

Fixed case: prompt `A cinematic editorial portrait of a clockmaker in a sunlit workshop, intricate brass tools, natural skin texture`, seed `43301611940728`, 1024-square, 8 steps, CFG 1, shift 3, `res_multistep`/`simple`.

Add typography, illustration, architecture, materials, faces/hands, and long-prompt cases.

Required scenarios: cold plus three changed-seed warm runs; changed prompt/dimensions; Z Turbo to Klein 4B to Z Turbo switching; cancellation during encoder, materialization, denoise, and decode; malformed transformer/encoder/VAE; portrait/landscape aligned buckets; explicit teardown.

Assertions: exact SHA/revision/schema fingerprints, layout counts, native dispatch counts, cache state, stage residency, effective schedule/canvas, and output hash. Record allocator memory plus approximate external GPU/RAM and Windows commit.

Quality failure modes: oversharpening, text corruption, face/hand errors, prompt truncation, aesthetic collapse relative to BF16, mixed-encoder drift, black/NaN output, and non-determinism.

## Ordered bounded slices

1. **Next: exact Turbo INT8 T2I closure and loader.** One operation/recipe and three artifacts. Tests cover TOML/schema, header maps, native dispatch, lifecycle, and malformed closure. Out of scope: Base, ControlNet, edit, LoRA.
2. **RTX 5080 acceptance.** Creator corpus, three meaningful warm runs, switching, dimensions, cancellation, teardown, provenance.
3. **Turbo BF16 Reference.** Same graph/settings on adequate hardware; no runtime-cast surrogate.
4. **Base T2I only if creator review justifies it.** Separate closure/schedule and recipe.
5. **Keep control/edit generic Comfy** until an exact first-party operation and creator value are demonstrated.

Stop on an unknown header, mismatched artifact hash, hidden dense copy, or native fallback.
