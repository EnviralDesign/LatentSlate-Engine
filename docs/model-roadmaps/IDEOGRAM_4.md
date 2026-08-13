# Ideogram 4 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official source audited: [`ideogram-oss/ideogram4@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [standard FP8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i.json)
- [INT8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i_int8.json)
- [ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)

## Decision

Ideogram 4 is a typography/design-focused T2I family with structured JSON prompting and separate conditional and unconditional diffusion branches. No dense BF16/FP16 public teacher was released, so the official NF4 Diffusers model is the public reference baseline, not a lossless source of truth.

The cleanest exact Comfy implementation packet is the pinned **four-file INT8 graph**:

1. conditional INT8 ConvRot model;
2. unconditional INT8 ConvRot model;
3. Qwen3-VL 8B scaled-FP8 encoder;
4. Flux2 VAE.

One diffusion file is never a complete Ideogram 4 path.

The standard FP8 template distributes an additional Gemma 4 prompt-assistant model beside Qwen. Its exact execution closure depends on the selected prompt mode. Do not call a four-file Qwen-only FP8 subset “the official standard graph” without labeling that mode/deviation. No checked-in official NVFP4 workflow was found at the pinned template commit; an NVFP4 graph is a derived substitution until one is published.

## Product and operation boundary

| Operation | Exact boundary | Disposition |
| --- | --- | --- |
| Structured-JSON T2I | user-authored JSON caption, dimensions, seed, explicit sampler preset | Native boundary worth owning after license approval |
| Plain text plus local prompt assistant | prompt assistant identity/mode and expanded JSON must be recorded | Separate mode; current standard template distributes Qwen plus Gemma support |
| Hosted magic prompt | hosted preprocessing with separate privacy/cost/version provenance | Hosted Fallback, never hidden inside local inference |
| Bounding boxes and palette | structured prompt fields, not post-hoc controls | Include in request schema after base JSON path works |
| I2I/edit/control | no exact native product boundary established here | Generic Comfy or hosted API |

The official source supports 256 to 2048 pixel dimensions in multiples of 16 and aspect ratios up to 6:1. Sampling presets remain distinct: quality 48-step, default 20-step, and turbo 12-step schedules.

## Exact official INT8 closure

| Role | Repository/file and revision | Bytes | SHA-256 | License/provenance |
| --- | --- | ---: | --- | --- |
| Conditional model | `Comfy-Org/Ideogram-4`, `diffusion_models/ideogram4_int8_convrot.safetensors`, `e18159a2e9a95cdb4ecd76f49cecdf5291849697` | 9,583,465,712 | `a9164002943463b4c7b2abd88c82a488c088acc35762651e4d8604d6ce4a163d` | Ideogram Non-Commercial Model Agreement |
| Unconditional model | `Comfy-Org/Ideogram-4`, `diffusion_models/ideogram4_unconditional_int8_convrot.safetensors`, `8532c0f76182375c10b8f082dc6b0be196ef0615` | 9,583,465,712 | `cd03ed94f244c9cb705e7d30ca0f40b5f5b004bb20674117adff88d16416c23d` | Ideogram Non-Commercial Model Agreement |
| Text/vision encoder | [`Comfy-Org/Qwen3-VL`, `qwen3vl_8b_fp8_scaled.safetensors`, `7f1d4413e3bd9ae24580b14d4113bfce872c55f0`](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/7f1d4413e3bd9ae24580b14d4113bfce872c55f0/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) | 10,588,637,512 | `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` | Apache-2.0 Qwen3-VL repack |
| VAE | [`Comfy-Org/flux2-dev`, `flux2-vae.safetensors`, `ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa`](https://huggingface.co/Comfy-Org/flux2-dev/blob/ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa/split_files/vae/flux2-vae.safetensors) | 336,213,556 | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` | Flux dev non-commercial license; separate from Ideogram weights |

Exact four-file total: `9,583,465,712 + 9,583,465,712 + 10,588,637,512 + 336,213,556 = 30,091,782,492` bytes.

Mutable discovery pages: [Comfy Ideogram artifacts](https://huggingface.co/Comfy-Org/Ideogram-4), [official NF4 Diffusers baseline](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers), and [official FP8 source](https://huggingface.co/ideogram-ai/ideogram-4-fp8). Mutable pages are not recipe locks.

The combined closure has multiple license obligations: Ideogram’s non-commercial terms govern model branches, Qwen is Apache-2.0, and the selected Flux2 VAE carries its own non-commercial license. Product/legal approval must cover the complete closure, not only the diffusion weights.

## Exact INT8 topology

Preserve the pinned `image_ideogram4_t2i_int8.json` graph:

1. structured JSON or explicit raw text input;
2. Qwen3-VL prompt/text path selected by the graph;
3. both conditional and unconditional INT8 ConvRot models;
4. asymmetric CFG behavior rather than a generic negative prompt string;
5. exact Ideogram sampler/preset behavior and flow-matching schedule;
6. dimensions from the resolution selector, rounded to the graph’s alignment;
7. Flux2 VAE decode and output save;
8. built-in safety filtering and its reported failure state.

The request provenance must store the original user text, the exact JSON caption, prompt-assistant provider/model/mode when used, and all layout fields. Prompt expansion quality must be evaluated separately from image inference.

## Alternate and deferred paths

| Path | Status | Reason |
| --- | --- | --- |
| Official NF4 Diffusers | Reference baseline | first-party public pipeline; no dense teacher exists |
| Exact four-file INT8 Comfy graph | Experimental | complete immutable closure and checked-in graph exist; no Engine loader or acceptance yet |
| Standard FP8 template | Deferred until mode is frozen | template distributes conditional/unconditional FP8, Qwen3-VL, Flux2 VAE, and an additional Gemma prompt model; active closure depends on prompt mode |
| NVFP4 | Deferred derived challenger | exact files exist, but no checked-in official NVFP4 graph at the pinned commit; staging still exceeds raw 16 GB before buffers |
| Hosted Ideogram API | Fallback | current hosted quality/magic-prompt path with different privacy/cost contract |
| Generic Comfy | Fallback | appropriate for prompt experiments and unsupported graphs |

## Recipe ladder and candidate contract

| Candidate key | Tier | Fixed contract |
| --- | --- | --- |
| `ideogram-4.text-to-image.comfy-int8-qwen-json` | Experimental | exact four files above; structured JSON; Qwen mode; dual branch; fixed sampler preset; staged residency; no runtime conversion |
| `ideogram-4.text-to-image.official-nf4` | Reference | exact first-party NF4 Diffusers repository and operation-matched prompt JSON |
| FP8/NVFP4 keys | Deferred | author only after one exact prompt mode and complete closure are pinned |

A typed Ideogram recipe needs `conditional_transformer`, `unconditional_transformer`, `text_encoder`, and `vae`. Generic `transformer` plus an optional second file is too ambiguous. Prompt-assistant components belong to a separate typed mode/closure.

## Loader and runtime implementation packet

Reuse immutable resources, deterministic closure, `stored_quant.py`, Kitchen dispatch proof, runtime fingerprints, byte-bounded prompt cache, manager poison/ejection, and staged residency.

Likely new files: `ideogram4_recipe.py`, `runtime/ideogram4.py`, `tools/ideogram4.py`, structured prompt models, built-in declarations, and tests.

Fail-closed checks:

- exact conditional and unconditional source-to-target maps, branch identity, ConvRot scales/markers, and no branch substitution;
- complete Qwen3-VL mapping and selected hidden-state extraction contract;
- VAE schema/license identity;
- JSON schema, bounding boxes, palette values, and normalized coordinates;
- reject missing branch, mixed precision families, hidden prompt assistant, runtime conversion, unsupported plain-text fallback, or dense duplicate.

Lifecycle: validate JSON and all four files; stage encoder; cache CPU-frozen prompt state; release encoder; load/execute conditional and unconditional branches according to the exact sampler; release both; decode VAE; save. Cancellation/failure must clear partial branch state and prompt JSON.

Native proof requires positive intended INT8/FP8 dispatch counts and zero eager/dequantized fallback. Branch residency and swaps must be explicit in provenance.

## Hardware and scientific acceptance

Fixed baseline: structured JSON, seed `43301611940728`, 1024-square, `V4_QUALITY_48`, exact four-file INT8 closure. Corpus covers short/long multilingual text, posters, menus, signage, labels, palette hex values, bounding boxes, object count, spatial relations, photography, illustration, product mockups, and long banners.

Required scenarios: cold plus three changed-seed warm runs; conditional/unconditional load order; malformed each-resource case; cancellation during JSON expansion, encoder, each branch, denoise, VAE, and save; Ideogram to Z-Image to Ideogram switching; 1024-square, portrait, landscape, and one 2048 diagnostic where feasible; teardown.

Record spelling/layout accuracy, exact JSON, safety-filter result, branch identities, stage residency, dispatch counts, fallback counts, memory, output hash, and creator review. No result may imply dense-teacher equivalence.

## Ordered bounded slices

1. **Next: license and structured JSON contract.** Resolve complete closure rights, define JSON schema and expansion provenance. No model loader yet.
2. **Exact four-file INT8 loader.** Implement dual branches, Qwen mode, VAE, fail-closed headers, and lifecycle tests.
3. **RTX 5080 acceptance.** Typography/layout corpus, cancellation, switching, malformed artifacts, and branch residency.
4. **NF4 reference study.** Run same JSON corpus on suitable hardware/runtime.
5. **FP8/NVFP4 only after mode closure is explicit.** Record Gemma prompt-assistant inclusion or deliberate omission; never call a reduced subset full official parity.
