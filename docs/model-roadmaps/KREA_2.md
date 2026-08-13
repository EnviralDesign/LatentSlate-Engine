# Krea 2 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official code audited: [`krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [INT8 T2I workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i_int8.json)
- [INT8 style-reference workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_int8_image_style_reference.json)
- [ComfyUI Krea source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)

## Decision

Krea 2 is not FLUX. It is a separate image family with Raw and Turbo lineages. Raw is the training/foundation line. Turbo is the creator-facing inference line.

The current official Comfy INT8 T2I graph is a **four-file execution closure**, not three files. It actively routes the model through `LoraLoaderModelOnly` with `krea2_darkbrush.safetensors` at strength `0.8`. Official parity therefore includes the transformer, text/vision encoder, VAE, and Darkbrush LoRA.

A three-file LoRA-disabled path may be a useful controlled Alternate, but it is a deliberate Engine deviation and must not be described as the official graph.

The Krea 2 Community License remains a product gate: commercial use under the community license has a revenue threshold, and deployment obligations include content filtering. No built-in recommendation should ship before legal/product review.

## Product and operation boundary

| Operation | Exact boundary | Disposition |
| --- | --- | --- |
| Turbo T2I, official parity | prompt, dimensions, seed, eight-step Turbo schedule, four fixed artifacts including Darkbrush LoRA at 0.8 | First implementation candidate after license gate |
| Turbo T2I, no LoRA | same transformer/encoder/VAE, Darkbrush omitted | Deliberate Alternate for isolating base Turbo behavior |
| Turbo style reference | separate official workflow with style image conditioning | Deferred until T2I loader and creator value pass |
| Raw T2I/training | Raw checkpoint, 52-step guided foundation behavior | Reference for training/foundation only |
| I2I/edit/inpaint | no exact official Krea 2 edit operation established in this audit | Keep generic Comfy; native Engine must fail closed |

The official repository states “TRAIN on Raw and RUN on Turbo.” Do not point a Turbo recipe at Raw or inherit FLUX request semantics.

## Exact official Comfy closure

The following immutable Hugging Face file pages were resolved during this correction.

| Role | Repository/file and immutable revision | Bytes | SHA-256 | License/provenance |
| --- | --- | ---: | --- | --- |
| Transformer | [`Comfy-Org/Krea-2`, `krea2_turbo_int8_convrot.safetensors`, `6b1d7191d84d5ded74d83a1a98211dad0ac8ae25`](https://huggingface.co/Comfy-Org/Krea-2/blob/6b1d7191d84d5ded74d83a1a98211dad0ac8ae25/diffusion_models/krea2_turbo_int8_convrot.safetensors) | 13,492,686,496 | `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1` | Comfy-Org stored INT8 ConvRot; Krea 2 Community License |
| Text/vision encoder | [`qwen3vl_4b_fp8_scaled.safetensors`, `4aa0eed112bd2780ceea37583edbdcd2df6c2c09`](https://huggingface.co/Comfy-Org/Krea-2/blob/4aa0eed112bd2780ceea37583edbdcd2df6c2c09/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | 5,242,467,968 | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` | Comfy-Org scaled FP8 Qwen3-VL 4B; repository license gate applies |
| VAE | [`qwen_image_vae.safetensors`, `a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e`](https://huggingface.co/Comfy-Org/Krea-2/blob/a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | Comfy-Org VAE; repository license gate applies |
| Fixed parity LoRA | [`krea2_darkbrush.safetensors`, `b5a1dcd1574c1d256408cbb5ae46a67b225481e6`](https://huggingface.co/Comfy-Org/Krea-2/blob/b5a1dcd1574c1d256408cbb5ae46a67b225481e6/loras/krea2_darkbrush.safetensors) | 469,291,992 | `f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6` | Comfy-Org/Krea distribution; Krea 2 Community License |

Arithmetic:

- LoRA-disabled three-file deviation: `13,492,686,496 + 5,242,467,968 + 253,806,246 = 18,988,960,710` bytes.
- Exact official four-file parity: `18,988,960,710 + 469,291,992 = 19,458,252,702` bytes.

The graph literal is authoritative: Darkbrush is selected and applied at `0.8`. Adjacent workflow notes and trigger/recommendation prose are not perfectly consistent with that literal setting. Engine parity must preserve the wired graph and record the conflict, not silently substitute a preferred strength.

### Publisher BF16 references

The mutable discovery pages are [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) and [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw). Turbo BF16 is the source-of-truth inference reference; Raw BF16 is a training/foundation reference. A built-in resource must resolve an immutable repository revision and every required support file before authoring. Mutable `main` is not a lock.

## Official Comfy topology

For the pinned INT8 T2I graph, preserve:

1. INT8 ConvRot Turbo transformer through core `UNETLoader`.
2. Qwen3-VL 4B scaled-FP8 encoder through the Krea text-encoder path.
3. Qwen image VAE.
4. Darkbrush through `LoraLoaderModelOnly`, strength `0.8`, in the active model path.
5. Turbo sampling defaults from the graph, including the fixed eight-step product schedule.
6. Prompt enhancement, dimensions, seed, and output encoding exactly as represented by the subgraph.

Do not claim that Krea’s general support for image conditioning proves an Engine edit operation. The separate style-reference template is evidence only for its exact style-conditioning graph.

## Recipe ladder and candidate contracts

| Candidate key | Tier | Fixed resources/runtime |
| --- | --- | --- |
| `krea-2-turbo.text-to-image.comfy-int8-darkbrush` | Experimental official-parity candidate | exact four resources above; Darkbrush fixed at 0.8; stored INT8/FP8; native Comfy/Kitchen dispatch; staged encoder/transformer/VAE; eight-step Turbo schedule; no runtime conversion |
| `krea-2-turbo.text-to-image.comfy-int8-no-lora` | Alternate, deliberate deviation | exact three-file closure; same schedule; explicit provenance `official_workflow_deviation = darkbrush_disabled` |
| `krea-2-turbo.text-to-image.native-bf16` | Reference | immutable official Turbo BF16 closure; same operation; likely larger-hardware/offload qualification |
| `krea-2-raw.text-to-image.native-bf16` | Reference for Raw only | separate Raw lineage and schedule; not normal product inference |

A typed Krea split recipe is a schema/runtime extension. Required roles are `transformer`, `text_encoder`, `vae`, and for official parity a fixed `model_lora`. The fixed parity LoRA must not be represented as an optional user slot.

## Loader and runtime implementation packet

Reuse `resources.py`, `recipes.py`, `variants.py`, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, runtime manager/residency, and the Klein mixed-Qwen/native-dispatch patterns.

Likely new files: `krea2_recipe.py`, `runtime/krea2.py`, `tools/krea2.py`, built-in resource/recipe declarations, and family tests.

Fail-closed header checks:

- complete Krea transformer source-to-target mapping, including fused projections, dense exceptions, ConvRot markers, row scales, group geometry, aliases, and sidecars;
- complete Qwen3-VL mapping, vision/text branch ownership, scaled-FP8 geometry, tied weights, and unsupported dense fallback detection;
- exact VAE schema;
- exact Darkbrush LoRA target set, rank, dtype, SHA, and fixed strength;
- reject Raw/Turbo mismatch, missing LoRA in official-parity key, extra unknown tensors, runtime conversion, or a dense duplicate.

Lifecycle: validate all four identities before load; stage encoder and produce CPU-frozen conditioning; release encoder device residency; materialize transformer; apply fixed LoRA without dequantizing the base; denoise; release transformer; decode with VAE; publish output. Cancellation or any fallback-integrity failure poisons and ejects the runtime.

Native proof requires positive Kitchen dispatch counts for every eligible INT8/FP8 layer and zero eager/dequantized fallback. A successful image alone is insufficient.

## Hardware and scientific acceptance

Fixed parity case:

- official four-file recipe;
- graph-default eight-step schedule;
- Darkbrush strength `0.8`;
- 1024 by 1024;
- seed `43301611940728`;
- prompt corpus covering editorial photography, illustration, materials, typography, faces/hands, long art direction, and dark/bright scenes.

Required scenarios:

- runtime-cold plus three changed-seed warm generations, not cache replays;
- four-file parity versus three-file no-LoRA deviation with identical prompt/seed/settings;
- Krea to Z-Image to Krea switching;
- cancellation during encoder load, LoRA application, transformer materialization, denoise, and decode;
- malformed transformer, encoder, VAE, and LoRA;
- 1024-square plus aligned portrait/landscape canvases;
- explicit teardown and memory return.

Provenance asserts all four SHA/revisions, Darkbrush strength, header/schema fingerprints, layout counts, native dispatch counts, cache state, stage residency, effective schedule, and output hash. External GPU/RAM sampling is approximate; also record allocator and Windows commit.

## Ordered bounded slices

1. **Next: license decision and exact four-file manifest.** No runtime code until product use, redistribution, revenue threshold, and content-filter obligations are approved. Reconfirm every HF identity through authenticated/public acquisition APIs.
2. **Official-parity INT8 T2I loader.** One operation and four fixed artifacts. Add closure/header/native-dispatch/lifecycle tests. Out of scope: Raw, style reference, edit, user LoRAs.
3. **Target RTX 5080 acceptance.** Run four-file parity and no-LoRA deviation as separate recipes; include cancellation and switching.
4. **Turbo BF16 Reference.** Qualify the exact same operation on adequate hardware; do not substitute a runtime-cast artifact.
5. **Style-reference operation only after T2I value is accepted.** Treat its graph, inputs, closure, and corpus separately.

Stop on any unresolved license gate, unknown header, incomplete LoRA mapping, hidden dense copy, native fallback, or ambiguity about whether Darkbrush ran.
