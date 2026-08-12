# FLUX.2 Klein 4B implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Current upstream evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c), [Comfy Kitchen `9816d220021ab526e2cc1700a68b68d1b72d961c`](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)

## Decision

Klein 4B remains the house reference for how an Engine family should be packaged. The ordinary Distilled line is already implementation- and hardware-proven: **BF16 Reference**, **first-party NVFP4 Recommended on qualified Blackwell**, and **first-party FP8 Fallback**. Base editing is an Alternate line, not a precision variant of Distilled. The next work is lifecycle coverage, not another loader.

Evidence labels used below are **workflow**, **publisher**, **code**, **measurement**, and **inference**. A workflow proves only the graph it contains; a model card proves publisher intent; Engine measurements are explicitly identified.

## Product and operation boundary

| Operation | Official/default boundary | Engine boundary | Disposition |
| --- | --- | --- | --- |
| Distilled T2I | 4 steps, guidance/CFG 1, Euler + Flux2 schedule, 1024² baseline | Native T2I recipes exist | Recommended NVFP4; FP8 Fallback; BF16 Reference |
| Distilled edit | One active reference; official templates demonstrate a second reference in a bypassed example | Engine deliberately accepts 1–3 ordered references | One/two refs require parity acceptance; third ref remains an Engine extension |
| Base edit | 20 steps, guidance/CFG 5; separate Base weights and small-decoder closure | FP8 Alternate exists | Retain; add matching Base BF16 before judging Base quantization |
| Inpaint/control | No bounded package-owned native contract in this tranche | Generic Comfy can express these graphs | Keep generic Comfy unless a creator workflow justifies a distinct recipe |

Official parity graphs:

- [Distilled edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_4b_distilled.json)
- [Base edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_4b_base.json)
- [T2I shell](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_text_to_image.json)

References are resized toward one megapixel with nearest-exact semantics before Flux reference conditioning. Conditioning order is reference order. The disabled two-reference subgraph chains reference conditioning once per image; it is not evidence for three references.

## Exact resource closures

| Tier | Transformer | Exact immutable identity | Shared closure | Declared closure bytes |
| --- | --- | --- | --- | ---: |
| Reference | BFL Distilled BF16 `flux-2-klein-4b.safetensors` | revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`; 7,751,105,712 bytes; SHA-256 `ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343` | Qwen3 4B BF16, Flux2 VAE, Distilled support shell | 16,148,187,592 |
| Recommended | BFL Distilled NVFP4 `flux-2-klein-4b-nvfp4.safetensors` | revision `286fd2fbb83294d929d5be472620826c28e6085b`; 2,460,413,488 bytes; SHA-256 `d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b` | Same exact shared closure | 10,857,495,368 |
| Fallback | BFL Distilled FP8 `flux-2-klein-4b-fp8.safetensors` | revision `5b4408e59397a4a37ccb46afe426d8ed86379441`; 4,070,624,520 bytes; SHA-256 `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6` | Same exact shared closure | 12,467,706,400 |
| Alternate | BFL Base FP8 `flux-2-klein-base-4b-fp8.safetensors` | revision `103db268c10d4d3921101b46057671f9ac460da6`; 4,089,498,488 bytes; SHA-256 `44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840` | Qwen3 4B BF16, small-decoder VAE, Base shell | 12,399,885,870 |

Shared exact resources:

- Qwen3 4B BF16 `qwen_3_4b.safetensors`: revision `d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e`, 8,044,982,048 bytes, SHA-256 `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`.
- Flux2 VAE `flux2-vae.safetensors`: revision `03d6521e6f6a47396b3f951cbea50f7e6c2f482e`, 336,213,556 bytes, SHA-256 `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`.
- Small decoder `full_encoder_small_decoder.safetensors`: revision `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`, 249,519,092 bytes, SHA-256 `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.

All BFL 4B weights are first-party Apache-2.0 artifacts. Component equality means byte-identical only where the same resource identity above is reused. Base and Distilled transformers/support shells are not substitutes.

## Candidate recipe contract

These keys already match the public grammar and current catalog:

| Key | Tier | Fixed runtime contract |
| --- | --- | --- |
| `flux2-klein-4b.text-to-image.bfl-distilled-nvfp4` | Recommended | stored NVFP4; native attention; Engine staged residency; no compile; prompt cache; VAE staged; keep pipeline loaded |
| `flux2-klein-4b.image-to-image.bfl-distilled-nvfp4` | Recommended | same resources; ordered 1–3 references; reference cache; one/two refs parity, three refs extension |
| `flux2-klein-4b.text-to-image.comfy-distilled-fp8` | Fallback | stored global FP8; otherwise same Distilled schedule and closure |
| `flux2-klein-4b.image-to-image.comfy-distilled-fp8` | Fallback | same FP8 closure and reference semantics |
| `flux2-klein-4b.image-to-image.comfy-base-fp8` | Alternate | Base transformer/support/small decoder; 20 steps; CFG 5 |

No schema extension is needed for those recipes. Dynamic LoRAs are already a bounded additive branch and must never dequantize the base. A componentized Base BF16 recipe is a catalog addition, not a new runtime family.

## Loader/runtime packet at the audited Engine commit

Reuse `klein_recipe.py`, `variants.py`, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, the Klein runtime/materializer modules, and the existing Klein tools. Preserve:

1. immutable component signatures before and after loading;
2. exact SafeTensors header/schema fingerprinting;
3. packed Kitchen tensors through module assignment;
4. measured `scaled_mm_nvfp4` or native FP8 dispatch, with zero dense/eager fallback;
5. prompt encoder → transformer → VAE staging, with only bounded caches resident;
6. pipeline fingerprint keying by exact resources, operation, optimization policy, and LoRAs;
7. runtime poisoning/ejection on cancellation, CUDA/materialization failure, or fallback-integrity failure.

The current stored-weight contract is already the proof pattern for later families: do not generalize it into a user-selected `quantization=` conversion switch.

## Hardware/scientific acceptance packet

Target: Windows 11, RTX 5080 15.9 GiB, 63.8 GiB RAM, CUDA 13/Kitchen-capable Blackwell.

Fixed parity case: prompt from the existing Klein hardware-study corpus, seed `43301611940728`, 1024×1024, 4 steps, CFG 1, Euler/Flux2 schedule. I2I uses the pinned source hash `9299067fd7912d4e6ac7c4cd0888082fb71cf3b1562fd6e8c9ee7fd3735c7fa5`.

Required scenarios:

- three runtime-cold and three warm repeats for NVFP4 and FP8;
- Recommended → Fallback → Recommended switching;
- one and two ordered references, then the explicit three-reference extension;
- cancellation during materialization, denoise, and decode, followed by a clean recovery job;
- malformed header/scale/sidecar rejection before model allocation;
- provenance assertions for every resource SHA, schema fingerprint, native dispatch count, cache hit, effective canvas, and runtime-cold/warm state.

Device-wide one-second VRAM sampling is approximate; keep allocator peaks and Windows process/system commit separately. Existing measurements prove NVFP4 materially faster than FP8 and BF16 on this workstation, but do not prove cross-format perceptual equivalence.

## Ordered bounded slices

1. **Next — Distilled edit lifecycle completion.** Operation: one/two-reference I2I for NVFP4 and FP8. Likely files: hardware-study scripts/tests only plus defects surfaced in Klein runtime/cache. Tests: cancellation/recovery, ordered-reference cache invalidation, A→B→A, provenance. Out of scope: new formats, Base, third-reference promotion. Stop on fallback, stale cache, poison, or output corruption.
2. **Engine extension acceptance.** Operation: three-reference Distilled I2I. Reuse exact resources/runtime. Prove deterministic ordering and provenance; label non-Comfy extension. Out of scope: changing preprocessing or adding arbitrary lists.
3. **Base BF16 comparison closure.** Operation: Base one-reference I2I. Add exact componentized BF16 recipe/resource closure and compare against Base FP8 at 20 steps/CFG 5. Out of scope: Base NVFP4 until BF16 succeeds.
4. **Stop.** Do not add ConvRot/GGUF/W4/Nunchaku without a new creator-visible requirement.

## Primary sources

- [BFL 4B family](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [ComfyUI quantized loading](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/quant_ops.py)
- [ComfyUI model management](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/model_management.py)
- [Comfy Kitchen pinned research source](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)
- [Engine audited source](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine)
