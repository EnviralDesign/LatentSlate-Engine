# FLUX.2 Klein 9B implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Upstream evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c), [Kitchen `9816d220021ab526e2cc1700a68b68d1b72d961c`](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c)

## Decision

Ordinary Distilled 9B is implemented and output-qualified on the RTX 5080: first-party NVFP4 is **Recommended on qualified Blackwell**, first-party FP8 is **Fallback**, and complete first-party BF16 remains the **Reference** even though it honestly OOMs on the 15.9 GiB card. Base and 9B-KV remain separate lines. The next work is cancellation/multi-reference lifecycle coverage, not format expansion.

## Product and operation boundary

| Line | Operations and exact semantics | Disposition |
| --- | --- | --- |
| Distilled 9B | T2I and ordinary ordered-reference edit; 4 steps, CFG 1 | Current product line |
| Base 9B | T2I/edit; official Comfy graphs use 20 steps, CFG 5 | Deferred separate line |
| 9B-KV | Repeated-reference editing with reference K/V created at step zero and reused on later denoise calls | Separate Experimental line; not a generic faster recipe |

Pinned official graphs:

- [9B T2I (Base selected)](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_text_to_image_9b.json)
- [ordinary Distilled edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_9b_distilled.json)
- [ordinary Base edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_image_edit_9b_base.json)
- [9B-KV edit](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_flux2_klein_9b_kv_image_edit.json)

The ordinary edit graph has one active reference and a bypassed second-reference example. KV has two active references and uses the full Flux2 VAE. Engine's ordinary third-reference support is an extension.

All first-party 9B weights are gated under the FLUX Non-Commercial License; filter/manual-review and product-distribution decisions are hard gates.

## Exact ordinary-Distilled closure

| Tier | Transformer identity | Shared closure | Minimum weight bytes |
| --- | --- | --- | ---: |
| Reference | authenticated BFL Distilled BF16; standalone file approximately 18.2 GB; production declaration must re-resolve an authenticated immutable revision; corroborated SHA-256 `0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58` | exact mixed Qwen3-8B, small decoder, matching support shell | larger than target GPU; exact closure is cataloged |
| Recommended | BFL NVFP4 revision `e882f64f6aa086fcf8915a7763550e05af10ef13`; 5,760,960,048 bytes; SHA-256 `5c72214496dd278f721a112e1bd1585fffed487bc0831c894bcbf30d12e9ee48` | same | 14,675,327,882 |
| Fallback | BFL FP8 revision `902d9d510b51533e07729f19211414a3648b77d2`; 9,433,061,528 bytes; SHA-256 `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee` | same | 18,347,429,362 |

Shared resources:

- `qwen_3_8b_fp8mixed.safetensors`: Comfy-Org revision `23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83`; 8,664,848,742 bytes; SHA-256 `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6`. Header contract: 141 global FP8 linear layers, 85 packed NVFP4 linear layers, 172 dense BF16 tensors; tied `lm_head.weight` omitted.
- `full_encoder_small_decoder.safetensors`: 249,519,092 bytes; SHA-256 `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.
- support shell: 15,886,279 bytes from Distilled revision `92196c8e11f7b6cf2b7493e037d8c5345c559216`.

KV uses a distinct first-party FP8 transformer (`a4d584032d9be4310b40531bc76b6b8398eba2c5`, 9,818,935,984 bytes, SHA-256 `33f7da5625a00798349a719742999d3c7dd20c1a7eda14663922c363640728f1`) and the full VAE. It is compatible by workflow, not byte- or numerically equivalent to ordinary Distilled.

## Candidate recipe contract

Existing keys:

- `flux2-klein-9b.text-to-image.bfl-distilled-nvfp4` — Recommended; stored NVFP4 transformer + exact mixed Qwen; staged residency; native Kitchen dispatch; prompt cache; no compile/runtime conversion.
- `flux2-klein-9b.image-to-image.bfl-distilled-nvfp4` — same resources, ordered 1–3 refs, bounded reference cache.
- `flux2-klein-9b.text-to-image.bfl-distilled-fp8` and `...image-to-image...` — Fallback, same schedule/components with stored FP8 transformer.
- complete BF16 keys remain Reference and should be capability-gated above the local workstation.

No schema extension is needed for ordinary 9B. A KV recipe requires a new typed cache contract because cached reference K/V must be keyed by transformer identity, ordered asset hashes, preprocessing, canvas, and model configuration and must be invalidated on cancellation or any key change.

## Loader/runtime packet at the audited Engine commit

Reuse the current Klein 9B planner/materializers, mixed-Qwen loader, `stored_quant.py`, `runtime/kit.py`, `runtime/cache.py`, runtime manager, and Klein tools. Preserve exact source-to-target tensor maps, dense QKV exceptions, tied-weight handling, sidecar scales, and schema fingerprints. Every successful generation must report positive native-dispatch deltas for all claimed FP8/NVFP4 modules and zero eager/dequantized fallback.

Lifecycle order: validate all headers → stage Qwen and materialize prompt conditioning → release device encoder residency → stage transformer blocks under the accepted partial-residency policy → release transformer before VAE decode → retain only bounded CPU caches and exact pipeline state. Warm switches are keyed by the complete pipeline fingerprint; LoRA identity participates in the key. Any cancellation, dispatch-integrity error, NaN, or CUDA exception poisons and ejects the runtime.

## Hardware/scientific acceptance packet

Use the existing public-API harness with seed `43301611940728`, 1024², four steps, CFG 1, and the established reference asset for I2I. Required next scenarios:

- three runtime-cold plus three warm runs for NVFP4 and FP8;
- NVFP4 → FP8 → NVFP4 switching;
- cancellation during Qwen load/encode, transformer materialization, denoise, and decode, followed by a clean job;
- one, two, then explicit three ordered references with cache hit/miss and invalidation proof;
- malformed mixed-Qwen and transformer headers, missing sidecars, aliases/tied-weight mismatch, and dense fallback rejection;
- provenance: exact resource SHA/revision, schema hashes, layout counts, native dispatch counts, residency mode, cache state, effective canvas, operation, schedule, and LoRA identity.

The BF16 reference should be run on larger hardware/Vast without weakening its closure to fit locally. Local OOM is a valid capability-gate result.

## Ordered bounded slices

1. **Next — ordinary Distilled lifecycle completion.** Operations: NVFP4/FP8 T2I and one-reference edit. Likely files: Klein runtime/cache tests and hardware-study scripts. Acceptance: cancellation/recovery, no fallback, manager empty after teardown. Out of scope: Base/KV/new formats.
2. **Two-reference official-parity edit.** Activate the bypassed topology, verify order/preprocessing/cache identity. Stop on stale-reference reuse or nondeterministic ordering.
3. **Three-reference Engine extension.** Same closure; explicit provenance flag that it is not official Comfy parity.
4. **Larger-hardware BF16 reference.** Vast-capable acceptance only; no artifact weakening.
5. **KV exploration only after ordinary lifecycle passes.** One operation: repeated-reference edit with exact KV FP8 closure. New typed cache key/teardown tests are mandatory. Out of scope: community KV NVFP4.

## Primary sources

- [BFL 9B Distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [BFL 9B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4)
- [BFL 9B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)
- [Comfy mixed Qwen repository](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b)
- [Comfy KV node](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy_extras/nodes_flux.py)
- [Engine audited source](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine)
