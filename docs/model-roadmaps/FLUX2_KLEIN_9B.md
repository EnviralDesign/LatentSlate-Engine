# FLUX.2 Klein 9B roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, lineage, license | first-party BFL repositories and exact Engine resource declarations; gated immutable identities must be re-resolved for any new artifact |
| Saved operation topology/defaults | accepted workflow baseline [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb) |
| Node and quantized dispatch schema | accepted ComfyUI [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) and Kitchen `0.2.28` at [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) |
| Acceptance and tier | Engine public-API outputs, native dispatch, deterministic repeats, switching, bounded BF16 OOM, and creator review |

Current research pins may advance independently. They do not rewrite the stack that produced accepted results.

## Product decision

Keep three separate lines:

1. **Ordinary Distilled 9B:** four-step T2I and ordered-reference edit. First-party NVFP4 is Recommended on qualified Blackwell, FP8 is Fallback, and matching BF16 is Reference.
2. **Base 9B:** separate 20-step/CFG-5 foundation line, deferred after ordinary Distilled.
3. **9B-KV:** separate repeated-reference experiment with reference K/V state, not a generic faster default.

Ordinary NVFP4/FP8 T2I and one-reference edit are hardware-proven on the RTX 5080. Complete BF16 produced one bounded local OOM; preserve it and move output comparison to high-memory hardware.

## Operation authority

| Pinned graph | Saved contract | Boundary |
| --- | --- | --- |
| `image_flux2_text_to_image_9b.json` | Base FP8, 20 steps, CFG 5 | no checked-in Distilled T2I graph at the accepted pin; Engine’s four-step Distilled T2I is derived and labeled |
| `image_flux2_klein_image_edit_9b_distilled.json` | Distilled FP8, four steps, CFG 1 | one active reference; disabled two-reference example |
| `image_flux2_klein_image_edit_9b_base.json` | Base FP8, 20 steps, CFG 5 | separate Base line |
| `image_flux2_klein_9b_kv_image_edit.json` | KV FP8, four steps, CFG 1, full Flux2 VAE | two active references and experimental `FluxKVCache` semantics |

No checked-in official 9B NVFP4 graph was identified. The accepted NVFP4 Engine graph is an explicit stored-artifact substitution with its own fingerprint and acceptance.

## Accepted ordinary closure

| Path | Minimum declared bytes | Status |
| --- | ---: | --- |
| Distilled BF16 | above local GPU envelope | Reference; bounded local OOM |
| Distilled NVFP4 | 14,675,327,882 | Recommended on Blackwell |
| Distilled FP8 | 18,347,429,362 | Fallback |
| Base FP8 | 18,481,646,306 | Deferred Base line |
| KV FP8 | 18,819,998,282 | separate experiment |

Exact artifact revisions, hashes, mixed-Qwen layout, schema fingerprints, and LoRA targets remain in Engine declarations and accepted study manifests.

## Engine proof preserved

Seed `43301611940728`, 1024-square, runtime-cold and warm studies established positive native NVFP4/FP8 dispatch, deterministic repeated outputs within each recipe, and NVFP4-to-FP8-to-NVFP4 switching. A header-proven compatible LoRA executed over first-party NVFP4 without accepted base dequantization.

Different quantized output hashes are not perceptual equivalence. Cancellation and multi-reference lifecycle are still incomplete.

Proof level: **Hardware-proven** for ordinary Distilled NVFP4/FP8 T2I and one-reference edit.

## KV-specific contract

Any KV cache is keyed by transformer identity, ordered reference hashes, preprocessing, dimensions, operation, and model config. Changed input or failed/canceled work invalidates it. Report first-generation and reuse timing separately; do not retain a Comfy model-object cache across jobs without explicit Engine ownership and teardown.

## Runtime contract to preserve

- exact gated identity and schema revalidation;
- complete fused/dense/sidecar/alias maps for transformer and mixed Qwen;
- stored partial residency with no dense duplicate;
- ordered media cache identity/invalidation;
- additive LoRA dispatch without base dequantization;
- positive Kitchen dispatch and zero hidden fallback;
- poison/ejection on cancellation or integrity failure.

## Next bounded slices

1. Cancellation and clean recovery for ordinary NVFP4/FP8.
2. Official two-reference edit from the disabled example.
3. Labeled three-reference Engine extension.
4. Held-input creator review.
5. BF16 Reference on high-memory hardware.
6. KV only after ordinary lifecycle passes.

Stop on graph drift, false availability, stale media/KV cache, hidden fallback, or treating Base/KV as ordinary Distilled.
