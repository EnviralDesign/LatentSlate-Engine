# LTX 2.3 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, lineage, license | Lightricks official LTX 2.3 sources and Engine’s immutable 50-file BF16 closure at upstream revision `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` |
| Saved operation topology/defaults | official Comfy T2V, I2V, and FLF workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| Node and quantized dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) and Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for the pending stored-FP8 path |
| Acceptance and tier | Engine public-API A/V artifacts, observed streams, cancellation/recovery, memory, and creator review; optimized Comfy repair is not on current main |

The Comfy pins are research authority, not evidence that local repair work landed or passed hardware acceptance.

## Product decision

LTX 2.3 is legacy relative to 2.5, but main owns a strong native BF16 structural Reference. Preserve it and implement only the decisive official optimized operations:

- T2V and first-frame I2V: **Dev FP8 plus official Distilled LoRA**;
- first+last-frame: **Distilled FP8**, a different graph/transformer line.

The native BF16 closure loaded coherently and produced one bounded RTX 5080 OOM at first execution. Do not schedule another local BF16 retry or weaken the Reference. Dense output comparison belongs to a batched high-memory Vast campaign.

There is no Recommended or accepted optimized LTX 2.3 recipe on current main.

## Operation boundaries

| Operation | Practical authority | Current main |
| --- | --- | --- |
| T2V | `video_ltx2_3_t2v.json`: Dev FP8 + Distilled LoRA | BF16 structural recipe/runtime only |
| first-frame I2V | `video_ltx2_3_i2v.json`: Dev FP8 + Distilled LoRA + ordered first image | BF16 operation only |
| first+last-frame | `video_ltx2_3_flf2v.json`: Distilled FP8 + ordered endpoints | BF16 operation only |
| V2V/audio-conditioned/IC-LoRA/upscale | separate graphs and closures | Deferred; prefer 2.5 for new features |

Do not apply the T2V/I2V lineage to FLF or use the FLF transformer for T2V/I2V.

## BF16 Reference truth

Current main declares an exact 50-file, **94,977,693,482-byte** closure containing transformer, text encoder/tokenizer, connectors, scheduler, video/audio VAEs, vocoder, and configs. It fixes 24 fps, 8 steps/CFG 1, `8n+1` frames, aligned dimensions, synchronized 48 kHz stereo audio, operation-specific schemas, and disposable-worker cancellation checks.

Proof level: **Cataloged / structurally validated Reference**, one bounded local OOM, no accepted output.

## Optimized implementation packet

For each operation:

1. fetch and hash the exact raw workflow;
2. normalize every subgraph/switch and dynamic placeholder;
3. verify ComfyUI node inputs/output slots, endpoint indexes, and A/V output behavior;
4. resolve the complete active closure, including fixed LoRA and audio components;
5. validate stored-FP8 headers/maps/sidecars/dense exceptions and Kitchen dispatch;
6. write independent fixtures before implementation;
7. implement operation-specific requests and disposable-worker/native lifecycle;
8. fail closed on conversion, graph drift, missing A/V components, or fallback.

The transformer alone is not the closure. Unlanded local repair work cannot be cited as catalog, runtime, or hardware proof.

## Acceptance and next slices

Use exact graph frame/dimension/fps rules and probe actual video/audio streams. Corpus covers dialogue, music, ambience, foley, silence, lip/action timing, camera motion, identity, first-frame fidelity, endpoint approach, and drift. Require cold plus meaningful warm jobs, operation switching, malformed resources, cancellation across encode/load/denoise/decode/mux, observed worker exit, fresh recovery, memory, and creator review.

Next: authenticate/normalize T2V; implement/accept T2V; operation-specific first-frame I2V; independently normalize/implement FLF; batch dense BF16 references on Vast.

Stop on unlanded-status claims, partial closure, lineage substitution, hidden conversion/fallback, assumed A/V metadata, or cancellation without observed cleanup.
