# Ideogram 4 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, prompt schema, license | Ideogram publisher source [`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2) and exact artifact identities |
| Saved operation topology/defaults | official [`image_ideogram4_t2i_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i_int8.json), Git blob `4e9a71db38bc0c6e09aafba658adb5b06d10c8fa` |
| Node and quantized dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), ConvRot source [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) |
| Acceptance and tier | Engine JSON/layout artifacts, dual-branch dispatch/lifecycle, safety result, observed media, and creator review; none exists on current main |

The publisher commit and referenced architecture/prompt/inference paths were independently resolved; the old ledger warning was stale.

## Product decision

Ideogram 4 is structured-JSON typography/design T2I with separate conditional and unconditional branches. No dense BF16/FP16 public teacher exists; official NF4 Diffusers is the public Reference baseline, not lossless dense truth. The narrowest practical local candidate is the exact four-file Comfy INT8 graph.

License review is a hard gate because the combined closure carries Ideogram, Qwen, and Flux2 VAE terms.

## Operation and closure

Structured JSON, optional boxes/palette, explicit prompt-assistant mode, preset, dimensions, seed, dual branches, Qwen3-VL, and VAE are part of the operation. Hosted magic prompt is a separate provider path; I2I/edit/control remain generic Comfy/provider.

| Role | Bytes | SHA-256 |
| --- | ---: | --- |
| conditional INT8 ConvRot branch | 9,583,465,712 | `a9164002943463b4c7b2abd88c82a488c088acc35762651e4d8604d6ce4a163d` |
| unconditional INT8 ConvRot branch | 9,583,465,712 | `cd03ed94f244c9cb705e7d30ca0f40b5f5b004bb20674117adff88d16416c23d` |
| Qwen3-VL 8B scaled-FP8 encoder | 10,588,637,512 | `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` |
| Flux2 VAE | 336,213,556 | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |
| **total** | **30,091,782,492** | — |

One branch is never a complete path. FP8/NVFP4 follow-ons must preserve both branches and matching encoder/VAE. The standard FP8 template may add a Gemma prompt assistant; active closure depends on prompt mode and must be labeled.

## Recipe and implementation order

1. `ideogram-4.text-to-image.comfy-int8-qwen-json` — Experimental first local path: exact four-file graph and explicit JSON/Qwen mode.
2. Official NF4 Diffusers — Reference baseline with operation-matched JSON.
3. FP8/NVFP4 — deferred until prompt-assistant mode and complete closure are fixed.
4. Hosted API — separate Fallback.

No local Ideogram recipe is runnable or accepted on current main.

A typed recipe owns `conditional_transformer`, `unconditional_transformer`, `text_vision_encoder`, and `vae`; prompt-assistant resources are a separate mode. The request retains original text, normalized JSON, boxes, palette, preset, dimensions, seed, expansion identity, and safety result.

Before coding: normalize the graph; verify JSON/prompt/output schemas; validate both branch maps/identity, ConvRot markers/scales/dense exceptions, Qwen hidden-state/scaled-FP8 contract, VAE, and deterministic JSON serialization; reject missing/mixed branches, hidden hosted expansion, conversion, or fallback.

Lifecycle stages Qwen then exact branch choreography, releases components, decodes VAE, and observes safety/output. Cancellation clears both branch states and partial prompt data.

## Acceptance and next slices

Use multilingual posters, menus, signs, logos, exact line breaks, boxes, palettes, object counts, long banners, products, photography, and illustration. Require cold plus meaningful warm jobs, branch-order evidence, switching, malformed/missing resources, invalid JSON fields, phase cancellation/recovery, observed metadata, teardown, and creator review.

Next: composite license decision; structured JSON schema; normalized four-file fixture; dual-branch loader; RTX 5080 acceptance; NF4 high-memory reference; only then FP8/NVFP4.

Stop on license ambiguity, missing branch, graph drift, hidden expansion, incomplete headers, fallback, false availability, or assumed output metadata.
