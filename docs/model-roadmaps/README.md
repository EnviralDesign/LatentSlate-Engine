# Model roadmaps and implementation authority

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

This directory is the active decision surface for model-family implementation. It is not permission to reconstruct a pipeline from memory.

## Authority hierarchy

Use the first applicable authority. Lower layers implement or validate higher layers; they may not silently replace them.

1. **First-party publisher repositories and immutable snapshots** own weight identity, architecture, configs, license, lineage, and dense Reference facts.
2. **Pinned official Comfy-Org workflow templates** own practical creator-facing topology and saved defaults: active roles, wiring, preprocessing, prompt enhancement, conditioning order, sampler/scheduler/sigmas, CFG, steps/stages, dimensions, frame/fps rules, LoRA strength, and switch state.
3. **The exact pinned ComfyUI checkout** owns node classes, required inputs, output slots, validation, preprocessing, loading, memory management, and graph execution behavior.
4. **Pinned Comfy Kitchen source/version plus exact file headers** own low-bit markers, sidecars, layouts, supported native dispatch, and fallback behavior. A kernel existing is not file compatibility proof.
5. **Engine public-API evidence** owns acceptance and tiering: effective request, artifacts, provenance, dispatch counters, cancellation/recovery, memory, output inspection, and creator review.
6. **Dense BF16 video Reference** is an authority/comparison contract, not a local-fit mandate. Pin and CPU-validate it, keep one bounded OOM when useful, then batch dense outputs on high-memory Vast. Local RTX 5080 time prioritizes exact practical Comfy/FP8/ConvRot/NVFP4/fixed-LoRA paths.

Tutorials, screenshots, filenames, publisher benchmarks, and successful outputs cannot override the source that owns the disputed fact.

## Pin classes

There is no universal “current Comfy pin.”

- **Accepted baseline:** exact workflow, ComfyUI, Kitchen, and package stack used by retained Engine evidence. Preserve until deliberately requalified.
- **Current research:** immutable snapshot used for the next implementation packet. It is not accepted runtime evidence.
- **Authoring baseline:** template package/source used for catalog and recipe ingestion.
- **Historical alternate:** older immutable source retained to explain prior results.
- **Mutable discovery:** model card, gated landing page, tutorial, or service docs. It is never an execution or acquisition lock.

Important examples:

- recipe authoring uses workflow templates [`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1), package `0.1.37`;
- current portfolio research uses workflows [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), and Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4);
- Klein acceptance preserves workflows [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb), ComfyUI [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d), and Kitchen `0.2.28` at [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028);
- Wan 5 acceptance preserves examples [`f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3) and executable ComfyUI [`eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f).

See [SOURCE_PIN_AUDIT.md](./SOURCE_PIN_AUDIT.md) for complete classification and mutable-link rationale.

## Mandatory roadmap authority map

Every family roadmap names four owners:

| Surface | Required owner |
| --- | --- |
| weights/architecture/lineage/config/license | immutable publisher source and exact artifact revisions where available |
| saved operation topology/defaults | exact official Comfy raw blob, or explicitly labeled publisher-native graph when no Comfy graph exists |
| node and quantized dispatch schema | exact ComfyUI checkout and Kitchen source/version, or “not applicable” for an unquantized non-Comfy path |
| acceptance/tier | exact Engine commit and public-API/hardware evidence, otherwise the truthful lower proof state |

Accepted and research pins are listed separately when they differ.

## Implementation-agent preflight

Before implementation code:

1. Fetch the pinned raw workflow; retain repository commit, Git blob SHA, bytes, and raw SHA-256.
2. Expand every subgraph/switch into a normalized API graph with nodes, edges, constants, output slots, active/disabled branches, and dynamic placeholders.
3. Fetch the pinned ComfyUI object schema/source; verify class names, required inputs, enums, output indexes, preprocessing, and output-object shape.
4. Enumerate the complete active closure, including prompt enhancers, VAEs, upscalers, fixed LoRAs, and support. Record configured-but-disabled resources separately.
5. Resolve immutable artifact identities, bytes, hashes, licenses/gates, and SafeTensors header/schema fingerprints. Template `resolve/main` URLs are discovery only.
6. Verify Kitchen markers, scales, sidecars, aliases, dense exceptions, group geometry, backend, and fallback against the exact headers.
7. Write independent fixtures from upstream evidence before implementation. Fixtures generated by the implementation under test are not independent.
8. Only then author resources, recipes, runtime code, and public-API acceptance scenarios.

## Review gates

Reject a packet or implementation when any of these are unproven:

- **graph drift:** compiled graph differs without a documented Engine deviation and separate fingerprint;
- **hidden native fallthrough:** claimed low-bit/native execution converts, dequantizes, or uses eager/dense fallback;
- **false availability:** files are installed/cataloged but a loader, dependency, backend, license, schema, or sibling component is missing;
- **assumed output metadata:** dimensions, frames, fps, duration, audio, codec, or output slot are copied from the request instead of observed;
- **unobserved cancellation:** API state changed but worker exit, accelerator release, temp cleanup, poisoned-state eviction, and recovery were not observed;
- **cataloged-versus-runnable confusion:** declarations, header parsing, or unit tests are described as generation proof;
- **self-confirming fixtures:** expected behavior was copied from the implementation rather than normalized upstream evidence.

## Status and proof vocabulary

Product tiers are **Reference**, **Recommended**, **Fallback**, **Alternate**, **Experimental**, **Deferred**, and **Rejected**.

Proof levels:

- **Hardware-proven:** target-class public-API output plus intended backend and lifecycle evidence.
- **Runtime-proven:** one end-to-end job, incomplete acceptance matrix.
- **Structurally tested:** independent graph/header/loader fixtures, no accepted output.
- **Cataloged:** declarations exist, execution unproven.
- **Direct tool only:** callable code outside the opinionated package surface.
- **Not implemented:** no Engine path.

## Dense video and local acceptance

Dense BF16 video is source-pinned and CPU-validated locally, with one bounded OOM when useful; full output comparison is batched on Vast. Locally practical candidates use the public API and retain exact authority packet, resources, effective request, native dispatch/fallback, cold plus meaningful warm execution, switching, malformed-artifact behavior, observed output metadata, cancellation/recovery, memory, hashes, and creator review.

Stop on authority mismatch, unknown layout, fallback, corrupt output, poisoned recovery, stale media conditioning, or silent reduction of Reference settings.

## Active roadmaps and indexes

- [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md)
- [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md)
- [Krea 2](./KREA_2.md)
- [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md)
- [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md)
- [Ideogram 4](./IDEOGRAM_4.md)
- [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md)
- [Wan 2.2 14B](./WAN22_14B.md)
- [LTX 2.3](./LTX_2_3.md)
- [LTX 2.5](./LTX_2_5.md)
- [MiniMax H3](./MINIMAX_H3.md)
- [Z-Image Turbo](./Z_IMAGE_TURBO.md)
- [Implementation packets](./IMPLEMENTATION_PACKETS.md)
- [Source-pin audit](./SOURCE_PIN_AUDIT.md)
- [Recipes](../RECIPES.md)
- [Hardware studies](../HARDWARE_STUDIES.md)
