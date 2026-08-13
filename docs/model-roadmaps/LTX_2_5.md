# LTX 2.5 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Surface | Authority |
| --- | --- |
| Weights/architecture/license | Lightricks [`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) and exact gated snapshots |
| Saved topology/defaults | official [`video_ltx2_5_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_t2v.json), blob `60683c9f3cd9c708581e1fb2e2030d987d540634` |
| Node/dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), exact headers |
| Acceptance/tier | Engine public-API A/V artifacts, observed dispatch/lifecycle/memory, and creator review; none exists on current main |

Local optimized repair work is unlanded and must not be described as runnable or accepted.

## Decision and boundaries

The first local candidate is the exact official Comfy **T2V** graph. The official FLF graph and publisher BF16 Distilled two-stage pipeline are separate operations. Dense BF16 is Reference/Experimental on adequate hardware, not the first RTX 5080 task. No Recommended path exists.

| Path | Boundary | Status |
| --- | --- | --- |
| Comfy T2V | stored low-bit transformer/encoder, prompt enhancer, x2 upscaler, video/audio VAEs | first candidate; Not implemented |
| publisher BF16 two-stage T2V | half-resolution stage, x2 latent upscale, short full-resolution refinement | separate cloud/reference path |
| first-frame I2V | operation-specific image conditioning | follow-on |
| Comfy FLF | two endpoints and different graph/closure | separate operation |

T2V, I2V, and FLF are not one tool with optional media.

## Official Comfy T2V selection

The pinned graph actively selects six files:

- `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`;
- `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`;
- `gemma4_e2b_it_bf16.safetensors` prompt enhancer;
- `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`;
- `ltx-2.5-video-vae-bf16.safetensors`;
- `ltx-2.5-audio-vae-bf16.safetensors`.

Saved defaults include prompt enhancement, 1280 by 720, five seconds, and 24 fps. They are topology authority, not feasibility proof. The FLF blob `6d3d732038151d46946538f11cc3213520980cde` is not a T2V substitute.

Authenticated metadata must still resolve immutable revisions, bytes, hashes, licenses/gates, and any support files. Until then the graph is exact but not acquisition-ready.

The BF16 two-stage Reference separately requires Distilled transformer, projected Gemma encoder, selected video VAE, audio VAE, x2 upscaler, and official support. Omitting the upscaler breaks parity. The audited source constrains Transformers below 5.15. Runtime casting is not a production recipe.

## Implementation and acceptance

First recipe: `ltx-2-5.text-to-video.comfy-int8` as Experimental. No LTX 2.5 recipe is cataloged, runnable, Hardware-proven, Recommended, or Fallback on current main.

Before coding: hash and normalize the raw graph; verify ComfyUI inputs/output slots, frame/fps mapping, enhancer, A/V paths, and save behavior; resolve all six resources plus support; validate both low-bit maps/sidecars/dense exceptions and Kitchen dispatch; write independent fixtures; fail closed on missing roles, graph drift, conversion, or fallback.

An isolated worker must verify its checkout, stage only validated files, disable custom nodes, record the submitted graph hash, observe output metadata, and unload on cancellation/failure.

Use a labeled smaller diagnostic while retaining exact parity settings. Require cold plus meaningful warm jobs, family switching, malformed resources, phase cancellation/recovery, observed video/audio streams, memory, teardown, and creator review across dialogue, music, ambience, foley, lip/action sync, camera motion, identity, coherence, and drift.

Next: authenticate/normalize T2V; implement with dispatch proof; target diagnostic/parity acceptance; BF16 high-memory Reference; first-frame I2V; separate FLF.

Stop on gated identity gaps, T2V/FLF substitution, incomplete mapping, missing enhancer/upscaler/audio, fallback, assumed metadata, or unobserved cancellation.
