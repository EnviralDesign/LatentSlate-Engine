# LTX 2.5 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Lightricks [`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) and exact gated snapshots |
| saved topology/defaults | [`video_ltx2_5_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_ltx2_5_t2v.json), blob `60683c9f3cd9c708581e1fb2e2030d987d540634` |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for future direct Engine dispatch |
| acceptance/tier | Engine public-API synchronized A/V evidence; none exists |

## Decision

The first local candidate is an Engine-native T2V runtime derived from the exact
six-role saved workflow. FLF and publisher BF16 two-stage T2V are separate contracts.
No LTX 2.5 path is implemented or accepted; unlanded repair work is not evidence.

## Six-role T2V selection

- `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`;
- `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`;
- `gemma4_e2b_it_bf16.safetensors` prompt enhancer;
- `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`;
- `ltx-2.5-video-vae-bf16.safetensors`;
- `ltx-2.5-audio-vae-bf16.safetensors`.

Saved defaults include prompt enhancement, 1280×720, five seconds, and 24 fps. Exact
gated revisions, bytes, hashes, licenses, and support remain blockers. FLF blob
`6d3d732038151d46946538f11cc3213520980cde` is not a T2V substitute.

Publisher BF16 two-stage Reference separately requires the Distilled transformer,
projected encoder, selected video/audio VAEs, x2 upscaler, support, half-resolution
stage, upscale, and short full-resolution refinement. Runtime casting is rejected.

## Required implementation

Normalize the workflow into independent fixtures; verify node behavior from source;
resolve six roles/support; validate both low-bit maps and direct Kitchen primitives;
implement Engine-owned prompt enhancement, stages, A/V decode, mux, cancellation,
cleanup, output, and provenance in Engine-owned disposable workers.

Engine must not install or launch ComfyUI, create upstream model-folder layouts, host custom nodes, or
submit a graph.

## Next slices

1. gated immutable six-role closure;
2. Engine-native Kitchen-backed T2V;
3. diagnostic and parity acceptance with observed A/V;
4. high-memory BF16 Reference;
5. first-frame I2V;
6. separate FLF.

Stop on ComfyUI dependency, gated identity gaps, T2V/FLF substitution, missing
enhancer/upscaler/audio, incomplete map, fallback, assumed metadata, or unobserved
cancellation.
