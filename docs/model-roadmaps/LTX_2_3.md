# LTX 2.3 roadmap

Last authority audit: **2026-08-13**

Engine source audited:
[`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Lightricks sources and Engine’s immutable BF16 closure at upstream `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` |
| saved topology/defaults | official T2V, I2V, and FLF workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for future direct Engine dispatch |
| acceptance/tier | Engine public-API A/V artifacts/lifecycle; optimized path has no conforming implementation or acceptance |

## Decision

Current main owns a strong native BF16 structural Reference, not a practical local
product path. One bounded RTX 5080 OOM is enough; dense output belongs on Vast.

Practical source contracts:

- T2V and first-frame I2V: Dev FP8 plus fixed Distilled LoRA;
- FLF: Distilled FP8 with ordered endpoints, a separate graph/transformer line.

These are awaiting Engine-native Kitchen-backed implementation. Local repair work is
unlanded and must not be described as accepted.

## BF16 truth

The exact 50-file closure totals **94,977,693,482 bytes** and includes transformer,
encoder/tokenizer, connectors, scheduler, video/audio VAEs, vocoder, and configs. It
fixes 24 fps, 8 steps/CFG 1, `8n+1` frames, aligned dimensions, synchronized 48 kHz
stereo, typed operations, and Engine disposable-worker cancellation boundaries.

Proof: **Cataloged / structurally validated Reference, one bounded local OOM**.

## Required optimized implementation

For each operation, hash/normalize its workflow, verify node behavior from source,
resolve all active A/V resources and fixed LoRA, validate stored-FP8 headers and direct
Kitchen dispatch, write independent fixtures, and implement Engine-owned typed
orchestration, workers, cancellation, mux, output, and provenance.

No workflow is submitted to ComfyUI. T2V/I2V and FLF must not share the wrong lineage.

## Next slices

1. exact T2V/I2V closure and fixtures;
2. Engine-native Dev-FP8 + Distilled-LoRA runtime;
3. target A/V acceptance;
4. independent FLF closure/runtime;
5. batched dense BF16 Vast comparison;
6. no new 2.3 feature without specific compatibility value.

Stop on ComfyUI dependency, unlanded-status claims, partial closure, lineage
substitution, hidden conversion/fallback, assumed A/V metadata, or unobserved cleanup.
