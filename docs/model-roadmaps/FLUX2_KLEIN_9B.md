# FLUX.2 Klein 9B roadmap

Last authority audit: **2026-08-15**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | first-party BFL sources and exact Engine resources |
| saved topology/defaults | workflows [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb) |
| node behavior / dispatch | ComfyUI source [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) as research; direct Kitchen `0.2.28` at [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) |
| acceptance/tier | Engine public-API outputs, dispatch counters, bounded BF16 OOM, switching, lifecycle, creator review |

No ComfyUI code or process participates in execution.

## Decision

Keep separate:

1. ordinary Distilled 9B: four-step T2I/edit; NVFP4 Recommended on Blackwell, FP8
   Fallback, matching BF16 Reference;
2. Base 9B: separate 20-step/CFG-5 foundation line;
3. 9B-KV: separate repeated-reference cache experiment.

All four optimized NVFP4/FP8 recipes are narrow Hardware-proven through
LatentSlate-originated T2I, ordered one-to-three-reference edit, cancellation, and
fresh-job recovery. BF16 has one bounded local OOM and remains an unaccepted
high-memory Reference.

## Behavioral findings

- the pinned 9B T2I workflow is Base, not Distilled;
- ordinary Distilled edit is four steps/CFG 1 with one active and one disabled second
  reference;
- Base edit is 20 steps/CFG 5;
- KV is four steps/CFG 1 with two active references and distinct state semantics;
- no official NVFP4 graph was identified, so Engine’s accepted NVFP4 path is a
  separately fingerprinted stored-artifact substitution.

## Declared closures

| Path | Bytes | Status |
| --- | ---: | --- |
| Distilled BF16 | above local envelope | Reference |
| Distilled NVFP4 | 14,675,327,882 | Recommended |
| Distilled FP8 | 18,347,429,362 | Fallback |
| Base FP8 | 18,481,646,306 | Deferred |
| KV FP8 | 18,819,998,282 | separate experiment |

## Proof to preserve

Fixed 1024-square studies established positive direct Kitchen NVFP4/FP8 dispatch,
deterministic repeated output within each recipe, and NVFP4-to-FP8-to-NVFP4 switching.
Exact transformer/text Kitchen closures recorded zero fallback. A header-proven LoRA
executed additively without accepted base dequantization. Final trace prefixes are:

| Operation/tier | App / Engine / artifact SHA-256 trace prefixes |
| --- | --- |
| T2I NVFP4 Recommended | `5fe29980` / `c02d0403` / `7BE125` |
| T2I FP8 Fallback | `e42834d7` / `3ec0bcf1` / `4E5DC3` |
| I2I NVFP4 Recommended | cancel `2f9ea51f` / `9d747ad2`; recovery `13b54a6f` / `fc2ad938` / `8DDE6F` |
| I2I FP8 Fallback | two refs `8f730e44` / `aef0e1dc` / `615E41`; three refs `e22854ff` / `766cd5a8` / `B8256F` |

Cancellation emitted no artifact and recovery completed in a new job. This is
operational acceptance, not pixel/latent parity with Comfy.

## KV boundary

Any Engine KV cache is explicitly owned and keyed by transformer, ordered media hashes,
preprocessing, dimensions, operation, and model config. Changed, failed, or canceled
work invalidates it. Upstream model-object behavior is research evidence only; Engine
does not retain or execute a ComfyUI model object.

## Next slices

1. broaden held-input creator and reference-diversity coverage;
2. retain cancellation/recovery, switching, exact module-count, and zero-fallback
   evidence as dependencies evolve;
3. high-memory BF16 Reference;
4. keep KV a separate research line from the accepted ordinary lifecycle.

Stop on ComfyUI dependency, Base/KV conflation, stale cache, fallback, or false
availability.
