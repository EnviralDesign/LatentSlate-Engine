# FLUX.2 Klein 4B roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md) and the shared
[roadmap preflight](./README.md#implementation-agent-preflight).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | first-party BFL repositories plus exact Engine resource identities |
| saved topology/defaults | workflow baseline [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb) |
| node behavior / quantized dispatch | ComfyUI source [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) as research; direct Kitchen `0.2.28` at [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) as accepted Engine dependency |
| acceptance/tier | Engine public-API artifacts, direct Kitchen counters, lifecycle, memory, and creator review |

Engine does not execute ComfyUI. Existing recipe editions containing `comfy` identify
artifact/workflow provenance only.

## Decision

Klein 4B is the golden Engine-native stored-weight implementation:

- matching Distilled BF16: **Reference**;
- first-party Distilled NVFP4: **Recommended** on qualified Blackwell;
- first-party Distilled FP8: **Fallback**;
- Base FP8 edit: **Alternate**, not a Distilled precision variant.

Ordinary 1024-square T2I and one-reference edit are Hardware-proven through
Engine-owned typed orchestration, staged residency, caches, workers, and direct
Kitchen/native dispatch.

## Operation boundaries

| Operation | Behavioral contract | Engine status |
| --- | --- | --- |
| Distilled T2I | four steps, CFG 1, Euler/Flux2 schedule | BF16/FP8/NVFP4 accepted ladder |
| Distilled edit | one active reference; disabled two-reference example; nearest-exact scaling toward one megapixel | one reference accepted; two pending; third is an Engine extension |
| Base edit | 20 steps, CFG 5, Base closure and small decoder | FP8 Alternate; matching Base BF16 gap |
| inpaint/control | no approved typed contract here | absent, not an implied fallback |

## Accepted closures

| Path | Declared bytes | Status |
| --- | ---: | --- |
| Distilled BF16 | package Reference closure | Reference |
| Distilled NVFP4 | 10,857,495,368 | Recommended |
| Distilled FP8 | 12,467,706,400 | Fallback |
| Base FP8 edit | 12,399,885,870 | Alternate |
| Base NVFP4 | 10,797,932,158 | blocked by missing Base BF16 comparison |

Exact revisions, hashes, header schemas, and typed roles remain in Engine declarations
and retained manifests.

## Proof to preserve

Seed `43301611940728`, 1024-square, three reset runs and three meaningful warm runs per
recipe established deterministic per-recipe output, positive direct Kitchen dispatch,
zero accepted dense/eager fallback, and Recommended-to-Fallback-to-Recommended
switching. This is operational proof, not perceptual equivalence.

## Next slices

1. cancellation during encoder load, materialization, denoise, and decode, followed by
   observed cleanup and recovery;
2. translate the disabled two-reference behavior into Engine-owned typed logic and
   independent fixtures;
3. separately labeled three-reference Engine extension;
4. matching Base BF16 creator comparison;
5. no new format without measured creator value.

Stop on ComfyUI dependency, graph drift, fallback, stale media cache, false
availability, or poisoned recovery.
