# FLUX.2 Klein 4B roadmap

Last authority audit: **2026-08-15**

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

All five optimized recipes are narrow Hardware-proven through LatentSlate-originated
Engine jobs: T2I, ordered one-to-three-reference edit, cancellation with no artifact,
and warm recovery. Every accepted job retained exact direct-Kitchen module closure and
zero dense/eager fallback. The two BF16 recipes remain unaccepted References.

## Operation boundaries

| Operation | Behavioral contract | Engine status |
| --- | --- | --- |
| Distilled T2I | four steps, CFG 1, Euler/Flux2 schedule | BF16/FP8/NVFP4 accepted ladder |
| Distilled edit | ordered one-to-three-reference Engine conditioning; nearest-exact scaling toward one megapixel | one, two, and three references accepted; three remains an explicitly labeled Engine extension |
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
switching. Final LatentSlate acceptance traces are retained by prefix:

| Operation/tier | App / Engine / artifact SHA-256 trace prefixes |
| --- | --- |
| T2I NVFP4 Recommended | `c3591472` / `c40e4982` / `B1F385` |
| T2I FP8 Fallback | `fafe72d7` / `b834455a` / `DA9834` |
| I2I NVFP4 Recommended, one ref | `ae44d09b` / `6b7740e7` / `8DD520` |
| I2I Base FP8 Alternate, two refs | `aaf5d514` / `0de5bdba` / `CCC72E` |
| I2I Distilled FP8, three refs | cancel `7919827f` / `4b6dade5`; recovery `78debad1` / `6ad068d2` / `1FAC71` |

The canceled job emitted no artifact; recovery used a new successful job. This is
operational acceptance, not perceptual, pixel, or latent parity with Comfy.

## Next slices

1. broaden held-input creator and reference-diversity coverage;
2. retain cancellation/recovery, switching, exact module-count, and zero-fallback
   evidence as dependencies evolve;
3. matching Base BF16 creator comparison on high-memory hardware;
4. no new format without measured creator value.

Stop on ComfyUI dependency, graph drift, fallback, stale media cache, false
availability, or poisoned recovery.
