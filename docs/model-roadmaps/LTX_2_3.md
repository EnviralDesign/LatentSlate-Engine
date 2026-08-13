# LTX 2.3 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Lightricks sources and Engine’s immutable BF16 closure at upstream `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` |
| saved topology/defaults | official T2V, I2V, and FLF workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research only; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for direct Engine dispatch |
| acceptance/tier | Engine public-API A/V artifacts/lifecycle; optimized implementation is CPU/source complete but has no acceptance or runnable claim |

## Decision

Current main owns a strong native BF16 structural Reference, not a practical local
product path. One bounded RTX 5080 OOM is enough; dense output belongs on Vast.

Practical source contracts:

- T2V and first-frame I2V: Dev FP8 plus fixed Distilled LoRA;
- FLF: Distilled FP8 with ordered endpoints, a separate graph/transformer line.

The pinned workflow widgets default to **1280×720**. Engine deliberately exposes
**768×512** as the initial 16 GB acceptance preset: Dev's two-stage Diffusers path
requires the final dimensions to be divisible by 64, while 720 is not. Both values
are carried in variant provenance; this is an explicit compatibility/footprint
deviation, not an inferred workflow default.

Engine-native optimized work is implemented, cataloged, naturally installed, and
independently CPU/source reviewed. It remains neither runnable nor accepted until it
passes the required public-API A/V and disposable-lifecycle evidence. The saved
workflow contracts remain the authority for its topology and fixed defaults.

## BF16 truth

The exact 50-file closure totals **94,977,693,482 bytes** and includes transformer,
encoder/tokenizer, connectors, scheduler, video/audio VAEs, vocoder, and configs. It
fixes 24 fps, 8 steps/CFG 1, `8n+1` frames, aligned dimensions, synchronized 48 kHz
stereo, typed operations, and Engine disposable-worker cancellation boundaries.

Proof: **Cataloged / structurally validated Reference, one bounded local OOM**.

## Optimized implementation truth

The three operations bind their exact workflow revision and raw JSON SHA-256, all
active A/V resources and fixed LoRAs, stored-FP8/NVFP4 header contracts, additive
LoRA targets, direct Kitchen dispatch, Engine-owned typed orchestration, disposable
workers, cancellation, mux, output probing, and provenance. The ordinary optimized
profile is **72,529,224,527 bytes** across seven canonical resources; each installed
file has one NTFS identity and acquisition rejects staged multiply-linked files.

No workflow is submitted to ComfyUI. T2V/I2V and FLF must not share the wrong lineage.

## Next slices

1. paired target-workstation T2V, first-frame I2V, and FLF public-API acceptance;
2. cancellation/recovery and post-job tree-empty proof for each operation;
3. promote only the operations whose observed A/V and native dispatch evidence passes;
4. optional batched dense BF16 Vast comparison;
5. no new 2.3 feature without specific compatibility value.

Stop on ComfyUI dependency, unlanded-status claims, partial closure, lineage
substitution, hidden conversion/fallback, assumed A/V metadata, or unobserved cleanup.
