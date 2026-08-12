# Cross-family implementation packets

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

This is the dispatch index for one bounded implementation or acceptance slice. Read the linked roadmap section before source discovery. Complexity means implementation delicacy and integration surface, not model status. **Terra-high** is appropriate for exact, well-bounded seams; **Sol escalation** is reserved for novel tensor mapping, dual/multi-stage lifecycle, dependency conflicts, or ambiguous upstream contracts.

## Priority-ordered packets

| Priority | One-agent packet | Reuse/dependency | Complexity and agent | First read | Human gate | Do not rediscover |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Wan 14B I2V: publish an exact filtered support acquisition manifest | existing typed five-role recipe, resource closure, I2V runtime | **Small, Terra-high** — acquisition only | [Wan 14B exact closure](./WAN22_14B.md#exact-engine-i2v-closure) | choose file-by-file official snapshot versus first-party packaged support | four weight resources are already exact; only the 529,069,044-byte support directory is unreproducible |
| 2 | Wan 14B I2V: target-workstation cold/warm/cancel/recovery acceptance | packet 1; current two-expert stored-FP8 runtime | **Medium, Terra-high** — lifecycle/provenance | [Wan 14B acceptance](./WAN22_14B.md#hardwarescientific-acceptance-packet) | dense BF16 comparison may run on Vast rather than local | 20 steps, split at 10, both CFG 3.5, shift 8; experts are ordered and must not coexist accidentally |
| 3 | LTX 2.3 Distilled BF16 T2V acceptance | existing complete-repository recipe/runtime and A/V mux | **Medium, Terra-high** — measurement, not new loader | [LTX 2.3 acceptance](./LTX_2_3.md#hardwarescientific-acceptance-packet) | approve large local closure or cloud reference execution | Engine already fixes 24 fps, 8 steps, CFG 1, `8n+1` frames, VAE tiling, prompt cache |
| 4 | LTX 2.3 first-frame and first+last conditioning reuse acceptance | packet 3; same repository/runtime | **Medium, Terra-high** — ordered condition lifecycle | [LTX 2.3 operation boundary](./LTX_2_3.md#productoperation-boundary) | none after packet 3 | first image is required; end image is index `-1`; this is not generic I2V input multiplicity |
| 5 | Klein 4B/9B cancellation and clean recovery | accepted FP8/NVFP4 runtimes and hardware harness | **Small, Terra-high** — missing lifecycle cell | [4B slices](./FLUX2_KLEIN_4B.md#ordered-bounded-slices), [9B slices](./FLUX2_KLEIN_9B.md#ordered-bounded-slices) | none | native dispatch, deterministic 1024² cold/warm, and A→B→A already passed; do not rerun broad discovery |
| 6 | Klein 4B/9B official two-reference, then deliberate three-reference extension | packet 5; existing ordered reference cache | **Medium, Terra-high** — cache key/invalidation | [4B boundary](./FLUX2_KLEIN_4B.md#product-and-operation-boundary), [9B boundary](./FLUX2_KLEIN_9B.md#product-and-operation-boundary) | creator corpus approval for third-reference extension | official ordinary graphs prove one active reference and demonstrate two disabled; three is Engine-specific |
| 7 | H3 FL2VA: re-pin exact current BF16 closure and diff it against Engine's older snapshot | existing H3 complete-repository validator/runtime | **Medium, Terra-high** — metadata/config audit | [H3 Comfy closure](./MINIMAX_H3.md#officialdefault-comfy-closure) | H3 Community License/product review | Engine already excludes `transformer_ref/**`; FL2VA and Ref2VA are separate checkpoints |
| 8 | H3 FL2VA T2VA/endpoints acceptance on target hardware | packet 7; current runtime | **Large, Terra-high** — 33B full-attention lifecycle | [H3 acceptance](./MINIMAX_H3.md#hardwarescientific-acceptance-packet) | approve smaller diagnostic canvas while retaining exact 768p cloud reference | 24 fps, stereo A/V, `17k+5` Engine frame grid; released inference is full attention, not sparse |
| 9 | Wan 5B split FP16 T2V implementation/acceptance | actively owned external tranche; reuse Wan BF16 runtime, resource components, UMT5 cache | **Medium, Terra-high** — coordinate, do not parallelize | [Wan 5B coordination boundary](./WAN22_TI2V_5B.md#decision-and-coordination-boundary) | confirm current owner/branch before touching files | official Comfy split is transformer + scaled-FP8 UMT5 + Wan 2.2 VAE; Engine's 50-step path and Comfy's 30-step graph are not equivalent |
| 10 | Z-Image Turbo INT8 ConvRot T2I exact loader | stored ConvRot planner/materializer, mixed encoder seam, Klein residency | **Medium-large, Terra-high** — new family but exact graph | [Z resource closure](./Z_IMAGE_TURBO.md#exact-turbo-int8-resource-closure), [runtime packet](./Z_IMAGE_TURBO.md#loaderruntime-implementation-packet) | approve new typed `transformer`/`text_encoder`/`vae` recipe contract | three-file closure is 12,168,299,735 bytes; graph is 8 steps, CFG 1, shift 3, `res_multistep`/`simple` |
| 11 | Z-Image Turbo target-workstation acceptance | packet 10 | **Medium, Terra-high** — deterministic study | [Z acceptance](./Z_IMAGE_TURBO.md#hardwarescientific-acceptance-packet) | creator corpus review | no official Z-Image Edit artifact was verified; do not add optional image input |
| 12 | Qwen 2511 one/two-image four-step BF16-LoRA request contract and reference harness | ordered media schema, prompt/media cache, generic resource closure | **Medium, Terra-high** — request semantics first | [Qwen operation boundary](./QWEN_IMAGE_EDIT_2511.md#productoperation-boundary) | decide whether Engine supports publisher-proven two inputs only or also labels a three-input extension | publisher example proves two images; current Comfy template exposes up to three sockets but top-level activates one |
| 13 | Qwen 2511 INT8 ConvRot Lightning loader | packet 12; stored ConvRot + scaled-FP8 encoder + VAE + fixed LoRA | **Large, Sol escalation** — 20B-class mapping and fused/LoRA semantics | [Qwen closure](./QWEN_IMAGE_EDIT_2511.md#exact-artifact-closure), [runtime packet](./QWEN_IMAGE_EDIT_2511.md#loaderruntime-implementation-packet) | approve exact Comfy artifact revision/license and 20-step-versus-four-step product contract | fused scaled-FP8 Lightning is not a precision-only copy of 40-step BF16; compare to BF16 plus the same four-step LoRA |
| 14 | Krea 2 Turbo: immutable official Comfy INT8 closure and license decision | resource authoring, stored ConvRot, Krea Comfy model/encoder source | **Medium, Terra-high** — research-to-contract | [Krea artifacts](./KREA_2.md#exact-artifact-closure) | Community License revenue/content-filter/product acceptance | Krea 2 is not FLUX; Raw is training/foundation, Turbo is 8-step product inference |
| 15 | Krea 2 Turbo INT8 T2I loader and 16 GB acceptance | packet 14; Z/Klein stored-loader seams | **Large, Sol escalation** — fused projections/encoder topology | [Krea runtime packet](./KREA_2.md#loaderruntime-implementation-packet) | packet 14 must clear | no I2I/edit operation was proven; do not invent one from generic image-conditioned research |
| 16 | SDXL Base-only native T2I value spike | Diffusers repository validator/runtime skeleton | **Small-medium, Terra-high** — familiar pipeline | [SDXL decision](./STABLE_DIFFUSION_XL.md#decision), [slices](./STABLE_DIFFUSION_XL.md#ordered-bounded-slices) | identify a concrete workflow that generic Comfy does not already serve adequately | Base fits FP16; low-bit work is not the reason to support SDXL; Refiner is a separate two-stage operation |
| 17 | Ideogram 4 structured JSON request and expansion provenance contract | generic request/resource layers only | **Medium, Terra-high** — schema/product semantics | [Ideogram boundary](./IDEOGRAM_4.md#product-and-operation-boundary) | Non-Commercial license and hosted/local expansion product decision | public weights provide no dense source of truth; prompt JSON and expansion provenance are part of the operation |
| 18 | Ideogram 4 complete dual-branch FP8 loader | packet 17; Kitchen scaled FP8, staged Qwen encoder | **Large, Sol escalation** — conditional/unconditional model choreography | [Ideogram closure](./IDEOGRAM_4.md#exact-resource-closures), [runtime packet](./IDEOGRAM_4.md#loaderruntime-implementation-packet) | license gate must clear | one diffusion file is incomplete; conditional and unconditional branches plus encoder and VAE are mandatory |
| 19 | Ideogram 4 NVFP4 Blackwell challenger | packet 18 | **Large, Sol escalation** — dual branch plus packed layout | [Ideogram candidates](./IDEOGRAM_4.md#recipe-ladder-and-candidates) | FP8 path must first be accepted | raw NVFP4 closure still exceeds 16 GB before VAE/buffers; staging and native dispatch are mandatory |
| 20 | LTX 2.5: immutable Distilled two-stage closure and isolated dependency proof | resource catalog, new family recipe, LTX 2.3 lifecycle lessons | **Large, Sol escalation** — gated artifacts and package constraint | [LTX 2.5 components](./LTX_2_5.md#exact-component-closure), [runtime packet](./LTX_2_5.md#loaderruntime-implementation-packet) | LTX license, gated access, Transformers compatibility | custom Gemma 4 encoder is not stock Gemma; Distilled two-stage, Dev, DFR, DubIt are separate products |
| 21 | LTX 2.5 Distilled two-stage T2V with convolutional video VAE | packet 20 | **Large, Sol escalation** — stage transition, spatial upscaler, synchronized A/V | [LTX 2.5 candidate](./LTX_2_5.md#recipe-ladder-and-candidate-contract) | approve local diagnostic settings and cloud parity settings | first path uses half-resolution stage 1, x2 latent upscaler, short full-resolution refinement; no runtime FP8 cast |
| 22 | LTX 2.5 first-frame I2V reuse | packet 21 | **Medium-large, Terra-high** after runtime exists | [LTX 2.5 boundary](./LTX_2_5.md#productoperation-boundary) | packet 21 acceptance | reuse exact Distilled closure; image conditioning is a separate request and corpus, not optional T2V input |
| 23 | H3 Ref2VA multimodal ingress/closure design | H3 FL2VA acceptance, generic ordered media/assets | **Large, Sol escalation** — images/video/audio multiplicity and separate checkpoint | [H3 operation boundary](./MINIMAX_H3.md#productoperation-boundary) | explicit creator demand and license/product approval | Ref2VA allows a documented multimodal mix and is not present in Engine's FL2VA closure; Context-IR/2K remain hosted services |
| 24 | H3 low-bit/sparse path | packets 7–8; only after first-party artifact/code exists | **Large, Sol escalation** | [H3 slices](./MINIMAX_H3.md#ordered-bounded-slices) | upstream artifact/release must exist | architecture mentions sparse attention, but current released inference is full attention; a Kitchen kernel alone proves nothing |

## Recommended dispatch

**Existing-path next work:** packet 1, then packet 2. It converts Wan 14B's already substantial implementation into a reproducible, reviewable product candidate without inventing another loader.

**Next new-family work:** packet 10, Z-Image Turbo INT8 T2I. It has the smallest exact new-family closure, an immutable official Comfy graph, strong reuse of Engine's accepted stored-weight seams, and a locally feasible acceptance target.

Do not dispatch packet 9 without checking the active Wan 5B owner. Do not dispatch packets 13, 15, 18–24 until their preceding gate packet is complete.

## Shared file-level handoff map

Likely reusable files at the audited commit:

- catalog/acquisition: `resources.py`, `recipes.py`, `variants.py`, built-in resource/recipe directories, deployment/profile closure;
- typed component contracts: `wan22_recipe.py`, `klein_recipe.py`; new families need equally narrow contracts rather than generic component dictionaries;
- exact stored formats: `stored_quant.py`, Klein stored transformer and mixed-Qwen materializers, Kitchen dispatch instrumentation;
- lifecycle: `runtime/kit.py`, `runtime/cache.py`, `runtime/manager.py`, residency policy, family runtime/tool modules;
- acceptance: public jobs API, hardware-study guidance, deterministic output/provenance records.

Every implementation packet must add or update schema/header tests, resource/recipe closure tests, fail-closed fallback tests, lifecycle/cancellation tests, and an opt-in public-API hardware scenario. No packet should modify a family it does not own merely to share code; extract a reusable seam only when the concrete implementation requires it.

## Universal stop conditions

Stop the packet and report evidence instead of improvising when:

- the exact official file/revision/bytes/hash cannot be resolved;
- a license/gate forbids the intended built-in acquisition or product use;
- source and target tensor names/shapes/sidecars cannot be mapped completely;
- Kitchen/native dispatch cannot be proven or falls back to eager/dequantized execution;
- the request would silently broaden official input multiplicity or operation semantics;
- a cancellation, failed load, or A→B→A switch leaves poisoned state;
- parity requires changing steps, guidance, dimensions, frame count, fps, preprocessing, or artifact lineage without recording a separate diagnostic run.
