# LatentSlate Engine model roadmaps

Last portfolio audit: **2026-08-12**  
Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Current upstream evidence set: [Comfy workflow templates `2b7f823`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ec`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c), [Comfy Kitchen `9816d22`](https://github.com/Comfy-Org/comfy-kitchen/tree/9816d220021ab526e2cc1700a68b68d1b72d961c), and [comfy-model-tools `1fe3410`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341001c27e8fe7e0450e8ce7fd3333d97c34c).

These are executable implementation briefs, not model roundups. Each document separates official workflow behavior, publisher claims, code-proven facts, community claims, and LatentSlate inference. The cross-family dispatch surface is [IMPLEMENTATION_PACKETS.md](./IMPLEMENTATION_PACKETS.md).

## Status vocabulary

- **Reference** — the source-of-truth comparison path for the same lineage and operation. It may require larger local or cloud hardware and is not automatically the product default.
- **Recommended** — the opinionated production default after measured creator, lifecycle, and hardware acceptance.
- **Fallback** — another strong production choice, normally broader in hardware or compatibility. It is not a warning tier.
- **Alternate** — a valid specialized quality, lineage, or workflow choice.
- **Experimental** — an exact plausible path whose implementation or meaningful qualification is incomplete.
- **Deferred** — real or plausible, but blocked by missing artifacts, evidence, license clearance, runtime support, or product value.
- **Rejected** — deliberately outside the current product direction.

A family may legitimately have no Recommended path. Stored artifacts are preferred; runtime quantization, repacking, or conversion is not a normal recipe.

## Proof vocabulary

- **Hardware accepted** — fixed public-API output, native backend, lifecycle, and provenance passed on the target workstation class.
- **Runtime proven** — an end-to-end Engine job completed, but the full target acceptance matrix is incomplete.
- **Structurally tested** — schema, header, materialization, or mocked lifecycle tests exist without accepted creator output.
- **Cataloged** — recipe/resource declarations exist, but execution proof is incomplete.
- **Direct runtime only** — callable family code exists outside a complete opinionated acquisition/recipe surface.
- **Not implemented** — no native Engine family path exists.

## Portfolio matrix

| Family | Native boundary worth owning | Reference | Current product disposition | Proof at audited commit | Next bounded packet |
| --- | --- | --- | --- | --- | --- |
| [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md) | Distilled T2I and ordered-reference edit; Base edit is separate | matching first-party BF16 | NVFP4 Recommended on Blackwell; FP8 Fallback; Base FP8 Alternate | Hardware accepted for 1024² T2I/one-reference edit; lifecycle gaps remain | cancellation/recovery, then official two-reference graph |
| [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md) | ordinary Distilled T2I/edit; Base and KV remain separate | matching authenticated BF16 | NVFP4 Recommended on Blackwell; FP8 Fallback | Hardware accepted for 1024² ordinary T2I/one-reference edit; BF16 honestly exceeds 15.9 GiB | cancellation/recovery, then two/three-reference lifecycle |
| [Krea 2](./KREA_2.md) | Turbo T2I only; Raw is training/foundation | official Turbo BF16 | no Recommended path; exact official Comfy INT8 topology is Experimental | Not implemented | resolve license/product gate and immutable INT8 closure, then Turbo T2I |
| [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md) | Base T2I only if native ecosystem value is proven | official FP16 Base | no Recommended path; Base-only Experimental | Not implemented | product-value gate before loader work |
| [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md) | ordered one/two-image editing; three inputs are an explicit extension | official 40-step BF16 edit | four-step BF16 LoRA Reference for distilled line; Comfy INT8 ConvRot Experimental | Not implemented | exact one/two-input four-step loader and lifecycle |
| [Ideogram 4](./IDEOGRAM_4.md) | structured-JSON T2I with dual conditional/unconditional branches | official NF4 public baseline; no public dense teacher | FP8 then NVFP4 Experimental; hosted API/Comfy Fallback | Not implemented | human license gate and JSON request contract |
| [Z-Image / Turbo](./Z_IMAGE_TURBO.md) | Turbo T2I; Base is separate; no invented edit | exact Turbo BF16 | exact three-file Comfy INT8 ConvRot closure is the strongest new Recommended candidate | Not implemented | **next new-family slice: Turbo INT8 T2I loader** |
| [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md) | T2V first, I2V only after reuse proof | official BF16 per operation | split official Comfy FP16 T2V is Experimental | BF16 T2V cataloged/direct runtime; acceptance pending | active-agent-owned split FP16 T2V; do not duplicate |
| [Wan 2.2 14B](./WAN22_14B.md) | exact two-expert I2V; T2V is a separate closure | official dense BF16 I2V pair | exact stored-FP8 I2V Experimental incumbent | typed recipe/runtime exists; reproducible support acquisition and acceptance incomplete | **safe support manifest, then hardware acceptance** |
| [LTX 2.3](./LTX_2_3.md) | Distilled T2V, first-frame I2V, and first+last conditioning | matching official Distilled BF16 | current BF16 path Experimental; official FP8 only after acceptance | cataloged native BF16 runtime with synchronized A/V | T2V acceptance, then conditioned reuse |
| [LTX 2.5](./LTX_2_5.md) | Distilled two-stage synchronized A/V; Dev/refinement separate | matching official Distilled BF16 component set | no Recommended path; exact Distilled BF16 two-stage Experimental | Not implemented | immutable gated closure and isolated dependency proof |
| [MiniMax H3](./MINIMAX_H3.md) | FL2VA T2VA/endpoints first; Ref2VA separate | exact official BF16 checkpoint matching operation | current BF16 FL2VA path Experimental; no low-bit recommendation | direct BF16 runtime exists against an older closure; output acceptance pending | re-pin exact current FL2VA closure and qualify T2VA |

## Priority rule

The portfolio is ordered by the cheapest trustworthy creator value, not model prestige:

1. close acquisition and acceptance gaps in existing Wan 14B, LTX 2.3, Klein, and H3 runtimes;
2. finish the actively owned Wan 5B split-component tranche without parallel duplication;
3. implement Z-Image Turbo as the next bounded new image family;
4. take Qwen editing after its ordered-input and four-step contracts are frozen;
5. defer license- or environment-heavy Krea, Ideogram, LTX 2.5, and H3 low-bit work until their explicit gates clear;
6. retain SDXL as an ecosystem decision, not an automatic legacy obligation.

See [the packet table](./IMPLEMENTATION_PACKETS.md#priority-ordered-packets) for one-agent slices.

## Engine architecture facts agents should reuse

At the audited commit, `resources.py`, `recipes.py`, and `variants.py` already provide immutable acquisition identity and deterministic resource closure. `runtime/kit.py` fingerprints exact resources and execution policy; `runtime/cache.py` provides byte-bounded CPU-frozen caches; manager/residency modules provide warm switching and ejection. `stored_quant.py` already recognizes exact global FP8, legacy scaled FP8, and tensorwise INT8 ConvRot contracts. Klein proves fail-closed Kitchen materialization and dispatch provenance. Wan 14B proves a typed multi-component recipe; H3 and LTX 2.3 prove complete-repository validation and synchronized A/V lifecycle.

Do not bypass these seams with a second catalog, generic `quantization=` switch, runtime conversion, or a family loader that accepts incomplete component sets. Only Wan 14B and Klein currently have typed component-recipe contracts; other split families require an explicitly reviewed schema extension.

## Shared scientific acceptance contract

Target workstation: Windows 11, RTX 5080 with 15.9 GiB usable VRAM, 63.8 GiB RAM, CUDA 13 and Kitchen-capable Blackwell. Each packet uses the public API and records:

- exact recipe/resource revisions, byte identities, header/schema fingerprints, selected backend, layout counts, fallback counts, and effective request;
- one runtime-cold job, three stable warm repeats for promotion, A→B→A switching, cancellation/recovery, malformed-artifact failure, teardown, and family-specific multi-input cases;
- prompt/media preprocessing order, cache hits, component residency, stage boundaries, output hash, and creator-reviewed output corpus;
- allocator VRAM plus approximate external GPU sampling, process RSS, Windows commit/system RAM, disk and PCIe traffic where practical.

Stop immediately on lineage/settings mismatch, unknown header/layout, hidden dense copy, eager/dequantized fallback in a claimed native path, OOM, NaN/black/corrupt output, poisoned cancellation recovery, or obvious creator-visible regression. Large BF16 references remain valid cloud/Vast qualification targets; never weaken their settings merely to fit locally.

## Research and implementation discipline

- Pin workflow JSON and source code by immutable commit. A tutorial or mutable `main` link is discovery evidence, not a recipe lock.
- Record exact file bytes and SHA/LFS identity. Parameter-count estimates are not artifact facts.
- Keep operation and lineage boundaries explicit: Base versus Distilled, T2I versus edit, T2V versus I2V, one-stage versus two-stage, FL2VA versus Ref2VA.
- Official workflow behavior, publisher claims, code proof, community claims, and inference must remain labeled separately.
- Generic Comfy is the correct product surface for graph-heavy controls or community extensions until one exact operation earns native support.
- Every roadmap ends with bounded Terra/Sol-sized slices, prerequisites, tests, exclusions, and stop conditions.

## Roadmaps

- [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md)
- [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md)
- [Krea 2](./KREA_2.md)
- [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md)
- [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md)
- [Ideogram 4](./IDEOGRAM_4.md)
- [Z-Image / Z-Image Turbo](./Z_IMAGE_TURBO.md)
- [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md)
- [Wan 2.2 14B](./WAN22_14B.md)
- [LTX 2.3](./LTX_2_3.md)
- [LTX 2.5](./LTX_2_5.md)
- [MiniMax H3](./MINIMAX_H3.md)
- [Cross-family implementation packets](./IMPLEMENTATION_PACKETS.md)
