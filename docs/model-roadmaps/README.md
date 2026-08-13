# LatentSlate Engine model roadmaps

Last portfolio correction: **2026-08-12**

Portfolio architecture baseline: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Post-dispatch Wan 5B truth reconciled without merging: [`f59c3970d7ca72d63533f9eb37d8f0dcc91b2810`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/f59c3970d7ca72d63533f9eb37d8f0dcc91b2810)

Verified upstream source set:

- [Comfy workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [ComfyUI `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [Comfy Kitchen `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4)
- [comfy-model-tools `1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768)
- [ComfyUI examples `f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3)

Every full GitHub commit or blob link in this directory was resolved through the GitHub API during this correction. Hugging Face links that do not contain a 40-character revision are intentionally mutable discovery/model-card links. Exact artifact identity is carried separately as revision, filename, byte count, and SHA-256; an unresolved gated identity remains a blocker rather than a guessed pin.

These documents are executable implementation briefs. They separate official workflow behavior, publisher claims, code-proven facts, community claims, and LatentSlate inference. Use [IMPLEMENTATION_PACKETS.md](./IMPLEMENTATION_PACKETS.md) to dispatch one bounded slice.

## Status vocabulary

- **Reference**: source-of-truth comparison path for the same lineage and operation. It may require cloud or larger local hardware.
- **Recommended**: opinionated production default after creator, lifecycle, and hardware acceptance.
- **Fallback**: another strong production choice, often broader in hardware or compatibility.
- **Alternate**: valid specialized quality, lineage, or workflow choice.
- **Experimental**: exact plausible path whose implementation or meaningful qualification is incomplete.
- **Deferred**: real or plausible but blocked by artifacts, evidence, license clearance, runtime support, or product value.
- **Rejected**: deliberately outside the current product direction.

Stored artifacts are preferred. Normal Engine execution must not quantize, repack, or convert weights.

## Proof vocabulary

- **Hardware accepted**: fixed public-API output, intended backend, lifecycle, and provenance passed on the target workstation class.
- **Runtime proven**: an end-to-end Engine job completed, but the target acceptance matrix is incomplete.
- **Structurally tested**: schema, header, materialization, or mocked lifecycle tests exist without accepted creator output.
- **Cataloged**: recipe/resource declarations exist, but execution proof is incomplete.
- **Direct runtime only**: callable family code exists outside a complete opinionated acquisition/recipe surface.
- **Not implemented**: no native Engine family path exists.

## Portfolio matrix

| Family | Native boundary worth owning | Reference | Product disposition | Proof | Next bounded packet |
| --- | --- | --- | --- | --- | --- |
| [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md) | Distilled T2I and ordered-reference edit; Base edit separate | matching first-party BF16 | NVFP4 Recommended on Blackwell; FP8 Fallback; Base FP8 Alternate | 1024-square T2I/one-reference edit hardware accepted | cancellation recovery, then official two-reference topology |
| [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md) | ordinary Distilled T2I/edit; Base and KV separate | authenticated matching BF16 | NVFP4 Recommended on Blackwell; FP8 Fallback | quantized ordinary paths hardware accepted; BF16 exceeds 15.9 GiB | cancellation recovery, then two/three-reference lifecycle |
| [Krea 2](./KREA_2.md) | Turbo T2I only; Raw is training/foundation | official Turbo BF16 | official Comfy four-file INT8 plus Darkbrush LoRA is Experimental | Not implemented | license decision and four-file header manifest |
| [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md) | Base T2I only if native ecosystem value is proven | official FP16 Base | Base-only Experimental; generic Comfy Fallback | Not implemented | product-value spike before loader work |
| [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md) | ordered one/two-image editing; three inputs explicit extension | official 40-step BF16 | saved-default INT8 graph and optional four-step Lightning mode are separate Experimental paths | Not implemented | freeze ordered inputs, then implement one exact mode |
| [Ideogram 4](./IDEOGRAM_4.md) | structured-JSON T2I with conditional and unconditional branches | official NF4 public baseline; no public dense teacher | exact four-file INT8 graph Experimental; FP8/NVFP4 deferred until mode closure is fixed | Not implemented | license decision and JSON request contract |
| [Z-Image / Turbo](./Z_IMAGE_TURBO.md) | Turbo T2I; Base separate; no invented edit | exact Turbo BF16 | exact three-file Comfy INT8 ConvRot is strongest new-family candidate | Not implemented | Turbo INT8 T2I loader |
| [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md) | distinct T2V and required-one-image I2V over one exact Comfy closure | complete BF16 per operation at a matching schedule | split Comfy path is an Experimental/Recommended candidate pending broader quality review | T2V, I2V, switching, cancellation, and one LoRA hardware accepted at `f59c397` | broaden creator-quality corpus; optional matching BF16 study |
| [Wan 2.2 14B](./WAN22_14B.md) | exact two-expert I2V; T2V separate | official dense BF16 I2V pair | exact five-resource stored-FP8 I2V Experimental incumbent | typed runtime exists; reproducible support acquisition incomplete | exact support manifest, then hardware acceptance |
| [LTX 2.3](./LTX_2_3.md) | Distilled T2V, first-frame, and first+last conditioning | matching official Distilled BF16 | current BF16 path Experimental; official FP8 only after acceptance | cataloged synchronized-A/V runtime | T2V acceptance, then conditioned reuse |
| [LTX 2.5](./LTX_2_5.md) | distinguish current Comfy FLF graph from publisher BF16 two-stage T2V/I2V | operation-matched publisher BF16 | no Recommended path; gated exact identities still unresolved | Not implemented | capture five-file Comfy and BF16 two-stage closures separately |
| [MiniMax H3](./MINIMAX_H3.md) | FL2VA T2VA/endpoints first; Ref2VA separate | exact official BF16 checkpoint matching operation | current BF16 FL2VA Experimental; no low-bit recommendation | direct older-closure runtime; output acceptance pending | re-pin current FL2VA closure and qualify T2VA |

## Priority rule

1. Close acquisition and acceptance gaps in existing Wan 14B, LTX 2.3, Klein, and H3 paths.
2. Treat Wan 5B implementation as complete; expand evidence only when needed.
3. Implement Z-Image Turbo as the next bounded new image family.
4. Take Qwen editing after ordered-input and saved-default-versus-Lightning semantics are frozen.
5. Keep license- or environment-heavy Krea, Ideogram, and LTX 2.5 behind explicit gates.
6. Retain SDXL as an ecosystem decision, not an automatic legacy obligation.

## Engine seams agents should reuse

At the portfolio baseline, `resources.py`, `recipes.py`, and `variants.py` provide immutable acquisition identity and deterministic closure. `runtime/kit.py` fingerprints exact resources and execution policy. `runtime/cache.py` provides byte-bounded CPU-frozen caches. Runtime manager and residency modules provide warm switching and ejection. `stored_quant.py` recognizes exact global FP8, legacy scaled FP8, and tensorwise INT8 ConvRot contracts. Klein proves fail-closed Kitchen materialization and dispatch provenance. Wan 14B proves typed multi-component recipes. H3 and LTX 2.3 prove complete-repository validation and synchronized-A/V lifecycle. Current `main` additionally proves a narrow three-role Wan 5B Comfy recipe.

Do not introduce a second catalog, a generic `quantization=` format switch, runtime conversion, or family loaders that accept incomplete component sets.

## Shared acceptance contract

Target workstation: Windows 11, RTX 5080 with 15.9 GiB usable VRAM, 63.8 GiB RAM, CUDA 13, and Kitchen-capable Blackwell.

Each packet records exact recipe/resource identities, schema fingerprints, effective request, selected backend, fallback counts, preprocessing order, cache state, component residency, stage boundaries, output hashes, allocator VRAM, approximate external GPU sampling, process RSS, Windows commit, and disk/PCIe traffic where practical.

Required scenarios are runtime-cold, three meaningful warm repeats, A-to-B-to-A switching, cancellation/recovery, malformed artifact, teardown, and operation-specific multi-input cases. A cache replay is labeled as such and is not counted as an independent stochastic repeat.

Stop immediately on lineage/settings mismatch, unknown header/layout, hidden dense copy, eager/dequantized fallback in a claimed native path, OOM, corrupt output, poisoned recovery, or obvious creator-visible regression. Large BF16 references remain valid cloud/Vast targets; never weaken parity settings merely to fit locally.

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
