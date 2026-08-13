# Ideogram 4 roadmap

Last reviewed: **2026-08-13**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Ideogram 4 is a compelling local design and typography target, but its public
artifact surface has an unusual qualification problem: **Ideogram publishes no dense
BF16/FP16 source of truth**. The first-party public model zoo contains NF4 and FP8,
and the pinned official workflow topology represents the conditional and unconditional model
branches as separate files.

That means Engine must not call any public low-bit artifact “lossless” or use one
branch in isolation. The official NF4 Diffusers pipeline remains the public baseline.
The current pinned workflow also supplies a complete four-file INT8 ConvRot
graph, making it the narrowest implementation candidate; FP8 and NVFP4 remain
separate follow-on closures whose prompt-assistant modes must be made explicit.

There is no native Engine family today and no Recommended path. A correct implementation
also needs a **structured JSON prompt contract**, not a casual plain-text string hidden
behind an unrecorded hosted “magic prompt” request.

## Evidence labels

- **Verified** — stated by Ideogram, Comfy, or the Engine source at the audited commit.
- **Publisher measurement** — an upstream benchmark or quality claim; not an Engine
  result.
- **Inference** — a roadmap judgment that must be validated on the target workstation.

## Scope and topology

Ideogram 4 is a 9.3B text-to-image foundation model trained from scratch. It is not a
FLUX derivative. Its official architecture uses:

- a 34-block fully single-stream DiT;
- Qwen3-VL-8B-Instruct as the text/vision-language encoder, concatenating hidden
  states from 13 layers;
- separate conditional and unconditional guidance branches;
- a VAE decode stage;
- structured JSON captions with optional bounding boxes, color-palette conditioning,
  and compositional decomposition.

Verified sources: the official
[Ideogram 4 repository at `990fe1c`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2),
[architecture document](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/model_architecture.md),
and [prompting guide](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/prompting.md).

### Canonical sampling lines

| Preset | Steps | Guidance schedule | Schedule | Use |
| --- | ---: | --- | --- | --- |
| `V4_QUALITY_48` | 48 | 45 steps at guidance 7, then 3 polish steps at guidance 3 | `mu=0.0`, `std=1.5` | Official default / quality reference |
| `V4_DEFAULT_20` | 20 | 18 steps at guidance 7, then 2 polish steps at guidance 3 | `mu=0.0`, `std=1.75` | Faster standard line |
| `V4_TURBO_12` | 12 | 11 steps at guidance 7, then 1 polish step at guidance 3 | `mu=0.5`, `std=1.75` | Speed preset, not a distinct checkpoint |

The official [inference reference](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/inference.md)
supports dimensions from 256 to 2048, each a multiple of 16, with aspect ratios up to
6:1. Do not compare different presets, prompt-expansion methods, or resolutions as a
quantization result.

## Public artifacts and exact component closures

### First-party model zoo

| Artifact | Format / runtime | Verified state | Disposition |
| --- | --- | --- | --- |
| [`ideogram-ai/ideogram-4-nf4`](https://huggingface.co/ideogram-ai/ideogram-4-nf4) / [`-nf4-diffusers`](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers) | NF4, CUDA, official Diffusers support | Gated; Ideogram 4 Non-Commercial Model Agreement | **Reference baseline**, explicitly not dense truth |
| [`ideogram-ai/ideogram-4-fp8`](https://huggingface.co/ideogram-ai/ideogram-4-fp8) | First-party FP8; official Ideogram runtime, not Diffusers | Gated; same non-commercial license | Provenance source for the Comfy repack; **Experimental** |
| Dense BF16/FP16 | Not published | No public artifact exists in the official model zoo | **Unverified / unavailable**, never invent a reference |

The official repository says JSON captions are required for the trained interface.
Its convenience CLI can call Ideogram's hosted magic-prompt API, while Diffusers also
provides a local Qwen3-VL prompt-upsample path that the model card says may reduce
quality. Hosted expansion and local expansion are different product operations and
must be recorded in the request provenance.

### Pinned official workflow topology

The current [Comfy-Org repository](https://huggingface.co/Comfy-Org/Ideogram-4)
requires a conditional diffusion model, an unconditional diffusion model, one
Qwen3-VL text encoder, and `flux2-vae.safetensors`.

| Component artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`ideogram4_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/b6c440e84da24539b8457c865e7994d4d87447f5/diffusion_models/ideogram4_fp8_scaled.safetensors) | Conditional scaled-FP8 model | **9.28 GB**; SHA-256 `49a946f1b0f8bcf5eab7d3b1ecc7b453c104e034cb1b592032745692724bd306` | Part of complete FP8 challenger |
| [`ideogram4_unconditional_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors) | Unconditional scaled-FP8 model | **9.28 GB**; SHA-256 `9b359007dae162cca7591d00868feea733eb7c56e56e3a214a4d5a9a2a07cd60` | Part of complete FP8 challenger |
| [`qwen3vl_8b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/b6c440e84da24539b8457c865e7994d4d87447f5/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) | Scaled-FP8 Qwen3-VL encoder | **10.6 GB**; SHA-256 `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` | Part of complete FP8 challenger |
| [`ideogram4_nvfp4_mixed.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9/diffusion_models/ideogram4_nvfp4_mixed.safetensors) | Conditional mixed NVFP4 | **5.49 GB**; SHA-256 `e7923b4b0a1129ae5afcc09e63046185688c8b09eb9a1a748cccdbde5d381609` | Part of complete Blackwell challenger |
| [`ideogram4_unconditional_nvfp4_mixed.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9/diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors) | Unconditional mixed NVFP4 | **5.49 GB**; SHA-256 `639e37bd1dd7ee35e23c7cfccf93a518ddc7f4587818956ec42b31e659fd6ac0` | Part of complete Blackwell challenger |
| [`qwen3vl_8b_nvfp4.safetensors`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/f2aa293eb4564d79d6bcdeb4ea263ab7af7f99f9/text_encoders/qwen3vl_8b_nvfp4.safetensors) | NVFP4 Qwen3-VL encoder | **6.31 GB**; SHA-256 `e462e9e0c3b9313ae17f82040d7c77beb92d7aef3e40692d7803228dab7c3b98` | Part of complete Blackwell challenger |
| Conditional and unconditional INT8 ConvRot | Comfy/Kitchen INT8 ConvRot | **9.58 GB each**; exact published files and pinned four-file workflow exist below | **Experimental first native candidate** |
| `flux2-vae.safetensors` | VAE | Required by the pinned workflow topology | Shared component; pin exact identity before implementation |

The active FP8 closure is already about **29.2 GB before the VAE**. The NVFP4 closure
is about **17.3 GB before the VAE**. Neither is a simple fully resident 16 GB path;
NVFP4 materially narrows the problem but still requires measured staging, temporary
buffers, and backend proof.

Comfy/Kitchen format support is relevant only after the exact four-role topology is
accepted. The existence of FP8, NVFP4, or INT8 kernels does not prove that Engine can
load these particular files, preserve the dual branches, and dispatch the intended
backend on SM120.

### Verified official INT8 workflow closure

The pinned [INT8 workflow template](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i_int8.json)
uses two separate ConvRot diffusion branches, Qwen3-VL, and Flux2 VAE. It is the
first exact Engine-native closure to implement; one diffusion file is never an Ideogram 4
recipe.

| Role | Immutable source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Conditional transformer | [`Comfy-Org/Ideogram-4`, `diffusion_models/ideogram4_int8_convrot.safetensors`, `e18159a2e9a95cdb4ecd76f49cecdf5291849697`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/e18159a2e9a95cdb4ecd76f49cecdf5291849697/diffusion_models/ideogram4_int8_convrot.safetensors) | 9,583,465,712 | `a9164002943463b4c7b2abd88c82a488c088acc35762651e4d8604d6ce4a163d` |
| Unconditional transformer | [`Comfy-Org/Ideogram-4`, `diffusion_models/ideogram4_unconditional_int8_convrot.safetensors`, `8532c0f76182375c10b8f082dc6b0be196ef0615`](https://huggingface.co/Comfy-Org/Ideogram-4/blob/8532c0f76182375c10b8f082dc6b0be196ef0615/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors) | 9,583,465,712 | `cd03ed94f244c9cb705e7d30ca0f40b5f5b004bb20674117adff88d16416c23d` |
| Text/vision encoder | [`Comfy-Org/Qwen3-VL`, `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`, `7f1d4413e3bd9ae24580b14d4113bfce872c55f0`](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/7f1d4413e3bd9ae24580b14d4113bfce872c55f0/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) | 10,588,637,512 | `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` |
| VAE | [`Comfy-Org/flux2-dev`, `split_files/vae/flux2-vae.safetensors`, `ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa`](https://huggingface.co/Comfy-Org/flux2-dev/blob/ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa/split_files/vae/flux2-vae.safetensors) | 336,213,556 | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |

The fixed total is **30,091,782,492 bytes**
(`9,583,465,712 + 9,583,465,712 + 10,588,637,512 + 336,213,556`). All four
file sizes and LFS SHA-256 values were rechecked through immutable Hugging Face file
metadata on 2026-08-13. The composite closure has multiple license gates: Ideogram's
non-commercial terms for the model branches, Apache-2.0 for the Qwen repack, and the
Flux2 VAE's own non-commercial terms. Product review must approve the whole closure.

The template's note confirms the essential operation semantics: structured JSON
captions; flow matching with asymmetric classifier-free guidance; and a built-in
safety-filter outcome. The native request/provenance contract must store the original
text, exact JSON caption, layout/palette fields, and any prompt-assistant identity;
plain text, hosted expansion, and local expansion are distinct modes.

## Current Engine truth at `2ba5709`

- **Not implemented.** Ideogram 4 is absent from Engine's family registry and tool
  catalog:
  [`model_store.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/model_store.py),
  [`tools/__init__.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/__init__.py).
- **No prompt contract.** Engine has no structured Ideogram caption schema, bounding-box
  normalization, palette representation, magic-prompt provider, or local expansion
  path.
- **No acquisition or loader contract.** There are no recipes, resource declarations,
  licenses/gates, component-role schemas, or conditional/unconditional lifecycle rules.
- **No proof.** No output, timing, target-backend, cancellation, reuse, or teardown
  evidence exists.
- **No external workflow is an Engine fallback.** A hosted API or pinned workflow may
  supply source or comparison evidence, but neither establishes native Engine support.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Official NF4 Diffusers pipeline | **Reference baseline** | Only first-party public Diffusers baseline; explicitly not a dense source of truth |
| Complete scaled-FP8 topology | **Experimental** | Exact official artifacts and layout exist, but closure is large and Engine has no loader/lifecycle proof |
| Complete NVFP4 topology | **Experimental challenger** | Best Blackwell memory candidate; still exceeds the raw 16 GB envelope before VAE/buffers |
| Complete INT8 ConvRot topology | **Experimental** | Pinned four-file official workflow topology; first bounded Engine-native candidate, pending schema/lifecycle/backend proof |
| Hosted Ideogram API | **Fallback** | Preserves current service quality and magic prompting without local residency burden |
| Dense BF16/FP16 reference | **Unverified / unavailable** | No official public artifact; do not fabricate one |
| Generic GGUF/Nunchaku/AWQ/W4 variants | **Rejected** | No first-party need or accepted end-to-end path |
| Recommended native path | **None** | No Engine or target-workstation evidence exists |

## Small qualification ladder

1. **Public reference baseline:** exact official NF4 Diffusers pipeline at
   `V4_QUALITY_48`, using a precomputed structured JSON caption and fixed seed.
2. **Incumbent candidate:** exact complete four-file INT8 ConvRot topology with
   the same JSON caption, dimensions, and an explicitly pinned sampler preset.
3. **Deferred alternate:** the standard FP8 graph only after its Qwen/Gemma prompt mode
   and complete closure are frozen.
4. **Deferred challenger:** exact complete NVFP4 topology on the native Blackwell
   backend, after an official graph or explicitly labeled derived contract is available.

The NF4 baseline cannot answer absolute quantization loss because no public dense
teacher exists. It can answer product questions: output quality, text/layout success,
latency, stability, and whether Comfy's alternative topology is materially better.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus a fixed **JSON** corpus covering:

- short and long multilingual typography, logos, posters, labels, menus, and signage;
- exact spelling, line breaks, hierarchy, font category, foreground/background
  contrast, and text placement;
- bounding-box layout with overlapping and non-overlapping elements;
- palette conditioning with exact hex values;
- object count, spatial relations, long-aspect-ratio banners, portrait, landscape,
  1024², and 2048² outputs;
- photography, illustration, graphic design, product mockups, and flat vector styles.

Store the exact JSON sent to the model. When starting from plain text, store the
original text, expansion provider/model/version, expanded JSON, and whether expansion
was hosted or local. Compare prompt expansion separately from image inference.

Measure conditional and unconditional branch loading/residency, Qwen encoding,
denoising, VAE decode, branch swaps, peak VRAM/RAM, disk/offload traffic, and backend
dispatch. Cancel during prompt expansion, encoder load, each branch load, denoising,
decode, and output write; the next request must not reuse stale JSON or a partial
branch.

## Hard gaps and source conflicts

1. **No dense source of truth:** public NF4 and FP8 cannot establish loss relative to an
   unpublished teacher. This limitation must remain visible in every result.
2. **Dual-branch closure:** one diffusion file is not a complete Ideogram 4 path. Both
conditional and unconditional models are required for the pinned official workflow topology.
3. **16 GB residency:** even the NVFP4 component sum exceeds 16 GB before VAE and
   temporary buffers. Stage order and transfer overhead are unproved.
4. **Prompting is part of the model operation:** hosted magic prompt, local Qwen prompt
   enhancement, and user-authored JSON can produce different outputs.
5. **License and gate:** all public weights are gated under the Ideogram 4
   Non-Commercial Model Agreement; product/distribution scope needs explicit review.
6. **Prompt-mode closure:** the current standard FP8 template distributes an additional
   Gemma prompt-assistant model beside Qwen. Its active artifact closure depends on
   prompt mode; do not label a reduced Qwen-only FP8 subset as official standard parity.
7. **“FP8 supports all hardware” is not a performance proof:** the target result must
   report the actual stored-layout and kernel/backend dispatch.

## Ordered next actions

1. Review the Ideogram non-commercial license and decide whether native local weights
   belong in the Engine product surface at all.
2. Define an Engine JSON prompt schema, including bounding boxes, palette, aspect ratio,
   safety result, and expansion provenance.
3. Build header-only manifests for both pinned INT8 branches, encoder, and VAE; reject
   a missing branch, mixed closure, hidden dense fallback, or runtime conversion.
4. Implement the exact four-file INT8 graph with typed conditional/unconditional roles,
   staged residency, and positive intended INT8/FP8 dispatch proof.
5. Run the typography/layout corpus on the RTX 5080, including branch-order,
   cancellation, switching, and teardown acceptance.
6. Prototype the official NF4 baseline separately with matching structured JSON.
7. Add FP8 or NVFP4 only after their complete prompt mode/closure is pinned; record
   Gemma inclusion or an explicitly labeled deliberate deviation.

## Explicit non-goals

- Do not call NF4 or FP8 a dense reference.
- Do not load only the conditional model and describe it as the official topology.
- Do not hide hosted magic prompting inside an otherwise local recipe.
- Do not convert weights at Engine runtime.
- Do not add all Comfy/Kitchen formats because their kernels exist.
- Do not recommend native Ideogram until license, structured prompting, output quality,
  and target-workstation lifecycle are accepted.

## Primary sources

- Official repository snapshot:
  <https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2>
- Official NF4 model card:
  <https://huggingface.co/ideogram-ai/ideogram-4-nf4>
- Official NF4 Diffusers repository:
  <https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers>
- Official FP8 model card:
  <https://huggingface.co/ideogram-ai/ideogram-4-fp8>
- Official Comfy repackaged artifacts:
  <https://huggingface.co/Comfy-Org/Ideogram-4>
- Official sampler reference:
  <https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/inference.md>
