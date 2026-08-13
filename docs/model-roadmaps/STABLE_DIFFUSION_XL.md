# Stable Diffusion XL roadmap

Last reviewed: **2026-08-13**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Stable Diffusion XL remains an important compatibility ecosystem, but it should not
be treated as an automatic first-class Engine priority in 2026. Its strongest product
argument is not frontier base-model quality; it is the enormous installed ecosystem
of LoRAs, ControlNets, adapters, fine-tunes, and familiar Comfy workflows.

The initial local path, if LatentSlate chooses to support SDXL, should be the official
**FP16 Base 1.0** checkpoint. Base+Refiner is a separate two-stage operation and must
prove a creator-visible improvement before it becomes a default. The model fits the
16 GB class in normal FP16 operation, so low-bit format work has little value. Defer
or reject quantization loaders unless a concrete plugin/workflow cannot otherwise run.

There is currently no SDXL Engine family, tool, recipe, resource declaration, or
proof. Therefore this roadmap has **no Recommended path**. The gating question is
whether compatibility with the SDXL creator ecosystem produces enough value to
outweigh adding an older runtime family.

## Evidence labels

- **Verified** — stated by an official model card, repository, license, or the Engine source at the audited commit.
- **Publisher measurement** — an upstream speed, memory, or quality claim; not an Engine result.
- **Inference** — a roadmap product judgment requiring target-workstation validation.

## Scope and operation boundaries

| Operation | Components | Verified reference behavior | Comparison boundary |
| --- | --- | --- | --- |
| Base-only T2I | SDXL Base UNet, VAE, OpenCLIP ViT/G, CLIP ViT/L | Official Diffusers example loads the `fp16` variant; Base can run standalone | Primary first milestone |
| Base + Refiner T2I | Base high-noise generation followed by Refiner low-noise img2img | Official example: 40 total steps with an 80/20 Base/Refiner denoise split | Compare only against Base-only using the same prompt, seed, scheduler, and total steps |
| Img2img / inpaint | SDXL-compatible operation-specific pipelines or fine-tunes | Broad ecosystem support, but not one canonical artifact/settings contract | Separate future acceptance corpus |
| Turbo / Lightning / custom checkpoints | Distilled or fine-tuned descendants | Different model lineages and schedules | Never use as a precision challenger to Base 1.0 |

Verified source: the official
[SDXL Base model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
and [Refiner model card](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0).

## Published artifacts and topology

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`sd_xl_base_1.0.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors) | Official all-in-one Base checkpoint | **6.94 GB**; SHA-256 `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b` | **Reference** |
| Official Diffusers `fp16` Base repository variant | Base components stored separately | First-party and supported by `StableDiffusionXLPipeline` | **Reference** and preferred implementation source if component contracts are clearer |
| [`sd_xl_refiner_1.0.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors) | Official Refiner checkpoint | **6.08 GB**; SHA-256 `7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f` | **Experimental**, optional second stage |
| Official Comfy SDXL examples | Base-only and Base+Refiner workflow images with embedded workflow metadata | Current example directory: <https://github.com/comfyanonymous/ComfyUI_examples/tree/master/sdxl> | Implementation reference; pin blob revisions before use |
| Community fine-tunes / LoRAs / ControlNets | Extensive ecosystem | Creator value is real, but each has its own lineage/license/inputs | **Deferred** until Base recipe exists |
| FP8 / INT8 / GGUF / TensorRT variants | Community or runtime-specific | Technically real; Base already fits target VRAM | **Rejected** for the initial roadmap |

The official Base model uses two fixed text encoders—OpenCLIP ViT/G and CLIP
ViT/L—and is licensed under CreativeML Open RAIL++-M. License and model-card
restrictions must remain attached to any built-in acquisition manifest.

### Pinned official Comfy workflow evidence

The official Comfy examples are PNGs containing embedded workflow metadata rather
than standalone JSON. The immutable source is
[`comfyanonymous/ComfyUI_examples@f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl):

| Operation | Exact PNG blob | Bytes | Required implementation action |
| --- | --- | ---: | --- |
| Base-only | [`sdxl_simple_example.png`, `285be9497da05ed151c9a505857c17485557c79b`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_simple_example.png) | 1,257,779 | Extract and check in the embedded graph before copying any settings |
| Base + Refiner | [`sdxl_refiner_prompt_example.png`, `8010ade35871a966cb616c3700256e341617813b`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_refiner_prompt_example.png) | 1,249,954 | Extract separately; preserve its recorded total steps and Base/Refiner handoff |

The commit and blob identities above were verified against GitHub on 2026-08-13. They
pin the workflow evidence only—not an SDXL component closure. A built-in Diffusers
path still needs an immutable Base/Refiner revision plus the complete allowlisted
configuration, tokenizers, dual text encoders, VAE, and license metadata. A single
checkpoint SHA is not enough for a componentized recipe.

## Current Engine truth at `2ba5709`

- **Not implemented.** SDXL is absent from the model-family registry and tool catalog:
  [`model_store.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/model_store.py),
  [`tools/__init__.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/__init__.py).
- **No catalog/acquisition.** Engine has no SDXL recipe, resource declaration, bundle,
  or deployment profile.
- **No proof.** There are no Engine schema tests, end-to-end outputs, target-hardware
  measurements, cancellation results, or lifecycle tests.
- **Comfy provider support is not native Engine support.** LatentSlate can already call
  user-supplied Comfy workflows, which weakens the case for duplicating the entire
  SDXL ecosystem in Engine unless an opinionated native path adds clear value.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Official FP16 Base 1.0 | **Reference** | Canonical first-party source and sufficient precision for the 16 GB workstation |
| Native Base-only Engine recipe | **Experimental** | Smallest truthful implementation if compatibility demand is established |
| Official Base + Refiner | **Experimental** | Separate two-stage quality candidate; not automatically better enough to justify double residency/load cost |
| SDXL community fine-tune/LoRA support | **Deferred** | High ecosystem value, but requires a stable Base component/adapter contract first |
| Img2img/inpaint native tools | **Deferred** | Separate operations; implement only after T2I lifecycle is stable |
| FP8/INT8/GGUF/NVFP4 | **Rejected** | Insufficient creator value for a model that already fits in FP16 |
| User-owned Comfy SDXL workflow | **Fallback** | Existing generic provider path; appropriate for arbitrary ecosystem graphs |
| Recommended native path | **None** | No Engine measurement or product-demand decision exists |

## Small qualification ladder

1. **Reference:** official FP16 Base-only, fixed prompt/seed, 1024², exact scheduler
   and step count recorded from the selected official workflow.
2. **Incumbent candidate:** the same Base artifact through one native Engine recipe.
3. **Optional challenger:** official Refiner with the canonical 40-step 80/20 split.

Stop if Base-only does not add meaningful product value beyond the existing Comfy
provider. Do not add a low-bit challenger.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- T2I corpus at 1024² and two non-square SDXL-native aspect ratios;
- cases for hands/faces, object count, spatial composition, illustration, photography,
  and the model's known weak area of legible text;
- Base-only versus Base+Refiner blind review, with denoise split and total steps held
  constant;
- one common LoRA and one ControlNet only after plain Base is accepted, to test whether
  Engine's native abstraction can preserve the ecosystem value that motivates support.

Record load time for one versus two UNets, peak VRAM/RAM, Base-to-Refiner transfer,
VAE reuse, prompt-embedding reuse, cancellation during each stage, and clean teardown.
The Refiner needs a material quality preference or workflow-enabling benefit; a small
pixel-detail change does not justify doubling the loader/lifecycle surface.

## Hard gaps and source conflicts

1. **Product-priority gap:** SDXL compatibility may be valuable, but generic Comfy
   workflows already cover it. Native Engine support needs a clearer creator use case.
2. **Workflow extraction required:** the exact Comfy source commit and PNG blob IDs are
   pinned above, but Engine must still extract, save, and structurally test each
   embedded graph rather than recreate settings from memory.
3. **Base versus Refiner is not a format comparison:** the Refiner adds a second model
   and changes the denoising topology.
4. **Ecosystem scope is enormous:** “support SDXL” cannot mean every custom checkpoint,
   ControlNet, adapter, LoRA, and extension in the first release.
5. **License propagation:** Open RAIL++-M terms and derivative licenses must remain
   visible through acquisition and provider metadata.

## Ordered next actions

1. Validate demand: identify the specific SDXL workflow or ecosystem assets that cannot
   be served adequately by LatentSlate's existing Comfy provider.
2. Pin the official Base repository revision and complete component allowlist; extract
   the Base-only graph from the exact PNG blob above and check it into Engine tests.
3. Build a header-only manifest for Base: two text encoders/tokenizers, VAE, UNet,
   scheduler/config identity, FP16 variant, and license metadata. Reject arbitrary
   checkpoints, runtime quantization, and a hidden Refiner in the Base key.
4. Implement Base-only T2I with one output, extracted scheduler/steps, explicit
   encoder/VAE ownership, cancellation, and teardown. Record prompt-cache state and
   phase timing.
5. Run the creator corpus and compare native Engine against the same artifact in Comfy.
6. Add Refiner only if blind review shows a material quality benefit worth its load and
   memory cost; use its separately extracted graph and a separate runtime fingerprint.
7. Add one narrowly selected LoRA/ControlNet compatibility check only after the Base
   contract is stable.

## Explicit non-goals

- Do not turn Engine into a general SDXL checkpoint manager in the first milestone.
- Do not add quantized formats merely because they exist.
- Do not mix SDXL Turbo, Lightning, custom fine-tunes, Base, and Refiner results.
- Do not claim a mutable Comfy example is immutable; pin its blob at implementation.
- Do not implement native SDXL if the existing Comfy provider already satisfies the
  creator workflow with lower maintenance cost.

## Primary sources

- Official Base model card and Diffusers examples:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
- Official Base single file:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors>
- Official Refiner model card:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0>
- Official Refiner single file:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors>
- Official Comfy examples directory (mutable until pinned):
  <https://github.com/comfyanonymous/ComfyUI_examples/tree/master/sdxl>
- SDXL technical report: <https://arxiv.org/abs/2307.01952>
