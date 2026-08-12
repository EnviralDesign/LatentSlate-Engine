# Qwen Image Edit 2511 roadmap

Last reviewed: **2026-08-11**  
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Qwen Image Edit 2511 is an **editing family, not a general T2I qualification
line**. Keep three execution lines separate:

1. **Standard BF16 edit** — the first-party 40-step reference.
2. **Lightning edit** — the same 2511 lineage with the official LightX2V four-step
   distilled LoRA.
3. **Stored FP8 Lightning** — a fused scaled-FP8 transformer that already includes
   the four-step distillation; it is not a precision-only substitute for the
   40-step BF16 reference.

There is no native Engine family today, and the official Comfy closure is far larger
than 16 GB: the BF16 diffusion model alone is 40.9 GB, while its standard Comfy text
encoder is 9.38 GB and the VAE is 254 MB. The product problem is therefore not
“which quantization enum should Engine expose?” It is designing a bounded multi-image
editing runtime with explicit encoder/transformer/VAE residency and then proving that
the four-step path preserves the edit behavior creators care about.

This roadmap has **no Recommended path**. The smallest useful ladder is standard BF16,
BF16 plus the four-step LoRA, and one exact stored scaled-FP8 Lightning challenger.
Do not add a generic Qwen format zoo before that comparison is complete.

## Evidence labels

- **Verified** — stated by Qwen, LightX2V, Comfy, or the Engine source at the audited
  commit.
- **Publisher measurement** — a performance or memory claim from an artifact publisher,
  not an Engine result.
- **Inference** — a roadmap judgment that requires target-workstation validation.

## Scope and operation boundaries

| Line | Operations | Canonical settings | Comparison boundary |
| --- | --- | --- | --- |
| Standard 2511 | Single-image and multi-image instruction editing | Qwen example: `true_cfg_scale=4.0`, negative prompt `" "`, 40 steps, guidance 1.0, seed 0 | High-precision source of truth for the same input set and edit request |
| Lightning LoRA | Same edit operations with step distillation | Four-step BF16 LoRA on the standard 2511 model | Compare quality and lifecycle against standard BF16; this is a different schedule, not a quant-only test |
| Fused scaled-FP8 Lightning | Same four-step distilled operation | FP8 E4M3 scaled transformer fused with the Lightning weights | Compare against BF16 + the same four-step LoRA, not standard 40-step BF16 alone |
| Comfy `fp8mixed` standard model | Standard 2511 topology with mixed stored precision | Workflow-specific standard edit path | Compare against standard BF16 with identical 40-step settings |

The official [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
uses `QwenImageEditPlusPipeline` and accepts a list of images. It demonstrates two
input images, 40 steps, true CFG 4.0, guidance 1.0, and a blank-space negative prompt.
Qwen describes improved character consistency, multi-person consistency, geometric
editing, design-material changes, and reduced unintended drift. Those capabilities
need separate acceptance cases; a single cat-to-dog example is not enough.

## Official and credible artifacts

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Complete first-party Diffusers BF16 model | Apache-2.0; canonical 40-step single/multi-image pipeline | **Reference** |
| [`qwen_image_edit_2511_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/blob/b6a0794717d3f5600f85c5edcdcd0c0eb93d7446/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors) | Official Comfy BF16 diffusion model | **40.9 GB**; SHA-256 `ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c` | **Reference component** |
| [`qwen_2.5_vl_7b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/bbdcab645099df455488b29f48957efbd91f996b/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | Official Comfy Qwen2.5-VL 7B text/vision encoder | **9.38 GB**; SHA-256 `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4` | **Shared component** |
| [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/qwen_image_vae.safetensors) | Official Comfy VAE | **253,806,246 bytes**; SHA-256 `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | **Shared component**; pin an immutable repository revision before implementation |
| [`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | Official LightX2V four-step distilled LoRA | Apache-2.0; separate LoRA applied to the BF16 model | **Experimental incumbent candidate** |
| [`qwen_image_edit_2511_fp8mixed.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/blob/44ae38a64b07261cdd61cca062c3c97ac73e839f/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors) | Comfy mixed-precision FP8-compatible standard transformer | **20.5 GB**; SHA-256 `c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e` | **Deferred challenger** until the standard BF16 runtime is bounded |
| [`qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/blob/b89b55b986fd724a8dd6db59cb4b9005d338941a/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors) | Fused scaled-FP8 Comfy transformer with Lightning | **20.4 GB**; SHA-256 `16d03fd9bab36b0a588e77db23825629df62ab99fc3e62f3a36887b201dcc36f` | **Experimental challenger** for the four-step line |
| Community GGUF, INT8, ConvRot, Nunchaku, AWQ, and other W4 paths | Various | File or kernel availability is not an accepted Qwen 2511 editing product path | **Rejected** from the first ladder |

The current official Comfy tutorial uses the BF16 diffusion model, the scaled-FP8
Qwen2.5-VL encoder, the Qwen image VAE, and optionally the four-step Lightning LoRA:
<https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511>.
The tutorial and repository default branches are mutable; an implementation change
must pin the exact workflow/template and every component revision.

## Current Engine truth at `2ba5709`

- **Not implemented.** Qwen Image Edit is absent from the model-family registry and
  tool catalog:
  [`model_store.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/model_store.py),
  [`tools/__init__.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/__init__.py).
- **No recipe or acquisition contract.** There is no Qwen 2511 built-in recipe,
  resource declaration, bundle, profile, loader schema, or artifact identity check.
- **No lifecycle proof.** Engine has no multi-image upload mapping, prompt/image
  encoder cache contract, offload plan, cancellation proof, output acceptance, or
  target-workstation timing for this family.
- **Generic Comfy remains the fallback.** LatentSlate can call a user-owned Comfy
  workflow today, but that does not establish a native Engine recommendation.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| First-party 40-step BF16 | **Reference** | Canonical highest-precision standard-edit source |
| Native standard BF16 Engine path | **Experimental** | Required to establish edit fidelity and lifecycle, but unlikely to fit without aggressive staging/offload |
| BF16 + four-step Lightning LoRA | **Experimental** | Best first creator-value challenger because it attacks 40-step latency without introducing a new base layout |
| Fused scaled-FP8 Lightning | **Experimental** | Exact published four-step artifact; compare only against BF16 + the same LoRA |
| Comfy `fp8mixed` standard model | **Deferred** | Valid standard-line quant candidate, but a fourth loader is premature before the two core lineages are accepted |
| Full BF16 Qwen2.5-VL encoder | **Deferred** | The official Comfy path deliberately uses a scaled-FP8 encoder; qualify that topology first |
| Community low-bit format zoo | **Rejected** | No creator benefit justifies multiple loaders before one native edit path exists |
| User-owned Comfy workflow | **Fallback** | Existing route for immediate experimentation and custom graphs |
| Recommended native path | **None** | No Engine measurement or accepted outputs exist |

## Small qualification ladder

### Standard edit

1. **Reference:** first-party BF16 2511, 40 steps, Qwen's exact CFG/guidance settings,
   and one fixed single- and multi-image corpus.
2. **Incumbent candidate:** the same BF16 transformer and components through one
   bounded Engine runtime with explicit staged residency.
3. **Optional standard challenger:** Comfy `fp8mixed`, but only if standard BF16 can
   be executed reliably enough to provide a truthful comparison.

### Four-step edit

1. **Reference for the distilled line:** BF16 transformer plus the exact four-step
   Lightning LoRA.
2. **Challenger:** fused scaled-FP8 Lightning artifact with the same encoder, VAE,
   prompts, image order, dimensions, and four-step schedule.

Do not collapse the two ladders. A four-step result can be a better product even while
being less similar to the 40-step teacher; that is a creator-value decision, not a
precision benchmark.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus a fixed edit corpus covering:

- no-op and surgical edits, where untouched regions should remain stable;
- identity-preserving single-person edits and two-person/multi-person consistency;
- two- and three-reference composition with input order recorded;
- exact text replacement, multilingual text, font/style preservation, and signage;
- object insertion/removal, relighting, material replacement, style transfer, camera
  or geometry edits, and industrial-design changes;
- repeated edits from the same source to expose encoder caching and cumulative drift.

Record image preprocessing, resize/crop behavior, encoder execution, prompt/image cache
hits, transformer stage time, VAE encode/decode, output serialization, peak VRAM/RAM,
and host-to-device traffic. Cancel during source encoding, text encoding, denoising,
VAE decode, and writeout; the next job must succeed without stale references or a
poisoned cache.

A production Lightning path requires a clear end-to-end win and creator acceptance on
identity, text, untouched-region stability, and multi-image composition. Publisher
claims such as “approximately 10× faster” are motivation only until reproduced under
Engine's full pipeline and target hardware.

## Hard gaps and source conflicts

1. **16 GB feasibility gap:** BF16 transformer plus encoder cannot be resident. The
   exact staging order, host RAM requirement, transfer cost, and cache lifetime are
   unproved.
2. **Two FP8 meanings:** Comfy `fp8mixed` is a standard 40-step model; the LightX2V
   scaled-FP8 file is fused with Lightning. They are not interchangeable artifacts.
3. **Reference versus product schedule:** standard BF16 is the quality source, while
   the likely product path is four-step. Both must be retained in qualification.
4. **Mutable workflow gap:** Comfy documentation points to current templates; pin the
   workflow JSON/template package and component commits before declaring a recipe.
5. **Input-order semantics:** multi-image behavior depends on ordered inputs and prompt
   references. Engine needs a stable schema, provenance, and repeatable upload mapping.
6. **No Engine proof:** no accepted output, cancellation, warm reuse, or teardown record
   exists for any Qwen 2511 path.

## Ordered next actions

1. Pin the official Qwen Diffusers revision and the exact Comfy component/workflow
   revisions used as the reference topology.
2. Specify one immutable multi-image Engine request schema, including ordered image
   roles, resize/crop policy, prompt, negative prompt, true CFG, guidance, steps, and
   seed.
3. Build header-only manifests for BF16, `fp8mixed`, the Lightning LoRA, the fused
   scaled-FP8 Lightning file, Qwen2.5-VL encoder, and VAE; reject ambiguous layouts.
4. Prototype the standard BF16 path with explicit CPU staging and measure whether its
   cold/warm lifecycle is usable on 16 GB without runtime weight conversion.
5. Add the four-step BF16 LoRA path and run the fixed edit corpus.
6. Add only the fused scaled-FP8 Lightning challenger and verify the actual quantized
   backend dispatch.
7. Promote a path only after target-workstation creator review, cancellation, reuse,
   teardown, and material-win gates pass.

## Explicit non-goals

- Do not treat Qwen Image Edit 2511 as a T2I family in this roadmap.
- Do not compare four-step Lightning output as though it were a pure FP8 version of
  the 40-step teacher.
- Do not quantize or fuse weights at Engine runtime.
- Do not add generic GGUF, INT8, AWQ, Nunchaku, or ConvRot loaders before the exact
  first-party/Comfy ladder is accepted.
- Do not expose arbitrary image-list semantics without ordered roles and provenance.
- Do not claim the publisher's speed or memory estimates as Engine measurements.

## Primary sources

- Official Qwen 2511 model card and canonical Diffusers example:
  <https://huggingface.co/Qwen/Qwen-Image-Edit-2511>
- Official Comfy Qwen 2511 tutorial:
  <https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511>
- Official Comfy component repository:
  <https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI>
- Shared Comfy Qwen encoder/VAE repository:
  <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI>
- LightX2V Lightning model repository:
  <https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning>
- Engine audited model registry and tool catalog:
  <https://github.com/EnviralDesign/LatentSlate-Engine/tree/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine>
