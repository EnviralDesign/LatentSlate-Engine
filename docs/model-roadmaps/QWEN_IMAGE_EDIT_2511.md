# Qwen Image Edit 2511 roadmap

Last reviewed: **2026-08-13**
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

This roadmap has **no Recommended path**. The publisher BF16 graph remains the
40-step Reference, while the exact current Comfy INT8 ConvRot graph below is the
smallest bounded native candidate. Its optional four-step Lightning mode is a separate
four-file operation, not a precision-only substitute for the 40-step Reference. Do
not add a generic Qwen format zoo before those comparisons are complete.

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
| Community GGUF, arbitrary INT8/ConvRot, Nunchaku, AWQ, and other W4 paths | Various | File or kernel availability is not an accepted Qwen 2511 editing product path | **Rejected** from the first ladder; this does not reject the pinned official Comfy INT8 graph below |

The current official Comfy tutorial uses the BF16 diffusion model, the scaled-FP8
Qwen2.5-VL encoder, the Qwen image VAE, and optionally the four-step Lightning LoRA:
<https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511>.
The tutorial and repository default branches are mutable; an implementation change
must pin the exact workflow/template and every component revision.

### Verified current Comfy INT8 closure

The pinned [INT8 template](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json)
provides the executable native reference. Its saved `Enable 4steps LoRA?` switch is
off: the saved-default is the three-file, **40-step / CFG 4** mode. Enabling the
Lightning branch requires all three coordinated changes—attach the fixed LoRA, use
**4 steps**, and use **CFG 1**—and creates a separate four-file mode.

| Role | Immutable source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Transformer | [`Comfy-Org/Qwen-Image-Edit_ComfyUI`, `split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors`, `e9e85de74a8f48c1e3e2656617626348675a2f21`](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/blob/e9e85de74a8f48c1e3e2656617626348675a2f21/split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors) | 20,499,083,824 | `11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829` |
| Text/vision encoder | [`Comfy-Org/HunyuanVideo_1.5_repackaged`, `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `f10daa9e51f1b192302ef701ffc918af0652830e`](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/blob/f10daa9e51f1b192302ef701ffc918af0652830e/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | 9,384,670,680 | `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4` |
| VAE | [`Comfy-Org/Qwen-Image_ComfyUI`, `split_files/vae/qwen_image_vae.safetensors`, `dfe60a0d63f0b946628080f070978594983b8b6e`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/dfe60a0d63f0b946628080f070978594983b8b6e/split_files/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| Fixed four-step LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`, `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, `fd3a43ffb5bc98c7d09b2238e5b09a63284a16f8`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/blob/fd3a43ffb5bc98c7d09b2238e5b09a63284a16f8/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors) | 849,608,296 | `22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f` |

The saved-default total is **30,137,560,750 bytes**
(`20,499,083,824 + 9,384,670,680 + 253,806,246`). Lightning adds
`849,608,296` bytes for a four-file total of **30,987,169,046 bytes**. The sizes and
LFS SHA-256 values were rechecked against immutable Hugging Face metadata on
2026-08-13; no model payload was downloaded.

The template activates one top-level image, while the official subgraph exposes
ordered `image1`, `image2`, and `image3` sockets. The first Engine schema should match
that real capability: one required image plus up to two optional ordered images, with
explicit preprocessing, roles, and hashes. One-, two-, and three-image quality still
require separate acceptance; input support itself is not an Engine-only extension.

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
| Exact Comfy INT8 standard edit | **Experimental** | Pinned three-file 40-step/CFG-4 graph; requires full lifecycle and dispatch proof |
| Exact Comfy INT8 Lightning edit | **Experimental** | Pinned four-file 4-step/CFG-1 graph; fixed LoRA and coordinated switches |
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
2. **Incumbent candidate:** the exact pinned three-file INT8 Comfy standard graph,
   with 40 steps, CFG 4, explicit staged residency, and no runtime conversion.
3. **Optional standard challenger:** Comfy `fp8mixed`, but only if the reference and
   INT8 lifecycles can be executed reliably enough to provide a truthful comparison.

### Four-step edit

1. **Reference for the distilled line:** BF16 transformer plus the exact four-step
   Lightning LoRA.
2. **Incumbent candidate:** the pinned INT8 transformer plus the exact same fixed
   LoRA, at 4 steps/CFG 1.
3. **Challenger:** fused scaled-FP8 Lightning artifact with the same encoder, VAE,
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
2. **Three stored-layout meanings:** the pinned INT8 graph is an independent standard
   40-step line; Comfy `fp8mixed` is a different standard model; the LightX2V
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

1. Specify one immutable multi-image Engine request schema: one required plus up to
   two optional ordered images, resize/crop policy, roles, hashes, prompt, negative
   prompt, steps, CFG, and seed. Qualify one-, two-, and three-image behavior separately.
2. Build header-only manifests for the pinned INT8 transformer, encoder, VAE, and
   Lightning LoRA; reject incomplete closures, hidden dense copies, or an invalid
   standard/Lightning switch combination.
3. Implement the three-file saved-default graph with stage ownership, image-order
   provenance, cancellation, and positive native INT8/FP8 dispatch proof.
4. Add the fixed four-file Lightning mode and atomically bind its LoRA, four steps,
   and CFG 1. Compare it with BF16 plus the same LoRA when that reference is viable.
5. Add the fused scaled-FP8 challenger only after these two modes are accepted.
6. Promote nothing until target-workstation creator review, cancellation, reuse,
   teardown, and material-win gates pass.

## Explicit non-goals

- Do not treat Qwen Image Edit 2511 as a T2I family in this roadmap.
- Do not compare four-step Lightning output as though it were a pure FP8 version of
  the 40-step teacher.
- Do not quantize or fuse weights at Engine runtime.
- Do not add generic GGUF, arbitrary INT8/ConvRot, AWQ, or Nunchaku loaders before the
  exact first-party/Comfy ladder is accepted.
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
