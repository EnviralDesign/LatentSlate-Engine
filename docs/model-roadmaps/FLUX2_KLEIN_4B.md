# FLUX.2 Klein 4B roadmap

Last reviewed: 2026-08-11

## Scope

This roadmap covers Black Forest Labs FLUX.2 Klein 4B on the primary local
qualification machine: Windows 11, NVIDIA RTX 5080 16 GB (SM 12.0), Python 3.12,
and the Engine-managed CUDA runtime tiers.

The two important upstream lineages are not interchangeable:

- **Distilled 4B** is the four-step, guidance-1 production path used for text to
  image and supported editing.
- **Base 4B** is the 20-step, guidance-5 editing path used by the current official
  Comfy image-edit workflow.

Quantization comparisons must remain within one lineage and operation. A Base FP8
edit cannot be evaluated against a Distilled BF16 edit as though quantization were
the only changed variable.

## Product position

- **Reference:** exact BFL BF16 for the matching lineage.
- **Recommended:** official BFL/Comfy FP8.
- **Next experimental challenger:** official BFL NVFP4 on a qualified cu130 Kitchen
  CUDA backend.
- **Later research candidate:** Distilled INT8 ConvRot, only if NVFP4 qualification
  leaves a meaningful gap.

The public recipe set should stay small. Kernel availability is not a reason to
publish a recipe; each edition must bind an exact artifact, loader contract,
component closure, runtime policy, and proof record.

## Current Engine truth

| Operation and lineage | Current recipe | State | Important caveat |
| --- | --- | --- | --- |
| Distilled T2I, BF16 | `flux2-klein-4b.text-to-image.native-distilled-bf16` | Reference implemented | Fresh recipe-derived install and formal benchmark corpus remain to be completed. |
| Distilled T2I, FP8 | `flux2-klein-4b.text-to-image.comfy-distilled-fp8` | Recommended implemented | Exact component loader exists; formal acceptance run remains. |
| Base I2I, FP8 | `flux2-klein-4b.image-to-image.comfy-base-fp8` | Recommended implemented | One-to-three-reference Engine path exists; formal acceptance run remains. |
| Distilled I2I, BF16 | `flux2-klein-4b.image-to-image.native-distilled-bf16` | Implemented, but not a Base reference | It must not be used as the scientific reference for Base FP8 editing. |
| Base I2I, BF16 | None | Missing reference | Highest-priority catalog gap before judging Base FP8 or Base NVFP4 quality. |

The stored-FP8 adapter accepts the exact official global E4M3 FP8 layout and
restores it without runtime conversion. Engine-owned staged residency is the
current 16 GB execution policy. The new bootstrap prefers cu130 when validated,
but a real cu130/Kitchen workstation acceptance run is still required.

## Artifact and execution matrix

| Representation | Published Klein 4B artifact | Comfy/Kitchen position | Engine state | Disposition |
| --- | --- | --- | --- | --- |
| BFL BF16 | Official Base and Distilled repositories | Ordinary Torch/Diffusers; source weights | Distilled loader/recipes exist; Base reference recipe missing | **Reference** |
| BFL FP8 | Official Base and Distilled single-file artifacts | Official Comfy path; Kitchen FP8 primitives | Exact Base and Distilled component recipes/loaders exist | **Recommended** |
| BFL NVFP4 | Official Base and Distilled single-file artifacts | Comfy understands NVFP4; native Kitchen path requires compatible Blackwell hardware and cu130 | No NVFP4 artifact contract or Klein materializer | **Experimental — next** |
| Plain tensor/row INT8 | Community Distilled artifact exists | Kitchen primitives exist; published Blackwell result shows little warm gain and large cold overhead | Unsupported | **Deferred** |
| INT8 ConvRot | Community Distilled artifact exists; no clean Base counterpart identified | Comfy/Kitchen format and kernels exist | No Klein loader or exact qualification | **Experimental — later** |
| MXFP8 | No compelling exact Klein artifact identified | Format exists; the pinned Kitchen execution path is not yet compelling enough | Unsupported | **Deferred** |
| ConvRot W4A4 | No clean, proven Klein artifact identified | Kernel/layout support is ahead of the model artifact ecosystem | Unsupported | **Deferred** |
| SVDQuant W4A4 | Community Distilled Nunchaku artifacts exist | Separate Nunchaku runtime/dependency stack, not the current Kitchen loader | Unsupported | **Deferred separate-runtime study** |
| Kitchen W4A8 | No clean published Klein artifact or end-to-end evidence identified | Computational layout exists | Unsupported | **Deferred** |
| GGUF | Community Base and Distilled families exist | Requires the ComfyUI-GGUF ecosystem; not a Kitchen path | Header inspection only; no Klein runtime | **Fallback candidate, deferred** |
| FP16 | No distinct first-party source-of-truth path worth qualifying | Would add no useful product rung beside BF16 | Unsupported by design | **Rejected** |
| AWQ W4A16 | No credible Klein product path identified | Better aligned with other architectures and pre-calibrated artifacts | Unsupported | **Rejected** |

The size trend makes NVFP4 worth testing even before a speed claim: the official
Distilled transformer is approximately 7.75 GB in BF16, 4.07 GB in FP8, and
2.46 GB in NVFP4. The Base artifacts follow the same broad progression. These are
artifact sizes, not peak workflow VRAM.

## Qualification ladder

### 1. Complete the references

1. Keep the existing Distilled BF16 T2I reference.
2. Add an exact Base BF16 I2I recipe matching Base FP8's 20-step, guidance-5
   semantics and component behavior.
3. Do not rename the existing Distilled BF16 I2I path into a Base reference.

### 2. Formalize FP8 as the incumbent

Run fresh recipe-derived installs and prove both official FP8 paths through the
public jobs API. Record exact resource identities, backend dispatch, cold/warm
timings, memory peaks, output metadata, cancellation, reuse, and teardown.

FP8 remains Recommended until a challenger produces a material measured win.

### 3. Qualify official NVFP4

Implement only the exact first-party Base and Distilled artifact contracts. Before
timing them, prove all of the following:

- selected Engine tier is `nvidia-cu130`;
- `torch.version.cuda` reports CUDA 13.x;
- the RTX 5080 is detected as SM 12.0;
- Comfy Kitchen's CUDA backend is qualified;
- the relevant NVFP4 matmul dispatches to the intended accelerated backend rather
  than eager fallback.

NVFP4 becomes Recommended only if it has no obvious creator-visible regression and
produces at least one material benefit: roughly 20–25% or better warm end-to-end
speed, a similarly meaningful cold/load improvement, or memory savings that unlock
a valuable workflow. A 3–8% timing difference alone does not justify another
production loader.

### 4. Optionally test INT8 ConvRot

Test the existing Distilled community artifact only after FP8 and NVFP4 results are
known. Require pinned provenance and an exact header/layout contract. If it does not
substantially improve the speed/quality/memory frontier on SM120, stop rather than
expanding into a generic INT/W4 matrix.

## Acceptance corpus

Use 8–12 fixed Distilled T2I prompts and 6–10 fixed editing cases. Include:

- photographic generation and editing;
- fine texture and repeated detail;
- legible text or signage;
- identity-preserving human edits;
- spatial or geometry instructions;
- one multi-reference Base edit.

For every case, hold constant the model lineage, prompt and negative prompt, source
images, seed, dimensions, scheduler, steps, guidance, text encoder, and VAE. Record:

- output artifacts and side-by-side human judgment;
- a lightweight perceptual/semantic similarity measure as supporting evidence;
- cold process-to-result time;
- warm sampling time and total job time over three to five runs;
- peak allocated and reserved VRAM;
- peak process/system RAM;
- model loading or compilation overhead;
- Torch, CUDA, driver, Kitchen, and actual dispatched backend;
- unload/reuse behavior and failure recovery.

## Ordered next actions

1. Run the first real cu130 bootstrap and Kitchen validation on the RTX 5080.
2. Finish the fresh-install and hardware acceptance pass for Distilled FP8 T2I and
   Base FP8 I2I.
3. Add the exact Base BF16 I2I reference recipe.
4. Create the reproducible Klein comparison corpus and result schema.
5. Add exact first-party NVFP4 resource contracts and a loader behind Experimental
   status.
6. Promote NVFP4 only if the acceptance threshold is met.
7. Consider Distilled INT8 ConvRot once; leave the remaining formats deferred until
   stronger artifacts or evidence appear.

## Non-goals

- No runtime weight conversion or save-quantized workflow.
- No generic user-facing quantization switch.
- No GPU-model lookup table that guesses a preferred recipe without qualification.
- No requirement to populate every format row.
- No Nunchaku or GGUF runtime solely to make the matrix look complete.
- No broad Klein 9B or other-family work in this roadmap.

## Primary references

- [Black Forest Labs FLUX.2 collection](https://huggingface.co/collections/black-forest-labs/flux2)
- [BFL FLUX.2 Klein 4B Distilled BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [BFL FLUX.2 Klein Base 4B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [BFL FLUX.2 Klein 4B Distilled FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)
- [BFL FLUX.2 Klein Base 4B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8)
- [BFL FLUX.2 Klein 4B Distilled NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4)
- [BFL FLUX.2 Klein Base 4B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-nvfp4)
- [Official ComfyUI Klein guide](https://docs.comfy.org/tutorials/flux/flux-2-klein)
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)
- [ComfyUI quantized-format dispatch](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/quant_ops.py)

