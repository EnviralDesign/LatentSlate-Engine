# Wan 2.2 TI2V 5B roadmap

Last reviewed: **2026-08-13**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Wan 2.2 TI2V 5B remains an existing Engine family with two distinct operations:
text-to-video and required-first-image video. Do not add another lineage or
speculative variant.

The complete BF16 Diffusers T2V recipe is the cataloged **Reference** contract.
Its 34.2 GB closure is not a local optimization target and should be exercised in
a batched high-memory reference campaign rather than repeatedly forced onto the
16 GB workstation.

The official split artifacts remain valuable optimized-source candidates:

- 10.0 GB FP16 TI2V transformer;
- 6.74 GB scaled-FP8 UMT5 encoder;
- 1.41 GB Wan 2.2 VAE.

They are catalog resources, not a runnable optimized recipe, until Engine loads
and executes them directly. The prior ComfyUI-backed recipes and profile were
removed. ComfyUI workflows and node source remain reference evidence only;
Comfy Kitchen is the allowed direct quantized-layout/dispatch dependency.

## Authority map

1. First-party Wan/Diffusers sources own architecture, tensor names, and dense
   reference behavior.
2. The pinned official Comfy workflow owns practical topology and fixed defaults:
   preprocessing, component wiring, 30 steps, CFG 5, `uni_pc` / `simple`, SD3
   shift 8, denoise 1, and 24 fps.
3. Pinned ComfyUI node source explains how the workflow realizes those semantics.
4. Pinned Comfy Kitchen APIs own supported quantized tensor restoration and
   dispatch facts.
5. Engine owns model materialization, conditioning, sampling, lifecycle,
   cancellation, provenance, storage, and acceptance.

See [the hard Engine policy](../COMFY_ENGINE_POLICY.md). Engine must not embed,
launch, proxy, or require ComfyUI.

## Current Engine truth

| Path | Proof | Disposition |
| --- | --- | --- |
| Dense BF16 T2V | Cataloged structural reference; local output acceptance incomplete | **Reference** |
| Split transformer + scaled-FP8 UMT5 + VAE T2V | Exact resources retained; no current runnable Engine recipe | **Experimental source contract** |
| Same split closure with required first image | Exact resources retained; direct Engine conditioning/runtime not yet restored | **Experimental source contract** |
| TI2V-5B LoRAs | Exact adapter resources retained; no runnable optimized base path | **Deferred with base runtime** |

No previous output produced through a ComfyUI backend counts as Engine-native
acceptance. It may inform parity investigation, but cannot establish current
runtime support or tier promotion.

## Existing-path overhaul

Use Klein as the golden implementation pattern:

1. Build the exact Wan TI2V model/text/VAE shells inside a disposable Engine
   worker.
2. Validate every stored artifact header and source-to-target mapping.
3. Restore the scaled-FP8 text-encoder tensors with direct Comfy Kitchen
   primitives and prove observed dispatch; do not create a dense duplicate.
4. Reproduce T2V and required-image conditioning from the pinned workflow and
   node source as typed Engine code.
5. Bind the exact 30-step schedule and 24 fps semantics in recipe identity.
6. Retain operation, component, conditioning, schedule, and actual backend facts
   in public provenance.
7. Prove disposable success, cancellation during heavy work, tree-empty cleanup,
   and fresh recovery before exposing optimized recipes again.

## Acceptance gate

For each existing operation, require:

- fixed-seed public-API output on the RTX 5080;
- valid media dimensions/frame rate/duration;
- observed Kitchen/native dispatch with no silent dense fallback;
- bounded parent memory and no dense duplicate residency;
- cancellation during loading and denoising, followed by clean recovery;
- creator review against the operation-matched official workflow output;
- explicit proof that no ComfyUI process, server, checkout, or HTTP graph runner is
  used.

Remain **Experimental** until the user reviews the Engine-native outputs. Do not
promote to Recommended automatically.

## Non-goals

- No ComfyUI execution backend, subprocess, loopback server, or workflow submission.
- No new Wan lineage or speculative format variant.
- No repeated local dense-BF16 offload tuning.
- No LoRA promotion before the direct optimized base runtime is accepted.
- No tier inheritance from historical non-native output evidence.

## Primary sources

- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Comfy-Org workflow templates](https://github.com/Comfy-Org/workflow_templates)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — node-source reference only
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) — direct Engine tensor/kernel dependency
