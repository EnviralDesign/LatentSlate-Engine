# Stable Diffusion XL implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

## Decision

SDXL is a compatibility ecosystem, not a frontier-model priority. Its highest value is existing LoRAs, ControlNets, adapters, fine-tunes, and creator workflows. The lowest-risk Engine stance is therefore:

- **Reference:** official FP16 Base 1.0 T2I.
- **Experimental native slice:** Base-only T2I, only if it provides product value beyond the generic Comfy provider.
- **Alternate:** Base+Refiner as a separate two-stage operation.
- **Fallback:** generic Comfy for arbitrary SDXL ecosystem graphs.
- **Rejected:** low-bit format work; Base already fits the target workstation.

## Product/operation boundary

| Operation | Native boundary | Provider boundary |
| --- | --- | --- |
| Base T2I | One exact Base 1.0 closure, fixed scheduler/steps, 1024-class output | Generic Comfy remains valid |
| Base+Refiner T2I | Separate two-stage contract; canonical 40 total steps with 80/20 handoff in the official Diffusers example | Keep arbitrary variants in Comfy |
| I2I/inpaint | Separate pipelines, masks, strengths, preprocessing, and acceptance | Prefer generic Comfy until specific demand |
| Control/reference/LoRA | Huge ecosystem with no single canonical closure | Generic Comfy by default; native only after Base seam is stable |

Official Comfy evidence remains in [ComfyUI examples at immutable commit `5ff76a4`](https://github.com/comfyanonymous/ComfyUI_examples/tree/5ff76a42c5f8fa15a8b18cde8c96cb3d2052b1ce/sdxl). The example images embed workflow JSON; an implementation agent must extract and check the metadata rather than transcribe settings from screenshots. Core loading/model-management truth is [ComfyUI `sd.py`](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/sd.py), [model detection](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/model_detection.py), and [model management](https://github.com/Comfy-Org/ComfyUI/blob/725e6ecf9f11561da664cae996e0ab27ed7bfc6c/comfy/model_management.py).

## Exact resources

| Tier | Artifact | Immutable identity | License/provenance |
| --- | --- | --- | --- |
| Reference | `sd_xl_base_1.0.safetensors` | 6.94 GB; SHA-256 `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`; production declaration must pin a concrete HF file revision instead of mutable `main` | Stability AI; CreativeML Open RAIL++-M |
| Alternate | `sd_xl_refiner_1.0.safetensors` | 6.08 GB; SHA-256 `7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f`; pin exact revision | Stability AI; separate model/stage |
| Reference alternative | official Diffusers FP16 variant | Pin one complete repository snapshot and allow-pattern closure for UNet, two text encoders/tokenizers, VAE, scheduler/config | First-party; componentized |

Base all-in-one and Diffusers component repositories are compatible representations, not assumed byte-identical or numerically interchangeable. Choose one as the recipe source and retain the other only as a controlled reference.

## Candidate recipes

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `sdxl.text-to-image.native-base-fp16` | Experimental/Reference | official Base FP16; native attention; no quantization; model offload or full CUDA; prompt cache; VAE slicing/tiling only if held fixed; exact sampler/scheduler/steps from pinned workflow |
| `sdxl.text-to-image.native-base-refiner-fp16` | Alternate | exact Base + Refiner resources; fixed 80/20 denoise split; shared prompt embeddings/VAE only where source code proves safe |

Current Engine has no SDXL family. A complete-repository recipe can reuse the current general resource contract, but operation/runtime registration, typed request validation, and a family adapter are still required. Do not invent an “all SDXL checkpoints” schema in the first slice.

## Loader/runtime packet

Likely reuse `resources.py`, `recipes.py`, `runtime/kit.py`, `runtime/cache.py`, `runtime/manager.py`, `runtime/diffusers_repository.py`, and generic dimension validation. Likely new `runtime/sdxl.py`, `tools/sdxl.py`, registry/config entries, tests, built-in declarations.

Validate model-index/config completeness; exact UNet architecture; both text encoders/tokenizers; VAE config; scheduler; FP16 variant files; no silently missing safety/watermark components. Lifecycle: text encoders → prompt embeddings → Base UNet → optional Refiner handoff → VAE decode. Pipeline key includes Base/Refiner identities, split, scheduler, dtype, attention/offload, VAE policy, and LoRAs. Cancellation in either stage must eject both-stage state if integrity is uncertain.

## Hardware/scientific acceptance packet

Fixed Base case: 1024², fixed seed/prompt, exact pinned scheduler and step count; portrait and landscape buckets. Run cold, three warm repeats, A→Base+Refiner→A, cancellation during encoder/Base/Refiner/decode, malformed closure, and explicit teardown. Provenance must report both text encoders, Base/Refiner hashes, split, cache reuse, actual attention/offload, and output hash.

Creator corpus: faces/hands, composition/counting, photography, illustration, known weak text rendering, one representative LoRA, and one ControlNet only after Base acceptance. Base+Refiner needs a blind creator-visible quality win; a subtle pixel change does not justify doubled lifecycle.

## Ordered bounded slices

1. **Decision gate — provider-versus-native value.** Identify one concrete SDXL workflow that generic Comfy does not serve well enough. Stop if none exists.
2. **Base-only FP16 T2I.** Exact Base closure, one output, no adapters. Tests: schema, load, cancellation, deterministic warm reuse. Out of scope: Refiner/I2I/inpaint/ControlNet/general checkpoints.
3. **Optional Base+Refiner.** Add only after blind review indicates value; fixed 80/20 contract.
4. **One ecosystem seam.** Add one LoRA or ControlNet path only after the Base runtime is stable; keep all other graphs generic Comfy.

## Primary sources

- [Official SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Official SDXL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- [Immutable Comfy SDXL examples](https://github.com/comfyanonymous/ComfyUI_examples/tree/5ff76a42c5f8fa15a8b18cde8c96cb3d2052b1ce/sdxl)
- [Diffusers SDXL pipeline source](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion_xl) — mutable discovery link; pin the Engine-installed revision during implementation
