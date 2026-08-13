# Stable Diffusion XL implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official Comfy example evidence:

- [ComfyUI examples `f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3)
- [SDXL example directory](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl)
- [base-only embedded-workflow PNG](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_simple_example.png)
- [Base+Refiner embedded-workflow PNG](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_refiner_prompt_example.png)

## Decision

SDXL remains valuable primarily as a compatibility ecosystem for fine-tunes, LoRAs, ControlNets, adapters, and familiar Comfy graphs. It is not a frontier-base-model priority.

A native Engine slice is justified only if a concrete creator workflow is not adequately served by the generic Comfy provider. The first slice would be official FP16 Base 1.0 T2I. Base+Refiner is a separate two-stage operation and must prove creator-visible benefit. SDXL already fits the target class in FP16, so low-bit loader work is rejected from the initial roadmap.

## Product and operation boundary

| Operation | Components/semantics | Disposition |
| --- | --- | --- |
| Base-only T2I | SDXL Base UNet, VAE, OpenCLIP ViT/G, CLIP ViT/L; one output | Smallest native value spike |
| Base+Refiner T2I | Base high-noise stage then Refiner low-noise stage; separate model lifecycle | Optional Alternate after Base acceptance |
| img2img/inpaint | operation-specific pipelines and corpora | Deferred follow-on |
| LoRA/ControlNet | ecosystem compatibility motivation, not first loader requirement | one narrow compatibility check after Base |
| Turbo/Lightning/custom checkpoints | different lineages and schedules | separate future recipes, not precision variants |
| FP8/INT8/GGUF/NVFP4 | little creator value for a model that fits FP16 | Rejected initially |

## Artifact boundary

| Artifact | Exact identity known | Disposition |
| --- | --- | --- |
| `sd_xl_base_1.0.safetensors` | 6.94 GB; SHA-256 `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b` | Reference |
| official Diffusers FP16 Base repository | complete componentized source | preferred implementation closure once immutable revision and allow-pattern are pinned |
| `sd_xl_refiner_1.0.safetensors` | 6.08 GB; SHA-256 `7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f` | optional two-stage Alternate |

Mutable discovery pages:

- [official Base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [official Refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

The exact single-file SHA is not by itself a complete Diffusers repository lock. A built-in componentized resource needs an immutable repository revision, exact support files, and license metadata. The CreativeML Open RAIL++-M terms and derivative licenses remain attached to acquisition/provenance.

## Official workflow handling

The Comfy examples are PNGs with embedded workflow metadata rather than standalone JSON. A future agent must extract and save the embedded workflow from the exact pinned blob instead of recreating settings from memory.

For Base+Refiner, preserve the official total step count and Base/Refiner denoise split represented by the extracted graph. Do not change scheduler, prompt embeddings, VAE, total steps, or split while attributing a result to the second model.

## Current Engine truth

At the audited commit, SDXL has no family, tool, runtime, recipe, resource declaration, deployment profile, output proof, or lifecycle test. Generic Comfy already serves arbitrary SDXL graphs, which raises the bar for native product value.

## Recipe ladder and candidate contract

| Candidate key | Tier | Fixed contract |
| --- | --- | --- |
| `sdxl-1-0.text-to-image.native-fp16-base` | Experimental | exact immutable Base component closure; extracted official Base-only graph; FP16; fixed scheduler/steps; explicit dual text encoders and VAE |
| `sdxl-1-0.text-to-image.native-fp16-base-refiner` | Alternate | exact Base plus Refiner closures; extracted two-stage graph; separate runtime fingerprint |
| user-owned Comfy SDXL | Fallback | arbitrary ecosystem graphs without native maintenance burden |

No quantized candidate key belongs in the first tranche.

## Loader and runtime implementation packet

Reuse complete-repository validation, `runtime/kit.py`, prompt cache, manager/residency, standard Diffusers loading, and public hardware-study patterns.

Likely new files: `runtime/sdxl.py`, `tools/sdxl.py`, typed recipe/resource declarations, and tests.

Fail-closed checks:

- exact Base repository class/components and FP16 variant;
- two text encoders and tokenizer/config files;
- VAE identity and dtype;
- Base/Refiner lineage and two-stage schedule for the Alternate;
- reject arbitrary checkpoints, missing components, runtime quantization, or hidden Refiner use in Base-only key.

Lifecycle: encode prompt once; load/run Base; release Base before Refiner when policy requires; optionally run Refiner; VAE decode; save. Cancellation during text encode, Base, transfer, Refiner, or decode ejects partial state and permits a clean next request.

## Hardware and scientific acceptance

Fixed Base case: 1024-square, seed `43301611940728`, exact extracted scheduler/steps, one output. Add two native non-square aspects and cases for faces/hands, object count, spatial composition, illustration, photography, and legible-text weakness.

Required scenarios: runtime-cold plus three changed-seed warm runs; Base to another family to Base switching; cancellation during every phase; malformed repository; explicit teardown. Record exact closure, phase timings, prompt-cache state, VRAM/RAM/Windows commit, output hashes, and creator review.

Base+Refiner admission requires a blind creator preference or a workflow-enabling benefit worth the additional model lifecycle. A small detail change is insufficient.

After Base acceptance, test exactly one common LoRA or ControlNet to prove the native abstraction preserves the ecosystem value that motivates support. Do not generalize that test into “all SDXL extensions supported.”

## Ordered bounded slices

1. **Next: product-value spike.** Name a concrete workflow/assets that generic Comfy cannot serve adequately. Stop if none exists.
2. **Exact Base closure and workflow extraction.** Pin repository, extract workflow from the exact PNG, define one T2I key, and add structural tests.
3. **Base-only runtime and acceptance.** Cold/warm/cancel/switch/teardown plus creator corpus.
4. **Base+Refiner only on demonstrated value.** Separate closure, fingerprint, two-stage tests, and blind review.
5. **One ecosystem compatibility check.** Narrow LoRA or ControlNet after Base stability.

Stop if generic Comfy already satisfies the use case with lower maintenance, or if implementation scope expands into a general checkpoint manager.
