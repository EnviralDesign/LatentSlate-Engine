# Stable Diffusion XL roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Surface | Authority |
| --- | --- |
| Weights/architecture/config/license | Stability AI official Base/Refiner repositories and operation-matched Diffusers configs; landing pages remain mutable discovery until exact revisions are authored |
| Saved topology/defaults | embedded workflow metadata in pinned Comfy examples at [`f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3): [`sdxl_simple_example.png`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_simple_example.png) and [`sdxl_refiner_prompt_example.png`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_refiner_prompt_example.png) |
| Node/dispatch schema | exact ComfyUI checkout selected by the packet; research begins from [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541); Kitchen is not applicable to initial FP16 |
| Acceptance/tier | Engine public-API image artifacts, lifecycle, and creator review; no SDXL Engine family exists on current main |

The former mutable `master/sdxl` link is no longer an operation pin.

## Product decision

SDXL’s Engine value is ecosystem compatibility, not historical obligation. The first packet is a **product-value and embedded-graph extraction spike**, not “implement native BF16.” Generic Comfy remains the Fallback for arbitrary fine-tunes, LoRAs, ControlNets, and adapter graphs.

A native Base FP16 recipe proceeds only when a concrete workflow or deployment need benefits from Engine ownership. FP16 Base fits the workstation, but defaults still come from the exact practical graph rather than remembered Diffusers examples.

## Operation boundaries

| Operation | Contract | Disposition |
| --- | --- | --- |
| Base T2I | exact Base closure, both text encoders, VAE, scheduler, extracted Comfy defaults | value gate, then Experimental native slice |
| Base + Refiner | distinct two-stage graph, closures, and denoise handoff | Alternate after blind creator value |
| I2I/inpaint | source/mask/strength/preprocessing contract | generic Comfy until explicit demand |
| Control/LoRA/adapters | broad non-canonical ecosystem | generic Comfy; add one seam only after Base stability |

Base and Base+Refiner are separate recipes.

## Graph and closure preflight

Before choosing defaults:

1. fetch the pinned PNG and extract embedded workflow JSON without OCR/transcription;
2. retain PNG blob, embedded bytes/hash, and normalized API graph;
3. verify nodes/output slots against pinned ComfyUI;
4. enumerate checkpoint, both text encoders/tokenizers, VAE, scheduler/config, and any watermark/safety behavior;
5. resolve immutable artifact revisions, bytes, hashes, and license text;
6. build independent fixtures for Base and Refiner separately.

A monolithic checkpoint and Diffusers component repository are compatible only after loader/config proof; do not assume byte or numerical identity.

## Recipe and acceptance order

1. Generic Comfy SDXL — Fallback ecosystem surface.
2. `sdxl.text-to-image.native-base-fp16` — Experimental/Reference only after graph extraction and value gate.
3. `sdxl.text-to-image.native-base-refiner-fp16` — separate Alternate after blind review.

There is no Recommended or implemented Engine SDXL path.

If native Base proceeds, require 1024-square plus portrait/landscape, cold plus meaningful warm jobs, malformed closure, cancellation across text encode/Base/VAE/save, observed output metadata, teardown, and creator review of faces/hands, composition, photography, illustration, text, and ecosystem compatibility. Refiner adds separate switching/cancellation and must show visible value worth doubled lifecycle.

Next: extract/normalize pinned graphs; decide product value; Base FP16 only if justified; optional Refiner; one ecosystem seam after stability.

Stop if generic Comfy already serves the need, exact defaults cannot be extracted, or availability would be inferred from an installed checkpoint.
