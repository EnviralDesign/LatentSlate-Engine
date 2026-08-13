# Stable Diffusion XL roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Stability AI Base/Refiner repositories and operation-matched configs |
| saved topology/defaults | embedded workflows in pinned examples [`f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3): [`sdxl_simple_example.png`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_simple_example.png), [`sdxl_refiner_prompt_example.png`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/sdxl/sdxl_refiner_prompt_example.png) |
| node behavior | pinned ComfyUI source selected by the packet; Kitchen not needed for initial FP16 |
| acceptance/tier | Engine public-API output/lifecycle/creator evidence; none exists |

The embedded graph is extracted and translated into Engine-owned code; it is never
submitted to ComfyUI.

## Decision

SDXL’s possible value is ecosystem compatibility, not historical obligation. First
perform a product-value and embedded-graph extraction spike. A Base FP16 Engine recipe
proceeds only for a concrete Engine-owned use case.

Unsupported fine-tunes, ControlNets, adapters, I2I, and inpaint remain outside the
built-in scope. Their existence does not create an alternate workflow-execution route.

## Boundaries

| Operation | Treatment |
| --- | --- |
| Base T2I | value gate, then typed Engine-native FP16 slice |
| Base + Refiner | separate two-stage Alternate after blind review |
| I2I/inpaint/control/adapters | absent until explicit product demand |

## Preflight and order

1. extract embedded workflow bytes without OCR;
2. retain source blob and normalized behavioral hash;
3. verify node semantics from pinned ComfyUI source;
4. resolve immutable Base closure: both text encoders/tokenizers, VAE, UNet,
   scheduler/config, license;
5. write independent Base and Refiner fixtures;
6. decide product value;
7. implement Engine-native Base only if justified;
8. add Refiner only for a material creator benefit.

There is no implemented, Recommended, or fallback SDXL recipe.

Stop on ComfyUI dependency, mutable-only defaults, checkpoint-only false availability,
or insufficient Engine-owned product value.
