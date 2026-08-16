# Model roadmaps and implementation authority

Last authority audit: **2026-08-15**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

All roadmaps are subordinate to the normative
[Comfy evidence and Engine execution policy](../COMFY_ENGINE_POLICY.md).

## Authority hierarchy

1. Publisher sources own weight identity, architecture, config, lineage, license, and
   dense Reference facts.
2. Pinned official workflows own practical topology and saved defaults.
3. Pinned ComfyUI **source** owns the node behavior studied during clean-room
   translation. It is never an Engine dependency or runtime.
4. Kitchen plus exact headers owns quantized layout and direct native dispatch.
5. Engine public-API evidence owns runnability, lifecycle, output facts, and tier.
6. Dense BF16 video is a high-memory comparison contract, not a local-fit mandate.

## Pin classes

There is no global “current Comfy pin.”

- **Accepted behavioral baseline:** workflow and ComfyUI source used to derive an
  accepted Engine-native implementation.
- **Accepted Kitchen baseline:** Kitchen version/source actually called directly by
  the accepted Engine runtime.
- **Current research:** source used for the next clean-room packet.
- **Authoring baseline:** template package/source used to ingest recipe evidence.
- **Historical alternate:** immutable source retained to explain older observations.
- **Mutable discovery:** landing page, tutorial, service, access, or legal page; never
  an execution or acquisition lock.

Examples:

- authoring workflow source:
  [`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1),
  package `0.1.37`;
- current research:
  workflows [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1),
  ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541),
  Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4);
- Klein accepted behavioral source:
  workflows [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb),
  ComfyUI source [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d),
  direct Kitchen `0.2.28` at
  [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028).

See [SOURCE_PIN_AUDIT.md](./SOURCE_PIN_AUDIT.md).

## Mandatory roadmap authority map

Every roadmap identifies:

| Surface | Owner |
| --- | --- |
| weights/architecture/config/license | immutable publisher source and artifacts |
| saved topology/defaults | exact workflow blob or publisher-native contract |
| node/quantized schema | ComfyUI source revision for research and Kitchen revision for direct Engine dispatch |
| acceptance/tier | exact Engine public-API evidence or truthful lower proof state |

Every roadmap also states explicitly that Engine does not execute ComfyUI.

## Implementation-agent preflight

Before implementation:

1. fetch and hash the raw workflow;
2. normalize subgraphs, switches, constants, output slots, active/disabled branches,
   and dynamic placeholders;
3. read pinned ComfyUI source to verify node inputs, outputs, preprocessing, and object
   semantics;
4. enumerate complete active closure and disabled resources;
5. resolve immutable artifacts, bytes, hashes, license/gate facts, and headers;
6. verify direct Kitchen layouts/primitives/fallback against exact headers;
7. write independent fixtures;
8. implement Engine-owned typed orchestration and disposable-worker lifecycle;
9. prove it through the Engine public API.

The normalized contract is never submitted to ComfyUI.

## Review gates

Reject:

- any ComfyUI package/import/process/server/graph/plugin/folder dependency;
- graph drift without a separate Engine deviation/fingerprint;
- hidden conversion or eager/dense fallback;
- false availability;
- assumed output metadata;
- unobserved cancellation/cleanup/recovery;
- cataloged or structurally tested work described as runnable;
- self-confirming fixtures;
- dense Reference settings weakened merely to fit local hardware.

## Status and proof vocabulary

Product tiers: **Reference**, **Recommended**, **Fallback**, **Alternate**,
**Experimental**, **Deferred**, **Rejected**.

Proof levels:

- **Hardware-proven:** target-class Engine output plus intended dispatch and lifecycle.
- **Runtime-proven:** one end-to-end Engine job; incomplete matrix.
- **Structurally tested:** independent behavioral/header/loader fixtures; no accepted
  output.
- **Cataloged:** declarations only.
- **Direct tool only:** callable Engine code outside the opinionated recipe surface.
- **Not implemented:** no conforming Engine path.

A nonconforming ComfyUI-executed prototype is research evidence, not an Engine proof
level.

## Portfolio decision surface

The built-in catalog contains **30 recipes**: all **21/21 optimized, non-Reference
recipes** have LatentSlate-originated Engine output acceptance, the **8 BF16
References** remain deliberately unaccepted high-memory structural/comparison
contracts, and one fixed Z-Image LoRA is Hardware-proven but remains Experimental
and local-only because its upstream license is not declared. Product tier and proof level are separate:
acceptance does not make every path Recommended. Comfy workflows and source remain
behavioral oracles only; no pixel/latent parity is claimed, and
ComfyUI does not participate in execution.

| Family | Practical Engine path | Current truth | Next bounded work |
| --- | --- | --- | --- |
| [Klein 4B](./FLUX2_KLEIN_4B.md) | Engine-native stored BF16/FP8/NVFP4; direct Kitchen | all five optimized recipes Hardware-proven, including ordered one-to-three-reference edit and cancel/recovery; BF16 structural only | broaden creator corpus; matching Base BF16 comparison |
| [Klein 9B](./FLUX2_KLEIN_9B.md) | Engine-native stored FP8/NVFP4; direct Kitchen | all four optimized recipes Hardware-proven, including ordered one-to-three-reference edit and cancel/recovery; BF16 bounded OOM | broaden creator corpus; high-memory BF16 and separate KV research |
| [Krea 2](./KREA_2.md) | Engine-native three-file saved-default INT8; separate Darkbrush mode | Not implemented | license gate, normalized fixture, direct Kitchen loader |
| [SDXL](./STABLE_DIFFUSION_XL.md) | Engine-native Base FP16 only if product value is proven | Not implemented | extract graph, value gate, then typed Base slice |
| [Qwen Edit 2511](./QWEN_IMAGE_EDIT_2511.md) | Engine-native stored INT8 standard; separate Lightning | Not implemented | ordered-input contract and direct Kitchen implementation |
| [Ideogram 4](./IDEOGRAM_4.md) | Engine-native dual-branch INT8 | Not implemented | JSON/license contract and direct Kitchen implementation |
| [Wan 5B](./WAN22_TI2V_5B.md) | Engine-native stored-mixed T2V/I2V from exact four-resource closure | narrow Hardware-proven Recommended: 24-fps T2V/strict-I2V, direct Kitchen, disposable-worker cancellation/recovery | broaden creator corpus and retain direct-dispatch/lifecycle proof; BF16 remains Reference |
| [Wan 14B](./WAN22_14B.md) | Engine-native stored expert runtimes derived from official workflows | I2V/T2V/FLF narrow Hardware-proven Fallbacks; LightX modes narrow Hardware-proven Alternates | broaden creator/switching evidence and retain direct-dispatch/lifecycle proof |
| [LTX 2.3](./LTX_2_3.md) | Engine-native optimized T2V/I2V and separate FLF | all three optimized A/V operations narrow Hardware-proven Recommended; BF16 retained only as unaccepted structural Reference | broaden creator corpus and retain exact A/V/dispatch/lifecycle proof |
| [LTX 2.5](./LTX_2_5.md) | Engine-native six-role T2V | Not implemented | gated closure and Engine-native Kitchen runtime |
| [MiniMax H3](./MINIMAX_H3.md) | Engine-native four-file optimized FL2VA after gate | BF16 direct-tool CPU/source contract only | authenticate optimized closure and implement direct Kitchen path |
| [Z-Image Turbo](./Z_IMAGE_TURBO.md) | Engine-native four-resource/three-weight-file INT8 T2I; separate exact fixed 70s Horror additive recipe | base is narrow Hardware-proven Recommended; fixed LoRA is target-hardware proven Experimental/local-only with exact lifecycle, switching, and dispatch evidence | clarify the upstream LoRA license; broaden creator evidence without claiming Comfy parity |

## Dense video policy

Source-pin and CPU-validate dense Wan/LTX/H3 references, retain one bounded local OOM
when useful, and batch full outputs on high-memory Vast. RTX 5080 work prioritizes the
conforming Engine-native practical path.

## Active roadmaps and indexes

- [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md)
- [FLUX.2 Klein 9B](./FLUX2_KLEIN_9B.md)
- [Krea 2](./KREA_2.md)
- [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md)
- [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md)
- [Ideogram 4](./IDEOGRAM_4.md)
- [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md)
- [Wan 2.2 14B](./WAN22_14B.md)
- [LTX 2.3](./LTX_2_3.md)
- [LTX 2.5](./LTX_2_5.md)
- [MiniMax H3](./MINIMAX_H3.md)
- [Z-Image Turbo](./Z_IMAGE_TURBO.md)
- [Implementation packets](./IMPLEMENTATION_PACKETS.md)
- [Source-pin audit](./SOURCE_PIN_AUDIT.md)
