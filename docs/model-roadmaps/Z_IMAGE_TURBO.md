# Z-Image Turbo roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Surface | Authority |
| --- | --- |
| Weights/architecture/lineage/license | Z-Image publisher source [`26f23eda626ffadda020b04ff79488e1d72004cd`](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd) and exact artifacts |
| Saved topology/defaults | official [`image_z_image_turbo_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json), Git blob `61bb66e258200a92db5626bb519d317e047807f4` |
| Node/dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), ConvRot source [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) |
| Acceptance/tier | Engine public-API image, exact header/resource provenance, observed native dispatch/lifecycle/memory, and creator review; no GPU acceptance exists on current main |

The publisher commit and README were independently resolved. They distinguish Turbo, Base, Omni, and Edit lineages; they do not create an available Engine edit operation.

## Product decision

The next bounded new image operation is **Z-Image Turbo T2I** from the exact official Comfy INT8 graph. Matching Turbo BF16 is Reference on adequate hardware. Base is a separate longer guided line. No released first-party Edit artifact was established for this packet, so native I2I/edit fails closed.

Current proof is catalog/source/header work only. Do not call it runnable, Hardware-proven, Recommended, or Fallback.

## Exact graph and closure

The 28,029-byte pinned workflow loads:

1. `z_image_turbo_int8_convrot.safetensors` through `UNETLoader`;
2. `qwen_3_4b_fp8_mixed.safetensors` through `CLIPLoader(type="lumina2")`;
3. `ae.safetensors` through `VAELoader`;
4. positive text conditioning and zeroed negative conditioning;
5. 1024-square `EmptySD3LatentImage`;
6. AuraFlow shift 3;
7. eight-step, CFG-1, `res_multistep`/`simple`, denoise-1 sampling;
8. VAE decode and image save.

| Role | Bytes | SHA-256 |
| --- | ---: | --- |
| INT8 ConvRot transformer | 6,201,001,296 | `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` |
| mixed-FP8 Qwen 4B encoder | 5,631,994,051 | `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` |
| VAE | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| **total** | **12,168,299,735** | — |

Template `resolve/main` URLs are discovery only. Exact revisions remain in the resource/header packet.

## Engine proof preserved

CPU/header work authenticates 202 intended transformer INT8 roles through per-layer markers, scales, and group size; the mixed Qwen file includes dense BF16, scalar-scale FP8, packed NVFP4, sidecars, and tied-weight handling. Immutable identities, schema fingerprints, and planned layer counts are recorded.

No GPU materialization, denoise, native dispatch, image, memory, cancellation, or warm-lifecycle evidence exists. Proof level: **Cataloged / Structurally tested in progress**.

## Implementation and acceptance

First recipe: `z-image-turbo.text-to-image.comfy-int8-convrot` as Experimental; typed roles are `transformer`, `text_encoder`, and `vae`. Before coding, normalize/hash the graph; verify ComfyUI inputs/output slots; validate all transformer and mixed-Qwen maps/markers/scales/aliases/dense exceptions plus VAE; bind resources, graph, ComfyUI/Kitchen, schedule, canvas, and policy into the fingerprint; reject image input, Base/Turbo mixing, conversion, or fallback.

Lifecycle stages encoder then transformer then VAE, retaining only bounded CPU conditioning. Cancellation or dispatch-integrity failure ejects the runtime.

Acceptance uses 1024-square plus typography, illustration, architecture, materials, faces/hands, long prompts, portrait, and landscape. Require cold plus changed-seed warm jobs, Z-to-Klein-to-Z switching, malformed resources, phase cancellation/recovery, observed image metadata, memory, teardown, native dispatch/zero fallback, hashes, and creator review.

Next: normalized graph/fixtures; materialization/dispatch; RTX 5080 acceptance; Turbo BF16 high-memory Reference; Base only for explicit demand; edit only after exact released authority.

Stop on graph drift, incomplete mapping, fallback, false availability, assumed metadata, or claiming header proof as GPU acceptance.
