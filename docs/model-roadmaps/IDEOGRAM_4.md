# Ideogram 4 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/prompt/license | Ideogram source [`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2) and exact artifacts |
| saved topology/defaults | [`image_ideogram4_t2i_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_ideogram4_t2i_int8.json), blob `4e9a71db38bc0c6e09aafba658adb5b06d10c8fa` |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for direct Engine dispatch |
| acceptance/tier | Engine JSON, branch dispatch, lifecycle, safety/output, creator review; none exists |

## Decision

Ideogram 4 is structured-JSON T2I with separate conditional and unconditional
branches. No dense public teacher exists; official NF4 Diffusers is the public
Reference baseline, explicitly not lossless dense truth.

The first local candidate is an Engine-native four-file INT8 implementation derived
from the pinned workflow. Hosted prompt expansion is a separate external operation.
I2I/edit/control are absent until separately specified. Product/legal review must
approve the complete closure: Ideogram model branches carry non-commercial terms,
the Qwen repack is Apache-2.0, and the Flux2 VAE has separate terms.

## Closure

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/Ideogram-4@e18159a2e9a95cdb4ecd76f49cecdf5291849697` / `diffusion_models/ideogram4_int8_convrot.safetensors` | 9,583,465,712 | `a9164002943463b4c7b2abd88c82a488c088acc35762651e4d8604d6ce4a163d` |
| `Comfy-Org/Ideogram-4@8532c0f76182375c10b8f082dc6b0be196ef0615` / `diffusion_models/ideogram4_unconditional_int8_convrot.safetensors` | 9,583,465,712 | `cd03ed94f244c9cb705e7d30ca0f40b5f5b004bb20674117adff88d16416c23d` |
| `Comfy-Org/Qwen3-VL@7f1d4413e3bd9ae24580b14d4113bfce872c55f0` / `text_encoders/qwen3vl_8b_fp8_scaled.safetensors` | 10,588,637,512 | `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` |
| `Comfy-Org/flux2-dev@ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa` / `split_files/vae/flux2-vae.safetensors` | 336,213,556 | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |
| total | 30,091,782,492 | — |

One branch is never complete. Prompt-assistant mode is part of the closure and request
fingerprint.

## Implementation order

1. composite license decision and deterministic JSON schema;
2. normalized dual-branch fixture;
3. complete maps/headers and direct Kitchen plan;
4. Engine-native dual-branch runtime;
5. RTX 5080 typography/layout acceptance;
6. NF4 high-memory Reference;
7. FP8/NVFP4 only after full mode closure.

No local Ideogram recipe is runnable or accepted.

Stop on ComfyUI dependency, missing branch, hidden hosted expansion, incomplete
headers, fallback, false availability, or assumed output metadata.
