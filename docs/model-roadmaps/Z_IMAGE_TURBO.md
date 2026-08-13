# Z-Image Turbo roadmap

Last authority audit: **2026-08-13**

Engine source audited:
[`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Z-Image source [`26f23eda626ffadda020b04ff79488e1d72004cd`](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd) and exact artifacts |
| saved topology/defaults | [`image_z_image_turbo_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json), blob `61bb66e258200a92db5626bb519d317e047807f4` |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for direct Engine dispatch |
| acceptance/tier | Engine public-API output/dispatch/lifecycle; no GPU acceptance exists |

## Decision

Next bounded operation: Engine-native Turbo T2I derived from the exact three-file INT8
behavior. Matching Turbo BF16 is high-memory Reference. Base is a separate line. No
released Edit artifact is established, so I2I/edit is absent.

## Contract and closure

Saved behavior: positive conditioning plus zeroed negative, 1024-square, AuraFlow
shift 3, eight steps, CFG 1, `res_multistep`/`simple`, denoise 1.

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/z_image_turbo@d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e` / `split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors` | 6,201,001,296 | `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` |
| `Comfy-Org/z_image_turbo@2f862278568d3f0a83167a16e5f11094da6dee72` / `split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors` | 5,631,994,051 | `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` |
| `Comfy-Org/z_image_turbo@93fae7d7f6189cc408fdd7cec36c91447b8506a2` / `split_files/vae/ae.safetensors` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| total | 12,168,299,735 | — |

CPU/header work identifies intended transformer and mixed-Qwen low-bit roles, markers,
scales, aliases, and sidecars. This is structural evidence only.

## Required implementation

Write normalized independent fixtures, validate all maps, implement Engine-owned staged
encoder/transformer/VAE lifecycle, and call Kitchen directly for every intended
quantized primitive. No ComfyUI dependency or graph execution is permitted.

Acceptance requires 1024-square output, meaningful warm jobs, Z-to-Klein-to-Z
switching, malformed artifacts, phase cancellation/recovery, observed metadata,
memory, teardown, positive dispatch/zero fallback, hashes, and creator review.

Stop on ComfyUI dependency, Base/Turbo mixing, incomplete mapping, fallback, false
availability, assumed metadata, or header proof presented as GPU acceptance.
