# Krea 2 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Krea publisher source [`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058), pinned license path, exact artifacts |
| saved topology/defaults | [`image_krea2_turbo_t2i_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i_int8.json) |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) and exact headers for future direct Engine dispatch |
| acceptance/tier | Engine public-API evidence; none exists |

Engine will translate the behavior into typed Engine code and call Kitchen directly;
it will not execute the workflow.

## Decision

Raw is the foundation/training line. Turbo is creator-facing inference. The first local
candidate is the exact saved-default three-file INT8 Turbo behavior. BF16 remains
Reference on adequate hardware, not the first RTX 5080 implementation.

The Krea Community License is a product/legal gate. At the pinned source it permits
community-license commercial use only below US$1 million trailing-twelve-month
company-wide revenue and requires reasonable content filtering; product/legal review
must confirm the intended built-in use.

## Saved-default contract

- INT8 ConvRot transformer;
- Qwen3-VL 4B scaled-FP8 encoder;
- Qwen image VAE;
- prompt enhancement enabled, 512 max tokens, thinking disabled;
- 1024-square, eight steps, CFG 1, Euler/`simple`, denoise 1;
- Darkbrush configured at 0.8 but `enable_lora=false`.

Saved default is three files, **18,988,960,710 bytes**. Darkbrush-enabled is a separate
four-file mode, **19,458,252,702 bytes**.

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/Krea-2@6b1d7191d84d5ded74d83a1a98211dad0ac8ae25` / `diffusion_models/krea2_turbo_int8_convrot.safetensors` | 13,492,686,496 | `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1` |
| `Comfy-Org/Krea-2@4aa0eed112bd2780ceea37583edbdcd2df6c2c09` / `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | 5,242,467,968 | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| `Comfy-Org/Krea-2@a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e` / `vae/qwen_image_vae.safetensors` | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| `Comfy-Org/Krea-2@b5a1dcd1574c1d256408cbb5ae46a67b225481e6` / `loras/krea2_darkbrush.safetensors` | 469,291,992 | `f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6` |

## Implementation order

1. license decision and normalized independent fixture;
2. Engine-native typed three-role loader and direct Kitchen dispatch;
3. RTX 5080 acceptance;
4. separate fixed Darkbrush mode;
5. high-memory BF16 Reference;
6. style-reference only after base value.

No Krea recipe is runnable or accepted. Candidate edition names must not imply a
ComfyUI backend.

Stop on license uncertainty, ComfyUI dependency, switch ambiguity, incomplete headers,
conversion/fallback, false availability, or unobserved cancellation.
