# Krea 2 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, lineage, license | Krea publisher source [`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058), its resolved license file, and immutable artifact identities |
| Saved operation topology/defaults | official [`image_krea2_turbo_t2i_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i_int8.json) |
| Node and quantized dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), and ConvRot source [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) |
| Acceptance and tier | Engine public-API output, observed native dispatch, lifecycle, and creator review; none exists for Krea on current main |

These are research pins, not accepted runtime provenance.

## Product decision

Krea is a separate family: **Raw** is the foundation/training line and **Turbo** is creator-facing inference. The first local implementation candidate is the exact saved-default official Comfy Turbo INT8 graph. BF16 remains operation-matched Reference on adequate hardware, not the first RTX 5080 engineering task.

The Krea Community License is a product gate. The source text is pinned; product/legal review must decide permitted use and obligations before built-in acquisition or promotion.

## Saved-default graph

The pinned graph uses:

- `krea2_turbo_int8_convrot.safetensors`;
- `qwen3vl_4b_fp8_scaled.safetensors`;
- `qwen_image_vae.safetensors`;
- prompt enhancement enabled, 512 maximum tokens, thinking disabled;
- 1024-square; eight steps, CFG 1, Euler/`simple`, denoise 1;
- Darkbrush configured at strength `0.8`, but `enable_lora=false`.

The saved-false switch selects the base model. Therefore:

- saved-default execution is a **three-file** closure;
- enabled Darkbrush-at-0.8 is a separate **four-file** mode;
- workflow prose recommending another strength does not override the literal configured branch value;
- neither mode may be described as the other.

## Practical closure

| Resource | Bytes | SHA-256 |
| --- | ---: | --- |
| INT8 ConvRot transformer | 13,492,686,496 | `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1` |
| Qwen3-VL 4B scaled-FP8 encoder | 5,242,467,968 | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| Qwen image VAE | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| Darkbrush variant LoRA | 469,291,992 | `f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6` |

Saved-default total: **18,988,960,710 bytes**. Darkbrush mode: **19,458,252,702 bytes**.

Mutable Hugging Face landing pages remain access/discovery surfaces. Resource declarations require immutable revisions, bytes, hashes, and headers.

## Recipe and implementation order

1. `krea-2-turbo.text-to-image.comfy-int8` — Experimental first implementation: exact three-file saved-default graph.
2. `krea-2-turbo.text-to-image.comfy-int8-darkbrush` — separate Alternate/Experimental fixed-LoRA mode at 0.8.
3. `krea-2-turbo.text-to-image.native-bf16` — Reference on adequate hardware.
4. Style-reference — separate operation after base T2I acceptance.

No Krea recipe is runnable or accepted on current main.

The typed recipe owns `transformer`, `text_encoder`, and `vae`; the variant adds fixed `model_lora`, not an optional user slot. Normalize the raw graph before coding, verify ComfyUI node/output schema, validate complete Krea and Qwen maps plus ConvRot/scaled-FP8 sidecars, and fail closed on Raw/Turbo mixing, ambiguous switch state, conversion, or fallback.

## Acceptance and next slices

Use fixed editorial, typography, materials, anatomy, illustration, and long-direction prompts. Require cold plus changed-seed warm jobs, base-to-Darkbrush-to-base switching, malformed each-resource cases, cancellation across enhancement/load/materialization/LoRA/denoise/decode, observed output metadata, teardown, and creator review.

Next: license decision; normalized saved-default fixture; three-file loader; RTX 5080 acceptance; separate Darkbrush mode; BF16 high-memory Reference.

Stop on license uncertainty, graph drift, incomplete headers, hidden fallback, false availability, or unobserved cancellation.
