# Qwen Image Edit 2511 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Contract surface | Authority |
| --- | --- |
| Weights, architecture, lineage, license | official Qwen 2511 repository/config; LightX2V owns the Lightning LoRA lineage; immutable file/header identities own stored representations |
| Saved operation topology/defaults | official [`image_qwen_image_edit_2511_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json), Git blob `251ffb5115cf8e6ab27b2ebc1038423737f22e72` |
| Node and quantized dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), ConvRot source [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) |
| Acceptance and tier | ordered-input Engine artifacts, observed dispatch, lifecycle, and creator review; no Qwen 2511 Engine path exists on current main |

The Comfy stack is a research baseline, not accepted runtime provenance.

## Product decision

Qwen 2511 is an **editing** family, not ordinary T2I. The first local implementation starts from the saved-default official Comfy INT8 graph. BF16 remains the 40-step quality Reference on adequate hardware, not the first 16 GB engineering target.

The request uses an ordered media list. Publisher examples establish one/two-image editing; the Comfy schema exposes a third socket. Any three-image Engine surface is separately qualified and labeled.

## Saved-default and Lightning modes

The graph selects the INT8 ConvRot transformer, Qwen2.5-VL 7B scaled-FP8 encoder, and Qwen image VAE. It exposes ordered image1/image2/image3, AuraFlow shift 3.1, and a Lightning switch saved **false**.

| Mode | Fixed closure | Schedule | Treatment |
| --- | --- | --- | --- |
| saved default | transformer + encoder + VAE | 40 steps, CFG 4 | first normalized practical implementation |
| Lightning enabled | same three + fixed official LoRA | 4 steps, CFG 1 | separate recipe/fingerprint |

Three-file saved-default closure: **30,137,560,750 bytes**. Four-file Lightning closure: **30,987,169,046 bytes**. The VAE is 253,806,246 bytes, SHA-256 `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f`, immutable revision `dfe60a0d63f0b946628080f070978594983b8b6e`.

Do not compare Lightning as a pure precision variant of the 40-step teacher. Compare it first with BF16 plus the same LoRA.

## Recipe and implementation order

1. `qwen-image-edit-2511.image-to-image.comfy-int8-standard` — Experimental saved-default three-file graph, ordered inputs, 40 steps/CFG 4.
2. `qwen-image-edit-2511.image-to-image.comfy-int8-lightning` — separate fixed four-file graph, four steps/CFG 1.
3. `qwen-image-edit-2511.image-to-image.native-bf16` — Reference on adequate hardware.
4. Fused scaled-FP8 Lightning — deferred challenger after both exact modes.

No Qwen 2511 recipe is runnable or accepted on current main.

## Typed contract and preflight

The request records stable input indexes/roles, content hashes, decoded dimensions, exact resize/crop results, prompts, seed, and effective canvas. The typed recipe owns `transformer`, `text_vision_encoder`, and `vae`; Lightning adds fixed `model_lora`.

Before coding: normalize all subgraphs/switches; verify `TextEncodeQwenImageEditPlus`, reference-latent method, sampler, and output slots against pinned ComfyUI; validate ConvRot and scaled-FP8 maps/sidecars/aliases/dense exceptions; validate VAE and Lightning target/rank/layout; bind LoRA identity, steps, and CFG atomically; reject ambiguous order, unsupported count, partial closure, conversion, or fallback.

Lifecycle: validate/decode media before model allocation, preprocess in order, cache conditioning by ordered hashes, stage/release encoder, stage transformer, apply fixed LoRA only in Lightning mode, denoise, release transformer, decode VAE, and observe output. Cancellation invalidates media conditioning and ejects uncertain state.

## Acceptance and next slices

Corpus: no-op/surgical edits, identity, text replacement, products/materials, insertion/removal, relighting, multi-person cases, and untouched-region stability. Require cold plus meaningful warm jobs, one-to-two-to-one inputs, separately labeled three-input case, standard-to-Lightning-to-standard switching, malformed resources, phase cancellation/recovery, observed output metadata, and teardown.

Next: normalized saved-default graph and one/two-image schema; header manifest/independent fixtures; three-file loader; RTX 5080 acceptance; separate Lightning recipe; three-image extension; BF16 high-memory Reference.

Stop on mutable-only identity, graph drift, ambiguous image order, incomplete mapping, hidden fallback, false availability, or unobserved cancellation.
