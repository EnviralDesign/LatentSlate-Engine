# Qwen Image Edit 2511 implementation roadmap

Last corrected: **2026-08-12**

Engine architecture audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)

Official Comfy evidence:

- [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [standard workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511.json)
- [INT8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json)
- [ComfyUI source `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)

## Decision

Qwen Image Edit 2511 is an editing family, not a general T2I line. Its official publisher reference and current Comfy template support ordered multi-image editing, but the exact operation must distinguish the saved 40-step graph from the optional four-step Lightning mode.

The pinned INT8 template contains a Lightning LoRA branch, but its saved `Enable 4steps LoRA?` switch is **off**. Therefore:

- the saved-default execution closure is three files and runs 40 steps at CFG 4;
- the optional four-step mode is a four-file closure with the Lightning LoRA active and CFG 1.

Do not call the four-step mode a precision-only version of the 40-step reference. It changes both artifact closure and schedule.

## Product and operation boundary

| Operation/mode | Inputs and ordering | Exact graph boundary | Disposition |
| --- | --- | --- | --- |
| Saved-default standard edit | one required image at top level; subgraph exposes ordered image 1/2/3 sockets | INT8 transformer, encoder, VAE; Lightning switch off; 40 steps, CFG 4 | Experimental reference implementation target for the Comfy INT8 graph |
| Four-step Lightning edit | same ordered image contract | same three files plus exact Lightning LoRA; switch on; 4 steps, CFG 1 | Separate Experimental product mode |
| Publisher BF16 edit | publisher pipeline accepts an ordered image list and demonstrates two images | 40-step high-precision source | Reference |
| Third image | current Comfy subgraph exposes a third socket | Engine extension unless independently accepted as official product behavior | Explicitly labeled extension |
| Inpaint/control/reference variants | different graphs, artifacts, or semantics | not part of this slice | Generic Comfy Or Deferred |

The top-level template activates one image. The publisher demonstrates two images. The subgraph’s three sockets do not by themselves prove that a three-reference product default should ship.

## Exact artifact closures

### Saved-default INT8 graph

| Role | Immutable identity | Bytes | SHA-256 | Provenance |
| --- | --- | ---: | --- | --- |
| Transformer | `Comfy-Org/Qwen-Image-Edit_ComfyUI`, `split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors`, revision `e9e85de74a8f48c1e3e2656617626348675a2f21` | 20,499,083,824 | `11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829` | Comfy-Org stored INT8 ConvRot, Apache-2.0 |
| Text/vision encoder | `Comfy-Org/HunyuanVideo_1.5_repackaged`, `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, revision `f10daa9e51f1b192302ef701ffc918af0652830e` | 9,384,670,680 | `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4` | Comfy-Org scaled FP8 Qwen2.5-VL 7B |
| VAE | [`Comfy-Org/Qwen-Image_ComfyUI`, revision `dfe60a0d63f0b946628080f070978594983b8b6e`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/dfe60a0d63f0b946628080f070978594983b8b6e/split_files/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | Apache-2.0 Qwen image VAE |

Saved-default total: `20,499,083,824 + 9,384,670,680 + 253,806,246 = 30,137,560,750` bytes.

The transformer and encoder revisions above were resolved from their file commit histories and exact pointer metadata. Their model-card links remain mutable discovery pages: [Qwen Edit Comfy](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) and [HunyuanVideo repackaged encoder source](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged).

### Optional four-step Lightning mode

| Additional role | Immutable identity | Bytes | SHA-256 | Provenance |
| --- | --- | ---: | --- | --- |
| Model-only LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`, revision `fd3a43ffb5bc98c7d09b2238e5b09a63284a16f8`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/blob/fd3a43ffb5bc98c7d09b2238e5b09a63284a16f8/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors) | 849,608,296 | `22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f` | LightX2V Apache-2.0 four-step LoRA |

Four-step total: `30,137,560,750 + 849,608,296 = 30,987,169,046` bytes.

The template’s model switch, step switch, and CFG switch must move together. Loading the LoRA while leaving 40 steps/CFG 4, or enabling four steps without routing through the LoRA, is an invalid hybrid.

### BF16 reference

The mutable publisher discovery page is [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). A Reference resource must pin the complete repository revision, tokenizer/processor/config closure, and exact weight shards. The publisher’s canonical comparison uses 40 steps and ordered image inputs.

## Official Comfy topology

Preserve the pinned template’s actual nodes and switches:

1. INT8 ConvRot `UNETLoader`.
2. scaled-FP8 Qwen2.5-VL `CLIPLoader(type="qwen_image")`.
3. Qwen image VAE.
4. positive and negative `TextEncodeQwenImageEditPlus` with ordered image sockets.
5. `FluxKontextMultiReferenceLatentMethod(index_timestep_zero)` around conditioning where represented.
6. `CFGNorm`, model/CFG/step switch nodes, and optional `LoraLoaderModelOnly`.
7. AuraFlow sampling shift `3.1`.
8. saved-default 40 steps/CFG 4 versus optional 4 steps/CFG 1.
9. VAE encode/decode and 8-bit sRGB PNG output.

Do not collapse the template into a generic “Turbo” Boolean without preserving all three coordinated switches and provenance.

## Recipe ladder and candidate contracts

| Candidate key | Tier | Fixed contract |
| --- | --- | --- |
| `qwen-image-edit-2511.image-to-image.comfy-int8-standard` | Experimental | exact three-file closure; 40 steps; CFG 4; one/two ordered images in initial public schema; Lightning disabled |
| `qwen-image-edit-2511.image-to-image.comfy-int8-lightning-4step` | Experimental | exact four-file closure; Lightning enabled; 4 steps; CFG 1; same ordered input semantics |
| `qwen-image-edit-2511.image-to-image.native-bf16` | Reference | complete first-party BF16 repository; 40-step publisher settings; adequate hardware/offload |

A typed split recipe needs roles `transformer`, `text_encoder`, `vae`, and optionally a fixed `model_lora` for the four-step key. The Lightning artifact is fixed recipe state, not an arbitrary user LoRA slot.

## Loader and runtime implementation packet

Reuse the generic resource closure, `stored_quant.py`, Klein mixed-encoder and native-dispatch patterns, runtime fingerprints, byte-bounded prompt/media caches, and manager poison/ejection behavior.

Likely new files: `qwen_image_edit_recipe.py`, `runtime/qwen_image_edit.py`, `tools/qwen_image_edit.py`, built-in declarations, and tests.

Header/schema gates:

- exact Qwen edit transformer mapping, including fused projections, dense exceptions, ConvRot markers/scales, aliases, and unsupported transpose detection;
- complete Qwen2.5-VL text/vision mapping and image-token preprocessing contract;
- exact VAE schema;
- exact Lightning target set/rank/dtype for the four-step recipe;
- ordered image count and hashes in cache keys and provenance;
- reject missing fixed LoRA in Lightning key, unexpected LoRA in standard key, incomplete component closure, hidden dense copies, or runtime conversion.

Lifecycle: validate media before model load; preprocess images in declared order; stage encoder and cache CPU-frozen conditioning; release encoder device residency; materialize transformer; optionally attach fixed Lightning branch; denoise; release transformer; decode and save. Cancellation or a dispatch-integrity failure ejects all request-local media state and the loaded runtime.

Native proof requires positive intended Kitchen dispatch counts and zero eager/dequantized fallback. LoRA application must not dequantize the base.

## Hardware and scientific acceptance

Fixed cases:

- one-image surgical edit and two-image composition;
- seed `43301611940728`;
- 1024-square output;
- standard mode at 40 steps/CFG 4;
- Lightning mode at 4 steps/CFG 1;
- same ordered images, prompt, negative prompt, preprocessing, encoder, VAE, and seed within each comparison.

Corpus: no-op/minimal edit, identity preservation, two-person consistency, exact text replacement, material/color change, object insertion/removal, relighting, geometry change, and repeated edits from the same source.

Required scenarios: cold plus three changed-seed warm generations; standard to Lightning to standard switching; one/two-image inputs; explicitly labeled three-image experiment only after two-image acceptance; cancellation during media preprocessing, encoder, LoRA attach, materialization, denoise, decode, and save; malformed each-resource cases; teardown.

Provenance asserts mode, all artifact identities, switch state, ordered image hashes, preprocessing, steps, CFG, shift, schema/layout counts, dispatch counts, cache state, and output hash.

## Ordered bounded slices

1. **Next: request contract and saved-default standard graph.** Expose one required and one optional ordered image; implement exact three-file 40-step graph. Out of scope: Lightning, third image, controls.
2. **Standard hardware acceptance.** Cold/warm, one/two images, cancellation, malformed artifacts, switching to another family, and creator review.
3. **Four-step Lightning mode.** Add exact fixed LoRA and coordinated model/step/CFG switches. Compare against BF16 plus the same LoRA when available.
4. **Third-image extension only after explicit approval.** Label it Engine-specific and keep separate acceptance/provenance.
5. **Stop.** Do not add GGUF, AWQ, Nunchaku, arbitrary LoRAs, or generic control graphs without a creator requirement.
