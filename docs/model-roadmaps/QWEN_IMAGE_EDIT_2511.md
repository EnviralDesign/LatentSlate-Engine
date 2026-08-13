# Qwen Image Edit 2511 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Qwen publisher source; LightX2V owns Lightning lineage; exact artifacts own stored layouts |
| saved topology/defaults | [`image_qwen_image_edit_2511_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_qwen_image_edit_2511_int8.json), blob `251ffb5115cf8e6ab27b2ebc1038423737f22e72` |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for future direct Engine dispatch |
| acceptance/tier | Engine ordered-input artifacts/lifecycle/creator review; none exists |

## Decision

This is an editing family, not ordinary T2I. First implement the saved-default
three-file INT8 behavior as Engine-owned typed orchestration. BF16 is the 40-step
Reference on adequate hardware.

Publisher evidence establishes one/two-image editing. The source node schema exposes a
third socket; any three-image Engine surface is separately qualified.

## Fixed modes

| Mode | Closure | Schedule |
| --- | --- | --- |
| standard | INT8 transformer + scaled-FP8 encoder + VAE | 40 steps, CFG 4 |
| Lightning | standard closure + fixed official LoRA | 4 steps, CFG 1 |

Totals: **30,137,560,750 bytes** standard; **30,987,169,046 bytes** Lightning.
Lightning is not a pure precision comparison to the teacher; compare first with BF16
plus the same LoRA.

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/Qwen-Image-Edit_ComfyUI@e9e85de74a8f48c1e3e2656617626348675a2f21` / `split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors` | 20,499,083,824 | `11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829` |
| `Comfy-Org/HunyuanVideo_1.5_repackaged@f10daa9e51f1b192302ef701ffc918af0652830e` / `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9,384,670,680 | `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4` |
| `Comfy-Org/Qwen-Image_ComfyUI@dfe60a0d63f0b946628080f070978594983b8b6e` / `split_files/vae/qwen_image_vae.safetensors` | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| `lightx2v/Qwen-Image-Edit-2511-Lightning@fd3a43ffb5bc98c7d09b2238e5b09a63284a16f8` / `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | 849,608,296 | `22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f` |

## Implementation order

1. ordered one/two-image request and normalized fixture;
2. exact header/LoRA maps and direct Kitchen plan;
3. Engine-native standard runtime;
4. RTX 5080 acceptance;
5. separate atomic Lightning recipe;
6. separately labeled three-image extension;
7. high-memory BF16 references.

No Qwen 2511 recipe is runnable or accepted. New candidate keys should describe
lineage/layout without implying ComfyUI execution.

Stop on ComfyUI dependency, ambiguous order/count, partial closure, non-atomic
Lightning switches, conversion/fallback, false availability, or unobserved
cancellation.
