# Source-pin and closure audit ledger

Last independently verified: **2026-08-13**. This ledger records only facts
rechecked for this branch or present in current `main`; it does not
retroactively certify claims from an external research packet.

## Rules

An immutable implementation source needs a resolvable full Git commit or a
Hugging Face file revision plus exact path, byte count, and LFS SHA-256. A
mutable model card can establish discovery, publisher intent, or license only;
it cannot be an acquisition lock. A parity closure includes every artifact
actively wired by the pinned workflow, including fixed LoRAs and unconditional
branches. Arithmetic must be shown and must include all such artifacts.

Before changing a roadmap, verify the source, the individual file identity, the
workflow connection, and `git diff --check`. If any identity is gated or cannot
be resolved, record it as a blocker rather than completing it by inference.

## Resolved shared Git authorities

| Purpose | Repository | Immutable commit |
| --- | --- | --- |
| External research architecture baseline | `EnviralDesign/LatentSlate-Engine` | `b2481702d7b888a8553a4ce8b3302258a7a1fd96` |
| Official workflow templates | `Comfy-Org/workflow_templates` | `2b7f823136606344f0bccce249898d771b809aa1` |
| ComfyUI source used by current packets | `Comfy-Org/ComfyUI` | `725e6ec60621c6f001af04769173e7dbb3c53541` |
| Comfy Kitchen | `Comfy-Org/comfy-kitchen` | `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4` |
| ConvRot reference | `Comfy-Org/comfy-model-tools` | `1fe341bb8a4e46f161a978b5faa2412d8c39c768` |

The following values were deliberately **not** carried forward merely because
they appeared in the external packet: a purported ComfyUI examples pin,
MiniMax, Ideogram, and Krea GitHub repository pins did not resolve under the
named repository paths during this audit. Their family documents must retain
their own evidence or be revalidated before treating any new Git link as
immutable.

## Corrected Z-Image Turbo closure

The external packet had a malformed Hugging Face URL for the mixed-FP8 text
encoder (`z_image_turboblob`) and an impossible VAE SHA ending in `g`. The
implementation packet uses the corrected URL and this valid VAE SHA:
`afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`.
The complete, independently checked three-file closure is in
[Z_IMAGE_TURBO.md](./Z_IMAGE_TURBO.md): 12,168,299,735 bytes.

## Krea 2 saved-default and enabled-LoRA closures

The pinned Comfy INT8 T2I workflow configures `krea2_darkbrush` through
`LoraLoaderModelOnly` at strength `0.8`, but its saved `enable_lora` switch is false
and selects the base transformer. Exact saved-default parity is therefore three files.
The immutable LoRA identity is retained for the explicitly enabled four-file variant.

| Artifact | Immutable revision/path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Transformer | [`6b1d7191d84d5ded74d83a1a98211dad0ac8ae25` / `diffusion_models/krea2_turbo_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/6b1d7191d84d5ded74d83a1a98211dad0ac8ae25/diffusion_models/krea2_turbo_int8_convrot.safetensors) | 13,492,686,496 | `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1` |
| Text/vision encoder | [`4aa0eed112bd2780ceea37583edbdcd2df6c2c09` / `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/4aa0eed112bd2780ceea37583edbdcd2df6c2c09/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | 5,242,467,968 | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| VAE | [`a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e` / `vae/qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| Optional Darkbrush LoRA | [`b5a1dcd1574c1d256408cbb5ae46a67b225481e6` / `loras/krea2_darkbrush.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/b5a1dcd1574c1d256408cbb5ae46a67b225481e6/loras/krea2_darkbrush.safetensors) | 469,291,992 | `f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6` |
| **Enabled four-file variant** | **base closure plus Darkbrush** | **19,458,252,702** | — |

Arithmetic: the saved-default base closure is
`13,492,686,496 + 5,242,467,968 + 253,806,246 = 18,988,960,710` bytes.
The explicitly enabled variant adds the LoRA:
`18,988,960,710 + 469,291,992 = 19,458,252,702` bytes.

The four-file path must not be called saved-default parity unless the workflow switch
is explicitly enabled and that distinct operation is named.
