# Z-Image Turbo implementation packet

Last independently verified: **2026-08-13**. This is a planning packet only:
Z-Image Turbo is **cataloged with a CPU/source contract in progress**. The
official three-file recipe is intentionally not hardware-accepted or promoted:
there is no installed payload, GPU execution, or output-parity claim yet.

## Bounded product decision

The next candidate is one native operation: **Z-Image Turbo text to image** using
the official Comfy INT8 ConvRot workflow. It is a three-artifact graph with a
fixed eight-step schedule. Z-Image Base is a separate, longer guided line and
is out of scope. No first-party edit/image-to-image artifact was established
for this packet; native edit and I2I must reject rather than silently route to
Turbo T2I or a generic workflow.

## Authorities

- [Comfy workflow templates at `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1)
- [Exact Turbo INT8 workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json)
- [ComfyUI at `725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541)
- [Comfy Kitchen at `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4)
- [ConvRot conversion reference at `1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341bb8a4e46f161a978b5faa2412d8c39c768/quant_int8_convrot.py)

The pinned workflow blob is `61bb66e258200a92db5626bb519d317e047807f4`
(28,029 bytes). It is the behavioral authority; its mutable `main` download
URLs are not acquisition locks.

## Exact graph and fixed contract

The pinned graph wires, in order:

1. `UNETLoader` for the Turbo INT8 ConvRot transformer.
2. `CLIPLoader` for the Qwen 3 4B mixed-FP8 encoder with type `lumina2`.
3. `VAELoader` for `ae.safetensors`.
4. Positive `CLIPTextEncode`; that conditioning goes through
   `ConditioningZeroOut` for the negative branch.
5. `EmptySD3LatentImage` (default 1024 × 1024, batch one).
6. `ModelSamplingAuraFlow` with shift `3`.
7. `KSampler`: 8 steps, CFG `1`, sampler `res_multistep`, scheduler `simple`,
   denoise `1`.
8. `VAEDecode` and save.

The first native recipe exposes only prompt and seed; resolution is fixed at
1024 × 1024 in the descriptor rather than accepted then rejected at runtime.
It must preserve the fixed schedule above and must fail closed on image input,
edit/I2I operation selection, Base/Turbo mixing, unknown resource headers,
or a fallback that dequantizes the stored INT8 artifact.

## Immutable Turbo INT8 closure

Each file below was checked through Hugging Face's revisioned tree API. `SHA-256`
is the LFS object identity; the listed file revision is the upload commit for
that exact path, not an implicit `main` reference.

| Role | Immutable source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Transformer | [`Comfy-Org/z_image_turbo` `split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors` at `d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e`](https://huggingface.co/Comfy-Org/z_image_turbo/blob/d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors) | 6,201,001,296 | `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` |
| Text encoder | [`Comfy-Org/z_image_turbo` `split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors` at `2f862278568d3f0a83167a16e5f11094da6dee72`](https://huggingface.co/Comfy-Org/z_image_turbo/blob/2f862278568d3f0a83167a16e5f11094da6dee72/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors) | 5,631,994,051 | `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` |
| VAE | [`Comfy-Org/z_image_turbo` `split_files/vae/ae.safetensors` at `93fae7d7f6189cc408fdd7cec36c91447b8506a2`](https://huggingface.co/Comfy-Org/z_image_turbo/blob/93fae7d7f6189cc408fdd7cec36c91447b8506a2/split_files/vae/ae.safetensors) | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| **Fixed closure** | **all three active workflow artifacts** | **12,168,299,735** | — |

Arithmetic: `6,201,001,296 + 5,631,994,051 + 335,304,388 =
12,168,299,735` bytes.

## Current CPU/source contract

- The transformer header has empty global metadata. Its complete **202** INT8
  roles are authenticated by the per-layer U8 `comfy_quant` markers (including
  `convrot_groupsize`) and per-row F32 scales; a global-metadata substitute is
  rejected.
- The Qwen header has **398** weights: 209 BF16, 177 scalar-scale FP8, and 12
  packed NVFP4. The latter carries FP8 block scales, F32 tensor scales, and
  `{"format":"nvfp4"}` markers; all 189 low-bit linears require native
  Kitchen dispatch proof. A standard Qwen shell may add only its tied
  `lm_head.weight` alias to the checkpoint's `model.embed_tokens.weight`; it
  is revalidated and re-tied explicitly, not treated as an arbitrary extra or
  a dense/generic FP8 Qwen.
- The catalog records the exact immutable revision, LFS SHA-256, header/schema
  identity, schedule, and no-warm-cache lifecycle. Source/header qualification
  reports stored layers as *planned*, never as observed native dispatch. GPU
  materialization and generation acceptance remain pending.

## Implementation and proof gates

- Add a typed split-image family with exactly `transformer`, `text_encoder`,
  and `vae` roles. Reject Base resources, arbitrary Lumina files, duplicate
  dense copies, metadata-less substitutions, and unsupported tensor layouts.
- Validate the ConvRot mapping and every intended INT8 native dispatch before
  generation. The runtime must not use eager conversion/dequantization as a
  hidden fallback. Validate the encoder and VAE keys/configuration separately.
- Stage the encoder, cache CPU-frozen prompt conditioning, release its device
  residency, then stage the transformer for the eight denoise steps and release
  it before VAE decode. Provenance fingerprints must include all resource
  revisions/hashes and every fixed schedule value. A cancellation or native
  fallback poisons/ejects the runtime.
- Acceptance begins at 1024 square with fixed prompt and seed, then measures a
  cold run, three changed-seed warm runs, prompt/dimension changes, switching
  to Klein and back, cancellation at each lifecycle phase, malformed resources,
  residency teardown, allocator/external-memory metrics, and output hashes.

Do not promote a tier or claim parity until the header, native-dispatch,
lifecycle, and hardware gates have passed. Base, ControlNet, LoRA, and all edit
operations remain separate future slices.
