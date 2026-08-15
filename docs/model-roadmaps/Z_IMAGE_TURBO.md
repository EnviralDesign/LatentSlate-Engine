# Z-Image Turbo roadmap

Last authority audit: **2026-08-14**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Z-Image source [`26f23eda626ffadda020b04ff79488e1d72004cd`](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd) and exact artifacts |
| saved topology/defaults | [`image_z_image_turbo_int8.json`](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_z_image_turbo_int8.json), blob `61bb66e258200a92db5626bb519d317e047807f4` |
| node behavior / dispatch | pinned local Comfy source `7fe8a6138504f90ff7be82f3babf416da32876b1` for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for direct Engine dispatch |
| acceptance/tier | narrow **Hardware-proven Recommended** through Engine public-API cold/warm output, exact dispatch, cancellation, cleanup, and recovery evidence |

## Decision

The exact four-resource/three-weight-file Engine-native Turbo T2I path is the
**Recommended** Z-Image recipe. Matching Turbo BF16 remains high-memory Reference.
Base is a separate line. No released Edit artifact is established, so I2I/edit is
absent. This promotion proves successful native execution and lifecycle behavior; it
does not claim pixel parity with Comfy.

## Contract and closure

Saved behavior: guidance-free BasicGuider with positive-only conditioning,
1024-square, AuraFlow shift 3, eight steps, `res_multistep`/`simple`, denoise 1.

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/z_image_turbo@d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e` / `split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors` | 6,201,001,296 | `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635` |
| `Comfy-Org/z_image_turbo@2f862278568d3f0a83167a16e5f11094da6dee72` / `split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors` | 5,631,994,051 | `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15` |
| `Comfy-Org/z_image_turbo@93fae7d7f6189cc408fdd7cec36c91447b8506a2` / `split_files/vae/ae.safetensors` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6` / exact four-file config/tokenizer support closure | 4,459,144 | exact per-file identity validation |
| total | 12,172,758,879 | — |

The Engine-native managed core and exact stored materializers are Hardware-proven on
the target workstation. The transformer executes all 202 INT8 ConvRot modules with
zero dense fallback. The mixed
Qwen closure retains 177 FP8 plus 12 NVFP4 stored Kitchen layouts; its Comfy-derived
`full_precision_matrix_mult` route uses F32 embeddings/hidden activations, explicit
Kitchen layout dequantization per low-bit linear, then ordinary F32 `F.linear`.
The Qwen implementation is an Engine-owned raw 398-key `model.*` shell with no
output head or cache seam. The
literal call path is prompt envelope -> unweighted Qwen tokenizer -> int64 IDs and
binary mask -> FP32 embedding rows -> FP32 causal-plus-padding mask and RoPE -> 36
separate-QKV/SwiGLU blocks. It captures block 34, still executes block 35 and the
final norm, discards the final-normalized result, and returns block 34 FP32.
It preserves source-derived Qwen parameter dtype (BF16 for this closure): each call
keeps the moved Kitchen wrapper logically BF16, extracts its public layout tensors,
and asks Kitchen's public FP8/NVFP4 API to dequantize directly to FP32. It never
dequantizes to BF16 and widens, or invokes a dtype-only wrapper cast.
It does not quantize activations or use `scaled_mm`; the temporary F32 weight is not
retained and no dense checkpoint fallback is permitted.

The 5.63 GB Qwen checkpoint remains a CPU master. Its stage does not call
`model.to(cuda)` or rebuild the complete Kitchen closure on CUDA. Instead, each
embedding, norm, dense linear, and Kitchen linear moves only its current operand to
the concrete execution device, with Kitchen qdata and every scale field moved as one
logical tensor; temporaries are released after the operation. This follows Comfy's
manual-cast/offload behavior and prevents a cold full-model CUDA allocation before
the first encoder block.

Closed conditioning diagnostics identify embedding, mask, RoPE, block ordinal,
linear ordinal, or final norm without exposing prompt text or local paths. The
forward runs under inference mode and the concrete CUDA device context; cancellation
is checked at block and linear boundaries. Ordered closed stages
`conditioning.edge_07` through `conditioning.edge_20` cover CPU-master validation,
dispatch snapshot, tokenization, tensor transfer, embedding movement, mask/position/
RoPE construction, block-34 capture, final norm, and returned-output validation.
Before tokenization, the child also performs a prompt-free first-linear preflight.
The authenticated header fixes `linear_000` as FP8 E4M3 with scalar FP32 scale and
logical shape `[4096,2560]`. The preflight moves its packed payload as exact bytes,
requests FP32 directly from the public dequantizer, produces a contiguous FP32
matrix, and executes `F.linear` against a fixed zero `[1,1,2560]` probe. Closed
CUDA-sync, uint8-allocation, ordinary-uint8-copy, post-copy-sync, uint8-readback,
origin-flat-prepare,
origin-uint8-copy, flat-dtype-view, shape-restore, scale-move, bit-verification,
direct-FP32-dequant, F.linear, and validation stages distinguish backend
failures without exposing module names, paths, prompts, or tensor contents. Preflight
counters are restored before the per-command dispatch baseline.

Packed movement uses one Engine-owned, source-backed raw-byte transport for preflight
and real linears. The public layout extracts qdata and scale fields from the immutable
CPU-master wrapper. FP8 E4M3 storage is reinterpreted with `.view(torch.uint8)`,
flattened to one contiguous byte buffer (never numerically converted), copied
synchronously on the concrete current CUDA stream using an ordinary-Tensor
`torch.empty_like(flat_bytes, device=device)` followed by blocking
`destination.copy_(flat_bytes, non_blocking=False)`, viewed back to FP8 while still
one-dimensional, and only then restored to its exact source shape; its FP32 scale is
copied separately. NVFP4 qdata is already `uint8`, so its raw qdata, FP32 global
scale, and blocked FP8 scale are copied independently without changing their dtype,
layout, or high-nibble-first convention. This is VBar-output-equivalent transport;
it deliberately does not dispatch Kitchen's typed `QuantizedTensor.copy_` path.
FP8 then calls `comfy_kitchen.dequantize_per_tensor_fp8(...,
output_type=torch.float32)`; NVFP4 calls `comfy_kitchen.dequantize_nvfp4(...,
output_type=torch.float32)` with its default high-nibble-first convention and crops
padding to `orig_shape`. Preflight proves byte count, bit identity, shape, scale
identity, current device, and that the CPU master was not mutated. The synchronous
current-stream branch needs no separate stream wait and avoids importing Comfy's
cast-buffer/offload-stream manager. Before touching the real FP8-origin buffer, the
prompt-free preflight synchronizes the concrete CUDA device (surfacing rather than
suppressing stale errors) and runs a bounded ordinary-`uint8` allocation/copy probe.
The worker checkpoints and Qwen preflight call the same shared implementation; it
uses explicit CPU source creation, `inference_mode`, normalized indexed CUDA device,
concrete current-device context, pre-copy sync, allocation, blocking copy, post-copy
sync, and CPU readback. Closed substages distinguish sync, allocation, copy,
post-copy sync, and readback so those paths cannot drift while retaining the same
diagnostic label.
The cold worker repeats that same model-free 16-byte synchronous CUDA health check at
fixed authenticated boundaries: before heavy runtime imports, after tokenizer/support,
after Qwen, after NextDiT, after Flux AE, after core construction, and immediately
before Qwen preflight. Successful results bind every phase to `pass`. Synthetic-only
failures expose only the closed phase, safe exception type, and one closed code:
`cuda_oom`, `illegal_memory_access`, `invalid_argument`, `operation_not_supported`,
`driver_error`, or `unknown_runtime`; raw exception text is never serialized.
Failure results also bind the exact ordered prefix of earlier passing health phases,
the failing phase, and the shared helper substage.
Before the first health checkpoint, the child resolves the authenticated device request
exactly once. Bare `cuda` becomes `cuda:<torch.cuda.current_device()>`; a valid
explicit indexed request remains exact and every CUDA operation enters its concrete
device context. Non-CUDA, unavailable, and invalid requests fail at the closed
`device_contract` boundary rather than being misclassified as a synthetic copy failure. Every health
phase, the core, the child runtime key, and the loaded-session binding then use only
that concrete indexed device. Success metadata retains both the original requested
device and the resolved execution device so result provenance cannot conflate them.
The authenticated first-linear backend label is exactly
`comfy_kitchen.dequantize_per_tensor_fp8+torch/f.linear`; the lower-level registered
custom-op name is not presented as the public API contract.

## Accepted target-workstation evidence

The 2026-08-14 LatentSlate public pipeline accepted the complete lifecycle. Every
successful command authenticated Qwen `177 FP8 + 12 NVFP4 = 189` dequantization and
F32 `F.linear` calls, plus all 202 NextDiT stored modules, with zero fallback.

- Cold: app `393de60e-c831-421c-b647-a41b31184469`, Engine
  `6eb4cc63-3e5c-4387-918a-c14ea6723177`, asset
  `28817b71-e1a5-4fed-ad34-657f9c840921`; worker `31392`; 36.63 s; RGB PNG
  1024x1024, 985,806 bytes, SHA-256
  `09B7FC41E0376A26585BAABD44385329811DD97E035E8E23A62BABA8CAF161A4`.
- Warm: app `f9e66022-eae5-46aa-a86c-28744afc5537`, Engine
  `d8dc7393-8b9b-41d4-a062-3bec9324e184`, asset
  `8a58d1ee-db1a-47d2-a978-1acbd03616ce`; the same worker `31392` reported
  `pipeline_warm=true`; 20.34 s; RGB PNG 1024x1024, 1,959,416 bytes, SHA-256
  `C15DA33A1D19717BD5D01A6C6F87FC734888BF6526E169ED210D3535C1F53E44`.
- Cancellation: app `5db2428d-e30d-4f56-a06d-7f25c946cc2e`, Engine
  `51c752df-bd35-4079-816b-104b9fc47ad3`; canceled while the app reported 22%;
  terminal state was canceled with `artifacts=[]` and `output_paths=[]`; cleanup
  completed, worker `31392` exited, and the runtime was evicted.
- Recovery: app `7e6d8c99-c8fb-455c-8cc5-b10414b141f3`, Engine
  `d3444cf5-ba94-4183-ac06-34227bb13064`, asset
  `0d80d2e6-03c3-429c-955d-58ab8ac33d1b`; new worker `10444` reported
  `pipeline_warm=false` and cold rematerialization; RGB PNG 1024x1024, 1,754,306
  bytes, SHA-256
  `8D0F0EE0644BEBFDB7B7D3D7DED0C2527CC40B1FF6B9EE3D554D81C2C28F40FA`.

The accepted recipe is therefore cataloged **Recommended** and is runnable when its
exact local closure is installed. These are LatentSlate-originated execution,
dispatch, output, and lifecycle facts—not a differential or pixel-parity result
against Comfy.

## Retained contract and next work

Retain normalized fixtures, exact maps, Engine-owned staged lifecycle, direct Kitchen
dispatch, all 189 Qwen calls, all 202 NextDiT modules, zero fallback, and the proven
cold/warm/cancel/recovery boundary. Next work may broaden creator prompts and compare
outputs without turning similarity into an unsupported parity claim. No ComfyUI
dependency or graph execution is permitted.

Stop on ComfyUI dependency, Base/Turbo mixing, incomplete mapping, fallback, false
availability, assumed metadata, or header proof presented as GPU acceptance.
