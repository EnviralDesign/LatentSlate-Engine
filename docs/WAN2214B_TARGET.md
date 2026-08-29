# Wan 2.2 14B canonical T2V target

## Authority and artifacts

The executable source of truth is the unmodified ComfyUI Export (API) prompt
`reference/comfy/wan2214b/t2v-pytorch-baseline-api.json`. It was validated and
executed on `Comfy C (PyTorch Baseline)`, pinned to ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f`, Torch 2.11 + CUDA 13.0,
comfy-kitchen 0.2.31, and comfy-aimdo 0.4.15. The process used
`--use-pytorch-cross-attention` without fast FP8 matmul flags.

The consumed artifacts are:

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| High checkpoint | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | 14,293,923,632 | `cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f` |
| High LoRA | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | 1,226,977,424 | `698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41` |
| Low checkpoint | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | 14,293,923,632 | `e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97` |
| Low LoRA | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | 1,226,977,424 | `ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| VAE | `wan_2.1_vae.safetensors` | 253,815,318 | `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b` |

## Exact recipe

- Wan 2.2 T2V 14B, turbo-only.
- High and low LightX2V four-step LoRA strengths are both
  `1.0000000000000002`.
- `ModelSamplingSD3` shift `5.000000000000001`; Euler/simple; four total
  steps; high steps 0–1; low steps 2–3; CFG 1.
- Sigmas are `[1.0, 0.9375, 0.8333333134651184, 0.625, 0.0]`.
- Seed `923510416338945`; CPU random noise shape `[1, 16, 21, 64, 64]`.
- 512×512, requested 5 seconds, 16 fps, with
  `floor(duration * fps) + 1 = 81` frames.
- Positive prompt: `a robot walks through the interior of a house, scanning ordinary objects such as a coffee table, a kitchen table, and a plant.`
- The exact Chinese negative prompt remains pinned in the fixture and
  `wan2214b.pipeline.NEGATIVE_PROMPT`.

The artifact is MP4/H.264 High profile, `avc1`, 8-bit `yuv420p`, TV range,
BT.709 primaries/matrix, sRGB transfer, 512×512, 16 fps, 81 frames, and an
effective duration of 5.0625 seconds.

## Exercised source and execution trace

UMT5 tokenization uses the checkpoint SentencePiece model without BOS and
with EOS, padded to 512 tokens. Positive and negative conditioning are each
`[1, 512, 4096]` float32 tensors. The session retains both conditioning tensors
and token data after the first request.

Each diffusion checkpoint has 40 transformer blocks, width 5120, FFN width
13824, 40 heads, and 16 latent channels. Native checkpoint FP8 linears use
comfy-kitchen `QuantizedTensor` input/weight matmul. Each exact LightX2V LoRA
contains 400 rank-64 targets with alpha 8. High and low adapters are opened by
their owning `WanWeights`; their state and cache dictionaries cannot cross.

The pinned Comfy run staged approximately 13,628 MB and attached 400 patches.
Its observed low-VRAM split kept 137 attention patches live and materialized
263 patches. Live patches dequantize the base and apply the FP16 LoRA matmul on
each use. Materialized patches use FP32 LoRA intermediates, add into the FP16
dequantized base, then use Comfy's float16 reciprocal scale clamp and seeded
stochastic FP8 requantization. A probed materialized FFN weight matched Engine
exactly in scale and every FP8 byte.

Engine retains immutable file-backed bases, pageable CPU copies of materialized
patched FP8 weights, and conditioning across requests. Checkpoint mappings are
rotated in bounded intervals while the cold cache is built so already-consumed
file pages do not remain resident. During a phase Engine uploads only that
model's materialized cache and the 160 q/k norm weights. High state is released
from VRAM before low state is activated. A bounded one-layer prefetch for live
patches retains its source views and records the main CUDA stream before use.
The low model is released before VAE decode. Both checkpoint stores and their
model-local CPU patch caches remain warm after a request; the two full 14B
checkpoints do not coexist in VRAM.

Sampling uses the canonical CPU noise and four sigma intervals. The first two
Euler updates use the high model. The resulting float32 latent is handed
directly to the low model, preserving leftover noise; the final two updates use
the low model. CFG 1 makes the negative conditioning a retained canonical input
but does not require a second denoiser evaluation. The final latent is converted
from Wan's normalized latent space before the Wan 2.1 VAE decode.

## Correctness and lifecycle evidence

- Token IDs, noise construction, latent geometry, and sigma sequence match the
  fixture.
- Engine block 0 versus a fresh Comfy graph capture has 1.92% relative error;
  means/stds are `0.007356/0.563978` versus `0.007405/0.568178`.
- The complete first Engine flow has 48.68% relative error versus the fresh
  graph-derived flow. A second direct invocation through Comfy's staged model
  itself differed from that graph seam by about 21.15%, proving material
  residency-dependent nondeterminism. The remaining Engine discrepancy is
  accumulated across 40 layers and is explicitly not bit parity.
- Exact Comfy final latent VAE decode and Engine VAE decode matched at the VAE
  seam. The final Engine artifact is coherent: the robot walks through a
  furnished kitchen/living interior over the full sequence. Its seed-specific
  framing differs from Comfy's centered composition.
- Tests prove live patches rebuild from an immutable base, materialized FP8
  cache bytes survive phase reactivation unchanged, and repeated use does not
  accumulate deltas.
- Artifact identity consumes both checkpoints, both LoRAs and strengths, text
  encoder, VAE, canonical settings, and prompts. Destruction clears retained
  state and makes the old session unusable. Replacement constructs a new
  session only after destructive invalidation.

## Fresh performance and memory evidence

Fresh matching Comfy results on the RTX 5080:

- cold: 48.94 seconds;
- genuine seed-only warm runs: 37.88, 52.09, and 36.48 seconds;
- warm median: 37.88 seconds;
- process-tree RAM: approximately 29–30 GiB;
- total GPU memory during inference: approximately 14–15 GB.

Both high- and low-noise phases reran in every warm request.

Final Engine results from one persistent session, including conditioning reuse,
both model phases, decode, and MP4 save:

- cold: 82.87 seconds;
- genuine seed-only warm runs: 43.21, 42.94, 42.57, and 42.64 seconds;
- warm median: 42.79 seconds, about 13.0% slower than Comfy's 37.88-second
  median and 1.13 seconds above a literal 10% ceiling;
- process RSS peak: 27.58 GiB cold and 32.07 GiB warm;
- peak PyTorch GPU allocation: 12.31 GiB.

The timing is close to, but does not literally meet, the approximate 10% target.
Permanent pinning reached a 41.64-second warm median but drove process RSS to
43.8 GiB, so the final runtime keeps the exact cache pageable and accepts the
small transfer cost. Cold RSS is below Comfy's observed 29–30 GiB and steady
warm RSS is about 2 GiB higher; GPU allocation remains lower than Comfy's
14–15 GB total inference usage.

## Family-specific lessons

- Wan's canonical identity naturally owns two checkpoints and two model-local
  LoRAs as one recipe.
- Prompt conditioning is worth retaining across seed-only requests.
- The high-to-low handoff is a direct Euler latent continuation with leftover
  noise, not a generic model-switch abstraction.
- LoRA arithmetic and cache ownership are part of the model's residency plan;
  live and materialized targets require distinct handling.
- Stream ownership is correctness-critical. Waiting on a prefetch stream is
  insufficient by itself; tensors must also record the consuming stream so the
  allocator cannot recycle their storage early.
- T2V supplies no evidence about content-derived reference/guide retention.

## Hypothesis classification

1. Identity-switch replacement: **confirmed**. Artifact/settings identity is
   complete, and replacement destructively invalidates the old session.
2. Prompt-conditioning retention: **confirmed**. Positive/negative UMT5
   outputs and token data are reused on seed-only requests.
3. Content-derived reference retention: **insufficient evidence**. Canonical
   T2V has no image, video, or guide input.
4. Euler-style flow/update and direct model handoff: **confirmed**. The high
   phase's float32 Euler latent, including leftover noise, is the low phase's
   input.
5. Isolated GPU worker/process containment: **insufficient evidence**. The
   canonical path is stable in-process; this target did not compare a worker
   boundary or prove that one is required.

I2V, FLF, non-turbo sampling, arbitrary LoRAs, and cross-family consolidation
remain outside this target.
