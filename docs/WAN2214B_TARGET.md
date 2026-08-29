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
`[1, 512, 4096]` float32 tensors. The session retains the conditioning tensors
under their prompt-pair key. Reusing the prompt pair reuses conditioning;
changing either prompt recomputes only conditioning while the high/low
checkpoint and LoRA caches remain warm.

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
in scale and all but 2,814 of 70,778,880 FP8 values after the runtime adopted
Comfy's module-name stochastic seed. That 0.00398% residual is the earliest
measured numerical difference after identical FFN input.

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
- Comfy graph flow versus a direct invocation of its staged model under the
  same canonical inputs has relative RMSE `6.47e-9`, absolute RMSE `7.77e-9`,
  and maximum error `1.19e-7`; the two Comfy paths are effectively identical.
- Engine first flow versus the graph Comfy flow has relative RMSE `0.107223`,
  absolute RMSE `0.128812`, maximum error `1.68262`, and cosine similarity
  `0.994264`. Engine mean/std are `0.060603/1.201419` versus Comfy
  `0.061065/1.199795`. Engine versus the direct staged-model flow is equivalent
  because the two Comfy controls coincide.
- Feeding Engine Comfy's exact conditioning reduces relative RMSE to
  `0.091030`. With that control, block 0 is exact through self-attention,
  normalized cross-attention input, every cross-attention projection, and the
  cross-attention output. The earliest residual is the sparse materialized FFN
  FP8 requantization difference above, which accumulates through 40 blocks.
  This supports bounded FP8 requantization variance; it does not rely on visual
  similarity or a claimed Comfy graph/direct residency divergence.
- Exact Comfy final latent VAE decode and Engine VAE decode matched at the VAE
  seam. The final Engine artifact is coherent: the robot walks through a
  furnished kitchen/living interior over the full sequence. Its seed-specific
  framing differs from Comfy's centered composition.
- Tests prove live patches rebuild from an immutable base, materialized FP8
  cache bytes survive phase reactivation unchanged, and repeated use does not
  accumulate deltas.
- Artifact identity consumes both checkpoints, both LoRAs and strengths, text
  encoder, VAE, and canonical settings, but excludes request prompt text.
  Focused tests prove same-pair reuse, positive- or negative-prompt invalidation
  without model replacement, and destructive invalidation for a true
  model/LoRA identity replacement.

## Fresh performance and memory evidence

Fresh matching Comfy results on the RTX 5080:

- cold: 48.52 seconds;
- five genuine same-process seed-only warm runs: 35.03, 33.64, 33.69,
  37.14, and 33.75 seconds;
- warm median: 33.75 seconds;
- process working set: 29.14 GiB idle, 31.25 GiB peak, 2.11 GiB
  incremental;
- WDDM total-device GPU use: 10.69 GiB idle, 15.45 GiB peak, 4.77 GiB
  incremental.

Both high- and low-noise phases reran in every warm request.

Final Engine results from one persistent session, including conditioning reuse,
both model phases, decode, and MP4 save:

- cold: 93.13 seconds;
- five genuine same-process seed-only warm runs: 35.97, 33.37, 33.30,
  33.14, and 33.21 seconds;
- warm median: 33.30 seconds, 1.33% faster than the matching Comfy median and
  inside the approximate 10% objective;
- process working set: 24.78 GiB idle, 31.86 GiB peak, 7.08 GiB incremental;
- WDDM total-device GPU use: 4.29 GiB idle, 15.22 GiB peak, 10.93 GiB
  incremental.

The memory figures use equivalent total-device and process-working-set
boundaries on both sides. The separately polled Engine telemetry run took
37.06 seconds; its perturbed timing is not used for the unpolled performance
median. Engine's total GPU peak is 0.23 GiB lower and working-set peak is
0.60 GiB higher than Comfy. No memory optimization is indicated, and the
pageable materialized cache remains the accepted choice.

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

1. Identity-switch replacement: **confirmed**. Artifact/settings identity
   excludes prompts but includes both model/LoRA owners; true replacement
   destructively invalidates the old session.
2. Prompt-conditioning retention: **confirmed**. Positive/negative UMT5
   outputs are reused for the same prompt pair. Changing either prompt
   recomputes conditioning without invalidating model/LoRA state.
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
