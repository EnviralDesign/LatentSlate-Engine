# Wan 2.2 14B canonical T2V target

## Authority and artifacts

The executable source of truth is the repo-pinned ComfyUI Export (API) prompt
`reference/comfy/wan2214b/t2v-pytorch-baseline-api.json`. It was validated and
executed on `Comfy C (PyTorch Baseline)`, pinned to ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f`, Torch 2.11 + CUDA 13.0,
comfy-kitchen 0.2.31, and comfy-aimdo 0.4.15. The process used
`--use-pytorch-cross-attention` without fast FP8 matmul flags.

The consumed artifacts are:

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| High checkpoint | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | 14,293,923,632 | `cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f` |
| High LoRA | `wan2.2_t2v_lightx2v_4steps_lora_v1_1_high_noise.safetensors` | 1,226,977,424 | `698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41` |
| Low checkpoint | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | 14,293,923,632 | `e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97` |
| Low LoRA | `wan2.2_t2v_lightx2v_4steps_lora_v1_1_low_noise.safetensors` | 1,226,977,424 | `ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| VAE | `wan_2.1_vae.safetensors` | 253,815,318 | `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b` |

The local T2V LoRA filenames normalize upstream's `v1.1` token to `v1_1`;
their hashes remain the accepted upstream artifacts.

## Product request domain

The canonical 512x512, 81-frame requests remain the protected numerical proof
points. The public T2V, I2V, and FLF request surface additionally accepts the
following shared, reject-without-coercion domain:

- integer width and height on the 16-pixel lattice, each at least 480;
- at most 921,600 pixels and at most a 16:9 long-side-to-short-side ratio;
- 17 through 81 frames on the `4n+1` lattice;
- at most 21,504 transformer tokens, where token count is
  `(((frames - 1) // 4) + 1) * (height // 16) * (width // 16)`; and
- an integer seed from 0 through `2^64 - 1`.

The token limit is a spatial/temporal tradeoff, not another fixed shape. It
retains the canonical 512x512x81 proof point, accepts 480x832 through 49
frames, and accepts 1280x720 at 17 frames. A live 480x832x81 I2V reference run
completed in Comfy, but the matching Engine run exhausted the 16 GB reference
GPU in the first high-noise transformer block. The next lower tested lattice
point, 480x832x49 (20,280 tokens), completed in both runtimes. The product
boundary therefore stops at the canonical 21,504-token budget rather than
claiming Comfy-only shapes that this Engine target cannot execute.

Frame rate remains fixed at 16 fps; requested frame count is the public
temporal primitive and artifact duration is `frames / 16`. Width, height,
frame count, prompts, and seed are request state rather than model identity.
Changing them retains the high/low checkpoint and LoRA caches. I2V derived
conditioning is keyed by source content plus target width, height, and frame
count; FLF uses the same key components for its ordered source pair. Geometry
or frame-count changes rebuild only that derived state. Source decoding and
bilinear center-crop behavior remain source-conformant, and FLF continues to
place the ordered endpoints and four-channel mask at the requested temporal
boundaries.

The spatial limits combine Comfy's native 16-pixel Wan node lattice with the
official Wan 480p/720p and LightX2V 81-frame training shapes. The temporal
lattice is Wan's public `4n+1` contract. No sampler, schedule, checkpoint,
LoRA, prompt, VAE, model-family, or operator change is part of this extension.

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

# Wan 2.2 14B canonical I2V operation

## Authority, input, and artifacts

The I2V executable source of truth is the ComfyUI API prompt
`reference/comfy/wan2214b/i2v-pytorch-baseline-api.json`, SHA-256
`045f08500ad16782fe5a99de01ead80db878c59948ac68627a8d64aa54a77e04`.
It was validated and executed unchanged after the four model/LoRA selections
were corrected in commit `3aad61e`. The matching process was
`Comfy C (PyTorch Baseline)` at the same pinned ComfyUI commit and PyTorch
cross-attention configuration as T2V.

The canonical request image is `C:\ComfyUI\input\front.png`: 1,085,126 bytes,
1024×1024 RGB, no EXIF data or orientation transform, SHA-256
`63b7e7401e75991f618db3181b6003816aab87954b1979bcb97fbf36c63323e5`.
It is request content, not model identity.

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| High checkpoint | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | 14,294,742,832 | `6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42` |
| High LoRA | `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1,226,977,424 | `d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e` |
| Low checkpoint | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | 14,294,742,832 | `5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e` |
| Low LoRA | `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1,226,977,424 | `024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9` |

I2V reuses the accepted UMT5 XXL FP8 encoder and Wan 2.1 VAE artifacts. Each
I2V LoRA has the same validated 400-target structure as the accepted T2V
adapters. The fixture requests 512×512, 5 seconds, 16 fps, 81 frames from
`floor(duration * fps) + 1`, four Euler/simple steps split 2+2, CFG 1,
`ModelSamplingSD3` shift 5, and seed `264244520398999`. The exact positive and
negative prompts remain pinned in the fixture and `wan2214b.i2v`.

## Source trace and operation boundary

Pinned Comfy loads the image as RGB float32 in `[0, 1]`, applies any decoded
rotation, then performs bilinear center-crop scaling to 512×512. For this square
source the resized tensor is `[1, 512, 512, 3]` and matches Engine exactly.
`WanImageToVideo` creates an 81-frame float32 video with the source as frame
zero and exact 0.5 gray for all remaining frames. The VAE receives
`[1, 3, 81, 512, 512]` BF16 in `[-1, 1]` and produces a raw source latent
`[1, 16, 21, 64, 64]` float32.

The conditioning mask is `[1, 1, 21, 64, 64]`: temporal position zero is 0 and
all later positions are 1. Comfy normalizes the source latent with Wan's fixed
16-channel mean/std, inverts the mask, repeats it to four channels, and forms
`mask4 + normalized_latent16`. The denoiser input is the current 16-channel
sample followed by those 20 conditioning channels, for
`[1, 36, 21, 64, 64]`. At the first high step the sample channels are exact
canonical CPU noise; the image channels are neither substituted into the
sample nor protected by a sampler mask. They condition every high and low
denoiser call unchanged.

The single-file I2V checkpoints have 36-channel patch input but no image
cross-attention projection weights. Pinned Comfy therefore exercises the same
transformer/cross-attention behavior, sigma schedule, direct float32 2+2
high-to-low Euler handoff, VAE decode, and media writer already accepted for
T2V. Only image load/resize, VAE encode, mask construction, and concatenated
model conditioning are new I2V behavior.

Comfy's same-process warm requests cached image loading, resize, source VAE
encode, both prompts, both model/LoRA owners, and sampling-model setup. Both
samplers, final VAE decode, and output write reran on every changed-seed request.
Engine retains source-derived latent/mask under a content identity of byte size
plus SHA-256. The same bytes at another path reuse it; changed bytes rebuild it
without disturbing model/LoRA or unchanged prompt state. Prompt changes use the
accepted prompt-pair invalidation independently of image state. True recipe
replacement destructively releases prompt, image, model/LoRA, and VAE state.

## Numerical and output evidence

The resized source tensor and mask match Comfy exactly. At the new VAE-encode
seam, Engine versus Comfy has absolute RMSE `0.002718`, relative RMSE
`0.000906`, maximum error `0.03125`, and cosine similarity `0.999882`; mean and
standard deviation agree to the reported precision. This is the first measured
I2V-specific residual and is bounded BF16 encoder-kernel variance.

At the first actual 36-channel denoiser input, canonical noise and mask are
exact. Normalized image conditioning has relative RMSE `0.001113`, absolute
RMSE `0.001442`, and maximum error `0.020996`; UMT5 context has relative RMSE
`2.72e-5`. The complete first high flow has relative RMSE `0.030772`, absolute
RMSE `0.044470`, maximum error `0.67114`, and cosine similarity `0.999537`.
The second high flow, first low flow, and second low flow have relative RMSE
`0.053882`, `0.085739`, and `0.110857`, respectively. The final latent at the
VAE input has relative RMSE `0.084507` and cosine similarity `0.996327`.
The I2V-specific seams are therefore substantially closer than the accepted
T2V first-flow FP8 residual before the same sparse materialized requantization
difference accumulates through the shared 40-layer transformer.

Comparing raw decoded RGB tensors before H.264 gives pixel MAE `0.008741`, RMSE
`0.038615`, and PSNR `28.265 dB`. Representative frame RMSE values are
`0.008028` (first), `0.040378` (middle), and `0.040857` (last); corresponding
MAE values are `0.002766`, `0.009737`, and `0.009581`. This comparison excludes
codec noise and is consistent with the internal seam evidence rather than
serving as its substitute.

The final artifact is MP4/H.264 High profile, 8-bit `yuv420p`, TV range,
BT.709 primaries/matrix, sRGB transfer, 512×512, 16 fps, 81 frames, and 5.0625
seconds, matching Comfy's media contract.

## Fresh performance and memory evidence

Fresh matching Comfy I2V results on the RTX 5080:

- cold: 50.910 seconds;
- five genuine same-process seed-only warm runs: 34.009, 32.982, 32.929,
  32.669, and 32.731 seconds;
- warm median: 32.929 seconds;
- process working set: 32,378,335,232 bytes idle, 33,375,969,280 bytes peak,
  997,634,048 bytes incremental;
- WDDM total-device GPU use: 10,720 MiB idle, 15,102 MiB peak, 4,382 MiB
  incremental.

Final Engine I2V results from one persistent unpolled session:

- cold: 80.874 seconds;
- five genuine same-process seed-only warm runs: 35.875, 32.278, 35.778,
  32.246, and 32.634 seconds;
- warm median: 32.634 seconds, 0.90% faster than matching Comfy and inside the
  approximate 10% objective;
- process working set: 26,772,344,832 bytes idle, 34,371,268,608 bytes peak,
  7,598,923,776 bytes incremental;
- WDDM total-device GPU use: 3,925 MiB idle, 15,582 MiB peak, 11,657 MiB
  incremental.

The separately polled Engine telemetry run took 48.894 seconds and is excluded
from the performance median. Engine retains much less idle RAM and VRAM; its
absolute working-set and total-device GPU peaks are each about 3% higher than
Comfy. The larger incremental figures are a consequence of that lower idle
residency, not a materially higher absolute resource ceiling. No I2V-specific
memory or performance optimization is indicated.

## Wan-local hypothesis observation

Content-derived reference/guide retention is **confirmed by the I2V
operation**: source-derived latent and mask are independent of model and prompt
identity, reusable for identical bytes at a changed path, and invalidated when
the bytes change. This is evidence only; no T2V/I2V consolidation or shared
reference framework is extracted before FLF has a vote.

# Wan 2.2 14B canonical FLF operation

## Canonical recipe and pinned-source trace

The FLF executable source of truth is
`reference/comfy/wan2214b/flf-pytorch-baseline-api.json`, SHA-256
`a4f64366096f2094a791f54bb59a4e088968d03068cd6ec10cc6c8bc0714ea88`.
It is an unchanged ComfyUI API prompt with resolved `class_type` and `inputs`.
Against pinned ComfyUI commit `12d5279438bfefc058a269eae805ceab6047777f`,
it selects:

- `wan22\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` with
  `wan\wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` at strength
  `1.0`;
- `wan22\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` with
  `wan\wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` at strength
  `1.0`;
- `WanFirstLastFrameToVideo`, 512x512, 81 frames, UMT5 XXL FP8, Wan 2.1 VAE,
  four Euler/simple steps, CFG 1, SD3 shift 5, and seed `984937593540091`.

The high sampler executes steps `[0, 2)` with noise and returns leftover noise.
The low sampler disables new noise and executes from step 2 through the end.
Its fixture value `end_at_step=10000` exceeds the four-step sigma tail, so
pinned KSampler semantics leave the tail untruncated; the sentinel is not
normalized to 4.

The start image is `front.png`: 1,085,126 bytes, SHA-256
`63b7e7401e75991f618db3181b6003816aab87954b1979bcb97fbf36c63323e5`,
1024x1024 RGB. The end image is `back.png`: 1,112,094 bytes, SHA-256
`b2ae3502ac181e45528da6bcfea2f7f3f1b90eabe6a6935d5dc1c29cb931c7da`,
1024x1024 RGB. Both are 8-bit sRGB PNGs with no EXIF orientation transform.
Each is decoded to float32 RGB `[0, 1]` and independently bilinear-scaled with
a centered crop to 512x512.

Pinned `WanFirstLastFrameToVideo` creates one float32 video tensor shaped
`[81, 512, 512, 3]`, initially filled with `0.5`. It places the resized start
image at frame 0 and resized end image at frame 80; frames 1 through 79 remain
gray. The joint video is transformed to `[-1, 1]`, converted to BF16, and sent
through one causal Wan VAE encode. The raw encoded result is one
`[1, 16, 21, 64, 64]` conditioning latent.

The pre-reshape mask is `[1, 1, 84, 64, 64]`: temporal indices 0 through 3 and
83 are zero and every other index is one. Viewing temporal groups of four and
transposing produces `[1, 4, 21, 64, 64]`; its first group is `[0,0,0,0]`,
ordinary middle groups are `[1,1,1,1]`, and its last group is `[1,1,1,0]`.
WAN 2.1 model conditioning inverts that mask and concatenates its four channels
before the normalized 16-channel encoded video. Each denoiser call therefore
receives sampled noise/latent channels first, inverted mask channels second,
and encoded-video channels last: 36 channels total. The 20 conditioning
channels remain constant across all four evaluations.

Execution is endpoint load/resize, joint conditioning-video construction and
VAE encode, retained prompt conditioning, two high-noise Euler updates, direct
float32 leftover-noise handoff, two low-noise updates, VAE decode, then MP4
write. Comfy caches the endpoint load and `WanFirstLastFrameToVideo` result on
seed-only requests while both sampler phases, final decode, and media save
rerun.

## Ownership and lifecycle

FLF owns one joint image-conditioning entry keyed by an ordered pair of
content identities (byte size plus SHA-256), not filesystem paths. The natural
cache unit follows the one joint causal VAE encode rather than inventing two
independent latent slots. Focused tests prove:

- the same ordered bytes at changed paths reuse image conditioning;
- changing only the first or only the last image rebuilds joint conditioning
  without disturbing prompt or model/LoRA state;
- swapping the same two payloads invalidates conditioning because first and
  last are semantic ordered roles;
- changing either prompt recomputes only prompt conditioning and retains image
  and model/LoRA state;
- changing only the seed reruns full sampling, decode, and write while prompt,
  ordered image conditioning, and model/LoRA state remain warm;
- true recipe/model/LoRA replacement destructively releases prompt, image,
  VAE, checkpoint, and LoRA-derived state.

## Numerical and output evidence

Engine and Comfy decoded start/end tensors, resized endpoints, composed
81-frame conditioning video, BF16 VAE encoder input, canonical CPU noise, and
final mask match exactly. At the new joint VAE-encode seam, Engine versus Comfy
has pixelwise latent MAE `0.000505`, absolute RMSE `0.002727`, relative RMSE
`0.000917`, maximum error `0.03125`, and cosine similarity `0.9999996`.

At the complete first 36-channel denoiser input, relative RMSE is `0.000886`,
absolute RMSE `0.000964`, and maximum error `0.021484`. Within it, noise and
mask are exact; normalized image channels have relative RMSE `0.001126`,
absolute RMSE `0.001445`, and maximum error `0.021484`; UMT5 context has
relative RMSE `3.35e-5` and maximum error `0.000244`.

The first high flow has relative RMSE `0.108333`, absolute RMSE `0.156112`,
maximum error `2.50586`, and cosine similarity `0.994121`. This is the earliest
material downstream residual and is the same scale as the frozen T2V control
(`0.107223`) produced by the already-characterized sparse stochastic FP8
requantization difference. Every earlier FLF-specific seam is exact or bounded
at the BF16 VAE-encode level. The second high, first low, and second low flows
have relative RMSE `0.20692`, `0.23410`, and `0.24883`; the final latent has
relative RMSE `0.25229` and cosine similarity `0.96800`. The increase is
accumulation through the shared 40-layer FP8 transformer, not an unexplained
endpoint, mask, VAE-input, concatenation, or sampling mismatch.

Before H.264, the complete raw RGB comparison has pixel MAE `0.032968`, RMSE
`0.120340`, PSNR `18.392 dB`, and cosine similarity `0.987657`. The endpoint
frames remain strongly anchored: first-frame MAE/RMSE are `0.003758/0.008384`
and last-frame MAE/RMSE are `0.004911/0.012219`. The unconstrained middle frame
has MAE `0.055210` and RMSE `0.169728`; qualitative inspection shows a
different but coherent stage of the same turn. This larger middle-trajectory
variance is accepted only because the deterministic FLF seams precede the
known shared FP8 residual quantitatively; visual plausibility is supplemental,
not the parity argument.

The Engine and Comfy artifacts are MP4/H.264 High profile, 8-bit `yuv420p`, TV
range, BT.709 primaries/matrix, sRGB transfer, 512x512, 16 fps, 81 frames, and
5.0625 seconds.

## Fresh performance and memory evidence

Fresh matching Comfy FLF results on the RTX 5080:

- cold: 51.253 seconds;
- five genuine same-process seed-only warm runs: 35.673, 34.487, 33.831,
  41.265, and 34.120 seconds;
- warm median: 34.487 seconds;
- process working set: 30,255,239,168 bytes idle, 32,526,929,920 bytes peak,
  2,271,690,752 bytes incremental;
- WDDM total-device GPU use: 10,416 MiB idle, 15,408 MiB peak, 4,992 MiB
  incremental.

Final Engine FLF results from one persistent unpolled session:

- cold: 87.609 seconds;
- five genuine same-process seed-only warm runs: 39.399, 33.455, 36.948,
  33.226, and 32.958 seconds;
- warm median: 33.455 seconds, 2.99% faster than matching Comfy and inside the
  approximate 10% objective;
- process working set: 26,731,171,840 bytes idle, 34,510,893,056 bytes peak,
  7,779,721,216 bytes incremental;
- WDDM total-device GPU use: 4,912 MiB idle, 15,804 MiB peak, 10,892 MiB
  incremental.

The separately polled Engine telemetry request took 39.282 seconds and is
excluded from the performance median. Engine's absolute working-set peak is
6.10% above Comfy and its total-device GPU peak is 2.57% above Comfy. Its much
lower idle residency makes the incremental figures larger; the equivalent
absolute peaks remain acceptable, so no FLF-specific memory or performance
optimization is indicated.

## Final Wan-local observation

Across T2V, I2V, and FLF, the checkpoint/LoRA ownership mechanics, UMT5 prompt
lifecycle, stochastic FP8 transformer execution, four-step 2+2 Euler sampling,
float32 high-to-low handoff, final VAE decode, and media write are now proven
identical. Image preprocessing and VAE encode are shared by the two image
operations, while FLF's joint gray-filled conditioning video, `+3` mask
topology, causal endpoint encode, and ordered-pair identity remain
operation-specific.

Content-derived reference retention is **confirmed and nuanced**: identical
bytes remain path-independent, but FLF proves that semantic role and ordering
must participate in identity and that the natural retained artifact may be one
joint ordered-pair encode. FLF naturally uses I2V's checkpoint and LoRA
substrate. This is evidence for a later architecture phase only; the three
operations are not consolidated here, and no broader reference abstraction is
yet justified.
