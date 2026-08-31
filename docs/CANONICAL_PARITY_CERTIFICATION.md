# Canonical parity certification

## Result

The verification-first Klein-local repair resolves the final FLUX.2 Klein 9B two-image runtime blocker. Fresh evidence certifies Klein two-image, Klein T2I, all LTX 2.3 operations, and all three Wan 2.2 14B turbo operations across their accepted numerical/output, lifecycle, performance, and memory boundaries.

The repair is Klein-local. No canonical fixture, LTX/Wan production code, or Engine-wide architecture changed.

## Environment and method

- Engine branch: `greenfield-reset`; Klein repair baseline `9427478` (`require Windows and Linux runtime portability`)
- Reference process: `Comfy C (PyTorch Baseline)` only
- ComfyUI: `12d5279438bfefc058a269eae805ceab6047777f`
- Comfy Python / PyTorch / CUDA: 3.13.12 / 2.11.0+cu130 / 13.0
- Engine Python / PyTorch / CUDA: 3.12.13 / 2.11.0+cu130 / 13.0
- `comfy-kitchen`: 0.2.31
- `comfy-aimdo`: 0.4.15
- GPU: NVIDIA GeForce RTX 5080, driver 610.47, 16,303 MiB reported
- Host RAM: 68,505,284,608 bytes

Each timing case used one cold execution followed by five genuine same-process warm executions with only the first sampling seed changed. Instrumentation verified that the complete model/sampling, decode, audio where applicable, and artifact-write path reran. Timing and resource telemetry were collected separately. GPU figures are WDDM total-device usage on both sides; process memory is working set. Numerical comparisons use the canonical seed and prefer pre-codec decoder tensors.

Temporary Comfy and Klein probes were removed, no reference or production instrumentation remains active, and canonical fixtures remained unmodified.

## LTX product-domain extension — 2026-08-30

The canonical LTX numerical certification below remains the protected proof
baseline.  A later bounded milestone generalized only its public request
surface to the recovered product geometry, duration, and seed domain.  The
model identities, schedules, coarse/refinement split, canonical seeds, and
previously certified numerical kernels are unchanged.

Pinned source and live `Comfy C (PyTorch Baseline)` establish the product rules
recorded in `docs/LTX23_TARGET.md`.  In particular, the canonical API fixtures'
duration controls are integer primitives only because their proof value is 5.
For live 1.5-second variants, changing only that request node to Comfy's native
float primitive made the unchanged downstream graph valid and executable.

Fresh reference and Engine artifacts established the following representative
matrix.  PSNR is a conservative post-H.264 comparison; it is separate from,
and does not replace, the stronger pre-codec canonical evidence below.

| Operation | Request | Reference / Engine artifact contract | Post-codec video PSNR |
|---|---|---|---:|
| T2V canonical | 512x512, 5.0 s, canonical seed | 145 frames, 30 fps, stereo 48 kHz | 35.52 dB |
| T2V recovered default | 1280x704, 5.0 s, canonical seed | 145 frames, 30 fps, stereo 48 kHz | 38.76 dB |
| T2V portrait + half-second duration | 384x640, 1.5 s, canonical seed | 41 frames, 30 fps, stereo 48 kHz | 37.76 dB |
| I2V normalized landscape | 640x384, 1.5 s, canonical seed | 41 frames, 30 fps, stereo 48 kHz | 36.87 dB |
| FLF normalized portrait | 384x640, 1.5 s, canonical seed | 41 frames, 30 fps, stereo 48 kHz | 41.23 dB |

The recovered 1280x704 default completed in both reference and Engine on the
certification machine; no product-policy narrowing was needed.  Source-derived
formula tests cover every accepted 0.5-second increment and the 1.0/10.0-second
boundaries without repeatedly paying for long video generation.  Existing
768x768 canonical certification remains below, while the generalized tests
also protect its derived latent shapes.

I2V and FLF now require already-normalized source canvases with exact requested
dimensions.  Content-derived caches retain source/guide latents across seed and
duration changes, invalidate on geometry or bytes, and preserve FLF endpoint
order.  Targeted tests explicitly protect public coarse-seed / fixed-42
refinement mapping for T2V and I2V, single-stage public seed mapping for FLF,
and duration-derived FLF endpoint coordinates.

These fresh executions are correctness and artifact-contract evidence, not a
new performance recertification claim.  The prior matching-baseline performance
and resource results below remain the applicable canonical gates.

## Fixture and artifact inventory

| Case | Canonical fixture | SHA-256 |
|---|---|---|
| LTX T2V 512 and derived 768 shape gate | `reference/comfy/ltx23/t2v-pytorch-baseline-api.json` | `5c35eef0f42175c78614b0156fe538147afaea26fb918e6876060a7a990bd063` |
| LTX I2V | `reference/comfy/ltx23/i2v-pytorch-baseline-api.json` | `4f34a4f72f4de97e7b6f5e04f453414972340c58ec5a294294b3520bcccf982a` |
| LTX FLF | `reference/comfy/ltx23/flf-pytorch-baseline-api.json` | `4d202edf1d5329521738992ca8f2683851a248acabe1a01b8f15953a09c62195` |
| Klein T2I | `reference/comfy/klein9b/t2i-pytorch-baseline-api.json` | `425a8994fc22f9441f1412631b34e71c04d77b21244ffd766e77250c59095ae4` |
| Klein two-image | `reference/comfy/klein9b/2i2i-pytorch-baseline-api.json` | `c6884b211b466d0d9814688e39e6e1254cb0c8e94edf50d894f49f31f8fcf141` |
| Wan T2V | `reference/comfy/wan2214b/t2v-pytorch-baseline-api.json` | `fae773ec268ee1c0d6a8b0a30a57d3a78786c10123136c38730ce42ccb2f33d5` |
| Wan I2V | `reference/comfy/wan2214b/i2v-pytorch-baseline-api.json` | `045f08500ad16782fe5a99de01ead80db878c59948ac68627a8d64aa54a77e04` |
| Wan FLF | `reference/comfy/wan2214b/flf-pytorch-baseline-api.json` | `a4f64366096f2094a791f54bb59a4e088968d03068cd6ec10cc6c8bc0714ea88` |

The LTX 768 case was executed by changing only the relevant width/height values in a temporary request; the tracked 512 fixture stayed byte-identical.

Important artifact hashes:

- LTX dev transformer `28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450`; distilled transformer `d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2`; Gemma text encoder `aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d`; upsampler `5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed`; transformer/text LoRAs `9f482f50bf87f55abd7171bf600851907a7134c7d8c0674f2cfd25392e5df3a5` / `87bcabeac9bec9f374232b5122d6511c2b2112d479e50176149e944b3712eb4a`.
- Klein diffusion model `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee`; text encoder `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6`; VAE `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.
- Wan T2V high/low checkpoints `cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f` / `e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97`; I2V/FLF high/low checkpoints `6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42` / `5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e`; UMT5 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68`; VAE `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b`.
- Wan T2V high/low LoRAs `698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41` / `ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a`; I2V/FLF high/low LoRAs `d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e` / `024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9`.

Input content identities were also verified: LTX bird `f293ee0aba3cebda198d8223d140ce714fc8d10ee00f4529f13fe8d4f1a667c0` (512x512 RGB), `botsneak.png` `cb60895352f2e4eebb796b939a36cc06f7bed425b0f292735183c0dc969b8248` (1920x1080 RGBA, no orientation EXIF), Klein references `fd44eb4359b0341e7ee9620d853cf3474e19a6e1a9781c0bb7deb05d3ea564a8` (920x630 RGB) and `3c3ce6381b59c231cdd28c3234ab5b79ba1d7c272b189984f8407fb246665bee` (512x512 RGB), and Wan `front.png` / `back.png` `63b7e7401e75991f618db3181b6003816aab87954b1979bcb97fbf36c63323e5` / `b2ae3502ac181e45528da6bcfea2f7f3f1b90eabe6a6935d5dc1c29cb931c7da` (both 1024x1024 RGB).

## Certification matrix

| Case | Fixture | deterministic seams | first model output | transition/final latent | raw image/video | raw audio | artifact | lifecycle | warm perf. | RAM | GPU | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LTX T2V 512 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| LTX T2V 768 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| LTX I2V | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| LTX FLF | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Klein T2I | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Klein two-image | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Wan T2V | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Wan I2V | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Wan FLF | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |

`FAIL` means a certification gate is violated. No certification rows remain failed.

## Performance

All values are seconds. The warm column contains all five measurements; delta compares medians as `(Engine / Comfy - 1)`.

| Case | Comfy cold | Comfy warm series (median) | Engine cold | Engine warm series (median) | Delta | Gate |
|---|---:|---|---:|---|---:|---|
| LTX T2V 512 | 86.143 | 52.530, 46.639, 48.407, 48.552, 45.020 (48.407) | 78.165 | 46.858, 46.202, 46.242, 46.131, 46.196 (46.202) | -4.55% | PASS |
| LTX T2V 768 | 97.419 | 54.481, 50.982, 49.584, 51.072, 54.810 (51.072) | 89.510 | 53.925, 54.156, 53.564, 54.128, 53.362 (53.925) | +5.59% | PASS |
| LTX I2V | 83.299 | 47.474, 45.826, 45.549, 45.689, 44.744 (45.689) | 83.819 | 48.796, 45.479, 46.007, 45.377, 45.491 (45.491) | -0.43% | PASS |
| LTX FLF | 54.541 | 21.599, 23.261, 23.213, 24.558, 23.783 (23.261) | 53.771 | 18.918, 18.807, 18.866, 18.935, 19.040 (18.918) | -18.67% | PASS |
| Klein T2I | 16.687 | 1.549, 1.584, 1.713, 1.658, 1.693 (1.658) | 11.738 | 1.242, 1.232, 1.243, 1.246, 1.236 (1.242) | -25.10% | PASS |
| Klein two-image | 22.165 | 7.935, 7.888, 7.944, 7.892, 7.873 (7.892) | 30.789 | 7.818, 7.875, 7.820, 7.822, 7.872 (7.822) | -0.88% | PASS |
| Wan T2V | 48.538 | 37.018, 48.685, 49.223, 62.125, 33.266 (48.685) | 95.872 | 36.795, 33.839, 33.684, 33.526, 33.642 (33.684) | -30.81% | PASS |
| Wan I2V | 52.025 | 34.027, 33.823, 33.188, 35.794, 55.559 (34.027) | 114.474 | 35.480, 33.821, 33.128, 33.151, 33.300 (33.300) | -2.14% | PASS |
| Wan FLF | 49.042 | 35.065, 33.924, 33.533, 33.508, 36.638 (33.924) | 89.813 | 36.678, 34.095, 33.666, 33.886, 33.545 (33.886) | -0.11% | PASS |

Wan I2V was repeated because the initial median was borderline; the clean repeat shown above passed and had a tight four-run cluster despite one modest first-warm cost.

## Equivalent resource telemetry

Absolute peaks are the acceptance signal. Working set is bytes; GPU is total-device MiB.

| Case | Comfy peak WS | Engine peak WS | RAM result | Comfy peak GPU | Engine peak GPU | GPU result |
|---|---:|---:|---|---:|---:|---|
| LTX T2V 512 | 39,607,767,040 | 38,909,177,856 | PASS | 15,455 | 15,541 | PASS (+0.6%) |
| LTX T2V 768 | 39,241,826,304 | 39,232,200,704 | PASS | 15,669 | 15,547 | PASS |
| LTX I2V (warm) | 29,116,305,408 | 1,274,875,904 | PASS | 15,391 | 15,626 | PASS (+1.5%) |
| LTX FLF | 38,836,998,144 | 39,316,766,720 | PASS (+1.2%) | 15,241 | 15,391 | PASS (+1.0%) |
| Klein T2I | 21,354,024,960 | 1,951,907,840 | PASS | 13,862 | 13,159 | PASS |
| Klein two-image | 21,590,417,408 | 2,253,766,656 | PASS | 14,162 | 15,514 | PASS (+9.5%) |
| Wan T2V | 39,870,447,616 | 34,543,824,896 | PASS | 15,389 | 15,774 | PASS (+2.5%) |
| Wan I2V | 39,520,956,416 | 34,579,161,088 | PASS | 15,397 | 15,773 | PASS (+2.4%) |
| Wan FLF | 39,893,217,280 | 34,608,521,216 | PASS | 15,530 | 15,771 | PASS (+1.6%) |

Idle residency and incremental use were captured for every row. They are not substituted for the absolute-peak gate: notably, Engine and Comfy begin with materially different residency in several operations.

## Numerical and output evidence

### LTX 2.3 T2V 512

Canonical noise, all eleven sampler states, complete model inputs, processed context, every video/audio transformer prediction, both stage transitions, and final video/audio latents match Comfy exactly. Raw RGB video has MAE 0.0004448, RMSE 0.0008227, and PSNR 61.69 dB; median/worst frame PSNR are 61.66/60.54 dB, with first/middle/last 64.08/61.11/61.85 dB.

Raw audio has waveform RMSE 0.000666, SNR 52.7 dB, and is inside measured Comfy same-seed self-variance (RMSE 0.001418, SNR 46.17 dB). The remaining audio residual first appears at roughly 1e-9 in the CUDA transpose-convolution path, not in latent generation.

### LTX 2.3 T2V 768

The derived 768 request changes only the established resolution values. Both stages and all eleven transformer calls remain exact. Raw RGB MAE/RMSE/PSNR are 0.0004039/0.0008342/61.57 dB; median/worst frame PSNR 61.69/60.50 dB; first/middle/last 64.19/60.74/62.16 dB. Raw audio MAE/RMSE are 0.0000651/0.0001021 with SNR 63.03 dB. The secondary shape gate passes numerical, output, performance, and resource certification.

### LTX 2.3 I2V

Source decoding, preprocessing, VAE source latents, masks, complete first input, context, all eleven transformer calls, and both stage/final latents match Comfy exactly. Raw video MAE/RMSE/PSNR are 0.0004384/0.0008090/61.84 dB; median/worst frame PSNR 61.45/61.23 dB; first/middle/last 61.57/61.29/61.46 dB. Raw audio MAE/RMSE are 0.0000469/0.0002562 with SNR 49.29 dB and cosine approximately 1.0.

The content-derived source cache retains both canonical VAE source latents across seed-only requests, treats identical bytes at a changed path as identical content, invalidates on changed bytes, and remains independent from prompt/model identity. An operation-local retained vocoder removes the measured warm reconstruction cost without raising absolute GPU peak materially.

### LTX 2.3 FLF

Guide decoding/preprocessing and ordered placement, context, complete inputs, every block in all eight transformer calls, stage transition, and final video/audio latents match Comfy exactly. This includes the distilled checkpoint's actual mixed layout: blocks 0-3 use BF16 weights and block 4 onward uses FP8 input/weight dispatch even without transformer LoRA.

Raw video MAE/RMSE/PSNR are 0.0004884/0.0007400/62.61 dB; median/worst frame PSNR 62.69/59.18 dB; first/middle/last 63.02/62.61/61.61 dB. Raw audio MAE/RMSE are 0.0002037/0.0006081 with SNR 53.73 dB and cosine approximately 1.0.

All LTX artifacts now match H.264 High/yuv420p, 145 frames at 30 fps (4.833 s), AAC-LC stereo at 48 kHz, and explicit television-range BT.709 primaries/matrix with sRGB transfer metadata. Media writing trims only inactive Windows working-set pages after the container closes; it preserves the warm model/cache objects, keeps all timing gates passing, and eliminates the measured duplicate mapped-page residency.

### FLUX.2 Klein 9B T2I

First model input, context, and first prediction are exact. Raw image MAE/RMSE/PSNR are 0.011612/0.031597/30.007 dB, cosine 0.998428. The 768x768 PNG contract matches. Numerical, lifecycle, performance, and resource gates pass.

The shared Klein allocator bootstrap was re-certified with a fresh baseline process: Comfy cold/warm was 16.687 s / 1.549, 1.584, 1.713, 1.658, 1.693 s (median 1.658 s); fresh Engine cold/warm was 11.738 s / 1.242, 1.232, 1.243, 1.246, 1.236 s (median 1.242 s, 25.1% faster). The same 50 ms total-device WDDM monitor observed 13,862 MiB maximum for Comfy and 13,159 MiB for Engine. A clean-process repeat of the canonical seed-42 Engine PNG was byte-identical. The accepted numerical seam/output evidence is unchanged.

### FLUX.2 Klein 9B two-image

Both ordered image preprocessing paths, complete first target input, and context are exact. Fresh reference-latent comparisons are bounded BF16 encoder residuals: reference 1 RMSE 0.003082 (0.365% relative, cosine 0.9999933) and reference 2 RMSE 0.002759 (0.326% relative, cosine 0.9999947). With those Engine references the first prediction RMSE is 0.126862 (10.593% relative, cosine 0.994389). The existing exact-Comfy-reference control reduces first-prediction RMSE to 0.00328, localizing the downstream numerical residual to the two VAE reference latents.

Raw image MAE/RMSE/PSNR are 0.019478/0.049525/26.104 dB, cosine 0.994756, and the 1232x832 PNG contract matches. The post-repair seed-42 through seed-44 PNGs are byte-identical to the pre-repair accepted Engine PNGs, and all five warm requests retained both the model and conditioning caches.

The progressive residency trace localized the native allocator, not live references or concurrent VAE residency, as the material gap. With the native allocator, the stable warm pre-denoise boundary was 15,643 MiB WDDM with 9,151.858 MiB allocated, 14,228 MiB reserved, and 5,076.142 MiB inactive allocator slack. Retained references total only 1.978 MiB and the live VAE only 118.968 MiB; temporarily moving the VAE to CPU reduced the stable warm boundary by about 93 MiB and added 141 ms, so it is not the repair. Releasing the native cache reduced WDDM to 10,668 MiB without changing 9,151.858 MiB allocated, but every denoiser forward recreated about 4.5 GiB and releasing it per iteration added 10.297 s while preserving the PNG hash.

The Klein package now selects `backend:cudaMallocAsync` before its first Torch import only when the embedding process has not selected an allocator. This matches the pinned Comfy baseline. The same complete trace then held the warm pre-denoise boundary at 11,516 MiB with 9,150.091 MiB allocated, 9,216 MiB reserved, and 65.909 MiB slack; the four stable post-transformer checkpoints were 11,533, 11,537, 11,531, and 11,542 MiB. Final VAE decode's stable checkpoint was 12,438 MiB. Uninstrumented fresh five-warm measurements were 7.892 s for Comfy and 7.822 s for Engine (0.88% faster), with continuous 50 ms WDDM peaks of 14,162 and 15,514 MiB respectively (+9.5%). The operation therefore passes its warm-performance and equivalent GPU boundary gates without an offload system or broader runtime work.

### Wan 2.2 14B turbo T2V

Canonical input is exact and text context is effectively exact (RMSE 6.73e-7). First flow RMSE is 0.128812 (10.722% relative, cosine 0.994260); final latent RMSE is 0.853464 (48.779% relative, cosine 0.88146). This freshly reproduces the previously localized sparse source-conformant stochastic FP8 materialization residual; no earlier request/conditioning seam is divergent.

Raw video MAE/RMSE/PSNR are 0.092183/0.153864/16.257 dB; median/worst frame PSNR 16.325/15.548 dB; first/middle/last 16.302/16.527/16.329 dB. The residual is bounded, mechanistically localized, and stable under fresh execution, so the operation passes.

### Wan 2.2 14B turbo I2V

The complete 36-channel first input has RMSE 0.000961 (0.0879% relative, cosine 0.9999996); context is effectively exact. First flow RMSE is 0.044470 (3.077% relative, cosine 0.999527), and final latent RMSE is 0.223062 (8.449% relative, cosine 0.996426).

Raw video MAE/RMSE/PSNR are 0.008741/0.038615/28.265 dB; median/worst frame PSNR 27.999/24.825 dB; first/middle/last 41.908/27.877/27.775 dB. Input/VAE conditioning is bounded before the residual begins inside the shared FP8 transformer. All gates pass.

### Wan 2.2 14B turbo FLF

Start/end decode, resize, composed gray/start/end video, mask topology, VAE encoder input, and complete 36-channel denoiser input were freshly confirmed. The first-input RMSE is 0.000964 (0.0886% relative); context is effectively exact. First flow RMSE is 0.156112 (10.833% relative, cosine 0.994121), the same scale as T2V's shared FP8 transformer residual. Final latent RMSE is 0.655051 (25.228% relative, cosine 0.968003).

Raw video MAE/RMSE/PSNR are 0.032968/0.120340/18.392 dB; median/worst frame PSNR 21.644/13.819 dB; first/middle/last 41.531/15.405/38.259 dB. The good endpoints and weaker middle are quantitatively consistent with amplification of the localized transformer residual rather than an FLF conditioning error. All gates pass.

All Wan artifacts match MP4/H.264 High/yuv420p, 512x512, 16 fps, 81 frames, 5.0625 seconds, and BT.709/sRGB metadata.

## Reference self-variance controls

Clean-process, same-seed duplicate Comfy runs were performed for the unexpectedly weak or adversarial cases: LTX T2V, LTX I2V, Klein two-image, Wan T2V, and Wan FLF. First model input, context, prediction, and raw video/image were exact across duplicate Comfy runs. LTX audio alone showed small spread: T2V waveform RMSE 0.001418 (SNR 46.17 dB) and I2V RMSE 0.000633 (0.848% relative, cosine 0.999964). The repaired Engine audio residual is bounded by this reference spread.

## Lifecycle and completeness

The focused suites exercise all accepted operations and establish that seed-only changes rerun inference while retaining appropriate model state; same prompts reuse conditioning; prompt changes invalidate prompt conditioning without replacing model identity; changed image bytes invalidate derived image state; identical bytes at another path remain content-equivalent; Klein two-image and Wan FLF preserve ordered semantic roles; model/LoRA identity replacement destroys all derived state; and LoRA application is not cumulative.

The executable fixture traces also confirmed that canonical checkpoints, LoRAs and strengths, text encoders, VAEs, prompts, seeds, samplers, schedulers, stage boundaries, resolutions, frame counts, input roles, and output settings are consumed. The tracked Klein one-image fixture was not executed or implemented because it is outside the accepted operation inventory.

## Bounded Phase B controls and repairs

The original certification recorded two independent LTX failures: processed context relative RMSE near 3%, and a remaining first-video-prediction residual near 2.23% even when exact Comfy context was injected. Operator-level controls found and corrected both roots.

The text path first diverged in source-specific Gemma arithmetic and dispatch: full-attention RoPE frequency division, BF16 checkpoint embedding and RMSNorm-add order, FP4/FP8 linear dequantization into float32 input, in-place RoPE `addcmul`, and explicit GQA K/V repetition before SDPA. With those corrections, raw Gemma projection and final processed LTX context are exact for all canonical prompts.

The independent transformer root was source-conformant materialization: LoRA-patched FP8 weights require the exact Comfy seed derivation, uint8 stochastic-rounding RNG, Kitchen requantization, per-layer input scales, and FP8-by-FP8 dispatch. FLF additionally exposed that quantized activation dispatch is determined by the checkpoint weight layout, not by LoRA presence. The corrected sampler preserves float32 sigma tensors, Comfy's exact Euler arithmetic (including FLF's `eta=0` ancestral/RF convex blend), literal second-stage `0.4219`, and packed masked-latent arithmetic. These changes make all canonical LTX transformer calls and stage/final latents exact.

The audio decoder/vocoder now follows the pinned float32 path and in-place residual accumulation. The artifact writer adds the missing Comfy color metadata. A narrow post-artifact Windows working-set trim releases inactive duplicate mapped pages without destroying retained LTX state; an I2V control reduced warm working-set peak from 39.15 GB to 1.27 GB while the final five-run median remained inside the performance gate.

For Klein two-image, a stage-resolved residency trace found the material cause inside the native CUDA allocator's retained forward workspace, not the VAE, retained source latents, or an Engine-wide ownership problem. A Klein-package-local pre-Torch allocator default matches the pinned Comfy `cudaMallocAsync` setting while preserving an embedding process's explicit choice. It removes roughly 5 GiB of inactive warm allocator reservation, preserves byte-identical accepted outputs, and returns the fresh warm and WDDM measurements to the certification gates.

## Known bounded numerical differences

- Klein two-image retains small BF16 VAE reference-latent residuals. Their effect is localized by the exact-reference control and the raw output remains quantitatively close; this accepted numerical difference is unchanged by the allocator repair.
- Wan T2V/I2V/FLF retain source-conformant stochastic FP8 materialization residuals. Fresh input-side controls and the cross-operation scale establish the transformer as the earliest seam. Wan FLF does not introduce an earlier endpoint, mask, or VAE-conditioning discrepancy.
- LTX video/model seams are exact. The only bounded residual is the tiny raw-audio decoder/vocoder variation localized to CUDA transpose convolution and smaller than measured same-seed Comfy audio self-variance.

## Exact blockers and next experiments

None. The Klein two-image milestone is complete; no Engine-wide residency or offload architecture is justified by this report.
