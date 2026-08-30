# Canonical parity certification

## Result

This verification-first campaign does **not** certify the complete accepted-operation inventory. Fresh evidence certifies FLUX.2 Klein 9B T2I and all three Wan 2.2 14B turbo operations. LTX 2.3 T2V/I2V retain deterministic numerical and output failures, LTX FLF remains affected by the same unresolved deterministic substrate, and Klein two-image fails the performance and absolute GPU-peak gates.

No production implementation, canonical fixture, family architecture, or family target document was changed. Bounded controls localized the failures but did not establish a sufficiently narrow source-conformant repair. The campaign therefore stops at the requested blocker boundary.

## Environment and method

- Engine branch/start: `greenfield-reset` at `8417c1afda5470a83903f9567354954bda79bdc2`
- Reference process: `Comfy C (PyTorch Baseline)` only
- ComfyUI: `12d5279438bfefc058a269eae805ceab6047777f`
- Comfy Python / PyTorch / CUDA: 3.13.12 / 2.11.0+cu130 / 13.0
- Engine Python / PyTorch / CUDA: 3.12.13 / 2.11.0+cu130 / 13.0
- `comfy-kitchen`: 0.2.31
- `comfy-aimdo`: 0.4.15
- GPU: NVIDIA GeForce RTX 5080, driver 610.47, 16,303 MiB reported
- Host RAM: 68,505,284,608 bytes

Each timing case used one cold execution followed by five genuine same-process warm executions with only the first sampling seed changed. Instrumentation verified that the complete model/sampling, decode, audio where applicable, and artifact-write path reran. Timing and resource telemetry were collected separately. GPU figures are WDDM total-device usage on both sides; process memory is working set. Numerical comparisons use the canonical seed and prefer pre-codec decoder tensors.

Temporary Comfy probes were removed and the two instrumented source files were restored byte-for-byte. Canonical fixtures remained unmodified.

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
| LTX T2V 512 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | PASS | PASS | **FAIL** |
| LTX T2V 768 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | PASS | PASS | **FAIL** |
| LTX I2V | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | PASS | PASS | **FAIL** |
| LTX FLF | PASS | NEEDS EXPLANATION | NEEDS EXPLANATION | NEEDS EXPLANATION | NEEDS EXPLANATION | NEEDS EXPLANATION | FAIL | PASS | PASS | PASS | PASS | **NEEDS EXPLANATION** |
| Klein T2I | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Klein two-image | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| Wan T2V | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Wan I2V | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Wan FLF | PASS | PASS | PASS | PASS | PASS | n/a | PASS | PASS | PASS | PASS | PASS | **PASS** |

`FAIL` means a certification gate is violated. `NEEDS EXPLANATION` means the case cannot be certified while a deterministic shared-family residual remains unresolved, even though its final metrics are substantially stronger than the failing LTX cases.

## Performance

All values are seconds. The warm column contains all five measurements; delta compares medians as `(Engine / Comfy - 1)`.

| Case | Comfy cold | Comfy warm series (median) | Engine cold | Engine warm series (median) | Delta | Gate |
|---|---:|---|---:|---|---:|---|
| LTX T2V 512 | 80.851 | 45.793, 39.860, 38.792, 38.767, 38.326 (38.792) | 76.146 | 45.799, 51.544, 45.696, 47.029, 45.836 (45.836) | +18.16% | FAIL |
| LTX T2V 768 | 84.922 | 48.095, 46.284, 43.466, 52.379, 45.972 (46.284) | 79.138 | 63.882, 60.444, 62.352, 61.601, 60.550 (61.601) | +33.09% | FAIL |
| LTX I2V | 85.330 | 48.554, 40.163, 39.840, 40.249, 39.839 (40.163) | 92.559 | 48.928, 47.704, 47.146, 45.755, 45.154 (47.146) | +17.39% | FAIL |
| LTX FLF | 52.086 | 23.951, 23.942, 25.946, 30.292, 26.728 (25.946) | 48.831 | 23.413, 23.617, 23.651, 23.581, 23.631 (23.617) | -8.98% | PASS |
| Klein T2I | 13.527 | 1.377, 1.360, 1.321, 1.344, 1.422 (1.360) | 24.388 | 1.228, 1.227, 1.231, 1.228, 1.224 (1.228) | -9.66% | PASS |
| Klein two-image | 15.982 | 7.437, 7.577, 7.412, 7.589, 7.405 (7.437) | 21.748 | 12.167, 11.783, 12.168, 11.765, 12.321 (12.167) | +63.60% | FAIL |
| Wan T2V | 48.538 | 37.018, 48.685, 49.223, 62.125, 33.266 (48.685) | 95.872 | 36.795, 33.839, 33.684, 33.526, 33.642 (33.684) | -30.81% | PASS |
| Wan I2V | 52.025 | 34.027, 33.823, 33.188, 35.794, 55.559 (34.027) | 114.474 | 35.480, 33.821, 33.128, 33.151, 33.300 (33.300) | -2.14% | PASS |
| Wan FLF | 49.042 | 35.065, 33.924, 33.533, 33.508, 36.638 (33.924) | 89.813 | 36.678, 34.095, 33.666, 33.886, 33.545 (33.886) | -0.11% | PASS |

The Klein two-image series was repeated in a clean persistent session and became slower (25.450 s median), confirming instability rather than supplying a passing alternative. The first complete five-run series remains the conservative comparison authority. Wan I2V was likewise repeated because the initial median was borderline; the clean repeat shown above passed and had a tight four-run cluster despite one modest first-warm cost.

## Equivalent resource telemetry

Absolute peaks are the acceptance signal. Working set is bytes; GPU is total-device MiB.

| Case | Comfy peak WS | Engine peak WS | RAM result | Comfy peak GPU | Engine peak GPU | GPU result |
|---|---:|---:|---|---:|---:|---|
| LTX T2V 512 | 39,230,210,048 | 39,544,373,248 | PASS | 15,301 | 15,241 | PASS |
| LTX T2V 768 | 38,156,926,976 | 39,564,304,384 | PASS | 15,345 | 15,278 | PASS |
| LTX I2V | 39,075,700,736 | 39,582,044,160 | PASS | 15,314 | 15,173 | PASS |
| LTX FLF | 38,580,326,400 | 39,084,576,768 | PASS | 15,306 | 11,254 | PASS |
| Klein T2I | 21,354,024,960 | 1,951,907,840 | PASS | 14,808 | 13,410 | PASS |
| Klein two-image | 21,590,417,408 | 2,253,766,656 | PASS | 13,768 | 15,695 | **FAIL (+14.0%)** |
| Wan T2V | 39,870,447,616 | 34,543,824,896 | PASS | 15,389 | 15,774 | PASS (+2.5%) |
| Wan I2V | 39,520,956,416 | 34,579,161,088 | PASS | 15,397 | 15,773 | PASS (+2.4%) |
| Wan FLF | 39,893,217,280 | 34,608,521,216 | PASS | 15,530 | 15,771 | PASS (+1.6%) |

Idle residency and incremental use were captured for every row. They are not substituted for the absolute-peak gate: notably, Engine and Comfy begin with materially different residency in several operations.

## Numerical and output evidence

### LTX 2.3 T2V 512

The complete first model input is exact, but processed text context is not: RMSE 0.029986, relative RMSE 2.999%, cosine 0.99955. First prediction video RMSE is 0.042454 (3.925% relative, cosine 0.99923); first prediction audio RMSE is 0.022574 (1.423% relative). The residual amplifies through sampling: first-stage video is 51.21% relative, final video latent 196.43%, and final audio latent 280.83%.

Raw RGB video: MAE 0.233970, RMSE 0.306806, PSNR 10.263 dB; median/worst frame PSNR 10.344/9.134 dB; first/middle/last 9.572/10.083/11.530 dB. Raw audio: waveform RMSE 0.572695, cosine 0.01129, SNR -5.955 dB. This is a numerical and output failure, not codec noise.

### LTX 2.3 T2V 768

The shape-specific request preserves exact first model inputs but reproduces the same text/transformer chain. First video prediction relative RMSE is 3.837%; stage and final video latent residuals are 68.49% and 178.64%. Raw RGB MAE/RMSE/PSNR are 0.174515/0.281301/11.017 dB; median/worst frame PSNR 12.830/7.820 dB; first/middle/last 7.910/13.299/12.817 dB. Raw audio RMSE is 0.458578, cosine 0.06619, SNR -10.020 dB. The secondary shape gate fails numerical, output, audio, and performance certification.

### LTX 2.3 I2V

Source preprocessing and complete first input are exact; the timestep differs only 0.018% relative. Processed context RMSE is 0.040269 (4.027% relative). First video prediction RMSE is 0.237752 (19.846% relative, cosine 0.98013); first audio prediction is 2.397% relative. Final video/audio latent residuals are 60.13%/24.82%.

Raw video MAE/RMSE/PSNR are 0.083044/0.141086/17.010 dB; median/worst frame PSNR 16.613/15.439 dB; first/middle/last 49.063/17.148/15.439 dB. Raw audio RMSE is 0.076151, cosine 0.50965, SNR -0.175 dB. Correct retention of the conditioned first frame does not rescue the remaining sequence.

### LTX 2.3 FLF

First model input is exact. Context RMSE is 0.009987 (0.999% relative); first video prediction RMSE is 0.059556 (4.962% relative, cosine 0.99877), while first audio prediction is 1.322% relative. Final video/audio latent residuals are 31.446%/1.779%.

Raw video MAE/RMSE/PSNR are 0.013491/0.047738/26.423 dB; median/worst frame PSNR 38.602/11.212 dB; first/middle/last 48.194/38.678/29.030 dB. Raw audio RMSE is 0.055831, cosine 0.98214, SNR 14.468 dB. These are much stronger results, but the operation remains uncertified because the same deterministic text/transformer seam is unresolved and its worst-frame residual is material.

All LTX artifacts are H.264 High/yuv420p, 145 frames at 30 fps (4.833 s), with AAC-LC stereo at 48 kHz and matching duration. Comfy writes BT.709/sRGB color tags; current Engine LTX artifacts omit the explicit color-space/transfer/primaries tags. Artifact completeness therefore also fails.

### FLUX.2 Klein 9B T2I

First model input, context, and first prediction are exact. Raw image MAE/RMSE/PSNR are 0.011612/0.031597/30.007 dB, cosine 0.998428. The 768x768 PNG contract matches. Numerical, lifecycle, performance, and resource gates pass.

### FLUX.2 Klein 9B two-image

Both ordered image preprocessing paths, complete first target input, and context are exact. Fresh reference-latent comparisons are bounded BF16 encoder residuals: reference 1 RMSE 0.003082 (0.365% relative, cosine 0.9999933) and reference 2 RMSE 0.002759 (0.326% relative, cosine 0.9999947). With those Engine references the first prediction RMSE is 0.126862 (10.593% relative, cosine 0.994389). The existing exact-Comfy-reference control reduces first-prediction RMSE to 0.00328, localizing the downstream numerical residual to the two VAE reference latents.

Raw image MAE/RMSE/PSNR are 0.019478/0.049525/26.104 dB, cosine 0.994756, and the 1232x832 PNG contract matches. Correctness and lifecycle pass, but the operation is not certified because warm time is 63.60% slower and absolute GPU peak is 14.0% above Comfy.

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

Clean-process, same-seed duplicate Comfy runs were performed for the unexpectedly weak or adversarial cases: LTX T2V, LTX I2V, Klein two-image, Wan T2V, and Wan FLF. First model input, context, prediction, and raw video/image were exact across duplicate Comfy runs. LTX audio alone showed small spread: T2V waveform RMSE 0.001535 (0.532% relative, cosine 0.999986) and I2V RMSE 0.000633 (0.848% relative, cosine 0.999964). This is far below the Engine audio residuals. Reference nondeterminism does not explain any failed certification gate.

## Lifecycle and completeness

The focused suites exercise all accepted operations and establish that seed-only changes rerun inference while retaining appropriate model state; same prompts reuse conditioning; prompt changes invalidate prompt conditioning without replacing model identity; changed image bytes invalidate derived image state; identical bytes at another path remain content-equivalent; Klein two-image and Wan FLF preserve ordered semantic roles; model/LoRA identity replacement destroys all derived state; and LoRA application is not cumulative.

The executable fixture traces also confirmed that canonical checkpoints, LoRAs and strengths, text encoders, VAEs, prompts, seeds, samplers, schedulers, stage boundaries, resolutions, frame counts, input roles, and output settings are consumed. The tracked Klein one-image fixture was not executed or implemented because it is outside the accepted operation inventory.

## Bounded Phase B controls and repairs

No production repair was made.

For LTX, an exact-context control invoked the Engine transformer with Comfy's processed context. T2V first-video-prediction relative RMSE improved from 3.925% to 2.227%, proving that text conditioning is one cause, but did not become correct. The remaining transformer/materialization residual is separately deterministic; both residuals then amplify across eight video steps. A valid repair would need to reopen at least two accepted LTX substrates and re-certify all LTX operations, which is not a narrow demonstrated correction suitable for this campaign.

For Klein two-image, the exact-Comfy-reference control confirms the numerical path, while a second clean performance series became substantially slower rather than exposing a narrow one-time cost. The combined warm-time and absolute-GPU failure points to kernel/residency behavior, not an operation-value bug. Correcting it would require performance work broader than bounded parity repair.

## Known bounded numerical differences

- Klein two-image retains small BF16 VAE reference-latent residuals. Their effect is localized by the exact-reference control and the raw output remains quantitatively close; these are accepted numerically, but do not waive the performance/resource failure.
- Wan T2V/I2V/FLF retain source-conformant stochastic FP8 materialization residuals. Fresh input-side controls and the cross-operation scale establish the transformer as the earliest seam. Wan FLF does not introduce an earlier endpoint, mask, or VAE-conditioning discrepancy.
- LTX deviations are **not** listed as accepted bounded differences. They remain deterministic, amplify materially, and are certification blockers.

## Exact blockers and next experiments

1. **LTX shared deterministic substrate:** processed text conditioning is the first divergent seam, followed by a separate transformer/materialization residual even under exact Comfy context. The next experiment is an operator-level comparison of the text-conditioning transform followed by the earliest differing transformer operator using identical materialized weights. Any correction must then re-certify T2V 512/768, I2V, and FLF, including audio and color metadata.
2. **Klein two-image runtime:** the numerical operation passes, but warm execution is 63.60% slower and absolute GPU peak is 14.0% higher than Comfy. The next experiment is a stage-resolved timing and total-device residency trace around the two VAE reference encodes and four denoiser steps, without changing the accepted Klein T2I path.

Until those blockers are resolved, this campaign is not complete and no Engine-wide architecture or consolidation work is justified by this report.
