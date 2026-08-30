# FLUX.2 Klein 9B canonical target

## Proven boundary

The second greenfield model family is the canonical FLUX.2 Klein 9B distilled
text-to-image recipe in
`reference/comfy/klein9b/t2i-pytorch-baseline-api.json`:

- 768x768 RGB PNG, batch 1
- prompt from the tracked fixture, empty negative prompt, seed 42
- four distilled Euler steps, CFG 1
- `flux2/flux-2-klein-9b-fp8.safetensors`
- `flux2/qwen_3_8b_fp8mixed.safetensors`
- `flux2/full_encoder_small_decoder.safetensors`

The implementation is isolated under `src/latentslate_engine/klein9b/`. It contains
only the concrete transformer, text-conditioning, VAE decode, scheduler, lifecycle,
and PNG output path required by this target. It does not import ComfyUI or share
runtime architecture with the frozen LTX family.

The model identity is the resolved diffusion, text-encoder, and VAE artifacts;
the resolved tokenizer path and the consumed `vocab.json`, `merges.txt`,
`tokenizer_config.json`, `special_tokens_map.json`, and `added_tokens.json`
artifacts; the consumed `text_encoder/config.json`; and the fixed recipe revision.
Repeating that identity retains the diffusion model, VAE, and same-prompt
conditioning. Any identity change calls the destructive release path before the
replacement identity is accepted.

## Reference and acceptance evidence

Reference behavior was measured from the exact live process named
`Comfy C (PyTorch Baseline)`, pinned at ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f`. Direct source-parity checks established:

- the four-step 768x768 Flux2 schedule is exactly
  `[1.0, 0.9622337222, 0.8946577907, 0.7389686704, 0.0]`;
- all 512 canonical prompt token IDs match, including 49 non-padding IDs (SHA-256
  `1f9aea661b39005924d7b850ab92631a7494d62369f500f6fb629c9172161f3c`);
- the selected Qwen layers 9, 18, and 27 reshape to `[1, 512, 12288]` and the
  Engine tensor is bit-identical to the pinned Comfy tensor;
- at canonical seed 42 and sigma 1, the Engine denoiser prediction matches pinned
  Comfy with MAE `3.39e-9`, RMSE `1.60e-8`, and maximum error `2.38e-7`;
- the final canonical RGB PNG compares at `29.99 dB` PSNR and `2.97` pixel MAE.
  Visual inspection shows the same composition, lighting, palette, and detail;
  later BF16 steps amplify sub-ULP differences even though each prediction on the
  same reference latent remains within `5.6e-8` RMSE.

Fresh performance runs changed only `RandomNoise.noise_seed` between outputs to
defeat Comfy graph caching. The newly started pinned baseline completed cold at seed
42 in `16.687 s`; warm seeds 43 through 47 took `1.549 s`, `1.584 s`, `1.713 s`,
`1.658 s`, and `1.693 s` (median `1.658 s`). Fresh standalone Engine completed cold
at seed 42 in `11.738 s`; retained same-identity warm seeds 43 through 47 took
`1.242 s`, `1.232 s`, `1.243 s`, `1.246 s`, and `1.236 s` (median `1.242 s`). This
is `25.1%` faster than matching Comfy. The same 50 ms total-device WDDM monitor
observed 13,862 MiB maximum for Comfy and 13,159 MiB for Engine. A clean-process
repeat of the Engine canonical seed-42 PNG was byte-identical; the accepted
numerical/output evidence remains unchanged.

These timings were recorded on the local RTX 5080 with PyTorch 2.11.0+cu130,
comfy-kitchen 0.2.31, and the fixture assets resolved from the configured ComfyUI
model paths. Cold timing includes checkpoint loading and text conditioning. Warm
timing includes noise creation, four sampling steps, VAE decode, and PNG writing.
Memory telemetry was collected separately because polling perturbs wall timing.

## Canonical two-image operation

The next proven operation is the exact Comfy API prompt in
`reference/comfy/klein9b/2i2i-pytorch-baseline-api.json` (SHA-256
`c6884b211b466d0d9814688e39e6e1254cb0c8e94edf50d894f49f31f8fcf141`). It
uses the same diffusion, Qwen text encoder, VAE, four-step Euler sampler, CFG 1,
and model identity as the accepted T2I operation. Its request is:

- prompt `the person from image 1 and the person from image 2 sitting at a table
  drinking coffee`, empty zeroed negative conditioning, and canonical seed 42;
- image 1 `lev-single-front.png`, 920x630 RGB PNG, SHA-256
  `fd44eb4359b0341e7ee9620d853cf3474e19a6e1a9781c0bb7deb05d3ea564a8`;
- image 2 `FfEAJDXXkAAmhI_.png`, 512x512 RGB PNG, SHA-256
  `3c3ce6381b59c231cdd28c3234ab5b79ba1d7c272b189984f8407fb246665bee`.

Image 1 is nearest-exact scaled to 1237x847 at one megapixel. That uncropped size
selects the 4093-token Flux2 schedule
`[0.9999999404, 0.9673759937, 0.9081227183, 0.7671545148, 0.0]` and the empty
target latent floors to 77x52 tokens. VAE encoding applies the pinned centered
multiple-of-16 crop to 1232x832 and produces a `[1, 128, 52, 77]` reference.
Image 2 is independently Lanczos scaled to 1024x1024 with the pinned uint8 PIL
round-trip and produces `[1, 128, 64, 64]`. There is no additional crop, mask,
strength, guidance, or reference-method node.

The two `ReferenceLatent` chains append the images in that order to both positive
and zeroed-negative conditioning. The detected Klein model's default `index`
method concatenates target, image 1, and image 2 before the single FP8 image-input
projection. Their four-axis position IDs distinguish target index 0, first
reference index 10, and second reference index 20; each image has independent
zero-origin row and column coordinates. Text retains the already-proven 512-token
Klein Qwen path and uses axis-3 positions 0 through 511.

Direct pinned-source checkpoints established that both cropped BF16 VAE input
tensors, the target noise latent after initial sigma scaling, the complete Qwen
context, all image/text position IDs, and the first timestep are exact. The
Diffusers and pinned Comfy BF16 VAE encoder implementations differ only at their
normal encoder arithmetic: posterior RMSE is `0.00517` and `0.00462`, yielding
normalized reference RMSE `0.00308` and `0.00276`. With the exact Comfy target,
context, and references, the Engine first denoiser prediction is within `0.00328`
RMSE and `0.03125` maximum error. The final seed-42 RGB output is 1232x832 and
compares at `26.09 dB` PSNR and `4.97` pixel MAE. Visual inspection confirms the
same two subjects, ordering, coffee-table composition, lighting, palette, and
major details. The remaining final-pixel difference is consistent with the
measured BF16 VAE-encoder seam rather than preprocessing, packing, text, schedule,
or positional geometry drift.

Fresh pinned Comfy evidence completed cold at seed 42 in `22.165 s`; seed-only warm
runs 43 through 47 took `7.935 s`, `7.888 s`, `7.944 s`, `7.892 s`, and `7.873 s`
(median `7.892 s`). Fresh Engine completed cold at seed 42 in `30.789 s`; retained
same-identity warm seeds 43 through 47 took `7.818 s`, `7.875 s`, `7.820 s`,
`7.822 s`, and `7.872 s` (median `7.822 s`). This is `0.88%` faster than matching
Comfy and passes the existing no-more-than-10%-slower objective. Continuous 50 ms
WDDM monitoring observed total-device peaks of 14,162 MiB for Comfy and 15,514 MiB
for Engine (+9.5%).

The material historical warm-residency gap was native CUDA allocator slack, not
retained source latents or concurrent VAE residency. The native trace had 9,151.858
MiB allocated versus 14,228 MiB reserved at the warm pre-denoise boundary; the
5,076.142 MiB difference was immediately released by the reversible cache control
and recreated after each transformer step. Per-step release preserved the PNG but
added 10.297 s, so it is not a production behavior. Klein instead sets the same
`cudaMallocAsync` allocator default used by the pinned Comfy baseline before its
first Torch import, unless the embedding process explicitly configured an allocator.
The matching warm boundary is 11,516 MiB WDDM with 9,150.091 MiB allocated,
9,216 MiB reserved, and only 65.909 MiB slack. Moving the 118.968 MiB live VAE to
CPU changed this boundary by only about 93 MiB and added 141 ms; it is deliberately
not retained as an offload path. Post-repair seed-42 through seed-44 PNGs are
byte-identical to their accepted pre-repair outputs, and all warm requests retain
both model and conditioning state.

Pinned Comfy retains the model objects and graph-cached text, scaled images, and
VAE references when only the seed changes. The Engine mirrors that relevant
lifetime with same-model residency, prompt-keyed text conditioning, and two
ordered content-hash-keyed reference slots. Changing either source invalidates
only its slot; swapping the sources invalidates both semantic slots; changing the
prompt invalidates text while retaining unchanged image references; and changing
any model/artifact identity destructively clears model, text, and reference state.

## Running the canonical path

Use `python -m latentslate_engine.klein9b` with explicit `--diffusion`,
`--text-encoder`, `--vae`, `--tokenizer`, `--prompt`, one or more `--seed` values,
and `--output`. Multiple seeds in one process exercise retained model and
conditioning state. No service or ComfyUI process is required.

Use `python -m latentslate_engine.klein9b.two_image` for the canonical two-image
path, adding explicit `--first-image` and `--second-image` inputs. Multiple
`--seed` values in one process exercise retained model, prompt, and ordered
reference state.

This proof stops after the canonical distilled T2I and exact two-image operation.
Other Klein image-editing workflows, base variants, other resolutions, broader
provider APIs, and cross-family consolidation remain outside this milestone.
