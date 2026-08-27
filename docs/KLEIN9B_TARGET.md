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
defeat Comfy graph caching. The pinned fixture completed cold at seed 42 in
`6.861 s`; warm seeds 43, 44, and 45 took `1.267 s`, `1.268 s`, and `1.267 s`
(median `1.267 s`). A separate monitored seed-46 run peaked at `19.880 GiB`
process-tree RAM and `14.291 GiB` total GPU memory.

The corrected standalone Engine completed cold at seed 42 in `11.103 s`; retained
same-identity warm seeds 43, 44, and 45 took `1.180 s`, `1.180 s`, and `1.181 s`
(median `1.180 s`). This is `6.9%` faster than matching Comfy and passes the existing
no-more-than-10%-slower objective. A separate monitored seed-46 cold run peaked at
`10.030 GiB` process-tree RAM and `11.525 GiB` incremental GPU memory (`14.113 GiB`
total from a `2.588 GiB` idle baseline).

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

Fresh pinned Comfy evidence completed cold at seed 42 in `19.64 s`; seed-only warm
runs 43, 44, and 45 took `7.55 s`, `7.39 s`, and `7.40 s` (median `7.40 s`). A
separate monitored seed-46 run peaked at `20.099 GiB` process-tree RAM and
`14.162 GiB` total GPU memory. The Engine completed cold at seed 42 in `17.806 s`;
same-identity warm seeds 43, 44, and 45 took `7.844 s`, `7.818 s`, and `7.757 s`
(median `7.818 s`). This is `5.6%` slower than matching Comfy and passes the
existing no-more-than-10%-slower objective. Its separately monitored seed-46 cold
run peaked at `10.140 GiB` process-tree RAM and `13.826 GiB` incremental GPU memory
(`15.304 GiB` total from a `1.478 GiB` idle baseline).

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
