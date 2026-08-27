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

## Running the canonical path

Use `python -m latentslate_engine.klein9b` with explicit `--diffusion`,
`--text-encoder`, `--vae`, `--tokenizer`, `--prompt`, one or more `--seed` values,
and `--output`. Multiple seeds in one process exercise retained model and
conditioning state. No service or ComfyUI process is required.

This proof stops at the canonical distilled T2I operation. Klein base variants,
image editing, reference images, other resolutions, broader provider APIs, and
cross-family consolidation remain outside this milestone.
