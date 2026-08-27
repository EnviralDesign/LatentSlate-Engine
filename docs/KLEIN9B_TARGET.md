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

The model identity is the resolved diffusion, text-encoder, VAE, and tokenizer
identity plus the fixed recipe revision. Repeating that identity retains the
diffusion model, VAE, and same-prompt conditioning. Any identity change calls the
destructive release path before the replacement identity is accepted.

## Reference and acceptance evidence

Reference behavior was measured from the exact live process named
`Comfy C (PyTorch Baseline)`, pinned at ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f`. The fixture completed cold in 6.762 s.
Three fresh warm reference runs measured 1.865 s, 1.474 s, and 1.545 s wall clock
(median 1.545 s). A monitored reference run peaked at 19.952 GiB process-tree RAM
and 14.070 GiB total GPU memory.

The standalone Engine path produced a valid 768x768 PNG with the requested vintage
motorcycle, retro diner, sunset, neon, and film-grain semantics. Its retained-state
warm runs measured 1.5310 s, 1.5312 s, and 1.5276 s (median 1.5310 s), about 0.9%
faster than the fresh reference median. A separately monitored cold Engine run took
12.070 s and peaked at 10.116 GiB process-tree RAM and 11.510 GiB incremental GPU
memory (13.263 GiB total device use from a 1.753 GiB idle baseline).

These timings were recorded on the local RTX 5080 with PyTorch 2.11.0+cu130,
comfy-kitchen 0.2.31, and the fixture assets resolved from the configured ComfyUI
model paths. Cold timing includes checkpoint loading and text conditioning. Warm
timing includes noise creation, four sampling steps, VAE decode, and PNG writing.

## Running the canonical path

Use `python -m latentslate_engine.klein9b` with explicit `--diffusion`,
`--text-encoder`, `--vae`, `--tokenizer`, `--prompt`, one or more `--seed` values,
and `--output`. Multiple seeds in one process exercise retained model and
conditioning state. No service or ComfyUI process is required.

This proof stops at the canonical distilled T2I operation. Klein base variants,
image editing, reference images, other resolutions, broader provider APIs, and
cross-family consolidation remain outside this milestone.
