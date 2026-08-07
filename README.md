# LatentSlate Engine

LatentSlate Engine is the first-party, opinionated generation backend for
[LatentSlate](https://github.com/EnviralDesign/LatentSlate). It exposes a small
catalog of creative tools instead of a general node graph.

The first wedge intentionally stays simple:

- a versioned tool catalog that LatentSlate can render as native controls;
- HTTP upload/download transport that works unchanged on localhost, a LAN host,
  or a remote GPU instance;
- asynchronous jobs with progress, cancellation requests, and downloadable artifacts;
- a model-bundle registry with a CLI download seam;
- two MiniMax-H3 video tools backed by the upstream Diffusers modular pipeline;
- FLUX.2 Klein 9B text-to-image and image-to-image tools backed by the upstream
  Diffusers pipeline.

There are no custom H3 kernels, WanGP-derived runtime changes, or custom Klein
kernels in this initial implementation. Model runtimes are deliberately isolated
so optimization can happen later without changing the client protocol or tool
schemas.

## Current tools

| Tool key | LatentSlate intent | Inputs |
| --- | --- | --- |
| `h3.text_to_video` | Text to Video | prompt, quality, duration, seed |
| `h3.first_last_frame_video` | First/Last Frame Video | prompt, first frame, optional last frame, quality, duration, seed |
| `flux2_klein9b.text_to_image` | Text to Image | prompt, size, seed |
| `flux2_klein9b.image_to_image` | Image to Image | prompt, source image, size, seed |

The H3 quality choices are deliberately constrained to the consumer-memory canvas
used by the upstream 12–16 GB Diffusers recipe:

| Quality | Canvas | Steps |
| --- | --- | --- |
| Draft | 832×480 | 16 |
| Balanced | 960×544 | 20 |
| Final | 960×544 | 30 |

H3 frame counts are aligned to the model's temporal contract. The currently
advertised duration range is 5.0–14.375 seconds rather than a nominal 15 seconds,
because the next legal aligned frame count would cross H3's 15-second limit.

The initial Klein tools use the distilled four-step model. Text to Image defaults
to 1024×1024. Image to Image defaults to the source image's resolved canvas, with
explicit square, landscape, and portrait sizes also available.

## Model lifecycle

The Engine currently runs one generation worker and keeps at most one heavyweight
model runtime active. H3's two tools share one H3 pipeline, and Klein's two tools
share one Klein pipeline. Switching model families unloads the previous pipeline
before the next one loads so H3 and Klein do not accumulate in system RAM or VRAM.

This is intentionally a small first step toward ComfyUI-style model lifecycle
management, not a general scheduler or multi-GPU model manager.

## Install

The API and development tools:

```bash
uv sync --extra dev
```

Include both current model runtimes:

```bash
uv sync --extra dev --extra h3 --extra klein
```

Download the canonical H3 bundle into the Hugging Face cache:

```bash
uv run latentslate-engine bundles install h3-basic
```

FLUX.2 Klein 9B is gated on Hugging Face and uses the FLUX non-commercial model
license. Accept the terms for both BFL repositories and authenticate Hugging Face
before installing the consumer bundle:

```bash
uv run latentslate-engine bundles install klein9b-basic
```

The Klein consumer bundle downloads only the pipeline metadata/VAE, the official
BFL NVFP4 transformer, and the official Qwen3-8B FP8 encoder. It deliberately
avoids downloading the redundant BF16 transformer and text encoder.

LatentSlate Engine V0 does not yet ship automated Klein input/output filters.
Klein 9B usage must remain human-reviewed and comply with the FLUX Non-Commercial
License and Acceptable Use Policy; do not expose this V0 endpoint as an unattended
public image-generation service.

Run the server:

```bash
uv run latentslate-engine serve --host 127.0.0.1 --port 8765
```

For a LAN, Vast.ai, or other remote deployment, bind to `0.0.0.0`, set a bearer
token, and expose the port through a secure tunnel or HTTPS reverse proxy:

```bash
export LATENTSLATE_ENGINE_TOKEN='replace-me'
uv run latentslate-engine serve --host 0.0.0.0 --port 8765
```

LatentSlate sends media as multipart HTTP uploads and downloads generated
artifacts over HTTP. It never assumes that the desktop app and engine share a
filesystem.

## Important environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LATENTSLATE_ENGINE_HOME` | platform user data directory | Uploaded assets and generated job artifacts |
| `LATENTSLATE_ENGINE_TOKEN` | unset | Optional bearer token required by every `/v1` route |
| `LATENTSLATE_H3_MODEL` | `MiniMaxAI/MiniMax-H3` | H3 Hugging Face repository |
| `LATENTSLATE_H3_PROFILE` | `consumer_int8` | `consumer_int8` or `bf16_auto_offload` |
| `LATENTSLATE_H3_DEVICE` | `cuda` | Torch device used by H3 |
| `LATENTSLATE_KLEIN_MODEL` | `black-forest-labs/FLUX.2-klein-9B` | Klein pipeline/config repository |
| `LATENTSLATE_KLEIN_PROFILE` | `consumer_nvfp4` | `consumer_nvfp4`, `consumer_int8`, `bf16_model_offload`, or `bf16_cuda` |
| `LATENTSLATE_KLEIN_DEVICE` | `cuda` | Torch device used by Klein |
| `LATENTSLATE_KLEIN_TRANSFORMER_MODEL` | `black-forest-labs/FLUX.2-klein-9b-nvfp4` | Consumer transformer repository |
| `LATENTSLATE_KLEIN_TRANSFORMER_FILE` | `flux-2-klein-9b-nvfp4.safetensors` | Consumer transformer checkpoint |
| `LATENTSLATE_KLEIN_TEXT_ENCODER_MODEL` | `Qwen/Qwen3-8B-FP8` | Consumer text encoder repository |

The H3 `consumer_int8` profile follows the upstream Diffusers low-VRAM loading
recipe. It is a correctness-first starting point, not the final optimized runtime.
MiniMax-H3 is large; 64 GB system RAM may still be tight until lower-RAM
checkpoints and loaders are added.

Klein's default `consumer_nvfp4` profile composes the official BFL NVFP4
transformer with the official Qwen3-8B FP8 encoder and moves the text encoder,
transformer, and VAE through the configured accelerator one at a time. This is the
best-effort RTX 5080 path. NVIDIA ModelOpt support is most predictable on Linux;
WSL2 or a Linux/Vast.ai host is recommended for this profile.

`consumer_int8` is a TorchAO weight-only fallback that builds the transformer from
the BF16 repository at load time. It is slower to initialize and downloads the
full transformer, but avoids the ModelOpt dependency. The BF16 profiles are for
larger remote GPUs and should not be expected to fit a 16 GB card.

## Protocol

The initial endpoints are:

```text
GET    /v1/health
GET    /v1/catalog
GET    /v1/bundles
POST   /v1/assets
POST   /v1/jobs
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/artifacts/{artifact_id}
```

Tool IDs and input keys are stable machine identities. Labels and descriptions
may evolve. Every job carries the tool's schema revision and hash; stale clients
receive a structured `schema_mismatch` error instead of an implicit migration.

Job records are currently held in memory. Uploaded inputs and completed artifacts
are stored under `LATENTSLATE_ENGINE_HOME`, but restarting the service clears the
pollable job catalog. Durable job recovery and cleanup policy are later runtime
work and do not require a protocol redesign.

## Validation boundary

The protocol, catalog, upload/download flow, schema mismatch handling, H3 frame
alignment, Klein size contracts, runtime eviction, packaging, and Python source
compilation are covered by lightweight CI on Python 3.11 and 3.12. CI does not
download either model or execute GPU inference.

Full H3 and Klein inference still require hardware validation on the target RTX
5080 / 64 GB workstation. The Klein path is best effort and follows the official
BFL checkpoints plus upstream Diffusers integration rather than a ComfyUI- or
WanGP-specific implementation.

## Development

```bash
uv run pytest
uv run ruff check .
```

The tests use lightweight fakes and do not download or execute H3 or Klein.
