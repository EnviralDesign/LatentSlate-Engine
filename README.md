# LatentSlate Engine

Runtime includes `comfy-kitchen==0.2.28` (Apache-2.0) for restoring supported,
already-quantized Comfy tensor layouts. Engine does not quantize or convert model
weights at runtime.

LatentSlate Engine is the first-party, opinionated generation backend for
[LatentSlate](https://github.com/EnviralDesign/LatentSlate). It exposes a small
catalog of creative tools instead of a general node graph.

The first wedge intentionally stays simple:

- a versioned tool catalog that LatentSlate can render as native controls;
- HTTP upload/download transport that works unchanged on localhost, a LAN host,
  or a remote GPU instance;
- asynchronous jobs with progress, cancellation requests, and downloadable artifacts;
- a model-bundle registry with a CLI download seam;
- MiniMax-H3, LTX 2.3, and Wan 2.2 video tools backed by upstream Diffusers paths;
- FLUX.2 Klein 4B and 9B text-to-image and one-to-three-reference image-editing
  tools backed by the upstream Diffusers pipeline.

There are no custom H3 kernels, WanGP-derived runtime changes, or custom model
kernels in this initial implementation. Model runtimes are deliberately isolated
so optimization can happen later without changing the client protocol or tool
schemas.

## Current tools

| Tool key | LatentSlate intent | Inputs |
| --- | --- | --- |
| `h3.text_to_video` | Text to Video | prompt, quality, duration, seed |
| `h3.first_last_frame_video` | First/Last Frame Video | prompt, first frame, optional last frame, quality, duration, seed |
| `ltx23.text_to_video` | Text to Video | prompt, width, height, duration, seed |
| `wan22.text_to_video` | Text to Video | prompt, width, height, duration, seed |
| `flux2_klein4b.text_to_image` | Text to Image | prompt, width, height, seed |
| `flux2_klein4b.image_to_image` | Image to Image | prompt, one to three reference images, optional width/height, seed |
| `flux2_klein9b.text_to_image` | Text to Image | prompt, width, height, seed |
| `flux2_klein9b.image_to_image` | Image to Image | prompt, one to three reference images, optional width/height, seed |

The two additional video tools are intentionally narrow:

- LTX 2.3 uses the Diffusers-converted distilled checkpoint and its required
  eight-step fixed-sigma recipe. It produces synchronized audio, defaults to
  768×512 at 24 fps, and exposes a conservative 1–10 second duration range.
  Its current variant contract accepts only complete Diffusers folders annotated
  `precision = "bf16"` and `quantization = "native"`; selecting a resource
  changes the exact load-plan fingerprint. GGUF, FP8, INT8, and partial folders
  are rejected because LTX has no corresponding stored-quant loader.
- Wan 2.2 uses the official dense TI2V-5B Diffusers checkpoint in text-only mode,
  defaults to 1280×704 at 24 fps, and exposes a 1–5 second duration range.

Neither tool currently exposes negative prompts, guidance, steps, upscaling,
LoRAs, reference inputs, expert variants, or quantization choices. Those remain
runtime/recipe decisions until local testing establishes useful presets.

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

Both Klein variants use their distilled four-step checkpoints. Klein 4B Text to
Image defaults to 512×512 for fast iteration, matching the imported LatentSlate
Comfy workflows; Klein 9B retains a 1024×1024 default. Image to Image defaults to
the first source image's resolved canvas and accepts up to two additional
references. The curated tools expose granular width and height rather than a
preset list; Engine aligns explicit requests to each model family's canvas grid
and rejects aligned canvases above its safe pixel budget. Omit both I2I dimensions
to use the EXIF-oriented source canvas through the model's native 16-pixel
preprocessing floor, or provide both for an explicit output canvas.

## Model lifecycle

The Engine currently runs one generation worker and keeps at most one heavyweight
model runtime active. Tools that share a model variant reuse one pipeline.
Switching among H3, LTX 2.3, Wan 2.2, Klein 4B, and Klein 9B unloads the previous
heavyweight runtime before the next one loads. This prevents models from
accumulating in system RAM or VRAM.

This is intentionally a small first step toward ComfyUI-style model lifecycle
management, not a general scheduler or multi-GPU model manager.

## Install

LatentSlate Engine is intentionally not an opt-in collection of model extras. A
plain sync installs the complete runtime currently supported by this branch:

```bash
uv sync
```

The repository pins Python 3.12 and resolves PyTorch from the official CUDA 12.8
wheel index on Windows and Linux. This avoids the CPU-only PyTorch wheel that PyPI
provides on Windows while keeping one CUDA baseline for a local RTX 5080 and
Linux/Vast.ai backends.

After pulling this change over an environment that was created with Python 3.13
or CPU-only PyTorch, recreate the virtual environment once:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

```bash
rm -rf .venv
uv sync
```

Development/test tools remain optional:

```bash
uv sync --extra dev
```

When `.env` is missing, the bootstrap scripts create it from `.env.example`.
They preserve an existing `.env`, then run `uv sync` and the preflight:

```powershell
.\scripts\bootstrap.ps1
```

```bash
./scripts/bootstrap.sh
```

Before downloading large bundles or attempting the first GPU job, run the local
preflight directly when needed:

```bash
uv run latentslate-engine doctor
```

Use `--json` for automation or remote bootstrap scripts. The doctor reports
CUDA/GPU details, system RAM, disk space, package versions, current profiles,
Hugging Face authentication presence, and local bundle state without loading a
model or printing credentials. See [docs/DIAGNOSTICS.md](./docs/DIAGNOSTICS.md).

## Hugging Face authentication

Copy the example file and place a Hugging Face read token in `HF_TOKEN` after
accepting the terms for any gated repositories:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=hf_replace_me
```

The Engine loads this ignored local `.env` before importing Hugging Face tooling.
A real process/container environment variable always wins. For Docker, Vast.ai,
or another hosted backend, inject `HF_TOKEN` as a secret or environment variable
instead of copying `.env` into the image. `.env` is excluded from Git and Docker
build contexts.

Model downloads remain explicit because the combined weights are very large and
some repositories require accepted terms:

```bash
uv run latentslate-engine bundles install h3-basic
uv run latentslate-engine bundles install ltx23-basic
uv run latentslate-engine bundles install wan22-basic
uv run latentslate-engine bundles install klein4b-basic
uv run latentslate-engine bundles install klein9b-basic
```

LatentSlate owns the model library rather than relying on a user-global Hugging
Face cache. The default ignored store is initialized at `LatentSlateEngineData/` in
this repository:

```text
LatentSlateEngineData/
├── models/
│   ├── h3/
│   ├── klein4b/
│   ├── klein9b/
│   ├── ltx23/
│   ├── wan22/
│   └── custom/
├── loras/
│   ├── h3/
│   ├── klein4b/
│   ├── klein9b/
│   ├── ltx23/
│   ├── wan22/
│   └── custom/
├── cache/
│   ├── huggingface/{hub,assets,xet}/
│   └── torch/
├── assets/
├── jobs/
├── logs/
└── temp/
```

Set `LATENTSLATE_ENGINE_HOME` in `.env` to move that entire tree to another
folder or drive. Relative values resolve from the repository. Hugging Face,
Diffusers, Transformers, and Torch cache paths are forced below this one root;
their user-global cache settings do not control Engine model storage.

```powershell
uv run latentslate-engine data path
uv run latentslate-engine data init
```

FLUX.2 Klein 9B is gated on Hugging Face and uses the FLUX non-commercial model
license. Accept its terms and authenticate Hugging Face before installing the
complete self-contained BF16 Diffusers bundle. Engine does not assemble partial
components or convert those weights during loading.

LatentSlate Engine V0 does not yet ship automated model input/output filters.
Usage must remain human-reviewed and comply with each model's license and
acceptable-use terms; do not expose this V0 endpoint as an unattended public
generation service.

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

## Hardware compatibility

The default install is CUDA-capable on Windows and Linux, but the creator-facing
tool catalog is not tied to Blackwell. Portable correctness paths remain the
baseline. Future hardware-specific artifact loaders—and future Sol-Attn,
SageAttention, fused kernels, or similar accelerators—must be detected and gated
inside the runtime with an actionable unavailable reason when the selected artifact
cannot run exactly as stored. Engine must never convert it or silently substitute a
different artifact. These paths must not create different tool IDs or project schemas.

The local RTX 5080 and suitable Blackwell Vast.ai instances can use supported
artifact loaders as they are validated. See
[docs/RUNTIME_COMPATIBILITY.md](./docs/RUNTIME_COMPATIBILITY.md).

## Important environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for gated/private model downloads |
| `LATENTSLATE_ENGINE_HOME` | repository-local `LatentSlateEngineData/` | Root for models, LoRAs, library caches, uploaded assets, and generated job artifacts |
| `LATENTSLATE_ENGINE_TOKEN` | unset | Optional bearer token required by every `/v1` route |
| `LATENTSLATE_H3_MODEL` | `MiniMaxAI/MiniMax-H3` | H3 Hugging Face repository; its bundle installs inside `models/h3/` |
| `LATENTSLATE_H3_PROFILE` | `bf16_auto_offload` | `bf16_auto_offload` |
| `LATENTSLATE_H3_DEVICE` | `cuda` | Torch device used by H3 |
| `LATENTSLATE_LTX23_MODEL` | `diffusers/LTX-2.3-Distilled-Diffusers` | Diffusers-converted distilled LTX 2.3 repository; its bundle installs inside `models/ltx23/` |
| `LATENTSLATE_LTX23_PROFILE` | `bf16_sequential_offload` | `bf16_sequential_offload`, `bf16_model_offload`, or `bf16_cuda` |
| `LATENTSLATE_LTX23_DEVICE` | `cuda` | Torch device used by LTX 2.3 |
| `LATENTSLATE_WAN22_MODEL` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Wan 2.2 dense TI2V-5B repository |
| `LATENTSLATE_WAN22_PROFILE` | `bf16_sequential_offload` | `bf16_sequential_offload`, `bf16_model_offload`, or `bf16_cuda` |
| `LATENTSLATE_WAN22_DEVICE` | `cuda` | Torch device used by Wan 2.2 |
| `LATENTSLATE_KLEIN4B_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | Klein 4B Diffusers repository |
| `LATENTSLATE_KLEIN4B_PROFILE` | `bf16_model_offload` | `bf16_model_offload` or `bf16_cuda` |
| `LATENTSLATE_KLEIN4B_DEVICE` | `cuda` | Torch device used by Klein 4B |
| `LATENTSLATE_KLEIN_MODEL` | `black-forest-labs/FLUX.2-klein-9B` | Complete Klein 9B Diffusers repository |
| `LATENTSLATE_KLEIN_PROFILE` | `bf16_model_offload` | `bf16_model_offload` or `bf16_cuda` |
| `LATENTSLATE_KLEIN_DEVICE` | `cuda` | Torch device used by Klein 9B |

LatentSlate Engine never quantizes or converts model weights at runtime. A model's
stored precision and quantization are properties of the artifact in
`LatentSlateEngineData/models`; variants select only compatible artifacts and a
family must advertise a proven loader before a quantized artifact becomes available.

LTX 2.3 and Wan 2.2 both default to sequential CPU offload because their initial
paths are not expected to remain resident on a 16 GB GPU. The LTX recipe is the
converted distilled eight-step checkpoint; Wan remains the official dense 5B
checkpoint. These are correctness-first integrations, not claims of acceptable
local speed or memory use. The optional model-offload and CUDA-resident profiles
are available for larger remote backends and future benchmarking.

Klein 4B defaults to the official BF16 pipeline with model CPU offload. It is the
least speculative local image path and the first one to validate on the RTX 5080.
The optional `bf16_cuda` profile is intended for a GPU with enough VRAM to keep
the complete pipeline resident. A file-dropped Klein 4B transformer in the exact
official/BFL Comfy-native stored-FP8 layout is also supported through variants. The
Engine restores its FP8 bytes and scales directly into the matching Diffusers shell,
keeps dense text/VAE components separately offloaded, and owns transformer CUDA
residency without invoking a quantizer or Diffusers transformer offload hooks.

Klein 9B currently requires a complete BF16 Diffusers repository. Pre-quantized
Klein artifacts remain unavailable until their artifact metadata and exact loaders
are implemented and verified.

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

The Engine applies schema defaults at submission and rejects unknown fields,
missing required inputs, incorrect scalar types, invalid choices, out-of-range
numbers, malformed media references, and missing uploaded assets before a job
enters the GPU queue.

Job records are currently held in memory. Uploaded inputs and completed artifacts
are stored under `LATENTSLATE_ENGINE_HOME`, but restarting the service clears the
pollable job catalog. Durable job recovery and cleanup policy are later runtime
work and do not require a protocol redesign.

## Validation boundary

The protocol, catalog, upload/download flow, schema mismatch handling, frame
alignment, tool contracts, runtime eviction, packaging, and Python source
compilation are covered by lightweight CI on Python 3.11 and 3.12. CI does not
download any model or execute GPU inference.

Full H3 and LTX 2.3 inference still require hardware validation
on the target RTX 5080 / 64 GB workstation or an appropriately sized remote GPU.
Native Wan 14B I2V and Klein 4B stored-FP8 text/image generation have been exercised
through the normal API on that workstation. Family adapters use Comfy-native stored
formats and execution lessons where proven, while reusing compatible Diffusers model
shells and orchestration components instead of copying ComfyUI itself.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The tests use lightweight fakes and do not download or execute the model weights.
