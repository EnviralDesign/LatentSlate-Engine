# LatentSlate Engine

LatentSlate Engine is the first-party generation backend for
[LatentSlate](https://github.com/EnviralDesign/LatentSlate). It exposes a small,
versioned catalog of creator-facing tools over HTTP rather than a general node
graph. Jobs, uploads, progress, cancellation, and generated artifacts work the
same way on a local workstation or a remote GPU host.

Engine treats a **recipe** as the runnable product boundary. A recipe binds a
tool to fixed resources or explicit resource slots plus runtime policy. A
lockable deployment profile selects a lean set of recipes, resolves an exact
resource closure, and deduplicates shared files. The older bundle commands remain
a download compatibility seam; installing every family bundle is neither required
nor recommended.

Runtime includes `comfy-kitchen==0.2.28` (Apache-2.0) to restore supported
already-quantized Comfy tensor layouts. Engine never quantizes or converts model
weights at runtime.

## Quick start on Windows

From the Engine repository:

```powershell
git pull
.\scripts\bootstrap.ps1
uv run latentslate-engine data init
uv run latentslate-engine doctor
uv run latentslate-engine recipes validate
uv run latentslate-engine recipes list
uv run latentslate-engine deployments profiles
```

The bootstrap preserves an existing ignored `.env`, creates it from
`.env.example` when absent, installs the pinned Python/runtime environment, and
runs preflight checks. The repository pins Python 3.12, CUDA 12.8 PyTorch wheels
on Windows/Linux, Diffusers, Transformers, and Comfy Kitchen.

Choose a deployment profile before downloading anything large:

```powershell
uv run latentslate-engine deployments plan klein4b-image
uv run latentslate-engine deployments plan ltx23-video
uv run latentslate-engine deployments plan wan22-14b-i2v-fp8
uv run latentslate-engine deployments plan wan22-ti2v5b-text-to-video
```

The plan reports recipes, the deduplicated resource closure, total and incremental
bytes, required secrets, and whether the profile is locally runnable or remotely
provisionable. `deployments lock` emits reproducible JSON for automation, but is
not yet an installer:

```powershell
uv run latentslate-engine deployments lock klein4b-image |
  Set-Content -Encoding utf8 .\klein4b-image.lock.json
```

Install only the canonical resource needed by the profile you intend to run:

```powershell
uv run latentslate-engine bundles install klein4b-basic
uv run latentslate-engine bundles install ltx23-basic
uv run latentslate-engine bundles install wan22-basic
```

Important: `wan22-basic` is the dense **Wan 2.2 TI2V 5B** repository. It does
not install the native Wan 14B I2V recipe. The 14B runtime currently uses an
explicit five-resource recipe (high/low transformers, UMT5, VAE, and pipeline
support). The package-owned 14B recipe becomes runnable only after those exact
local artifacts are supplied. Its deployment plan is not an installer, and lock
generation intentionally remains unavailable: the filtered support directory is
not represented as an upstream whole-snapshot acquisition source.

These additional compatibility downloads exist, but are not part of the first
lean built-in profile set:

```powershell
uv run latentslate-engine bundles install h3-basic
uv run latentslate-engine bundles install klein9b-basic
```

Do not run every install command unless you deliberately want every repository;
the combined footprint is hundreds of gigabytes.

## What is ready now

Package-owned built-in recipes currently cover:

| Recipe key | Operation | Resource | Current status |
| --- | --- | --- | --- |
| `klein4b.distilled.text-to-image` | Text to Image | complete Klein 4B BF16 Diffusers folder | built-in |
| `klein4b.distilled.image-to-image` | Image to Image, one to three references | same shared Klein 4B folder | built-in |
| `ltx23.distilled.text-to-video` | Text to Video with synchronized audio | complete LTX 2.3 distilled BF16 folder | built-in |
| `ltx23.distilled.image-to-video` | Image(s) to Video with synchronized audio | same shared LTX 2.3 distilled BF16 folder | built-in |
| `wan22.ti2v5b.text-to-video` | Text to Video | complete Wan 2.2 TI2V 5B BF16 folder | built-in |
| `wan22.comfy-org-14b-i2v-fp8` | Image to Video | five exact Comfy-Org FP8/native support artifacts | built-in when locally present |

Additional runtime paths exist but are not yet equivalent built-in defaults:

| Family/path | Status |
| --- | --- |
| Wan 2.2 14B Comfy FP8 I2V | Native stored-weight runtime is workstation-proven; package recipe validates an exact local five-resource closure without runtime conversion or automatic acquisition |
| Klein 4B stored FP8 | Native stored-weight transformer path is workstation-proven; BF16 remains the public baseline |
| Klein 9B T2I/I2I | Direct complete-folder tools exist; 9B is not in the first lean built-in profiles and I2I still needs hands-on diagnosis |
| MiniMax H3 | T2V/first-last runtime tools exist; curated Comfy-aligned artifacts and Ref2VA remain active work |
| LTX 2.3 I2V/anchored video | First-frame and optional final-frame anchor use the pinned ConditionPipeline; 24fps/product defaults remain fixed |
| Wan 14B T2V/first-last and Wan 5B I2V | Official workflows are mapped; Engine runtime operations are not implemented yet |

Run `recipes list` for the authoritative catalog on the current machine. A recipe
may be present but unavailable when its resource is absent or does not satisfy the
exact loader contract. Catalog errors are actionable and must be resolved rather
than ignored.

## Dimensions and source sizing

H3, Klein, LTX, and Wan tools expose granular `width` and `height`, not a short preset
list. Engine accepts project-oriented dimensions, aligns them to the model grid,
and rejects requests above the current family safety budget before loading a
pipeline:

- H3: nearest 32 pixels, each side at least 64 pixels, effective aspect ratio
  from 1:4 through 4:1, and up to 1,032,192 output pixels;
- Klein: nearest 16 pixels, up to 1,048,576 output pixels;
- LTX 2.3: nearest 32 pixels, up to 942,080 output pixels;
- Wan 5B: nearest 16 pixels, up to 901,120 output pixels.

For Klein Image to Image, omit both fields to inherit the first source's
EXIF-oriented canvas through the pinned model's native floor-to-16 behavior.
Supplying one dimension without the other is invalid. Result metadata reports
both requested and effective dimensions.

H3 exposes a separate integer `steps` input (1–30, default 20) alongside the
960×544 default canvas. This carries forward the previous balanced policy;
legacy draft was 16 steps and final was 30. Canvas selection no longer changes
the step count.

## Inspecting recipes, resources, and storage

```powershell
uv run latentslate-engine data path
uv run latentslate-engine bundles list
uv run latentslate-engine resources list
uv run latentslate-engine recipes list
uv run latentslate-engine recipes validate
uv run latentslate-engine deployments profiles
uv run latentslate-engine deployments plan klein4b-image
```

LatentSlate owns its model library instead of relying on the user-global Hugging
Face cache. Set `LATENTSLATE_ENGINE_HOME` in `.env` to place the full tree on the
desired drive. Relative values resolve from the repository. Engine redirects its
Hugging Face, Diffusers, Transformers, and Torch caches below that root.

```text
LATENTSLATE_ENGINE_HOME/
├── models/
├── loras/
├── recipes/
├── profiles/
├── resource_declarations/
├── cache/
├── assets/
├── jobs/
├── logs/
└── temp/
```

Private recipes can live in `${LATENTSLATE_ENGINE_HOME}/recipes`, with matching
resource declarations and deployment profiles in the adjacent directories.
Additional private catalog roots can be supplied with `LATENTSLATE_RECIPE_PATHS`
and `LATENTSLATE_DEPLOYMENT_PROFILE_PATHS`. See
[docs/RECIPES.md](./docs/RECIPES.md).

## Authentication

Copy `.env.example` to ignored `.env`, accept any gated model terms, and set a
read-only Hugging Face token:

```powershell
Copy-Item .env.example .env
```

```dotenv
HF_TOKEN=hf_replace_me
```

A real process/container environment variable wins over `.env`. Remote hosts
should inject `HF_TOKEN` and `LATENTSLATE_ENGINE_TOKEN` as secrets; never bake
them into an image or deployment lock. Civitai source credentials are represented
by environment-variable names rather than serialized values.

## Reset and rebootstrap

To recreate only the Python environment while preserving every downloaded model,
recipe, job, and setting:

```powershell
Remove-Item -LiteralPath .\.venv -Recurse -Force
.\scripts\bootstrap.ps1
```

To reinitialize the data-directory structure without deleting its contents:

```powershell
uv run latentslate-engine data init
```

For a true full data reset, first print and inspect the exact target:

```powershell
$engineHome = [System.IO.Path]::GetFullPath((uv run latentslate-engine data path).Trim())
$engineHome
```

Stop the Engine and delete that directory manually only after verifying it is the
dedicated Engine data root. This deletes all models, caches, uploads, job outputs,
local recipes, profiles, and declarations below it. The README intentionally does
not provide a generic recursive-delete command: `LATENTSLATE_ENGINE_HOME` is
user-configurable, so no portable script can prove that a broad existing directory
was not selected accidentally. After deletion, run `uv run latentslate-engine data
init` to recreate the empty layout.

On Linux, the equivalent environment reset is:

```bash
rm -rf .venv
./scripts/bootstrap.sh
```

## Run and connect LatentSlate

Start a local-only server:

```powershell
uv run latentslate-engine serve --host 127.0.0.1 --port 8765
```

In LatentSlate, set the Engine endpoint to `http://127.0.0.1:8765` and refresh
the Engine provider catalog after any tool schema change.

For a LAN or remote GPU host, bind externally only with a bearer token and a
secure tunnel or HTTPS reverse proxy:

```powershell
$env:LATENTSLATE_ENGINE_TOKEN = 'replace-me'
uv run latentslate-engine serve --host 0.0.0.0 --port 8765
```

LatentSlate uploads media and downloads artifacts over HTTP; desktop and Engine
never need a shared filesystem. This V0 has no automated model input/output
filters, so keep use human-reviewed and do not expose it as an unattended public
generation service.

## Model lifecycle

Engine runs one generation worker and keeps at most one heavyweight runtime
active. Recipes sharing a model can reuse one pipeline; switching families evicts
the prior runtime so models do not accumulate indefinitely in RAM or VRAM. Native
stored-weight runtimes own physical tensor residency and do not delegate their
quantized storage to generic Diffusers offload hooks.

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
| `LATENTSLATE_RECIPE_PATHS` | unset | Additional private recipe catalog roots, separated by the platform path separator |
| `LATENTSLATE_DEPLOYMENT_PROFILE_PATHS` | unset | Additional private deployment-profile roots |
| `LATENTSLATE_H3_MODEL` | `MiniMaxAI/MiniMax-H3` | Direct H3 fallback repository under `models/h3/` |
| `LATENTSLATE_H3_PROFILE` | `bf16_auto_offload` | `bf16_auto_offload` |
| `LATENTSLATE_H3_DEVICE` | `cuda` | Torch device used by H3 |
| `LATENTSLATE_LTX23_MODEL` | `diffusers/LTX-2.3-Distilled-Diffusers` | Direct LTX 2.3 fallback repository under `models/ltx23/` |
| `LATENTSLATE_LTX23_PROFILE` | `bf16_sequential_offload` | `bf16_sequential_offload`, `bf16_model_offload`, or `bf16_cuda` |
| `LATENTSLATE_LTX23_DEVICE` | `cuda` | Torch device used by LTX 2.3 |
| `LATENTSLATE_WAN22_MODEL` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Wan 2.2 dense TI2V-5B repository |
| `LATENTSLATE_WAN22_PROFILE` | `bf16_sequential_offload` | `bf16_sequential_offload`, `bf16_model_offload`, `bf16_group_leaf` (experimental recovery), or `bf16_cuda` |
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
Klein 9B artifacts remain unavailable until their artifact metadata and exact
loaders are implemented and verified.

## Troubleshooting

- If a traceback ends in `KeyboardInterrupt`, Python was externally interrupted;
  it is not a loader diagnosis. Run long downloads in a dedicated PowerShell
  window and let the process return to a prompt before closing it.
- If a recipe disappeared after pulling a schema change, run `recipes validate`,
  fix or remove the reported stale local TOML, then refresh LatentSlate's Engine
  catalog.
- If a bundle appears installed but a recipe is unavailable, compare
  `resources list` with `deployments plan <profile>`; recipes require exact
  resource identity and completeness, not merely a similarly named folder.
- Run `uv run latentslate-engine doctor --json` for automation-friendly hardware,
  package, authentication, disk, and bundle diagnostics without loading a model.

See [docs/DIAGNOSTICS.md](./docs/DIAGNOSTICS.md) for deeper checks.

## Protocol

The current endpoints are:

```text
GET    /v1/health
GET    /v1/catalog
GET    /v1/resources
GET    /v1/variants
GET    /v1/recipes
GET    /v1/deployment/profiles
GET    /v1/deployment/plan/{profile_key}
GET    /v1/deployment/lock/{profile_key}
GET    /v1/bundles
GET    /v1/runtime
DELETE /v1/runtime/cache
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

Full H3, LTX 2.3, and dense Wan 5B inference still require hardware validation on
the target RTX 5080 / 64 GB workstation or an appropriately sized remote GPU.
Native Wan 14B I2V and Klein 4B stored-FP8 text/image generation have been
exercised through the normal API on that workstation. Family adapters use
Comfy-native stored formats and execution lessons where proven, while reusing
compatible Diffusers model shells and orchestration components instead of copying
ComfyUI itself.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The tests use lightweight fakes and do not download or execute the model weights.
