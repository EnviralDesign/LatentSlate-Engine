# LatentSlate Engine

LatentSlate Engine is the first-party generation backend for
[LatentSlate](https://github.com/EnviralDesign/LatentSlate). It exposes a small,
versioned catalog of creator-facing tools over HTTP rather than a general node
graph. Jobs, uploads, progress, cancellation, and generated artifacts work the
same way on a local workstation or a remote GPU host.

Engine treats a **recipe** as the runnable product boundary. A recipe binds a
tool to fixed resources or explicit resource slots plus runtime policy. A
deployment profile is a named, reusable selection of recipes for a workstation,
cloud machine, or reproducible lock; it is useful, never required for a normal
recipe install. The older bundle commands are legacy download compatibility
seams—installing every family bundle is neither required nor recommended.

Engine studies pinned official ComfyUI workflows and node implementations as
architecture and behavior sources, but never runs ComfyUI as a backend. Engine
owns inference and uses Comfy Kitchen directly for supported quantized loading
and dispatch. Its NVIDIA tiers also use standalone low-level
`comfy-aimdo==0.4.15` for Engine-owned DynamicVRAM residency; no ComfyUI module
or process participates. This hard boundary is specified in
[docs/COMFY_ENGINE_POLICY.md](./docs/COMFY_ENGINE_POLICY.md).

Runtime includes `comfy-kitchen==0.2.31` (Apache-2.0) to restore supported
already-quantized Comfy tensor layouts and `comfy-aimdo==0.4.15` (GPL-3.0) in
the NVIDIA extras. Engine is licensed GPL-3.0-or-later; retained third-party
terms are listed in `THIRD_PARTY_NOTICES.md`. Engine never quantizes or converts
model weights at runtime.

The active documentation map is [docs/README.md](./docs/README.md). Model and
optimization priorities live in structured
[model roadmaps](./docs/model-roadmaps/README.md); superseded implementation notes
are isolated under `docs/archive/` and are not current setup guidance.

## Quick start on Windows

From the Engine repository:

```powershell
git pull
.\scripts\bootstrap.ps1
.\scripts\engine.ps1 data init
.\scripts\engine.ps1 doctor
.\scripts\engine.ps1 recipes list
.\scripts\engine.ps1 recipes show flux2-klein-4b.text-to-image.comfy-distilled-fp8
.\scripts\engine.ps1 recipes plan flux2-klein-4b.text-to-image.comfy-distilled-fp8
```

The bootstrap preserves an existing ignored `.env`, creates it from
`.env.example` when absent, selects a reproducibly locked runtime tier, and
runs a lightweight Torch/CUDA/Comfy Kitchen validation without downloading any
models. `Auto` is the default: it selects CUDA 13.0 when the platform, Python,
NVIDIA driver, and GPU architecture qualify; otherwise it selects CUDA 12.8
when compatible, then protocol-only when no compatible NVIDIA runtime exists.
Both NVIDIA tiers require the actual/default CUDA device to meet Comfy Kitchen's
tested SM 7.5 floor; a secondary capable adapter cannot override an unsupported
default device.

The choice is always printed and saved below the Engine data root, so `doctor`
can report the selected tier, validation result, actual Kitchen backend, and
hardware-gated capabilities. It never claims NVFP4 just because CUDA 13 is
installed. An automatic downgrade is limited to a classified Torch/CUDA or
Comfy Kitchen backend validation problem; network, hash, lock, resolver, and
general installation failures stop with their original error.

Override the auto policy only when diagnosing or deliberately testing a tier:

```powershell
.\scripts\bootstrap.ps1 -Backend Auto      # default; prefer cu130
.\scripts\bootstrap.ps1 -Backend Cu130     # require CUDA 13.0 prerequisites
.\scripts\bootstrap.ps1 -Backend Cu128     # require the CUDA 12.8 compatibility tier
.\scripts\bootstrap.ps1 -Backend Protocol  # HTTP/catalog tooling only
```

After bootstrap, use `scripts\engine.ps1` for normal Engine commands. It reads
the recorded tier and applies the matching locked `uv run` flags, so a later
command cannot accidentally resync the environment as protocol-only.

On Linux, use the equivalent Bash entrypoints and lower-case tier values:

```bash
./scripts/bootstrap.sh                         # auto; prefer cu130
./scripts/bootstrap.sh --backend cu130         # require CUDA 13.0 prerequisites
./scripts/bootstrap.sh --backend cu128         # require the CUDA 12.8 compatibility tier
./scripts/bootstrap.sh --backend protocol      # HTTP/catalog tooling only
./scripts/engine.sh doctor
./scripts/engine.sh recipes list
```

The Bash bootstrap follows the same locked selection, validation, visible
fallback, and persisted-state rules as the PowerShell bootstrap.

For low-level debugging, the equivalent locked commands are kept explicit:

```powershell
uv sync --locked --extra nvidia-cu130 --group runtime
uv sync --locked --extra nvidia-cu128 --group runtime
uv sync --locked --extra protocol                 # same dependency set as plain `uv sync`

uv run --locked --extra nvidia-cu130 --group runtime latentslate-engine doctor
uv run --locked --extra nvidia-cu128 --group runtime latentslate-engine doctor
uv run --locked --extra protocol latentslate-engine doctor
```

The NVIDIA tiers both pin `torch==2.11.0`, sourced only from the official
PyTorch `cu130` or `cu128` index. CUDA 13 uses Comfy Kitchen's `cublas` extra;
CUDA 12.8 intentionally remains the compatibility path. Plain `uv sync` is
coherent but protocol-only—it does not install model runtimes.

The commands use compact, scan-friendly terminal views: tables for catalogs and
closures, labeled detail panels, and a clearly separated next action. They wrap
on narrow PowerShell terminals and respect normal color/`NO_COLOR` behavior.
Add `--json` when a script needs the complete structured payload; API endpoints
remain JSON. `doctor` presents system/CUDA readiness, model-family prerequisites,
and grouped checks, while recipe/resource commands report whether exact artifacts
are installed, whether a recipe is runnable, and whether the fixed closure is
automatically provisionable. `recipes list` includes a dedicated tier column for
recommended, fallback, reference, alternate, and experimental choices. The HTTP tool catalog
publishes runnable-recipe entries only; the family adapters they inherit from remain
internal implementation seams rather than duplicate provider choices.

During a human `recipes install` or `deployments install`, the terminal keeps a
single live progress display for closure preflight, the active file/resource,
bytes, transfer rate, verification, publication, and skipped resources. Use
`--json` for automation; its stdout remains one structured result document.

Install one recipe, then verify the catalog:

```powershell
$env:HF_TOKEN = 'hf_replace_me' # only if `recipes plan` reports it
.\scripts\engine.ps1 recipes install flux2-klein-4b.text-to-image.comfy-distilled-fp8
.\scripts\engine.ps1 recipes list
.\scripts\engine.ps1 recipes validate
```

Several recipe keys share one deduplicated resource closure:

```powershell
.\scripts\engine.ps1 recipes plan `
  flux2-klein-4b.text-to-image.comfy-distilled-fp8 flux2-klein-4b.image-to-image.comfy-distilled-fp8 flux2-klein-4b.image-to-image.comfy-base-fp8
.\scripts\engine.ps1 recipes install `
  flux2-klein-4b.text-to-image.comfy-distilled-fp8 flux2-klein-4b.image-to-image.comfy-distilled-fp8 flux2-klein-4b.image-to-image.comfy-base-fp8
```

The installer stages resumable downloads below the Engine data root, verifies
declared sizes, hashes, and repository structure, and publishes each resource
without overwriting an existing target. Publication atomically moves the verified
staged file into its canonical Engine path; resource installation never uses
hard links or shares file identity with staging. Shared resources are downloaded once.
After a full file hash succeeds, Engine caches that verification against the exact
expected SHA-256 plus filesystem device, identity, size, modification time, and
change time. Unchanged artifacts therefore do not get reread on every CLI or server
startup; any ordinary replacement or mutation forces a new full hash. Delete
`cache/resource-integrity-v1` below the Engine home when an explicit full recheck is
needed.

For a repeatable workstation or cloud target, save that selection as a deployment
profile. Profiles have their own discovery and inspection commands:

```powershell
.\scripts\engine.ps1 deployments profiles
.\scripts\engine.ps1 deployments plan klein4b-image
.\scripts\engine.ps1 deployments install klein4b-image
.\scripts\engine.ps1 deployments plan klein9b-image
.\scripts\engine.ps1 deployments install klein9b-image
```

`deployments lock` remains JSON-only and emits the exact reproducible closure for
automation:

```powershell
.\scripts\engine.ps1 deployments lock klein4b-image |
  Set-Content -Encoding utf8 .\klein4b-image.lock.json
```

Compatibility note: Engine is pre-1.0. The default output of `doctor`, recipe
and resource catalog/detail commands, deployment profile/plan commands, and
installations is human-readable. Existing automation must add `--json`;
`deployments lock` remains JSON-only.

The older bundle installer remains available only for legacy compatibility and
direct-tool testing; it is not the recipe workflow:

```powershell
.\scripts\engine.ps1 bundles install klein4b-basic
.\scripts\engine.ps1 bundles install ltx23-basic
.\scripts\engine.ps1 bundles install wan22-basic
```

Important: `wan22-basic` is the dense **Wan 2.2 TI2V 5B reference** repository. It does
not install the native Wan 14B I2V recipe. The 14B runtime currently uses an
explicit five-resource recipe (high/low transformers, UMT5, VAE, and pipeline
support). The package-owned 14B recipe becomes runnable only after those exact
local artifacts are supplied. `deployments install wan22-14b-i2v-fp8`
intentionally refuses before downloading anything: its filtered support directory
is not yet represented by a safe upstream acquisition manifest, so the complete
profile is not remotely provisionable and lock generation remains unavailable.

These additional compatibility downloads exist, but are not part of the first
lean built-in profile set:

```powershell
.\scripts\engine.ps1 bundles install h3-basic
.\scripts\engine.ps1 bundles install klein9b-basic
```

Do not run every install command unless you deliberately want every repository;
the combined footprint is hundreds of gigabytes.

## Recipe catalog files

The built-in TOML catalogs live in the source tree and are a useful starting point
for authoring or auditing a recipe:

- [`src/latentslate_engine/builtin_recipes`](./src/latentslate_engine/builtin_recipes)
  contains `[runnable_recipe]` definitions;
- [`src/latentslate_engine/builtin_profiles`](./src/latentslate_engine/builtin_profiles)
  contains `[profile]` saved recipe selections;
- [`src/latentslate_engine/builtin_resource_declarations`](./src/latentslate_engine/builtin_resource_declarations)
  contains `[resource]` acquisition/artifact declarations.

At runtime the local equivalents live in
`${LATENTSLATE_ENGINE_HOME}/recipes`, `${LATENTSLATE_ENGINE_HOME}/profiles`, and
`${LATENTSLATE_ENGINE_HOME}/resource_declarations`. `LATENTSLATE_RECIPE_PATHS`
and `LATENTSLATE_DEPLOYMENT_PROFILE_PATHS` add private recipe/profile roots using
the operating system path separator. `recipes show <key>` and `resources show <id>`
report the defining recipe path and artifact path; see
[docs/RECIPES.md](./docs/RECIPES.md) for the schema and acquisition rules.


### Custom catalog authoring

Use `resources inspect` before mutation, `resources add` to publish exact metadata
(or import a local file), and `resources fetch` to materialize one declared resource.
`recipes create <file>` saves an editable typed draft; `recipes publish <key>`
atomically moves a valid draft into the runnable catalog. A running Engine reports
catalog staleness and requires restart after API publication instead of pretending a
new recipe is active in the existing job-manager registry. See
[docs/CATALOG_AUTHORING.md](./docs/CATALOG_AUTHORING.md) for supported sources,
CLI/API examples, TOML editing, lifecycle behavior, reproducibility, and security
boundaries.

For resource declarations, the local Engine also includes a small browser editor. In
one terminal run `.\scripts\engine.ps1 serve`; in another run
`.\scripts\engine.ps1 author`. It opens
`http://127.0.0.1:8765/authoring/` (or an explicit loopback `--url`) against the
already-running Engine. The editor groups resources by family, keeps built-ins
read-only, and supports inspecting, previewing, publishing/updating local resources,
fetching a declared resource, and dependency-safe deletion of local declarations.
Artifact removal is a separate explicit choice; referenced resources cannot be deleted.
Browser authoring accepts Hugging Face and CivitAI
sources only; local imports and direct HTTPS remain CLI-only. Recipe authoring remains
TOML/CLI-only. Restart Engine after browser/API publication before using the changed
catalog for jobs.

## What is ready now

Package-owned built-in recipes currently cover:

| Recipe key | Operation | Resource | Current status |
| --- | --- | --- | --- |
| `flux2-klein-4b.text-to-image.bfl-distilled-nvfp4` | Text to Image | first-party BFL NVFP4 transformer + unchanged Distilled Qwen/full-VAE/support closure; measured native Kitchen CUDA dispatch | recommended on qualified Blackwell hardware |
| `flux2-klein-4b.image-to-image.bfl-distilled-nvfp4` | Image to Image, one to three ordered references | same first-party NVFP4 transformer and official Distilled four-step edit closure; mandatory partial residency | recommended on qualified Blackwell hardware |
| `flux2-klein-4b.text-to-image.comfy-distilled-fp8` | Text to Image | v0.1.37 distilled FP8 transformer + exact Qwen/full-VAE/support roles | Engine-native non-Blackwell path |
| `flux2-klein-4b.image-to-image.comfy-distilled-fp8` | Image to Image, one to three ordered references | same exact Distilled FP8/Qwen/full-VAE closure; Euler, 4 steps, guidance 1 | Engine-native non-Blackwell path |
| `flux2-klein-4b.image-to-image.comfy-base-fp8` | Image to Image, one to three references | current Base FP8 transformer + exact Qwen/small-decoder/support roles; 20 steps, guidance 5 | quality-alternate built-in |
| `flux2-klein-4b.text-to-image.native-distilled-bf16` | Text to Image | complete Klein 4B BF16 Diffusers folder | source-of-truth/reference built-in |
| `flux2-klein-4b.image-to-image.native-distilled-bf16` | Image to Image, one to three references | same complete BF16 folder | source-of-truth/reference built-in |
| `flux2-klein-9b.text-to-image.bfl-distilled-nvfp4` | Text to Image | first-party Distilled 9B NVFP4 transformer + exact mixed Qwen3-8B/small-decoder/support closure | recommended on qualified Blackwell hardware; controlled RTX 5080 acceptance passed |
| `flux2-klein-9b.image-to-image.bfl-distilled-nvfp4` | Image to Image, one to three ordered references | same 9B NVFP4 closure and four-step edit contract | recommended on qualified Blackwell hardware; controlled one-reference RTX 5080 acceptance passed |
| `flux2-klein-9b.text-to-image.bfl-distilled-fp8` | Text to Image | first-party Distilled 9B FP8 transformer + same mixed Qwen/small-decoder/support closure | non-Blackwell fallback; controlled RTX 5080 acceptance passed |
| `flux2-klein-9b.image-to-image.bfl-distilled-fp8` | Image to Image, one to three ordered references | same 9B FP8 closure and four-step edit contract | non-Blackwell fallback; controlled one-reference RTX 5080 acceptance passed |
| `flux2-klein-9b.text-to-image.native-distilled-bf16` | Text to Image | complete first-party Distilled 9B BF16 Diffusers closure | source-of-truth/reference built-in |
| `flux2-klein-9b.image-to-image.native-distilled-bf16` | Image to Image, one to three references | same complete BF16 closure | source-of-truth/reference built-in |
| `ltx-2-3.text-to-video.native-distilled-bf16` | Text to Video with synchronized audio | complete LTX 2.3 distilled BF16 folder | built-in |
| `ltx-2-3.image-to-video.native-distilled-bf16` | Image(s) to Video with synchronized audio | same shared LTX 2.3 distilled BF16 folder | built-in |
| `wan-2-2-5b-ti2v.text-to-video.native-bf16` | Text to Video | complete first-party Wan 2.2 TI2V 5B BF16 folder | reference |
| `wan-2-2-5b-ti2v.text-to-video.engine-stored-mixed` | Text to Video | exact FP16 transformer/VAE + stored-FP8 UMT5 + bounded support | Hardware-proven Recommended Engine-native direct-Kitchen path |
| `wan-2-2-5b-ti2v.image-to-video.engine-stored-mixed` | Image to Video, required first frame | same exact four-resource closure | Hardware-proven Recommended Engine-native direct-Kitchen path |
| `wan-2-2-14b-i2v.image-to-video.comfy-org-fp8` | Image to Video | five exact Comfy-Org-published FP8/native support artifacts | Engine-native accepted RTX 5080 path |
| `wan-2-2-14b-flf.first-last-frame-to-video.comfy-org-fp8` | First/Last Frame Video, required start and end images | same exact Comfy-Org-published I2V FP8/native support closure | Engine-native accepted single-pair RTX 5080 path |
| `wan-2-2-14b-flf.first-last-frame-to-video.comfy-org-fp8-lightx2v-4step` | First/Last Frame Video, required start and end images | same I2V closure plus the pinned official LightX2V high/low LoRA pair | experimental; accepted one fixed-pair RTX 5080 success/cancel/recovery path |
| `wan-2-2-14b-t2v.text-to-video.comfy-org-fp8` | Text to Video | exact official FP8 high/low pair + UMT5/VAE + T2V support closure | Engine-native accepted RTX 5080 path |
| `z-image-turbo.text-to-image.comfy-int8-convrot` | Text to Image | exact official four-resource Turbo closure with INT8 ConvRot NextDiT and mixed Qwen | Hardware-proven Recommended Engine-native direct-Kitchen path |
| `z-image-turbo.text-to-image.kutches-70s-horror-int8-convrot` | Text to Image | exact fixed Kutches rank-16 BF16 LoRA at strength 1.0 beside the immutable Z-Image Turbo INT8 ConvRot base | target-hardware proven Experimental; local-only while the upstream license remains undeclared |

Additional runtime paths exist but are not yet equivalent built-in defaults:

| Family/path | Status |
| --- | --- |
| Wan 2.2 14B FP8 I2V/FLF | Engine-native stored-weight I2V and first/last-frame paths are workstation-proven; each uses an exact local five-resource closure without runtime conversion. FLF acceptance is one fixed endpoint pair on one RTX 5080, not a corpus-quality claim. |
| Klein 4B stored quantized | First-party Distilled BFL NVFP4 T2I/I2I is recommended on qualified Blackwell hardware after successful RTX 5080 LatentSlate smoke tests; exact Distilled FP8 is the Engine-native non-Blackwell path. Compatible LoRAs use an additive branch without dequantizing the native Kitchen base weight |
| Klein 9B T2I/I2I | Package recipes mirror the ordinary Distilled 4B ladder: first-party NVFP4 recommended on Blackwell, first-party FP8 Engine-native non-Blackwell path, and complete BF16 reference. Controlled fixed-seed 1024² NVFP4/FP8 T2I and one-reference I2I acceptance passes; a real custom Hugging Face LoRA also passed native NVFP4 cold/warm deterministic API generation. The exact BF16 reference honestly OOMs on the 15.9 GiB workstation |
| MiniMax H3 | T2V/first-last runtime tools exist; curated source-pinned artifacts and Ref2VA remain active work |
| LTX 2.3 I2V/anchored video | First-frame and optional final-frame anchor use the pinned ConditionPipeline; 24fps/product defaults remain fixed |
| Wan 14B T2V/FLF | Engine-native stored-weight T2V baseline and FLF are accepted on RTX 5080; the official LightX v1.1 T2V recipe has separate success/cancel/recovery acceptance as Experimental. FLF LightX separately passed one fixed-pair RTX 5080 success, live-worker cancellation, and fresh-worker byte-identical recovery as Experimental; it still needs broader quality qualification. |

Run `recipes list` for the authoritative catalog on the current machine. A recipe
may be present but unavailable when its resource is absent or does not satisfy the
exact loader contract. Catalog errors are actionable and must be resolved rather
than ignored.

## Dimensions and source sizing

H3, Klein, LTX, and Wan tools expose granular `width` and `height`, not a short preset
list. Engine publishes each model's legal grid and rejects explicit illegal
dimensions; it does not silently rewrite an explicit request. LatentSlate may snap
picker or fallback values to that published grid before submission. Engine also
rejects requests above the current family safety budget before loading a pipeline:

- H3: nearest 32 pixels, each side at least 64 pixels, effective aspect ratio
  from 1:4 through 4:1, and up to 1,032,192 output pixels;
- Klein: nearest 16 pixels, up to 1,048,576 output pixels;
- LTX 2.3: nearest 32 pixels, up to 942,080 output pixels;
- Wan 5B dense Reference publishes its existing model grid; the stored-mixed T2V/I2V
  recipes require explicit 32-pixel alignment and at most 901,120 output pixels.

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
.\scripts\engine.ps1 data path
.\scripts\engine.ps1 bundles list
.\scripts\engine.ps1 resources list
.\scripts\engine.ps1 recipes list
.\scripts\engine.ps1 recipes validate
.\scripts\engine.ps1 deployments profiles
.\scripts\engine.ps1 deployments plan klein4b-image
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

Copy `.env.example` to ignored `.env`, accept any gated model terms, and set only
the source tokens required by the deployment plan:

```powershell
Copy-Item .env.example .env
```

```dotenv
HF_TOKEN=hf_replace_me
CIVITAI_TOKEN=replace_me_if_a_recipe_uses_civitai
```

A real process/container environment variable wins over `.env`. Remote hosts
should inject `HF_TOKEN`, `CIVITAI_TOKEN` when required, and
`LATENTSLATE_ENGINE_TOKEN` as secrets; never bake them into an image or deployment
lock. Source credentials are represented by environment-variable names rather
than serialized values.

## Reset and rebootstrap

To recreate only the Python environment while preserving every downloaded model,
recipe, job, and setting:

```powershell
Remove-Item -LiteralPath .\.venv -Recurse -Force
.\scripts\bootstrap.ps1
```

To reinitialize the data-directory structure without deleting its contents:

```powershell
.\scripts\engine.ps1 data init
```

For a true full data reset, first print and inspect the exact target:

```powershell
$engineHome = [System.IO.Path]::GetFullPath((.\scripts\engine.ps1 data path).Trim())
$engineHome
```

Stop the Engine and delete that directory manually only after verifying it is the
dedicated Engine data root. This deletes all models, caches, uploads, job outputs,
local recipes, profiles, and declarations below it. The README intentionally does
not provide a generic recursive-delete command: `LATENTSLATE_ENGINE_HOME` is
user-configurable, so no portable script can prove that a broad existing directory
was not selected accidentally. After deletion, rerun `.\scripts\bootstrap.ps1` to
recreate the runtime-selection record and empty data layout.

On Linux, the equivalent environment reset is:

```bash
rm -rf .venv
./scripts/bootstrap.sh
```

## Run and connect LatentSlate

Start a local-only server:

```powershell
.\scripts\engine.ps1 serve --host 127.0.0.1 --port 8765
```

In LatentSlate, set the Engine endpoint to `http://127.0.0.1:8765` and refresh
the Engine provider catalog after any tool schema change.

For a LAN or remote GPU host, bind externally only with a bearer token and a
secure tunnel or HTTPS reverse proxy:

```powershell
$env:LATENTSLATE_ENGINE_TOKEN = 'replace-me'
.\scripts\engine.ps1 serve --host 0.0.0.0 --port 8765
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
artifact loaders as they are validated. Runtime-tier diagnostics live in
[docs/DIAGNOSTICS.md](./docs/DIAGNOSTICS.md), while model-specific qualification
decisions live in [docs/model-roadmaps](./docs/model-roadmaps/README.md).

## Important environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for gated/private model downloads |
| `CIVITAI_TOKEN` | unset | Civitai token for exact recipe resources that require authentication |
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
| `LATENTSLATE_WAN22_DEVICE` | `cuda` | Torch device used by Wan 2.2; compatibility alias for the neutral execution device when that variable is unset |
| `LATENTSLATE_EXECUTION_DEVICE` | `cuda` | Neutral execution device used by Z-Image and future device-neutral recipe tools; takes precedence over the Wan compatibility alias |
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

LTX 2.3 and the legacy dense Wan 2.2 path both default to sequential CPU offload
because those complete-folder paths are not expected to remain resident on a 16 GB
GPU. The native LTX reference recipes use the exact publisher BF16 closure; the
legacy Wan recipe remains the official dense 5B checkpoint. Those dense paths are
correctness-first integrations, not claims of acceptable local speed or memory use.

Wan 2.2 TI2V 5B now has two typed Engine-native stored-mixed recipes. They load the
exact FP16 transformer/VAE and scaled-FP8 UMT5 directly from their canonical resource
paths, dispatch the text linears through Kitchen without dense fallback, and run in a
fresh Engine-owned disposable worker. The `wan22-ti2v5b-video` profile installs both;
the separate `wan22-ti2v5b-text-to-video` profile retains only the dense BF16
**Reference** recipe for remote scientific comparison. The optimized T2V and I2V
recipes are narrow **Hardware-proven Recommended** paths after paired RTX 5080
output, direct-dispatch, lifecycle, and recovery acceptance.

`klein4b-image` is the practical image profile. It installs the recommended
first-party BFL NVFP4 transformer, the non-Blackwell Distilled FP8 fallback, the
exact standalone Qwen3-4B BF16 encoder and full Flux2 VAE shared by those four-step
T2I/I2I recipes, and the Base FP8 transformer plus BFL full-encoder/small-decoder
VAE for the optional 20-step/guidance-5 I2I quality alternate. Distilled I2I preserves
ordered references and scales each toward 1 MP with PIL nearest-neighbor before
Diffusers floors it to the 16px VAE grid; this is the Engine-native approximation
of Comfy's tensor `nearest-exact`, with the first reference driving the canvas.
Engine loads those roles directly; the support shells contain no substitute
weights. Stored FP8 bytes/scales retain Engine-owned staged CUDA residency
without runtime quantization or converted model copies.

`klein4b-reference-bf16-image` installs the complete native BF16 Diffusers
repository. Keep it as the source-of-truth path for scientific comparisons with
quantized/optimized recipes, not as the everyday 16 GB default. The optional
direct-tool `bf16_cuda` profile requires enough VRAM for a resident full pipeline.

`klein9b-image` installs the ordinary Distilled 9B consumer ladder: first-party
NVFP4 and FP8 transformers, the exact Comfy mixed FP8/NVFP4 Qwen3-8B encoder,
the shared BFL full-encoder/small-decoder VAE, and a weight-free 9B support shell.
Both representations preserve the four-step/guidance-1 T2I/I2I contract and
ordered one-to-three-reference Engine surface. The mixed Qwen and transformer
paths use explicit native Comfy Kitchen dispatch with no runtime weight conversion;
controlled fixed-seed 1024² NVFP4/FP8 T2I and one-reference I2I output,
runtime-cold, warm-cache, memory-sampling, and determinism acceptance passes on the
RTX 5080. See the 9B roadmap for the measured distributions and remaining lifecycle
gates.

`klein9b-reference-bf16-image` installs the complete first-party Distilled BF16
Diffusers closure for source-of-truth comparisons. Base and KV are intentionally
not cataloged yet. The 9B roadmap tracks an
[unqualified community KV-NVFP4 conversion](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/tree/1c119e68f2741d0ad46ff56940ca54f622af1a24)
as a research lead; it is not an Engine recipe or recommended artifact.

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
- Run `.\scripts\engine.ps1 doctor --json` for automation-friendly hardware,
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
DELETE /v1/runtime
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

Manual fixed-seed generation acceptance lives outside routine CI. See
[docs/HARDWARE_STUDIES.md](./docs/HARDWARE_STUDIES.md) for the public-API harness,
the Klein 4B/9B cold/warm/benchmark/switch/family scenarios, state assertions,
timing summaries, manifests, and best-effort Reference policy. Benchmark scenarios
prove independent runtime resets, record three runtime-cold plus three
pipeline/cache-warm jobs per recipe, and never infer process-cold state.

Full H3 and dense LTX/Wan Reference paths still require appropriately sized remote
hardware. Engine-native optimized LTX 2.3 and Wan 5B recipes have target-hardware
public-API acceptance from exact installed closures. Their roadmaps retain the
remaining corpus-broadening and comparison work without downgrading accepted
Recommended paths.
Native Wan 14B I2V and the Klein 4B stored-FP8 transformer path have been
exercised through the normal API on that workstation. All seven current Klein 4B
recipes now have fixed 1024² public-API generation proof there, including NVFP4
warm reuse and NVFP4/FP8 recipe switching. Family adapters use pinned upstream
stored-format and topology evidence where proven, with Engine-owned model shells and
orchestration components plus direct Comfy Kitchen dispatch where supported.

## Development

```bash
# Preserve the selected accelerator tier while adding dev tools.
uv sync --locked --extra nvidia-cu130 --group runtime --extra dev # cu130 selection
uv sync --locked --extra nvidia-cu128 --group runtime --extra dev # cu128 selection
uv sync --locked --extra protocol --extra dev                     # protocol selection
uv run --no-sync pytest
uv run --no-sync ruff check .
```

Use the line matching the tier recorded by bootstrap; `scripts/engine.ps1` or
`./scripts/engine.sh` remains the safer day-to-day command wrapper. The explicit
`uv` forms above are for development/debugging because test and lint commands are
not Engine subcommands.

The tests use lightweight fakes and do not download or execute the model weights.
