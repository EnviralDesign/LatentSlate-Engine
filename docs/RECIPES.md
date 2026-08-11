# Recipes, resources, and deployment profiles

LatentSlate Engine treats a runnable recipe—not an accumulating download bundle—as
the product-facing unit. The first implementation intentionally preserves legacy
`variants/` files and bundle installers while establishing the replacement boundary.

## Catalog roots

Recipes are loaded deterministically from:

1. Engine-owned built-ins under `latentslate_engine/builtin_recipes`;
2. `${LATENTSLATE_ENGINE_HOME}/recipes`;
3. durable private directories from `LATENTSLATE_RECIPE_PATHS` (OS path separator);
4. `${LATENTSLATE_ENGINE_HOME}/variants` as a compatibility catalog.

Deployment profiles are loaded from Engine-owned built-ins,
`${LATENTSLATE_ENGINE_HOME}/profiles`, and private directories from
`LATENTSLATE_DEPLOYMENT_PROFILE_PATHS`.

Duplicate recipe/profile keys are authoring errors. Private catalogs are expected to
live outside the public repository and may be gitignored or maintained in a separate
private repository.

## Resource acquisition metadata

Installed resource sidecars may declare one or more sources:

```toml
[[sources]]
type = "huggingface"
repo_id = "organization/repository"
revision = "exact-revision-or-commit"
filename = "optional/file.safetensors"
sha256 = "optional-64-character-hash"

[[sources]]
type = "civitai"
model_version_id = 123456
file_id = 789012
```

Supported source types are `huggingface`, `civitai`, and `manual`. Empty sources are
also valid and represent a source-less local resource. Authentication values are never
serialized. Deployment plans report only the required environment-variable names:
`HF_TOKEN`, `CIVITAI_TOKEN`, or an explicit `token_env` override.

This tranche does not implement a general downloader. Existing bundle installation
remains a compatibility mechanism while recipe-derived acquisition is built out.

## Deployment profiles and locks

A profile selects the exact recipes needed by one local or remote deployment:

```toml
[profile]
schema_version = 1
key = "editor-5080"
name = "Editor workstation"
recipes = [
  "wan22.native.production",
  "klein4b.distilled.production",
]
target = "windows-sm120-16gb"
```

`latentslate-engine deployments plan <key>` computes the deduplicated fixed-resource
closure, total bytes, incremental bytes, local runnability, remote provisionability,
dynamic resource slots, and required secret names. `deployments lock <key>` emits a
JSON-safe lock containing Engine version, recipe IDs/schema hashes, exact resources,
sources, sizes, and target metadata—never credential values.

Recipes with exposed model/LoRA selectors remain runnable, but a deployment profile
cannot claim a complete remote lock until those dynamic choices are fixed.

## Comfy-first reference policy

The initial default research is pinned to:

- repository: `Comfy-Org/workflow_templates`;
- commit: `1206ea94470a5b66948f1758a8feea5b00801ed1`;
- locally installed package evidence: `comfyui-workflow-templates-json==0.1.37`.

Comfy templates are behavioral/reference inputs only. Engine recipes are clean-room
implementations and do not embed ComfyUI or copy GPL implementation code.

Current reference set includes:

- H3 T2V, first/last-frame, and distinct reference-to-video templates;
- LTX 2.3 T2V, I2V, true first/last-frame, image+audio, IC/ID LoRA,
  ingredients, and style-transition templates;
- Wan 2.2 14B T2V, 14B I2V/FLF, and distinct 5B TI2V templates;
- Klein task/model-specific base and distilled T2I/I2I templates.

Intentional product deviations are recorded next to each built-in recipe as they land.
Public operation labels remain compact (`Text to Video`, `Image(s) to Video`, and
`Reference to Video`) even when recipes expose different optional inputs.

## Optimization reference policy

ComfyUI remains the primary behavioral reference. Wan2GP revision
`7e45fe7e21105807b43f6285827d9ebb5fa72906` is a secondary optimization reference
only. Its closest draft concept is an atomic accelerator-LoRA profile; there is no
`draft` or `super-draft` symbol to reproduce.

Dual-transformer Wan 2.2 I2V must not advertise TeaCache. MagCache remains opt-in and
calibration-bound to an exact model, resolution, and step schedule. Any future
clean-room cache implementation must reset state at every high/low model transition,
phase transition, and temporal-window boundary. No Wan2GP Community License 2.0 or
MMGP GPL-3.0 implementation code is copied or linked.

## Deferred work

This foundation deliberately does not implement a full storage manager, pruning UI,
Vast orchestration, or network acquisition executor. SDXL is the next high-priority
lightweight family after the current video/image operation tranche. Krea 2's exact
identifier and Ideogram 4 structured editing remain research items. A two-boundary
bridge is expected to be a LatentSlate timeline preset unless a model exposes a truly
distinct semantic operation.
