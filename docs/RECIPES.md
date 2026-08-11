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

`latentslate-engine deployments install <profile-key>` acquires the fixed,
deduplicated closure of a remotely provisionable profile. It supports pinned Hugging
Face snapshots, exact Hugging Face files, and exact Civitai files. Downloads are
staged below the Engine data root, resumable where the source permits, and accepted
only after declared size, hash, and repository-structure verification. Civitai
streaming additionally enforces the declared byte ceiling during transfer. Resources
are published without overwriting an existing target. Manual and source-less resources,
dynamic selector slots, inexact sources, and unsupported directory source shapes
fail before network access. The legacy bundle installer remains only a compatibility
path for direct tools and older setup instructions.

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

The normal recipe-first sequence is:

```text
latentslate-engine deployments plan <key>
latentslate-engine deployments install <key>
latentslate-engine recipes validate
```

Profile installation is intentionally all-or-refuse at preflight: every missing
resource must have an exact supported source, every required secret must be present,
and the complete missing closure must fit on the staging volume before the first
request. Directory publication currently requires Windows or Linux no-clobber
filesystem primitives; other platforms fail closed rather than weaken the guarantee.

Recipes with exposed model/LoRA selectors remain runnable, but a deployment profile
cannot claim a complete remote lock until those dynamic choices are fixed.

## Comfy-first reference policy

The initial default research is pinned to:

- repository: `Comfy-Org/workflow_templates`;
- commit: `1206ea94470a5b66948f1758a8feea5b00801ed1`;
- locally installed package evidence: `comfyui-workflow-templates-json==0.1.37`.

Comfy templates are behavioral/reference inputs only. Engine recipes are clean-room
implementations and do not embed ComfyUI or copy GPL implementation code.

The first package-owned baseline recipes intentionally cover complete native
BF16 Diffusers folders already supported by a runtime—Klein 4B T2I/I2I (one
shared resource), LTX 2.3 distilled T2V/I2V (one shared resource), and Wan 2.2 TI2V 5B T2V—plus the
independently validated Wan 2.2 14B Comfy-Org FP8 I2V five-resource closure. They are
family/workflow-derived substitutions, not the same artifacts as the Comfy
templates: Klein T2I uses a standalone transformer plus components while Klein I2I
uses the standalone FP8 transformer; Comfy's LTX templates use an FP8 development
checkpoint plus distilled LoRA, while Engine uses a native BF16 Diffusers substitution;
Wan5 uses a split FP16 artifact. The LTX I2V recipe derives first/last-frame endpoint
semantics from Comfy v0.1.37 templates but invokes the pinned `LTX2ConditionPipeline`
with Engine's 24fps/product defaults. Each declaration
pins the exact Hugging Face revision and upstream snapshot size (excluding both
`.cache` and the Engine-generated `.latentslate-model.toml` sidecar). Built-in
deployment profiles stay family-specific so a normal install never pulls an
all-model superset.

The Wan 14B profile has four individually pinned and hash-validated Hugging Face
files, but its `pipeline_support` directory is a deliberately filtered subset of
`Wan-AI/Wan2.2-I2V-A14B-Diffusers`. It therefore has provenance only—not a
`ResourceSource`—because a whole-snapshot acquisition would pull additional
weights and could not reproduce the declared support directory. Its plan is
locally runnable only when all five exact artifacts exist and is intentionally not
remote-provisionable or installable until filtered acquisition support exists.

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
or Vast orchestration. Acquisition is currently limited to exact fixed-resource
deployment profiles; filtered snapshot manifests, dynamic recipe-slot resolution,
pruning, and remote instance lifecycle remain future work. SDXL is the next
high-priority lightweight family after the current video/image operation tranche.
Krea 2's exact identifier and Ideogram 4 structured editing remain research items. A
two-boundary bridge is expected to be a LatentSlate timeline preset unless a model
exposes a truly distinct semantic operation.
