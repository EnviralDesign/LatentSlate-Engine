# Resource taxonomy and storage migration specification

Status: **Back-burner design specification. Do not implement until the LTX 2.3
optimized-path tranche and its user pair-testing session are complete.**

This document defines a future migration of LatentSlate Engine resource authoring,
catalog metadata, and on-disk storage. It is intentionally a specification rather
than a session plan. Existing declarations, installed artifacts, resource IDs, and
recipes remain authoritative until a separately reviewed implementation lands.

## Problem statement

Engine currently has two top-level resource kinds, `model` and `lora`, and stores
artifacts primarily below:

```text
<ENGINE_HOME>/models/<family>/...
<ENGINE_HOME>/loras/<family>/...
```

Text encoders, VAEs, denoisers, pipeline support, and upscalers are represented as
`model` resources with optional free-text `component` metadata. This is functional:
the Resource Editor can inspect a Hugging Face or CivitAI source, prefill fields,
publish a declaration, and fetch the artifact. It is not a durable semantic or UX
contract:

- unrelated artifact roles share the broad `model` kind;
- the Resource Editor exposes a free-text component field rather than a guided type;
- generated paths vary by family and recipe history;
- cataloging an artifact can be confused with proving runtime compatibility;
- LoRAs live outside the otherwise model-like storage root;
- importing ComfyUI's entire folder list would copy legacy aliases and unrelated
  product surface into Engine.

The target is a small, Engine-owned taxonomy informed by current ComfyUI behavior,
Comfy Kitchen's technical contracts, and the exact artifacts required by Engine's
selected recipe superset.

## Design principles

1. **Classify by runtime role, not loader history.** ComfyUI currently treats `unet`
   as a legacy alias of `diffusion_models` and `clip` as a legacy alias of
   `text_encoders`. Engine should not preserve those duplicate concepts.
2. **Keep quantization orthogonal.** BF16, FP16, FP8, ConvRot INT8, NVFP4, GGUF, and
   future layouts are format and dispatch metadata on an artifact, not storage kinds.
3. **Use only categories required by selected recipes.** Do not pre-create the full
   ComfyUI folder catalog.
4. **Prefer human-browsable paths.** Canonical depth is category, then model
   family/variant, then files. Source/vendor provenance belongs in the declaration,
   not another directory level.
5. **Separate catalogability from executability.** A resource may be safely inspected,
   declared, and installed before any Engine runtime supports its architecture or
   quantization. The UI must say so plainly.
6. **Never require redownload to adopt metadata.** Existing files remain valid through
   aliases until an explicit, verified physical migration is requested.
7. **Fail closed on ambiguity.** Publication rejects unsafe paths, incompatible type
   assertions, ambiguous filename collisions, and source facts that changed after
   inspection.

## Canonical minimum taxonomy

The initial canonical storage layout is:

```text
<ENGINE_HOME>/models/
  denoisers/
    <family-or-variant>/
      <artifact files>
  bundled_checkpoints/
    <family-or-variant>/
      <artifact files>
  text_encoders/
    <family-or-variant>/
      <artifact files>
  autoencoders/
    <family-or-variant>/
      <artifact files>
  loras/
    <family-or-variant>/
      <artifact files>
  latent_upscalers/
    <family-or-variant>/
      <artifact files>
  pipeline_repositories/
    <family-or-variant>/
      <repository snapshots>
  pipeline_support/
    <family-or-variant>/
      <tokenizer, scheduler, configuration, and support files>
```

Definitions:

| Category | Meaning | Current examples |
| --- | --- | --- |
| `denoisers` | The primary generation backbone: UNet, DiT, diffusion transformer, or flow-matching transformer | Wan high/low experts, standalone Z-Image transformer |
| `bundled_checkpoints` | One checkpoint exposing multiple pipeline components | LTX 2.3 Comfy checkpoint with model, video VAE, audio VAE, and vocoder weights |
| `text_encoders` | Standalone text or multimodal prompt encoders used in the text-conditioning role | UMT5, Gemma, Qwen text/vision encoder used as one prompt component |
| `autoencoders` | Standalone image, video, or audio encoders/decoders | Wan VAE, Flux/Klein VAE |
| `loras` | Additive low-rank adapters for model or encoder targets | Wan LightX, LTX Distilled model LoRA, Gemma text LoRA |
| `latent_upscalers` | Models that operate on latent representations between generation stages | LTX spatial x2 upscaler |
| `pipeline_repositories` | Exact whole-repository component closures used by native/reference runtimes | Diffusers reference snapshots for LTX, H3, and Wan |
| `pipeline_support` | Non-weight or small-weight support closure that cannot stand alone | Tokenizers, schedulers, model configs, filtered support snapshots |

The UI label for `denoisers` should be **Generation model (denoiser)** so the role is
clear without requiring users to know UNet/DiT implementation details.

### Categories intentionally deferred

Add these only when an approved recipe requires them:

- `vision_encoders` for a distinct vision-conditioning artifact;
- `control_adapters` for ControlNet or T2I Adapter;
- `image_upscalers` for decoded-pixel super-resolution;
- embeddings, style models, detection, optical flow, interpolation, and other
  task-specific assets.

Do not create empty directories or schema enum values merely because ComfyUI has a
folder with that name.

## Path rules

1. Paths use `<category>/<family-or-variant>/<files>` with no vendor/source subfolder.
2. Family/variant slugs are stable, readable, and normalized by Engine; examples are
   `ltx-2.3`, `wan-2.x`, `flux-klein-4b`, and `z-image-turbo`.
3. Immutable source identity remains in the resource declaration: provider, repo/model
   ID, revision/version ID, upstream filename/file ID, expected bytes, and SHA-256.
4. When two different immutable artifacts would occupy the same canonical path,
   publication must not silently overwrite either file. It should recommend a concise
   distinguishing filename and require preview/confirmation.
5. Several recipes may reference one resource ID and one physical file. Shared use
   must not create copies.
6. Comfy workers stage canonical artifacts into their required folder vocabulary by
   same-volume hardlink or another explicitly validated zero-copy mechanism. Canonical
   Engine storage does not need to mirror every Comfy loader directory.

## Resource schema

Replace the semantic overload of `ResourceKind.MODEL` with a typed artifact role. The
exact enum/API shape is an implementation decision, but it must represent every
canonical category above without relying on free-text component values.

A resource declaration must retain at least:

- stable resource ID;
- canonical artifact type;
- family/variant;
- display name and description;
- relative canonical path;
- container format and stored precision;
- quantization format/layout and required loader/dispatch capability when known;
- architecture/base-model compatibility metadata when known;
- exact immutable source and integrity facts;
- installed state derived from the canonical or legacy-resolved path;
- explicit status distinguishing cataloged, installable, structurally supported, and
  runnable/accepted.

Quantization metadata must be extensible enough for per-tensor or mixed layouts. Do
not encode every format as a top-level artifact type.

## Resource Editor UX

The existing source-first flow remains:

1. Paste or enter a Hugging Face/CivitAI locator.
2. Inspect source without publication or download.
3. Select an exact candidate file when a source contains several artifacts.
4. Review immutable facts and recommended declaration fields.
5. Preview and validate.
6. Publish the declaration.
7. Fetch/install as an explicit separate action.

Change the declaration form as follows:

- Replace `Kind: Model / LoRA` plus free-text `Component` with an **Artifact type**
  selector populated from the canonical supported taxonomy.
- Inspection recommends an artifact type from path, repository metadata, file header,
  and tensor signals. The recommendation is editable before publication.
- Display the generated canonical destination path immediately when type or family
  changes.
- Explain that **Cataloged** means the artifact can be retained and managed, not that
  a recipe can execute it.
- Show compatible built-in/custom recipes when compatibility is proven.
- Show `No compatible Engine runtime currently declared` without blocking safe
  publication of an otherwise valid artifact.
- Preserve source/integrity facts as read-only after inspection.
- Keep local/NAS imports CLI-managed until the browser has an explicitly safe local
  selection mechanism.

### Inspection recommendations

Inspection should be best-effort and evidence-labeled:

- recognize LoRA tensor pairs and target families;
- distinguish standalone denoiser schemas from bundled checkpoints;
- recognize common text-encoder and VAE architectures from keys/config/header data;
- identify mixed/quantized tensor layouts without claiming runtime support;
- retain `unknown` when evidence is insufficient rather than guessing;
- explain which evidence produced each recommendation.

## ComfyUI and Comfy Kitchen relationship

ComfyUI, its pinned official workflows, current node input/output schemas, and model
folder aliases are primary interoperability evidence. Comfy Kitchen is primary
evidence for quantized tensor representation, layout metadata, native dispatch, and
fallback behavior. They do not directly dictate Engine's product taxonomy.

Canonical compatibility mappings initially include:

| Comfy vocabulary | Engine canonical category |
| --- | --- |
| `diffusion_models`, legacy `unet` | `denoisers` |
| `checkpoints` when one file exposes several components | `bundled_checkpoints` |
| `text_encoders`, legacy `clip` | `text_encoders` |
| `vae` | `autoencoders` |
| `loras` | `loras` |
| `latent_upscale_models` | `latent_upscalers` |
| `diffusers` | `pipeline_repositories` |

Staging code owns translation from Engine categories to the exact folders expected by
a pinned Comfy operation. The submitted graph and worker provenance must bind the
canonical resource IDs, exact paths/integrity facts, staged names, Comfy revision/node
schema, and actual dispatch evidence where applicable.

## Compatibility and migration

This is a compatibility-first migration.

### Phase 1: schema and resolver compatibility

- Add typed artifact categories and canonical path generation.
- Continue reading legacy `model:` and `lora:` declarations and current physical paths.
- Build a deterministic mapping from known legacy `component` values to canonical
  categories.
- Reject ambiguous mappings and report actionable diagnostics.
- Do not move or redownload files.

### Phase 2: new-publication cutover

- Make newly authored resources use canonical categories and category-first paths.
- Keep existing IDs resolvable through compatibility aliases.
- Update recipe authoring selectors to filter by canonical type, architecture,
  format, and proven runtime capability.
- Add catalog and CLI reporting for legacy versus canonical storage.

### Phase 3: migration planning

- Provide a read-only command that produces an exact migration plan: source path,
  destination path, resource IDs, dependent recipes/profiles, size, integrity, volume,
  collision status, and whether hardlink/reflink/rename/copy is possible.
- Require the plan to prove no destination escapes Engine home and no unrelated file
  will be replaced.
- Support a dry run and machine-readable output.

### Phase 4: explicit physical migration

- Run only after user approval of an exact plan.
- Prefer same-volume atomic rename or verified hardlink transition; avoid redownload.
- Verify byte size and SHA before changing declarations.
- Publish declaration changes transactionally.
- Retain rollback information until every dependent recipe resolves.
- Never recursively delete legacy roots as part of automatic startup or catalog load.

## Recipe and runtime behavior

- Typed recipes reference stable resource IDs, not inferred filenames.
- Each role declares accepted canonical types and architecture/format capabilities.
- A resource being installed is insufficient for recipe availability; the tool must
  also prove its loader/runtime/backend dependency and format support.
- Whole-repository reference paths remain separate from optimized split-artifact
  recipes.
- Comfy-backed recipes translate canonical resources to exact staged Comfy folders
  without changing canonical storage.
- Provenance records canonical type, resource ID, immutable source/integrity identity,
  recipe role, staged basename, and observed dispatch/output facts.

## Security and data-safety requirements

- Preserve current source allowlisting, immutable-revision enforcement, size/SHA
  verification, reparse-point rejection, same-volume staging checks, and transactional
  declaration publication.
- Canonical relative paths must be generated/validated components, never arbitrary
  traversal supplied by a remote source.
- Filename collisions fail closed.
- Migration cannot overwrite an existing different artifact.
- Deletion remains dependency-aware and separately authorized.
- Logs/errors must not expose prompts, tokens, credentials, or private absolute paths.

## Required test and acceptance gates

1. Unit tests for every canonical type, legacy mapping, slug/path rule, and collision.
2. Resource Editor tests for inspection recommendations, manual override, destination
   preview, validation, publication, and install separation.
3. Catalog tests proving legacy and canonical declarations resolve to one descriptor,
   not duplicates.
4. Recipe tests proving type/architecture/format filtering and unavailable reasons.
5. Migration-plan tests covering collisions, cross-volume files, missing artifacts,
   changed hashes, reparse points, dependent recipes, and rollback data.
6. A fixture representing the current accepted recipe superset and all artifact roles.
7. No-network routine test suite; remote inspection tests remain mocked or opt-in.
8. Before physical migration, audit the real Engine inventory and produce a reviewed
   machine-readable plan with aggregate bytes and exact target paths.

## Delegation boundary

This task should be delegated only after the LTX 2.3 optimized implementation and user
pair-testing session are complete. It is broad enough for one primary Terra-high
implementation agent with strict file boundaries, followed by an independent Sol-high
review. Use additional read-only scouting only for inventory/path mapping if needed.

The implementation agent should receive:

- this specification;
- a current real-inventory report;
- explicit instruction not to move/delete installed artifacts;
- ownership of resource schema, resolver/path generation, Authoring API/UI, focused
  tests, and documentation;
- a requirement to stop at a reviewed compatibility seam before any physical migration.

The first deliverable is schema/UI/resolver compatibility and a read-only migration
plan. Physical migration is a separate user-authorized tranche.

## Out of scope for the first tranche

- implementing new model architectures or quantization kernels;
- adding every ComfyUI model folder;
- moving or deleting current installed artifacts;
- changing recipe tiers based solely on storage migration;
- declaring cataloged resources runnable without loader and hardware evidence;
- redesigning recipe authoring beyond the resource selection changes required here.

## Completion criteria

The taxonomy tranche is complete when:

- new resources can be authored under explicit canonical types;
- canonical paths follow category/family/files with no vendor-depth requirement;
- existing declarations and installed resources continue to resolve without copies or
  redownload;
- recipe availability remains truthful;
- the Resource Editor clearly distinguishes cataloged, installable, supported, and
  runnable states;
- a reviewed dry-run migration report can describe every legacy resource safely;
- all focused and repository gates pass;
- no physical migration has occurred without separate user approval.
