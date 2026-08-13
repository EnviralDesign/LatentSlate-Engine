# Resource taxonomy and storage migration specification

Status: **Deferred design specification.** Existing declarations, installed artifacts,
resource IDs, and recipes remain authoritative until a separately reviewed migration
lands.

This specification is subordinate to
[COMFY_ENGINE_POLICY.md](./COMFY_ENGINE_POLICY.md): Engine never stages resources for,
launches, imports, or submits work to ComfyUI. Comfy folder vocabulary is research
metadata only.

## Objective

Move from broad `model`/`lora` kinds and free-text components toward a small,
Engine-owned runtime-role taxonomy without redownloading or silently moving existing
files.

## Design principles

1. Classify by Engine runtime role, not by an upstream loader name.
2. Keep quantization orthogonal to artifact role.
3. Add only categories required by approved recipes.
4. Prefer human-readable category/family/file paths.
5. Separate catalogability, installability, structural support, runnability, and tier.
6. Preserve existing files through aliases until an explicit migration is approved.
7. Fail closed on ambiguous role, path, collision, or changed source facts.
8. Engine workers load canonical resources directly; no ComfyUI workspace or folder
   translation exists.

## Canonical minimum taxonomy

```text
<ENGINE_HOME>/models/
  denoisers/<family-or-variant>/
  bundled_checkpoints/<family-or-variant>/
  text_encoders/<family-or-variant>/
  autoencoders/<family-or-variant>/
  loras/<family-or-variant>/
  latent_upscalers/<family-or-variant>/
  pipeline_repositories/<family-or-variant>/
  pipeline_support/<family-or-variant>/
```

| Category | Meaning |
| --- | --- |
| `denoisers` | primary UNet/DiT/diffusion/flow backbone or expert |
| `bundled_checkpoints` | one file exposing several Engine-owned components |
| `text_encoders` | standalone prompt or multimodal encoders |
| `autoencoders` | image, video, or audio encoders/decoders |
| `loras` | additive low-rank adapters |
| `latent_upscalers` | latent-space upscalers between Engine stages |
| `pipeline_repositories` | exact whole-repository dense/reference closure |
| `pipeline_support` | tokenizer, scheduler, configs, processor, and support |

Add vision encoders, control adapters, decoded-image upscalers, embeddings, or other
categories only when an approved typed recipe requires them.

## Relationship to Comfy evidence

Pinned workflows and ComfyUI source help identify practical roles and aliases such as
`diffusion_models`, `text_encoders`, `vae`, `loras`, and
`latent_upscale_models`. Engine maps those research labels into its own taxonomy during
authoring; it does not reproduce ComfyUI folders or loaders.

Comfy Kitchen remains an allowed direct Engine dependency for quantized layouts. Layout
and dispatch metadata belong on the artifact declaration, not in the storage category.

## Path and schema rules

- Paths are generated under Engine home and reject traversal/symlink/reparse escape.
- Source/vendor identity remains in declarations, not directory depth.
- Shared recipes reference one resource ID and one physical file.
- Collisions fail closed; no silent overwrite.
- Declarations retain exact source/revision/path, bytes/hash, role, lineage, format,
  layout, license/gate, and proof state.
- Installed does not imply runnable.

## Migration phases

1. Add typed categories and legacy resolution without moving files.
2. Author new resources under canonical categories.
3. Produce a read-only migration plan with source/destination, dependent recipes,
   bytes/hash, volume, collision, and supported move/link/copy operation.
4. Perform physical migration only after explicit approval, exact integrity checks,
   transactional declaration updates, and rollback data.

Never redownload merely to change metadata. Never recursively delete legacy roots.

## Runtime and provenance

Typed recipes reference stable resource IDs and Engine roles. Engine-owned runtimes
validate architecture/layout, load canonical paths directly, call Kitchen directly
where required, and record resource identity, role, header, direct dispatch, and output
facts.

No category, alias, installed artifact, or workflow citation may create ComfyUI
availability or a fallback execution path.

## Required gates

- unit tests for categories, legacy mapping, paths, and collisions;
- catalog/UI tests for state distinctions and destination preview;
- recipe tests for role/architecture/layout filtering and unavailable reasons;
- migration-plan tests for collisions, changed hashes, reparse points, cross-volume
  behavior, dependencies, and rollback;
- no-network routine tests with synthetic headers;
- reviewed inventory and machine-readable dry-run before physical migration;
- explicit architecture test that no ComfyUI dependency, process, server, graph,
  plugin, workspace, or folder staging is introduced.

## Completion criteria

The tranche is complete when new resources use explicit roles, legacy resources still
resolve without copies/redownload, recipe availability stays truthful, the UI
distinguishes proof states, and a reviewed dry-run can describe every migration safely.
Physical migration remains separately authorized.
