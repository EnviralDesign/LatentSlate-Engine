# Catalog authoring

Catalog authoring records exact artifacts and recipes without confusing discovery,
installation, structural support, and accepted execution.

Read [COMFY_ENGINE_POLICY.md](./COMFY_ENGINE_POLICY.md) and
[RECIPES.md](./RECIPES.md) before publishing a built-in path.

## State model

| State | Meaning | Does not prove |
| --- | --- | --- |
| Inspected | source metadata/header observed | immutable lock or compatibility |
| Cataloged | declaration and deterministic closure exist | installed or runnable |
| Installable | source/auth/path rules permit acquisition | loader support |
| Installed | bytes pass integrity checks | operation availability |
| Structurally supported | independent fixtures match a typed loader | dispatch/output |
| Runnable | complete Engine runtime can start | target acceptance |
| Hardware-proven | public-API output plus lifecycle/provenance | Recommended tier |

A resource being installed is never a shortcut to recipe availability.

## Authority ownership

Publisher sources own artifact/architecture/license facts. Official workflows own
practical active closure and saved defaults. Pinned ComfyUI source is read to verify
node semantics only. Kitchen plus exact headers owns quantized layout and direct
dispatch. Engine evidence owns runtime status.

Cataloging never creates a ComfyUI dependency. Engine does not stage files into Comfy
folders, create upstream execution workspaces, install plugins, or require a Comfy executable.
Acquisition likewise never hard-links staged payloads into inventory. A verified file
is atomically moved into its one canonical Engine path, with no shared file identity.

## Required identity

A built-in resource retains stable ID, typed role, family/lineage, canonical Engine
path, format/precision/layout, exact source/revision/path, bytes, SHA-256/LFS/Xet
identity where available, license/gating/auth posture, provenance, and truthful proof
state.

Mutable workflow-note `resolve/main` links are discovery only. Resolve exact identities
before publication; never complete SHA suffixes from memory.

## Active closure

Normalize the workflow evidence and include every active transformer/expert/branch,
encoder, VAE, vocoder, upscaler, fixed LoRA, prompt enhancer, tokenizer, scheduler,
processor, and support file. Record configured-but-disabled artifacts separately.

A switch changing resources, schedule, or operation normally creates a separate
recipe. Deduplicate shared resources by stable identity.

## Header/schema inspection

Before load, record tensor names, shapes, dtypes, aliases, fused projections, dense
exceptions, quantization markers, sidecars/scales, group geometry, packing,
source-to-target mapping, and fixed-LoRA targets/rank/layout.

For Kitchen-backed layouts, declarations name the accepted/research Kitchen source and
expected direct primitive. Positive observed dispatch and zero hidden fallback remain
runtime acceptance requirements.

## Engine storage and safety

- Canonical paths remain under Engine home.
- Reject traversal, reparse-point escape, unexpected symlinks, and collisions.
- Never overwrite a different artifact.
- Engine workers load canonical Engine resources directly; there is no translation to
  ComfyUI folder vocabulary.
- Publication is transactional and deletion is dependency-aware.
- Logs never expose tokens, prompts, secrets, or private absolute paths.

## Recipe authoring

Typed roles must express expert pairs, dual branches, A/V components, fixed LoRAs,
enhancers, and support. Validate lineage, operation, header/layout, complete closure,
normalized behavior, direct Kitchen capability, dynamic slots, and deterministic
deployment bytes.

Existing edition names containing `comfy` are provenance labels only. Authoring must
not interpret them as a backend or require ComfyUI.

## Availability gates

Fail closed on missing Engine runtime, direct Kitchen/native backend, dependency,
license, node-behavior fixture, sibling component, unsupported input multiplicity,
unknown output semantics, hidden conversion, or graph drift.

Unknown/custom resources may be cataloged while reporting no compatible Engine runtime.
They do not create an alternate workflow-execution route.

## Review checklist

- immutable source/path and license/gate evidence resolve;
- active closure is complete and disabled resources are separate;
- typed role and header map match independent fixtures;
- no ComfyUI package/process/server/graph/plugin/folder dependency exists;
- Kitchen is called directly where required;
- availability and proof states remain distinct;
- output metadata and cancellation are observed, not assumed;
- deployment closure is deterministic and deduplicated.

## Related documentation

- [Execution policy](./COMFY_ENGINE_POLICY.md)
- [Runnable recipes](./RECIPES.md)
- [Hardware studies](./HARDWARE_STUDIES.md)
- [Source-pin audit](./model-roadmaps/SOURCE_PIN_AUDIT.md)
