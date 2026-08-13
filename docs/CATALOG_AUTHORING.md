# Catalog authoring

Catalog authoring records exact artifacts and runnable recipes without confusing discovery, installation, structural compatibility, and accepted execution.

Read [RECIPES.md](./RECIPES.md) and the normative [model authority policy](./model-roadmaps/README.md) before publishing a built-in family path.

## Core distinctions

| State | Meaning | What it does not prove |
| --- | --- | --- |
| **Inspected** | Source metadata/header were read without publication or download | identity is not frozen until source facts are retained and rechecked |
| **Cataloged** | A declaration exists with stable ID, role, source, path, and integrity facts | file is not installed or runnable |
| **Installable** | Auth/gate/source/path rules permit acquisition | loader/runtime compatibility is not proven |
| **Installed** | Expected file/directory exists and passes size/hash checks | operation is not available merely because bytes exist |
| **Structurally supported** | Typed role and independent header/schema fixtures match a loader contract | backend dispatch and output are unproven |
| **Runnable** | Complete recipe, runtime, dependencies, and backend can start | target hardware and creator quality are unproven |
| **Hardware-proven** | Public-API job produced accepted output with lifecycle/provenance evidence | Recommended tier still requires product judgment |

The UI/API must use these states and actionable unavailable reasons. “Resource installed” is never a shortcut to recipe availability.

## Authority ownership

Catalog declarations use the same hierarchy as roadmaps:

1. Publisher sources own lineage, architecture, config, license, and first-party artifact identity.
2. Pinned official Comfy workflows own the active practical artifact roles and saved operation defaults.
3. Pinned ComfyUI source owns loader/node schema and output slots.
4. Pinned Comfy Kitchen source/version and exact file header own low-bit layout and dispatch compatibility.
5. Engine public-API evidence owns runnable/accepted status and tier.

Catalog authoring must not infer practical closure from a dense parameter count or native pipeline when an official Comfy graph already selects a different split closure.

## Source-first workflow

1. Enter a Hugging Face, CivitAI, manual, or approved source locator.
2. Inspect without publishing or downloading a model payload.
3. Select the exact file or complete allow-pattern closure.
4. Review immutable and mutable facts separately.
5. Preview declaration, destination, dependencies, and availability impact.
6. Validate source/path/integrity/role rules.
7. Publish the declaration.
8. Install as a separate explicit action.
9. Revalidate before every load.

Inspection is best-effort. It may recommend a family, component role, precision, or layout, but it is not a compatibility oracle. Unknown remains unknown.

## Required resource identity

A built-in resource retains:

- stable resource ID;
- artifact role/type and family/variant;
- human name/description;
- canonical relative path;
- container format and stored precision;
- exact quantization/layout contract when applicable;
- architecture/base-model compatibility metadata;
- source provider and repository/model ID;
- immutable revision/version/file ID;
- exact filename or directory allow-pattern closure;
- expected byte count;
- SHA-256/LFS/Xet identity where available;
- license, gating, authentication, and redistribution posture;
- provenance: publisher, Comfy-Org, runtime-generated, or community;
- proof state and unavailable reason.

Mutable `main` URLs in workflow notes are discovery links only. Resolve them to immutable identities before publication. Do not complete a partial SHA from memory.

For gated repositories, record what was actually resolved. A model-card landing page is not an immutable acquisition contract. If authenticated metadata is unavailable, publication stops or remains a clearly labeled manual/gated placeholder without false installability.

## Complete closure rules

Author the active graph, not the filename list someone expected:

- expand subgraphs and switches;
- include every active transformer/expert/branch, encoder, VAE, vocoder, upscaler, fixed LoRA, prompt enhancer, tokenizer, scheduler, processor, and support file;
- distinguish saved-default active artifacts from configured-but-disabled artifacts;
- make a switch mode a separate recipe when it changes fixed resources, schedule, input semantics, or quality lineage;
- deduplicate shared resources by stable identity;
- never infer file bytes/hash from parameter count.

Examples:

- Krea’s saved default has Darkbrush configured but disabled; the base saved-default closure is three files, while enabled Darkbrush at 0.8 is a separate four-file mode.
- Qwen 2511’s saved INT8 graph is 40-step/CFG-4 with Lightning disabled; enabling the fixed LoRA also changes steps and CFG and therefore creates a separate mode.
- Wan 14B T2V and I2V use operation-specific expert pairs and support closures; shared naming does not make experts interchangeable.

## Header and schema inspection

SafeTensors inspection is CPU-safe and should happen before download when range/header access is possible. Record:

- file/header/schema fingerprints;
- tensor names, shapes, dtypes, aliases, and tied-weight expectations;
- quantization markers, sidecars/scales, group geometry, transpose/packing rules;
- fused projections and dense exceptions;
- expected source-to-target role map;
- fixed LoRA targets/rank/layout;
- architecture and component signals;
- unknown or contradictory metadata.

Recognition is not execution support. A resource may be cataloged safely while reporting `No compatible Engine runtime currently declared`.

For Kitchen-backed formats, the declaration must name the accepted/research Kitchen source and the expected native primitive. The runtime still must observe positive dispatch and zero hidden fallback before the recipe becomes runnable/accepted.

## Paths and safety

- Relative paths are generated/validated under Engine home; remote names never control traversal.
- Reject absolute paths, `..`, reparse-point escape, unexpected symlinks, and filename collisions.
- Existing different artifacts are never overwritten.
- Same-volume hardlink staging into an isolated Comfy workspace is allowed only after source and target paths are validated.
- Publication is transactional; incomplete declarations are not exposed.
- Deletion is dependency-aware and separately authorized.
- Logs do not expose auth tokens, prompts, private absolute paths, or source secrets.

## Recipe authoring

A recipe may reference only resources whose roles fit its typed contract. The authoring surface validates:

- exact required role set and no duplicate path for distinct roles;
- family/lineage/operation match;
- format/precision/layout contract;
- complete active closure and fixed resources;
- Comfy workflow, ComfyUI, and Kitchen pins when applicable;
- dynamic LoRA slots separately from workflow-fixed LoRAs;
- capability gate and unavailable reason;
- deterministic closure bytes and profile deduplication.

Do not accept generic dictionaries when an expert pair, conditional/unconditional pair, audio/video component set, or fixed-LoRA mode needs typed ownership.

## Availability review gates

Publication or discovery must fail closed against:

- **false availability:** installed resource but missing runtime, backend, dependency, license, object schema, or sibling component;
- **graph drift:** recipe closure/settings no longer match the normalized pinned workflow;
- **hidden conversion/fallback:** declared stored precision is not the executed layout;
- **assumed output:** tool descriptor claims media metadata the backend does not observe;
- **cataloged-versus-runnable confusion:** a declaration or green header test is surfaced as generation support.

## Custom and community artifacts

Custom support remains possible, but built-in recommendations require creator-visible value and strong provenance.

A community artifact declaration records author/repository/revision/license, claimed base, exact header identity, conversion method when known, expected loader/layout, and qualification status. Similar names such as FP8, NVFP4, ConvRot, GGUF, AWQ, or “Comfy” do not establish compatibility or LoRA interchangeability.

Community files begin Experimental or Cataloged. They earn a production tier only through the same independent graph, header, dispatch, lifecycle, and creator acceptance gates as first-party artifacts.

## Review checklist

Before merging catalog changes:

- source/revision/path resolve and immutable facts match;
- license/gating claims cite authoritative text and avoid unsupported legal conclusions;
- active closure is complete and disabled resources are separate;
- resource role/type matches the normalized graph and typed recipe;
- declaration destination is safe and collision-free;
- installability, structural support, runnable state, and product tier are distinct;
- no runtime conversion or fallback is implied by a format label;
- tests use independent fixtures rather than implementation-generated expectations;
- deployment closure is deterministic and deduplicated.

## Related documentation

- [Runnable recipes](./RECIPES.md)
- [Hardware studies](./HARDWARE_STUDIES.md)
- [Authority policy](./model-roadmaps/README.md)
- [Source-pin audit](./model-roadmaps/SOURCE_PIN_AUDIT.md)
