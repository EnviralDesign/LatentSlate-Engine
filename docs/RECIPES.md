# Runnable recipes

A runnable recipe is the public discovery and deployment boundary for one exact
Engine operation. It binds a typed Engine tool to fixed lineage, exact resource roles,
saved operation semantics, bounded dynamic slots, and Engine runtime policy.

Read the normative [Comfy evidence and Engine execution policy](./COMFY_ENGINE_POLICY.md)
and [model-roadmap authority policy](./model-roadmaps/README.md).

## Boundaries

- A resource is an artifact identity and acquisition contract, not an operation.
- A deployment profile is a saved recipe selection and deduplicated closure.
- ComfyUI is never an Engine backend, dependency, worker, server, or fallback.
- Cataloged, installed, structurally tested, runnable, Hardware-proven, and
  Recommended are distinct states.
- Reference and Recommended are independent.

## Public key grammar

Use `<family-or-line>.<operation>.<edition>`. The edition is an opaque stable
identifier, not an execution-backend field.

Existing keys containing `comfy` retain compatibility and mean only that artifact
selection or operation defaults were derived from official Comfy evidence. They never
mean that Engine launches or imports ComfyUI. New keys should prefer unambiguous
lineage/layout names where compatibility permits.

Base versus Distilled, T2V versus I2V, ordinary versus KV, and one-stage versus
two-stage are separate contracts, not hidden switches.

## Typed recipe shape

Roadmaps must not invent authorable fields. Expert pairs, conditional/unconditional
branches, audio/video VAEs, fixed LoRAs, prompt enhancers, support closures, and
upscalers require reviewed typed roles.

A workflow-fixed LoRA is a fixed resource in the recipe fingerprint. It is not a
user-selectable slot. Dynamic LoRAs require an explicit target/header compatibility
contract and fail closed.

## Workflow-derived Engine authority

For an operation with an official workflow, implementation starts from its pinned raw
behavioral contract, not from a dense pipeline reconstructed from memory.

Authority is divided as follows:

1. publisher source owns weights, architecture, config, license, and dense Reference;
2. official workflow owns practical topology and saved defaults;
3. pinned ComfyUI source owns the node behavior being studied;
4. Kitchen plus exact headers owns low-bit layout and direct dispatch;
5. Engine public-API evidence owns runnability and tier.

Engine translates the normalized contract into Engine-owned typed code. It never
submits the normalized graph to ComfyUI and never requires a local ComfyUI source tree at runtime.
Kitchen is imported directly inside Engine-owned disposable workers.

Authoring baseline workflow source remains
[`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1),
package `0.1.37`. Family accepted and research pins may differ and remain separately
labeled.

## Pre-implementation contract

Before a recipe can be runnable:

1. fetch and hash the raw workflow;
2. normalize all subgraphs, switches, constants, outputs, and placeholders;
3. verify node semantics against pinned ComfyUI source;
4. enumerate active and configured-but-disabled artifacts;
5. resolve immutable identities, licenses/gates, and headers;
6. verify direct Kitchen/header compatibility and fallback behavior;
7. write independent fixtures;
8. implement Engine-owned orchestration, materialization, lifecycle, and output;
9. pass public-API acceptance.

The normalized contract records resource roles, prompt enhancement, preprocessing,
conditioning order, sampler/scheduler/sigmas, steps/stages, CFG/guidance, dimensions,
frames/fps, fixed LoRAs, and output semantics.

## Reference versus practical ordering

Dense BF16 remains Reference when published, but large video Reference is not a
requirement to force onto the local RTX 5080. Source-pin and CPU-validate it, retain
one bounded OOM when useful, then batch dense output comparisons on high-memory Vast.

Local work prioritizes an exact Engine-native practical path derived from official
workflow evidence and using stored FP8, ConvRot, NVFP4, or fixed LoRAs where justified.

## Complete resource closure

A recipe includes every active fixed resource: transformers/experts/branches,
encoders, image/video/audio VAEs, vocoders, upscalers, patches, fixed LoRAs, prompt
enhancers, tokenizer/scheduler/processor/config support, and exact artifact identities.

Do not omit subgraph-loaded resources. Do not include disabled resources in the
saved-default closure. Modes changing resources, schedule, or semantics normally
become separate recipes.

## Availability

A recipe is available only when every resource and typed role is valid, the
Engine-owned runtime and dependencies exist, direct Kitchen/native dispatch is
supported, license/auth gates are satisfied, the operation/input multiplicity is
implemented, and no active resource is missing.

A ComfyUI installation, executable, checkout, server, plugin, graph, or model folder
must never participate in availability.

Engine must not silently swap lineage, drop components, convert/dequantize stored
weights, route to another backend, or infer output metadata from the request.

## Runtime fingerprint and provenance

Fingerprint exact resources, normalized workflow authority, ComfyUI source revision,
Kitchen revision, operation, input order, preprocessing, prompt enhancement, fixed
LoRAs, schedule, output policy, and Engine runtime policy.

Provenance reports what Engine actually ran: effective request, direct
Kitchen/native dispatch, fallback counters, cache/cold state, lifecycle, output slot,
and observed artifact metadata. Workflow and ComfyUI pins are provenance for the
behavioral contract, not deployed dependencies.

## Merge gates

Reject graph drift, any forbidden external-UI execution dependency, hidden native fallthrough, false
availability, assumed output metadata, unobserved cancellation,
cataloged-versus-runnable confusion, and self-confirming fixtures.

## Related documentation

- [Execution policy](./COMFY_ENGINE_POLICY.md)
- [Catalog authoring](./CATALOG_AUTHORING.md)
- [Hardware studies](./HARDWARE_STUDIES.md)
- [Model roadmaps](./model-roadmaps/README.md)
- [Implementation packets](./model-roadmaps/IMPLEMENTATION_PACKETS.md)
