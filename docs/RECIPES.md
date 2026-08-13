# Runnable recipes

A runnable recipe is the public discovery and deployment boundary for one exact operation. It binds a base tool to a fixed lineage, exact resource roles, operation defaults, bounded dynamic slots, and runtime policy.

Read the normative [model-roadmap authority policy](./model-roadmaps/README.md) before authoring or implementing a family.

## Boundaries

- A resource is an artifact identity and acquisition contract, not a runnable operation.
- A deployment profile is a saved selection of recipes and their deduplicated resource closure, not a recipe.
- A generic Comfy provider call is not native Engine acceptance.
- Cataloged, installed, structurally tested, runnable, Hardware-proven, and Recommended are distinct states.
- Reference and Recommended are independent: Reference is operation-matched comparison; Recommended is the accepted product default.

## Public key grammar

Use a stable family/variant prefix, operation, and implementation lineage:

```text
<family-or-line>.<operation>.<implementation>
```

Examples:

```text
flux2-klein-4b.text-to-image.bfl-distilled-nvfp4
flux2-klein-4b.image-to-image.comfy-distilled-fp8
wan-2-2-5b-ti2v.text-to-video.comfy-fp16
wan-2-2-5b-ti2v.image-to-video.comfy-fp16
```

Do not encode display prose, target GPU, transient experiments, or mutable branch names. Base versus Distilled, T2V versus I2V, ordinary versus KV, and one-stage versus two-stage are lineage/operation boundaries, not hidden flags.

## Current TOML shape

Do not invent fields in roadmaps and present them as authorable TOML.

```toml
[runnable_recipe]
key = "family.operation.implementation"
name = "Human name"
description = "Exact bounded operation"
family = "family"
base_tool = "family.tool_key"
tags = ["builtin", "operation", "lineage"]

[runnable_recipe.recipe]
type = "typed_recipe_contract"
base_model = "exact-base-model"
transformer = "resource:id"
text_encoder = "resource:id"
vae = "resource:id"

[runnable_recipe.optimizations]
keep_pipeline_loaded = true
```

Expert pairs, conditional/unconditional branches, audio/video VAEs, fixed LoRAs, prompt enhancers, support directories, and upscalers require reviewed typed roles. Do not hide them in arbitrary metadata or infer them from filenames.

Optional user LoRAs use declared slots only when the runtime validates base, target schema, rank/layout, source identity, and execution behavior. A workflow-fixed LoRA is a fixed recipe resource and participates in the fingerprint; it is not an optional user slot.

## Comfy-first operation authority

For a practical operation with an official Comfy graph, implementation starts from the pinned raw workflow—not from a dense Diffusers pipeline reconstructed from memory.

Authority is split deliberately:

1. Publisher source owns weights, architecture, configs, license, and dense Reference facts.
2. Pinned official Comfy raw JSON owns practical topology and saved defaults.
3. Pinned ComfyUI owns node schema, preprocessing, output slots, loading, and execution behavior.
4. Pinned Comfy Kitchen plus exact headers own low-bit layout, dispatch support, and fallback.
5. Engine public-API evidence owns availability, acceptance, and tier.

The workflow is not “behavioral inspiration.” Engine may compile it into native code or an isolated pinned worker, but deviations need separate fingerprints and acceptance.

Authoring baseline: workflow templates [`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1), package `comfyui-workflow-templates-json==0.1.37`. Family accepted baselines may differ and must remain exact historical provenance.

The mandatory normalization steps are in [model-roadmaps/README.md](./model-roadmaps/README.md#implementation-agent-preflight).

## Workflow compilation contract

Before a recipe is runnable:

1. fetch the exact raw workflow and retain commit, Git blob, bytes, and raw SHA-256;
2. expand all subgraphs/switches into a normalized API graph with dynamic placeholders;
3. verify exact ComfyUI object schema, required inputs, enum values, output indexes, and preprocessing;
4. enumerate the active artifact closure and separately record configured-but-disabled resources;
5. resolve immutable artifacts, licenses/gates, and headers; template `resolve/main` URLs are discovery only;
6. verify Kitchen/header compatibility and refuse unknown layout or fallback;
7. build independent fixtures from upstream evidence before implementation.

The normalized contract records active/disabled branches, artifact roles, prompt enhancement, media preprocessing/conditioning order, sampler/scheduler/sigmas, steps/stages, CFG/guidance, shift, dimensions, frame/fps rules, fixed LoRAs, dynamic slots, and output-object semantics.

## Reference versus practical ordering

Dense BF16 remains Reference when the publisher provides it. It does not automatically become the first local recipe.

For large video:

- source-pin and CPU-validate the dense closure;
- retain one bounded local OOM when useful;
- stop repeated local offload tuning;
- batch dense outputs on high-memory Vast;
- implement and accept the decisive official Comfy/FP8/ConvRot/NVFP4/fixed-LoRA path locally first.

An image-family dense path may still be first when it fits and the roadmap explains why the Comfy graph adds no material product value.

## Complete resource closure

A recipe includes every active fixed resource:

- denoiser/transformer or expert/branch pair;
- text or multimodal encoders;
- image/video/audio VAEs and vocoder;
- upscalers, patches, fixed LoRAs, and prompt enhancers;
- tokenizer, scheduler, processor, and support files;
- exact pinned worker/runtime dependency when applicable.

Do not omit small or subgraph-loaded artifacts. Do not include disabled artifacts in saved-default closure. Switch modes that change resources, schedule, or operation semantics normally become separate recipes.

Deployment planning unions selected recipe closures by stable resource identity, downloads shared components once, and reports incremental bytes/missing resources.

## Availability and fail-closed behavior

A recipe is available only when:

- every required resource is installed and revalidated;
- every role matches the typed contract and exact header/schema;
- runtime, Comfy checkout, package, codec, and backend dependencies are present;
- license/auth/gating requirements are satisfied;
- operation and input multiplicity are supported;
- the stored format has a proven loader/dispatch path;
- no active graph resource is missing.

False availability is a correctness defect. Engine must not silently swap lineages/operations, drop an expert/branch/VAE/upscaler/prompt enhancer/fixed LoRA, convert/dequantize stored weights, route to another backend under the same key, or infer output metadata from the request.

## Runtime fingerprint and provenance

The key includes recipe/contract revision, exact resources and schema fingerprints, workflow/ComfyUI/Kitchen pins, operation/input order/preprocessing/prompt enhancement, fixed LoRAs, schedule, dimensions/frames/fps, attention/offload/cache/compile/VAE policy, and keep-loaded policy.

Job provenance reports what actually ran: effective values, backend/dispatch/fallback, cache and cold/warm state, output slot, and observed artifact metadata. A submitted request is not backend proof.

## Merge gates

Reject graph drift, hidden native fallthrough, false availability, assumed output metadata, unobserved cancellation, cataloged-versus-runnable confusion, and fixtures generated by the implementation under test. See the central [review gates](./model-roadmaps/README.md#review-gates).

## Related documentation

- [Catalog authoring](./CATALOG_AUTHORING.md)
- [Hardware studies](./HARDWARE_STUDIES.md)
- [Authority policy](./model-roadmaps/README.md)
- [Implementation packets](./model-roadmaps/IMPLEMENTATION_PACKETS.md)
- [Source-pin audit](./model-roadmaps/SOURCE_PIN_AUDIT.md)
