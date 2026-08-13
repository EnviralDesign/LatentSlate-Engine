# Hardware and acceptance studies

Hardware studies are the authority for Engine proof level and product tier. Publisher measurements, workflow defaults, header inspection, catalog publication, and unit tests are inputs to a study; none of them is an Engine result.

Read the normative [model authority policy](./model-roadmaps/README.md), especially the [implementation-agent preflight](./model-roadmaps/README.md#implementation-agent-preflight) and [review gates](./model-roadmaps/README.md#review-gates).

## Evidence boundary

A study is valid only when it runs through the public Engine catalog/job/artifact surface and retains enough evidence to reproduce what actually happened:

- Engine commit and dirty/clean state;
- recipe key and typed contract revision;
- exact resource identities, file/header/schema fingerprints, and acquisition pins;
- normalized upstream workflow hash plus ComfyUI/Kitchen/runtime pins where applicable;
- submitted request and effective request after validation/alignment;
- observed backend, native dispatch counters, fallback counters, and component residency;
- cold/warm/cache state and phase timing;
- cancellation and cleanup observations;
- output-object slot and observed artifact metadata;
- output bytes and SHA-256;
- allocator and approximate device/process/system memory;
- creator-review notes and known failure modes.

A submitted request is not proof that the backend honored it. Artifact metadata must come from probing the produced file or from a backend response whose semantics are pinned and tested.

## Proof levels

| Proof level | Minimum evidence |
| --- | --- |
| Cataloged | exact declaration and deterministic closure only |
| Structurally tested | independent fixtures validate graph/header/loader contract without accepted output |
| Runtime-proven | one end-to-end Engine job with observed backend and valid artifact |
| Hardware-proven | target-class output plus lifecycle, memory, provenance, and recovery evidence |
| Recommended/Fallback | Hardware-proven plus operation-matched creator review and an explicit product decision |

Do not promote a recipe because its catalog entry appears available, a model object loaded, a fixture passed, or a file was written.

## Study manifest

Retain one machine-readable manifest per job or scenario. It should include:

```json
{
  "engine_commit": "40-char-sha",
  "recipe_key": "family.operation.implementation",
  "authority": {
    "workflow_commit": "40-char-sha",
    "workflow_blob": "40-char-git-blob",
    "workflow_sha256": "64-char-sha256",
    "comfyui_commit": "40-char-sha-or-null",
    "kitchen_commit": "40-char-sha-or-null"
  },
  "request": {},
  "effective_request": {},
  "resources": [],
  "runtime": {
    "backend": "observed-backend",
    "native_dispatch": {},
    "fallbacks": {},
    "cold_start": true,
    "cache": {}
  },
  "timings_seconds": {},
  "memory": {},
  "artifact": {
    "sha256": "64-char-sha256",
    "observed_metadata": {}
  },
  "cancellation": null
}
```

The schema may evolve, but evidence categories may not be replaced by prose-only summaries.

## Required local scenario matrix

For a practical RTX 5080 candidate:

1. **Preflight only:** source/graph/header validation with no payload execution.
2. **Runtime-cold:** fresh runtime or disposable worker, fixed prompt/media/seed.
3. **Three meaningful warm jobs:** changed seed or prompt so execution occurs; label any execution-cache replay separately.
4. **A-to-B-to-A switching:** prove recipe/component fingerprints, cache invalidation, and memory ownership.
5. **Malformed artifact:** fail before expensive allocation when possible.
6. **Cancellation/recovery:** cancel during every meaningful lifecycle phase, observe cleanup, then complete a fresh job.
7. **Multi-input cases:** exact official multiplicity and order, followed by any clearly labeled Engine extension.
8. **Explicit teardown:** runtime/worker gone, temporary output removed, owned caches cleared, expected memory returned.
9. **Creator corpus:** fixed family-specific prompts/media reviewed blind where comparisons matter.

## Cancellation evidence

A canceled API state is not enough. Record:

- cancellation request and terminal job state;
- worker/process tree exit or in-process runtime eviction;
- accelerator synchronization and memory release;
- temporary input/output cleanup;
- prompt/reference/cache invalidation;
- poisoned-state ejection;
- a fresh recovery job with exact provenance.

If a third-party call cannot be interrupted inside one phase, document the cooperative boundary and ensure the uncertain runtime is discarded afterward.

## Native and quantized dispatch evidence

A low-bit path is accepted only when:

- exact header/schema validation matches the planned source-to-target map;
- every intended Kitchen/native module reports positive dispatch;
- no eligible module silently uses eager/dense/dequantized execution;
- dense exceptions are expected and enumerated;
- sidecars/scales/aliases remain intact through assignment;
- LoRA application does not dequantize the base unless the recipe explicitly says so and is separately classified;
- the manifest records Kitchen version/source and backend capability.

A Kitchen import, startup banner, capability probe, or successful image/video is not dispatch proof.

## Output observation

Probe produced artifacts independently. For images record dimensions, color mode/bit depth where relevant, format, and hash. For video/audio record:

- container and stream codecs;
- encoded width/height;
- frame count and observed frame rate;
- duration and time base;
- audio sample rate, channel count/layout, duration, and start/end alignment;
- whether audio is absent by design;
- output bytes and SHA-256.

Never populate these fields solely from the request or tool descriptor. A graph may save at a different fps, emit a different slot, or omit audio.

## Memory and timing

Separate:

- Engine API elapsed time;
- worker startup;
- source/media preprocessing;
- text/vision encoding;
- model materialization and staging;
- each denoise/refinement stage;
- VAE/video/audio decode;
- mux/export/download;
- teardown.

Record Torch allocator peaks when meaningful, device-wide sampled VRAM, process RSS/private bytes, system/Windows commit, and disk/PCIe traffic where practical. Device polling is approximate and must be labeled as such.

## Dense BF16 video policy

Dense video Reference is not a recurring local optimization target.

For Wan, LTX, H3, and similarly large references:

1. pin the publisher repository and exact operation closure;
2. validate configs, file list, shards, schema, and request contract on CPU;
3. record one bounded local OOM if it establishes the workstation capability gate;
4. stop repeated local retries and pathological offload tuning;
5. batch dense reference outputs on high-memory Vast hardware using operation-matched prompts/media/settings;
6. retain cloud hardware, driver/runtime, memory, timings, output hashes, and costs in the same manifest format.

Local 5080 time should qualify exact practical official Comfy paths and first-party stored FP8/ConvRot/NVFP4/fixed-LoRA candidates. A local diagnostic with reduced dimensions or duration is allowed only when labeled non-parity; it does not replace the full reference case.

## Comparison rules

A precision/layout comparison holds constant:

- operation and lineage;
- prompt and ordered media;
- raw/effective prompt enhancement mode;
- dimensions, frames, fps, and preprocessing;
- sampler/scheduler/sigmas, steps/stages, CFG/guidance, shift, denoise;
- fixed LoRAs and strengths;
- output container and postprocessing.

A four-step Lightning path is compared first against BF16 plus the same Lightning LoRA, not against a 40-step teacher as though only precision changed. Base, Distilled, KV, T2V, I2V, and first/last-frame results have separate ladders.

## Creator review

Operational success is necessary but insufficient. The corpus should expose family-specific failure modes:

- image: text rendering, anatomy, identity, untouched-region stability, composition, color/material, long prompts;
- video: temporal coherence, identity, motion onset/arrival, camera intent, endpoint fidelity, texture, looping/freezing;
- audio-video: dialogue/singing/music/foley, action-to-sound timing, lip sync, channel placement, clipping, silence, drift;
- multi-reference: input order, source contribution, stale-cache reuse, and extension behavior.

Record reviewer decisions separately from objective metadata. Promotion requires a creator-visible reason, not merely lower memory or a different hash.

## Review gates

Reject the study when:

- workflow or artifact authority is mutable/unresolved;
- normalized graph differs from the recipe without an explicit deviation;
- runtime reports hidden fallback or missing dispatch;
- availability was inferred from installed files;
- output metadata was assumed;
- cancellation cleanup was not observed;
- warm results are only cache replays;
- dense reference settings were silently weakened;
- comparison changes lineage, schedule, prompt enhancement, or operation;
- fixtures or expected outputs came from the implementation under test.

## Related documentation

- [Runnable recipes](./RECIPES.md)
- [Catalog authoring](./CATALOG_AUTHORING.md)
- [Authority policy](./model-roadmaps/README.md)
- [Implementation packets](./model-roadmaps/IMPLEMENTATION_PACKETS.md)
