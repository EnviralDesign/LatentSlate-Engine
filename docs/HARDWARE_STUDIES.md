# Hardware and acceptance studies

Hardware studies own Engine proof level and product tier. Publisher measurements,
workflow defaults, header inspection, catalog publication, and unit tests are inputs;
none is an Engine result.

Read [COMFY_ENGINE_POLICY.md](./COMFY_ENGINE_POLICY.md) and the
[model-roadmap preflight](./model-roadmaps/README.md#implementation-agent-preflight).

## Evidence boundary

A valid study runs through the public Engine catalog/job/artifact API and records:

- Engine commit and recipe/typed-contract revision;
- exact resources, headers, and acquisition pins;
- normalized workflow hash and ComfyUI source revision as research provenance;
- Kitchen revision and direct primitives used;
- submitted and effective requests;
- observed native dispatch and fallback counters;
- component residency, cold/warm/cache state, timings, and memory;
- cancellation, worker exit, cleanup, and recovery;
- observed output slot/media metadata, bytes, and SHA-256;
- creator review and limitations.

ComfyUI is never launched or imported during a study. A study that depends on a
ComfyUI process, graph queue, server, plugin, or checkout is nonconforming and cannot
establish Engine acceptance.

The ignored manifests are the detailed local evidence. Accepted claims also
require a compact retained record under [`evidence/acceptance`](../evidence/README.md).
The validator binds each claim to a runtime-relevant source fingerprint,
resource closure, observed fallback counters, and creator review without
retaining prompts, media, credentials, absolute paths, or account/job IDs.

## Proof levels

| Level | Minimum evidence |
| --- | --- |
| Cataloged | exact declaration and closure |
| Structurally tested | independent behavioral/header fixtures |
| Runtime-proven | one end-to-end Engine job with observed backend |
| Hardware-proven | target output plus lifecycle/memory/provenance |
| Product tier | Hardware-proven plus creator review and decision |

## Required local matrix

For a practical RTX 5080 candidate:

1. CPU/source/header preflight;
2. runtime-cold Engine-owned worker;
3. three meaningful warm executions, not output-cache replay;
4. A-to-B-to-A switching;
5. malformed-artifact failure;
6. cancellation during each meaningful phase, observed cleanup, fresh recovery;
7. exact official input multiplicity/order, then separately labeled Engine extensions;
8. explicit teardown and expected memory return;
9. fixed creator corpus.

## Cancellation evidence

A canceled API state is insufficient. Observe terminal state, Engine worker/process
exit or in-process ejection, accelerator synchronization, temp cleanup, cache
invalidation, poisoned-state eviction, memory return, and a fresh successful job.

## Direct Kitchen/native dispatch

A low-bit path is accepted only when exact headers match the one-to-one map, every
intended module reports positive direct Kitchen/native dispatch, dense exceptions are
enumerated, sidecars/aliases survive assignment, and eligible modules report zero
eager/dense/dequantized fallback.

Kitchen availability or a successful artifact is not dispatch proof.

## Output observation

Probe produced files independently. Record image dimensions/mode/format/hash or video
container, codecs, dimensions, frame count/rate, duration/time base, audio
rate/channels/layout/duration/alignment, and bytes/hash. Do not populate these solely
from the request.

## Dense BF16 video

Pin and CPU-validate dense publisher references. Keep one bounded local OOM when useful,
stop repeated RTX 5080 fitting attempts, and batch operation-matched dense outputs on
high-memory Vast. Local work prioritizes Engine-native stored practical paths derived
from official workflow evidence.

A reduced local diagnostic is non-parity and cannot replace the full reference.

## Comparison rules

Hold operation, lineage, prompt/media order, enhancement, preprocessing, dimensions,
frames/fps, sampler/scheduler/sigmas, steps/stages, CFG/guidance, shift, LoRAs, and
output policy constant. Separate Base/Distilled, teacher/Lightning, T2V/I2V/FLF, and
single/two-stage ladders.

## Review gates

Reject any study with mutable/unresolved authority, normalized-contract drift,
forbidden external-UI execution dependency, hidden fallback, false availability, assumed metadata,
unobserved cancellation, cache-only “warm” results, weakened reference settings, or
fixtures generated from the implementation under test.

## Related documentation

- [Execution policy](./COMFY_ENGINE_POLICY.md)
- [Recipes](./RECIPES.md)
- [Catalog authoring](./CATALOG_AUTHORING.md)
- [Implementation packets](./model-roadmaps/IMPLEMENTATION_PACKETS.md)
