# LTX 2.3 implementation target

LTX 2.3 is the first model family in the greenfield Engine.

Implement in this order:

1. text-to-video
2. image-to-video
3. first/last-frame-to-video

Do not implement the three paths concurrently.

## First principle

Prove inference before building the surrounding product shell.

The first executable milestone is a standalone Engine-native LTX 2.3 T2V core
that can run the pinned workflow on the fixed benchmark in one process and
measure a cold request followed by a same-context warm request.

Do not build the HTTP API, generic recipe/resource system, model manager,
serving layer, or cross-family framework during the LTX proving phase.

Keep the LTX implementation independently runnable and benchmarkable through all
three operation paths. The LatentSlate service contract in
`docs/ENGINE_CONTRACT.md` remains a future integration constraint, not work for
this family milestone.

## Runtime boundary

The intended eventual product behavior is simple:

- the service/API layer owns requests, jobs, assets, progress, and artifacts;
- GPU/native inference state belongs below that layer;
- an isolated GPU worker is the preferred failure boundary;
- one active model identity is sufficient;
- same identity should remain warm and reusable;
- a real identity switch completely destroys the previous context.

During this milestone, implement only the inference-side behavior needed to
prove those model identity/lifetime semantics. Do not build the general service
or a Comfy-like global model manager.

Worker replacement remains an acceptable future implementation of a destructive
model switch or unsafe native failure boundary.

## Canonical T2V reference

The operational reference workflow for the first milestone is:

`reference/comfy/ltx23/t2v-pytorch-baseline-api.json`

It must be a ComfyUI **Export (API)** prompt, not editable frontend workflow
JSON. It is supplied by the human after configuring the exact benchmark case;
do not reconstruct a replacement from the visual workflow.

It must be executed against the live-discovered Comfy process named exactly
`Comfy C (PyTorch Baseline)` using the ComfyUI Process Manager at
`http://127.0.0.1:47827` and the installed `comfy-local` MCP.

Use `comfy-local` to inspect the actual workflow nodes/settings, resolve their
implementations, execute the fixed reference case, and compare behavior. Do not
reconstruct the graph from memory or substitute another Comfy variant.

If the workflow file still contains the explicit placeholder marker, stop
reference execution and wait for the user-provided API-format export rather
than inventing a replacement.

## T2V benchmark

Routine resolution:

`512x512`

Final/heavy gate:

`768x768`

Do not use 1280x704 as an acceptance gate.

Prompt:

> Dynamic cinematic close-up of high-tech modular machinery self-assembling in
> midair, precision robotic parts, magnetic connectors, and glowing circuits
> clicking together, with a clean surface displaying large glowing engraved text
> “LTX-2.3” centered and unobstructed.

Contract:

- 5 requested seconds
- 30 fps
- 145 effective frames
- 4.833 seconds
- H.264
- AAC
- 48 kHz stereo

## Fresh Comfy baseline requirement

No timing or memory baseline is hard-coded in this document.

Before the first Engine performance comparison for the canonical fixture, run
that exact fixture on the live-discovered `Comfy C (PyTorch Baseline)` process
and establish a fresh reference on the current machine/environment.

Capture at minimum:

- cold wall-clock runtime;
- same-process warm wall-clock runtime;
- peak system RAM;
- peak GPU memory use;
- output/media contract and any material reference metadata needed to prove the
  run used the intended fixture.

The cold/warm procedure itself must be explicit enough to repeat. The warm run
must use the same surviving Comfy model context intended by the comparison.

If the canonical API fixture, model selections, frame cadence, pinned Comfy
runtime, relevant dependency versions, or materially relevant execution
environment changes, establish a new baseline before comparing Engine again.

Do not reuse historical timing/memory numbers from another workflow revision or
benchmark shape merely because they were previously measured in this project.

Initial parity objective:

approximately <=10% slower than the freshly measured matching Comfy baseline,
without materially worse RAM/VRAM behavior.

## Accepted 512 evidence — 2026-08-26

This evidence applies only to the exact API fixture
`reference/comfy/ltx23/t2v-pytorch-baseline-api.json`, including the verbatim
node 373 prompt value (its trailing newline is significant to repeat the
measurement). It is not a timeless timing target.

The matching reference used the live-discovered `Comfy C (PyTorch Baseline)`
process with pinned ComfyUI v0.34.0 at
`12d5279438bfefc058a269eae805ceab6047777f`, launched with
`--use-pytorch-cross-attention`. On this fixture/environment its cold run was
73.20 s. Its 37.58 s warm run retained that same process/model context; a
temporary submission changed only the noise seed to force a real execution
rather than return Comfy's graph-cached result.

The accepted Engine checkpoint is `573e22c` (`perf: prefetch LTX T2V LoRA
blocks`), atop `47ec36e` (`perf: retain LTX T2V dynamic source weights`). Its
recorded cold/warm result was 74.13 s / 40.07 s, against a warm comparison
threshold of 41.34 s. A no-change repeat that read the prompt directly from
the API fixture measured 39.35 s and 39.15 s for two same-identity warm
requests (cold: 62.33 s), confirming that the accepted warm result was not an
obvious timing outlier.

The repeated Engine output was verified with `ffprobe` as H.264 512x512, 145
frames at 30 fps and 4.833333 s, plus AAC stereo at 48 kHz and 4.833 s. The
decoded tensors were `[1, 145, 512, 512, 3]` video and `[1, 2, 240480]` audio.

The observed environment was an NVIDIA GeForce RTX 5080 (16,303 MiB, driver
610.47), PyTorch `2.11.0+cu130`, PyAV `18.0.0`, comfy-aimdo `0.4.15`, and
comfy-kitchen `0.2.31`. Refresh this evidence before comparison if the fixture,
model selections, pin/dependency versions, launch mode, or materially relevant
hardware/software environment changes.

The narrow implementation lesson is LTX-local: retain the source-conformant
dynamic source weights and stage/prefetch LoRA blocks in source-conformant
order. The ordinary-CUDA and resident-requant LoRA experiments regressed the
full path and were removed; they are not fallbacks or general Engine policy.

## Accepted 768 evidence — 2026-08-26

This evidence uses the same canonical API fixture and environment recorded
above. The matching 768 prompt was derived temporarily from
`reference/comfy/ltx23/t2v-pytorch-baseline-api.json` by changing only node
354's value, node 367's value, and node 376's width/height from 512 to 768. The
repo fixture itself remains unchanged. All prompt, model, seed, sigma, cadence,
and duration inputs remained canonical.

The live-discovered `Comfy C (PyTorch Baseline)` process measured 78.65 s cold,
then 43.07 s and 41.01 s for two real same-process warm executions. Each warm
submission changed only node 332's noise seed to bypass Comfy graph caching.
The 42.04 s warm median established a 46.24 s approximate 10% comparison
threshold. The Comfy process-tree working-set peak was 36,104,179,712 bytes and
total observed GPU use peaked at 15,068 MiB.

Engine checkpoint `78ee733` (`feat: pass LTX T2V 768 gate`) measured 67.65 s
cold, then 43.80 s and 43.82 s warm with one retained model identity (43.81 s
median). Its process-tree working-set peak was 39,270,604,800 bytes, about 8.8%
above the matching Comfy observation, and total observed GPU use peaked at
15,345 MiB. The output tensors were `[1, 145, 768, 768, 3]` video and
`[1, 2, 240480]` audio. `ffprobe` verified H.264 768x768, 145 frames at 30 fps
and 4.833333 s, plus AAC stereo at 48 kHz and 4.833 s. A representative frame
preserved the reference's clear `LTX-2.3` sign and scene semantics.

The bounded 768 changes kept the accepted 512 substrate and added only the two
canonical resolution shapes. The performance changes follow the pinned source's
large-attention SDPA backend priority and allow in-place LoRA addition only for
disposable dequantized weights; persistent plain weights retain the
non-mutating path. Factorized LoRA, resident/requantized LoRA, and indiscriminate
in-place LoRA experiments failed performance or correctness checks and were
removed.

## Accepted I2V evidence — 2026-08-26

This evidence applies only to
`reference/comfy/ltx23/i2v-pytorch-baseline-api.json` and its canonical 512x512
PNG source, SHA-256
`F293EE0ABA3CEBDA198D8223D140CE714FC8D10EE00F4529F13FE8D4F1A667C0`.
The source fixture is a real API export with 51 nodes and no placeholder marker.

The matching reference used the live-discovered `Comfy C (PyTorch Baseline)`
process and the same pinned environment recorded for T2V. Its cold execution
was 72.97 s. A real same-process warm execution changed only the first-stage
noise seed at node 331 and measured 37.07 s, establishing a 40.78 s approximate
10% comparison threshold. The Comfy process-tree working-set peaks were
41,544,302,592 bytes cold and 41,269,227,520 bytes warm; total observed GPU use
peaked at 15,566 MiB cold and 15,467 MiB warm.

Engine checkpoint `485a4e7` (`feat: run canonical LTX I2V operation`) measured
66.12 s cold, then 38.19 s and 37.91 s for two same-identity warm requests.
Their complete generation-and-MP4 times were 40.01 s and 39.65 s. A monitored
run peaked at 39,998,484,480 bytes of process-tree working set and 15,464 MiB of
total observed GPU use. Closing the runtime released its retained inference
context, while same-identity requests reused it.

The output tensors were `[1, 145, 512, 512, 3]` video and
`[1, 2, 240480]` audio. `ffprobe` verified H.264 512x512, 145 frames at 30 fps
and 4.833008 s, plus AAC stereo at 48 kHz and 4.833000 s. The repeated Engine
MP4 SHA-256 was
`3BD5324BF2C041D0CCA7AA7D0DF8DC3E7EACD1243484B512A70CC31E61556D57`.
Decoded frame inspection confirmed that frame zero preserves the supplied
workshop/bird source, followed by the prompted Egyptian royal and robot
formation with the prescribed forward push-in.

The standalone I2V path adds only canonical image preprocessing and VAE encode,
conditioned latent masks, and masked two-stage sampling. It reuses the proven
T2V transformer, decoders, vocoder, upsampler, and media writer without changing
the frozen T2V operation or introducing a generalized LTX-family layer.

## Development sequence

### 1. Reference

Verify the canonical repo workflow is a real API-format export, not a frontend
workflow or placeholder stub.

Run it on `Comfy C (PyTorch Baseline)` using `comfy-local` and establish the
fresh cold/warm reference measurements above **before making any Engine
performance comparison**.

If fresh matching reference evidence does not exist yet, do not infer a target
from historical docs, old Engine evidence, or a different Comfy workflow.

Trace the exact T2V workflow and relevant persistent state from that execution
into the pinned Comfy/AIMDO/Kitchen source.

### 2. Standalone T2V — 512

Bootstrap only enough Python/package/test infrastructure to execute and benchmark
T2V.

Use AIMDO/Kitchen directly and source-port narrowly from pinned Comfy where
appropriate.

Benchmark hardware as soon as a valid generation exists.

Reach the 512 correctness/performance target before broadening the path.

### 3. Standalone T2V — 768

Run the same proven T2V substrate through the 768x768 heavy gate.

Treat problems exposed by 768 as evidence about the T2V implementation, not as
permission to build a general memory/runtime framework.

T2V is complete only after both 512 and 768 gates pass.

### 4. I2V

Trace the pinned Comfy I2V workflow before implementation.

Reuse only T2V behavior that has already proved genuinely common. Temporary
duplication is acceptable while the second operation establishes the family
seam.

Expected new work should primarily concern image conditioning, VAE encode,
initial latent construction, and masks rather than a new memory/residency system.

Establish equivalent correctness, memory, and performance comparisons against a
fresh matching PyTorch Comfy reference for the canonical I2V fixture.

### 5. FLF

Trace the pinned Comfy FLF workflow before implementation.

Add the genuinely different first/last-frame conditioning and distinct model
identity. Temporary duplication remains acceptable while this third operation
proves what the LTX family actually shares.

Verify that switching to the FLF identity actually purges the previous model
context.

Establish equivalent correctness, memory, and performance comparisons against a
fresh matching PyTorch Comfy reference for the canonical FLF fixture.

### 6. LTX-family consolidation

Only after T2V, I2V, and FLF are all working and measured, perform a bounded
family-local deduplication pass.

Extract only behavior that the three proven LTX operation paths actually share.
Do not promote LTX structures to model-neutral Engine framework code during this
pass.

Prefer a little remaining duplication over an abstraction whose semantics have
not yet been exercised outside LTX.

Then stop the LTX milestone.

The next evidence stage is a contrasting second model family implemented on its
own terms. Cross-family extraction is a later task governed by `AGENTS.md`.

## Correctness

Do not make the historical Engine implementation a correctness oracle.

Do not require bit-identical cold/warm outputs unless an exact same-input pinned
Comfy experiment establishes that behavior.

Media validity, workflow semantics, reference-source behavior, and explicit
numerical comparisons are the authority.
