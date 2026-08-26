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

> A tiny silver wind-up bird flutters across a sunlit workshop table, its metal
> wings clicking softly as the camera follows at eye level.

Contract:

- 5 requested seconds
- 30 fps
- 145 effective frames
- 4.833 seconds
- H.264
- AAC
- 48 kHz stereo

Pinned 512 Comfy baseline:

- cold: 141.709 seconds
- warm: 44.157 seconds
- peak RAM: 60.484 / 61.393 GiB
- peak GPU use: 15,097 / 15,074 MiB

Initial parity objective:

approximately <=10% slower than pinned Comfy without materially worse RAM/VRAM
behavior.

## Development sequence

### 1. Reference

Verify the canonical repo workflow is a real API-format export, not a frontend
workflow or placeholder stub.

Run or verify it on `Comfy C (PyTorch Baseline)` using `comfy-local`.

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

Establish equivalent correctness, memory, and performance comparisons against the
pinned PyTorch Comfy reference.

### 5. FLF

Trace the pinned Comfy FLF workflow before implementation.

Add the genuinely different first/last-frame conditioning and distinct model
identity. Temporary duplication remains acceptable while this third operation
proves what the LTX family actually shares.

Verify that switching to the FLF identity actually purges the previous model
context.

Establish equivalent correctness, memory, and performance comparisons against the
pinned PyTorch Comfy reference.

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
