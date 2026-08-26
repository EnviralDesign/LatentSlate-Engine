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

Do not build the full HTTP API, generic recipe/resource system, model manager, or
cross-family framework before this core is proven.

Once standalone inference is correct and fast, integrate that proven core behind
the smallest LatentSlate-compatible service/worker shell and verify that
integration does not materially regress it.

## Runtime boundary

The intended product behavior is simple:

- the service/API layer owns requests, jobs, assets, progress, and artifacts;
- GPU/native inference state belongs below that layer;
- an isolated GPU worker is the preferred failure boundary;
- one active model identity is sufficient;
- same identity should remain warm and reusable;
- a real identity switch completely destroys the previous context.

Worker replacement is an acceptable and intentionally simple implementation of a
destructive model switch or unsafe native failure.

Do not build a Comfy-like global model manager.

## Canonical T2V reference

The operational reference workflow for the first milestone is:

`reference/comfy/ltx23/t2v-pytorch-baseline.json`

It must be executed against the live-discovered Comfy process named exactly
`Comfy C (PyTorch Baseline)` using the ComfyUI Process Manager at
`http://127.0.0.1:47827` and the installed `comfy-local` MCP.

Use `comfy-local` to inspect the actual workflow nodes/settings, resolve their
implementations, execute the fixed reference case, and compare behavior. Do not
reconstruct the graph from memory or substitute another Comfy variant.

If the workflow file still contains the explicit placeholder marker, stop
reference execution and wait for the user-provided flattened workflow rather
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
- 25 fps
- 121 effective frames
- 4.84 seconds
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

Verify the canonical repo workflow is the real flattened fixture, not the
placeholder stub.

Run or verify it on `Comfy C (PyTorch Baseline)` using `comfy-local`.

Trace the exact T2V workflow and relevant persistent state from that execution
into the pinned Comfy/AIMDO/Kitchen source.

### 2. Standalone T2V

Bootstrap only enough Python/package/test infrastructure to execute and benchmark
T2V.

Use AIMDO/Kitchen directly and source-port narrowly from pinned Comfy where
appropriate.

Benchmark hardware as soon as a valid generation exists.

### 3. T2V integration

Only after standalone 512 parity is credible:

- add the minimal Engine service/worker integration required by
  `ENGINE_CONTRACT.md`;
- rerun the same benchmark;
- investigate any integration overhead before adding features.

Then pass the 768 gate.

### 4. I2V

Trace pinned Comfy I2V.

Reuse only the T2V substrate that has already proved common.

Expected new work should primarily concern image conditioning, VAE encode,
initial latent construction, and masks rather than a new memory/residency system.

### 5. FLF

Trace pinned Comfy FLF.

Add the genuinely different first/last-frame conditioning and distinct model
identity.

Verify that switching to the FLF identity actually purges the previous model
context.

## Correctness

Do not make the historical Engine implementation a correctness oracle.

Do not require bit-identical cold/warm outputs unless an exact same-input pinned
Comfy experiment establishes that behavior.

Media validity, workflow semantics, reference-source behavior, and explicit
numerical comparisons are the authority.
