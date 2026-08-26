# Comfy LTX 2.3 parity fixtures

This directory contains the canonical Comfy workflows used for Engine parity
work. Keep only known-good reference fixtures here, not exploratory workflows.

## T2V

`t2v-pytorch-baseline.json` is the canonical operational T2V reference fixture.

It is intended to be a flattened, user-prepared derivative of ComfyUI's official
`Text to Video (LTX-2.3)` workflow at pinned ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f` (v0.34.0).

The flattening exists so the installed `comfy-local` MCP can inspect and execute
the workflow reliably without depending on subgraph handling. Flattening must
not intentionally change the effective inference path.

The repo fixture defines the concrete parity case, including any user-corrected
model selections or settings. Do not silently alter it while debugging Engine.
Intentional changes may invalidate prior benchmark evidence and should be
re-baselined.

### Execution environment

Use the ComfyUI Local Process Manager at:

`http://127.0.0.1:47827`

Always discover `GET /processes` live and locate the entry whose display name is
exactly:

`Comfy C (PyTorch Baseline)`

Target the ID returned by that discovery. Do not store its current UUID, PID, or
status in this repository, and do not substitute Sage or another Comfy process
for parity runs unless the benchmark is deliberately changed.

Use the installed `comfy-local` MCP to inspect nodes/settings, resolve node
implementations, execute the workflow, and inspect reference results.

## Placeholder safety

The initial `t2v-pytorch-baseline.json` committed by the greenfield bootstrap is
an explicit non-Comfy placeholder. Replace the entire file with the real
flattened workflow before any reference run.

If `_latentslate_reference_placeholder` is present and true, the file is not a
valid parity fixture and must not be executed or used as source evidence.

Future canonical fixtures should follow the same naming pattern:

- `i2v-pytorch-baseline.json`
- `flf-pytorch-baseline.json`
