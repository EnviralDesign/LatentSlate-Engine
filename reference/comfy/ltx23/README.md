# Comfy LTX 2.3 parity fixtures

This directory contains the canonical Comfy workflows used for Engine parity
work. Keep only known-good reference fixtures here, not exploratory workflows.

## T2V

`t2v-pytorch-baseline-api.json` is the canonical operational T2V reference
fixture.

It is a user-provided **File > Export (API)** payload from ComfyUI's official
`Text to Video (LTX-2.3)` workflow at pinned ComfyUI commit
`12d5279438bfefc058a269eae805ceab6047777f` (v0.34.0). It is a JSON object
keyed by node ID; every entry carries a `class_type` and resolved `inputs`.

The API export is the exact queue payload, allowing `comfy-local` to inspect and
execute it without reproducing frontend workflow/subgraph behavior. Do not
hand-edit it while debugging Engine. Keep any frontend-format workflow only as
a separate human-editable companion; it is not parity evidence.

`comfy-local` can execute this API export, but its `list_workflow_slots`
inspector is frontend-format-only. Inspect API node objects directly when
tracing settings; do not treat that inspector's format error as a fixture
failure.

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

## Reference safety

If the expected API export is absent, a frontend-format workflow, or an explicit
placeholder, stop before reference execution and have the human put the
working API export in this directory. Do not construct a substitute.

If `_latentslate_reference_placeholder` is present and true, the file is not a
valid parity fixture and must not be executed or used as source evidence.

Future canonical fixtures should follow the same API-export naming pattern:

- `i2v-pytorch-baseline-api.json`
- `flf-pytorch-baseline-api.json`
