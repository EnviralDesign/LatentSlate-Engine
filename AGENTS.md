# LatentSlate Engine project guidance

This repository is a greenfield rebuild. Historical Engine code is preserved in
Git but is not design authority.

## Product boundary

LatentSlate Engine is a local, Engine-native inference service consumed by
LatentSlate. It is not a graph engine, plugin host, or ComfyUI reimplementation.

Current LatentSlate is authoritative for the external Engine HTTP/tool contract.
The distilled contract is in `docs/ENGINE_CONTRACT.md`.

Model/runtime implementation stays independent from the service protocol. GPU,
Torch, AIMDO, Kitchen, model weights, and native CUDA state belong below the
service boundary, preferably inside an isolated GPU worker.

Same model identity should be maximally warm and reusable. A real model identity
change must completely purge the previous model context. If native state becomes
unsafe or unknowable, worker replacement is a valid recovery boundary.

Do not build a general model manager, recipe framework, resource framework,
plugin system, or cross-family inference abstraction before working model
families demonstrate that such a seam is actually shared.

## Local stack and process control

The local development stack is normally controlled through the loopback-only
Local Process Manager REST API.

Current default control endpoint:

`http://127.0.0.1:47634`

Treat this endpoint as local development tooling, not an Engine product API or
runtime dependency.

Prefer the Process Manager for starting, stopping, restarting, inspecting, and
reading logs from locally managed LatentSlate/Engine processes. Do not replace
it with ad-hoc process spawning, process-name killing, or baked-in PIDs when the
manager is available.

### Discover before acting

Process Manager definitions are mutable external state. Never copy currently
observed process IDs, group IDs, display names, membership, PIDs, status, CPU,
or RAM values into Engine source, tests, scripts, or durable project guidance.

Before operating the stack:

1. `GET /health` to confirm the manager is reachable.
2. `GET /processes` to discover current process IDs and state.
3. `GET /groups` when group control is relevant, or `GET /topology` when the
   current relationship between entries matters.
4. Target individual processes by the stable ID returned by live discovery,
   not by display name.
5. Target groups by the group ID returned by live discovery.
6. After any control `POST`, poll `GET /processes` or `GET /groups` until the
   intended state is actually visible.

Useful read endpoints:

- `GET /processes/{id}`
- `GET /processes/{id}/logs?limit=N`
- `GET /groups/{id}`
- `GET /topology`

Control endpoints:

- `POST /processes/{id}/start`
- `POST /processes/{id}/stop`
- `POST /processes/{id}/restart`
- `POST /processes/{id}/reload`
- `POST /groups/{id}/start`
- `POST /groups/{id}/stop`
- `POST /groups/{id}/restart`
- `POST /stack/start`
- `POST /stack/stop`
- `POST /stack/restart`

Use stack- or group-wide actions only when the requested operation actually
applies to that whole discovered set. Individual process control is preferred
for bounded development work.

### Reload semantics

`POST /processes/{id}/reload` rereads only that process definition from the
external `processes.json`.

`POST /stack/reload` is materially broader: it rereads the stack definition and
**stops all managed processes first**, regardless of their current status or
stack-control settings. Do not use stack reload as a routine restart or
refresh operation.

Group definitions also live in the external Process Manager configuration. If
regrouping is required, edit that external configuration and then deliberately
reload the stack. Do not mirror group membership into this repository.

The absence or presence of any particular group/process is not an Engine
invariant. Always rediscover the current topology.

If the Process Manager API is unavailable, report that fact rather than
silently assuming stale IDs or falling back to broad unmanaged process control.
Use another launch/control path only when the user explicitly asks for it or the
current task requires bootstrapping the manager itself.

## Source authority for Comfy-derived inference

For LTX 2.3:

1. official pinned Comfy workflow behavior;
2. pinned ComfyUI source;
3. comfy-aimdo source;
4. comfy-kitchen source;

are authoritative for the behavior they own.

Read `docs/COMFY_REFERENCE.md` before Comfy-derived implementation work.

For T2V parity, the canonical operational workflow fixture is:

`reference/comfy/ltx23/t2v-pytorch-baseline.json`

Use the installed `comfy-local` MCP as the preferred interface for loading the
fixture, inspecting its nodes, resolving node implementations/source, executing
reference runs, and comparing results.

Reference execution must use the ComfyUI Process Manager at
`http://127.0.0.1:47827`. Discover `/processes` live and select the process whose
current display name is exactly `Comfy C (PyTorch Baseline)`, then target the ID
returned by that discovery. Do not bake its current UUID into project files.
Do not substitute Sage or another Comfy process for parity measurements unless
the user explicitly requests it.

If the canonical workflow file is still the explicit placeholder stub, do not
invent or reconstruct a replacement. The user-provided flattened workflow must
replace it before reference execution.

Use Comfy as an executable source reference:

- trace the working workflow into the exact model/operation path;
- trace which state survives calls, stages, and requests;
- reproduce the smallest relevant state transitions;
- use AIMDO/Kitchen directly when they already own the primitive;
- narrowly adapt upstream source when licensing permits and doing so reduces
  semantic drift.

Do not translate Comfy nouns into Engine abstractions merely because they exist
in Comfy.

Do not import or reproduce Comfy's graph executor, global model manager, node
runtime, plugin machinery, UI policy, or broad `comfy.*` runtime.

## Historical Engine quarantine

The pre-reset tag exists for recovery, not implementation guidance.

Do not inspect or copy historical Engine runtime code unless the user explicitly
authorizes a bounded archaeology task.

If a historical public identifier or product fact is needed, prefer the
distilled contracts in `docs/`. If a fact is missing, inspect current LatentSlate
or the relevant upstream reference before consulting old Engine.

A bounded legacy scout may answer a specific factual question; the implementing
agent should not browse the historical runtime for inspiration.

## LTX 2.3 execution order

Read `docs/LTX23_TARGET.md`.

Implement in this order:

1. T2V
2. I2V
3. first/last-frame video

T2V must be proven before its substrate is generalized.

Measure early on real hardware. Once a change is safe to benchmark, benchmark it
before speculative cleanup or architecture work.

Performance counters are diagnostics, not product contracts. They must not
reject an otherwise valid generation unless they prove an actual safety
invariant.

Do not invent determinism or bit-identity requirements unless the pinned
reference demonstrates them under the same inputs.
