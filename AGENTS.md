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

## Source authority for Comfy-derived inference

For LTX 2.3:

1. official pinned Comfy workflow behavior;
2. pinned ComfyUI source;
3. comfy-aimdo source;
4. comfy-kitchen source;

are authoritative for the behavior they own.

Read `docs/COMFY_REFERENCE.md` before Comfy-derived implementation work.

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
