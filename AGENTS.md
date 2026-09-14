# LatentSlate Engine project guidance

This repository is a greenfield rebuild. Historical Engine code remains in Git for recovery, not as design authority.

## Product and runtime boundary

LatentSlate Engine is a local, Engine-native inference service consumed by LatentSlate. It is not a graph engine, plugin host, or ComfyUI reimplementation. Current LatentSlate is authoritative for the external HTTP/tool contract; the distilled contract is `docs/ENGINE_CONTRACT.md`.

Keep model/runtime implementation independent from the service protocol. GPU/Torch/AIMDO/Kitchen/model state belongs below that boundary, preferably inside an isolated GPU worker.

Shared Engine, service, recipe, and authoring semantics must remain OS-agnostic across Windows, Linux, and macOS. CUDA execution is backend/hardware-dependent, currently Windows-tested and Linux-targeted. Linux NVIDIA deployment is a product constraint. Isolate and guard platform-specific optimizations; never make WDDM, Windows-only APIs, Linux-specific mechanisms, or host filesystem/process conventions part of inference correctness. Measure target-platform behavior before adding compensating mechanisms.

Reuse the same model identity maximally. A true model identity change must completely purge prior model/request state. If native state is unsafe or unknowable, worker replacement is a valid recovery boundary.

## Private hardening material

Custom checkpoints, LoRAs, other custom adapters, and anything that identifies them must remain outside this repository, including during hands-on hardening. Do not commit their files, names, paths, URLs, hashes, inventories, authored recipes, screenshots, outputs, logs, or identifying evidence in source, tests, docs, commit messages, PRs, or issues.

Support custom models/adapters through neutral format, architecture, loading, execution, and lifecycle behavior. Model/adapter selection belongs to the user. Do not choose, recommend, promote, bundle, advertise, or create named defaults/examples/compatibility lists for specific custom artifacts. Private testing may earn generic fixes and synthetic regression coverage only.

All Comfy reference workflows and raw experiment evidence are local-only and gitignored, including official baseline workflows. Never force-add `reference/`, `evidence/`, or `workflows/` material. Existing docs may describe local diagnostic paths that are absent from a clone; obtain the external reference when needed rather than reconstructing it.

Official built-in model identifiers required by product behavior may remain in source/docs. That does not authorize tracking model weights or reference workflows.

## App-managed state

`LatentSlateEngineData` and other app-managed runtime state are never source material. This includes Recipe Studio user recipes/revisions, authoring roots, downloaded/materialized artifacts, jobs, caches, and credentials. Use synthetic, non-identifying fixtures for durable tests; keep raw runtime evidence outside the repository.

## Earned architecture

Grow architecture only from proven implementations:

- Prefer operation- or family-local code until multiple working consumers demonstrate identical semantics.
- Small model-neutral utilities may be extracted after contrasting families prove the seam; a reusable framework abstraction should normally have at least three proven consumers.
- Modest duplication is preferable to speculative model managers, recipe/resource frameworks, plugin systems, or cross-family runtime abstractions.
- If a new family fights an extracted seam, reconsider the seam rather than adding adapters solely to preserve it.
- For compatibility/stress work, choose cases for the distinct assumption they can falsify; prefer an evidence-producing lattice over a mechanical Cartesian product.

## Local stack and process control

The loopback Local Process Manager is the canonical control path for the managed LatentSlate UI/Engine stack. Default endpoint: `http://127.0.0.1:47634`. It is development tooling, not an Engine product dependency.

Before operating the stack, discover `/health`, `/processes`, and when relevant `/groups` or `/topology`. Target the stable IDs returned by live discovery, never baked-in IDs/PIDs/display names/group membership. Prefer bounded individual-process control; after control requests, poll until the intended state is visible. Use the manager's configured UI build entry for LatentSlate release builds.

`POST /stack/reload` is broad and stops all managed processes before rereading configuration; do not use it as a routine refresh. If the manager is unavailable, report that rather than silently assuming stale topology or falling back to broad unmanaged process control.

## Comfy-derived inference

For any Comfy-derived implementation, parity, performance, or lifecycle work, **read `docs/COMFY_REFERENCE.md` first**. That document owns the pinned reference environment, reference-process discovery, workflow-fixture rules, certification procedure, telemetry conventions, lifecycle cases, AIMDO/Kitchen responsibilities, and diagnostic guidance.

Cross-cutting rules that always apply:

- The default Engine baseline is the **actual model selection and effective settings executed by Comfy's curated workflow**, frozen to a recorded revision. Inspect switches and executed graph state; links/notes or supplemental BF16/core examples do not override the curated default unless the user explicitly chooses them.
- Use Comfy as an executable behavioral oracle, not as Engine architecture. Do not port its graph executor, global model manager, node runtime, plugin machinery, UI policy, or broad `comfy.*` runtime.
- Establish the exact API workflow/reference runtime/assets/effective inputs before comparing. Prove the reference actually executed rather than being satisfied by graph caching.
- Compare semantic boundaries, not just similarly shaped tensors. For a new path, use coarse forward checkpoints; when a boundary differs, localize the first unexplained divergence and run the cheapest discriminating experiment that can falsify the current hypothesis.
- Treat source explanations as provisional until the live exercised branch supports them. Remove failed experiments and temporary probes; authoritative correctness/performance measurements run with diagnostic instrumentation removed or inactive.
- Reuse family behavior only when measured reference boundaries prove the semantics are the same. Similar names or graph shapes are not evidence.
- AIMDO/Kitchen own primitives they already implement; application code should express execution order and safe dependencies rather than recreate their global memory/quantization policies.
- Before performance comparison, establish a fresh equivalent Comfy baseline in the current pinned environment. Historical timing/memory numbers are not standing product gates.
- Do not invent determinism or bit-identity requirements where the pinned reference is itself nondeterministic. Performance/resource counters are diagnostics unless they establish a real safety invariant.

## Historical Engine quarantine

Do not inspect or copy pre-reset runtime code unless the user explicitly authorizes a bounded archaeology task. For historical public identifiers/product facts, prefer current `docs/`, current LatentSlate, or the relevant upstream reference. Legacy code is recovery material, not inspiration.

## Working discipline

Measure early on real hardware once a change is safe to benchmark. Preserve exact artifact/model provenance where it matters, but keep compatibility claims specimen-specific unless broader evidence exists. Do not turn temporary diagnostics, experiment harnesses, or one-off fixes into permanent frameworks without repeated proven need.
