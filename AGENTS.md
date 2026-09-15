# LatentSlate Engine project guidance

Greenfield rebuild: historical Engine code is for recovery, not design authority. Do not inspect or copy pre-reset runtime code without explicit authorization for bounded archaeology. For historical public product facts/identifiers, prefer current docs, LatentSlate or upstream references.

## Product and runtime boundary

Engine is a local, native inference service consumed by LatentSlate, whose current HTTP/tool contract is authoritative; see `docs/ENGINE_CONTRACT.md`. Keep model implementation and GPU/Torch/AIMDO/Kitchen state below the service-protocol boundary, preferably in an isolated GPU worker.

Shared service, recipe and authoring semantics must remain OS-agnostic across Windows, Linux and macOS. CUDA is hardware/backend-dependent, currently Windows-tested and Linux-targeted; Linux NVIDIA deployment is a product constraint. Guard platform-specific optimizations. Inference correctness must not depend on WDDM, platform-only APIs or host filesystem/process conventions. Measure target-platform behavior before adding compensating mechanisms.

Maximize reuse within one model identity. A true identity change must purge all prior model/request state; worker replacement is valid when native state is unsafe or unknowable.

## Private hardening material

Use `C:/Users/envir/Documents/LatentSlate-Diagnostics/README.md` for local reference workflows, pairings, specimens and evidence. Agents may maintain/reorganize that external workspace to keep the repository clean, without publishing its contents.

Custom checkpoints, LoRAs, other custom adapters and identifying material must stay outside this repository, including during hardening. Never commit their files, names, paths, URLs, hashes, inventories, authored recipes, screenshots, outputs, logs or identifying evidence in source, tests, docs, commit messages, PRs or issues.

Support custom artifacts through neutral format, architecture, loading, execution and lifecycle behavior. Selection belongs to the user: do not choose, recommend, promote, advertise, bundle or create named defaults/examples/compatibility lists for specific custom artifacts. Private testing may yield only generic fixes and synthetic, non-identifying regression coverage.

All reference workflows and raw experimental evidence, including official baselines, are local-only and gitignored. Never force-add `reference/`, `evidence/` or `workflows/`. If a documented local reference is absent from a clone, obtain it externally rather than reconstructing it. Official built-in identifiers needed by product behavior may remain in source/docs; weights and reference workflows may not.

`LatentSlateEngineData` and other app-managed state are never source material: user recipes/revisions, authoring roots, downloaded/materialized artifacts, jobs, caches and credentials stay outside the repository. Durable tests use synthetic, non-identifying fixtures.

## Earned architecture

- Keep code operation/family-local until working consumers prove identical semantics. Extract small model-neutral utilities after contrasting families prove the seam; reusable frameworks should normally have at least three proven consumers.
- Prefer modest duplication to speculative managers, recipe/resource frameworks, plugins or cross-family runtime abstractions. Reconsider a seam that a new family fights instead of adding adapters to preserve it.
- Choose compatibility/stress cases by the distinct assumptions they can falsify, not a mechanical Cartesian product.

## Local stack and process control

The loopback Local Process Manager is the canonical control path for the managed UI/Engine stack; default `http://127.0.0.1:47634`. It is development tooling, not an Engine dependency.

Before control, discover `/health`, `/processes` and relevant `/groups` or `/topology`. Use live stable IDs, never baked-in IDs/PIDs/names/group membership. Prefer individual-process control and poll the resulting state. Use the manager's configured UI build entry for release builds.

`POST /stack/reload` stops all managed processes before rereading configuration; never use it as routine refresh. If the manager is unavailable, report it instead of assuming stale topology or falling back to broad unmanaged process control.

## Comfy-derived inference

For implementation, parity, performance or lifecycle work derived from Comfy, **read `docs/COMFY_REFERENCE.md` first**. It owns reference versions/discovery, fixtures, reconciliation, certification, telemetry, lifecycle checks and AIMDO/Kitchen responsibilities.

- Engine is not a graph engine, plugin host or Comfy reimplementation. Use Comfy as an executable behavioral oracle; do not port its graph executor, global model manager, node/plugin runtime, UI policy or broad `comfy.*` runtime.
- Baseline the **actual model selection and effective settings executed by the curated workflow**, frozen to a recorded revision. Inspect switches/executed state; notes, links and supplemental BF16/core examples do not override it without explicit user direction.
- Establish the exact API workflow, runtime, assets and effective inputs; prove execution was not satisfied by graph caching. Use a fresh equivalent reference baseline; historical timing/memory numbers are not standing gates.
- Localize before porting: map executed stages to Engine equivalents and compare semantic boundaries, cold/warm time, RAM/VRAM, entering state and allocation lifetimes. Start coarse, isolate the first unexplained divergence and choose the cheapest discriminating experiment.
- Before editing, identify the exercised reference implementation, concrete disagreement and evidence connecting it to the measured gap. Adapt the smallest coherent implementation with its ownership/dependencies; do not substitute speculative Engine-only optimizations. Verify the affected boundary and a representative pressure case before full certification.
- Treat source explanations as provisional until supported by the exercised branch. Remove failed experiments and temporary probes; authoritative correctness/performance measurements require instrumentation removed or inactive.
- Share family behavior only when measured boundaries prove identical semantics; similar names, tensors or graphs are insufficient. AIMDO/Kitchen own their existing primitives and global memory/quantization policies; application code expresses execution order and safe dependencies.
- Do not impose determinism or bit identity on a nondeterministic reference. Performance/resource counters are diagnostic unless they establish a real safety invariant.

Measure early on real hardware once safe. Preserve exact artifact/model provenance externally and keep compatibility claims specimen-specific without broader evidence. Do not turn temporary diagnostics, harnesses or one-off fixes into permanent frameworks without repeated proven need.
