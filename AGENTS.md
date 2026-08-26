# AI Developer Guidelines

LatentSlate Engine is a curated Python inference service for LatentSlate. It is
not a general graph engine, plugin host, or reimplementation of another inference
application.

Prefer the smallest runtime that faithfully satisfies the product contract.

## Product boundaries

- Keep tool discovery explicit in `latentslate_engine.tools.default_registry()`.
- Tool IDs and input keys are persistent protocol identities. Do not change them
  casually; labels and descriptions may evolve.
- Keep model-specific loading and inference under `runtime/` so API/schema work
  does not depend on one model implementation.
- Media transport is always HTTP upload/download. Do not add shared-filesystem
  shortcuts to the public protocol.
- Do not introduce dynamic plugin loading, a general graph executor, or global
  model-management policy.
- Treat WanGP, ComfyUI, InvokeAI, and other mature runtimes as behavioral and
  architectural references.
- Narrow source adaptation is allowed when licensing is compatible, provenance is
  recorded, and adaptation produces a smaller or less divergent runtime.
- Runtime independence is mandatory. Source independence is not.

## Implementation discipline

- Prefer direct use of an existing primitive or supported API over introducing a
  wrapper, manager, coordinator, policy layer, or parallel ownership model.
- New abstractions must pay rent immediately: they must remove more complexity
  than they add or represent a requirement that cannot be expressed cleanly with
  existing primitives.
- Do not generalize from a single use case. Fix the narrow case first. Promote
  code into shared framework only after at least two proven consumers demonstrate
  the same seam.
- Treat persistent state as expensive. Before adding a stateful class, manager,
  coordinator, registry, lifecycle object, lease, descriptor, proof object, or
  cache, justify why simpler local state or an existing owner is insufficient.
- Prefer functions and ordinary object fields over new stateful abstractions when
  they express the required lifetime clearly.
- Fix exceptional edge cases at the narrowest responsible layer. Do not turn a
  rare failure into a general recovery framework unless recovery is itself a
  product requirement.
- When process isolation provides a safe recovery boundary, prefer failing closed
  and restarting the isolated process over maintaining complex recovery machinery
  for ambiguous native or external state.
- Make one meaningful hypothesis-driven change at a time when debugging behavior
  or performance. Measure its predicted effect before adding another mechanism.
- Tests protect product behavior and important invariants, not historical
  implementation structure. Rewrite or delete implementation-detail tests when
  the architecture they encode is superseded.
- Keep observability proportional to the question being answered. Temporary
  diagnostics, counters, proof state, and failure taxonomy must not become durable
  architecture merely because they were useful during debugging.
- When matching a known-good reference implementation, reproduce its observable
  behavior, state transitions, ownership, and lifetimes before translating its
  concepts into local abstractions.
- If a fix appears to require a new persistent architectural noun, stop and
  reconsider whether the existing primitive or process boundary already solves
  the problem.
- A successful static implementation pass should be followed by the smallest meaningful hardware/reference comparison as soon as safety permits. Do not perform speculative cleanup, generalization, or architecture improvement between the static gate and that measurement.

## Model and capability work

Read [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md), the
[runtime framework ADR](docs/architecture/ADR_RUNTIME_FRAMEWORK.md), and the
[runtime inventory](docs/architecture/RUNTIME_FRAMEWORK_INVENTORY.md) before
adding or structurally changing a model runtime.

These documents describe existing capabilities; they do not make existing
abstractions mandatory.

- For an ordinary model addition, begin with the model's concrete behavioral and
  resource requirements. Reuse an existing Engine capability when it already
  satisfies them.
- Do not create a generic capability merely because a new model needs something
  that sounds reusable.
- Model adapters own model architecture, artifact maps, conditioning, schedules,
  sampler mathematics, model-specific loading, progress vocabulary, and
  model-specific acceptance behavior.
- Use the closed recipe-handler registry for new typed recipes. Do not add central
  type-condition branches or dynamic plugin loading.
- Write the reuse matrix required by `docs/ADDING_A_MODEL.md` before
  implementation.
- Do not import another model family's private implementation or retain obsolete
  implementations in active source as examples.
- Do not recreate worker or IPC infrastructure inside a family adapter.
- Run model-free worker and architecture gates before affected-family hardware
  gates.
- After two failed hypotheses, acquire new evidence: a source trace, fixture, log
  boundary, profiler result, or discriminating test. Do not continue stacking
  speculative fixes.

Shared framework is an output of demonstrated reuse, not a prerequisite for
implementing a model.

## Comfy architecture boundary

Read and follow [docs/COMFY_ENGINE_POLICY.md](docs/COMFY_ENGINE_POLICY.md) before
implementing or reviewing a Comfy-derived model runtime, optimized recipe, or
roadmap.

ComfyUI workflows and pinned source are behavioral and architectural references.
ComfyUI is never an Engine execution backend.

Do not embed, import, launch, proxy, or require ComfyUI from Engine.

Comfy Kitchen and comfy-aimdo are permitted low-level dependencies. Use their
supported tensor, quantization, transfer, residency, and memory primitives
directly where appropriate rather than recreating their responsibilities inside
Engine.

Narrow, attributed, license-compatible source ports are preferred over a new
Engine abstraction when they preserve the relevant upstream state machine with
less code and semantic drift.

## Comfy parity lockdown

Explicit Comfy parity work is a different engineering mode from ordinary
capability development.

Read [docs/COMFY_PARITY_LOCKDOWN.md](docs/COMFY_PARITY_LOCKDOWN.md).

During parity lockdown:

- Pinned Comfy behavior is authoritative for the behavior under review.
- Existing Engine implementation, abstractions, tests, ADR assumptions, and
  framework boundaries are not authoritative if they conflict with measured
  parity.
- Product/API boundaries, artifact integrity, process isolation, cancellation,
  and explicit model-identity semantics remain authoritative.
- Do not design a generalized Engine equivalent of a Comfy concept merely because
  the concept has a name.
- Preserve upstream state transitions and lifetimes at the lowest practical
  abstraction level.
- A parity path may bypass, replace, or delete a generic Engine residency,
  quantization, transfer, cache, or lifecycle abstraction when direct use of
  AIMDO/Kitchen is smaller and more faithful.
- No new model-neutral abstraction should be introduced during parity work unless
  the current parity task cannot be expressed without it.
- Do not generalize the parity substrate to another operation or model family
  until the current acceptance gates pass.

### Parity research workflow

Before editing a parity path, produce two evidence ledgers.

**Call ledger**

Trace the working reference from workflow/node entry through model forward to the
low-level primitive. Include exact source locations and important control edges.

**State-survival ledger**

For every state that affects correctness or performance, record:

- its owner;
- where it is created;
- where it is mutated;
- whether it survives an operation;
- whether it survives a stage;
- whether it survives a request;
- what invalidates it;
- what destroys it;
- which stream or native lifetime constraints apply.

A correct call trace without a correct state-survival trace is insufficient.

### Scout roles

Use Luna scouts aggressively for bounded, read-only work when doing so materially
reduces main-session context or uncertainty.

- A **trace scout** reports exact calls, mutations, and execution order. It does
  not propose Engine architecture.
- A **primitive scout** classifies each required behavior as:
  - already owned by AIMDO;
  - already owned by Kitchen;
  - a thin caller responsibility;
  - or genuinely missing.
- A **delta scout** compares the reference with Engine and lists Engine-only state
  objects, ownership layers, and control edges. It classifies each as a required
  product invariant or a deletion candidate.

Research scouts are archaeologists, not architects.

The main agent owns synthesis and assigns one bounded behavioral delta to an
implementer.

Use a separate reviewer for consequential CUDA, native-lifetime, quantization,
source-port, or destructive-lifecycle changes.

### Parity implementation loop

One hypothesis, one patch, and one fixed benchmark is the normal unit of parity
work.

Before implementation, state the expected observable consequence, such as:

- a specific counter changing;
- a source reread disappearing;
- signature hits appearing;
- H2D traffic falling;
- a stage timing moving;
- a memory peak changing.

If the predicted movement does not occur, stop and return to evidence gathering.
Do not compensate by adding another cache, scheduler, recovery path, or
abstraction.

Prefer deletion over coexistence. Once a replacement path is proven, remove the
superseded implementation rather than retaining both as active architecture.

## Review discipline

Reviewers are responsible for architecture-regrowth detection in addition to
correctness, safety, and regression review.

For consequential runtime changes, the reviewer should explicitly report:

- new persistent state introduced;
- new stateful classes, protocols, managers, coordinators, registries, leases,
  descriptors, policies, or lifecycle objects;
- the exact product or upstream behavior requiring each;
- whether direct use of an existing primitive would be simpler;
- whether process restart is a sufficient failure boundary;
- production lines added and deleted;
- superseded machinery that can now be removed.

A reviewer should reject a change that is locally correct but introduces
unjustified persistent machinery.

Do not enlarge a patch merely to satisfy implementation-detail tests belonging to
an architecture being replaced.

## Failure and native-lifetime policy

Normal resource pressure, cache eviction, and documented fallback paths are part
of ordinary execution and must be handled correctly.

Ambiguous or corrupted native state is different.

- Preserve live references conservatively until ownership is known to have ended.
- Do not claim a resource was released when the underlying native release failed.
- Do not build general recovery machinery solely to keep a GPU worker alive after
  native ownership or stream quiescence becomes unknowable.
- When safe continuation cannot be established cheaply and locally, fail the
  operation and terminate the isolated GPU worker.
- Reconstructing a clean worker is an intentional recovery mechanism, not a
  failure of architecture.

Optimize the common success path. Correctly handle expected pressure. Fail closed
on exceptional native corruption.

## Checks

Run before yielding:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
````

Run `ruff check .` when the development extra is installed.

Hardware-specific and H3 tests must remain opt-in when they require large model
downloads or supported GPU hardware and must not become mandatory for routine CI.

```