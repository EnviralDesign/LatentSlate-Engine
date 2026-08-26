# ADR: capability-oriented runtime framework

Status: accepted and implemented

This ADR governs ordinary capability reuse. For an active Comfy parity gap,
[Comfy parity lockdown](../COMFY_PARITY_LOCKDOWN.md) takes precedence: a generic
residency abstraction may be bypassed or retired when direct AIMDO/Kitchen use is
smaller and more faithful. Parity-proven duplication, not predicted reuse, is the
entry criterion for later extraction.

## Context

Engine has several accepted heavyweight runtimes. Their model mathematics and
artifact contracts differ, but their process isolation, file transport,
authentication, quantized-layout restoration, residency coordination, and proof
accounting repeat. Later runtimes contain stronger versions of mechanisms that
earlier runtimes implemented independently.

The refactor must preserve public tools, recipes, outputs, direct Kitchen
dispatch, zero-fallback proofs, privacy, and lifecycle behavior. It must also
remain an Engine-owned implementation: ComfyUI is a pinned source oracle, never
an executable dependency or backend.

## Decision

Engine will organize reusable runtime behavior by capability. Dependency flow is:

```text
tools and recipes
  -> model-family adapter
  -> model architecture / checkpoint map / conditioning / sampler
  -> model-neutral worker, stored-quant, residency, and proof capabilities
  -> PyTorch, SafeTensors, Comfy Kitchen, and operating-system primitives
```

The current `manager.py`, `windows_process.py`, `stored_quant.py`, artifact
probing, cache helpers, and residency policy are starting points. We will not
create parallel replacements merely to satisfy a proposed folder layout.

### Worker boundary

A shared worker layer owns canonical envelopes, keyed authentication, bounded
and atomic files, process-tree ownership, liveness policy, cancellation,
poisoning, cleanup, and safe failure publication. Worker entry points are a
closed source-controlled registry. Requests cannot select a Python module or
callable.

A model handler owns typed request parsing, artifact revalidation, load/run/
unload, model progress stages, model result fields, and model provenance. Both
disposable and persistent sessions are explicit policies; neither is simulated
with model-name switches.

File IPC remains the default because it is already accepted and auditable.
Envelope schema versioning is separate from model request/result schemas.

The disposable lifecycle remains deliberately narrow:
`DisposableWorkerSupervisor` owns fixed-command spawn, Job Object lifecycle,
bounded result/progress transport, cancellation, and cleanup, while
`run_disposable_child` invokes a statically constructed handler. Wan 5, the LTX
BF16 Reference path, and the reachable Wan prompt encoder consume this policy.

Persistent sessions are a separate capability, not flags on the disposable
class. `PersistentWorkerSupervisor` and `run_persistent_child` own fixed-command
spawn, the serial command loop, bounded progress and heartbeat transport,
independent hard/stage/heartbeat clocks, boundary drains, cooperative cancel
grace, poisoning support, Job Object termination, and cleanup. Z-Image, LTX
Kitchen, and Wan 14 consume this policy. Their statically wired handlers retain
family HMAC composition, concrete session identity, result/proof,
safe-failure, and model-runtime decisions.

### Stored-quant boundary

SafeTensors/header validation, known Kitchen layout restoration, byte-preserving
transport, and generic observed-dispatch proof are shared. Source-to-target key
mapping and exact checkpoint closure remain model-specific.

Execution differences are represented by small strategies, not flags on one
universal linear: direct stored kernels, full-precision dequant-then-linear,
grouped stored residency, per-operation streaming, and dense resident execution
remain distinct operations.

### Residency boundary

The shared layer currently owns only process-wide exclusive stored-session
coordination and concrete-device normalization. Model adapters still own their
component groups, movement transitions, rollback, cleanup, and telemetry;
those obligations are not yet uniform enough to generalize honestly.
Runtime-wrapper caching remains the responsibility of the existing
`RuntimeManager`.

A broader residency lifecycle is a possible future capability, not present
architecture. It should be added only after at least two characterized consumers
demonstrate the same state transitions and rollback obligations.

### Proof boundary

Shared proof objects validate generic observed facts—expected/executed counts,
native calls, dequantizations, linear calls, rejection/fallback counts, dtype,
and residency. Models may add exact fields. Configuration never counts as proof
of execution.

## Deliberately model-specific behavior

Architecture, source key vocabulary, fused/sliced targets, conditioning order,
tokenization, scheduler/sigmas, sampler, guidance, dimensions, A/V timing,
component staging order, output interpretation, and exact proof extensions stay
with their model families.

This is not a graph engine, plugin system, dynamic loader, universal model base
class, or ComfyUI host.

## Migration and compatibility

Each structural change first gains model-free or model-specific characterization
tests. One capability and one model consumer migrate at a time. A compatibility
facade is allowed only when the same change records its removal step; the final
architecture contains no permanent private re-export shims.

Initial migration order is worker primitives and a fake handler, Wan 5 worker,
Z-Image worker, neutral FP8/NVFP4 restoration, Z/LTX quant consumers, shared
residency state, then remaining production workers. Z-Image Qwen is split only
after worker, quant, and residency contracts exist, so accepted mathematics are
not rewritten during infrastructure extraction.

The worker primitives, model-free disposable and persistent handlers, all active
heavyweight worker migrations, neutral FP8/NVFP4 restoration/execution, and
shared exclusive-residency lease are now implemented. Persistent policies
preserve each family's heartbeat, deadline, cooperative-cancel, poison,
authentication, and warm-reuse contract without generalizing model-owned
schemas into the framework.

The former Z-Image mixed-Qwen monolith is also split along the established
boundaries: a Torch-only architecture shell owns attention/RoPE/block-34
capture, checkpoint planning owns the exact config/header/398-key closure, and
runtime composition owns shared restore, per-operation residency, direct F32
dequantization/linear execution, preflight, and proof. Dependencies flow from
runtime to checkpoint and architecture only; there is no facade or reverse edge.

## Guardrails

AST-based tests enforce that shared framework code imports no model family,
contains no model-name dispatch, exposes no arbitrary module/callable loading,
and creates no ComfyUI execution surface. A fixed exemption set records current
private cross-family imports and may only shrink.

## Consequences

The program favors deletion and narrow composable capabilities over a quick
directory reshuffle. Some model files remain large until a proven shared seam
exists. Hardware acceptance remains mandatory after each model migration; CI
alone cannot prove GPU residency, dispatch, or cleanup behavior.
