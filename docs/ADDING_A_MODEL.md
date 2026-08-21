# Adding a model or recipe

Start from the capability the model needs, not from the nearest model file to
copy. Engine is a curated set of typed native runtimes; it is not a graph host,
dynamic plugin loader, or universal model abstraction.

## 1. Fix the public contract first

- Choose the stable tool ID, operation, input identities, canvas, and output
  contract.
- Pin the publisher artifacts and the behavioral source revision.
- Declare the exact resource closure and fail closed when any role is missing or
  incompatible.
- Add a built-in recipe only when the operation is independently verified.

## 2. Trace the behavioral authority

For Comfy-derived behavior, follow [the Comfy/Engine policy](./COMFY_ENGINE_POLICY.md).
Use a read-only source oracle to return the complete call path, schedule,
conditioning, sampler, decode, and lifecycle facts. Engine owns the clean native
implementation and calls supported Comfy Kitchen primitives directly where useful;
it never runs ComfyUI.

## 3. Reuse capabilities, keep mathematics local

Before implementation, normalize the effective source call graph and constants,
write independent semantic fixtures, and complete this small reuse matrix:

| Required behavior | Existing Engine capability | Model-specific code | Missing shared capability | Evidence |
|---|---|---|---|---|
| _one row per required behavior_ | _reuse or none_ | _irreducible ownership_ | _only demonstrated gaps_ | _source/fixture/run_ |

- Use `DisposableWorkerSupervisor` and `run_disposable_child` for isolated
  one-job workers.
- Use `PersistentWorkerSupervisor` and `run_persistent_child` only when measured
  warm reuse justifies a serial exact-recipe session.
- Use shared bounded file IPC, stored-quant restoration/execution, proof counters,
  process memory observation, and exclusive residency coordination.
- Keep architecture, checkpoint key maps, conditioning order, schedules, sampler
  math, model loading, progress vocabulary, and accepted-result proof in the model
  adapter.
- Add one static `_RecipeHandler` entry in `variants.py`; do not add central
  `isinstance` branches or a dynamic plugin mechanism.

Shared framework code must contain no model-name switches. A model adapter must
not recreate subprocess launch, Job Object ownership, polling, heartbeat,
cancellation transport, JSONL draining, or generic cleanup.

## 4. Prove boundaries before hardware

Add model-free and adapter characterization for request binding, duplicate-key and
size rejection, result authentication, progress bounds, cancellation, poisoning,
fresh recovery, warm reuse when applicable, exact cleanup, and public lifecycle
status. Extend the recursive architecture tests so the old structure cannot return.

Run the model-free gates before renting or loading hardware:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\ruff.exe check .
```

Preserve the selected accelerator extra in a hardware-capable local environment;
do not run a dev-only sync that removes its runtime packages.

## 5. Promote only from evidence

Exercise affected families through the Engine public API. Record output identity,
runtime cold/warm state, direct-dispatch and zero-fallback proof, cancellation and
fresh recovery, process-tree cleanup, and host/GPU memory. Hardware-proven catalog
claims must be backed by compact versioned evidence generated from real manifests.
Reference recipes that exceed local hardware remain explicit remote comparison
contracts rather than being silently skipped or downgraded.

See the [runtime framework ADR](./architecture/ADR_RUNTIME_FRAMEWORK.md),
[runtime inventory](./architecture/RUNTIME_FRAMEWORK_INVENTORY.md), and
[hardware study policy](./HARDWARE_STUDIES.md) for the current boundaries.
