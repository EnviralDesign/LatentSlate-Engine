# AI Developer Guidelines

LatentSlate Engine is a curated Python inference service for LatentSlate. It is
not a general graph engine or plugin host.

## Boundaries

- Keep tool discovery explicit in `latentslate_engine.tools.default_registry()`.
- Tool IDs and input keys are persistent protocol identities. Do not change them
  casually; labels and descriptions may evolve.
- Keep model-specific loading and inference under `runtime/` so API/schema work
  does not depend on one implementation.
- Media transport is always HTTP upload/download. Do not add shared-filesystem
  shortcuts to the public protocol.
- Treat WanGP, ComfyUI, and InvokeAI as behavioral and architectural references.
  Narrow source adaptation is allowed when licensing is compatible, provenance is
  recorded, and it produces a smaller, less divergent runtime.

## Parity work and capability work are different modes

Read [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md), the
[runtime framework ADR](docs/architecture/ADR_RUNTIME_FRAMEWORK.md), and the
[runtime inventory](docs/architecture/RUNTIME_FRAMEWORK_INVENTORY.md) before
adding or structurally changing a model runtime.

- For ordinary model additions, start from the required worker, IPC, stored-quant,
  proof, cache, residency, or memory-observation capability.
- For explicit Comfy parity work, follow
  [COMFY_PARITY_LOCKDOWN.md](docs/COMFY_PARITY_LOCKDOWN.md). Pinned Comfy behavior
  is authoritative for the behavior under review; existing Engine abstractions and
  tests are not.
- Shared framework code must remain model-neutral and statically wired.
- Model adapters own architecture, artifact maps, conditioning, schedules,
  sampler math, model loading, progress vocabulary, and result proof.
- Use the closed recipe-handler registry for a new typed recipe. Do not add
  central type-condition branches or dynamic plugin loading.
- Write the reuse matrix required by `docs/ADDING_A_MODEL.md` before implementation.
- Do not import another family's private implementation or retain historical
  implementations in active source as examples.
- Do not recreate worker or IPC infrastructure inside a family adapter. A parity
  path may bypass or replace a generic residency/quant abstraction when direct
  AIMDO/Kitchen use more faithfully preserves the pinned source state machine.
- Run model-free worker and architecture gates before affected-family hardware.
- After two failed hypotheses, acquire a new source trace, fixture, log boundary,
  or discriminating test before changing code again.

## Comfy architecture boundary

Read and follow [docs/COMFY_ENGINE_POLICY.md](docs/COMFY_ENGINE_POLICY.md) before
implementing or reviewing any model runtime, optimized recipe, or roadmap.
ComfyUI workflows and pinned source are architecture/behavior references;
ComfyUI is never an Engine execution backend. Do not embed, import, launch,
proxy, or require ComfyUI from Engine. Comfy Kitchen and comfy-aimdo are permitted
low-level dependencies. Runtime independence is mandatory; source independence is
not. Narrow, attributed, license-compatible source ports are preferred over a new
Engine abstraction when they preserve the relevant Comfy state transitions with
less code and drift.

### Comfy parity research workflow

Before editing a parity path, produce two evidence ledgers:

1. a call ledger from workflow node through model forward to low-level primitive;
2. a state-survival ledger naming each state owner and whether it survives an
   operation, stage, request, and recipe-identity change.

Use Luna scouts aggressively and independently for bounded read-only work:

- a trace scout reports exact calls and state mutations, without proposing an
  Engine architecture;
- a primitive scout classifies each required behavior as already owned by AIMDO,
  already owned by Kitchen, a thin caller responsibility, or genuinely missing;
- a delta scout lists Engine-only state objects/control edges and classifies them
  as product invariants or deletion candidates.

The main agent synthesizes those ledgers and assigns one behavioral delta to an
implementer. Use a separate reviewer for consequential CUDA, native-lifecycle, or
source-port changes. Do not ask research scouts to design a framework.

One hypothesis, one patch, and one fixed benchmark is the normal unit of parity
work. If the predicted counter or timing movement does not occur, stop and return
to source tracing. Do not compensate with another cache or abstraction.

## Checks

Run before yielding:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

Run `ruff check .` when the development extra is installed. H3 tests must remain
opt-in and may not be required for routine CI because they require large model
downloads and a supported GPU.
