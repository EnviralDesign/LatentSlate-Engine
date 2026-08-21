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
- Treat WanGP, ComfyUI, and InvokeAI as architectural references. Do not copy
  implementation code without checking and preserving compatible licensing.

## Capability-first implementation

Read [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md), the
[runtime framework ADR](docs/architecture/ADR_RUNTIME_FRAMEWORK.md), and the
[runtime inventory](docs/architecture/RUNTIME_FRAMEWORK_INVENTORY.md) before
adding or structurally changing a model runtime.

- Start from the required worker, IPC, stored-quant, proof, cache, residency, or
  memory-observation capability; do not copy another family wholesale.
- Shared framework code must remain model-neutral and statically wired.
- Model adapters own architecture, artifact maps, conditioning, schedules,
  sampler math, model loading, progress vocabulary, and result proof.
- Use the closed recipe-handler registry for a new typed recipe. Do not add
  central type-condition branches or dynamic plugin loading.
- Write the reuse matrix required by `docs/ADDING_A_MODEL.md` before implementation.
- Do not import another family's private implementation or retain historical
  implementations in active source as examples.
- Do not recreate worker, quant transport, or residency infrastructure inside a
  family adapter.
- Run model-free worker and architecture gates before affected-family hardware.
- After two failed hypotheses, acquire a new source trace, fixture, log boundary,
  or discriminating test before changing code again.

## Comfy architecture boundary

Read and follow [docs/COMFY_ENGINE_POLICY.md](docs/COMFY_ENGINE_POLICY.md) before
implementing or reviewing any model runtime, optimized recipe, or roadmap.
ComfyUI workflows and pinned node source are architecture/behavior references;
ComfyUI is never an Engine execution backend. Do not embed, import, launch,
proxy, or require ComfyUI from Engine. Comfy Kitchen is intentionally different:
use its supported tensor/layout/kernel/dispatch primitives directly inside
Engine-owned disposable workers. "Comfy-first" means study and reproduce the
effective operation cleanly in Engine, not run a graph server. Select capabilities
from the framework; no model family is a wholesale golden implementation.

### Comfy source oracle workflow

Before implementing a new model or recipe, and whenever runtime behavior hits a
new bug or unexplained mismatch, consult a dedicated read-only Comfy source
oracle and wait for its findings before changing the Engine implementation.

Keep that oracle strictly outside LatentSlate and Engine code:

- Give it only the desired behavior, recipe facts, or privacy-safe failure
  symptom and focused questions.
- Restrict its inspection to the pinned ComfyUI source tree and the installed
  Comfy Kitchen package source. It must not inspect LatentSlate, Engine source,
  worktrees, model payloads, user data, logs, or runtime state.
- Ask it to return the relevant source locations, execution chain, numerical
  conventions, lifecycle patterns, and implementation traps.
- Use a separate Engine-aware implementer or reviewer to compare those findings
  with local code and decide the clean-room Engine changes.
- Keep the oracle read-only and available for follow-up questions; do not let it
  drift into local implementation or acceptance work.

This separation is intentional: the oracle stays an independent map of proven
Comfy behavior, while the Engine team owns adaptation, direct Kitchen dispatch,
local debugging, and LatentSlate acceptance.

For difficult Comfy-derived runtime work, prefer a ChatGPT Pro source-oracle and
implementation-planning pass before further local iteration. Give Pro a current,
pushed repository state and ask it to inspect the pinned ComfyUI source, installed
Comfy Kitchen source, and the Engine implementation together. Require it to trace
the working Comfy call path, compare the Engine edge by edge, and audit the model
runtime against mature Engine patterns—especially Klein—for broader architectural
drift or code smell. Pro should return a diagnosis and sufficiently detailed local
implementation plan; a patch or zip is optional and must not be required. Wait for
the Pro answer when it is an execution gate. Use a local Sol expert to implement
the resulting plan and a separate Sol reviewer to verify it.

For Comfy-derived model reconstruction, sampler mathematics, quantized model
execution, and GPU/process lifecycle implementation, use a Sol expert as the
default implementer and a separate Sol reviewer. These paths are sufficiently
coupled and consequential that Terra should be limited to mechanical source
mapping, bounded scaffolding, or straightforward follow-up edits. Give the Sol
implementer the oracle's ordered source call map and require an edge-by-edge
conformance comparison; do not preserve an existing abstraction merely to avoid
rewriting it.

## Checks

Run before yielding:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

Run `ruff check .` when the development extra is installed. H3 tests must remain
opt-in and may not be required for routine CI because they require large model
downloads and a supported GPU.
