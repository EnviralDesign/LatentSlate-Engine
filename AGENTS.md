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

## Checks

Run before yielding:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

Run `ruff check .` when the development extra is installed. H3 tests must remain
opt-in and may not be required for routine CI because they require large model
downloads and a supported GPU.
