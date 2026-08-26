# LatentSlate Engine

LatentSlate Engine is being rebuilt from a deliberately clean greenfield baseline.
The pre-reset implementation was removed so the new runtime can earn its architecture
from working model families rather than inherit historical framework assumptions.

The first implementation in progress is a standalone LTX 2.3 inference family under
`src/latentslate_engine/ltx23/`, developed against pinned ComfyUI behavior with direct
AIMDO/Kitchen use. The serving/API layer remains intentionally deferred until model-family
evidence has earned the shared seams described in `AGENTS.md`.

Start with:

- [`AGENTS.md`](AGENTS.md)
- [`docs/GREENFIELD_RESET.md`](docs/GREENFIELD_RESET.md)
- [`docs/ENGINE_CONTRACT.md`](docs/ENGINE_CONTRACT.md)
- [`docs/COMFY_REFERENCE.md`](docs/COMFY_REFERENCE.md)
- [`docs/LTX23_TARGET.md`](docs/LTX23_TARGET.md)

The pre-reset implementation remains recoverable at the annotated Git tag
`ltx23-pre-greenfield-reset-2026-08-26`
(`86419a7b943a2dcd9a172c817aafb3f05728331d`). It is a historical checkpoint,
not the architecture for this rebuild.
