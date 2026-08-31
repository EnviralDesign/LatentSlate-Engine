# LatentSlate Engine

LatentSlate Engine is being rebuilt from a deliberately clean greenfield baseline.
The pre-reset implementation was removed so the new runtime can earn its architecture
from working model families rather than inherit historical framework assumptions.

The rebuild now contains three independently proven model families: LTX 2.3 under
`src/latentslate_engine/ltx23/`, FLUX.2 Klein 9B under
`src/latentslate_engine/klein9b/`, and Wan 2.2 14B turbo under
`src/latentslate_engine/wan2214b/`. Their first evidence-earned shared request
invariants are described in `docs/ENGINE_ARCHITECTURE.md`; inference, lifecycle,
cache, and artifact ownership otherwise remain family-local. The serving/API layer
and LatentSlate reintegration remain intentionally deferred.

Start with:

- [`AGENTS.md`](AGENTS.md)
- [`docs/GREENFIELD_RESET.md`](docs/GREENFIELD_RESET.md)
- [`docs/ENGINE_ARCHITECTURE.md`](docs/ENGINE_ARCHITECTURE.md)
- [`docs/ENGINE_CONTRACT.md`](docs/ENGINE_CONTRACT.md)
- [`docs/COMFY_REFERENCE.md`](docs/COMFY_REFERENCE.md)
- [`docs/LTX23_TARGET.md`](docs/LTX23_TARGET.md)
- [`docs/KLEIN9B_TARGET.md`](docs/KLEIN9B_TARGET.md)
- [`docs/WAN2214B_TARGET.md`](docs/WAN2214B_TARGET.md)
- [`docs/CANONICAL_PARITY_CERTIFICATION.md`](docs/CANONICAL_PARITY_CERTIFICATION.md)

The pre-reset implementation remains recoverable at the annotated Git tag
`ltx23-pre-greenfield-reset-2026-08-26`
(`86419a7b943a2dcd9a172c817aafb3f05728331d`). It is a historical checkpoint,
not the architecture for this rebuild.
