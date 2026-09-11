# LatentSlate Engine

LatentSlate Engine is a local inference service for LatentSlate, with a versioned
HTTP catalog, media uploads, asynchronous generation, and downloadable artifacts.

The runtime contains three independently proven model families: LTX 2.3 under
`src/latentslate_engine/ltx23/`, FLUX.2 Klein 9B under
`src/latentslate_engine/klein9b/`, and Wan 2.2 14B turbo under
`src/latentslate_engine/wan2214b/`. Their first evidence-earned shared request
invariants are described in `docs/ENGINE_ARCHITECTURE.md`; inference, lifecycle,
cache, and artifact ownership otherwise remain family-local. The serving/API layer
now exposes the three stable LTX 2.3 tools, the proven Klein 9B text-to-image
and two-image tools, and the three accepted Wan video operations to LatentSlate.

For Hugging Face artifact pinning/materialization, install the lightweight
official client in the Engine environment: `python -m pip install huggingface_hub==1.27.0`.
Public files need no token; private/gated repositories use the host's `HF_TOKEN`
or normal Hub login. In Recipe Studio, choose **Hugging Face** on a file slot,
pin the source, then explicitly **Materialize** the recipe. Import never starts
downloads automatically. See the authoring contract below for cache and task
semantics. Inference dependencies are unchanged.

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
