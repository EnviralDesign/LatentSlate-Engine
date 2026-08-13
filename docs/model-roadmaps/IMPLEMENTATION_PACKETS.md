# Cross-family implementation packets

All packets are subordinate to the hard
[Comfy authority and Engine execution policy](../COMFY_ENGINE_POLICY.md). A packet
may use pinned Comfy workflows and node source to specify behavior and may use
Comfy Kitchen directly, but it must never propose ComfyUI as an Engine runtime,
server, plugin host, or recipe dependency.

Last reviewed: **2026-08-13**

This index turns the family roadmaps into bounded handoffs. It is intentionally
current-state driven: accepted Engine work is not redispatched, and an upstream
workflow is not called exact until every active artifact and fixed setting is pinned.

Complexity describes integration risk, not model quality. **Terra-high** is the
default for an explicit closure and operation. Escalate to **Sol** for unresolved
tensor mappings, multi-stage lifecycle design, dependency isolation, or contradictory
upstream authorities.

## Priority packets

| Priority | Bounded packet | Reuse and boundary | Suggested agent | Stop conditions |
| ---: | --- | --- | --- | --- |
| 1 | [Z-Image Turbo](./Z_IMAGE_TURBO.md): exact official INT8 ConvRot T2I closure, typed recipe, stored loader, and public-API smoke | Reuse stored ConvRot, mixed-Qwen, residency, and dispatch proof; only 1024-square Turbo T2I at 8 steps, CFG 1, AuraFlow shift 3, `res_multistep`/`simple` | Terra-high | Unknown header mapping, fallback/dequantized execution, or an unpinned component |
| 2 | [LTX 2.3](./LTX_2_3.md): replace full-folder substitution with exact components, then T2V and endpoint-conditioned acceptance | Existing complete-repository runtime, A/V mux, cancellation, and cache seams; preserve 24 fps and `8n+1` frame rules | Terra-high | Audio/video drift, repository substitution, dependency conflict, or poisoned recovery |
| 3 | [MiniMax H3](./MINIMAX_H3.md): re-pin the current FL2VA BF16 closure and qualify T2VA plus endpoint modes | Existing older-pinned direct runtime; keep Ref2VA and hosted Context-IR/2K separate | Terra-high, escalate if architecture drift is material | Incomplete release identity, transformer-ref ambiguity, license gate, or incompatible source layout |
| 4 | [LTX 2.5](./LTX_2_5.md): pin the official T2V graph and its six-resource closure before any runtime | New family path; do not substitute the five-resource FLF graph or publisher BF16 path for official Comfy T2V | Terra-high discovery, Sol for runtime if needed | Missing immutable gated identities, Gemma environment conflict, or Windows decoder failure |
| 5 | [Qwen Image Edit 2511](./QWEN_IMAGE_EDIT_2511.md): ordered input contract and saved-default INT8 edit reference | Generic ordered media schema, staged encoder/offload, and exact closure tooling; Lightning is a separate four-step recipe | Terra-high | Input multiplicity ambiguity, incomplete mixed-precision map, or unqualified Lightning fusion |
| 6 | [Krea 2](./KREA_2.md): license decision and exact saved-default Turbo manifest | Saved parity is the three-file base graph; Darkbrush at 0.8 is configured but disabled and becomes a separate four-file variant when explicitly enabled | Terra-high manifest, Sol for loader | License/product rejection, switch-state ambiguity, or ConvRot/native fallback |
| 7 | [Ideogram 4](./IDEOGRAM_4.md): product/license gate and exact structured-request closure | Preserve JSON prompt semantics and conditional/unconditional topology; no invented dense reference | Terra-high contract, Sol for loader | License failure, missing model branch, prompt-contract drift, or unverifiable VAE |
| 8 | [Stable Diffusion XL](./STABLE_DIFFUSION_XL.md): base-only FP16 value spike | Reuse generic Diffusers lifecycle; Base+Refiner is a separate operation | Terra-high | No creator-visible value over newer supported families |

## Accepted families: evidence expansion only

These paths are implemented and must not be handed off as new loader work:

- [FLUX.2 Klein 4B](./FLUX2_KLEIN_4B.md) and
  [9B](./FLUX2_KLEIN_9B.md): broaden cancellation, reference-count, source-change,
  peer-switch, creator-corpus, and matching BF16 comparison evidence. Preserve the
  accepted NVFP4/FP8 operation contracts and native-dispatch proof.
- [Wan 2.2 TI2V 5B](./WAN22_TI2V_5B.md): retain its dense reference and split artifacts
  as source evidence. Rebuild only the existing optimized T2V/I2V paths as Engine-native
  direct-Kitchen operations before gathering creator-quality evidence.
- [Wan 2.2 14B](./WAN22_14B.md): broaden corpus, changed-source, peer-switch, and FLF
  endpoint-pair evidence. I2V, T2V, and FLF Engine-native baselines are accepted;
  their official LightX operations are accepted Experimental paths. ConvRot remains
  catalog-only until a clean operation and exact-header planner/native dispatch are
  proven.

## Shared handoff requirements

Every implementation packet must:

1. Pin the workflow authority, every active artifact, exact revision, byte count,
   content identity, license, and operation-specific fixed settings.
2. Add catalog/schema/header tests and prove the selected closure contains no hidden
   checkpoint weights or missing active components.
3. Map every tensor, alias, sidecar, scale, and dense exception one-to-one. Fail closed
   on leftovers or runtime conversion.
4. Prove the intended native backend dispatched. A low-bit file that executed through
   eager/dequantized fallback is not accepted evidence.
5. Use the public Engine job API for success, live cancellation, fresh recovery,
   switching, provenance, artifact publication, and teardown evidence.
6. Keep official operations separate. Do not collapse Base/Distilled, T2I/edit,
   T2V/I2V/FLF, single/two-stage, or fixed-LoRA/base graphs.
7. Record output hashes and media properties, device-wide versus process-local memory,
   worker/process observations, cache state, and honest evidence limitations.

## Universal stop conditions

Stop and report evidence rather than improvising when:

- an immutable source, file, revision, size, or hash cannot be resolved;
- the active workflow closure or fixed switch state is ambiguous;
- the license or gated-access posture blocks built-in acquisition;
- source and target tensors cannot be mapped completely;
- native dispatch cannot be proven;
- required dependencies destabilize already accepted families;
- cancellation, failure, or switching leaves poisoned runtime state;
- parity would require silently changing dimensions, frames, fps, schedule,
  preprocessing, conditioning, or model lineage.
