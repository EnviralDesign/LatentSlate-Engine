# LTX 2.3 roadmap

Status: **parity reset in progress**

Last authority audit: **2026-08-25**

Follow [Comfy parity lockdown](../COMFY_PARITY_LOCKDOWN.md) and the
[Comfy/Engine execution policy](../COMFY_ENGINE_POLICY.md). Historical implementation
logs remain in git history and `evidence/`; they are not architectural authority.

## Authority pins

| Surface | Authority |
| --- | --- |
| model architecture and weights | Lightricks sources and pinned artifacts |
| workflow topology/defaults | official LTX 2.3 T2V, I2V, and FLF workflow bytes |
| execution behavior | ComfyUI v0.34.0 `12d5279438bfefc058a269eae805ceab6047777f` |
| low-level residency/transport | `comfy-aimdo` 0.4.15 |
| stored quantization/kernels | `comfy-kitchen` 0.2.31 |
| Engine reset checkpoint | `dd68307` |

Engine never imports or executes `comfy.*`. Narrow, attributed source adaptation is
allowed where it preserves pinned behavior with less code and drift.

## Current truth

LTX 2.3 is the proving ground for a smaller Engine inference substrate. The current
`LeafResidencyScheduler` → `AimdoDynamicResidency` → `AimdoHostSourcePool` route is
frozen for LTX and will not receive more features or implementation-detail tests.
The benchmark falsified its central warm-residency behavior, so the LTX hot path will
bypass it and then delete its LTX-specific machinery.

T2V is the only active parity-hardening target. I2V and FLF have historical
functional/output evidence, but they have not passed the new Comfy timing, memory,
state-lifetime, or simplified-substrate gates. They resume only after T2V passes.

The reference canvas is no longer 1280×704. Routine iteration is **512×512** and the
heavier/final local gate is **768×768**, both using 121 frames, 25 fps, and 48 kHz
stereo for the current five-second workflow contract.

## Fixed baseline and rejected checkpoint

Exact shared prompt:

> A tiny silver wind-up bird flutters across a sunlit workshop table, its metal
> wings clicking softly as the camera follows at eye level.

| Runtime | Cold | Warm | RAM peak cold/warm | GPU peak cold/warm |
| --- | ---: | ---: | ---: | ---: |
| pinned Comfy, 512×512 | 141.709 s | 44.157 s | 60.484 / 61.393 GiB | 15097 / 15074 MiB |
| Engine `dd68307`, 512×512 | 438.088 s | 60.542 s | 63.001 / 62.432 GiB | 15739 / 15423 MiB |

Both Engine outputs passed the media contract, and the latest stage-boundary OOM is
closed. Timing parity is rejected. The decisive counters are:

- cold prompt enhancement: 288.837 s, 1.573 TB H2D, 338.856 GB source reads;
- warm AV: 37,048 faults, 37,048 misses, zero signature hits, 37,048 `fault(None)`
  destinations, and 260.956 GB H2D.

Evidence:

- `evidence/comfy-baselines/2026-08-25-ltx23-t2v-comfy-512-pytorch-no-fast.json`
- `evidence/comfy-baselines/2026-08-25-ltx23-t2v-engine-512-stream-boundary.json`

## Stable seam

Keep the Engine shell:

- typed recipes and public contracts;
- artifact identity, header/span validation, and checkpoint/LoRA maps;
- persistent isolated worker, cancellation, hard recovery, and prompt cache;
- request topology, progress, media mux/probe/hash, provenance, and public telemetry.

The replacement hot path is a small LTX-local source port of these state transitions:

1. one model VBAR with module-local allocations, saved signatures/views, persistent
   model-local source pins, and `prioritize()`;
2. fault → compare signature → fill only on miss → VBAR or stream-local
   `VRAMBuffer` on `fault(None)` → compute → unpin;
3. exactly one transformer block/layer ahead prefetch;
4. AIMDO file/HostBuffer transport and Kitchen flatten/unflatten at their native
   abstraction level.

Do not add a descriptor hierarchy, backend protocol, lease abstraction, source-pool
cache-key system, predictive stage budget manager, or ordinary-operation retirement
state machine.

One recipe fingerprint is one model context. Dev T2V/I2V may own Gemma, fixed text
LoRA, AV transformer/model LoRA, latent upscaler, and video/audio VAEs together.
Component transitions may pressure pages but do not purge identity. A different
checkpoint/recipe fingerprint destructively replaces the whole context.

## Keep, freeze, replace, delete

### Keep

- `ltx23_kitchen_recipe.py` and resource/catalog declarations;
- worker/session/cancel/cache/output infrastructure;
- validated checkpoint/config/key/LoRA mappings;
- prompt generation and conditioning semantics already matched to pinned Comfy;
- API, artifact, cancellation, media, and provenance tests.

### Simplify

- `ltx23_kitchen.py`: orchestration only, not memory policy;
- `ltx23_kitchen_managed.py`: worker/session behavior without private residency
  schema validation;
- `ltx23_kitchen_text.py`: retain Gemma semantics and mappings, replace bespoke
  residency;
- `ltx23_av_stored_adapter.py`: retain artifact/model mapping, replace storage,
  capture, scheduling, and activation machinery.

### Freeze until both LTX AV and Gemma stop importing them

- `runtime/framework/residency/leaf.py`
- `runtime/framework/residency/aimdo.py`
- `runtime/framework/residency/host_source_pool.py`
- `runtime/framework/residency/host_registration.py`

A read-only import audit found no non-LTX production consumers of these five files.
`runtime/framework/residency/session.py` is separate, genuinely shared by Klein,
Wan, UMT5, and stored-quant execution, and must remain. Delete the frozen files after
both AV and Gemma stop importing them; do not broaden the runtime reset before LTX
proves the smaller route.

### Delete or rewrite for LTX

- AV and Gemma leaf descriptors/scheduling groups/leases;
- root-as-residency-group and activation/restore bindings;
- custom HostSourcePool lifetimes/cache keys and registration ledger;
- `prepare_stage(required_free_bytes)` and explicit VBAR free targets;
- ordinary-operation retirement batches and detailed poison taxonomies;
- tests asserting those private objects, counters, or diagnostic shapes.

## T2V phases and gates

### Phase 1: AV direct substrate

Primary files: `ltx23_av_stored_adapter.py`, `ltx23_kitchen.py`, and at most one
small LTX-local AIMDO helper.

First falsifiable gate:

- non-zero warm VBAR signature hits;
- warm AV H2D below 200 GB;
- whole warm 512×512 request below 55 seconds;
- valid media and no stream-reuse/lifecycle fault.

If H2D remains near 261 GB or every fault remains `None`, stop and inspect native
VBAR priority/watermark/residency against pinned Comfy. Do not add another cache.

### Phase 2: Gemma on the same substrate

Use the same fault/resolve/release/prefetch functions. Do not create a text residency
subsystem. Initialize AIMDO before Torch/Kitchen-dependent GPU construction.

Gate before lifecycle cleanup:

- text H2D below 0.787 TB;
- source rereads below 84.715 GB;
- meaningful autoregressive signature hits;
- cold 512×512 runtime below 220 seconds.

If transfers collapse but runtime does not, compare Gemma math/kernel selection with
Comfy's `llama.py`. Do not invent more residency.

### Phase 3: lifecycle and final parity

Preserve VBAR/signature/pin/MMAP state for the same context, clear transient
prefetch/cast/patch state, and let AIMDO pressure component pages naturally. Purge
the complete context only on external identity change.

Initial final timing gate is within 10% of the pinned baseline:

- 512×512 cold ≤ 155.88 seconds;
- 512×512 warm ≤ 48.57 seconds;
- memory no materially worse than the pinned Comfy envelope;
- the same behavioral/memory/timing contract then passes at 768×768.

If residency counters become Comfy-like but warm remains above the gate, the next
measured comparison is Diffusers LTX forward versus Comfy's Lightricks/Kitchen fused
forward. Diffusers is not sacred, but change that axis only after residency is proven.

## After T2V

Implement I2V on the proven Dev context by adding image preprocessing, VAE encode,
initial latent insertion, and noise mask construction. Implement FLF as its separately
fingerprinted Distilled checkpoint context with guide/keyframe conditioning. Neither
operation receives a residency layer.

Generalization is deduplication of proven T2V/I2V/FLF code, not a prediction made
before parity.
