# Runtime framework inventory

This is the Phase 0 baseline for consolidating Engine runtime infrastructure. It
describes the production tree at `0f069939311b8d88b9dd4c1502412858a6bf6d6e`
on `main` (2026-08-19). The earlier design brief referenced
`ba10140af7239d7742482c441b55ae78ab74213a`; the current tree adds the exact
Z-Image fixed LoRA and exact canvas contracts. Those accepted behaviors are part
of this baseline.

The objective is deletion of repeated mechanisms. This inventory does not make
the proposed `runtime/framework/` directory structure a requirement: the
existing small shared modules should be extended or moved only when they own the
right capability.

## Current extraction checkpoint

The current bounded implementation includes separate disposable and persistent
process/session policies:

- `runtime/wan22_canvas.py` now owns the import-light Wan canvas contract, so
  catalog and CLI discovery do not import Torch through conditioning code;
- `runtime/framework/worker/` owns canonical JSON, SHA/HMAC bindings, bounded
  JSON reads, atomic JSON publication, and bounded partial-record-safe JSONL
  append/drain; all five managed-worker families use those primitives, while
  family adapters retain their characterized authentication and progress-schema
  policy;
- the same worker package now owns a model-neutral disposable parent supervisor
  and child harness. A static model-free handler proves cold success, bounded
  progress, safe failure, malformed/oversized request rejection, forced
  process-tree cancellation, and cleanup without importing model or GPU code;
  Wan 5 is the first production consumer;
- a separate model-neutral persistent parent supervisor and static child
  command-loop harness own fixed-command launch, warm process reuse, bounded
  progress/heartbeat drains, independent watchdog clocks, boundary rechecks,
  cooperative cancel grace, Job Object poisoning, and cleanup. An
  actual-subprocess model-free handler covers cold/warm PID reuse, command
  sequence/replay/tamper rejection, full-result binding, clock independence,
  cancellation, corruption, crashes, safe failure, unload-once, and recovery;
  Z-Image is the first production consumer;
- `stored_quant.py` owns bounded duplicate-safe SafeTensors header reads and the
  public global-FP8/NVFP4 restoration primitives used by Klein, LTX, and Z;
- `runtime/framework/residency/` owns the process-wide exclusive stored-session
  lease and concrete-device normalization used by Klein, Wan, UMT5, and the Wan
  VAE;
- `runtime/framework/stored_quant/` owns the neutral FP8/INT8, direct FP8,
  direct NVFP4, and additive-LoRA execution modules used by Wan, Wan 5, UMT5,
  Klein, and LTX Gemma;
- the private cross-family import exemption set is now empty;
- Z-Image Qwen is split into a Torch-only architecture shell, exact checkpoint
  planner, and runtime composition/execution/proof module. The former mixed-Qwen
  monolith and its compatibility surface are deleted;
- architecture tests keep framework/shared modules model-neutral, reject new
  private or public model-named cross-family reach-throughs and dynamic loaders;
  both cross-family ledgers are now empty.

Wan 5 still owns its unkeyed binding,
typed recipe/generation contract, progress-stage meaning, exact safe failure
fields, result/proof validation, and output publication rules. Z-Image still
owns its per-session HMAC schemas, concrete device/session identity, typed
commands, model progress mapping, exact result/proof validation, safe failure,
private DACL, and output publication rules; only its generic persistent process,
IPC, watchdog, command-loop, and cleanup mechanics moved.

## Classification

- **I** — behavior is effectively identical.
- **E** — structurally equivalent, but policy or schema differs.
- **M** — irreducibly model-specific.
- **H** — historical or compatibility implementation, not the selected optimized path.
- **U** — uncertain and must be characterized before migration.

## Existing shared capabilities

| Capability | Current owner | Assessment |
|---|---|---|
| Runtime selection, bounded wrapper cache, unload/evict | `runtime/manager.py` | Reuse; do not create a second manager. |
| Windows kill-on-close process tree and tree-empty proof | `runtime/windows_process.py` | Reuse; process messages should become model-neutral. |
| Process memory observation | `runtime/process_memory.py` | Reuse. |
| Pipeline/prompt/media cache helpers | `runtime/cache.py` | Reuse where the model contract permits caching. |
| Stored FP8, legacy FP8, and INT8 ConvRot description/restoration | `stored_quant.py` | Extend carefully; it is already the neutral restoration seam. |
| Capacity-based full/grouped residency decision | `runtime/residency_policy.py` | Retain as policy; it is not a lifecycle/session implementation. |
| Artifact identity and bounded SafeTensors probing | `artifacts.py` | Reuse; do not duplicate header readers in the final state. |

## Managed-worker inventory

### Production paths

| Runtime | Process/session | IPC and authentication | Liveness/cancellation | Result, cleanup, and model-owned work |
|---|---|---|---|---|
| Wan 5 Kitchen | Disposable child per job through the shared supervisor/harness; one `DisposableProcessTree`; no warm reuse. | Output-adjacent request/result/progress/gate/staging files. Request binding remains the same unkeyed SHA-256 fingerprint; no result HMAC. 1 MiB JSON/progress bounds and 4096 progress records. | The shared disposable policy preserves the 100 ms poll, no heartbeat/deadline, and forced tree termination on parent cancellation. | Wan owns exact recipe/endpoint revalidation, output path/size/SHA, stream facts, dispatch/zero-fallback proof, progress stages, and safe diagnostic fields. The framework owns request publication, fixed-command launch, progress transport, tree lifecycle, and owned-path cleanup. |
| Z-Image Turbo | Persistent exact-recipe child through the shared persistent supervisor/harness; serial commands; healthy success remains warm in the same PID; any command error/cancel/timeout poisons the session. | Random private temp directory with protected Windows owner/SYSTEM DACL (0700 elsewhere); request/command/result/progress/heartbeat/gate/cancel files. Per-session 32-byte secret; Z-owned HMAC session, request, and complete result bindings. Atomic JSON; 1 MiB bounds. | The shared persistent policy preserves the 100 ms poll, independent 30-minute hard deadline, 12-minute stage deadline, 45-second heartbeat, ordered boundary-race recheck, cooperative cancel marker and 5-second grace, then Job Object termination. | Parent validates fresh 1024-square PNG, SHA/size, exact Qwen/NextDiT/LoRA/CUDA-health proof, failure schema, and safe locations. The Z handler owns recipe rehydration, concrete identity, CUDA health, exact materializers, progress mapping, sampling, decode, result publication, and safe diagnostics. |
| LTX 2.3 Kitchen | Persistent request-bound child; serial commands; healthy success remains warm; command failure/cancel poisons the session. | Random temp directory with 0700 best effort; request/command/result/progress/gate files. Per-session secret; HMAC request and complete result bindings. Canonical JSON rejects NaN. Atomic JSON; 1 MiB and 4096-record bounds. | 100 ms poll; progress and process-death observation; no independent heartbeat or absolute job deadline. Cancellation terminates the full worker tree. | Parent validates exact recipe/guides, MP4 identity, closed A/V metadata, source-derived audio timing, dispatch/LoRA proof, failure schema, and cleanup. Child owns LTX load, staged execution, audio/video decode, mux, and safe diagnostics. |
| Wan 14 native | Persistent exact-recipe/device child; serial commands; healthy success remains warm; failure/cancel poisons the session. | Random private temp directory with protected Windows owner/SYSTEM DACL (0700 elsewhere); request/command/result/progress/gate files. Per-session secret and HMAC session/request bindings. Result only echoes the request binding; the complete result envelope is not independently HMAC-bound. Atomic JSON; 1 MiB and 4096-record bounds. | 100 ms poll; process-death and cancellation observation; no heartbeat or absolute job deadline. Cancellation forces full tree termination. | Parent validates endpoints, stream metadata, exact component identity, expert/LoRA/dispatch proof, output path/size, and cleanup. Child owns operation-specific Wan load, stage policy, sampling, encode, and model provenance. |
| LTX 2.3 BF16 reference | Disposable child per job; no warm reuse. | Output-adjacent request/result/progress/gate files; unkeyed request fingerprint; no complete result authentication. Atomic JSON; 1 MiB and 4096-record bounds. | 100 ms poll; no heartbeat/deadline; parent cancellation forces tree termination. | Parent validates request-bound output metadata and cleanup. Child owns Diffusers reference load/execution. This remains reachable for reference recipes, not dead code. |

### Secondary and historical paths

| Path | Classification | Notes |
|---|---|---|
| `_DisposableNativeWanI2VRuntime` and the first half of `wan22_native_managed.py` | **H** | Retained validation/source-history precursor. The exported production class is the later persistent `ManagedNativeWanI2VRuntime`; the file currently contains both implementations. |
| `wan22_prompt_worker.py` launched from `runtime/wan22.py` | **M/E** | Disposable CPU-only prompt encoder using a temporary directory and explicit timeout. It predates the authenticated model-worker protocols, writes a raw request and worker log, and returns SafeTensors conditioning. It needs a separate privacy/reachability decision before migration. |
| Worker launch helpers embedded in remaining managed modules | **I/E** | Wan 5 and Z-Image no longer embed `Popen`, start-gate, Job Object assignment, polling/drain, termination, command-loop, heartbeat, or cleanup mechanics. Remaining managed families and the LTX BF16 disposable reference retain characterized family implementations pending separate migrations. |

### Worker mechanism comparison

| Mechanism | Wan 5 | Z-Image | LTX Kitchen | Wan 14 | LTX BF16 |
|---|---:|---:|---:|---:|---:|
| Atomic canonical JSON | I | I | I | I | I |
| Bounded JSON/progress | I | I | I | I | I |
| Private capability directory | no | yes | partial | yes | no |
| Keyed request authentication | no | yes | yes | yes | no |
| Complete result authentication | no | yes | yes | no | no |
| Persistent command loop | no | yes | yes | yes | no |
| Heartbeat | no | yes | no | no | no |
| Absolute/stage deadline | no | yes | no | no | no |
| Cooperative child cancel signal | no | yes | no | no | no |
| Parent-forced tree termination | yes | yes | yes | yes | yes |
| Poison-on-command-failure | n/a | yes | yes | yes | n/a |
| Warm worker reuse | no | yes | yes | yes | no |
| Output identity revalidation | yes | yes | yes | yes | yes |
| Closed safe failure schema | yes | yes | yes | limited | limited |

The shared worker design must therefore be based on policy objects/callbacks, not
the assumption that all workers are persistent or that all models expose the
same progress stages. Z-Image is the strongest existing lifecycle contract;
Wan 5 is the smallest disposable consumer.

## Quantized execution inventory

| Consumer / format | Stored and logical representation | Sidecars and marker | Restore / transfer / execution | Residency and proof |
|---|---|---|---|---|
| Shared FP8 | 2-D `F8_E4M3`, logical F16/BF16/F32, same logical shape | F32 scalar scale; optional bounded U8 Comfy marker depending on contract | `stored_quant.restore_stored_quantized_tensor` builds Kitchen `TensorCoreFP8Layout`; neutral execution supports fixed/dynamic activation scaling | Used by Wan/UMT5/Klein/LTX Gemma with model-owned proof. |
| Shared INT8 ConvRot | 2-D INT8, logical dense shape | F32 `[rows,1]` scale plus bounded U8 marker/global layer metadata with ConvRot group size | Shared restore builds `TensorWiseINT8Layout` | Z NextDiT and Wan direct Kitchen paths; consumers own dispatch counters. |
| Klein global FP8 | 2-D `F8_E4M3`, source logical dtype | positive F32 `weight_scale`; fixed or dynamic activation scale | Shared restore and neutral `StoredFP8Linear`; activation FP8 quantization then Kitchen `scaled_mm_v2`; dense fallback forbidden | Whole/grouped transformer residency; per-module native/rejected/fallback counters. |
| Klein NVFP4 | packed 2-D U8 with explicit logical shape | F32 tensor scale, F8 block scale `[stored rows, stored cols/8]`, positive F32 input scale | Shared restore and neutral `StoredNVFP4Linear`; Kitchen `quantize_nvfp4` + `scaled_mm_nvfp4`; dense fallback forbidden | Whole/grouped residency and native proof. |
| Wan 14 / Wan 5 / UMT5 | FP8, legacy FP8, or INT8 ConvRot depending on exact artifact | weight scale, optional input scale, marker/global ConvRot metadata | Shared restore; neutral `StoredFP8Int8Linear` executes FP8 through Kitchen `scaled_mm_v2`, INT8 through Kitchen-aware `F.linear` | Block-group/staged sessions; native, INT8, rejection, and dense-fallback counters. |
| LTX 2.3 A/V | 2-D FP8 with BF16/F32 dense state | F32 scale and input-scale contract | LTX-local restore and direct Kitchen FP8 linear | Module-storage capture plus staged transformer residency; aggregate exact module/call proof. |
| LTX 2.3 Gemma | 34 FP8 + 302 NVFP4 stored text linears | Shared scale/block-scale topology | Uses shared restoration and neutral FP8/NVFP4 execution; direct Kitchen kernels, no dense fallback | LTX-local stage/residency plus 336-module dispatch proof. |
| Z Qwen | 177 FP8 + 12 NVFP4, CPU-master bytes; logical F32 operation weight | FP8 scalar scale; NVFP4 scalar + F8 block scale | Uses shared restoration. Per operation, moves raw bytes/scales, calls public Kitchen direct F32 dequant, then F32 `F.linear`, and releases the dense temporary. No activation quantization or scaled-mm. | Per-operation streaming; exact 189 dequant/linear deltas, zero rejection/fallback, CPU-master retention proof. Raw transport is Z-local but represents a stable layout operation. |
| Z NextDiT | 202 INT8 ConvRot linears plus BF16/F32 state | row scale and ConvRot marker/group size | Shared restore; Z-specific direct Kitchen `int8_linear@cuda`; fixed LoRA bypass remains model-specific | Staged whole component movement; 202-module direct dispatch proof and zero fallback. |

Model-specific checkpoint key maps, layer counts, fused-QKV slicing, scheduler
semantics, and accepted proof extensions stay with the model. SafeTensors header
parsing, FP8/NVFP4 layout restoration, byte-preserving transport, generic
counter arithmetic, and zero-fallback validation are extraction candidates.

## Residency inventory

| Strategy currently present | Implementations | Shared versus model-specific |
|---|---|---|
| Whole-model or grouped stored residency | `KleinTransformerResidencySession`, `SynchronousBlockResidencyManager`, `WanTransformerResidencySession` | Capacity decision is shared; safe group definitions and physical movement are model-specific. Guard/transition/rollback/proof mechanics are structurally equivalent. |
| Staged component residency | Wan expert/text/VAE sessions; `_LTX23TransformerResidency`; Z Qwen/NextDiT/VAE stages | Component order and safe overlap are model-specific. State transitions, exclusive leases, rollback, cancellation cleanup, and telemetry are shared candidates. |
| CPU-master retention with per-operation streaming | Z Qwen; parts of LTX module storage | Byte/layout movement and temporary-release mechanics are stable candidates; target selection and operation math remain model-specific. |
| Tensor/module storage capture | LTX `LTX23ModuleStorage`/`LTX23ModuleBinding`; stored-linear move methods in Klein/Wan/Z | Similar ownership problem with incompatible current representations. Characterize before unifying. |
| Runtime-wrapper residency | `RuntimeManager` | Already shared and orthogonal to tensor residency. |

## Cross-family dependencies

High-priority private reach-throughs at the baseline:

- the former Z-Image mixed-Qwen monolith imported private Klein restoration
  helpers;
- `ltx23_kitchen_text.py` imports those two private restore helpers and Klein's
  model-named FP8/NVFP4 linear wrappers.
- `umt5_stored_adapter.py` and `wan21_vae_adapter.py` import the private
  `_read_safetensors_header` from `wan22_stored_adapter.py`.

The extraction removes all six private imports above and both remaining public
model-named execution dependencies. LTX Gemma and Klein now consume the same
neutral stored execution and additive-LoRA implementation directly; UMT5 and
Wan consume the same neutral FP8/INT8 implementation. No compatibility-only
facade or model-named re-export remains.

The temporary Klein restoration facades have also been deleted: Klein, its
mixed text encoder, LTX, Z, and their restoration tests call the public neutral
FP8/NVFP4 functions directly.

Wan 5 intentionally reuses Wan-family tokenizer, VAE, UMT5, and stored-linear
implementations. That is real reuse, but several owners are version/model-named;
the migration must extract neutral capabilities rather than merely relabel Wan
2.2 modules as shared.

## Baseline measurements

| Measurement | Baseline |
|---|---:|
| Runtime Python files | 56 |
| Runtime production lines | 33,162 |
| Test lines | 32,186 |
| Collected tests | 1,195 |
| Z runtime / test lines | 6,035 / 3,781 |
| Wan runtime / test lines | 10,277 / 5,544 |
| LTX 2.3 runtime / test lines | 8,302 / 5,481 |
| Klein runtime / test lines | 4,674 / 3,639 |

At the recorded baseline, production files over 1,000 lines were `klein_stored_adapter.py` (2,032),
`ltx23_kitchen.py` (1,989), `wan22_stored_adapter.py` (1,766),
`z_image_mixed_qwen.py` (1,544), `ltx23_av_stored_adapter.py` (1,461),
`wan22_native_managed.py` (1,299), `klein.py` (1,255),
`ltx23_kitchen_managed.py` (1,188), `z_image_turbo_managed.py` (1,135), and
`z_image_turbo_worker.py` (1,013). Files between 800 and 999 lines are
`ltx23_kitchen_text.py`, `wan5_kitchen.py`, `kit.py`, and
`umt5_stored_adapter.py`.

The current tree replaces that 1,544-line Z file with
`z_image_qwen_architecture.py`, `z_image_qwen_checkpoint.py`, and
`z_image_qwen_runtime.py`; the baseline measurement remains above for audit
comparison rather than current ownership.

Repeated helpers remain in unmigrated families for typed bindings, paths,
endpoint hashing, safe failure validation, output removal, and last-worker
status construction. Wan 5 and Z-Image now share their policy-appropriate
generic transport, process-tree, drain/wait, termination, child-loop, and
cleanup mechanics without sharing model schemas.

The first locked full-suite run at this SHA produced two baseline failures:

1. terminal color output was disabled by the invoking environment
   (`NO_COLOR=1`, `TERM=dumb`); it passes with a terminal-capable test environment;
2. `cli_product` imported Torch because the new catalog canvas constant lived in
   `wan22_i2v_conditioning.py`; the Phase 0 dependency correction moves that
   static contract to `runtime/wan22_canvas.py`.

## Accepted hardware evidence that must survive

- Klein 4B and 9B optimized T2I/I2I paths have target-hardware output,
  ordered-reference, direct Kitchen, switching, cancellation, recovery, and
  zero-fallback evidence; BF16 recipes remain references.
- Wan 5 optimized T2V/I2V have target-hardware output, direct Kitchen,
  disposable-worker cancellation/recovery, and zero-fallback evidence.
- Wan 14 base T2V/I2V/FLF and fixed LightX variants have target-hardware output,
  persistent-worker lifecycle, direct Kitchen, cancellation/recovery, and
  zero-fallback evidence.
- LTX 2.3 Dev T2V/I2V and Distilled FLF have target-hardware A/V timing,
  direct Kitchen, cancellation/recovery, and zero-fallback evidence.
- Z-Image Turbo base and the fixed 70s Horror LoRA have target-hardware cold,
  warm, cancel, recovery, recipe switching, deterministic switch-back, exact
  Qwen/NextDiT/LoRA dispatch, and zero-fallback evidence.

## Characterization coverage and gaps

Existing suites already pin worker envelopes, result rejection, progress bounds,
warm reuse, poison/eviction, cancellation races, process-tree cleanup, output
identity, exact dispatch structures, stored restoration, and CPU-master
immutability. The strongest suites are `test_z_image_turbo_managed.py`,
`test_ltx23_kitchen_managed.py`, `test_native_wan_managed.py`,
`test_wan5_kitchen_managed.py`, and the stored-adapter contract suites.

Before each migration, add model-free characterization for the exact mechanism
being extracted. Persistent replay/duplicate commands, heartbeat/deadline
policy, oversize handling, complete result authentication, forced tree exit,
and poison/fresh recovery are now characterized. Do not weaken a model-specific
proof to fit a shared schema.

The shared bounded reader deliberately classifies absent, empty, oversized,
unstable, invalid-UTF8, and invalid-JSON files as one fail-closed framework
error. Family adapters retain their existing public safe error envelopes, but
malformed private IPC is not promised its former incidental parser exception
type. Valid canonical JSON and every authenticated binding remain byte-for-byte
unchanged.

## First bounded migrations

1. Restore the import-light catalog baseline.
2. Add architecture guardrails with a fixed, shrinking exemption set.
3. Extract canonical JSON, HMAC/result signing, bounded reads, and atomic writes
   behind model-neutral tests; do not yet change process/session policy.
4. Build a model-free worker fixture and shared disposable parent/child
   lifecycle. **Implemented.**
5. Migrate Wan 5 first, then treat Z-Image as a separate persistent stress test.
   **Wan 5 and the separate Z-Image persistent extraction implemented.**
6. Extend the existing `stored_quant.py` seam with neutral NVFP4 restoration;
   migrate Z and LTX off Klein-private helpers.

This order preserves model mathematics. The disposable seam must gain a second
production consumer only when that migration deletes its existing mechanics;
the persistent seam must not become a flag-heavy extension of the disposable
class.
