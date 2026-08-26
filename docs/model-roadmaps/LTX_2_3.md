# LTX 2.3 roadmap

Last authority audit: **2026-08-25**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Lightricks sources and Engine’s immutable BF16 closure at upstream `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` |
| saved topology/defaults | official T2V, I2V, and FLF workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| node behavior / dispatch | ComfyUI v0.34.0 [`12d5279438bfefc058a269eae805ceab6047777f`](https://github.com/Comfy-Org/ComfyUI/tree/12d5279438bfefc058a269eae805ceab6047777f) for research only; Kitchen [`7c6ca3a5b63857d42c2d49777d6afb69de23f13f`](https://github.com/Comfy-Org/comfy-kitchen/tree/7c6ca3a5b63857d42c2d49777d6afb69de23f13f) for direct Engine dispatch |
| acceptance/tier | all three optimized operations are narrow Hardware-proven Recommended through LatentSlate-originated Engine public-API A/V, exact dispatch, and lifecycle evidence; BF16 remains structural only |

## Decision

Current main owns a strong native BF16 structural Reference, not a practical local
product path. One bounded RTX 5080 OOM is enough; dense output belongs on Vast.

Practical source contracts:

- T2V and first-frame I2V: Dev FP8 plus fixed Distilled LoRA;
- FLF: Distilled FP8 with ordered endpoints, a separate graph/transformer line.

The pinned workflow widgets request **1280×720**, while the source latent grid
produces **1280×704**. Engine keeps its strict Dev `/64` boundary and defaults to
the honest effective 1280×704 canvas; it never accepts or silently coerces 720.
Both the source request and Engine target remain explicit in provenance.

Parity iteration no longer uses that source canvas as an acceptance gate. Routine
cross-family comparisons use **512×512**, and the heavier/final local gate uses
**768×768**. Both are exact `/64` canvases shared with image and other video model
families; the older 1280×704 runs remain historical evidence only.

All three Engine-native optimized operations are **narrow Hardware-proven
Recommended**: Dev T2V, strict first-frame Dev I2V, and ordered-endpoint Distilled
FLF. Recommended is operation-local here: FLF is its own typed operation, not a quality
alternate to T2V/I2V. The saved workflow contracts remain the topology/default oracle,
but no ComfyUI process or graph participates and no pixel/latent parity is claimed.

## BF16 truth

The exact 50-file closure totals **94,977,693,482 bytes** and includes transformer,
encoder/tokenizer, connectors, scheduler, video/audio VAEs, vocoder, and configs. It
fixes 24 fps, 8 steps/CFG 1, `8n+1` frames, aligned dimensions, synchronized 48 kHz
stereo, typed operations, and Engine isolated-worker cancellation boundaries.

Proof: **Cataloged / structurally validated Reference, one bounded local OOM**.

## Optimized implementation truth

The three operations bind their exact workflow revision and raw JSON SHA-256, all
active A/V resources, operation-specific fixed LoRAs (none for FLF), stored-FP8/NVFP4
header contracts, additive LoRA targets, direct Kitchen transformer dispatch, strict
Comfy full-precision Gemma execution, Engine-owned typed
orchestration, persistent isolated workers, cancellation, mux, output probing, and provenance.
The ordinary optimized
profile is **72,529,224,527 bytes** across seven canonical resources; each installed
file has one NTFS identity and acquisition rejects staged multiply-linked files.

The current source timing contract is preserved directly in Engine: at five seconds,
Comfy requests 126 frames at 25 fps and the LTX temporal grid floors that to 121
effective frames (4.84 seconds). The corresponding audio closure is latent `L=121`,
mel `M=481`, decoded `230880`, target `232320`, pad `1440`, policy
`source_derived_exact_duration_v1`. The prompt cache is bounded to eight CPU-owned
entries / 1 GiB. The bound covers both 385,359,872-byte positive and negative Gemma
node outputs plus prompt/proof metadata, and cached provenance retains the source text/LoRA proof
without claiming a new dispatch. No workflow is submitted to ComfyUI.

The output proofs below predate parity revision `ltx23-comfy-baseline-v2` and remain
historical lifecycle evidence. Current strict-text comparison evidence is recorded in
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-engine-strict-text-v3.json`; the earlier
single-conditioning comparison remains in
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-engine-parity-v2.json`.

### Parity-v2 acceptance checkpoint (2026-08-25)

Implementation, focused tests, and independent review are complete for 25 fps
flooring, strict 1280x704 geometry, exact Comfy VAE tiling, and the bounded prompt
cache. The first candidate cold job (`2ded3dae-1cb0-44ec-991c-e9e46483ea7d`)
succeeded in 460.589 seconds and proved the 1280x704 / 121-frame / 25 fps media
contract, but also proved that the original 256 MiB cache bound could not retain the
385,359,872-byte conditioning value. The bound is now 512 MiB and a prompt-policy
miss must fail closed unless it publishes a non-empty cache entry.

The corrected cold retry (`e6af1ba1-df9b-4b53-a647-bee99c6327ac`) was blocked at
the first transformer refill by external TouchDesigner VRAM use: only about
10.25-10.57 GiB was free versus the runtime's approximately 11.14 GiB conservative
root/refill threshold. This was a safe residency refusal, not a cache allocation or
Kitchen failure. Engine is unloaded and no corrected cold/warm cache proof has been
claimed. Resume after TouchDesigner exits: restart Engine for a clean cold boundary,
confirm free VRAM exceeds the refill threshold, then rerun the same prompt at seeds
`810138461690240` and `810138461690241` and require cold publication followed by a
warm prompt hit.

A later corrected cold attempt passed prompt publication and denoise, but host RAM
peaked at 68,407,611,392 bytes and the parent hit `MemoryError` while polling progress
during upscale. The 385,359,872-byte CPU prompt copy had been retained before denoise.
Publication now occurs only after mux/probe/hash and large media/latent references are
released; bounded JSONL polling also limits each read allocation to 64 KiB. Acceptance
was then repeated successfully from a clean GPU boundary.

The accepted comparison pair used the same captured Comfy prompt and seed-only warm
change. Cold job `d2c83722-ecf5-41c0-b57c-25509f661669` completed in 446.199 seconds,
published one 385,361,554-byte prompt entry, and produced 1,126,804 bytes with SHA-256
`35a576dc467cd0d4559b453dc882646c90272a9aeffc7a9d56d26ff63004baee`.
Warm job `e491f617-9a92-41c3-a729-c07280109707` completed in 92.978 seconds, reused
the same persistent worker and entry, skipped new text dispatch, and produced 1,695,915
bytes with SHA-256 `69bfb0d3576df30c01307368273213039398ccc4f0ea0d15af82e32d2f5f39f9`.
Both outputs are 1280x704, 121 frames, 25 fps, 48 kHz stereo. Warm execution is close
to strict Comfy (92.978 versus 101.011 seconds); cold remains approximately 1.96x
slower (446.199 versus 227.674 seconds), isolated to text materialization, enhancement,
and encoding scheduling. This is comparison evidence, not final recipe acceptance.

That pair predates the strict text-node parity slice and is retained as historical
diagnostic evidence. The current path uses one Gemma patcher with LoRA active only for
enhancement, base positive and exact base negative encodes, strict Comfy
`full_precision_mm`, exactly two bounded CUDA transfer streams, best-effort in-place
host registration of eligible authoritative parameters, per-layer completion barriers,
and a 1 GiB dual-node-output warm cache. The registration ledger is kept only for the
short Engine text stage and unregistered after synchronized text offload; pageable
fallback is authenticated rather than rejected. Authenticated metadata carries
phase/cumulative timings and transfer/patch counters.

A fresh strict-text cold/warm pair is now complete. Cold job
`32fb3fc1-69e7-4b1d-a33e-01352da2d993` completed in 445.708 seconds of authenticated
worker time (449.423 seconds polling wall time). It published one 770,722,916-byte
dual-conditioning entry and produced 1,179,237 bytes with SHA-256
`d82b1a87edba60da5f9787fc5cb6474b5cbd9d9993fd5c485287ec54679c1b27`.
Warm seed-only job `a80e2748-3295-4c4b-ae1e-c0adca4ea431` completed in 91.096
seconds of worker time (92.741 seconds polling wall time), hit the same entry, performed
no new text or negative dispatch, and produced 1,161,873 bytes with SHA-256
`3e45bbd2293ed92ceff479083d642ad98cc1f74b3a9d92aa688594151ca65ccd`.
Both outputs are 1280x704 H.264, 121 frames at 25 fps / 4.84 seconds, with 48 kHz
stereo AAC. Warm remains close to strict Comfy (91.096 versus 101.011 seconds); cold
remains approximately 1.96x slower (445.708 versus 227.674 seconds). This is valid
strict-text behavior evidence, not cold-time recipe acceptance.

The cold trace isolates the remaining gap. Materialization took 58.654 seconds, text
onload 1.387, LoRA prompt enhancement 205.215, base positive and negative encodes
2.558 combined, synchronized text offload 0.387, and downstream execution 176.890.
All 1,300 eligible physical allocations (8,561,786,368 bytes) registered in place and
were unregistered after text with zero failures or leaks. The text stage independently
hardcodes root-plus-one-layer residency and therefore performed 9,216
transfer/wait/completion-barrier cycles. The separately reported 10,256,685,465-byte
activation reserve belongs to downstream transformer/VAE policy and does not cause
text streaming. The current implementation slice adds direct standalone
comfy-aimdo 0.4.13 DynamicVRAM residency for Gemma only while leaving downstream
policy unchanged. Kitchen values flatten to qdata plus every tensor sidecar,
use Comfy's 1024-byte sub-layout inside 512-aligned VBAR allocations, and rebuild
as identical typed CUDA views. Signature hits reuse without copying; misses
refill authoritative CPU state. This was the superseded coarse-transfer slice:
its 49 scheduling groups were not Comfy-like allocation units and its single
transfer buffer intentionally did not provide downstream prefetch overlap.
Signature hits still copy nothing, and fault-None uses an operation-scoped Torch CUDA
buffer. LoRA enhancement and base encoding use distinct dirty epochs without
merging or checkpoint reload. This is implementation evidence only until a real
hardware run proves output, timings, memory, cleanup, and zero live VBAR bytes;
prompt, model, sampler, denoiser, VAE, geometry, timing, and cache semantics are
unchanged.

Failed-fill ownership and teardown are fail-closed. The backend installs an
owned pending token (including any fault-None CUDA buffer) before reconstruction,
copy, event, or wait operations. A proven device barrier permits ordinary cleanup
and rethrows the primary failure; a failed barrier retains every reference and
marks the session terminal. The persistent LTX GPU child then skips runtime
unload and uses exit code 86 through `os._exit`, after a best-effort bounded
failure record, bypassing Python and AIMDO finalizers. The managed parent discards
that session and creates a new child for the next request.

Production's unindexed `cuda` device is resolved once to the active `cuda:N`
before AIMDO initialization; VBAR, control initialization, copy streams, and
module bindings retain that identity. Hardware acceptance sets
`LATENTSLATE_LTX23_REQUIRE_AIMDO=1` in the GPU child environment so a supported
run rejects pre-allocation Engine-hook fallback instead of publishing it as
strict AIMDO evidence. This is an operator smoke seam, not a recipe parameter.
The adapter now passes Comfy's current initialization arguments (`nvml_pressure=True`,
no simple-VRAM headroom, and zero per-device headroom), but it still initializes
lazily when the text stage is constructed. Because Torch may already have created a
CUDA context by then, this is not yet initialization-order parity; a pre-context
GPU-child bootstrap remains a separate required slice.

Required-AIMDO job `8e3def94...` is also rejected evidence: residency setup and
copies completed, but prompt enhancement returned no suffix after roughly 475 seconds.
The runtime now uses Comfy's pinned 39-line system instruction, exact manual turn
template, stop token 106, `<think>` removal, and source-prompt fallback. An opt-in
operator diagnostic separately checks the actual installed Gemma root and
first/middle/last layer physical tensors byte-for-byte across miss, signature hit,
and forced-refill paths before another full generation is attempted.

That real-weight diagnostic is now accepted for storage residency. The first
attempt correctly rejected layer 0 at `_param_scale`: wrapping the freshly
unflattened Kitchen tensor in `Parameter` detached it and cloned scale sidecars
before the raw VBAR fill. Dynamic reconstruction now preserves custom-tensor
Parameter identity in place without detach or copy. The rerun used the installed
artifact with header SHA-256 `7fcb15cd...`, and root plus layers 0, 24, and 47
passed exact physical-byte comparison for initial miss, identity-preserving
signature hit, and forced refill. Teardown ended with 49 allocations reduced to
zero live allocations, zero live/loaded bytes, one native free, no host-registration
leaks, and no poisoned state. The retained record is
`evidence/comfy-baselines/2026-08-25-ltx23-gemma-aimdo-residency.json`. This accepts
storage/signature/cleanup only; logits, output, cold/warm cost, initialization
order, and Comfy-granularity transfer parity remain open.

The next bounded transport slice is also accepted on the same real-weight oracle.
At that point, one fixed native AIMDO `HostBuffer` packed the same 1024-aligned
physical layout and performed one gathered H2D/event/wait per miss.
The accepted rerun reported `copy_strategy=gathered_host_buffer`, no fallback,
8 misses with exactly 8 events and 8 waits instead of 172 per-physical events,
4 zero-copy signature hits, and a single 2,013,770,752-byte reusable buffer.
Close unregistered and freed that buffer and again left zero live/loaded VBAR bytes.
That isolated transport evidence remains historical; the coarse schedule was later
replaced by per-leaf allocation and block scheduling/prefetch.

The next implementation slice removes the 9,447,702,218-byte base Gemma CPU
materialization: immutable authenticated SafeTensors spans and real meta Kitchen
FP8/NVFP4 templates now describe base physical fields, while LoRA and generated
runtime buffers remain bounded CPU sources. At that stage, misses filled the gathered
HostBuffer with blocking host-only file reads before H2D.
The file is revalidated immediately before open and retained until synchronized
HostBuffer/VBAR disposal; post-open read or quiescence failures retain ownership
and fail closed. HostBuffer setup/capability fallback is auto-policy-only; the
operator required-AIMDO seam never materializes the CPU fallback. Hardware
acceptance for this source-backed path is recorded below.

Required-AIMDO T2V job `ac965cb6...` then completed the same bounded 1-second
smoke that previously failed, producing a valid synchronized MP4 with nonempty
enhancement output and no fallback. Internal cold time was 269.285 seconds:
69.879 materialization, 56.865 enhancement, 3.332 combined base encodes, and
137.505 downstream. Its paired prompt-cache hit `9238052e...` skipped all text
dispatch and completed internally in 65.896 seconds. Cold text residency recorded
7,301 signature hits, 98 gathered misses, exactly 98 events/waits, no per-physical
misses, and zero-live cleanup. Machine-wide monitored peaks were 13,440 MiB VRAM /
68,380,340,224 bytes RAM cold and 13,379 MiB / 68,263,477,248 bytes warm. Therefore
functional cold/warm behavior is accepted, but RAM and full 5-second cost parity are
not. The retained record is
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-aimdo-gathered-smoke.json`.

Base Gemma source residency is accepted as authenticated file-backed storage.
The language shell retains meta dense tensors and real Kitchen meta FP8/NVFP4
templates; active base fields are read by absolute SafeTensors spans into Engine-owned
HostBuffer source lanes, while LoRA and generated runtime fields remain identity-bound
CPU or patch sources. The real-weight
diagnostic completed in 28.413 seconds, read 4,804,813,992 bytes across 162 calls for
the initial and forced misses, performed no source read/copy on signature hits, and
matched every selected qdata/scale/block-scale byte. Before close the source and one
file handle were live; afterward that handle had exactly one matched close and the
source, HostBuffer view, HostBuffer, and all VBAR residency were dead/zero. Required
mode cannot CPU-materialize on setup failure. Direct file-to-device DMA is still out
of scope; the accepted path uses blocking host-only `HostBuffer.read_file_slice`
followed by the gathered H2D.

The paired required-AIMDO file-backed T2V smoke is accepted for this bounded slice.
Cold job `a1f6f7bf...` produced a valid synchronized MP4 in 161.021 internal seconds:
16.113 materialization, 42.158 enhancement, 3.902 combined base encodes, and 97.618
downstream. This is 40.2% below the preceding 269.285-second CPU-master smoke, with
materialization down 76.9%. It read 17,200,367,232 authenticated base bytes over
2,528 HostBuffer file calls, retained the same 7,301 signature hits / 98 misses /
98 events and waits, and closed its single source handle with zero live VBAR or
HostBuffer state. Paired warm job `cf33cb34...` hit the prompt cache, performed no
text dispatch, and completed in 62.283 seconds. Runtime unload then returned the
host to a 589,103,104-byte working set with no cleanup errors.

This is not full parity acceptance. The cold whole-machine RAM peak remained
68,124,815,360 bytes (about 63.45 GiB), nearly unchanged from the prior smoke,
because downstream model loading/paging now dominates after the text boundary.
The 1-second run is also not directly comparable to the retained 5-second Comfy
oracle. Evidence is retained in
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-aimdo-file-backed-smoke.json`.

The Windows checkpoint-map crash is now closed for the bounded T2V smoke. Two
pre-fix cold workers (`4ee1708d...`, `bce59f3b...`) died with native status
`0xC0000005` while materializing the embedded vocoder. The runtime had been opening
the same 29 GB checkpoint seven times. It now owns one mapping across transformer,
connectors, video VAE, audio VAE, and vocoder construction/materialization, while
the latent upscaler retains its independent artifact mapping. Independent review
found no retry-blocking issue, and cold job `5e43f323...` crossed the old boundary
and completed in 154.606 internal seconds (13.186 materialization, 42.975 prompt
enhancement, 4.117 combined base encodes, 93.005 downstream). Paired warm job
`25afe339...` hit the prompt cache and completed in 59.769 seconds. Runtime unload
left no cleanup errors and returned the host working set to 587,563,008 bytes.

The new signed phase telemetry made the pre-pivot gap explicit. Cold
process-private memory peaked at 61,650,317,312 bytes and whole-system used RAM at
68,461,969,408 bytes; warm began at 62,658,920,448 private bytes and whole-system
used RAM peaked at 66,555,428,864 bytes. Both runs retained a 12,366,905,344-byte
CUDA reservation. Those measurements characterize the retired 23,722,941,536-byte
synchronous CPU-master / 49-coarse-group design, not the current leaf residency
implementation. The checkpoint-map and 1-second cold/warm evidence remain historical;
RAM/cost parity and the five-second oracle remain open. Evidence is retained in
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-shared-checkpoint-map-smoke.json`.

Benchmark job `8912bd2a...` is rejected evidence: the prior implementation attempted
`Tensor.pin_memory()` copies and required complete physical-tensor coverage, duplicating
roughly the full Gemma CPU master before failing at the root. The corrected path never
copies or canonicalizes CPU masters; it deduplicates registration by physical pointer
and size under the Comfy-consistent 40%-of-system-RAM Windows budget.

### Current dynamic-residency policy

LTX uses Kitchen plus standalone `comfy-aimdo` low-level primitives; it imports no
`comfy.*` execution or memory-policy code. Every weight-bearing leaf receives its
own VBAR allocation and signature. The root and 48 transformer blocks are scheduling
groups only: each forward prefetches the next block, waits by stream dependency,
computes the current block, and unpins the consumed leaves. Signature hits transfer
no bytes.

The Engine-owned HostBuffer pool has four logical source lanes (`base`/`patch` crossed
with warm/temporary lifetime). Lanes are lazy fixed-address append stores with 64 MiB
base and 8 MiB patch prewarm and a 40%-of-system-RAM registration cap. Like Comfy's
narrow native seam, each append first extends with `register=False`, authenticates the
exact appended view, and then calls `cudaHostRegister` explicitly. Registration or
physical-RAM admission refusal rolls back exactly and uses direct file/device or
pageable copying without poisoning the model; structural ownership failures remain
terminal. Immutable base
and runtime-identity-bound LoRA sources may remain warm; temporary, mixed, and OOM
sources are reclaimed after fences. Stage cleanup retains safe warm sources, while
patch invalidation purges patch lanes. A model identity switch transactionally drains
and destroys VBAR, HostBuffer, file, cache, and wrapper ownership before loading the
new identity. If quiescence fails, the exact graph remains resident only for hard
child-process exit.

The current required-AIMDO 1-second acceptance pair is cold job
`31cf9282-daaa-4146-9c86-a7573a16b641` and identical-request warm job
`441546b5-034c-47dd-8d51-0c7b1b11f5f4`. Cold completed in 243.161 seconds
(139.325 text, 103.590 downstream); warm hit the prompt cache, performed no text
dispatch, and completed in 69.575 seconds. Both published valid 1280x704, 25 fps,
48 kHz stereo MP4s without poison. Monitored peaks were approximately 63.42 GiB RAM
and 15,678 MiB VRAM. This accepts the leaf-VBAR, split-registration, pressure-fallback,
and same-model warm lifecycle; it is not a replacement for the five-second Comfy
timing oracle. Retained evidence:
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-comfy-leaf-hostbuffer-smoke.json`.

The strict five-second apples comparison is now complete and rejects timing parity.
Cold job `9540970f-d0c8-43b0-bf31-ab30ef009bfd` completed in 542.384 seconds
against Comfy's 227.674; identical-request warm job
`b0309954-5faf-4a22-9f8d-705e88f0cd23` hit the prompt cache, spent zero time in
all text phases, and completed in 344.010 seconds against Comfy's 101.011. Both
outputs correctly contain 121 frames at 1280x704 / 25 fps with synchronized 48 kHz
stereo. Memory is close: Engine cold/warm peaks were 62.73/61.11 GiB RAM and
15,717/15,844 MiB VRAM versus Comfy's 61.736/61.161 GiB and 15,386/15,258 MiB.

The warm request isolates the gap to downstream AV execution: 37,048 faults, zero
VBAR signature hits, 37,048 fault-None temporaries, 36,432 leaf prefetches, and about
243 GiB H2D. Source caching works (18,797 hits and only four new base-file reads),
but every successful hot-path lease release still performs a host-blocking CUDA event
synchronization. The next parity slice is stream-ordered AV completion/cleanup like
Comfy, retaining whole-device barriers only for failure and final close. Evidence:
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-leaf-hostbuffer-5s-comparison.json`.

The active low-cost oracle was recaptured after updating ComfyUI to v0.34.0,
`comfy-aimdo` 0.4.15, and `comfy-kitchen` 0.2.31. At 512×512 with the exact shared
silver-bird prompt, five-second/25 fps contract, and no Sage or FP8-matrix-multiply
launch flag, Comfy completed in 141.709 seconds cold and 44.157 seconds warm. The
warm run cached 37 nodes. Sampled whole-machine peaks were 60.484/61.393 GiB RAM
and 15,097/15,074 MiB VRAM. Both outputs are 121-frame, 4.84-second H.264/AAC
files with 48 kHz stereo audio. This is the current T2V timing and memory oracle:
`evidence/comfy-baselines/2026-08-25-ltx23-t2v-comfy-512-pytorch-no-fast.json`.

The current Engine checkpoint closes the asynchronous sampler-to-upscaler lifecycle
failure but does **not** establish timing parity. Cold job
`fe16fbe3-7915-4f89-8950-7ab85cceeae4` and same-session prompt-cache-hit job
`17615569-a2c6-4261-b00c-f6ea2a2d7e87` both produced valid 512×512, 121-frame,
25 fps H.264/AAC outputs with 48 kHz stereo. Engine runtime totals were 438.088
seconds cold and 60.542 seconds warm, versus Comfy's 141.709 and 44.157 seconds.
Sampled Engine peaks were 63.001/62.432 GiB whole-system RAM and 15,739/15,423
MiB GPU use, so memory is close while timing remains rejected.

The stage boundary now performs one explicit retirement drain, leaves zero pending
retirements, and preserves warm source/VBAR ownership; the prior upscaler OOM did not
recur. The remaining costs are directly observed: cold prompt enhancement alone took
288.837 seconds and moved 1,573,431,776,256 bytes H2D while rereading
338,856,403,776 source bytes. The warm AV request still recorded 37,048 misses,
zero signature hits, 37,048 fault-None destinations, and 260,956,151,808 H2D bytes.
This checkpoint is intentionally available for independent architecture review;
neither its implementation nor its tests constrain a future Comfy-aligned rewrite.
Evidence: `evidence/comfy-baselines/2026-08-25-ltx23-t2v-engine-512-stream-boundary.json`.

| Operation | LatentSlate app job / Engine job | Output proof |
| --- | --- | --- |
| Dev T2V cold | `625a798e-5afc-4393-a5b1-ef5bb9f38d2f` / `7a03e1f2-cf83-4a19-986a-4612ddad4624` | 1,331,572 B; SHA-256 `2ddccdf05f4167c5c558a96eb593b9ced6cead703ffc0156ca3adbd33dc3a711` |
| Dev T2V recovery | `ddcca150-cadc-4394-9aec-3765e0c6d96d` / `f7548f73-7599-4585-9560-1e8dcf49f69e` | fresh worker; 1,225,703 B; SHA-256 `f5bdf109d0676ccd96f2a6afb22e4d6e13fe64cc992b6abdf1a18c3225617634` |
| Dev I2V, strict one start image | `94dde246-912b-4468-81c3-f27653f14d7a` / `02bb5e57-f342-4a49-b085-a1d1cdf46602` | 1,126,917 B; SHA-256 `7a0d847c3d9cf5c48faa221ec3d95e29fc12f8cfe4370f60139e77be15a5142a` |
| Distilled FLF, strict ordered endpoints | `cabb3a3a-64a6-4c93-9961-97b4d8b88926` / `263f2b6a-3843-4605-9937-5397961712b8` | 801,222 B; SHA-256 `0728298a9bef42468cd4d6cc433f67acdc92da5b79c45eb5e195f00fc9448ff7`; intentionally no LoRAs |

Dev T2V/I2V recorded exact Kitchen `1496/1496` module closure with `16456` native
calls; Distilled FLF recorded `1462/1462` with `11696` native calls. Reject/fallback
counts were zero. Cancel job `d6b22fe5-e06e-499b-98dd-55bec97cb2ae` /
`78134062-206e-4501-b191-e2efad9e80d5` stopped at 54%, emitted no artifact, and left
the output tree empty before the fresh-worker T2V recovery.

## Next slices

1. close T2V's stream-ordered residency, reusable fault-buffer, and managed-stage
   lifetime gaps against the 512×512 Comfy cold/warm oracle without changing sampler
   semantics or importing ComfyUI;
2. pass the 768×768 T2V cold/warm gate, then stabilize the proven model-local seam as
   reusable Engine framework rather than LTX-only orchestration;
3. apply that seam to Dev I2V and Distilled FLF using separately pinned authoritative
   Comfy workflow contracts and the same 512×512 / 768×768 harness;
4. optional batched dense BF16 Vast comparison, without calling it locally accepted;
5. no new 2.3 feature without specific compatibility value.

Stop on ComfyUI dependency, partial closure, lineage
substitution, hidden conversion/fallback, assumed A/V metadata, or unobserved cleanup.
