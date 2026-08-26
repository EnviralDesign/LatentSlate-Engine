# Comfy evidence and Engine execution policy

Status: **Normative architecture policy**

LatentSlate Engine uses official Comfy evidence aggressively, but it never runs
ComfyUI.

## Non-negotiable boundary

Engine must never:

- embed or import ComfyUI;
- launch, proxy, supervise, or require a ComfyUI process or server;
- submit, queue, or execute a ComfyUI graph;
- host ComfyUI plugins, custom nodes, model folders, or a checkout;
- expose any user-supplied ComfyUI execution route as an Engine fallback;
- make ComfyUI availability part of recipe availability.

The permitted low-level Comfy ecosystem dependencies are **Comfy Kitchen** and
standalone **comfy-aimdo** only. Kitchen supplies stored-tensor layouts and
kernels; AIMDO supplies low-level control, VBAR, and HostBuffer primitives inside
an Engine-owned persistent GPU child. Engine imports no `comfy.*` policy,
patcher, model manager, graph, or execution code, and does not use AIMDO's ComfyUI
integration or allocator plugin. Engine may wrap AIMDO's model-local `VRAMBuffer`
and authenticated device-directed file-slice DMA as low-level transfer primitives
inside its own persistent GPU child; Engine owns their scheduling, identity,
budgets, synchronization, failure handling, and teardown.

Engine owns the typed request, orchestration, materialization, residency, caches,
cancellation, synchronization, cleanup, storage, provenance, output encoding,
and public API. It may retain authenticated immutable base sources and
runtime-identity-bound LoRA sources in Engine-owned HostBuffer source storage; this
is not a duplicate model copy. Source storage is four logical lanes
(`base`/`patch` × `warm`/temporary), allocated lazily with fixed addresses,
64 MiB base and 8 MiB patch prewarm, and a 40%-of-system-RAM registration cap.
Base and patch warm slices are reused only while their identities remain valid;
mixed, temporary, and OOM fallback slices are released after their CUDA fences.
`HostBuffer.read_file_slice` and device-directed file reads remain available only
after the SafeTensors header and absolute spans are authenticated. Whether a fill
is host-only or also targets an Engine-owned GPU destination is an Engine transfer
policy decision; file identity, destination lifetime, and stream ordering must stay
explicit and fail closed.

Source-backed LTX Gemma setup/capability failure may materialize the authenticated
CPU fallback only under automatic policy before file activation. The operator
required-AIMDO seam fails closed instead; it never silently materializes or changes
backend when HostBuffer capability/setup is unavailable.

The one-shot runtime-bootstrap child may import Kitchen only to validate the
selected installation and checks AIMDO package metadata without initializing
it. The API/managed parent never initializes or owns AIMDO. Only the persistent
GPU runtime child may call `control.init()`/`init_devices()` and allocate VBARs.

Existing recipe edition names containing `comfy` describe artifact or workflow
provenance only. They do not identify an execution backend. New names should avoid
backend ambiguity where compatibility permits.

## Dynamic residency policy

Dynamic residency is a reusable Engine framework, not a ComfyUI reimplementation.
Each weight-bearing leaf receives a VBAR allocation and a signature; a signature
hit rebinds without transfer. For LTX A/V, the root and 48 transformer blocks are
scheduling groups only: the runtime prefetches one following block, waits through
stream dependencies, computes the current block, then unpins its leaves. Stage
cleanup drops active/prefetch and temporary state while preserving valid warm base
and identity-bound patch sources. Patch invalidation purges patch-lane state.

A model-identity change is transactional and destructive: Engine drains fences,
then destroys all VBAR, HostBuffer, file-reader, cache, and wrapper ownership
before constructing the new identity. If native quiescence is not provable, the
poisoned child retains the exact graph for hard child exit rather than attempting
Python-side cleanup.

## Authority split

1. Immutable first-party publisher repositories and snapshots own weight identity,
   architecture, configs, lineage, license, and dense Reference facts.
2. Pinned official Comfy-Org workflows own practical creator-facing topology and
   saved defaults: active and disabled branches, artifact roles, preprocessing,
   prompt enhancement, conditioning order, sampler, scheduler, sigmas, steps,
   stages, CFG/guidance, dimensions, frame/fps rules, and fixed LoRA strengths.
3. Pinned ComfyUI **source** owns the observed node contract used for research:
   class names, required inputs, enum values, output slots, preprocessing, loading
   semantics, and object behavior. It is read, normalized, and tested; it is never
   imported or executed by Engine.
4. Pinned Comfy Kitchen source/version plus exact artifact headers own quantized
   marker, sidecar, scale, packing, geometry, native-dispatch, and fallback claims.
5. Engine public-API evidence owns runnability, cancellation/recovery, memory,
   provenance, output metadata, creator review, and product tier.

A lower authority may implement or verify a higher one; it may not silently replace
it.

## Clean-room translation workflow

Before Engine implementation:

1. Fetch the exact raw workflow and retain repository commit, Git blob, byte count,
   and raw SHA-256.
2. Expand subgraphs and switches into a normalized behavioral contract containing
   nodes, edges, constants, output slots, active/disabled branches, and dynamic
   placeholders.
3. Read the pinned ComfyUI source revision to verify node inputs, outputs,
   preprocessing, and execution semantics.
4. Enumerate the complete active resource closure and record configured-but-disabled
   resources separately.
5. Resolve immutable artifact identities, bytes, hashes, license/gating facts, and
   SafeTensors header/schema fingerprints.
6. Verify Kitchen layouts and direct primitives against exact headers.
7. Write independent fixtures from upstream evidence before implementation.
8. Implement the contract in Engine-owned typed orchestration and Engine-owned
   disposable workers. Call Kitchen directly where the stored layout requires it.
9. Prove the Engine implementation through the public API.

The normalized contract is a research artifact and test oracle. It is never submitted
to ComfyUI.

## Golden implementation pattern

FLUX.2 Klein is the house example:

- official workflows and pinned ComfyUI source define operation behavior;
- exact publisher and Comfy-Org files define resource identity;
- Engine owns typed recipes, loading, staged residency, caches, lifecycle, and API;
- Engine calls Kitchen directly for accepted stored quantized paths;
- native dispatch counters and zero-fallback assertions prove the intended path;
- no ComfyUI process, module, graph executor, folder staging, or plugin host exists.

Wan 14B follows the same architecture: operation-specific expert closures and
saved defaults are derived from pinned official workflows, while Engine-owned stored
materialization, Engine workers, and direct Kitchen/native dispatch perform execution.

## Status correction for nonconforming paths

A prototype that imported or launched ComfyUI, submitted a graph, or depended on a
local ComfyUI source tree is not an accepted Engine implementation, even when it produced valid
media. Its workflow, artifacts, settings, and observations may remain research
evidence, but the recipe must be treated as unavailable until an Engine-native
implementation passes acceptance.

Accordingly:

- the historical Wan 5 ComfyUI-executed/imported-graph prototype remains
  nonconforming and is not acceptance. The exact stored-mixed T2V/I2V recipes are
  narrow Hardware-proven
  Recommended through LatentSlate target-hardware output, direct Kitchen dispatch,
  and disposable-worker cancellation/recovery. Comfy is a source oracle only; no
  ComfyUI process or graph participates and no pixel/latent parity is claimed;
- LTX 2.3 optimized Engine-native Dev T2V/I2V and Distilled FLF are narrow
  Hardware-proven Recommended through LatentSlate public-API output, exact dispatch,
  and lifecycle evidence. Comfy is a source oracle only; no pixel/latent parity is
  claimed. LTX 2.5 optimized workflows remain source contracts awaiting an
  Engine-native Kitchen-backed implementation;
- accepted Klein and Wan 14 paths are described as Engine-native stored runtimes,
  not ComfyUI backends.

## Review gates

Reject any design, documentation, or implementation that:

- adds a ComfyUI import, package dependency, process, server, graph submission,
  custom-node host, workspace, or folder-staging requirement;
- labels workflow-derived Engine execution as an upstream execution backend;
- treats a pinned ComfyUI source revision as a deployable dependency;
- substitutes native/Diffusers defaults for a decisive saved workflow without a
  separately fingerprinted deviation;
- claims Kitchen compatibility without exact header and positive native-dispatch
  evidence;
- treats installed or cataloged artifacts as runnable;
- infers output metadata from the request;
- marks cancellation complete without observing worker exit, cleanup, memory return,
  poisoned-state eviction, and fresh recovery.

All active model roadmaps and implementation handoffs must link to this policy.
