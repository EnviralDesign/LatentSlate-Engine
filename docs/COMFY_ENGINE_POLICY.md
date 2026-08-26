# Comfy evidence and Engine execution policy

Status: **normative architecture policy**

LatentSlate Engine uses pinned Comfy behavior aggressively but never runs
ComfyUI. See [COMFY_PARITY_LOCKDOWN.md](./COMFY_PARITY_LOCKDOWN.md) for the
mandatory operating mode while closing a parity gap.

## Runtime independence

Engine must never:

- embed or import `comfy.*`;
- launch, proxy, supervise, or require a ComfyUI process or server;
- submit, queue, or execute a ComfyUI graph;
- host ComfyUI plugins, custom nodes, model folders, or a checkout;
- expose a user-supplied ComfyUI execution route as an Engine fallback; or
- make ComfyUI availability part of recipe availability.

Comfy Kitchen and standalone comfy-aimdo are permitted low-level dependencies.
Engine does not use ComfyUI's graph executor, global model manager, plugin system,
node registry, UI policy, or AIMDO allocator integration.

Runtime independence does not require source independence. When licenses are
compatible, Engine may directly adapt a narrow ComfyUI or AIMDO helper/state
machine if that produces a smaller and less divergent implementation. Record the
pinned upstream revision, preserve required notices and attribution, and identify
Engine modifications. Do not copy a general manager when only a small inference
transition is needed.

## Ownership seam

Engine owns:

- typed recipe and public request identity;
- authenticated artifact identity and SafeTensors structural validation;
- checkpoint-name mapping and fixed LoRA identity/phase;
- model-context identity and request execution order;
- persistent worker, cancellation, hard recovery, and destructive identity switch;
- prompt/output caches, media mux/probe/hash, provenance, and public telemetry; and
- the small module-local table needed to associate weights with AIMDO allocations,
  signatures, views, and source pins.

Delegate directly to comfy-aimdo:

- `ModelVBAR`, native fault/signature/watermark/pressure behavior, and priority;
- `HostBuffer`, `ModelMMAP`, file-slice/direct-DMA transport; and
- stream-local `VRAMBuffer` physical backing.

Delegate directly to comfy-kitchen:

- `QuantizedTensor` physical layout and sidecars;
- flatten/unflatten, movement, and copy semantics; and
- quantized kernel dispatch and fallback behavior.

Engine owns execution order and the stream dependencies needed to consume and
reuse those primitives. It does not implement a second VRAM-pressure algorithm,
predictive stage budget manager, quant-layout model, or generic lifecycle above
them without demonstrated need from at least two parity-proven consumers.

## Identity and lifetime

A recipe fingerprint is one model context. For LTX 2.3 Dev T2V/I2V, that context
may own Gemma, its fixed prompt LoRA, the AV transformer and model LoRA, latent
upscaler, video VAE, and audio VAE/vocoder. Moving between those components is a
stage transition, not a model-identity switch.

Within the same context, preserve model VBAR allocations, signatures, persistent
source pins, ModelMMAP ownership, and valid caches. Clear only operation-scoped
prefetch, patch, cast, and temporary state when its source lifetime requires it.
Let new component allocations create native AIMDO pressure rather than proactively
erasing another component's identity state.

An external recipe/model identity change is destructive. Stop new work, establish
required stream quiescence, destroy temporary VRAM buffers, VBARs, HostBuffers,
file readers/mappings, fixed-LoRA state, and identity-bound caches, then construct
the new context. If native quiescence cannot be proven, hard-kill the isolated GPU
child instead of attempting clever Python cleanup.

## Authority split

1. Publisher sources own architecture, weight identity, configs, lineage, and
   license facts.
2. Pinned official workflow bytes own creator-facing topology and saved defaults.
3. Pinned ComfyUI source owns the effective node/model execution behavior used for
   parity research and narrow source adaptation.
4. Pinned comfy-aimdo source/version owns the low-level residency and transfer
   primitive contract.
5. Pinned comfy-kitchen source/version and exact headers own stored quantization,
   sidecars, layout, and kernel behavior.
6. Engine public-API evidence owns runnability, lifecycle, memory, output, and
   product-tier claims.

A lower authority may implement or verify a higher one; it may not silently
replace it.

## Source and state tracing

Before a parity edit, retain the exact workflow/revision/hash and produce both:

- a call ledger from graph node to low-level primitive; and
- a state-survival ledger naming each owner and its operation/stage/request/context
  lifetime.

For each low-level behavior, determine whether AIMDO owns it, Kitchen owns it, the
caller supplies a thin ordering/identity responsibility, or it is genuinely absent.
Then list Engine-only state objects and control edges that can be deleted.

The normalized workflow remains research evidence and is never submitted to
ComfyUI.

## Review gates

Reject any change that:

- introduces a ComfyUI runtime dependency or graph-execution surface;
- translates AIMDO/Kitchen concepts into a new stateful framework without a
  measured missing primitive;
- preserves an Engine abstraction only because tests encode it;
- changes more than one benchmark hypothesis at a time during parity lockdown;
- adds predictive generalization before the 512×512 and 768×768 gates pass;
- claims Kitchen compatibility without exact header and positive dispatch evidence;
- substitutes library defaults for decisive workflow/source behavior without a
  separately fingerprinted deviation; or
- marks cleanup/cancellation complete without observed worker/process and native
  ownership recovery.

Every parity checkpoint must report production LOC added/deleted, stateful objects
removed/added, direct AIMDO/Kitchen use, new persistent state, and the predicted and
observed benchmark movement.
