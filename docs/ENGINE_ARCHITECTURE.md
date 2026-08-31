# Engine architecture after three proven families

This is the first Engine-wide extraction earned by the completed LTX 2.3,
FLUX.2 Klein 9B, and Wan 2.2 14B turbo families. It starts from accepted
baseline `3c68910` and preserves the family behavior certified in
`CANONICAL_PARITY_CERTIFICATION.md`.

The Engine still consists primarily of three independent family packages. The
shared layer contains only request invariants whose meaning is already identical
across the packages.

## Extracted seams

### Unsigned 64-bit request values

`latentslate_engine.validation.validate_u64` owns the common type and range
check used by every public seed. LTX, Klein, and Wan retain their existing
family request validators, exported maximum-seed constants, and error labels.
Only the identical unsigned 64-bit invariant is shared.

This seam has three proven family consumers and replaces repeated bool/type and
range checks without changing request coercion or family-specific geometry,
duration, or frame rules.

### Request-file content identity

`latentslate_engine.identity.FileContentIdentity` owns the path-independent
identity of request-supplied files. It records the resolved path for the current
load, but equality uses only byte size and SHA-256. Consequently, identical
bytes at another path reuse derived state and changed bytes invalidate it.

The value is consumed by LTX I2V and FLF, Klein two-image, and Wan I2V and FLF.
Each family still owns its cache entries, semantic roles, derived tensors, and
invalidation dimensions:

- LTX I2V retains two resolution-specific source latents; LTX FLF retains two
  independently encoded ordered guides.
- Klein retains two ordered reference slots with distinct preprocessing rules.
- Wan I2V retains one latent/mask entry keyed by source plus request shape; Wan
  FLF retains one joint causal encode keyed by an ordered endpoint pair plus
  request shape.

Sharing the content value therefore removes duplicate hashing without flattening
the cache units or their lifetimes.

## Deliberate non-seams

The following responsibilities remain family-local because the current evidence
does not support one simpler ownership model:

- **Family-internal runtime replacement.** LTX closes an operation runtime and returns a
  new object; Klein destructively resets one reusable object in place; Wan
  destroys a session, rejects use of the dead session, and updates request-state
  recipe fields when model identity is unchanged. A common helper would require
  lifecycle adapters and obscure which object remains valid.
- **Model identity composition.** LTX recipe identities are concrete operation
  fields, Klein includes resolved model and consumed tokenizer/config artifacts,
  and Wan combines two checkpoint/LoRA owners with fixed recipe settings.
- **Request and shape derivation.** The three families use different spatial
  lattices, limits, temporal domains, and latent formulas. Only seed range is
  identical.
- **Prompt, source, and guide caches.** Prompt keys, retained encoders, natural
  source cache units, semantic ordering, and shape invalidation differ by family
  and operation.
- **Results and artifact writing.** LTX returns video plus stereo waveform for
  an audio/video writer, Klein writes PNGs and reports reuse flags, and Wan
  returns video metadata and stage timings.
- **Sampling, conditioning, text, LoRA, residency, and media pipelines.** These
  are part of the certified family inference and resource behavior, not common
  Engine mechanisms.

No generic family interface, sampler, model manager, registry, cache manager,
or residency layer is introduced.

## Public service boundary after the third consumer

The current LatentSlate client now supplies the concrete service call sites.
`latentslate_engine.service` owns the consumed HTTP protocol, bounded
session-local uploads and in-memory jobs, one FIFO GPU queue, and downloadable
job artifacts. Its public catalog contains the three preserved LTX 2.3
operations, the two explicit FLUX.2 Klein 9B operations, and the three accepted
Wan 2.2 14B turbo video operations.

Torch and native GPU state stay outside the HTTP process in one spawned,
active-family worker. A small process owner synchronously closes the current
worker before starting an incompatible one, so LTX, Klein, and Wan GPU contexts
never coexist. LTX retains its proven operation-specific workers. Klein uses one
family worker backed by `Klein9BTwoImageRuntime`; because it extends the base
runtime, that process serves T2I and two-image requests without replacing the
model, prompt conditioning, or ordered reference caches. Live T2I -> two-image
-> T2I -> two-image execution retained one PID and reported model/prompt reuse
throughout, with both reference slots still reused after the intervening T2I.

Wan likewise uses one family process, but owns one current operation session at
a time. Repeating an operation retains that session and its natural prompt and
image-conditioning caches. Switching T2V, I2V, or FLF destructively closes the
old session before creating the exact accepted operation session in the same
process. This preserves the family's proven operation-local ownership instead
of introducing a generic runtime interface.

This is process ownership, not a family runtime interface. Dispatch remains
explicit for the eight public tools. Per-operation availability checks only the
required artifacts, and the single-primary-artifact service path chooses MP4
for LTX and Wan or PNG for Klein. Family request validation and media semantics
stay explicit: LTX still requires canvas-matched source images, while Klein and
Wan uploads are only image-validated and their original bytes reach family-owned
preprocessing.

The accepted Klein paths resolve from `LATENTSLATE_ENGINE_HOME` under
`models/klein9b`. `LATENTSLATE_KLEIN9B_VAE` is the one optional file override
for installations that share the accepted VAE from another local model folder;
there is no model discovery or search-path system.

Wan paths resolve from `LATENTSLATE_WAN_MODEL_ROOT`, defaulting to
`LATENTSLATE_ENGINE_HOME/models/wan2214b`, using an explicit ComfyUI model
layout. The Wan child removes the service's LTX-oriented `cudaMallocAsync`
allocator setting before its first Torch import, preserving the native allocator
under which the accepted Wan sessions were certified.

The HTTP process still owns cancellation truth. A running cancellation sets a
request flag while the single native call continues to a safe boundary. Only
after the worker replies does the job become `canceled`, and its output is
discarded. No incompatible job starts before that reply. `DELETE /v1/runtime`
exits the idle active worker; a busy runtime returns a conflict instead of
releasing native state under execution.

Live service evidence exercised LTX -> Wan -> Klein -> Wan with a different PID
for each family activation and each predecessor gone before its successor ran.
Same-operation Wan requests retained one PID and reused prompt conditioning;
identical I2V and ordered FLF bytes at new paths reused image conditioning,
while swapped FLF endpoint roles invalidated it. A canceled native Wan request
remained running while incompatible Klein work stayed queued, then exposed no
artifact before Klein began. Idle release terminated the final Wan worker and
reduced total device use from 5,519 MiB to 2,450 MiB.
