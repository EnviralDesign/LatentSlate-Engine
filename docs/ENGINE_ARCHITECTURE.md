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

- **Active-runtime replacement.** LTX closes an operation runtime and returns a
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
residency layer, service protocol, or worker owner is introduced.

## Deferred integration boundary

LatentSlate reintegration may eventually require one owner for the currently
active family runtime and a service-level request/result boundary. Those call
sites do not exist in this milestone, so defining them now would be speculative.
They remain deferred until reintegration supplies concrete ownership and
lifetime requirements.
