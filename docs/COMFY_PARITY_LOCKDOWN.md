# Comfy parity lockdown

Status: **normative for active parity work**

This mode exists to make an Engine runtime converge on a pinned Comfy execution
path before Engine generalization. LTX 2.3 is the proving ground.

## Mental contract

Pinned Comfy source is authoritative for the behavior under review. Current
Engine implementation and implementation-detail tests are not authoritative.

Do not translate Comfy's nouns into new Engine abstractions. Preserve its state
transitions at the same abstraction level and call comfy-aimdo and comfy-kitchen
directly where they already own the primitive.

Runtime independence is mandatory: Engine never imports or executes `comfy.*`,
runs a ComfyUI process, or submits a graph. Source independence is not required:
narrow GPL-compatible source adaptation is allowed with pinned provenance,
attribution, license notices, and clearly identified modifications.

## Before editing

Produce five compact artifacts:

1. **Call ledger:** workflow node → wrapper → model forward → cast/prefetch →
   low-level primitive.
2. **State-survival ledger:** owner and lifetime across operation, stage, request,
   and recipe-identity change.
3. **Primitive ledger:** delegate to AIMDO, delegate to Kitchen, thin caller
   responsibility, or genuinely missing.
4. **Deletion ledger:** Engine-only layers made unnecessary by the direct path.
5. **Benchmark hypothesis:** one predicted counter and timing movement.

Luna scouts are archaeologists, not architects. Use independent scouts for the
call trace, primitive ownership, and Engine delta. The main session owns the
design and final decision.

## During implementation

- Implement one behavioral delta.
- Add no model-neutral framework or backend protocol.
- Add at most one small family-local helper until parity is established.
- Prefer direct AIMDO/Kitchen objects over wrappers that rename their concepts.
- Adapt the smallest relevant upstream state machine when that is simpler than a
  clean-room rewrite.
- Preserve Engine's typed API, artifact identity, persistent worker, cancellation,
  prompt/output cache, output contract, provenance, and destructive external
  recipe-identity switch.
- Treat all subcomponents of one recipe fingerprint as one model context. Loading
  its text model, AV transformer, upscaler, or VAE is not an identity switch.
- Let AIMDO own VBAR pressure, watermarks, physical residency, HostBuffer
  implementation, file transport, and VRAMBuffer physical backing.
- Let Kitchen own quantized physical layout, sidecars, flatten/unflatten, movement,
  and quantized operator dispatch.
- Add instrumentation only when it decides the active benchmark hypothesis.
- Delete or rewrite tests that name a retired internal abstraction unless they
  can be restated as an external or low-level behavioral invariant.

## Stopping rule

After the change, run the fixed 512×512 cold/warm benchmark. If the predicted
movement does not occur, checkpoint or revert and return to the pinned source
trace. Do not add another compensating layer.

No extraction or predictive generalization is allowed until T2V passes both
512×512 and 768×768 parity gates. I2V is then implemented on the proven T2V
substrate. Shared code is extracted only from demonstrated duplication. FLF may
select a different checkpoint identity, but it never gets a residency subsystem.

## Change accounting

Every parity checkpoint reports:

```text
production LOC added / deleted
stateful classes before / after
Engine wrappers removed
direct AIMDO/Kitchen calls added
new persistent state and the pinned source behavior requiring it
benchmark prediction and observed result
```

A simplification is expected to have a strongly negative production LOC delta.
If it adds more machinery than it removes, presume the direction is wrong.

## Tests that survive a pivot

Retain tests for:

- public request/recipe and artifact identity;
- SafeTensors structural authentication;
- exact checkpoint and fixed-LoRA mappings;
- VBAR miss fill, signature hit with zero copy, eviction/refill, and `fault(None)`;
- Kitchen qdata plus every tensor sidecar;
- source-pin reuse without a new file read;
- stream-safe temporary reuse and deterministic context destruction;
- cancellation, hard worker recovery, media mux/probe/hash, and provenance.

Internal scheduler groups, leases, stage budgets, lane generations, retirement
batches, diagnostic dictionary shapes, and poison taxonomies are not product
contracts.
