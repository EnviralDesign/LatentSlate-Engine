# Recipe model V1.1

This note records the smallest product-definition seam earned by the existing
LTX 2.3, FLUX.2 Klein 9B, and Wan 2.2 14B implementations. V1.1 separates the
family capability declaration from recipe policy and proves the separation with
two different products over one LTX T2V operation. It remains deliberately
smaller than an inference architecture.

## Three layers

### Family capability

A Capability names one semantic value genuinely consumed by an operation. It
declares its value type, whether None is meaningful, whether it is an ordered
collection, its media role when applicable, and inherent scalar bounds,
increments, or choices. A CapabilitySet groups the exact capability objects
for an operation and owns cross-value validation that cannot be expressed by a
single scalar domain.

Family modules declare these objects once:

- latentslate_engine.ltx23.recipes owns LTX T2V capabilities, including the
  64-pixel geometry lattice, duration domain, seed domain, ordered transformer
  adapter artifacts and strengths, and the final LTX request validator.
- latentslate_engine.klein9b.recipes owns the paired optional dimensions,
  ordered LoRA artifacts, ordered reference roles, and Klein validation.
- latentslate_engine.wan2214b.recipes owns separate high/low checkpoint and
  adapter capabilities, turbo settings, duration conversion, and Wan
  validation.

The capability layer does not own model loading, sampling, adapter arithmetic,
residency, preprocessing, lifecycle, caches, or media output.

### Recipe policy

A Recipe contains one Field for each object in a family capability set. A
field either fixes a value or exposes it to callers with a default or required
state. Exposed policy may narrow a capability range, increment, or choice set,
but construction fails if policy would admit values outside the family domain.
Fixed fields are hidden and attempts to override them fail.

Requiring fields to reuse the declared capability objects makes the ownership
boundary concrete: recreating an equal-looking capability inside a recipe is
rejected. The recipe selects a product; it does not redefine what the family
operation supports.

### Exposed caller surface

Recipe.surface() is derived only from exposed fields. It reports semantic type,
required/default state, effective constraints, nullability, media role, and
collection ordering. Hidden artifacts and fixed settings do not appear. The
result contains no slider, dropdown, label, grouping, layout, or other UI
policy.

The current service catalog still owns LatentSlate-facing labels, widget hints,
canvas/timing metadata, tool identity, and request-schema hashes. V1.1 does not
derive or change the catalog.

## LTX falsification experiment

LTX23_T2V_CAPABILITIES is one declaration shared by all current LTX T2V
product recipes. The V1-compatible recipe preserves its existing exposed
geometry surface. Two additional proving recipes deliberately make different
policy choices over the same capability object identities:

- ltx23_t2v_locked_recipe fixes checkpoint, text checkpoint, upsampler,
  ordered adapters and strengths, device, 768x512 dimensions, and five-second
  duration. It exposes only prompt and seed.
- ltx23_t2v_tunable_recipe fixes the same artifact ownership but exposes
  prompt, seed, dimensions, duration, and the ordered adapter-strength
  collection. Its geometry and duration domains are narrower than LTX's family
  domain, and its adapter strengths are constrained to 0.0 through 1.0.

Both use resolve_ltx23_t2v. Resolution produces the existing
Ltx23T2VIdentity plus the keyword arguments already accepted by
Ltx23T2VRuntime.generate; there is no second execution path.

### Adapter control that survived

The V1 Adapter value was sufficient for fixed ordered adapters but could not
expose artifact and strength independently without replacing a whole nested
value. V1.1 does not add a nested field system. LTX instead declares the two
values it already consumes: an ordered artifact tuple and a parallel ordered
strength tuple. Family validation requires matching lengths. The tunable recipe
fixes the artifact tuple and exposes only strengths; resolution zips them in
order into LTX's existing single- or multi-LoRA identity fields.

This representation is intentionally local to LTX's proven need. Klein keeps
its ordered unweighted LoRA artifacts, and Wan keeps strength-bearing adapters
in separate high/low primary and secondary ownership.

## Preserved V1 family adapters

- The original LTX T2V recipe still resolves one adapter through the native
  single-LoRA fields and multiple adapters through ordered transformer_loras.
- Klein two-image still resolves to Klein9BIdentity plus
  Klein9BTwoImageRuntime.generate_two_image arguments. Its fixed four-step
  sampler remains absent from the caller surface.
- Wan T2V still resolves to WanRecipe plus WanSession.generate arguments.
  High/low and primary/secondary adapter ownership remains distinct. Prompt,
  dimensions, frame count, and seed remain request state outside
  WanRecipe.identity.

The family adapters live beside their family contracts. The generic
latentslate_engine.recipe module now contains only inert values and policy
resolution primitives and imports no family or Torch runtime.

## Falsification result and boundaries

The experiment supports a real distinction between family capabilities and
recipe policy: the locked and tunable LTX products share one capability set,
derive different caller surfaces, and converge on the same family identity and
request structures. It also forced one simplification of V1's design: adapter
controls are expressed as the smallest concrete ordered values LTX needs,
rather than through a universal nested adapter editor.

This seam would be forced if it required a common family runtime, generic model
identity, resolver dispatch, special-case inheritance, fake Klein sampling
controls, flattened Wan high/low ownership, or loss of adapter order. Those
remain explicit reasons to shrink the seam rather than change a family.

V1.1 is not a registry, file format, loader, discovery system, plugin system,
model or LoRA manager, graph, sampler, cache, lifecycle, residency layer,
service protocol replacement, or LatentSlate UI. Wan FLF and additional
operation breadth are not modeled here.
