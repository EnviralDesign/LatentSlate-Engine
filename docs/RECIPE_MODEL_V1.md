# Recipe model V1

This note records the first product-definition seam earned by the existing LTX
2.3, FLUX.2 Klein 9B, and Wan 2.2 14B implementations. It is deliberately
smaller than an inference architecture.

## Evidence from the three families

The genuinely shared information is a named operation capability, its semantic
value type, a recipe-selected value, whether callers may override that value,
and any recipe-level range or choice restriction. Artifact values and ordered
collections also recur, but their identity and execution meaning do not.

The following remains family-specific:

- LTX operation identities, the single-versus-multiple transformer LoRA shape,
  strength arithmetic, two-pass T2V sampling, and operation replacement;
- Klein artifact identity, tokenizer/config consumption, ordered unweighted
  LoRAs, reference-slot preprocessing, and the fixed four-step sampler; and
- Wan high/low checkpoint ownership, primary/secondary adapter placement,
  identity composition, request-default replacement, 2+2 sampling, and direct
  high-to-low latent handoff.

The current service catalog already proves useful external descriptor concepts:
`key`, semantic `type`, `required`, `default`, and media `role`. It separately
owns LatentSlate-facing labels, widget hints, canvas/timing metadata, stable
tool identity, and request-schema hashes. V1 reuses the semantic descriptor
ideas but does not move service presentation policy or hashes into recipes.

The smallest generic vocabulary is therefore:

- a **capability**, which names and types a value the concrete operation really
  supports;
- a recipe **field**, which binds that capability to a fixed value or an
  exposed caller value and may narrow it with a range or choices;
- an **artifact** value and a strength-bearing **adapter** value; and
- a **recipe**, which resolves caller overrides and derives only its exposed
  surface.

Optionality is represented by an exposed field whose default is `None` and
whose capability accepts `None`. Ordering is a capability property and remains
visible in the derived surface. There is no schema language beyond these
in-code values.

## Resolution boundary

Recipe resolution validates override names first. A fixed or hidden field is
not an accepted override, so attempts to change it fail rather than being
ignored. Exposed fields use the caller value when present and otherwise use the
recipe default; an exposed field without a default is required. Type, range,
choice, and final family request validation run before a family adapter builds
its existing inputs.

The three proving adapters end at current family-owned structures:

- LTX T2V resolves to `Ltx23T2VIdentity` plus the keyword arguments already
  accepted by `Ltx23T2VRuntime.generate`. One adapter keeps the native
  single-LoRA fields; multiple adapters keep their tuple order in
  `transformer_loras`.
- Klein two-image resolves to `Klein9BIdentity` plus the keyword arguments
  already accepted by `Klein9BTwoImageRuntime.generate_two_image`. Optional
  `width`/`height` remain the family's existing paired optional values, and
  ordered LoRA artifacts remain part of Klein identity.
- Wan T2V resolves to the existing `WanRecipe` plus the keyword arguments
  accepted by `WanSession.generate`. Fixed ordered high/low adapter collections
  map to each model owner's primary then secondary slots. Public duration is
  converted by Wan's existing timing rule. Prompt, dimensions, native frame
  count, and seed remain request/default state and are excluded by
  `WanRecipe.identity`.

`Recipe.surface()` returns only exposed fields with semantic type, required or
default state, recipe constraints, nullability, and collection ordering. It
does not reveal hidden artifacts or fixed sampler settings and does not choose
sliders, dropdowns, text boxes, or other widgets.

## Falsification and boundaries

This seam would be forced if it required a common family runtime, generic model
identity, special-case inheritance, fake Klein sampling controls, flattened Wan
high/low ownership, loss of adapter order, or conversion of request state into
model identity. In that case the generic vocabulary should shrink rather than
the family implementation changing to fit it.

V1 is not a registry, file format, discovery system, plugin system, model or
LoRA manager, graph, sampler, cache, lifecycle, residency layer, service
protocol replacement, or LatentSlate UI. The current catalog remains unchanged;
the exposed recipe surface is proven directly in unit tests.
