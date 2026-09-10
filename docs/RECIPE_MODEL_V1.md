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

- latentslate_engine.ltx23.recipes owns LTX T2V, I2V, and FLF capabilities, including
  their distinct geometry lattices, shared duration and seed domains, T2V's
  ordered transformer adapter artifacts and strengths, and the final LTX
  request validators.
- latentslate_engine.klein9b.recipes owns the paired optional dimensions,
  ordered LoRA artifacts, ordered reference roles, and Klein validation.
- latentslate_engine.wan2214b.recipes owns separate high/low checkpoint and
  adapter capabilities, turbo settings, duration conversion, Wan validation,
  and the explicit first/last image roles used by FLF.

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

## LTX FLF breadth experiment

LTX23_FLF_CAPABILITIES represents the existing first/last-frame operation
without changing the generic vocabulary. It reuses the exact LTX T2V
capability objects for checkpoint, text checkpoint, device index, prompt,
duration, and seed because those values have the same family meaning and
inherent domain in both operations. It owns separate required start_image and
end_image capabilities with distinct semantic roles and maps them explicitly
to Ltx23FlfRuntime.generate's first_image_path and last_image_path arguments.

Width and height are deliberately not reused. LTX T2V requires a 64-pixel
lattice, while LTX FLF's existing request contract requires a 32-pixel lattice.
The FLF capability set therefore owns distinct width and height objects even
though their keys and roles match T2V. A shared field name is not sufficient
evidence of capability identity; both the semantic meaning and inherent domain
must match.

The FLF product fixes checkpoint, text checkpoint, and device index, and
exposes prompt, both ordered endpoints, width, height, duration, and seed.
Resolution produces the existing Ltx23FlfIdentity plus the existing generate
arguments. Endpoint paths, prompt, geometry, duration, and seed remain outside
model identity. Content-based ordered guide identity, separate preprocessing
and VAE encoding, width/height-sensitive guide caching, temporal placement,
fixed guide strength and sigma schedule, AV sampling/decoding, and FLF-specific
residency remain family-runtime behavior below the recipe boundary.

## LTX I2V breadth experiment

LTX23_I2V_CAPABILITIES reuses all eleven exact T2V capability objects:
checkpoint, text_checkpoint, upsampler, transformer_adapter_artifacts,
transformer_adapter_strengths, device_index, prompt, width, height,
duration_seconds, and seed. Both runtimes consume the same model and adapter
inputs, use the same text encoding and upsampler contracts, and validate through
the same LTX request domain. Their public seed controls first-pass noise.
T2V and I2V share the 64-pixel geometry lattice, minimum side 64, and maximum
pixel budget 942080. FLF retains distinct 32-pixel width/height capabilities.

I2V also reuses FLF's exact start_image capability. At the recipe boundary both
represent one required image conditioning the start of the video, with the
start_image role and a canvas-matched input contract. Different preprocessing,
encoding, and cache structures below that boundary do not change this semantic
input. I2V has no end_image capability.

ltx23_i2v_recipe fixes model artifacts, ordered adapter artifacts and strengths,
and device; it exposes prompt, start_image, width, height, duration_seconds,
and seed. resolve_ltx23_i2v maps start_image explicitly to the existing runtime's
image_path and returns Ltx23I2VIdentity. The identity is now defined once in
Torch-free ltx23/contracts.py and imported by i2v.py. The shared family-local
adapter conversion preserves no-adapter, native single-LoRA, and ordered
multiple-LoRA representations, including each artifact's paired strength.

Source paths, prompt, geometry, duration, and seed remain outside model identity.
Resolution passes paths without opening, hashing, or decoding them. The runtime
still owns FileContentIdentity, content/width/height cache invalidation,
preprocessing, low/full-resolution source encodes, conditioning, two-pass AV
sampling and decoding, fixed refinement behavior, and lifecycle/residency.
Internal conditioning strength, schedules, and second-pass seed are not recipe
capabilities.

Recipe V1.1 survived I2V unchanged. This completes the complementary reuse
experiment: identical T2V/I2V geometry shares actual objects, while FLF's
same-named but different domain remains distinct. Together with both LTX policy
variants, Klein two-image, and all three Wan operations, this is enough breadth
to begin a bounded recipe-to-service/catalog derivation experiment. Such an
experiment still needs to prove the consumed public contract; this milestone
does not integrate recipes into the service or change its schemas.

## Wan FLF breadth experiment

WAN2214B_FLF_CAPABILITIES represents the existing Wan first/last-frame
operation without changing the generic vocabulary. It reuses the exact Wan T2V
capability objects whose family semantics are identical: high/low artifacts
and adapters, text encoder, VAE, negative and positive prompt values, fixed
turbo settings, dimensions, duration, and seed.

FLF adds two family-owned capabilities: required start_image and end_image
values with distinct start_image and end_image semantic roles. They are not one
generic image collection because order is not merely presentation order: the
first and last endpoints occupy different temporal positions and swapping them
changes the request.

The FLF recipe fixes model artifacts, ordered high/low adapter collections, the
negative prompt, and the 2+2 turbo settings. It exposes prompt, both endpoints,
dimensions, duration, and seed. Resolution produces the existing WanFLFRecipe
and WanFLFSession.generate arguments. Public duration is converted with Wan's
native_frame_count rule, and endpoint order becomes first_path then last_path.

Source paths, prompt, geometry, frame count, and seed remain request/source
state and do not enter WanFLFRecipe.identity. OrderedSourceIdentity and joint
endpoint conditioning remain owned by WanFLFSession; the recipe boundary only
preserves the two semantic inputs that the session consumes.

## Wan I2V breadth experiment

WAN2214B_I2V_CAPABILITIES reuses the exact common capability objects from Wan
T2V and the same start_image capability that FLF uses. It adds no new generic
vocabulary: I2V is the common Wan video operation plus one required semantic
source image, while FLF adds the distinct end_image endpoint.

The I2V recipe fixes the same phase-owned artifacts, ordered adapters, negative
prompt, and turbo settings as the other Wan recipes. It exposes prompt,
start_image, dimensions, duration, and seed. Resolution produces the existing
WanI2VRecipe and WanI2VSession.generate arguments; source_path remains request
state and does not enter model identity.

Wan's currently fixed turbo values are now family capability domains rather
than recipe policy alone. shift, steps, split_step, and cfg each declare their
single proven value. A future recipe cannot expose an unsupported value without
first widening the Wan family capability from new execution evidence.

## Preserved V1 family adapters

- The original LTX T2V recipe still resolves one adapter through the native
  single-LoRA fields and multiple adapters through ordered transformer_loras.
- LTX FLF resolves to Ltx23FlfIdentity plus Ltx23FlfRuntime.generate
  arguments. Its endpoint order is explicit, its 32-pixel geometry domain stays
  distinct from T2V's 64-pixel domain, and guide state remains runtime-local.
- Klein two-image still resolves to Klein9BIdentity plus
  Klein9BTwoImageRuntime.generate_two_image arguments. Its fixed four-step
  sampler remains absent from the caller surface.
- Wan T2V still resolves to WanRecipe plus WanSession.generate arguments.
  High/low and primary/secondary adapter ownership remains distinct. Prompt,
  dimensions, frame count, and seed remain request state outside
  WanRecipe.identity.
- Wan I2V resolves to WanI2VRecipe plus WanI2VSession.generate arguments. It
  reuses FLF's start_image semantic input while source conditioning and cache
  identity remain below the recipe boundary.
- Wan FLF resolves to WanFLFRecipe plus WanFLFSession.generate arguments.
  Endpoint order remains explicit while source/cache identity stays below the
  recipe boundary.

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

Wan FLF did not falsify or expand the generic model. Two explicit image
capabilities represented its endpoint semantics naturally, while its ordered
content identity and joint conditioning remained family-local. It did provide
the first evidence that identical capability objects can recur naturally
across two operations in one family.

LTX FLF also did not falsify or expand Recipe V1.1. It strengthened the reuse
rule: exact capability objects are shared only when both meaning and inherent
domain match. Its 32-pixel geometry objects intentionally differ from LTX
T2V's same-named 64-pixel objects, while the genuinely identical LTX values are
shared. Recipe V1.1 therefore survived this cross-family FLF breadth case
unchanged.

This seam would be forced if it required a common family runtime, generic model
identity, resolver dispatch, special-case inheritance, fake Klein sampling
controls, flattened Wan high/low ownership, or loss of adapter order. Those
remain explicit reasons to shrink the seam rather than change a family.

V1.1 is not a registry, file format, loader, discovery system, plugin system,
model or LoRA manager, graph, sampler, cache, lifecycle, residency layer,
service protocol replacement, or LatentSlate UI. The service catalog is not
derived from these recipes, and additional operation breadth is not modeled
here.

## Two-tool service projection experiment: Outcome B

The experiment started at `ee7e883067dc6c190eeeff872cb27962db200a5d` and
used exactly `ltx23_i2v_recipe` and `wan2214b_flf_recipe`. Before any projection
code was added, the complete eight-tool `TOOLS` value was captured in
`tests/fixtures/catalog-ee7e883.json`. This is an immutable pre-change oracle,
not an expectation generated through the projection under test.

`tests/test_service_recipe_projection.py` retains a small **test-only** service
projection and explicit presentation overlay. Both probes achieve exact input
list and complete public dictionary equality against that oracle. Input order,
canvas, timing, revisions, and hashes match; the other six tools and catalog
order also match. Baseline and final hashes are:

- LTX I2V (`5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7`):
  `sha256:8364fcc55ec44ae780d49d9c9404768c81a5680783106934f9a17bd990be7efa`
- Wan FLF (`d0c202bf-7dd5-4df8-b116-f7633dc94cfe`):
  `sha256:9cf28f66f4a51f1631f4f527d26081bf72ba9644d453b1e6f65b34acbcf5601a`

### What composes, and what does not

The surface supplies input order, keys, semantic types, defaults, roles and
effective scalar constraints. Required state feeds the projection unless the
existing service wire contract explicitly overrides it. Labels (including
Start Image versus First Frame/Last Frame), multiline/placeholder hints, units,
and the choice of which constraints to publish remain service-owned. Only
width/height min and step, and duration min/max/step, enter public UI metadata.
The legal u64 seed range and scalar geometry maxima are not published there.
Hidden artifacts, adapters, and fixed sampling settings never enter the list.
Neither probe exposes nullable values or ordered collections; the proof does
not establish their mapping to the public protocol.

There is a real required/default distinction: Recipe.surface() marks width,
height, duration and seed as not required because Recipe.resolve() supplies
defaults. The catalog marks all four as required, and EngineService._validate_job
rejects their omission rather than applying those defaults. The shadow overlay
therefore explicitly retains required=True for those wire inputs. Copying
recipe required flags verbatim would change both schema hashes and admission
behavior. This is protocol policy, not a reason to alter recipe semantics.

Tool UUID/key/revision, name/description, workflow/output type, canvas and timing
remain entirely service-owned. The proof replaces only inputs on a copy of the
service definition, hashes the base without timing or schema_hash, then checks
the whole dictionary against the frozen oracle. Timing remains attached after
hashing, exactly as in production.

### Lifecycle gate and next question

Production derivation was **not wired**. `TOOLS = _tool_definitions()` and
`TOOLS_BY_ID` are constructed during service module import. Only later does
`create_app()` load .env, select the home/model roots, construct LtxModelPaths,
KleinModelPaths and WanModelPaths, and establish runtime availability. The
catalog endpoint adds availability/reasons without changing these static schemas.
Tests preserve the catalog both before configuration and with no installed models.

Both family builders require concrete hidden artifact bindings before they can
return a Recipe. Construction does not open those files, so nonexistent paths
are legitimate test fixtures, but inventing paths at production import time
would still create a fictitious bound product. CapabilitySet alone cannot
replace the product: it lacks exposed/fixed policy, defaults and recipe field
order. Recipe also requires policy for every capability, with values for fixed
fields. No existing unbound product surface satisfies the static catalog call.

The experiment falsifies direct use of the current fully bound Recipe as an
honest static catalog declaration under the unchanged lifecycle. It does not
falsify semantic input projection itself. Generic recipe.py, family definitions,
service.py and runtime behavior remain unchanged; their semantic duplication is
deliberate pending this boundary decision. No production helper or unused
framework was retained.

The next smallest question is whether these same two family products can declare
caller policy once, independently of hidden artifact binding, while a concrete
Recipe still requires real bindings. Investigate that separation before any
production integration or broader catalog migration; it is not implemented here.
