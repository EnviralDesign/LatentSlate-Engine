# Recipe model V1.1

This note records the smallest product-definition seam earned by the existing
LTX 2.3, FLUX.2 Klein 9B, and Wan 2.2 14B implementations. V1.1 separates the
family capability declaration from recipe policy and proves the separation with
two different products over one LTX T2V operation. It remains deliberately
smaller than an inference architecture. All eight current public tools now use
ProductPolicy for production caller semantics and configured Recipe binding.

## Capability, policy and binding

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
- latentslate_engine.klein9b.recipes owns concrete T2I and paired optional
  two-image dimensions, ordered LoRA artifacts, ordered reference roles,
  and Klein validation.
- latentslate_engine.wan2214b.recipes owns separate high/low checkpoint and
  adapter capabilities, turbo settings, duration conversion, Wan validation,
  and the explicit first/last image roles used by FLF.

The capability layer does not own model loading, sampling, adapter arithmetic,
residency, preprocessing, lifecycle, caches, or media output.

### Recipe policy

A Field either fixes a value or exposes it to callers with a default or required
state. Exposed policy may narrow a capability range, increment, choice set or nullability,
but construction fails if policy would admit values outside the family domain.
Fixed fields are hidden and attempts to override them fail.

For all six LTX and Wan video products, ProductPolicy declares those Fields once
before concrete model selection. Capabilities absent from its fields are required hidden
bindings; no placeholder Field or artifact value represents them. Already-known
fixed policy, such as Wan's turbo settings, can still use fixed Fields.

### Concrete binding

ProductPolicy.bind supplies exactly the deferred hidden values and returns a
Recipe containing one Field for every family capability. The existing family
builder APIs perform this binding for all six video products. The locked/tunable
LTX demonstrations and Klein two-image still construct Recipe directly.

Requiring fields to reuse the declared capability objects makes the ownership
boundary concrete: recreating an equal-looking capability inside a recipe is
rejected. The recipe selects a product; it does not redefine what the family
operation supports.

### Exposed caller surface

ProductPolicy.surface() and Recipe.surface() share the same projection of exposed
fields. It reports semantic type,
required/default state, effective constraints, nullability, media role, and
collection ordering. Hidden artifacts and fixed settings do not appear. The
result contains no slider, dropdown, label, grouping, layout, or other UI
policy.

The current service catalog still owns LatentSlate-facing labels, widget hints,
canvas/timing metadata, tool identity, and request-schema hashes. All six video
tools source their input semantics from unbound ProductPolicy surfaces; their
public schemas remain exactly unchanged. The two Klein tools retain explicit
service input declarations.

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
service protocol replacement, or LatentSlate UI. All six video tools now derive
their service input semantics from product policy. Klein is not migrated.

## First two-tool service projection experiment: Outcome B

The experiment started at `ee7e883067dc6c190eeeff872cb27962db200a5d` and
used exactly `ltx23_i2v_recipe` and `wan2214b_flf_recipe`. Before any projection
code was added, the complete eight-tool `TOOLS` value was captured in
`tests/fixtures/catalog-ee7e883.json`. This is an immutable pre-change oracle,
not an expectation generated through the projection under test.

`tests/test_service_recipe_projection.py` initially hosted a small **test-only**
projection and explicit presentation overlay. Both probes achieved exact input
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

At that starting implementation, the only product declarations were builders
requiring concrete hidden artifact bindings to return a Recipe. Construction
did not open those files, but inventing paths at production import time would
still create a fictitious bound product. CapabilitySet alone could not replace
the product: it lacked exposed/fixed policy, defaults and recipe field order.
Recipe required policy for every capability, with values for fixed fields.

That experiment falsified direct use of the fully bound Recipe as an honest
static catalog declaration. It did not falsify semantic projection itself. It
changed no production code and left the lifecycle question for the following
bounded experiment.

## Two-product lifecycle separation

Starting from `4d7b32981583311f3ae2357279b75d7b4d3fbca2`, the same LTX I2V
and Wan FLF products now declare unbound caller policy once:

```text
CapabilitySet (family types, domains, roles and cross-value validation)
    -> ProductPolicy (exposure, defaults, narrowing and field order)
    -> bind(concrete hidden values) -> Recipe
    -> resolve(caller overrides) -> existing family identity and request
```

`LTX23_I2V_POLICY` and `WAN2214B_FLF_POLICY` live in their family recipe modules.
Their surfaces are available at import time without paths, model configuration
or a Recipe instance. They reuse the existing Field/fixed/exposed vocabulary;
ProductPolicy is the only new public generic concept. A capability not included
in the policy fields must be bound as hidden. Deferred fields precede declared
fields in the resulting Recipe, in capability-set order, preserving the two
existing products' complete field order. Exposed field objects are reused, not
redeclared for catalog and runtime consumers.

The unchanged ltx23_i2v_recipe and wan2214b_flf_recipe APIs now supply concrete
hidden bindings through their corresponding policy. LTX still separates ordered
adapter artifacts and strengths; Wan retains separate high/low ordered adapter
collections. Wan's fixed turbo values live in its policy, while checkpoints,
adapters, text encoder, VAE and negative prompt arrive at binding time.

Binding rejects missing or unknown keys, exposed-field bindings, and attempts
to replace policy-fixed values. Each supplied value passes the existing
capability normalization/domain validation through fixed(). Cross-field checks
stay on CapabilitySet and run during Recipe.resolve(), at the same point as
before; binding does not load models or resolve family identities.

Complete surface, resolved-value, family-identity and request expectations were
verified against the pre-change builders before implementation. They still pass
for both default and overridden requests, alongside the existing LTX adapter
order/strength and Wan ownership tests. Narrowing the shared policy changes
both its surface and the existing builder's bound behavior, proving that there
is no second caller-policy declaration. Generic choices, nullable defaults and
ordered values retain their existing meanings.

A clean-process test imports and reconstructs both policies with filesystem APIs,
artifact creation and binding blocked; Python's normal source/bytecode loading
is the only filesystem activity permitted. Torch, dotenv and service imports
are forbidden, and the environment (including the CUDA allocator setting) stays
unchanged. The shadow service tests at that milestone consumed unbound policies
directly without fake artifacts. Both complete public dictionaries and the
hashes above equaled the untouched frozen oracle.

This supported the lifecycle separation without introducing a FieldPolicy type,
binding object, registry or second resolver. That milestone left production
service.py and TOOLS unchanged, earning the following two-tool integration.

## Two-tool production catalog integration

Starting from `62a0ef7d1379abf9bc3cc1b55baff207817faab1`, the static service
catalog consumes `LTX23_I2V_POLICY.surface()` for I2V_ID and
`WAN2214B_FLF_POLICY.surface()` for WAN_FLF_ID. No other public tool migrated.

```text
CapabilitySet -> ProductPolicy
                    -> surface() -> static semantic service projection
                    -> bind(real hidden values) -> Recipe -> runtime resolution
```

The service-local `_video_policy_inputs` helper supplies existing `_input`
descriptors. Input order, keys, semantic types, defaults, roles and effective
scalar constraints come from the policy surface. Recipe required state also
comes from the surface, except that the service explicitly requires width,
height, duration_seconds and seed on the wire. Recipe defaults still do not
make those HTTP inputs optional, and `_validate_job` is unchanged.

The service owns labels, including Start Image versus First Frame/Last Frame,
the multiline prompt hint and placeholder, and the seconds unit. It publishes
only width/height min and step and duration min/max/step. Seed bounds, scalar
geometry maxima and arbitrary choices are not copied into UI metadata. The
existing service schema builders retain tool UUID/key/revision, name/description,
workflow kind, output type and canvas. Timing stays on its existing service path
and is still attached after schema hashing. Intentional canvas/timing duplication
is not expanded into another derivation.

All eight complete tool dictionaries, ordering, revisions and hashes match
`catalog-ee7e883.json` exactly, including TOOLS, fresh `_tool_definitions()` output
and TOOLS_BY_ID. The two hashes recorded above are unchanged in production.
Catalog HTTP tests preserve both available and unavailable tool output, and
missing defaulted wire inputs still receive the existing 422 response.

Static construction uses ordinary Torch-free imports of the two family policy
modules. A clean-process service import succeeds without configured model roots,
with filesystem APIs, artifact construction, Recipe binding and dotenv loading
blocked; normal Python module loading remains permitted. Neither Torch nor
family inference modules are imported. Runtime initialization and availability
checks still happen later through create_app(), not while defining the catalog.

The old test-only projection and presentation implementation were removed.
Tests now exercise the production helper and catalog directly against the
independent frozen oracle. Altering a policy in a test changes that tool's input
order/defaults/constraints while preserving service presentation, canvas, timing
and the other seven tools, demonstrating real production ownership rather than
coincidentally equal duplicated declarations.

This integration did not falsify the lifecycle seam or require changes to the
generic recipe layer, family policy, inference or HTTP behavior. It earns the
pattern for further bounded, oracle-checked migrations of compatible products;
it does not establish that every remaining recipe has the same input semantics.
At that stage, the remaining six tools had not been migrated.

## Six-video-product end-to-end production ownership

LTX I2V and Wan FLF first proved runtime binding from
`157f18a45cb9d9407020a434c8b3fbf8cb321af9`. Starting from
`56d6fb1aebb4c634377da9f383a88b8e71926337`, four independent native parity
gates extended that ownership to LTX T2V/FLF and Wan T2V/I2V. All six video
products now use the complete lifecycle; Klein remains outside it:

```text
CapabilitySet -> ProductPolicy
                    -> surface() -> static catalog semantics
                    -> bind(real configured artifacts) -> Recipe
                        -> family identity/request resolver
                        -> existing runtime/session
```

Each normal video builder binds its family-owned `LTX23_*_POLICY` or
`WAN2214B_*_POLICY`. Existing builder APIs, keys, field order, capability object
identities and caller defaults are preserved. Locked and tunable LTX T2V recipes
remain independent policy-depth examples over the same CapabilitySet.

LTX T2V and I2V bind the configured dev checkpoint, text checkpoint, upsampler,
single transformer adapter at strength 0.5 and device 0. Their operation-local
fixed-identity resolvers share the identical two-pass model-field selection and
native keyword mapping. FLF binds only distilled checkpoint, text checkpoint
and device 0; its fixed-identity resolver retains that distinct three-field
shape. These resolvers reject exposed model fields and need no invented caller
values. Native runtimes still construct before the first job. Each real job uses
its existing family request resolver, including I2V's image_path mapping and
FLF's ordered first_image_path/last_image_path mapping and 32-pixel lattice.

Wan binds at first use from the first real request. T2V uses its configured
T2V high/low checkpoints and adapters; I2V/FLF use their configured image-model
paths. T2V and I2V preserve native primary strengths `1.0000000000000002`, while
FLF preserves `1.0`. Each binds the existing operation-specific negative prompt,
text encoder and VAE. Secondary slots and all fixed turbo settings retain their
existing meaning. Every generation resolves only exposed keys, excluding the
service's internal frame_count; each resolver derives its own count. The first
call resolves once for construction and again for generation. Subsequent calls
retain the same session and bound product; operation switches destroy the old
session before constructing its replacement. No replacement API or cache was added.

Native-boundary tests were run against the direct constructors before production
edits and retained as parity oracles for each operation. LTX identities and native
requests compare exactly, including the single-adapter representation, strength,
device and empty multi-adapter tuple for T2V/I2V and FLF's smaller identity.
Wan's complete native recipe equals the old constructor for the native-default
request, including its identity tuple, adapter slots and
strengths, negative prompt and sampling settings. For other requests, the full
recipe equals that same constructor with only positive prompt, width, height and
frame_count replaced by the real request. Those values were already explicit
generate arguments before this integration and remain explicit on every call;
the session's initial request defaults cannot override later jobs. The now
explicit negative_prompt exactly equals the previously omitted session default.
Effective native requests, including ordered endpoint paths, compare exactly.

Tests exercise real service admission and uploaded asset resolution for every
accepted quarter-second duration from 1 through 5 seconds for all three Wan
operations: service and resolver both yield 17 through 81 native frames.
Prompt, source, geometry, duration and
seed changes preserve Wan model identity and session reuse. Conditioning reuse,
content-derived source reuse, endpoint swaps and operation switches retain their
previous behavior. LTX worker construction, reuse, shutdown, output path and
progress messages also retain their previous behavior. Native runtime/session
classes are replaced with bounded fakes in these wiring tests; this milestone
does not claim a new GPU inference or performance measurement.

All eight catalog dictionaries and schema hashes still match the untouched
frozen oracle. Static catalog construction still binds no artifacts or Recipes,
reads no configuration or model files, and imports no Torch. HTTP admission,
assets, availability, presentation, canvas/timing, worker/process lifecycle,
progress and artifact delivery remain service-owned. The generic `recipe.py`
and all inference implementations are unchanged. The two Klein operations
retain their existing catalog and runtime paths. Explicit video-input fallback
declarations were removed once all six consumers used the policy projection.

The four migrations did not falsify the existing product seam or require a
generic execution layer. LTX's pre-request boundary needed two more family-local
identity entry points; the different Wan strengths and negative prompts required
exact operation-specific bindings. A subsequent Klein experiment should test its
own nullable geometry, ordered references/adapters and shared T2I/two-image runtime
before assuming the video projection or lifecycle applies.

## Klein product and native-parity experiment

Starting from `876f15dd9206bb4be39b60bb6243694258809483`, Klein was tested
separately because its two public operations share one model/runtime while its
existing flexible two-image recipe permits ordered LoRAs and nullable geometry.
The service exposes neither of those controls. This experiment declared the
actual products and proved shadow parity before production integration. The
final integration is recorded below.

`KLEIN9B_T2I_CAPABILITIES` contains diffusion, text_encoder, vae, tokenizer,
loras, prompt, width, height and seed. It reuses the exact two-image objects for
the four artifacts, ordered LoRAs, prompt and seed: both native methods consume
the same identity, prompt conditioning and unsigned seed semantics. T2I owns
distinct width/height objects because its native contract does not accept None.
The non-null numeric domain remains alignment 16, minimum side 256, maximum
pixel budget 1,048,576 and maximum aspect ratio 4:1. Pixel/aspect and paired
geometry validation remain Klein-local.

The two service products are:

- `KLEIN9B_T2I_POLICY`: prompt, width, height, seed; defaults 768, 768, 0.
- `KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY`: prompt, image_1, image_2, width, height,
  seed; defaults 768, 768, 0 and non-null geometry.

Both fix the ordered LoRA capability to an empty tuple and defer only the four
configured artifacts. `klein9b_t2i_recipe` and
`klein9b_two_image_explicit_recipe` bind these products. The existing
`klein9b_two_image_recipe` retains its key and complete surface: exposed ordered
loras, prompt, image_1, image_2, nullable width/height defaulting to None, and
seed defaulting to zero. Both dimensions may be None or concrete; a mixed pair
is invalid. Its resolver preserves LoRA and reference order. Runtime helpers
still derive auto geometry from the first reference, and swapping references
can change that geometry without changing model identity.

### Earned nullability narrowing

Klein falsified one assumption in the generic policy layer: family nullability
cannot always be copied to the product surface. Previously an exposed optional
capability advertised and accepted None even with a concrete default. HTTP-only
rejection would leave that semantic surface inaccurate for downstream callers.
A focused failing test preceded the small `Field.nullable` / `exposed(nullable=)`
addition. None inherits the capability domain, False rejects None, and True is
legal only if the capability already permits it. This allows narrowing but never
widening. `surface()` reports the effective nullability and resolution enforces
it, including validation of defaults. Key presence and omission/default behavior
remain separate. Existing fields inherit their previous behavior unchanged.

### Native and shadow parity

Before family edits, tests captured the flexible recipe's complete surface,
identity, ordered request mapping and paired geometry behavior, plus the current
service's direct native calls. Real small model/tokenizer/config files support
the unchanged metadata-backed `Klein9BIdentity.from_paths`. Both proposed products
resolve to exactly the existing no-LoRA service identity: diffusion, text encoder,
VAE, tokenizer path and file identities, text encoder config, empty ordered LoRAs
and native recipe identity value all match. A tiny Klein-local identity mapping
is shared by the T2I and existing two-image resolvers. Neither resolver opens
reference images; image_1/image_2 map to first_image/second_image in order.

Tests run the unchanged service worker and a test-only resolver-fed path through
the real CPU Klein runtime methods with bounded transformer, VAE and encoding
fakes. Exact request values, identity, output geometry, progress delivery and
reuse results match. One runtime serves T2I -> two-image -> T2I; model and prompt
reuse survive operation changes. Content-identical reuploads reuse reference
slots, individual slot changes invalidate independently, and slots persist
through intervening T2I calls. The existing process-owner test retains one Klein
worker across those operation changes. No GPU/numerical parity claim is added.

Test-only projection overlays service labels, the "Describe the image" prompt
hint, geometry min/step and wire-required width/height/seed on the two policy
surfaces. Complete dictionaries equal the untouched eight-tool oracle, including:

- T2I: `sha256:2e94d609c2db43e883da19fb0c73faa1bef7f3459c916760079f7cedd212c6b3`
- Two-image: `sha256:d756bc62e593edd29f3c2c909f3c92fd22d10cb2fb44a2b51bdd93afdb605ed8`

Canvas, metadata, hashing, presentation and HTTP admission remain service-owned.
Clean-process policy import constructs no artifacts or identities, performs no
application file IO/configuration, imports no inference/Torch/diffusers/transformers
or service modules, and leaves the environment unchanged. Native identity
resolution legitimately inspects configured files later.

The two products compose after this narrow nullability correction without losing
the richer flexible recipe or splitting Klein's runtime. This supported the final
bounded integration below, preserving the shared worker and pre-request identity.


## Final production lifecycle: all eight tools

The final Klein integration starts from
`a5236e212a13c3cd0a85234385466a33564284dd`. All eight existing public Engine
products now follow:

```text
CapabilitySet
  -> ProductPolicy
       -> surface() -> static service semantic inputs
       -> bind(real configured state) -> Recipe
            -> family-native identity/request resolution -> existing runtime
```

ProductPolicy owns caller semantics, not UI presentation. The six video paths
remain unchanged. Klein uses its own small `_klein_policy_inputs` projection:
policy field order/types/defaults/roles plus service labels and prompt hint,
geometry min/step, and HTTP-required width/height/seed. It publishes no LoRAs,
nullable geometry, scalar geometry max or seed bounds. The duplicate test-only
Klein projection was removed; production itself is checked against the untouched
complete eight-tool catalog oracle and the exact hashes recorded above.

The configured Klein worker binds `klein9b_t2i_recipe` and
`klein9b_two_image_explicit_recipe` using the same diffusion, text encoder, VAE
and tokenizer paths before receiving jobs. `resolve_klein9b_fixed_identity`
requires all four artifact fields and LoRAs to be fixed, then returns the existing
metadata-backed native identity. It runs once per product binding at startup
(two resolutions total); exact identity inequality fails before runtime creation.
The check covers all artifact sizes/timestamps/paths, tokenizer files, text encoder
config, empty ordered LoRAs and native recipe identifier. The worker then retains
one identity object and one `Klein9BTwoImageRuntime` for both operations.

Per-job `resolve_klein9b_t2i_request` and `resolve_klein9b_two_image_request`
resolve caller values only, without reconstructing identity or inspecting model
metadata. Their native argument mappings are shared with the existing full
resolvers, whose standalone behavior is preserved. The two-image mapping retains
explicit first/second reference order. Asset IDs and upload validation are still
resolved by HTTP admission into local Paths before recipe resolution; output
paths and progress remain service/runtime plumbing.

Before worker edits, an eleven-job native CPU baseline passed with real small
artifact/tokenizer/config files and bounded transformer/VAE/encoding fakes. The
same production test now verifies exact identity and native arguments, one model
load, prompt reuse, reference reuse across intervening T2I calls, independent
reference-slot invalidation, true image swaps, output geometry, progress, and
shutdown. Guards reject identity construction and model-path stat/resolve/open
once jobs begin. A separate request-only check forbids all Path IO, and an unequal
identity test proves startup fails before receiving work. Clean-process service
import binds no Recipe, reads no application/config/artifact files, constructs no
native identity, imports no Klein inference/Torch/diffusers/transformers, and
preserves the existing allocator-environment behavior. These are wiring and
lifecycle proofs, not GPU inference or performance measurements.

The lifecycle differences remain intentional: LTX constructs operation identities
before requests; Wan binds operation sessions from the first real request; Klein
binds two products to one equal pre-request identity and shared runtime. The
richer non-service `klein9b_two_image_recipe` still exposes ordered LoRAs and
paired nullable auto geometry. Its caller-exposed LoRAs correctly prevent use of
the fixed pre-request identity helper. HTTP requiredness remains separate from
recipe defaults, and canvas, timing, presentation and availability remain
service-owned. Generic `recipe.py` and all inference implementations are unchanged
in this final integration. The eight-product milestone is complete.
