# Using Comfy as the inference reference

## Reference versions

The user's shared Comfy installation follows its normal upstream update branch.
Record the actual Comfy commit, dependency versions and launch flags in the
external diagnostics workspace for every new comparison. Keep that environment
fixed during a comparison, but do not leave the shared checkout detached or
permanently pinned afterward. Historical evidence retains its original pins;
an update requires a fresh baseline before claiming current parity or speed.

Historical versions and family results belong in
[`CANONICAL_PARITY_CERTIFICATION.md`](CANONICAL_PARITY_CERTIFICATION.md) and the
linked external campaigns, not as standing defaults in this general procedure.

Find canonical fixtures through the external diagnostics workspace linked in
`AGENTS.md`. Its pairing index identifies the operational API export, Engine
case, provenance and validation limits; its historical index locates older
fixtures. Do not assume a repo-local reference directory exists.

The selected fixture is the canonical operational workflow used for automated
inspection and parity runs. It must be exported from the working ComfyUI graph
with **File > Export (API)**, or Comfy's native `app.graphToPrompt().output`
serializer, after all benchmark values and model paths are set.
The export is a JSON object keyed by node ID, each with `class_type` and
resolved `inputs`; it is the exact form submitted to Comfy's queue. The pinned
upstream frontend workflow remains a semantic/editing reference; intentional
differences in the API fixture define the concrete benchmark case and should be
documented in the fixture README.

Do not silently edit the canonical fixture while debugging Engine. A deliberate
fixture change requires acknowledging that reference evidence may need to be
re-baselined.

All reference workflows and raw evidence are local-only and gitignored. Paths
in this document identify local diagnostic material, not tracked files available
in a clone. Custom-model hardening material must live outside the checkout and
must not enter source, tests, documentation, or commit messages. Only generic
implementation lessons and synthetic regression fixtures belong in the repo.

## Local Comfy reference harness

The installed and validated `comfy-local` MCP is the preferred control and
inspection interface for Comfy itself.

Use it to:

- load/inspect the canonical workflow;
- inspect individual workflow nodes and effective widget/input values;
- resolve node classes to their respective implementations/source paths;
- inspect schemas and runtime behavior relevant to the traced path;
- execute fixed reference runs;
- inspect execution results/history/output metadata;
- compare Comfy behavior against Engine without reconstructing the graph from
  memory.

`comfy-local`'s `list_workflow_slots` is intentionally limited to Comfy's
editable frontend format (`nodes[]` / `links[]`). Do not use that command as an
API-fixture validity check: inspect the API object's `class_type` and resolved
`inputs` directly, and use `run_workflow` for execution.

Do not use frontend-format workflow JSON (`nodes[]` / `links[]`) as an
operational fixture, and do not hand-reconstruct an API prompt. If an API export
is absent, resolve and export the official template using the procedure below;
ask the user only if that route fails or the intended case is ambiguous. It
must preserve the effective nodes, settings, links, model selections, samplers,
schedules, seeds, dimensions, conditioning, and outputs of the intended
reference case. Validate it with `comfy-local` and execute it on the pinned
baseline before using it as parity evidence.

### Resolving official templates

Discover templates with `comfy-local.search_templates` / `fetch_template`, or
the installed `comfyui_workflow_templates.iter_templates()` and `get_asset_path()`
APIs. The running frontend serves installed assets at `/templates/<filename>`;
`app/frontend_management.py` owns that mapping. Record the installed template
package version and retain the original JSON and hash externally.

Load the selected JSON in a dedicated Comfy browser tab. Set benchmark widgets
and model selections, then export through File > Export (API). Browser automation
may equivalently import `/scripts/app.js`, call `app.loadGraphData(workflow)`,
then save `(await app.graphToPrompt()).output`. This uses the installed frontend's
native serializer, including subgraph expansion; do not flatten links or infer
widget positions by hand. Preserve the configured frontend graph alongside its
API export in the external pairing record.

Check the resulting node classes, effective inputs and model selections, validate
against the running server, and execute successfully before declaring the fixture
operational. An export alone does not prove model availability or output parity.

Trace exposed controls through the serialized connections to their consumers.
In multi-pass graphs, record which sampler/noise node receives the user seed and
which seeds stay fixed. Selecting a node by its title, position or the first
matching class can leave the expensive pass unchanged and cached.

### ComfyUI Process Manager

Comfy processes are managed by a separate loopback-only Local Process Manager:

`http://127.0.0.1:47827`

Process definitions may change. Discover them live rather than storing their
current IDs.

For parity work:

1. `GET /health` on the Process Manager.
2. `GET /processes` and locate the process whose display name is exactly
   `Comfy C (PyTorch Baseline)`.
3. Use the ID returned by that live response for any process start/stop/restart
   action.
4. Poll `/processes` until the desired state is visible.
5. Once the baseline Comfy process is healthy, use `comfy-local` for workflow
   inspection/execution.

Do not bake the current process UUID, PID, or status into the repository.
Do not use `Comfy C`, `Comfy C (Sage)`, `Comfy D`, or another Comfy variant as a
performance-parity reference unless the user explicitly changes the benchmark.

As with the Engine stack manager, `POST /stack/reload` is a broad operation that
stops all managed processes first. Prefer bounded control of the discovered
baseline process when that is sufficient.

## What Comfy is for

Comfy is an executable source oracle for:

- exact model topology and forward behavior;
- checkpoint/component composition;
- prompt enhancement and conditioning;
- sampler and schedule behavior;
- quantized tensor execution;
- model weight residency;
- transfer ordering;
- block/layer prefetch;
- VAE/upscaler behavior;
- state lifetime across repeated execution.

Start from the actual canonical workflow fixture and trace only the code it
touches, checking the pinned upstream workflow/source whenever semantic intent is
unclear.

For important paths, produce both:

1. call order;
2. state ownership/lifetime.

A call trace without the lifetime trace is incomplete.

## Reconciliation procedure

The objective is equivalent cold/warm execution, RAM and VRAM behavior while
preserving output agreement and the supported request domain. Locate how the
exercised Comfy implementation differs before designing an Engine change.

1. **Map the executed path.** Maintain a compact correspondence in the external
   pairing record: Comfy node and runtime function, Engine equivalent, effective
   inputs, and state created/reused/released. Include initialization, effective
   allocator/backend selection, callers, model switching and cleanup outside
   visible nodes. Include that surrounding execution context in source-review
   packets; a model function alone may omit the owner of a critical lifetime.
   Reuse this map until the exercised path changes; an available source helper
   is not proof that Comfy called it.
2. **Measure coarse boundaries together.** Start with startup, conditioning,
   sampling/model phases, decode and save. Compare matching cold and warm state;
   distinguish worker startup and model preparation, and record OS file-cache
   conditions rather than equating process restart with cold storage. Capture
   stage wall time and RAM/VRAM entry, peak and exit. Retained memory belongs to
   its owner/lifetime; node peaks cannot be added or treated as node allocations.
   Use existing Comfy execution events for coarse node intervals before adding
   probes. Check where Engine progress callbacks fire: a sampling update after
   step one makes the preceding UI stage include that step. Labels alone do not
   define equivalent timing boundaries.
3. **Narrow the disagreement.** Descend only into stages that explain a material
   gap, or move sideways when they agree. Check incoming residency and the prior
   release boundary before blaming a stage's computation. Function returns can
   be allocation-release boundaries: moving identical math into a larger scope
   can retain large temporaries through the next expensive operation. Compare
   actual temporary lifetimes, not just tensor values or arithmetic. Within the
   divergent interval, distinguish preparation, transfers/casts/patches, compute
   and waits.
   Use temporary CPU/CUDA traces or counters as needed; host enqueue duration is
   not GPU completion time. Do not insert pervasive synchronization that changes
   the overlap or residency being diagnosed. Diagnostic timings remain separate
   from uninstrumented acceptance.
4. **Establish the port before editing.** Record the exercised Comfy source
   revision/function, observed ownership and dependency behavior, Engine's
   differing behavior, and the observation connecting that difference to cost.
   If causality remains uncertain, run a discriminating comparison rather than
   implement an optimization idea. Adapt the narrow implementation together with
   the lifetimes it needs; use AIMDO/Kitchen primitives where they own the work.
   If it cannot be separated coherently from broader Comfy machinery, identify
   that concrete dependency before deciding the next scope.
5. **Verify locally, then certify.** First repeat the affected boundary with
   equivalent inputs and entering state, preserving output equality/tolerances.
   Exercise a representative case that can falsify the change's main assumption
   before the full matrix: for residency, include a supported larger canvas.
   Then run the uninstrumented certification below. Short diagnostic runs locate
   differences; they do not replace final cold-plus-five-warm measurements.

Keep evidence and the next unanswered comparison in the existing external
pairing/campaign record. Do not build a new profiling framework or repeatedly
rerun complete videos when a smaller faithful boundary can answer the question.
Preserve required state lifetimes in isolated replay; matching tensor values
alone does not make a memory/performance microbenchmark equivalent.

For each resolved disparity, retain the smallest runnable proof in that external
record: exact versions and effective runtime settings, required setup, invocation,
expected observation and cleanup. Reuse existing runners and fixtures so another
agent can verify the finding without reconstructing the campaign. Keep private
reproducers external; durable repository regression coverage must be synthetic
and non-identifying.

## Certifying a model operation

Use a fresh execution of the exact canonical fixture on the matching pinned
Comfy process. Re-baseline after a relevant fixture, environment, or dependency
change rather than treating historical measurements as standing targets.

For timing and resources, run one cold request followed by five same-process
warm requests that change only the sampling seed. Verify that every request
reruns the required model, sampling, decode, audio, and artifact path. Compare
equivalent end-to-end boundaries and report the warm median; cold timing is
diagnostic unless the operation target says otherwise.

Before collecting that block, prove first/repeat reuse through the real Engine
service: check worker identity, model retention and expected conditioning-cache
behavior. A family-local runtime test does not prove service registration or
worker reuse; fix unintended restarts before measuring warm performance.

Gate the measurement block on its first request and first seed-only repeat.
Check the required sampling, upscale, decode and save nodes against
`execution_cached` and execution events; fail the runner if a required stage was
skipped. Seed-independent conditioning may remain cached. Preserve this check on
every measured request; no separate duplicate preflight block is needed.
Exclude an invalid block explicitly and rerun it after
correcting the fixture or runner; do not optimize Engine against it.

When reference timing variation is comparable to or larger than the suspected
gap, use short, closely paired checks under comparable idle/resource conditions
to establish whether the gap is reproducible before changing Engine. Report the
repeat distribution alongside the median and qualify unresolved variability;
an unusually slow reference median does not establish a general speed advantage.
These diagnostic checks do not replace the final cold-plus-five-warm block.

Use equivalent telemetry:

- process working set for regular memory;
- WDDM-compatible total-device GPU usage on Windows;
- idle, absolute peak, and incremental use when idle residency differs.

Absolute peaks are the primary resource signal. Do not compare PyTorch allocator
statistics with total-device usage or invent a universal RAM/VRAM tolerance.

Compare correctness from the earliest deterministic seam through conditioning,
model inputs/outputs, stage/final latents, and raw decoded image, video, or audio.
At a divergent boundary, compare execution context as well as equations and
weights: dtype, device, operation order and effective backend/reduction settings.
CPU/GPU placement of even a final rescale can change results. Match the exercised
reference setting where needed and restore any process-global setting afterward.
Validate encoded artifacts and media metadata separately. Use exact equality
where expected and measures such as MAE, RMSE, PSNR, cosine similarity, or SNR
where variance is legitimate. Thresholds are operation-specific: localize and
explain residuals, and measure same-seed Comfy self-variance when nondeterminism
is suspected. Visual inspection is supplemental.

If encoded pixels differ, compare the raw decoder output before changing model
math; codec settings can explain the difference. For a small residual after a
matching boundary, replay that boundary's identical input in both implementations
and measure reference self-variance there rather than regenerating whole videos.

Exercise the lifecycle cases relevant to the operation, including:

- seed-only reruns with appropriate warm-state retention;
- same-prompt reuse and prompt-change invalidation where conditioning is cached;
- content-input identity, reuse, and invalidation for images or other media;
- ordered semantic roles when multiple references are consumed;
- destructive invalidation on a true model, recipe, or LoRA identity change;
- non-cumulative LoRA application where applicable.

For mutable model state, include a return-to-baseline sequence: base, adapter,
zero/absent adapter, then base again; likewise alternate model then original.
The restored configuration must reproduce its original output within the
established reference tolerance. This checks stale caches and cumulative patches
more directly than independent successful generations.

Choose compatibility cases by actual tensor keys, quantization metadata, logical
shapes and sidecars, not filenames alone. For adapters, verify that the intended
weights were matched and patched and that a nonzero test exercises their effect;
a successful generation can silently be the base model. Use the reference's
exercised name mapping and Kitchen layout/requantization path before adding a
new loader or quantization implementation. Keep specimen details external.

Record inapplicable cases explicitly. Concrete thresholds, accepted residuals,
and resource judgments belong in the operation or family target document.

## High-value Comfy source sites

Framework integration:

- `comfy/model_patcher.py`
- `comfy/ops.py`
- `comfy/model_prefetch.py`
- `comfy/model_management.py`
- `comfy/memory_management.py`
- `comfy/pinned_memory.py`
- relevant SafeTensors/ModelMMAP loading in `comfy/utils.py`

LTX:

- `comfy/ldm/lightricks/av_model.py`
- `comfy/text_encoders/llama.py`
- `comfy/text_encoders/lt.py`
- `comfy_extras/nodes_textgen.py`
- `comfy_extras/nodes_lt_upsampler.py`
- relevant `nodes_lt*` audio/video/latent/conditioning nodes

`execution.py` and workflow/node code are useful for discovering the effective
workflow path. They are not runtime architecture to port.

## comfy-aimdo

Study and use the package itself, not an Engine approximation of it.

Important primitives include:

- `control`
- `ModelMMAP`
- `HostBuffer`
- `ModelVBAR`
- `VRAMBuffer`
- file-slice/direct transfer support
- native VBAR signatures
- native fault/unpin behavior

AIMDO owns native physical residency and pressure behavior.

Application code should normally express model execution order and safe stream
dependencies, not recreate AIMDO's VRAM budgeting/watermark system.

## comfy-kitchen

Treat `QuantizedTensor` as the owner of its logical quantized representation.

Use its supported:

- tensor flatten/unflatten protocol;
- qdata and sidecar movement;
- device casting;
- reconstruction;
- native quantized dispatch;
- fallback behavior;
- reusable fused operations.

Do not teach an Engine memory layer the internal physical structure of every
Kitchen quantization layout unless an upstream API genuinely requires it.

## What not to port

Do not port or depend on:

- Comfy graph execution
- global model-manager policy
- node/plugin runtime
- UI behavior
- global caching policy
- arbitrary multi-model coexistence machinery

LatentSlate Engine should reproduce the required inference behavior with a
smaller product-specific runtime.

## Source adaptation

This project is GPLv3.

Narrow, attributed adaptation of compatible pinned Comfy/AIMDO source is allowed
when it is the shortest path to faithful behavior.

Prefer removing unrelated Comfy dependencies from a small proven source seam
over re-expressing that seam as a new generalized Engine architecture.
