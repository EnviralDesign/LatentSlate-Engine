# Using Comfy as the inference reference

## Pins

ComfyUI:

- repository: `Comfy-Org/ComfyUI`
- commit: `12d5279438bfefc058a269eae805ceab6047777f`
- version: v0.34.0

Low-level dependencies:

- comfy-aimdo 0.4.15
- comfy-kitchen 0.2.31

Official upstream LTX 2.3 workflows of interest:

- `blueprints/Text to Video (LTX-2.3).json`
- `blueprints/Image to Video (LTX-2.3).json`
- `blueprints/First-Last-Frame to Video (LTX-2.3).json`

Canonical Engine parity fixtures live under:

`reference/comfy/ltx23/`

For the first T2V milestone, use exactly:

`reference/comfy/ltx23/t2v-pytorch-baseline.json`

The repo fixture is the canonical operational workflow used for automated
inspection and parity runs. It is a flattened, user-prepared derivative of the
pinned official workflow so local MCP tooling does not need to reproduce Comfy
subgraph behavior. The pinned upstream workflow remains semantic/source
reference; intentional differences in the repo fixture define the concrete
benchmark case and should be documented in the fixture README.

Do not silently edit the canonical fixture while debugging Engine. A deliberate
fixture change requires acknowledging that reference evidence may need to be
re-baselined.

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

If `comfy-local` cannot execute a future canonical workflow because of subgraph
handling, a separate flattened derivative may be created for MCP automation. It
must preserve the effective nodes, settings, links, model selections, samplers,
schedules, seeds, dimensions, conditioning, and outputs of the intended
reference case. Validate such a derivative in normal Comfy before using it as
parity evidence.

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
