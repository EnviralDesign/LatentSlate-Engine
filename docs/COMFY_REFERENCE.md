# Using Comfy as the inference reference

## Pins

ComfyUI:

- repository: `Comfy-Org/ComfyUI`
- commit: `12d5279438bfefc058a269eae805ceab6047777f`
- version: v0.34.0

Low-level dependencies:

- comfy-aimdo 0.4.15
- comfy-kitchen 0.2.31

Official LTX 2.3 workflows of interest:

- `blueprints/Text to Video (LTX-2.3).json`
- `blueprints/Image to Video (LTX-2.3).json`
- `blueprints/First-Last-Frame to Video (LTX-2.3).json`

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

Start from the actual pinned workflow and trace only the code it touches.

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

`execution.py` and blueprint/node code are useful for discovering the effective
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
