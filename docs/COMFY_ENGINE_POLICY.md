# Comfy authority and Engine execution policy

This is a hard architectural boundary for LatentSlate Engine.

## Goals

- Use pinned official ComfyUI workflows as the primary practical reference for
  operation topology, active branches, component roles, preprocessing,
  conditioning, sampling, frame/FPS behavior, and output handling.
- Read pinned ComfyUI node implementations and schemas to understand how modern
  optimized artifacts are wired and how their fast paths use Comfy Kitchen.
- Use **Comfy Kitchen directly inside Engine-owned workers** for supported
  quantized tensor layouts, restoration, kernels, and dispatch.
- Implement the effective operation as a typed Engine runtime with Engine-owned
  lifecycle, cancellation, provenance, storage, and public API.

## Non-goals

- Never embed, import, launch, proxy, or require ComfyUI as an Engine execution
  backend.
- Never submit workflow JSON to a ComfyUI server from an Engine recipe.
- Never make a ComfyUI checkout, custom-node installation, HTTP endpoint, or
  workspace part of recipe availability or execution.
- Never interpret "Comfy-first" or "Comfy-aligned" as running ComfyUI.
- Never copy license-incompatible implementation code.

## Authority hierarchy

1. First-party repositories own weights, architecture, tensor naming, and
   high-precision reference behavior.
2. Pinned official Comfy workflows own practical saved-operation topology and
   defaults when they are the decisive creator path.
3. Pinned ComfyUI node source and schemas explain wiring and preprocessing that
   Engine reproduces cleanly.
4. Pinned Comfy Kitchen APIs own supported quantized representation, layout,
   restoration, kernel, dispatch, and fallback facts.
5. Engine source, typed recipes, and tests own implementation.
6. Engine public-API runs on target hardware own acceptance claims.

A workflow proves topology, not native Engine dispatch. A Kitchen capability
proves a primitive exists, not that an Engine family uses it correctly. A
successful upstream ComfyUI run is reference evidence, not Engine acceptance.

## Required optimized-runtime evidence

- immutable model, workflow, node-source, and Kitchen pins;
- a normalized operation contract derived from the pinned workflow;
- exact artifact/header/schema and source-to-target mapping validation;
- direct Engine materialization without dense duplicate residency;
- observed Kitchen/native dispatch counters, not planned-layer counts;
- Engine-owned cancellation, teardown, cleanup, and recovery;
- public provenance binding artifacts, operation, conditioning, schedule, and
  actual backend dispatch;
- target-hardware output acceptance before promotion.

Klein is the golden implementation pattern for this boundary. Existing families
must converge on that pattern rather than introducing a graph server or plugin
host.
