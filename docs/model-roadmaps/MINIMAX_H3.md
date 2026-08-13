# MiniMax H3 roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | MiniMax source [`fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf) and exact snapshots |
| saved topology/defaults | H3 T2V/I2V/R2V workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for future direct Engine dispatch |
| acceptance/tier | Engine synchronized A/V public-API evidence; current main has CPU/source proof only |

## Decision

FL2VA and Ref2VA are separate checkpoints. Current main owns a direct dense-BF16
FL2VA-style tool and hardened CPU/source closure, but no package recipe or accepted
target output.

The optimized four-file FL2VA contract is the practical priority. Dense work led only
because gated access prevented exact optimized identities and headers; this is an
evidence blocker, not a dense-first policy.

## Dense source contract

Revision `42ed227ee7df40d41602854ae760620d6eb651fe`, 61 files,
**144,051,143,011 bytes**, normal `transformer/` rather than `transformer_ref/`,
with tokenizer/processor, Qwen encoder, visual/audio VAEs, configs, 24 fps, and
32 kHz stereo.

Proof: **Direct tool only / CPU-source hardened; output acceptance pending**.
Do not repeat local dense fitting; use high-memory reference hardware.

## Optimized source contract

The T2V workflow selects:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`;
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- `minimax_h3_video_vae_fp16.safetensors`;
- `minimax_h3_audio_vae_fp32.safetensors`.

This is topology/filename evidence only. Gated identities, support, headers, maps, and
Kitchen geometry remain unresolved, so the path is not catalogable or runnable.

## Required implementation

Authenticate the closure; normalize T2VA and endpoint contracts; verify node semantics
from source; map H3-specific layouts; call Kitchen directly; implement Engine-owned
typed requests, workers, A/V lifecycle, cancellation, mux, output, and provenance.

No external-UI execution or graph execution is permitted.

Next: optimized closure; Engine-native T2VA; endpoint modes; dense Vast Reference;
Ref2VA only after FL2VA value; wait for published sparse inference.

Stop on ComfyUI dependency, gate gaps, FL2VA/Ref2VA mixing, inferred sparse behavior,
partial closure, fallback, false availability, or assumed A/V metadata.
