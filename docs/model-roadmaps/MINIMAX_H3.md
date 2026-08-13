# MiniMax H3 roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Surface | Authority |
| --- | --- |
| Weights/architecture/lineage/license | MiniMax publisher source [`fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf) and exact official snapshots |
| Saved topology/defaults | official Comfy H3 T2V/I2V/R2V templates at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| Node/dispatch schema | ComfyUI [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541), Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4), exact optimized headers |
| Acceptance/tier | Engine public-API synchronized A/V artifacts, lifecycle, memory, and creator review; current main has CPU/source proof only |

The publisher commit and README were independently resolved; the old ledger warning was stale.

## Product decision

H3 has separate local FL2VA and Ref2VA checkpoints. Engine current main owns only a direct dense-BF16 FL2VA-style T2VA/endpoint path plus a hardened CPU/source closure. It has no package recipe and no target-hardware output acceptance.

The official Comfy FL2VA low-bit graph is the practical priority. Dense BF16 work led only because the decisive Comfy artifact repository is gated and the four-file optimized closure has not been authenticated. That is an access/evidence blocker, not a dense-first product policy.

Do not schedule repeated RTX 5080 BF16 retries. Preserve one bounded capability result if useful and move dense output studies to high-memory hardware.

## Operation boundaries

| Line | Contract | Engine state |
| --- | --- | --- |
| FL2VA T2VA | prompt, no images, video plus stereo audio | direct BF16 tool only |
| FL2VA endpoints | first image, last image, or both ordered endpoints | direct BF16 tool only |
| Ref2VA | separate checkpoint and ordered image/video/audio references | absent; separate project |
| Context-IR / Regenerate-2K | hosted preprocessing and 2K regeneration | provider-only, not open local features |

Open H3-Base output and the full hosted H3 product are different capability claims.

## Dense BF16 source contract

Main pins official revision `42ed227ee7df40d41602854ae760620d6eb651fe` and a 61-file FL2VA allowlist totaling **144,051,143,011 bytes**. It selects the normal `transformer/` partition, not `transformer_ref/`, and preserves tokenizer/processor/text encoder, visual/audio VAEs, configs, and synchronized 24 fps / 32 kHz stereo semantics.

The historical Engine revision was compared with the current revision; direct-closure artifact paths, sizes, and LFS identities showed no observed weight/layout drift. This is CPU/source evidence, not a GPU or license-acceptance result.

Proof level: **Direct tool only / CPU-source contract hardened; output acceptance pending**.

## Official Comfy optimized graph

The pinned T2V graph selects:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`;
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- `minimax_h3_video_vae_fp16.safetensors`;
- `minimax_h3_audio_vae_fp32.safetensors`.

This proves topology and filenames only. Gated access prevented immutable four-file identities, headers, tensor maps, Kitchen geometry, and complete support closure from being authenticated. Therefore it is **Deferred practical research**, not catalogable/runnable support.

The initial open release executes full-attention inference. Sparse attention remains an upstream future claim and must not be invented from architecture notes.

## Implementation and acceptance

Next packet: authenticate the four selected files and support closure; hash/normalize T2V and endpoint graphs; verify ComfyUI schemas/output slots and A/V behavior; map pruned transformer and H3-specific Qwen low-bit layouts; prove Kitchen dispatch/fallback; write independent fixtures; implement operation-specific requests and lifecycle.

Use local diagnostics only after the practical closure is exact. Acceptance covers T2VA, first-only, last-only, first+last, 24 fps stereo A/V, meaningful warm jobs, operation switching, malformed resources, cancellation across encode/load/denoise/decode/mux, observed stream metadata, teardown, and creator review. Dense BF16 output is a high-memory reference campaign.

Next: optimized closure authentication; optimized T2VA; endpoint reuse; dense Vast reference; Ref2VA only after FL2VA value; wait for official sparse inference.

Stop on gated identity gaps, FL2VA/Ref2VA mixing, inferred sparse behavior, partial closure, fallback, false availability, or assumed A/V metadata.
