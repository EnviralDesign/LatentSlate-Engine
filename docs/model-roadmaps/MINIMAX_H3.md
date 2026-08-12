# MiniMax H3 implementation roadmap

Last audited: **2026-08-12**  
Engine source audited: [`b2481702d7b888a8553a4ce8b3302258a7a1fd96`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b2481702d7b888a8553a4ce8b3302258a7a1fd96)  
Official H3 source audited: [`fa6891f`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf)  
Official Comfy evidence: [workflow templates `2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1), [ComfyUI `725e6ecf9f11561da664cae996e0ab27ed7bfc6c`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ecf9f11561da664cae996e0ab27ed7bfc6c)

## Decision and next slice

H3 has two distinct local checkpoint families:

- **FL2VA**: zero/one/two endpoint images for text-to-audio-video and first/last-frame conditioning.
- **Ref2VA**: separate omni-reference checkpoint for images, video, and audio references.

Engine already implements complete-repository BF16 FL2VA-style T2VA and first+optional-last audio-video. The next slice is **current BF16 closure reconciliation and target-hardware acceptance**, not immediate quantization. The official current Comfy path provides a plausible stored INT8/NVFP4 FL2VA closure, but its exact identities and family tensor mapping must be pinned/proven after BF16. Ref2VA is a separate large project. Hosted Context-IR and Regenerate-2K remain provider services, not local-model features.

## Product/operation boundary

| Operation | Exact input contract | Engine state | Disposition |
| --- | --- | --- | --- |
| T2VA | prompt, no images | implemented BF16 | first acceptance |
| first-frame FL2VA | prompt + one first image | implemented | separate endpoint corpus |
| last-frame FL2VA | prompt + one last image | official FL2VA supports endpoint semantics; Engine one-image public path currently means first frame | schema/runtime extension after base acceptance |
| first+last FL2VA | prompt + ordered first/last | implemented | acceptance after T2VA |
| Ref2VA | text + up to 9 images, 3 videos, 3 audio clips, or 12 mixed files within publisher limits | absent; different transformer | separate Deferred family slice |
| Context-IR / Regenerate-2K | hosted preprocessing and 2K regeneration | absent | generic provider; preserve privacy/cost provenance |

Pinned workflows:

- [T2V/T2VA](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_t2v.json)
- [I2V/endpoint](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_i2v.json)
- [reference-to-video](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_r2v.json)

Current T2V template uses a high-level H3 subgraph, defaults to 1344×768 and five seconds, produces video plus native stereo audio, and saves through Comfy's video output node. Template notes: 24 fps; frame length snaps to `17k+5`; dimensions are multiples of 32; native canvas short edge 768 with cap 768×1344.

## Official/default Comfy closure

Current FL2VA Comfy T2V closure:

| Role | Exact workflow filename | Identity status |
| --- | --- | --- |
| Transformer | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | official Comfy stored INT8 ConvRot; exact immutable repository/revision/bytes/SHA must be resolved before declaration |
| Text/vision encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | H3-specific Qwen3-VL-32B encoder; exact immutable identity required |
| Visual VAE | `minimax_h3_video_vae_fp16.safetensors` | exact identity required |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | exact identity required |
| Support | tokenizer/processor/config/scheduler/vocoder/mux semantics selected by H3 source/workflow | exact allow-pattern closure required |

Unknown bytes are a publication stop. Do not treat similarly named Qwen3-VL/NVFP4/AWQ files as substitutes. The complete first-party BF16 `MiniMaxAI/MiniMax-H3` repository is the Reference. Engine's current compatibility closure pins HF revision `9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6` and excludes `transformer_ref/**`, so it is FL2VA-only; reconcile it with current official source before promotion.

Architecture facts that affect loading: Qwen3-VL-32B hidden states, 33B dense single-stream H3 transformer, visual VAE `f16t4d24` plus patchification, independent stereo audio VAE at 32 kHz, and full-attention inference in the initial open release. Sparse attention is a future upstream feature; never claim it from architecture alone.

## Recipe ladder

| Key | Tier | Fixed contract |
| --- | --- | --- |
| `minimax-h3-fl2va.text-to-video.native-bf16` | Reference/Experimental | complete exact FL2VA BF16 repository; T2VA; model offload; full attention; synchronized audio; no cache/LoRA/quant conversion |
| `minimax-h3-fl2va.first-last-frame-to-video.native-bf16` | Reference/Experimental | same closure; required first + optional last; explicit endpoint roles |
| `minimax-h3-fl2va.text-to-video.comfy-int8` | Experimental future | exact INT8 transformer + H3 Qwen NVFP4/AWQ + video/audio VAEs + support; native dispatch required |
| `minimax-h3-ref2va.reference-to-video.native-bf16` | Deferred | separate Ref2VA transformer and ordered multimodal list/limits |

Current complete BF16 path does not need a new component recipe, but package-owned resource/recipe declarations are absent. The Comfy low-bit path requires a typed H3 component contract with `transformer`, `text_vision_encoder`, `video_vae`, `audio_vae`, `support`; Ref2VA additionally needs an ordered multimodal request schema. These are schema extensions.

## Loader/runtime packet at audited Engine commit

Current `runtime/h3.py` is the authoritative Engine truth:

- 24 fps; frame grid `17k+5`; min 124, max 345; default 960×544; alignment 32; max pixel area 1,032,192;
- 1–30 steps, default 20; workflows `t2va` and `fl2va`;
- complete BF16 repository contract; native attention; ComponentsManager automatic CPU offload; no VAE tiling/slicing/cache/LoRA;
- synchronized video/audio output and exact workflow in pipeline fingerprint;
- runtime lock, hook removal, GC/CUDA cleanup on unload.

Source: [H3 runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/h3.py) and [repository contract](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/diffusers_repository.py).

Before low-bit work, compare current Engine snapshot/config/tensor schema to current official FL2VA and prove no topology/weight drift. Package resource acquisition must pin exact complete closure and license/gate. For Comfy low-bit later, map pruned transformer tensors/dense exceptions/sidecars, H3-specific Qwen AWQ/NVFP4 geometry and aliases, both VAE contracts, and actual Kitchen/native kernels. Fail closed on full-attention/sparse mismatch, eager/dequant fallback, or Ref2VA/FL2VA mix.

Lifecycle: validate media before model allocation → processor/Qwen encode → stage transformer/full attention → visual/audio latent generation → release transformer → visual/audio decode → synchronized mux. Cancellation during third-party generation may be cooperative only between phases; uncertain state must be ejected. Provenance records workflow (`t2va`/`fl2va`), endpoint roles, full-attention backend, every artifact, fps/frames/audio rate, and hosted Context-IR use if any.

## Hardware/scientific acceptance packet

Parity request: 1344×768, five seconds, 24 fps, exact aligned `17k+5` frame count, seed `43301611940728`, fixed 20-step prompt. The local BF16 path may not fit; run a clearly labeled 960×544/124-frame diagnostic on the RTX 5080, while preserving full parity for larger/Vast hardware.

Scenarios: T2VA cold/warm, T2VA→FL2VA→T2VA, first-only and first+last, cancellation during repository load/processor/denoise/video decode/audio decode/mux, malformed closure, endpoint order, explicit teardown. Assertions: exact snapshot/config, workflow, endpoint indices, full-attention backend, offload/residency, frame grid/fps, audio 32 kHz stereo, A/V duration/drift, output hash. Publisher four-GPU serving results are not single-5080 evidence.

Corpus: dialogue/singing/music/ambience/foley/impacts/silence/channel placement; lip/action/sound sync; identity; endpoint composition; camera motion; temporal coherence. Context-IR-expanded and raw prompts are separate corpora.

## Ordered bounded slices

1. **Next — BF16 closure reconciliation/package resource.** Compare Engine revision `9ac0dd7...` with current official FL2VA configs/tensors; inventory exact files/bytes/hashes/license. Stop on unresolved gate or topology mismatch.
2. **BF16 T2VA target/cloud acceptance.** Diagnostic local + parity cloud; synchronized A/V, cold/warm/cancel/recovery/provenance.
3. **BF16 endpoint acceptance.** First-only and first+last; add last-only request semantics only if source proves exact mapping.
4. **Comfy INT8 FL2VA loader (Sol escalation).** Exact five-role closure, complex H3/Qwen mapping, native dispatch proof. Out of scope: Ref2VA/sparse attention/hosted services. Stop on any unknown layout/fallback.
5. **Ref2VA separate project.** Ordered multimodal schema, limits, separate transformer/closure/corpus; only after FL2VA earns value.
6. **Wait for official sparse attention.** Do not invent it.

## Primary sources

- [MiniMax H3 source `fa6891f`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf)
- [MiniMax H3 weights](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Current official H3 T2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/video_minimax_h3_t2v.json)
- [Engine audited H3 runtime](https://github.com/EnviralDesign/LatentSlate-Engine/blob/b2481702d7b888a8553a4ce8b3302258a7a1fd96/src/latentslate_engine/runtime/h3.py)
