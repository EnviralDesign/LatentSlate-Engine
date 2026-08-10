# Wan 2.2 TI2V-5B recovery runtime

This adapter keeps the curated `wan22.text_to_video` tool and its default
`1280x704`, five-second schema unchanged. The default remains the reference
BF16/sequential-offload recipe. The recovery implementation changes model
lifecycle, not output semantics or quality settings.

## Why the runtime is staged

The official Wan 2.2 TI2V-5B recipe expects at least 24 GB of VRAM and runs the
T5 encoder on CPU with model offload. On a 16 GB RTX 5080 / 64 GB RAM host, the
earlier monolithic Diffusers load reached approximately 15.9/16.3 GB VRAM and a
39.9 GB Engine working set before CUDA OOM.

Wan prompt conditioning is now computed in a short-lived CPU subprocess:

1. Check the bounded prompt-conditioning cache.
2. On a miss, unload any hot Wan generation pipeline.
3. Load UMT5 on CPU in the child process and encode positive/negative prompts.
4. Apply the same prompt normalization used by the pinned Diffusers Wan pipeline.
5. Write only the small embeddings to a temporary safetensors file.
6. Exit the child process so the operating system reclaims text-encoder memory.
7. Load the transformer, VAE, and scheduler in the Engine process.
8. Generate using precomputed embeddings; no text encoder lives in the pipeline.

A repeated prompt can keep the generation pipeline hot because the subprocess
stage is skipped. A new prompt deliberately trades a pipeline reload for a
bounded host-memory peak.

## Exact first-tranche capability matrix

| Kit part | Implemented |
| --- | --- |
| Model | Complete Wan 2.2 Diffusers directory |
| Prompt stage | Isolated CPU UMT5 subprocess, bounded prompt cache |
| Attention | `native` only |
| Conservative BF16 | `sequential` offload |
| Optional BF16 offload experiment | `group_leaf`, streams disabled |
| Pre-quantized artifacts | Not yet supported by a proven Wan loader; unavailable |
| VAE | FP32, tiling must remain `on`; slicing remains `off` |
| Cache | `none` or `prompt` |
| Residency | `keep_pipeline_loaded = true/false` |
| LoRA | Not implemented in this tranche |
| Compile | Not implemented in this tranche |
| GGUF / FP8 / NVFP4 | Not implemented in this tranche |
| Group-offload streams | Disabled; no prefetch or `record_stream` |

The adapter rejects cross-products that have not been reviewed. In particular,
BF16 plus model offload as a 16 GB recovery
variant, streamed group offload, VAE tiling off, compile, LoRA, and alternate
attention backends remain unavailable.

## Suggested variants for the first local matrix

Use one-second duration for the first allocation probe. Do not begin with the
five-second curated default.

### 1. Staged BF16 sequential baseline

This is the first test because sequential offload remains the most conservative
existing weight-residency policy. It isolates whether removing the live UMT5
encoder from the parent process is sufficient to recover the previous failure.

```toml
schema_version = 1
key = "wan22.recovery.bf16_sequential"
name = "Wan 2.2 Recovery — BF16 Sequential"
family = "wan22"
base_tool = "wan22.text_to_video"

[fixed]
duration_seconds = 1.0
size = "1280x704"

[inputs.prompt]
[inputs.seed]

[optimizations]
attention = "native"
quantization = "bf16"
offload = "sequential"
vae_tiling = "on"
cache = "prompt"
keep_pipeline_loaded = false
```

### 2. Optional BF16 group-leaf throughput experiment

Do not assume this uses less VRAM than sequential offload. Test it only after the
staged sequential baseline establishes the new memory floor.

```toml
schema_version = 1
key = "wan22.recovery.bf16_group_leaf"
name = "Wan 2.2 Recovery — BF16 Group Leaf"
family = "wan22"
base_tool = "wan22.text_to_video"

[fixed]
duration_seconds = 1.0
size = "1280x704"

[inputs.prompt]
[inputs.seed]

[optimizations]
attention = "native"
quantization = "bf16"
offload = "group_leaf"
vae_tiling = "on"
cache = "prompt"
group_offload_use_stream = false
group_offload_record_stream = false
keep_pipeline_loaded = false
```

## Expected memory risks

Staged BF16 removes the UMT5/transformer overlap in the parent process, but 720p
video activations and FP32 VAE decode can still approach the 16 GB VRAM ceiling.
Sequential offload was already the most memory-conservative built-in Diffusers
policy, so staging improves host lifecycle without guaranteeing that the original
activation peak disappears.

The current Wan adapter supports only complete native BF16 Diffusers artifacts.
It does not convert a dense repository to another precision. A pre-quantized Wan
drop will stay unavailable until its stored format and loader are explicitly added.

## Local scale-up order

1. BF16/sequential, 1 second, `keep_pipeline_loaded=false`.
2. Verify MP4, provenance, and full RAM/VRAM recovery.
3. Repeat BF16/sequential with `keep_pipeline_loaded=true`, same prompt/new seed.
4. Verify prompt hit and warm pipeline.
5. Change the prompt and verify the pipeline unloads before isolated UMT5 starts.
6. Optionally compare BF16/group-leaf after the sequential memory floor is known.
7. Only after a clean result, increase duration to 2, 3, then 5 seconds.
8. Test portrait separately after landscape is stable.

Record Engine working set, system available RAM, dedicated VRAM, job state,
runtime status, pipeline fingerprint, prompt-stage metadata, artifact validity,
and teardown recovery for every run.
