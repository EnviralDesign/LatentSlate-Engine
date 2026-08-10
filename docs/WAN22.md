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

## Stored-quant adapter boundary

The stored FP8/INT8 Wan adapter is currently a CPU/meta validation and
per-layer primitive only; it is not a generation runtime yet. Its future
integration is constrained to Engine-owned `offload = "group_block"` at block
granularity. The Engine synchronously calls `module.to(onload_device)` before a
managed block and `module.to(offload_device)` after it; it never calls
Diffusers/Accelerate group hooks. Sequential CPU offload/meta reconstruction,
leaf-level group offload, whole-model offload, and disk offload are rejected
because Accelerate cannot safely reconstruct the third-party `QuantizedTensor`
parameter type from meta storage. A tiny CUDA residency proof has passed for the
Engine-owned block primitive; full-model generation, output quality, and memory
residency remain unproven, so stored-quant execution is still unavailable.

The staged UMT5 XXL legacy-FP8 and ConvRot-INT8 text encoders also have an
exact CPU-only `UMT5EncoderModel` materialization plan. It restores stored
linear layouts and the tied `shared.weight` alias without conversion, but does
not yet own prompt orchestration. A future conditioning path must make its
sequence policy explicit; Comfy-first staging uses a padded/masked 512-token
sequence rather than silently inheriting Diffusers' default length.

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

The current CPU-only adapter can validate and materialize complete stored current
FP8, legacy scaled-FP8, and ConvRot INT8 transformer artifacts into a Diffusers
meta skeleton without converting weights. It binds the exact artifact/config and
stored dense-role contract: the proven current Comfy FP8 layout keeps only
`patch_embedding` weight and bias in F32, keeps remaining dense state in F16,
and transiently returns F16 activations; legacy FP8 and ConvRot currently require
uniform F16 dense state. It is not yet a full Wan pipeline or generation path.

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

## 14B I2V component recipe inspection

The Engine can now inspect stored SafeTensors and GGUF headers without loading a
tensor. The CPU-only probe records container format, Comfy-native key-prefix and
marker signals, and an explicit stored quantization contract when the artifact
header proves one. It does not infer a loader, convert a weight, or make a
quantized artifact executable.

Wan transformer recognition is deliberately strict: the header must expose the
40-block, 36-channel, modulation/head topology of the staged 14B I2V artifacts.
UMT5 XXL requires 24 encoder blocks, T5 markers, and its embedded SentencePiece
payload. The standalone VAE requires the Wan 2.1 decoder/encoder shape signature;
an arbitrary `decoder.*` or `encoder.block.*` drop is not accepted by role name
alone.

A Wan 2.2 14B I2V recipe is explicitly composed from a high-noise transformer,
low-noise transformer, UMT5 text encoder, and a standalone Wan 2.1 VAE. The
high/low pair must share one declared stored format, quantization contract, and
header architecture signature. Text-encoder compatibility is role-specific:
an FP8 or INT8 UMT5 may pair with GGUF, FP8, or INT8 transformers when its own
stored contract is proven. The VAE is likewise a separate role, not an embedded
Diffusers directory requirement.

Tokenizer, scheduler, configuration, and other orchestration files may be
recorded as optional support metadata. They are not required to validate the
four stored Comfy-oriented component artifacts. This matches the embedded
SentencePiece payload in Comfy text-encoder drops and the standalone VAE used by
the official workflow.

The result can be serialized as a versioned, executor-neutral runtime request
manifest containing only the validated local resource IDs and paths. The Engine
now has CPU-tested transformer materialization and Engine-owned root/block
residency primitives; full 14B loading, pipeline orchestration, and generation
remain unavailable. Future native work can add component composition, LoRA
patching, and the high/low switch. ComfyUI remains a behavioral reference, not a
bundled backend. No Hugging Face or Comfy cache/model universe is introduced, and
conversion or save-quantized nodes are out of scope.

Artifact probes validate container offsets and file bounds without reading tensor
payloads, then record resolved path, size, modification time, and a header/table
digest in the runtime request. A future executor must revalidate that identity
immediately before opening each artifact and keep the opened handle; this narrows
but cannot eliminate filesystem TOCTOU risk on a user-writable model root.
