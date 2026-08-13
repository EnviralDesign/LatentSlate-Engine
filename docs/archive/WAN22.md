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

## Legacy TI2V-5B capability matrix

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

## Native stored-quant I2V boundary

The stored FP8/INT8 Wan adapter now forms a native 14B I2V generation runtime.
Its execution is constrained to Engine-owned `offload = "group_block"` at
block granularity. The Engine synchronously calls `module.to(onload_device)`
before a managed block and `module.to(offload_device)` after it; it never calls
Diffusers/Accelerate group hooks. Sequential CPU offload/meta reconstruction,
leaf-level group offload, whole-model offload, and disk offload are rejected
because Accelerate cannot safely reconstruct the third-party `QuantizedTensor`
parameter type from meta storage. Real local four-step generation has passed
with UMT5, first-frame VAE conditioning, high/low 14B denoising, and VAE decode
while preserving CPU cleanup and bounded block residency. Normal catalog/tool
exposure is provided by data-defined `wan22_i2v_14b` recipe variants; the hidden
native base is not advertised without a complete validated five-role recipe.
The Engine synchronizes the active CUDA device once at each transformer-stage
teardown before reconstructing stored tensor parameters on CPU; this prevents
asynchronous kernels from retaining storage across the high/low boundary.

The staged UMT5 XXL legacy-FP8 and ConvRot-INT8 text encoders use an exact
`UMT5EncoderModel` materialization plan. It restores stored linear layouts and
the tied `shared.weight` alias without conversion. Engine-owned prompt
orchestration uses an explicit padded/masked 512-token Comfy-first sequence
rather than silently inheriting Diffusers' default length.

The Comfy-first prompt boundary uses the staged raw SentencePiece model
directly (no BOS, EOS appended, pad ID 0) and rejects prompts over 512 tokens
instead of silently truncating them. Positive and negative prompts are encoded
together by the stored UMT5 residency session and split into `[1,512,4096]`
conditioning tensors after the encoder has returned to CPU.

The standalone Wan 2.1 BF16 VAE has an exact CPU/meta `AutoencoderKLWan`
adapter: 16-channel latents, temporal ratio 4, spatial ratio 8, and the
artifact's Diffusers mean/std normalization. Its whole-component job
residency is Engine-owned and returns to CPU synchronously. The adapter only
wraps the pinned Diffusers public tiling/slicing controls and participates in
the native conditioning and final decode stages.

High/low denoising is Engine-owned rather than delegated to the Diffusers
pipeline call so only one 14B transformer can be resident. Stage policy is
explicit: `expert_split` uses a contiguous half-and-half split (10/10 at 20
steps, 2/2 at 4), while `diffusers_boundary` follows scheduler timesteps and
the configured training-timestep boundary (the official 0.9 boundary is 6/14
with the pinned 20-step scheduler). Each stage has independent guidance, and
CFG 1 skips the unconditional forward.

I2V conditioning follows the pinned/Comfy 16-channel Wan 2.1 latent path:
the first RGB frame is VAE-encoded under its own residency session, a
four-channel causal first-frame mask is added, and the immutable 20-channel
condition is concatenated with 16 FP32 scheduler-noise channels. Dimensions
must be divisible by 16 and frame counts must be `4k+1`; seeds are expanded by
a CPU generator for deterministic initial noise before device transfer.

The stored-transformer forward boundary is also explicit: FP32 scheduler
latents and the 20-channel condition are concatenated, transiently cast to the
artifact's F16 compute dtype, and run under the model's independent `cond` or
`uncond` cache context. The returned F16 prediction remains mixed-dtype-safe
for the pinned FP32 UniPC scheduler; no weight conversion occurs.

The official support directory is independently identity-bound. The model
index, scheduler config, raw SentencePiece model, both transformer configs,
UMT5 config, and VAE config are size-bounded, SHA-256 hashed, and
identity-bound before use.
Runtime scheduler/tokenizer objects are built from that validated config/byte
snapshot rather than reopening untrusted support files.

`NativeWanI2VRuntime` is the Engine-owned composition boundary for these
parts. It materializes the validated high/low transformers, stored UMT5, and
BF16 VAE on CPU, then runs prompt encoding, first-frame conditioning, staged
denoising, and decode under their separate residency sessions. Only one large
component is resident at a time, cancellation unwinds the active context, and
the returned video is detached on CPU with exact component provenance. The
normal tool/catalog path and output serialization are intentionally separate
integration seams.

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
width = 1280
height = 704

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
width = 1280
height = 704

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

The stored adapter validates and materializes complete current
FP8, legacy scaled-FP8, and ConvRot INT8 transformer artifacts into a Diffusers
meta skeleton without converting weights. It binds the exact artifact/config and
stored dense-role contract: the proven current Comfy FP8 layout keeps only
`patch_embedding` weight and bias in F32, keeps remaining dense state in F16,
and transiently returns F16 activations; legacy FP8 and ConvRot currently require
uniform F16 dense state. `NativeWanI2VRuntime` composes this adapter with the
stored UMT5, standalone VAE, scheduler, conditioning, and output serializer.

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

The validated local components now feed `NativeWanI2VRuntime`, which owns
prompt, conditioning, high/low stage switching, scheduling, and decode. Its
result carries exact support and artifact provenance. The normal catalog/tool path
and atomic MP4 serialization are implemented; stage-specific LoRA patching remains
unavailable and therefore cannot be selected by a variant. ComfyUI remains a
behavioral reference, not a bundled backend. No Hugging Face or Comfy cache/model
universe is introduced, and conversion or save-quantized nodes are out of scope.

Artifact probes validate container offsets and file bounds without reading tensor
payloads, then record resolved path, size, modification time, and a header/table
digest in the runtime request. Materializers revalidate that identity around
their bound SafeTensors handle before payload use. Support files use canonical
open-handle paths on Windows and are hashed from bounded snapshots.
