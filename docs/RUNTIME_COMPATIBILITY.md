# Runtime compatibility policy

LatentSlate Engine is opinionated about the creative tools it ships, but it must
not make a single GPU architecture part of the public tool contract.

## Baseline runtime

- Windows and Linux are first-class NVIDIA CUDA targets.
- A plain `uv sync` installs the complete currently supported model runtime.
- Windows and Linux resolve PyTorch from the official CUDA 12.8 wheel index.
  CUDA 12.8 was selected because it supports Blackwell while retaining a broad
  NVIDIA compatibility baseline.
- Python is pinned to 3.12 for the native-extension-heavy inference ecosystem.
- macOS can still resolve the normal PyPI PyTorch build, but the current Engine
  model catalog is designed and tested around NVIDIA CUDA execution.

## Architecture-specific acceleration

Future Sol-Attn, SageAttention, pre-quantized FP8/NVFP4 artifact loaders, fused kernels, compile profiles, and
similar accelerators are implementation capabilities, not different LatentSlate
tools. An optimization may be enabled only when all of these are true:

1. The selected model/recipe supports it.
2. The package and kernel are installed for the current operating system.
3. The detected GPU meets the minimum compute capability.
4. The path has been validated for the selected resolution and memory profile.

If any condition is false, Engine must mark that specific artifact recipe
unavailable with an actionable explanation. It must never convert a model as a
fallback. Unsupported hardware must not prevent unrelated tools from starting.

Optimization selection must not change tool IDs, creator-facing input schemas, or
project meaning. Completed jobs should record the resolved runtime profile and
optimizations in provenance so results remain auditable.

## Blackwell

Blackwell systems such as the local RTX 5080 and suitable Vast.ai instances can
use Blackwell-specific pre-quantized artifacts only after their exact loader is
validated. NVFP4 is not a universal fallback. Non-Blackwell NVIDIA systems should
continue to use a supported native BF16 artifact rather than attempting
Blackwell-only kernels.

## Secrets and remote deployment

`HF_TOKEN` and `LATENTSLATE_ENGINE_TOKEN` are ordinary environment variables.
For local development they can live in an ignored `.env` file. Remote templates
should inject them as secrets or container environment variables; they must never
be copied into an image or committed to the repository.
