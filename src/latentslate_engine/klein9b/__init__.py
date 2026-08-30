import os

# Must be set before this package imports torch. Match the pinned Comfy baseline
# while preserving an explicit embedding-process allocator choice.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from .runtime import GenerationResult, Klein9BIdentity, Klein9BRuntime

__all__ = ["GenerationResult", "Klein9BIdentity", "Klein9BRuntime"]
