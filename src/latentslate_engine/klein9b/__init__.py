import os
from typing import Any

# Must be set before this package imports torch. Match the pinned Comfy baseline
# while preserving an explicit embedding-process allocator choice.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

__all__ = ["GenerationResult", "Klein9BIdentity", "Klein9BRuntime"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runtime import GenerationResult, Klein9BIdentity, Klein9BRuntime

        return {
            "GenerationResult": GenerationResult,
            "Klein9BIdentity": Klein9BIdentity,
            "Klein9BRuntime": Klein9BRuntime,
        }[name]
    raise AttributeError(name)
