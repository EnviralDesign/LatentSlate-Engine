"""Establish process-global CUDA configuration before test collection imports Torch."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
