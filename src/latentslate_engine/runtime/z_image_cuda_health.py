"""Shared bounded CUDA health probe for the exact Z-Image worker."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

_HEALTH_SUBSTAGES = (
    "sync_before",
    "allocate",
    "copy",
    "sync_after",
    "readback",
)


def z_image_cuda_health_check(
    torch_module: Any,
    execution_device: Any,
    *,
    checkpoint: Callable[[str], None] = lambda _stage: None,
) -> dict[str, str | int | bool]:
    """Copy 16 ordinary CPU bytes through CUDA and return fixed safe facts."""

    target = torch_module.device(execution_device)
    if target.type not in {"cpu", "cuda"} or (
        target.type == "cuda" and target.index is None
    ):
        raise ValueError("Z-Image CUDA health requires CPU conformance or indexed CUDA")
    device_context = (
        torch_module.cuda.device(target) if target.type == "cuda" else nullcontext()
    )
    with torch_module.inference_mode(), device_context:
        checkpoint("sync_before")
        if target.type == "cuda":
            torch_module.cuda.synchronize(target)
        source = torch_module.arange(
            16,
            dtype=torch_module.uint8,
            device=torch_module.device("cpu"),
        )
        if (
            source.device.type != "cpu"
            or source.dtype is not torch_module.uint8
            or source.shape != (16,)
            or not source.is_contiguous()
            or source.storage_offset() != 0
        ):
            raise RuntimeError("Z-Image synthetic CPU source differs")
        checkpoint("allocate")
        destination = torch_module.empty_like(source, device=target)
        if (
            destination.device != target
            or destination.dtype is not torch_module.uint8
            or destination.shape != source.shape
            or not destination.is_contiguous()
            or destination.storage_offset() != 0
        ):
            raise RuntimeError("Z-Image synthetic CUDA destination differs")
        checkpoint("copy")
        destination.copy_(source, non_blocking=False)
        checkpoint("sync_after")
        if target.type == "cuda":
            torch_module.cuda.synchronize(target)
        checkpoint("readback")
        readback = destination.to(device="cpu", non_blocking=False)
        if not torch_module.equal(readback, source):
            raise RuntimeError("Z-Image synthetic CUDA readback differs")
    return {
        "source_device": "cpu",
        "target_device": str(target),
        "dtype": "uint8",
        "numel": 16,
        "contiguous": True,
        "storage_offset": 0,
        "blocking_copy": True,
        "readback_equal": True,
    }
