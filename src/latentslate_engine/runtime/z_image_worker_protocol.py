"""Closed model-specific vocabulary shared by the Z worker parent and child."""

from __future__ import annotations

import json

SCHEMA_VERSION = 1
CUDA_HEALTH_PHASES = (
    "pre_import",
    "post_tokenizer",
    "post_qwen",
    "post_nextdit",
    "post_vae",
    "post_core",
    "pre_qwen_preflight",
)
CUDA_HEALTH_STAGES = frozenset(f"cuda_health_{phase}" for phase in CUDA_HEALTH_PHASES)
CUDA_ERROR_CODES = frozenset(
    {
        "cuda_oom",
        "illegal_memory_access",
        "invalid_argument",
        "operation_not_supported",
        "driver_error",
        "unknown_runtime",
    }
)
QWEN_FAILURE_STAGES = frozenset(
    {
        *(f"conditioning.edge_{index:02d}" for index in range(7, 21)),
        *(
            f"conditioning.preflight_{kind}_{substage}"
            for kind in ("fp8", "nvfp4")
            for substage in (
                "cuda_sync",
                "uint8_allocate",
                "ordinary_uint8_copy",
                "ordinary_uint8_sync",
                "ordinary_uint8_readback",
                "origin_flat_prepare",
                "origin_uint8_copy",
                "flat_dtype_view",
                "shape_restore",
                "scale_move",
                "bit_verify",
                "direct_fp32_dequant",
                "f_linear",
                "validate",
            )
        ),
        "conditioning.embedding",
        "conditioning.mask",
        "conditioning.rope",
        "conditioning.final_norm",
        *(f"conditioning.block_{index:02d}" for index in range(36)),
        *(f"conditioning.linear_{index:03d}" for index in range(252)),
    }
)
FAILURE_STAGES = frozenset(
    {
        "auth",
        "canonical_validation",
        "device_contract",
        "rehydrate",
        "runtime_import",
        "tokenizer",
        "qwen_materialize",
        "nextdit_materialize",
        "lora_install",
        "transformer_onload",
        "vae_materialize",
        "core_ready",
        "conditioning",
        "noise",
        "sampling",
        "decode",
        "publish",
        *CUDA_HEALTH_STAGES,
        *QWEN_FAILURE_STAGES,
    }
)
FAILURE_LOCATIONS = frozenset(
    {
        "z_image_turbo_worker._read_json",
        "z_image_turbo_worker._secret",
        "z_image_turbo_worker._validate",
        "z_image_turbo_worker._resolve_worker_cuda_device",
        "z_image_turbo_recipe.rehydrate_z_image_turbo_runtime_request",
        "z_image_turbo_worker._load_core",
        "z_image_turbo_worker._execute",
        "z_image_turbo_worker._validate_artifact",
        "z_image_turbo.generate",
    }
)
SAFE_EXCEPTION_CLASSES: dict[type[BaseException], str] = {
    AssertionError: "AssertionError",
    AttributeError: "AttributeError",
    BaseException: "BaseException",
    EOFError: "EOFError",
    Exception: "Exception",
    FileNotFoundError: "FileNotFoundError",
    ImportError: "ImportError",
    IsADirectoryError: "IsADirectoryError",
    json.JSONDecodeError: "JSONDecodeError",
    KeyError: "KeyError",
    MemoryError: "MemoryError",
    ModuleNotFoundError: "ModuleNotFoundError",
    NotADirectoryError: "NotADirectoryError",
    OSError: "OSError",
    OverflowError: "OverflowError",
    PermissionError: "PermissionError",
    RuntimeError: "RuntimeError",
    SystemExit: "SystemExit",
    TypeError: "TypeError",
    UnicodeDecodeError: "UnicodeDecodeError",
    ValueError: "ValueError",
}
SAFE_EXCEPTION_NAMES = frozenset(
    {
        *SAFE_EXCEPTION_CLASSES.values(),
        "OutOfMemoryError",
        "ZImageDecodeCancelled",
        "ZImagePngPublicationCancelled",
        "ZImageSamplingCancelled",
        "ZImageTurboCancelled",
    }
)


def safe_exception_name(exc: BaseException) -> str:
    """Return one fixed public exception label without reading exception content."""

    if type(exc).__module__.startswith("torch") and type(exc).__name__ == "OutOfMemoryError":
        return "OutOfMemoryError"
    return SAFE_EXCEPTION_CLASSES.get(type(exc), "Exception")


def valid_failure_labels(*, error_type: object, stage: object, location: object) -> bool:
    """Validate the shared closed portion of a Z failure envelope."""

    return (
        error_type in SAFE_EXCEPTION_NAMES
        and stage in FAILURE_STAGES
        and location in FAILURE_LOCATIONS
    )


__all__ = (
    "CUDA_ERROR_CODES",
    "CUDA_HEALTH_PHASES",
    "CUDA_HEALTH_STAGES",
    "FAILURE_LOCATIONS",
    "FAILURE_STAGES",
    "QWEN_FAILURE_STAGES",
    "SAFE_EXCEPTION_CLASSES",
    "SAFE_EXCEPTION_NAMES",
    "SCHEMA_VERSION",
    "safe_exception_name",
    "valid_failure_labels",
)
