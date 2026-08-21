"""Z-Image Qwen runtime composition, stored execution, residency, and proof.

The exact Z-Image path constructs the raw 398-weight ``model.*`` Qwen closure,
runs all 36 blocks,
captures block 34, still executes block 35 and the final norm, and returns the
unnormalized block-34 state.  This module reproduces that path without an
external graph runtime.  Comfy Kitchen is used only for the authenticated
stored FP8/NVFP4 tensor layouts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import revalidate_artifact
from ..stored_quant import restore_global_fp8_tensor, restore_nvfp4_tensor
from ..z_image_turbo_recipe import ZImagePipelineSupportPlan, revalidate_z_image_pipeline_support
from . import z_image_cuda_health as _cuda_health
from .z_image_qwen_architecture import (
    QWEN_HIDDEN_SIZE,
    QWEN_NORM_EPS,
    QWEN_VOCAB_SIZE,
    ZImageQwenTextEncoder,
)
from .z_image_qwen_architecture import (
    QWEN_WEIGHT_COUNT as _QWEN_WEIGHT_COUNT,
)
from .z_image_qwen_architecture import (
    Cancel as _Cancel,
)
from .z_image_qwen_architecture import (
    Diagnostic as _Diagnostic,
)
from .z_image_qwen_architecture import (
    checkpoint as _checkpoint,
)
from .z_image_qwen_checkpoint import (
    QWEN_FIRST_LINEAR_SHAPE as _QWEN_FIRST_LINEAR_SHAPE,
)
from .z_image_qwen_checkpoint import (
    ZImageMixedQwenPlan,
    revalidate_z_image_mixed_qwen,
    validate_support_qwen_config,
)


class ZImageQwenDenseLinear(nn.Module):
    """BF16 CPU-master linear with exact per-operation F32 execution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str = "meta",
        ordinal: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.ordinal = ordinal
        self._cancelled: _Cancel = lambda: False
        self._diagnostic: _Diagnostic = lambda _stage: None
        self.per_op_move_count = 0

    def set_runtime_callbacks(self, cancelled: _Cancel, diagnostic: _Diagnostic) -> None:
        self._cancelled = cancelled
        self._diagnostic = diagnostic

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _checkpoint(
            self._cancelled,
            self._diagnostic,
            f"conditioning.linear_{self.ordinal:03d}",
        )
        if value.dtype is not torch.float32:
            raise TypeError("Z-Image Qwen dense linear requires F32 activations")
        weight = self.weight.to(device=value.device, dtype=value.dtype)
        if weight.device != value.device or weight.dtype is not torch.float32:
            raise RuntimeError("Z-Image Qwen dense per-op residency changed its contract")
        self.per_op_move_count += 1
        output = F.linear(value, weight)
        del weight
        return output


class ZImageQwenEmbedding(nn.Module):
    def __init__(self, *, device: torch.device | str = "meta") -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                (QWEN_VOCAB_SIZE, QWEN_HIDDEN_SIZE),
                device=device,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self._cancelled: _Cancel = lambda: False
        self._diagnostic: _Diagnostic = lambda _stage: None
        self.per_op_move_count = 0

    def set_runtime_callbacks(self, cancelled: _Cancel, diagnostic: _Diagnostic) -> None:
        self._cancelled = cancelled
        self._diagnostic = diagnostic

    def forward(self, input_ids: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
        _checkpoint(self._cancelled, self._diagnostic, "conditioning.edge_13")
        weight = self.weight.to(device=input_ids.device, dtype=torch.bfloat16)
        if weight.device != input_ids.device or weight.dtype is not torch.bfloat16:
            raise RuntimeError("Z-Image Qwen embedding per-op residency changed its contract")
        self.per_op_move_count += 1
        _checkpoint(self._cancelled, self._diagnostic, "conditioning.edge_14")
        output = F.embedding(input_ids, weight).to(dtype=out_dtype)
        del weight
        return output


class ZImageQwenRMSNorm(nn.Module):
    def __init__(self, width: int, *, device: torch.device | str = "meta") -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(width, device=device, dtype=torch.bfloat16), requires_grad=False
        )
        self.per_op_move_count = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=value.device, dtype=value.dtype)
        self.per_op_move_count += 1
        output = F.rms_norm(value, (value.shape[-1],), weight=weight, eps=QWEN_NORM_EPS)
        del weight
        return output


class _RuntimeModuleFactory:
    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str,
        ordinal: int,
    ) -> nn.Module:
        return ZImageQwenDenseLinear(
            in_features, out_features, device=device, ordinal=ordinal
        )

    def embedding(self, *, device: torch.device | str) -> nn.Module:
        return ZImageQwenEmbedding(device=device)

    def norm(self, width: int, *, device: torch.device | str) -> nn.Module:
        return ZImageQwenRMSNorm(width, device=device)


def build_z_image_mixed_qwen_shell(support: ZImagePipelineSupportPlan) -> nn.Module:
    """Compose the exact raw 398-key shell with production execution modules."""

    if not revalidate_z_image_pipeline_support(support):
        raise ValueError("Z-Image pipeline support changed before Qwen construction")
    validate_support_qwen_config(support)
    model = ZImageQwenTextEncoder(device="meta", factory=_RuntimeModuleFactory())
    keys = tuple(model.state_dict())
    if (
        len(keys) != _QWEN_WEIGHT_COUNT
        or any(not key.startswith("model.") for key in keys)
        or any("lm_head" in key for key in keys)
    ):
        raise RuntimeError("Z-Image Qwen shell closure differs from the raw 398-key model")
    return model


def materialize_z_image_mixed_qwen(
    plan: ZImageMixedQwenPlan,
    model: nn.Module,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: _Cancel = lambda: False,
) -> nn.Module:
    """Materialize the exact raw closure without a dense-checkpoint fallback."""

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    if not revalidate_z_image_mixed_qwen(plan):
        raise ValueError("Z-Image mixed Qwen changed after planning")
    target = model.state_dict()
    expected = set(plan.source_to_target.values())
    if set(target) != expected or len(target) != _QWEN_WEIGHT_COUNT:
        missing, extra = sorted(expected - set(target)), sorted(set(target) - expected)
        raise ValueError(
            f"Z-Image Qwen shell does not exactly match raw closure: missing={missing[:2]}, extra={extra[:2]}"
        )
    consumed: set[str] = set()
    native: dict[str, str] = {}
    total, completed = len(plan.source_to_target), 0

    def committed() -> None:
        nonlocal completed
        completed += 1
        if progress is not None and (completed == total or completed % 8 == 0):
            progress(completed, total)

    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("Z-Image mixed Qwen changed before materialization")
        for source in plan.fp8_sources:
            if cancelled():
                raise RuntimeError("Z-Image Qwen materialization canceled")
            _replace_quantized_linear(
                model,
                source,
                restore_global_fp8_tensor(
                    handle.get_tensor(source),
                    handle.get_tensor(source.removesuffix(".weight") + ".weight_scale"),
                    torch.bfloat16,
                ),
                "fp8",
            )
            consumed.add(source)
            native[source.removesuffix(".weight")] = "fp8"
            committed()
        for source in plan.nvfp4_sources:
            if cancelled():
                raise RuntimeError("Z-Image Qwen materialization canceled")
            stem = source.removesuffix(".weight")
            qdata = handle.get_tensor(source)
            _replace_quantized_linear(
                model,
                source,
                restore_nvfp4_tensor(
                    qdata,
                    handle.get_tensor(stem + ".weight_scale"),
                    handle.get_tensor(stem + ".weight_scale_2"),
                    (qdata.shape[0], qdata.shape[1] * 2),
                    torch.bfloat16,
                ),
                "nvfp4",
            )
            consumed.add(source)
            native[stem] = "nvfp4"
            committed()
        for source in plan.dense_sources:
            if cancelled():
                raise RuntimeError("Z-Image Qwen materialization canceled")
            value = handle.get_tensor(source)
            if tuple(value.shape) != tuple(target[source].shape):
                raise ValueError(f"Z-Image Qwen dense shape mismatch: {source}")
            set_module_tensor_to_device(model, source, "cpu", value=value, dtype=torch.bfloat16)
            consumed.add(source)
            committed()
    if consumed != set(plan.source_to_target):
        raise ValueError("Z-Image mixed Qwen source materialization is incomplete")
    unresolved = [key for key, value in model.state_dict().items() if value.is_meta]
    if unresolved:
        raise ValueError(f"Z-Image Qwen retains meta parameters: {unresolved[:2]}")
    model._latentslate_z_image_quant_modules = MappingProxyType(native)
    model._latentslate_z_image_first_linear_format = plan.first_linear_format
    model._latentslate_z_image_qwen_identity = plan.identity
    model.eval()
    return model


class _ZImageFullPrecisionLinear(nn.Module):
    """Kitchen stored weight with Comfy's exact ``full_precision_mm`` chain."""

    def __init__(
        self,
        weight: Any,
        *,
        layout: str,
        storage_dtype: torch.dtype,
        ordinal: int,
    ) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != layout
            or weight._qdata.dtype is not storage_dtype
        ):
            raise TypeError("Z-Image full-precision wrapper received an invalid Kitchen layout")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.ordinal = ordinal
        self._layout = layout
        self._storage_dtype = storage_dtype
        self._cancelled: _Cancel = lambda: False
        self._diagnostic: _Diagnostic = lambda _stage: None
        self.native_dequant_count = 0
        self.f_linear_count = 0
        self.rejected_dispatch_count = 0
        self.dense_checkpoint_fallback_count = 0
        self.activation_quantized = False
        self.scaled_mm_calls = 0
        self.per_op_move_count = 0
        self.last_dispatch_error: str | None = None

    def set_runtime_callbacks(self, cancelled: _Cancel, diagnostic: _Diagnostic) -> None:
        self._cancelled = cancelled
        self._diagnostic = diagnostic

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        _checkpoint(
            self._cancelled,
            self._diagnostic,
            f"conditioning.linear_{self.ordinal:03d}",
        )
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("Z-Image Qwen input feature count differs from stored weight")
        if input.dtype is not torch.float32:
            self.rejected_dispatch_count += 1
            raise TypeError("Z-Image Qwen full-precision matrix multiply requires F32 input")
        try:
            weight_for_input = _transport_z_image_quantized_weight(
                self.weight,
                input.device,
            )
            if (
                weight_for_input.layout != self._layout
                or weight_for_input.orig_shape != tuple(self.weight.shape)
                or weight_for_input.qdata.dtype is not self._storage_dtype
                or tuple(weight_for_input.qdata.shape) != tuple(self.weight._qdata.shape)
                or weight_for_input.qdata.device != input.device
                or not weight_for_input.qdata.is_contiguous()
                or weight_for_input.scale.device != input.device
                or (
                    weight_for_input.block_scale is not None
                    and weight_for_input.block_scale.device != input.device
                )
            ):
                raise RuntimeError("Z-Image Qwen raw-byte transport changed stored payload")
            self.per_op_move_count += 1
            dense_weight = _direct_fp32_dequantize_z_image_weight(weight_for_input)
            if (
                dense_weight.ndim != 2
                or tuple(dense_weight.shape) != tuple(self.weight.shape)
                or dense_weight.dtype is not torch.float32
                or dense_weight.device != input.device
                or not dense_weight.is_contiguous()
            ):
                raise RuntimeError("Z-Image Qwen Kitchen dequantization changed F32 geometry")
            self.native_dequant_count += 1
            output = F.linear(input, dense_weight)
            self.f_linear_count += 1
        except BaseException as exc:
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = type(exc).__name__
            raise RuntimeError(
                "Z-Image Qwen full-precision Kitchen dispatch failed; fallback is forbidden"
            ) from exc
        finally:
            if "dense_weight" in locals():
                del dense_weight
            if "weight_for_input" in locals():
                del weight_for_input
        self.last_dispatch_error = None
        return output


class ZImageFullPrecisionFP8Linear(_ZImageFullPrecisionLinear):
    def __init__(self, weight: Any, *, ordinal: int = 0) -> None:
        super().__init__(
            weight,
            layout="TensorCoreFP8Layout",
            storage_dtype=torch.float8_e4m3fn,
            ordinal=ordinal,
        )


class ZImageFullPrecisionNVFP4Linear(_ZImageFullPrecisionLinear):
    def __init__(self, weight: Any, *, ordinal: int = 0) -> None:
        super().__init__(
            weight,
            layout="TensorCoreNVFP4Layout",
            storage_dtype=torch.uint8,
            ordinal=ordinal,
        )


def _replace_quantized_linear(model: nn.Module, source: str, weight: Any, kind: str) -> None:
    stem = source.removesuffix(".weight")
    module = model.get_submodule(stem)
    if (
        type(module) is not ZImageQwenDenseLinear
        or tuple(module.weight.shape) != tuple(weight.shape)
    ):
        raise TypeError(f"Z-Image Qwen quantized target is not exact bias-free Linear: {stem}")
    parent_path, _, leaf = stem.rpartition(".")
    replacement: nn.Module = (
        ZImageFullPrecisionFP8Linear(weight, ordinal=module.ordinal)
        if kind == "fp8"
        else ZImageFullPrecisionNVFP4Linear(weight, ordinal=module.ordinal)
    )
    setattr(model.get_submodule(parent_path), leaf, replacement)


def z_image_mixed_dispatch_snapshot(model: nn.Module) -> dict[str, tuple[int, int, int, int]]:
    expected = getattr(model, "_latentslate_z_image_quant_modules", None)
    if (
        not isinstance(expected, Mapping)
        or len(expected) != 189
        or sum(value == "fp8" for value in expected.values()) != 177
        or sum(value == "nvfp4" for value in expected.values()) != 12
        or set(expected.values()) != {"fp8", "nvfp4"}
    ):
        raise ValueError("Z-Image mixed Qwen native module closure is incomplete")
    values: dict[str, tuple[int, int, int, int]] = {}
    for name in expected:
        module = model.get_submodule(name)
        _require_z_image_mixed_native_wrapper(name, module, expected[name])
        counters = tuple(
            getattr(module, counter, None)
            for counter in (
                "native_dequant_count",
                "f_linear_count",
                "rejected_dispatch_count",
                "dense_checkpoint_fallback_count",
            )
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError(f"Z-Image mixed Qwen dispatch counters are invalid: {name}")
        if counters[2] != 0 or counters[3] != 0:
            raise ValueError(
                f"Z-Image mixed Qwen has rejected or checkpoint-fallback history: {name}"
            )
        values[name] = counters
    return values


def _require_z_image_mixed_native_wrapper(name: str, module: nn.Module, kind: object) -> None:
    if kind == "fp8":
        expected_type, layout, dtype = (
            ZImageFullPrecisionFP8Linear,
            "TensorCoreFP8Layout",
            torch.float8_e4m3fn,
        )
    elif kind == "nvfp4":
        expected_type, layout, dtype = (
            ZImageFullPrecisionNVFP4Linear,
            "TensorCoreNVFP4Layout",
            torch.uint8,
        )
    else:
        raise ValueError(f"Z-Image mixed Qwen has unknown native wrapper kind: {name}")
    if type(module) is not expected_type:
        raise TypeError(f"Z-Image mixed Qwen native wrapper type differs from pin: {name}")
    weight = getattr(module, "weight", None)
    if (
        getattr(weight, "_layout_cls", None) != layout
        or getattr(getattr(weight, "_qdata", None), "dtype", None) is not dtype
        or getattr(module, "activation_quantized", object()) is not False
        or getattr(module, "scaled_mm_calls", object()) != 0
    ):
        raise TypeError(f"Z-Image mixed Qwen stored wrapper layout differs from pin: {name}")
    params = getattr(weight, "params", None)
    scale = getattr(params, "scale", None)
    if not isinstance(scale, torch.Tensor) or scale.dtype is not torch.float32 or scale.numel() != 1:
        raise TypeError(f"Z-Image mixed Qwen stored wrapper scale differs from pin: {name}")
    if kind == "nvfp4":
        block_scale = getattr(params, "block_scale", None)
        if not isinstance(block_scale, torch.Tensor) or block_scale.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"Z-Image mixed Qwen NVFP4 block scale differs from pin: {name}")


def verify_z_image_mixed_dispatch(
    model: nn.Module,
    before: Mapping[str, tuple[int, int, int, int]],
) -> dict[str, int | str | bool]:
    after = z_image_mixed_dispatch_snapshot(model)
    if set(before) != set(after):
        raise RuntimeError("Z-Image mixed Qwen module set changed during dispatch")
    dequant = {name: after[name][0] - before[name][0] for name in after}
    linear = {name: after[name][1] - before[name][1] for name in after}
    rejected = sum(after[name][2] - before[name][2] for name in after)
    fallback = sum(after[name][3] - before[name][3] for name in after)
    if (
        any(value <= 0 for value in dequant.values())
        or any(value <= 0 for value in linear.values())
        or any(dequant[name] != linear[name] for name in after)
        or rejected
        or fallback
    ):
        raise RuntimeError("Z-Image mixed Qwen did not dequantize and F.linear every stored layer")
    return {
        "contract": "full_precision_mm",
        "backend": "comfy-kitchen/public-direct-fp32-dequant+torch/f.linear",
        "module_count": 189,
        "dequantized_modules": 189,
        "f_linear_modules": 189,
        "complete": True,
        "total_dequantizations": sum(dequant.values()),
        "min_module_dequant_delta": min(dequant.values()),
        "max_module_dequant_delta": max(dequant.values()),
        "total_f_linear_calls": sum(linear.values()),
        "fp8_modules": 177,
        "nvfp4_modules": 12,
        "rejected_dispatch_count": rejected,
        "dense_checkpoint_fallback_count": fallback,
        "activation_quantized": False,
        "scaled_mm_calls": 0,
        "per_op_residency": True,
        "stored_transport": "source-backed-raw-byte/vbar-equivalent",
        "full_module_cuda_onload": False,
        "cpu_master_retained": True,
    }


@dataclass(frozen=True, slots=True)
class _ZImageTransportedWeight:
    layout: str
    qdata: torch.Tensor
    scale: torch.Tensor
    block_scale: torch.Tensor | None
    orig_shape: tuple[int, ...]


def _allocate_z_image_raw_bytes(
    flat_bytes_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_like(flat_bytes_cpu, device=device)


def _copy_z_image_bytes_into(
    destination: torch.Tensor,
    flat_bytes_cpu: torch.Tensor,
) -> None:
    destination.copy_(flat_bytes_cpu, non_blocking=False)


def _copy_z_image_raw_bytes(
    flat_bytes_cpu: torch.Tensor,
    device: torch.device,
    *,
    before_allocate: Callable[[], None] = lambda: None,
    before_copy: Callable[[], None] = lambda: None,
) -> torch.Tensor:
    from comfy_kitchen.tensor import QuantizedTensor

    if isinstance(flat_bytes_cpu, QuantizedTensor):
        raise TypeError("Z-Image Qwen byte transport requires an ordinary Tensor")
    before_allocate()
    destination = _allocate_z_image_raw_bytes(flat_bytes_cpu, device)
    if isinstance(destination, QuantizedTensor):
        raise TypeError("Z-Image Qwen byte destination is not an ordinary Tensor")
    before_copy()
    _copy_z_image_bytes_into(destination, flat_bytes_cpu)
    return destination


def _prepare_z_image_flat_raw_bytes(qdata: torch.Tensor) -> torch.Tensor:
    if (
        not qdata.is_contiguous()
        or qdata.storage_offset() != 0
        or qdata.element_size() != 1
    ):
        raise RuntimeError("Z-Image Qwen qdata is not flat-byte-view compatible")
    raw = qdata.view(torch.uint8).reshape(-1)
    if (
        raw.ndim != 1
        or raw.dtype is not torch.uint8
        or raw.element_size() != 1
        or not raw.is_contiguous()
        or raw.storage_offset() != 0
        or raw.stride() != (1,)
    ):
        raise RuntimeError("Z-Image Qwen source byte buffer is not flat contiguous")
    return raw


def _view_z_image_flat_raw_as_dtype(
    raw: torch.Tensor,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    return raw.view(dtype=storage_dtype)


def _restore_z_image_qdata_shape(
    typed_flat: torch.Tensor,
    storage_shape: torch.Size,
) -> torch.Tensor:
    return typed_flat.view(storage_shape)


def _probe_z_image_source_shaped_fp8_view_capability(
    source_qdata: torch.Tensor,
    execution_device: torch.device | str,
) -> torch.Tensor:
    """Exercise native source-shaped FP8 dtype-view support outside production."""

    if (
        source_qdata.dtype is not torch.float8_e4m3fn
        or not source_qdata.is_contiguous()
        or source_qdata.storage_offset() != 0
    ):
        raise ValueError("Z-Image Qwen FP8 capability probe source differs")
    shaped_raw = source_qdata.view(torch.uint8)
    copied = _copy_z_image_raw_bytes(shaped_raw, torch.device(execution_device))
    return copied.view(dtype=source_qdata.dtype)


def _move_z_image_scale_field(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    destination = torch.empty_like(value, device=device)
    destination.copy_(value, non_blocking=False)
    return destination


def _transport_z_image_quantized_weight(
    source: Any,
    execution_device: torch.device | str,
    *,
    cancelled: _Cancel = lambda: False,
    diagnostic: _Diagnostic = lambda _stage: None,
    diagnostic_prefix: str | None = None,
    verify_bits: bool = False,
) -> _ZImageTransportedWeight:
    """Move exact source bytes and scale fields without typed-wrapper dispatch.

    The Engine intentionally uses the synchronous current-stream branch. This
    is VBar-output-equivalent without importing Comfy's buffer/stream manager.
    """

    device = torch.device(execution_device)

    def boundary(substage: str) -> None:
        if diagnostic_prefix is not None:
            _checkpoint(cancelled, diagnostic, f"{diagnostic_prefix}_{substage}")

    plain = source.layout_cls.get_plain_tensors(source)
    if source._layout_cls == "TensorCoreFP8Layout":
        if len(plain) != 2 or plain[0].dtype is not torch.float8_e4m3fn:
            raise RuntimeError("Z-Image Qwen FP8 source closure differs")
        block_scale_source = None
    elif source._layout_cls == "TensorCoreNVFP4Layout":
        if len(plain) != 3 or plain[0].dtype is not torch.uint8:
            raise RuntimeError("Z-Image Qwen NVFP4 source closure differs")
        block_scale_source = plain[2]
    else:
        raise TypeError("Z-Image Qwen raw-byte transport received an unsupported layout")
    qdata_source, scale_source = plain[0], plain[1]
    if (
        qdata_source.device.type != "cpu"
        or scale_source.device.type != "cpu"
        or scale_source.dtype is not torch.float32
        or (block_scale_source is not None and block_scale_source.device.type != "cpu")
    ):
        raise RuntimeError("Z-Image Qwen transport source is not the CPU-master closure")
    source_state = (
        qdata_source,
        scale_source,
        block_scale_source,
        qdata_source.dtype,
        tuple(qdata_source.shape),
        tuple(qdata_source.stride()),
        scale_source.dtype,
        tuple(scale_source.shape),
        tuple(scale_source.stride()),
        (
            None
            if block_scale_source is None
            else (
                block_scale_source.dtype,
                tuple(block_scale_source.shape),
                tuple(block_scale_source.stride()),
            )
        ),
        tuple(source.params.orig_shape),
        source.params.orig_dtype,
    )

    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        boundary("origin_flat_prepare")
        raw_source = _prepare_z_image_flat_raw_bytes(qdata_source)
        byte_count = qdata_source.numel() * qdata_source.element_size()
        if raw_source.numel() != byte_count or tuple(raw_source.shape) != (byte_count,):
            raise RuntimeError("Z-Image Qwen qdata byte count changed")

        boundary("origin_uint8_copy")
        raw_destination = _copy_z_image_raw_bytes(raw_source, device)
        if (
            raw_destination.ndim != 1
            or raw_destination.dtype is not torch.uint8
            or raw_destination.element_size() != 1
            or raw_destination.numel() != byte_count
            or tuple(raw_destination.shape) != (byte_count,)
            or not raw_destination.is_contiguous()
            or raw_destination.storage_offset() != 0
            or raw_destination.stride() != (1,)
            or raw_destination.device != device
        ):
            raise RuntimeError("Z-Image Qwen raw qdata copy changed byte geometry")

        if qdata_source.dtype is torch.float8_e4m3fn:
            boundary("flat_dtype_view")
            typed_flat = _view_z_image_flat_raw_as_dtype(
                raw_destination,
                qdata_source.dtype,
            )
            if (
                typed_flat.ndim != 1
                or typed_flat.dtype is not qdata_source.dtype
                or typed_flat.numel() != qdata_source.numel()
                or not typed_flat.is_contiguous()
                or typed_flat.storage_offset() != 0
                or typed_flat.stride() != (1,)
            ):
                raise RuntimeError("Z-Image Qwen flat dtype view changed storage")
        else:
            typed_flat = raw_destination

        boundary("shape_restore")
        qdata = _restore_z_image_qdata_shape(typed_flat, qdata_source.shape)
        if (
            qdata.dtype is not qdata_source.dtype
            or tuple(qdata.shape) != tuple(qdata_source.shape)
            or qdata.device != device
            or not qdata.is_contiguous()
        ):
            raise RuntimeError("Z-Image Qwen qdata reinterpretation changed storage")

        boundary("scale_move")
        scale = _move_z_image_scale_field(scale_source, device)
        block_scale = (
            _move_z_image_scale_field(block_scale_source, device)
            if block_scale_source is not None
            else None
        )
        if (
            scale.dtype is not torch.float32
            or scale.device != device
            or tuple(scale.shape) != tuple(scale_source.shape)
            or tuple(scale.stride()) != tuple(scale_source.stride())
            or (
                block_scale is not None
                and (
                    block_scale.dtype is not block_scale_source.dtype
                    or tuple(block_scale.shape) != tuple(block_scale_source.shape)
                    or tuple(block_scale.stride()) != tuple(block_scale_source.stride())
                    or block_scale.device != device
                )
            )
        ):
            raise RuntimeError("Z-Image Qwen transported scale fields changed")

        if verify_bits:
            boundary("bit_verify")
            if (
                not torch.equal(
                    qdata.view(torch.uint8).reshape(-1).to("cpu"), raw_source
                )
                or not torch.equal(scale.to("cpu"), scale_source)
                or (
                    block_scale is not None
                    and not torch.equal(block_scale.to("cpu"), block_scale_source)
                )
            ):
                raise RuntimeError("Z-Image Qwen raw-byte transport changed stored bits")

    if (
        source_state[0] is not source._qdata
        or source_state[1] is not source.params.scale
        or source_state[2] is not getattr(source.params, "block_scale", None)
        or source_state[3] is not source._qdata.dtype
        or source_state[4] != tuple(source._qdata.shape)
        or source_state[5] != tuple(source._qdata.stride())
        or source_state[6] is not source.params.scale.dtype
        or source_state[7] != tuple(source.params.scale.shape)
        or source_state[8] != tuple(source.params.scale.stride())
        or source_state[9]
        != (
            None
            if getattr(source.params, "block_scale", None) is None
            else (
                source.params.block_scale.dtype,
                tuple(source.params.block_scale.shape),
                tuple(source.params.block_scale.stride()),
            )
        )
        or source_state[10] != tuple(source.params.orig_shape)
        or source_state[11] is not source.params.orig_dtype
        or source._qdata.device.type != "cpu"
    ):
        raise RuntimeError("Z-Image Qwen CPU-master source mutated during transport")
    return _ZImageTransportedWeight(
        source._layout_cls,
        qdata,
        scale,
        block_scale,
        tuple(source.params.orig_shape),
    )


def _direct_fp32_dequantize_z_image_weight(
    weight: _ZImageTransportedWeight,
) -> torch.Tensor:
    """Use Kitchen's public layout APIs without mutating logical wrapper dtype."""

    import comfy_kitchen

    qdata = weight.qdata
    if weight.layout == "TensorCoreFP8Layout":
        dense = comfy_kitchen.dequantize_per_tensor_fp8(
            qdata,
            weight.scale,
            output_type=torch.float32,
        )
    elif weight.layout == "TensorCoreNVFP4Layout":
        if weight.block_scale is None:
            raise RuntimeError("Z-Image Qwen NVFP4 plain-tensor closure differs")
        dense = comfy_kitchen.dequantize_nvfp4(
            qdata,
            weight.scale,
            weight.block_scale,
            output_type=torch.float32,
        )
    else:
        raise TypeError("Z-Image Qwen direct dequant received an unsupported layout")
    orig_shape = weight.orig_shape
    if tuple(dense.shape) != orig_shape:
        dense = dense[tuple(slice(0, extent) for extent in orig_shape)]
    return dense.contiguous()


def _preflight_z_image_full_precision_linear(
    module: _ZImageFullPrecisionLinear,
    execution_device: torch.device | str,
    *,
    expected_shape: tuple[int, int],
    cancelled: _Cancel = lambda: False,
    diagnostic: _Diagnostic = lambda _stage: None,
) -> dict[str, str | bool | int]:
    """Exercise one stored wrapper without mutating per-command proof counters."""

    device = torch.device(execution_device)
    kind = "fp8" if type(module) is ZImageFullPrecisionFP8Linear else "nvfp4"
    prefix = f"conditioning.preflight_{kind}"
    health_stage = {
        "sync_before": "cuda_sync",
        "allocate": "uint8_allocate",
        "copy": "ordinary_uint8_copy",
        "sync_after": "ordinary_uint8_sync",
        "readback": "ordinary_uint8_readback",
    }
    health_facts = _cuda_health.z_image_cuda_health_check(
        torch,
        device,
        checkpoint=lambda substage: _checkpoint(
            cancelled,
            diagnostic,
            prefix + "_" + health_stage[substage],
        ),
    )
    if health_facts != {
        "source_device": "cpu",
        "target_device": str(device),
        "dtype": "uint8",
        "numel": 16,
        "contiguous": True,
        "storage_offset": 0,
        "blocking_copy": True,
        "readback_equal": True,
    }:
        raise RuntimeError("Z-Image Qwen shared CUDA health facts differ")

    from comfy_kitchen.tensor import get_layout_class

    expected_layout = (
        "TensorCoreFP8Layout" if kind == "fp8" else "TensorCoreNVFP4Layout"
    )
    expected_storage_dtype = torch.float8_e4m3fn if kind == "fp8" else torch.uint8
    registered_op = "dequantize_fp8" if kind == "fp8" else "dequantize_nvfp4"
    public_backend = (
        "comfy_kitchen.dequantize_per_tensor_fp8+torch/f.linear"
        if kind == "fp8"
        else "comfy_kitchen.dequantize_nvfp4+torch/f.linear"
    )
    if not hasattr(torch.ops.comfy_kitchen, registered_op):
        raise RuntimeError("Z-Image Qwen Kitchen dequantization operator is not registered")
    expected_storage_shape = (
        expected_shape if kind == "fp8" else (expected_shape[0], expected_shape[1] // 2)
    )
    weight = module.weight
    if (
        module._layout != expected_layout
        or module._storage_dtype is not expected_storage_dtype
        or weight._layout_cls != expected_layout
        or weight.layout_cls is not get_layout_class(expected_layout)
        or tuple(weight.shape) != expected_shape
        or tuple(weight.params.orig_shape) != expected_shape
        or weight.params.orig_dtype is not torch.bfloat16
        or tuple(weight._qdata.shape) != expected_storage_shape
        or weight._qdata.dtype is not expected_storage_dtype
        or weight._qdata.device.type != "cpu"
        or not weight._qdata.is_contiguous()
        or any(
            getattr(weight.params, field).device.type != "cpu"
            for field in weight.params._tensor_fields()
        )
    ):
        raise RuntimeError("Z-Image Qwen first-linear CPU-master structure differs from pin")

    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        current_weight = _transport_z_image_quantized_weight(
            weight,
            device,
            cancelled=cancelled,
            diagnostic=diagnostic,
            diagnostic_prefix=prefix,
            verify_bits=True,
        )
        if (
            current_weight.layout != expected_layout
            or current_weight.orig_shape != expected_shape
            or tuple(current_weight.qdata.shape) != expected_storage_shape
            or current_weight.qdata.dtype is not expected_storage_dtype
            or current_weight.qdata.device != device
            or not current_weight.qdata.is_contiguous()
            or current_weight.scale.device != device
            or (
                current_weight.block_scale is not None
                and current_weight.block_scale.device != device
            )
        ):
            raise RuntimeError("Z-Image Qwen first-linear current-device structure differs")

        _checkpoint(cancelled, diagnostic, prefix + "_direct_fp32_dequant")
        dense_weight = _direct_fp32_dequantize_z_image_weight(current_weight)
        if (
            tuple(dense_weight.shape) != expected_shape
            or dense_weight.dtype is not torch.float32
            or dense_weight.device != device
            or not dense_weight.is_contiguous()
            or not bool(torch.isfinite(dense_weight).all())
        ):
            raise RuntimeError("Z-Image Qwen first-linear dequantized structure differs")

        _checkpoint(cancelled, diagnostic, prefix + "_f_linear")
        probe = torch.zeros((1, 1, expected_shape[1]), device=device, dtype=torch.float32)
        output = F.linear(probe, dense_weight)
        if (
            tuple(output.shape) != (1, 1, expected_shape[0])
            or output.dtype is not torch.float32
            or output.device != device
            or not bool(torch.isfinite(output).all())
        ):
            raise RuntimeError("Z-Image Qwen first-linear F.linear result differs")
        _checkpoint(cancelled, diagnostic, prefix + "_validate")
        del output, probe, dense_weight, current_weight

    return {
        "first_linear_preflight": True,
        "first_linear_format": kind,
        "first_linear_logical_shape": f"{expected_shape[0]}x{expected_shape[1]}",
        "first_linear_storage_dtype": str(expected_storage_dtype).removeprefix("torch."),
        "first_linear_compute_dtype": "float32",
        "first_linear_output_shape": f"1x1x{expected_shape[0]}",
        "first_linear_backend": public_backend,
        "first_linear_layout_registered": True,
        "first_linear_transfer": "source-backed-raw-byte/current-stream/blocking",
        "first_linear_transport_equivalence": "vbar-output-equivalent",
        "first_linear_bit_identity": True,
        "first_linear_byte_count": _QWEN_FIRST_LINEAR_SHAPE[0]
        * _QWEN_FIRST_LINEAR_SHAPE[1],
        "first_linear_logical_wrapper_cast": False,
        "first_linear_dequant_contract": "public-direct-fp32",
    }


def _preflight_z_image_first_linear(
    model: nn.Module,
    execution_device: torch.device,
    cancelled: _Cancel,
    diagnostic: _Diagnostic,
) -> dict[str, str | bool | int]:
    if getattr(model, "_latentslate_z_image_first_linear_format", None) != "fp8":
        raise RuntimeError("Z-Image Qwen first-linear authenticated format is not FP8")
    matches = [
        module
        for module in model.modules()
        if isinstance(module, _ZImageFullPrecisionLinear) and module.ordinal == 0
    ]
    if len(matches) != 1 or type(matches[0]) is not ZImageFullPrecisionFP8Linear:
        raise RuntimeError("Z-Image Qwen first-linear wrapper differs from authenticated plan")
    counters = (
        matches[0].native_dequant_count,
        matches[0].f_linear_count,
        matches[0].rejected_dispatch_count,
        matches[0].dense_checkpoint_fallback_count,
        matches[0].per_op_move_count,
        matches[0].last_dispatch_error,
    )
    try:
        return _preflight_z_image_full_precision_linear(
            matches[0],
            execution_device,
            expected_shape=_QWEN_FIRST_LINEAR_SHAPE,
            cancelled=cancelled,
            diagnostic=diagnostic,
        )
    finally:
        (
            matches[0].native_dequant_count,
            matches[0].f_linear_count,
            matches[0].rejected_dispatch_count,
            matches[0].dense_checkpoint_fallback_count,
            matches[0].per_op_move_count,
            matches[0].last_dispatch_error,
        ) = counters


class ZImageMixedQwenStage:
    def __init__(
        self,
        model: nn.Module,
        execution_device: torch.device | str,
        cancelled: _Cancel = lambda: False,
        diagnostic: _Diagnostic = lambda _stage: None,
    ) -> None:
        self.model = model
        self.execution_device = torch.device(execution_device)
        self.cancelled = cancelled
        self.diagnostic = diagnostic
        self._before: dict[str, tuple[int, int, int, int]] | None = None
        self._preflight_proof: dict[str, str | bool | int] | None = None

    def onload(self) -> None:
        _checkpoint(self.cancelled, self.diagnostic, "conditioning.edge_07")
        _require_z_image_qwen_cpu_master(self.model)
        _checkpoint(self.cancelled, self.diagnostic, "conditioning.edge_08")
        self._preflight_proof = _preflight_z_image_first_linear(
            self.model,
            self.execution_device,
            self.cancelled,
            self.diagnostic,
        )
        self._before = z_image_mixed_dispatch_snapshot(self.model)
        _checkpoint(self.cancelled, self.diagnostic, "conditioning.edge_09")

    def verify_dispatch(self) -> dict[str, int | str | bool]:
        if self._before is None:
            raise RuntimeError("Z-Image mixed Qwen was not staged")
        if self._preflight_proof is None:
            raise RuntimeError("Z-Image mixed Qwen first-linear preflight was not completed")
        return {**verify_z_image_mixed_dispatch(self.model, self._before), **self._preflight_proof}

    def offload(self) -> None:
        _require_z_image_qwen_cpu_master(self.model)
        self._before = None
        self._preflight_proof = None


def move_z_image_mixed_qwen_storage(model: nn.Module, device: torch.device | str) -> None:
    """Compatibility seam: Qwen remains CPU-master regardless of execution device."""

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Z-Image Qwen execution device must be CPU or CUDA")
    _require_z_image_qwen_cpu_master(model)


def _require_z_image_qwen_cpu_master(model: nn.Module) -> None:
    parameters = tuple(model.parameters())
    if not parameters or any(value.is_meta or value.device.type != "cpu" for value in parameters):
        raise RuntimeError("Z-Image Qwen CPU-master residency is incomplete")
    for module in model.modules():
        if isinstance(module, (ZImageFullPrecisionFP8Linear, ZImageFullPrecisionNVFP4Linear)):
            weight = module.weight
            if weight._qdata.device.type != "cpu" or any(
                getattr(weight.params, field).device.type != "cpu"
                for field in weight.params._tensor_fields()
            ):
                raise RuntimeError("Z-Image Qwen Kitchen CPU-master fields diverged")


