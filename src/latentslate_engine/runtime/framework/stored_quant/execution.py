"""Direct stored-weight execution modules shared by model adapters."""

from __future__ import annotations

import re
from dataclasses import replace as dataclass_replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..residency import canonical_device

_PATCH_FINGERPRINT_ATTRIBUTE = "_latentslate_ltx23_patch_fingerprint"
_AIMDO_SIGNATURE_CACHEABLE_ATTRIBUTE = "_latentslate_aimdo_signature_cacheable"


def comfy_string_to_seed(value: str | bytes) -> int:
    """Return Comfy's exact unsigned CRC32 stochastic-rounding seed."""

    crc = 0xFFFFFFFF
    for byte in value.encode() if isinstance(value, str) else value:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


class StoredFP8Int8Linear(nn.Module):
    """Linear execution for restored FP8 or tensor-wise INT8 Kitchen weights."""

    def __init__(
        self,
        weight,
        bias: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(weight, QuantizedTensor) or weight.ndim != 2:
            raise TypeError("stored linear requires a 2D restored QuantizedTensor weight")
        if weight._layout_cls not in {"TensorCoreFP8Layout", "TensorWiseINT8Layout"}:
            raise ValueError(f"stored linear does not support {weight._layout_cls!r}")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("stored linear bias must match output features")
        if input_scale is not None and (
            input_scale.dtype != torch.float32
            or input_scale.ndim != 0
            or not bool(torch.isfinite(input_scale))
            or not bool(input_scale > 0)
        ):
            raise ValueError("stored linear input_scale must be one positive finite F32 scalar")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.input_scale = float(input_scale.item()) if input_scale is not None else None
        self.native_dispatch_count = 0
        self.dense_fallback_count = 0
        self.fallback_dispatch_count = 0
        self.native_rejection_count = 0
        self.int8_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("stored linear input feature count does not match its weight")
        original_shape = input.shape
        flat_input = input.reshape(-1, original_shape[-1])
        if self.weight._layout_cls == "TensorWiseINT8Layout":
            output = F.linear(flat_input, self.weight, self.bias)
            self.int8_dispatch_count += 1
        else:
            try:
                output = self._native_fp8_matmul(flat_input)
            except BaseException:
                self.native_rejection_count += 1
                raise
            self.native_dispatch_count += 1
        return output.reshape(*original_shape[:-1], self.weight.shape[0])

    def _native_fp8_matmul(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Invoke only the pinned Kitchen CUDA FP8 operations."""

        import comfy_kitchen as ck
        from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2
        from comfy_kitchen.tensor import QuantizedTensor

        if flat_input.device.type != "cuda":
            raise RuntimeError("stored FP8 native dispatch requires CUDA input")
        scale = (
            torch.tensor(self.input_scale, device=flat_input.device, dtype=torch.float32)
            if self.input_scale is not None
            else torch.clamp(
                torch.amax(flat_input.abs()).to(dtype=torch.float32)
                / torch.finfo(torch.float8_e4m3fn).max,
                min=1e-12,
            )
        )
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("stored FP8 weight lost its Kitchen wrapper")
        with ck.use_backend("cuda"):
            quantize = ck.registry.get_implementation(
                "quantize_per_tensor_fp8", backend="cuda"
            )
            qdata = quantize(flat_input, scale, torch.float8_e4m3fn)
            output = scaled_mm_v2(
                qdata,
                weight._qdata.t(),
                scale,
                weight.params.scale,
                out_dtype=flat_input.dtype,
            )
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def move_stored_storage(self, device: torch.device | str) -> None:
        """Move stored qdata, scale, and bias without changing representation."""

        from comfy_kitchen.tensor import QuantizedTensor

        target = torch.device(device)
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("stored linear weight is no longer a QuantizedTensor")
        if weight._qdata.dtype != weight.storage_dtype:
            raise RuntimeError("stored linear qdata dtype changed")
        params = dataclass_replace(weight.params, scale=weight.params.scale.to(device=target))
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        self._parameters["weight"] = nn.Parameter(restored, requires_grad=False)
        if self.bias is not None:
            self._parameters["bias"] = nn.Parameter(
                self.bias.to(device=target), requires_grad=False
            )


class _StoredLoraAdapter(nn.Module):
    """Additive low-rank branch beside one immutable base linear."""

    def __init__(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        alpha: float | None,
    ) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError("stored LoRA weights must be rank-2 tensors")
        if up.shape[1] != down.shape[0]:
            raise ValueError("stored LoRA up/down rank does not match")
        if not down.dtype.is_floating_point or not up.dtype.is_floating_point:
            raise ValueError("stored LoRA weights must use floating-point storage")
        rank = int(down.shape[0])
        if rank <= 0:
            raise ValueError("stored LoRA rank must be positive")
        self.down = nn.Parameter(down.contiguous(), requires_grad=False)
        self.up = nn.Parameter(up.contiguous(), requires_grad=False)
        self.patch_identity = (id(self.down), id(self.up))
        self.scale = 1.0 if alpha is None else float(alpha) / rank
        self.strength = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        down = self.down.to(dtype=input.dtype)
        up = self.up.to(dtype=input.dtype)
        return F.linear(F.linear(input, down), up) * (self.scale * self.strength)


class _AdditiveLoraMixin:
    _lora_adapters: nn.ModuleDict
    lora_dispatch_count: int
    weight: torch.Tensor

    def _initialize_lora(self) -> None:
        self._lora_adapters = nn.ModuleDict()
        self.lora_dispatch_count = 0
        self.patch_seed_key: str | None = None
        self.patched_resident_merge_misses = 0
        self.patched_resident_hits = 0
        self.signature_none_patch_rematerializations = 0
        self.patched_resident_writebacks = 0

    def add_lora_adapter(
        self,
        name: str,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        alpha: float | None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("stored LoRA adapter name is unsafe")
        if name in self._lora_adapters:
            raise ValueError(f"stored LoRA adapter {name!r} is already loaded")
        if down.shape[1] != self.weight.shape[1] or up.shape[0] != self.weight.shape[0]:
            raise ValueError("stored LoRA geometry differs from its target linear")
        self._lora_adapters[name] = _StoredLoraAdapter(down, up, alpha=alpha)

    def set_lora_strength(self, name: str, strength: float) -> None:
        try:
            adapter = self._lora_adapters[name]
        except KeyError as exc:
            raise KeyError(f"stored LoRA adapter {name!r} is not loaded") from exc
        value = float(strength)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("stored LoRA strength must be finite")
        adapter.strength = value

    def delete_lora_adapter(self, name: str) -> None:
        if name in self._lora_adapters:
            self._lora_adapters.pop(name)

    def _apply_lora(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        active = False
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            output = output + adapter(input)
            active = True
        if active:
            self.lora_dispatch_count += 1
        return output

    def _merge_lora_weight(
        self,
        weight: torch.Tensor,
        *,
        count_dispatch: bool = True,
    ) -> torch.Tensor:
        """Match Comfy's full-precision patch-before-linear rounding order."""

        active = False
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            down = adapter.down.to(device=weight.device, dtype=weight.dtype)
            up = adapter.up.to(device=weight.device, dtype=weight.dtype)
            weight = weight + torch.mm(up, down) * (adapter.scale * adapter.strength)
            active = True
        if active and count_dispatch:
            self.lora_dispatch_count += 1
        return weight

    def _active_patch_fingerprint(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                name,
                *adapter.patch_identity,
                float(adapter.scale),
                float(adapter.strength),
            )
            for name, adapter in self._lora_adapters.items()
            if adapter.strength != 0.0
        )

    def _merged_strict_comfy_weight(self) -> torch.Tensor:
        """Resolve Comfy's patched-resident full-precision weight contract."""

        weight = self.weight
        dense_weight = weight.dequantize()
        fingerprint = self._active_patch_fingerprint()
        resident_fingerprint = getattr(weight, _PATCH_FINGERPRINT_ATTRIBUTE, None)
        if not fingerprint:
            if resident_fingerprint is not None:
                raise RuntimeError(
                    "stored linear retained patched bytes after its patch state changed"
                )
            return dense_weight

        self.lora_dispatch_count += 1
        cacheable = bool(getattr(weight, _AIMDO_SIGNATURE_CACHEABLE_ATTRIBUTE, False))
        if resident_fingerprint is not None:
            if not cacheable or resident_fingerprint != fingerprint:
                raise RuntimeError(
                    "stored linear patch fingerprint changed on one resident value"
                )
            self.patched_resident_hits += 1
            return dense_weight

        merged = self._merge_lora_weight(dense_weight, count_dispatch=False)
        if not cacheable:
            self.signature_none_patch_rematerializations += 1
            return merged

        seed_key = self.patch_seed_key
        if not seed_key:
            raise RuntimeError("stored linear patched residency requires a stochastic seed key")
        requantized = weight.requantize_from_float(
            merged,
            scale="recalculate",
            stochastic_rounding=comfy_string_to_seed(seed_key),
        )
        with torch.no_grad():
            weight.copy_(requantized)
        setattr(weight, _PATCH_FINGERPRINT_ATTRIBUTE, fingerprint)
        self.patched_resident_merge_misses += 1
        self.patched_resident_writebacks += 1
        return merged


class StoredFP8Linear(_AdditiveLoraMixin, nn.Module):
    """Bias-free direct Kitchen FP8 linear with optional additive LoRAs."""

    def __init__(self, weight: Any, *, input_scale: torch.Tensor | None) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorCoreFP8Layout"
            or weight._qdata.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("stored FP8 linear requires TensorCore FP8 weight data")
        if input_scale is not None:
            _validate_positive_scalar(input_scale, "input_scale")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.input_scale = None if input_scale is None else float(input_scale.item())
        self.native_dispatch_count = 0
        self.rejected_dispatch_count = 0
        self.dense_fallback_count = 0
        self.last_dispatch_error: str | None = None
        self.execution_policy = "native_quantized_mm"
        self.full_precision_dispatch_count = 0
        self._initialize_lora()

    def set_execution_policy(self, policy: str) -> None:
        if policy not in {"native_quantized_mm", "strict_comfy_full_precision_mm"}:
            raise ValueError("stored FP8 execution policy is unsupported")
        self.execution_policy = policy
        if not hasattr(self, "full_precision_dispatch_count"):
            self.full_precision_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("stored FP8 input feature count does not match weight")
        original_shape = input.shape
        flat_input = input.reshape(-1, original_shape[-1])
        if getattr(self, "execution_policy", "native_quantized_mm") == (
            "strict_comfy_full_precision_mm"
        ):
            dense_weight = self._merged_strict_comfy_weight()
            dense_weight = dense_weight.to(device=input.device, dtype=input.dtype)
            output = F.linear(flat_input, dense_weight)
            self.full_precision_dispatch_count += 1
            output = output.reshape(*original_shape[:-1], self.weight.shape[0])
            return output
        if self.input_scale is None:
            scale = torch.amax(flat_input.abs()).to(dtype=torch.float32)
            scale = torch.clamp(scale / torch.finfo(torch.float8_e4m3fn).max, min=1e-12)
        else:
            scale = torch.tensor(self.input_scale, device=input.device, dtype=torch.float32)
        try:
            output = _direct_kitchen_fp8_linear(flat_input, self.weight, scale=scale)
        except BaseException as exc:
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "direct Kitchen FP8 dispatch failed; dense fallback is forbidden"
            ) from exc
        self.native_dispatch_count += 1
        self.last_dispatch_error = None
        output = output.reshape(*original_shape[:-1], self.weight.shape[0])
        return self._apply_lora(input, output)

    def move_stored_storage(self, device: torch.device | str) -> None:
        """Move FP8 qdata and its F32 scale without changing representation."""

        from comfy_kitchen.tensor import QuantizedTensor

        target = canonical_device(device)
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("stored FP8 weight is no longer a QuantizedTensor")
        if weight._qdata.dtype is not torch.float8_e4m3fn:
            raise RuntimeError("stored FP8 qdata dtype changed")
        params = dataclass_replace(weight.params, scale=weight.params.scale.to(device=target))
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        self._parameters["weight"] = nn.Parameter(restored, requires_grad=False)


class StoredNVFP4Linear(_AdditiveLoraMixin, nn.Module):
    """Bias-free direct Kitchen NVFP4 linear with optional additive LoRAs."""

    def __init__(self, weight: Any, *, input_scale: torch.Tensor | None) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorCoreNVFP4Layout"
            or weight._qdata.dtype is not torch.uint8
            or weight.params.block_scale.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("stored NVFP4 linear requires packed TensorCore NVFP4 data")
        if input_scale is not None:
            _validate_positive_scalar(input_scale, "input_scale")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.input_scale = None if input_scale is None else float(input_scale.item())
        self.native_dispatch_count = 0
        self.rejected_dispatch_count = 0
        self.dense_fallback_count = 0
        self.last_dispatch_error: str | None = None
        self.execution_policy = "native_quantized_mm"
        self.full_precision_dispatch_count = 0
        self._initialize_lora()

    def set_execution_policy(self, policy: str) -> None:
        if policy not in {"native_quantized_mm", "strict_comfy_full_precision_mm"}:
            raise ValueError("stored NVFP4 execution policy is unsupported")
        self.execution_policy = policy
        if not hasattr(self, "full_precision_dispatch_count"):
            self.full_precision_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("stored NVFP4 input feature count differs from weight")
        original_shape = input.shape
        flat = input.reshape(-1, original_shape[-1])
        if getattr(self, "execution_policy", "native_quantized_mm") == (
            "strict_comfy_full_precision_mm"
        ):
            dense_weight = self._merged_strict_comfy_weight()
            dense_weight = dense_weight.to(device=input.device, dtype=input.dtype)
            result = F.linear(flat, dense_weight)
            self.full_precision_dispatch_count += 1
            result = result.reshape(*original_shape[:-1], self.weight.shape[0])
            return result
        try:
            result = _direct_kitchen_nvfp4_linear(
                flat,
                self.weight,
                input_scale=self.input_scale,
                output_dtype=input.dtype,
            )
        except BaseException as exc:
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "direct Kitchen NVFP4 dispatch failed; dense fallback is forbidden"
            ) from exc
        result = result[: flat.shape[0], : self.weight.shape[0]]
        self.native_dispatch_count += 1
        self.last_dispatch_error = None
        result = result.reshape(*original_shape[:-1], self.weight.shape[0])
        return self._apply_lora(input, result)


class StoredDenseLoraLinear(_AdditiveLoraMixin, nn.Module):
    """Dense linear with the same additive LoRA contract as stored linears."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if type(base) is not nn.Linear:
            raise TypeError("dense LoRA wrapper requires an exact nn.Linear")
        self.base = base
        self._initialize_lora()

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._apply_lora(input, self.base(input))


def _validate_positive_scalar(value: torch.Tensor, name: str) -> None:
    if (
        value.dtype is not torch.float32
        or value.ndim != 0
        or not bool(torch.isfinite(value))
        or not bool(value > 0)
    ):
        raise ValueError(f"{name} must be one positive finite F32 scalar")


def _direct_kitchen_fp8_linear(
    input: torch.Tensor,
    weight: Any,
    *,
    scale: torch.Tensor,
) -> torch.Tensor:
    if input.device.type != "cuda":
        raise RuntimeError("direct FP8 dispatch requires CUDA input")
    import comfy_kitchen as ck
    from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

    with ck.use_backend("cuda"):
        quantize = ck.registry.get_implementation("quantize_per_tensor_fp8", backend="cuda")
        qdata = quantize(input, scale, torch.float8_e4m3fn)
        return scaled_mm_v2(
            qdata,
            weight._qdata.t(),
            scale,
            weight.params.scale,
            out_dtype=input.dtype,
        )


def _direct_kitchen_nvfp4_linear(
    input: torch.Tensor,
    weight: Any,
    *,
    input_scale: float | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

    if input.device.type != "cuda":
        raise RuntimeError("NVFP4 native dispatch requires CUDA input")
    if input_scale is None:
        scale = torch.amax(input.abs()).to(dtype=torch.float32)
        scale = torch.clamp(scale / (448.0 * 6.0), min=1e-12)
    else:
        scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    padded = TensorCoreNVFP4Layout.get_padded_shape(tuple(input.shape)) != tuple(input.shape)
    with ck.use_backend("cuda"):
        quantize = ck.registry.get_implementation("quantize_nvfp4", backend="cuda")
        native_mm = ck.registry.get_implementation("scaled_mm_nvfp4", backend="cuda")
        aqdata, block_scale_a = quantize(input, scale, pad_16x=padded)
        if aqdata.dtype is not torch.uint8:
            raise RuntimeError("NVFP4 activation did not remain packed U8")
        if not isinstance(weight, QuantizedTensor):
            raise TypeError("NVFP4 weight lost its QuantizedTensor wrapper")
        return native_mm(
            aqdata,
            weight._qdata,
            tensor_scale_a=scale,
            tensor_scale_b=weight.params.scale,
            block_scale_a=block_scale_a,
            block_scale_b=weight.params.block_scale,
            out_dtype=output_dtype,
        )
