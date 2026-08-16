"""Stored-only ConvRot modules for the Engine-owned Z-Image NextDiT shell.

This module intentionally has no external graph-runtime dependency. It accepts the exact
``TensorWiseINT8Layout`` restored from the official file and asks Kitchen for a
CUDA implementation directly.  There is no dequantize/dense branch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .z_image_stored_lora import apply_z_image_fixed_lora

Z_IMAGE_CONVROT_NATIVE_PRIMITIVE = "comfy_kitchen.registry/int8_linear@cuda"


def _dispatch_z_image_convrot_cuda(
    value: torch.Tensor,
    weight: object,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve and invoke Kitchen 0.2.28 without changing registry policy."""

    from comfy_kitchen import registry
    from comfy_kitchen.tensor import QuantizedTensor

    if not isinstance(weight, QuantizedTensor):
        raise TypeError("Z-Image ConvRot weight lost its Kitchen wrapper")
    if weight._qdata.dtype is not torch.int8 or weight.params.scale.dtype is not torch.float32:
        raise TypeError("Z-Image ConvRot dispatch requires INT8 weight and F32 scale")
    if weight.params.convrot is not True or weight.params.convrot_groupsize != 256:
        raise ValueError("Z-Image ConvRot dispatch requires groupsize 256")
    kwargs = {
        "x": value.contiguous(),
        "weight": weight._qdata.contiguous(),
        "weight_scale": weight.params.scale,
        "bias": bias,
        "out_dtype": value.dtype,
        "convrot": True,
        "convrot_groupsize": 256,
        "input_act": None,
    }
    with torch.cuda.device(value.device):
        implementation = registry.get_implementation(
            "int8_linear", backend="cuda", kwargs=kwargs
        )
        if not callable(implementation):
            raise TypeError("Kitchen INT8 CUDA resolver did not return a callable")
        result = implementation(**kwargs)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Kitchen INT8 CUDA implementation did not return a tensor")
    return result


def move_z_image_quantized_weights(
    model: nn.Module,
    device: torch.device | str,
    *,
    module_types: tuple[type[nn.Module], ...],
    poison_attribute: str,
    label: str,
) -> None:
    """Failure-atomically rebuild selected Kitchen weights on a target device."""

    from comfy_kitchen.tensor import QuantizedTensor

    target = torch.device(device)
    poisoned = getattr(model, poison_attribute, None)
    if poisoned:
        if target.type == "cpu" and all(value.device.type == "cpu" for value in model.parameters()):
            return
        raise RuntimeError(f"{label} residency is poisoned: {poisoned}")
    stored: list[tuple[nn.Module, object]] = []
    for module in model.modules():
        if isinstance(module, module_types):
            stored.append((module, module.weight))
            module._parameters["weight"] = None
    rebuilt: list[tuple[nn.Module, object]] = []
    try:
        model.to(device=target)
        for module, weight in stored:
            params = weight.params.to_device(target)
            moved = QuantizedTensor(weight._qdata.to(target), weight._layout_cls, params)
            if moved._qdata.device != target or moved.params.scale.device != target:
                raise RuntimeError(f"{label} residency changed stored layout")
            rebuilt.append((module, moved))
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            model.to(device="cpu")
        except Exception as candidate:  # noqa: BLE001 - cleanup must not mask the move error
            rollback_error = candidate
        finally:
            for module, weight in stored:
                module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
        setattr(
            model,
            poison_attribute,
            f"{type(exc).__name__}; "
            f"rollback={type(rollback_error).__name__ if rollback_error else 'ok'}",
        )
        if rollback_error is not None:
            raise RuntimeError(f"{label} residency rollback failed") from exc
        raise
    for module, weight in rebuilt:
        module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)


class ZImageStoredConvRotLinear(nn.Module):
    """Stored INT8 ConvRot linear with an optional exact F32 bias seam."""

    def __init__(self, weight: object, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorWiseINT8Layout"
            or weight._qdata.dtype is not torch.int8
            or weight.params.convrot is not True
        ):
            raise TypeError("Z-Image ConvRot module requires stored TensorWise INT8 data")
        if bias is not None and (
            bias.dtype is not torch.float32
            or bias.ndim != 1
            or tuple(bias.shape) != (weight.shape[0],)
        ):
            raise TypeError("Z-Image ConvRot bias must be exact F32 [out_features]")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.native_dispatch_count = 0
        self.rejected_dispatch_count = 0
        # This exists solely as an auditable invariant.  It is never incremented
        # because this implementation has no dense fallback path.
        self.dense_fallback_count = 0
        self.last_dispatch_error: str | None = None
        # Fixed Z LoRAs remain independent BF16 branches. The authoritative
        # Kitchen weight above is never replaced, merged, or dequantized.
        self._fixed_lora_branches = nn.ModuleDict()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("Z-Image ConvRot input feature count differs from weight")
        flat = input.reshape(-1, input.shape[-1])
        if flat.device.type != "cuda":
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = "RuntimeError: native dispatch requires CUDA input"
            raise RuntimeError(
                "Z-Image direct Kitchen INT8 ConvRot dispatch failed; dense fallback is forbidden"
            )
        result = self._dispatch_flat(flat)
        result = result.reshape(*input.shape[:-1], self.weight.shape[0])
        return apply_z_image_fixed_lora(self, input, result)

    def _dispatch_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """Count one exact registry dispatch; split out for bounded failure injection."""

        try:
            weight = self.weight
            if flat.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError("Z-Image ConvRot requires F32, F16, or BF16 activations")
            result = _dispatch_z_image_convrot_cuda(flat, weight, self.bias)
        except BaseException as exc:
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "Z-Image direct Kitchen INT8 ConvRot dispatch failed; dense fallback is forbidden"
            ) from exc
        self.native_dispatch_count += 1
        self.last_dispatch_error = None
        return result


def z_image_convrot_dispatch_snapshot(model: nn.Module) -> dict[str, tuple[int, int, int]]:
    """Capture the exact materialized ConvRot module set and its counters."""

    expected = getattr(model, "_latentslate_z_image_convrot_modules", None)
    if not isinstance(expected, tuple) or len(expected) != 202:
        raise ValueError("Z-Image ConvRot module closure is incomplete")
    actual = {
        name: (
            module.native_dispatch_count,
            module.rejected_dispatch_count,
            module.dense_fallback_count,
        )
        for name, module in model.named_modules()
        if isinstance(module, ZImageStoredConvRotLinear)
    }
    if tuple(actual) != expected or len(actual) != 202:
        raise ValueError("Z-Image ConvRot module set differs from its exact materialization")
    biased = tuple(
        name for name in expected if getattr(model.get_submodule(name), "bias", None) is not None
    )
    expected_biased = tuple(name for name in expected if name.endswith("adaLN_modulation.0"))
    if biased != expected_biased:
        raise ValueError("Z-Image ConvRot bias closure differs from the exact adaLN mapping")
    for name in biased:
        module = model.get_submodule(name)
        bias = module.bias
        if (
            not isinstance(bias, nn.Parameter)
            or bias.dtype is not torch.float32
            or bias.ndim != 1
            or tuple(bias.shape) != (module.weight.shape[0],)
        ):
            raise ValueError(f"Z-Image ConvRot bias differs from stored F32 contract: {name}")
    if any(
        any(not isinstance(value, int) or value < 0 for value in counts)
        for counts in actual.values()
    ):
        raise ValueError("Z-Image ConvRot dispatch counter is invalid")
    return actual


def verify_z_image_convrot_dispatch(
    model: nn.Module, before: Mapping[str, tuple[int, int, int]]
) -> dict[str, int | str | bool]:
    """Prove every one of the 202 modules used direct INT8 Kitchen dispatch."""

    after = z_image_convrot_dispatch_snapshot(model)
    if tuple(before) != tuple(after):
        raise RuntimeError("Z-Image ConvRot module set changed during dispatch")
    native = {name: after[name][0] - before[name][0] for name in after}
    rejected = sum(after[name][1] - before[name][1] for name in after)
    fallback = sum(after[name][2] - before[name][2] for name in after)
    missing = [name for name, delta in native.items() if delta <= 0]
    if missing or rejected or fallback:
        raise RuntimeError(
            "Z-Image direct Kitchen INT8 ConvRot proof failed: "
            f"modules={len(after) - len(missing)}/{len(after)}, rejected={rejected}, "
            f"dense_fallback={fallback}"
        )
    values = tuple(native.values())
    return {
        "status": "proven",
        "backend": f"comfy-kitchen/cuda/{Z_IMAGE_CONVROT_NATIVE_PRIMITIVE}",
        "native_primitive": Z_IMAGE_CONVROT_NATIVE_PRIMITIVE,
        "module_count": len(values),
        "dispatched_modules": len(values),
        "complete": True,
        "total_dispatch_delta": sum(values),
        "min_module_dispatch_delta": min(values),
        "max_module_dispatch_delta": max(values),
        "rejected_dispatch_count": rejected,
        "dense_fallback_count": fallback,
    }


def move_z_image_nextdit_storage(model: nn.Module, device: torch.device | str) -> None:
    """Move dense state and rebuild every stored wrapper on the target device."""

    from comfy_kitchen.tensor import QuantizedTensor

    target = torch.device(device)
    poisoned = getattr(model, "_latentslate_z_image_residency_poisoned", None)
    if poisoned:
        if target.type == "cpu" and all(value.device.type == "cpu" for value in model.parameters()):
            return
        raise RuntimeError(f"Z-Image NextDiT residency is poisoned: {poisoned}")
    stored: list[tuple[ZImageStoredConvRotLinear, object, torch.Tensor | None]] = []
    for module in model.modules():
        if isinstance(module, ZImageStoredConvRotLinear):
            stored.append((module, module.weight, module.bias))
            module._parameters["weight"] = None
            module._parameters["bias"] = None
    rebuilt: list[tuple[ZImageStoredConvRotLinear, object, torch.Tensor | None]] = []
    try:
        model.to(device=target)
        for module, weight, bias in stored:
            params = weight.params.to_device(target)
            moved_weight = QuantizedTensor(weight._qdata.to(target), weight._layout_cls, params)
            moved_bias = None if bias is None else bias.detach().to(target)
            if (
                moved_weight._qdata.device != target
                or moved_weight.params.scale.device != target
                or moved_weight.params.convrot is not True
            ):
                raise RuntimeError("Z-Image ConvRot residency move changed stored layout")
            rebuilt.append((module, moved_weight, moved_bias))
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            model.to(device="cpu")
        except Exception as candidate:  # noqa: BLE001 - best-effort rollback must not mask the move error
            rollback_error = candidate
        finally:
            for module, weight, bias in stored:
                module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
                module._parameters["bias"] = bias
        model._latentslate_z_image_residency_poisoned = (
            f"{type(exc).__name__}; "
            f"rollback={type(rollback_error).__name__ if rollback_error else 'ok'}"
        )
        if rollback_error is not None:
            raise RuntimeError("Z-Image NextDiT residency rollback failed") from exc
        raise
    for module, weight, bias in rebuilt:
        module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
        module._parameters["bias"] = (
            None if bias is None else nn.Parameter(bias, requires_grad=False)
        )


class ZImageNextDiTStage:
    def __init__(self, model: nn.Module, execution_device: torch.device | str) -> None:
        self.model = model
        self.execution_device = torch.device(execution_device)
        self._before: dict[str, tuple[int, int, int]] | None = None

    def onload(self) -> None:
        move_z_image_nextdit_storage(self.model, self.execution_device)
        self._before = z_image_convrot_dispatch_snapshot(self.model)

    def verify_dispatch(self) -> dict[str, int | str | bool]:
        if self._before is None:
            raise RuntimeError("Z-Image NextDiT was not staged")
        return verify_z_image_convrot_dispatch(self.model, self._before)

    def offload(self) -> None:
        move_z_image_nextdit_storage(self.model, "cpu")
        self._before = None
