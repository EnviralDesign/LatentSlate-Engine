"""CPU/meta-only adapter planning for stored-quantized Comfy Wan 2.2 weights.

The mapping is intentionally local to the pinned Diffusers revision in
``pyproject.toml``.  It maps Comfy's native Wan topology to the public
``WanTransformer3DModel`` module layout without loading a checkpoint.  A future
loader may restore one stored quantized layer at a time only after this plan has
validated complete parameter coverage.

``NativeStoredLinear`` never quantizes or converts weights.  Its FP8 path may
quantize a *runtime activation* to execute a stored FP8 weight; that transient
compute operation is distinct from model-weight conversion and is never saved.

Stored quant is deliberately limited to Diffusers block-level group offload.
The Engine owns those synchronous block moves directly: Diffusers/Accelerate
group hooks must never be used because their meta reconstruction does not move a
third-party ``QuantizedTensor``'s internal storage. This CPU/meta plan does not
prove a CUDA group-offload runtime; that needs a separate device-residency
validation first.
"""

from __future__ import annotations

import json
import struct
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import _MAX_HEADER_BYTES, ArtifactIdentity, probe_safetensors

# This schema is the official I2V 14B config staged with the resources.  The
# model implementation is pinned by the Diffusers Git revision in pyproject.toml.
WAN22_14B_I2V_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "in_channels": 36,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 13824,
        "num_layers": 40,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
    }
)

_PREFIX = "model.diffusion_model."
_QUANT_SUFFIXES = (".comfy_quant", ".weight_scale", ".scale_weight", ".scale_input")
_FOREIGN_COMPONENT_PREFIXES = ("vae.",)
_LEGACY_QUANT_SENTINELS = ("scaled_fp8",)
_SUPPORTED_ARTIFACT_CONTRACTS = frozenset(
    {
        "comfy_quant/float8_e4m3fn",
        "comfy_legacy/scaled_fp8_e4m3fn",
        "comfy_quant/int8_tensorwise_convrot",
    }
)
SUPPORTED_STORED_QUANT_OFFLOAD_MODES = frozenset({"group_block"})
BlockModules: TypeAlias = Mapping[str, nn.Module]


@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    """One source parameter whose shape differs from Diffusers' meta skeleton."""

    source_key: str
    target_key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WanStoredAdapterPlan:
    """A fail-closed header-only mapping plan for one Comfy transformer artifact."""

    identity: ArtifactIdentity
    artifact_contract: str
    source_to_target: Mapping[str, str]
    quant_auxiliary: tuple[str, ...]
    foreign_component_extras: tuple[str, ...]
    unexpected_extras: tuple[str, ...]
    missing_targets: tuple[str, ...]
    duplicate_targets: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]

    @property
    def available(self) -> bool:
        """Whether every required Diffusers parameter maps exactly once."""

        return not (
            self.unexpected_extras
            or self.missing_targets
            or self.duplicate_targets
            or self.shape_mismatches
        )

    @property
    def errors(self) -> tuple[str, ...]:
        """Short actionable reasons that prevent a future layer loader."""

        errors: list[str] = []
        if self.missing_targets:
            errors.append(f"missing Diffusers parameters: {len(self.missing_targets)}")
        if self.duplicate_targets:
            errors.append(f"duplicate mapped parameters: {len(self.duplicate_targets)}")
        if self.shape_mismatches:
            errors.append(f"parameter shape mismatches: {len(self.shape_mismatches)}")
        if self.unexpected_extras:
            errors.append(f"unrecognized non-auxiliary source keys: {len(self.unexpected_extras)}")
        return tuple(errors)

    def require_available(self) -> None:
        """Raise before any future tensor materialization when the plan has gaps."""

        if not self.available:
            raise ValueError("Wan stored adapter unavailable: " + "; ".join(self.errors))


def build_wan_transformer_skeleton(
    config: Mapping[str, Any] = WAN22_14B_I2V_CONFIG,
) -> nn.Module:
    """Construct a Diffusers Wan skeleton with all parameter storage on ``meta``."""

    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        return WanTransformer3DModel(**dict(config))


def plan_comfy_wan_transformer(
    artifact_path: Path,
    config: Mapping[str, Any] = WAN22_14B_I2V_CONFIG,
) -> WanStoredAdapterPlan:
    """Map and validate a Comfy Wan checkpoint header without reading payloads."""

    source = Path(artifact_path).resolve(strict=True)
    probe = probe_safetensors(source)
    if probe.quantization_contract not in _SUPPORTED_ARTIFACT_CONTRACTS:
        raise ValueError(
            "Wan stored adapter: unsupported artifact contract "
            f"{probe.quantization_contract!r}; a supported stored-quant artifact is required"
        )
    header = _read_safetensors_header(source, probe.identity.size_bytes)
    skeleton = build_wan_transformer_skeleton(config)
    expected_shapes = {name: tuple(value.shape) for name, value in skeleton.state_dict().items()}

    source_to_target: dict[str, str] = {}
    quant_auxiliary: list[str] = []
    foreign_component_extras: list[str] = []
    unexpected_extras: list[str] = []
    target_sources: dict[str, list[str]] = {}
    shape_mismatches: list[ShapeMismatch] = []

    for raw_key, entry in header.items():
        if raw_key == "__metadata__":
            continue
        key = _normalize_comfy_key(raw_key)
        if key in _LEGACY_QUANT_SENTINELS:
            quant_auxiliary.append(raw_key)
            continue
        if _is_quant_auxiliary(key):
            if _quant_auxiliary_target(key) is None:
                unexpected_extras.append(raw_key)
            else:
                quant_auxiliary.append(raw_key)
            continue
        if key.startswith(_FOREIGN_COMPONENT_PREFIXES):
            foreign_component_extras.append(raw_key)
            continue
        target = map_comfy_wan_parameter_key(key)
        if target is None:
            unexpected_extras.append(raw_key)
            continue
        source_to_target[raw_key] = target
        target_sources.setdefault(target, []).append(raw_key)
        expected_shape = expected_shapes.get(target)
        source_shape = _entry_shape(entry, raw_key)
        if expected_shape is None:
            unexpected_extras.append(raw_key)
        elif source_shape != expected_shape:
            shape_mismatches.append(ShapeMismatch(raw_key, target, source_shape, expected_shape))

    duplicate_targets = tuple(sorted(target for target, sources in target_sources.items() if len(sources) != 1))
    missing_targets = tuple(sorted(set(expected_shapes) - set(target_sources)))
    return WanStoredAdapterPlan(
        identity=probe.identity,
        artifact_contract=probe.quantization_contract,
        source_to_target=MappingProxyType(dict(sorted(source_to_target.items()))),
        quant_auxiliary=tuple(sorted(quant_auxiliary)),
        foreign_component_extras=tuple(sorted(foreign_component_extras)),
        unexpected_extras=tuple(sorted(set(unexpected_extras))),
        missing_targets=missing_targets,
        duplicate_targets=duplicate_targets,
        shape_mismatches=tuple(sorted(shape_mismatches, key=lambda item: item.source_key)),
    )


def map_comfy_wan_parameter_key(source_key: str) -> str | None:
    """Map one normalized Comfy Wan parameter name to the pinned Diffusers layout."""

    key = _normalize_comfy_key(source_key)
    top_level = {
        "head.modulation": "scale_shift_table",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    }
    if key in top_level:
        return top_level[key]
    replacements = (
        (".modulation", ".scale_shift_table"),
        (".self_attn.q.", ".attn1.to_q."),
        (".self_attn.k.", ".attn1.to_k."),
        (".self_attn.v.", ".attn1.to_v."),
        (".self_attn.o.", ".attn1.to_out.0."),
        (".self_attn.norm_q.", ".attn1.norm_q."),
        (".self_attn.norm_k.", ".attn1.norm_k."),
        (".cross_attn.q.", ".attn2.to_q."),
        (".cross_attn.k.", ".attn2.to_k."),
        (".cross_attn.v.", ".attn2.to_v."),
        (".cross_attn.o.", ".attn2.to_out.0."),
        (".cross_attn.norm_q.", ".attn2.norm_q."),
        (".cross_attn.norm_k.", ".attn2.norm_k."),
        (".norm3.", ".norm2."),
        (".ffn.0.", ".ffn.net.0.proj."),
        (".ffn.2.", ".ffn.net.2."),
    )
    if not key.startswith("blocks."):
        return None
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    return None


class NativeStoredLinear(nn.Module):
    """A linear module backed by one already-restored Comfy ``QuantizedTensor``.

    Quantized weights and biases are frozen ``nn.Parameter`` objects so planned
    block-group residency accounting sees them. The optional activation scale is
    stored as an immutable Python float so a module dtype move cannot downcast it.
    """

    def __init__(
        self,
        weight,
        bias: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(weight, QuantizedTensor) or weight.ndim != 2:
            raise TypeError("NativeStoredLinear requires a 2D restored QuantizedTensor weight")
        if weight._layout_cls not in {"TensorCoreFP8Layout", "TensorWiseINT8Layout"}:
            raise ValueError(f"NativeStoredLinear does not support {weight._layout_cls!r}")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("NativeStoredLinear bias must match output features")
        if input_scale is not None and (
            input_scale.dtype != torch.float32
            or input_scale.ndim != 0
            or not bool(torch.isfinite(input_scale))
            or not bool(input_scale > 0)
        ):
            raise ValueError("NativeStoredLinear input_scale must be one positive finite F32 scalar")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.input_scale = float(input_scale.item()) if input_scale is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute with native kitchen layouts; only FP8 activations are transiently quantized."""

        if input.shape[-1] != self.weight.shape[1]:
            raise ValueError("NativeStoredLinear input feature count does not match stored weight")
        if self.weight._layout_cls == "TensorWiseINT8Layout":
            return F.linear(input, self.weight, self.bias)
        activation = _quantize_fp8_activation(input, self.input_scale)
        return F.linear(activation, self.weight, self.bias)


class SynchronousBlockResidencyManager:
    """Engine-owned, non-reentrant residency for explicit transformer blocks.

    Every pre-hook moves the whole block to the execution device with
    ``module.to``. Every post-hook is registered with ``always_call=True`` and
    moves the whole block back to the offload device, including after a failed
    forward. Outputs are deliberately not moved during cleanup, so they stay on
    the execution device for the next block.

    This does not use Diffusers or Accelerate hooks. A tiny CUDA residency proof
    covers this primitive, but full-model stored-quant generation remains
    unproven and unavailable.
    """

    def __init__(
        self,
        blocks: BlockModules,
        *,
        onload_device: torch.device | str,
        offload_device: torch.device | str = "cpu",
    ) -> None:
        if not blocks:
            raise ValueError("stored-quant block residency requires explicit blocks")
        ordered = OrderedDict(blocks)
        if len({id(block) for block in ordered.values()}) != len(ordered):
            raise ValueError("stored-quant block residency does not permit duplicate block modules")
        if not all(isinstance(name, str) and name and isinstance(block, nn.Module) for name, block in ordered.items()):
            raise TypeError("stored-quant block residency requires named nn.Module blocks")
        self._blocks = ordered
        self.onload_device = torch.device(onload_device)
        self.offload_device = torch.device(offload_device)
        self._handles: list[Any] = []
        self._active_name: str | None = None
        self._closed = False
        self._failed_reason: str | None = None
        self._transitioning = False
        self._lock = threading.RLock()

    @property
    def attached(self) -> bool:
        """Whether this manager currently owns hooks on its explicit blocks."""

        with self._lock:
            return bool(self._handles)

    @property
    def active_block(self) -> str | None:
        """The one block executing synchronously, if any."""

        with self._lock:
            return self._active_name

    def attach(self) -> None:
        """Attach paired pre/post hooks exactly once."""

        with self._lock:
            if self._closed:
                raise RuntimeError("stored-quant block residency manager is closed")
            if self._failed_reason:
                raise RuntimeError(f"stored-quant block residency manager failed: {self._failed_reason}")
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            if self._handles:
                raise RuntimeError("stored-quant block residency hooks are already attached")
            for name, block in self._blocks.items():
                self._handles.append(block.register_forward_pre_hook(self._make_pre_hook(name)))
                self._handles.append(block.register_forward_hook(self._make_post_hook(name), always_call=True))

    def force_offload(self) -> None:
        """Synchronously move every managed block to the configured offload device."""

        with self._lock:
            if self._active_name is not None:
                raise RuntimeError("cannot force offload while a stored-quant block is active")
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
        try:
            self._offload_all_blocks()
        finally:
            with self._lock:
                self._transitioning = False

    def _offload_all_blocks(self) -> None:
        errors: list[str] = []
        for name, block in self._blocks.items():
            try:
                block.to(self.offload_device)
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure for fail-closed state
                errors.append(f"{name}: {exc}")
        if errors:
            reason = "force-offload failed: " + "; ".join(errors)
            with self._lock:
                self._failed_reason = reason
            raise RuntimeError(reason)

    def remove(self, *, force_offload: bool = True) -> None:
        """Remove hooks and, by default, synchronously return all blocks to CPU."""

        with self._lock:
            if self._active_name is not None:
                raise RuntimeError("cannot remove stored-quant residency while a block is active")
            if self._closed:
                return
            if self._transitioning:
                raise RuntimeError("stored-quant block residency transition is in progress")
            self._transitioning = True
            handles = tuple(self._handles)
        try:
            for handle in handles:
                handle.remove()
            if force_offload:
                self._offload_all_blocks()
        finally:
            with self._lock:
                self._handles.clear()
                self._closed = True
                self._transitioning = False

    def _make_pre_hook(self, name: str):
        def pre_hook(module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            with self._lock:
                if self._closed or self._failed_reason or self._transitioning:
                    raise RuntimeError("stored-quant block residency is unavailable")
                if self._active_name is not None:
                    self._failed_reason = f"non-reentrant block execution: {self._active_name} -> {name}"
                    raise RuntimeError("stored-quant block residency is non-reentrant")
                self._active_name = name
            try:
                module.to(self.onload_device)
            except Exception as exc:
                reason = f"onload failed for {name}: {exc}"
                with self._lock:
                    self._active_name = None
                    self._failed_reason = reason
                raise RuntimeError(reason) from exc

        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            try:
                module.to(self.offload_device)
            except Exception as exc:
                reason = f"offload failed for {name}: {exc}"
                with self._lock:
                    self._failed_reason = reason
                raise RuntimeError(reason) from exc
            finally:
                with self._lock:
                    self._active_name = None
            return output

        return post_hook


def attach_native_stored_linear(
    parent: nn.Module,
    attribute: str,
    weight,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
) -> NativeStoredLinear:
    """Replace one ``nn.Linear`` child with a stored-quant wrapper, fail-closed."""

    if not isinstance(getattr(parent, attribute, None), nn.Linear):
        raise TypeError(f"{attribute!r} is not an nn.Linear child")
    replacement = NativeStoredLinear(weight, bias, input_scale)
    setattr(parent, attribute, replacement)
    return replacement


def validate_stored_quant_offload_mode(mode: str) -> str:
    """Require the sole planned stored-quant residency mode.

    ``group_block`` means Engine-owned synchronous whole-block moves only; it must
    never select Diffusers or Accelerate group hooks. This function does not imply
    that stored-quant CUDA generation is available today.
    """

    if mode not in SUPPORTED_STORED_QUANT_OFFLOAD_MODES:
        raise ValueError(
            "stored quant requires Engine-owned block-level group offload "
            "(offload='group_block'); sequential/meta, leaf-level, whole-model, and disk modes are unsupported"
        )
    return mode


def _quantize_fp8_activation(input: torch.Tensor, input_scale: float | None):
    """Make a temporary FP8 activation; this never mutates or converts weights."""

    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    if input_scale is None:
        scale = input.detach().abs().amax().to(dtype=torch.float32) / torch.finfo(torch.float8_e4m3fn).max
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    else:
        scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    qdata, params = TensorCoreFP8Layout.quantize(input, scale=scale, dtype=torch.float8_e4m3fn)
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _read_safetensors_header(path: Path, size_bytes: int) -> dict[str, Any]:
    """Read the already structurally-probed header, never any tensor payload."""

    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("Wan adapter: SafeTensors header is truncated")
        length = struct.unpack("<Q", raw_length)[0]
        if length > _MAX_HEADER_BYTES or length > size_bytes - 8:
            raise ValueError("Wan adapter: SafeTensors header exceeds bounds")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("Wan adapter: SafeTensors header is truncated")
    parsed = json.loads(raw_header)
    if not isinstance(parsed, dict):
        raise TypeError("Wan adapter: SafeTensors header must be an object")
    return parsed


def _normalize_comfy_key(key: str) -> str:
    return key.removeprefix(_PREFIX)


def _is_quant_auxiliary(key: str) -> bool:
    return key.endswith(_QUANT_SUFFIXES)


def _quant_auxiliary_target(key: str) -> str | None:
    for suffix in _QUANT_SUFFIXES:
        if key.endswith(suffix):
            target = map_comfy_wan_parameter_key(key.removesuffix(suffix) + ".weight")
            return target if target and target.endswith(".weight") else None
    return None


def _entry_shape(entry: Any, key: str) -> tuple[int, ...]:
    if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
        raise TypeError(f"Wan adapter: invalid source shape for {key!r}")
    shape = tuple(entry["shape"])
    if not all(isinstance(item, int) and item >= 0 for item in shape):
        raise ValueError(f"Wan adapter: invalid source shape for {key!r}")
    return shape
