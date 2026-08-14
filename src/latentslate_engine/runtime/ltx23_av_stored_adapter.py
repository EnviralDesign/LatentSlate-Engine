"""Clean-room stored-FP8 seam for the LTX 2.3 joint audio/video transformer.

This module owns checkpoint structure, materialization, and native linear dispatch.
It builds the functional pinned Diffusers transformer and connector classes on the
meta device, then populates them directly from the combined checkpoint. Engine adds
no second transformer forward implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from torch import nn

from .signatures import path_signature

LTX23AVVariant = Literal["dev", "distilled"]

LTX23_AV_DIFFUSERS_RUNTIME_REVISION = "f53d552036a0d1bd5570782a39cd40cfabf112bc"
LTX23_AV_DIFFUSERS_MAPPING_REVISION = "788e80206a9722b133fc4907bbb8da2ba26d5181"

_PREFIX = "model.diffusion_model."
_SCALE_SUFFIXES = (".weight_scale", ".input_scale")
_EXPECTED_STATE_KEYS = 4_444
_EXPECTED_TRANSFORMER_STATE_KEYS = 4_186
_EXPECTED_CONNECTOR_STATE_KEYS = 258
_EXPECTED_EXTERNAL_CONNECTOR_STATE_KEYS = 4
_EXPECTED_LINEARS = 1_660
_EXPECTED_FP8 = {"dev": 1_496, "distilled": 1_462}
_EXPECTED_STATE_DTYPES = {
    "dev": {"BF16": 2_658, "F8_E4M3": 1_496, "F32": 290},
    "distilled": {"BF16": 2_692, "F8_E4M3": 1_462, "F32": 290},
}
_EXPECTED_LORA_TARGETS = 1_660
_EXPECTED_MISSING_ALPHA = "time_embed.linear"
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

_CONNECTOR_PREFIXES = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "text_embedding_projection",
    "connectors.",
    "video_connector",
    "audio_connector",
    "text_proj_in",
)
_TRANSFORMER_RENAMES = (
    ("patchify_proj", "proj_in"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)
_CONNECTOR_RENAMES = (
    ("connectors.", ""),
    ("video_embeddings_connector", "video_connector"),
    ("audio_embeddings_connector", "audio_connector"),
    ("transformer_1d_blocks", "transformer_blocks"),
    ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
    ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)

_DIFFUSERS_LTX23_CONFIG: dict[str, Any] = {
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "cross_attention_dim": 4096,
    "vae_scale_factors": (8, 32, 32),
    "pos_embed_max_pos": 20,
    "base_height": 2048,
    "base_width": 2048,
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_in_channels": 128,
    "audio_out_channels": 128,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16_000,
    "audio_hop_length": 160,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "num_layers": 48,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 3840,
    "attention_bias": True,
    "attention_out_bias": True,
    "rope_theta": 10_000.0,
    "rope_double_precision": True,
    "causal_offset": 1,
    "timestep_scale_multiplier": 1000,
    "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split",
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

_DIFFUSERS_LTX23_CONNECTOR_CONFIG: dict[str, Any] = {
    "caption_channels": 3840,
    "text_proj_in_factor": 49,
    "video_connector_num_attention_heads": 32,
    "video_connector_attention_head_dim": 128,
    "video_connector_num_layers": 8,
    "video_connector_num_learnable_registers": 128,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 32,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_layers": 8,
    "audio_connector_num_learnable_registers": 128,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10_000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 4096,
    "audio_hidden_dim": 2048,
    "proj_bias": True,
}


@dataclass(frozen=True, slots=True)
class LTX23AVStateSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LTX23AVMappedStateSpec:
    source_key: str
    target_key: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LTX23AVLinearSpec:
    module_name: str
    weight: LTX23AVStateSpec
    bias: LTX23AVStateSpec
    quantized: bool
    source_weight_key: str
    source_bias_key: str
    source_weight_scale_key: str | None
    source_input_scale_key: str | None


@dataclass(frozen=True, slots=True)
class LTX23AVArtifactContract:
    path: Path
    variant: LTX23AVVariant
    artifact_signature: dict[str, Any]
    header_fingerprint: str
    state: tuple[LTX23AVStateSpec, ...]
    transformer_state: tuple[LTX23AVMappedStateSpec, ...]
    connector_state: tuple[LTX23AVMappedStateSpec, ...]
    external_connector_state: tuple[LTX23AVMappedStateSpec, ...]
    linears: tuple[LTX23AVLinearSpec, ...]

    @property
    def quantized_linear_count(self) -> int:
        return sum(item.quantized for item in self.linears)

    @property
    def dense_linear_count(self) -> int:
        return len(self.linears) - self.quantized_linear_count


@dataclass(frozen=True, slots=True)
class LTX23AVMaterializationPlan:
    contract: LTX23AVArtifactContract
    shell_type: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class LTX23ConnectorMaterializationPlan:
    contract: LTX23AVArtifactContract
    shell_type: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class LTX23AVLoraTargetSpec:
    module_name: str
    down_key: str
    up_key: str
    alpha_key: str | None
    rank: int
    # A missing alpha has the conventional alpha=rank meaning, hence 1.0.
    alpha_over_rank: float


@dataclass(frozen=True, slots=True)
class LTX23AVLoraContract:
    path: Path
    artifact_signature: dict[str, Any]
    header_fingerprint: str
    targets: tuple[LTX23AVLoraTargetSpec, ...]


@dataclass(frozen=True, slots=True)
class LTX23AVLoraInstallation:
    adapter_name: str
    contract_fingerprint: str
    modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LTX23StorageSlot:
    module: nn.Module
    name: str
    parameter: bool
    cpu_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class LTX23ModuleStorage:
    """Exact immutable CPU state for one independently resident module group.

    QuantizedTensor parameters are counted and copied through their physical
    qdata/sidecar tensors.  The logical wrapper is deliberately not counted a
    second time.  Restoring a group rebinds these original CPU objects, so a
    streamed forward never performs a device-to-host weight copy.
    """

    slots: tuple[_LTX23StorageSlot, ...]
    physical_bytes: int

    def copy_to(self, device: torch.device | str) -> LTX23ModuleBinding:
        target = torch.device(device)
        copies: dict[int, torch.Tensor] = {}
        values: list[torch.Tensor] = []
        for slot in self.slots:
            key = id(slot.cpu_value)
            if key not in copies:
                copies[key] = _copy_storage_value(slot.cpu_value, target)
            values.append(copies[key])
        copied = tuple(values)
        return LTX23ModuleBinding(self, copied, target)

    def restore_cpu(self) -> None:
        for slot in self.slots:
            _assign_storage_slot(slot, slot.cpu_value)


@dataclass(slots=True)
class LTX23ModuleBinding:
    storage: LTX23ModuleStorage
    values: tuple[torch.Tensor, ...]
    device: torch.device
    active: bool = False

    def activate(self) -> None:
        if self.active:
            raise RuntimeError("LTX module storage binding is already active")
        if len(self.values) != len(self.storage.slots):
            raise RuntimeError("LTX module storage binding is incomplete")
        assigned: list[_LTX23StorageSlot] = []
        try:
            for slot, value in zip(self.storage.slots, self.values, strict=True):
                _assign_storage_slot(slot, value)
                assigned.append(slot)
        except BaseException:
            for slot in assigned:
                _assign_storage_slot(slot, slot.cpu_value)
            raise
        self.active = True

    def restore_cpu(self) -> None:
        self.storage.restore_cpu()
        self.active = False


def capture_ltx23_module_storage(
    module: nn.Module,
    *,
    exclude_children: frozenset[str] = frozenset(),
) -> LTX23ModuleStorage:
    """Capture exact CPU parameter/buffer slots without double-counting storage."""

    slots: list[_LTX23StorageSlot] = []
    physical: dict[tuple[int, int, int, torch.dtype], int] = {}
    excluded_ids = {
        id(nested)
        for name, child in module.named_children()
        if name in exclude_children
        for nested in child.modules()
    }
    for nested in module.modules():
        if id(nested) in excluded_ids:
            continue
        for parameter, values in ((True, nested._parameters), (False, nested._buffers)):
            for name, value in values.items():
                if value is None:
                    continue
                if value.is_meta or value.device.type != "cpu":
                    raise ValueError("LTX module storage capture requires materialized CPU state")
                slots.append(_LTX23StorageSlot(nested, name, parameter, value))
                for tensor in _physical_storage_tensors(value):
                    key = _physical_tensor_key(tensor)
                    physical.setdefault(key, tensor.numel() * tensor.element_size())
    if not slots:
        raise ValueError("LTX module storage capture found no materialized state")
    return LTX23ModuleStorage(tuple(slots), sum(physical.values()))


def ltx23_module_physical_bytes(module: nn.Module) -> int:
    """Count materialized physical state while ignoring intentional meta shells."""

    physical: dict[tuple[int, int, int, torch.dtype], int] = {}
    for nested in module.modules():
        for values in (nested._parameters, nested._buffers):
            for value in values.values():
                if value is None or value.is_meta:
                    continue
                for tensor in _physical_storage_tensors(value):
                    key = _physical_tensor_key(tensor)
                    physical.setdefault(key, tensor.numel() * tensor.element_size())
    return sum(physical.values())


def _copy_storage_value(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    qdata = getattr(value, "_qdata", None)
    params = getattr(value, "params", None)
    tensor_fields = getattr(params, "_tensor_fields", None)
    if isinstance(qdata, torch.Tensor) and callable(tensor_fields):
        from comfy_kitchen.tensor import QuantizedTensor

        replacements = {
            field: getattr(params, field).to(device=device) for field in tensor_fields()
        }
        restored = QuantizedTensor(
            qdata.to(device=device), value._layout_cls, dataclass_replace(params, **replacements)
        )
        return nn.Parameter(restored, requires_grad=value.requires_grad)
    copied = value.to(device=device)
    if isinstance(value, nn.Parameter):
        return nn.Parameter(copied, requires_grad=value.requires_grad)
    return copied


def _assign_storage_slot(slot: _LTX23StorageSlot, value: torch.Tensor) -> None:
    values = slot.module._parameters if slot.parameter else slot.module._buffers
    values[slot.name] = value


def _physical_storage_tensors(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    qdata = getattr(value, "_qdata", None)
    params = getattr(value, "params", None)
    tensor_fields = getattr(params, "_tensor_fields", None)
    if isinstance(qdata, torch.Tensor) and callable(tensor_fields):
        sidecars = tuple(getattr(params, field) for field in tensor_fields())
        if not all(isinstance(item, torch.Tensor) for item in sidecars):
            raise TypeError("LTX quantized storage contains a non-tensor sidecar")
        return (qdata, *sidecars)
    return (value,)


def _physical_tensor_key(value: torch.Tensor) -> tuple[int, int, int, torch.dtype]:
    return (
        value.untyped_storage().data_ptr(),
        value.storage_offset(),
        value.numel(),
        value.dtype,
    )


class LTX23StoredFP8Linear(nn.Module):
    """Bias-capable direct Kitchen FP8 linear with no dense fallback path."""

    def __init__(
        self,
        weight: Any,
        bias: torch.Tensor,
        *,
        input_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        from comfy_kitchen.tensor import QuantizedTensor

        if (
            not isinstance(weight, QuantizedTensor)
            or weight.ndim != 2
            or weight._layout_cls != "TensorCoreFP8Layout"
            or weight._qdata.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("LTX stored linear requires 2D TensorCore FP8 E4M3 data")
        if bias.dtype is not torch.bfloat16 or tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("LTX stored linear requires one BF16 bias per output feature")
        _validate_positive_scalar(input_scale, "input_scale")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.input_scale = float(input_scale.item())
        self.native_dispatch_count = 0
        self.rejected_dispatch_count = 0
        self.dense_fallback_count = 0
        self.last_dispatch_error: str | None = None
        self._lora_adapters = nn.ModuleDict()
        self.lora_dispatch_count = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, torch.Tensor) or not input.is_floating_point():
            raise TypeError("LTX stored FP8 dispatch requires floating-point activations")
        if input.ndim < 1 or input.shape[-1] != self.weight.shape[1]:
            raise ValueError("LTX stored linear input feature count does not match weight")
        # Keep any upstream work (notably FP32 normalization) intact until the
        # native linear seam.  Kitchen's direct TensorCore FP8 path and the
        # official BF16 LoRA weights must receive the *same* BF16 activation.
        execution_input = input
        if execution_input.dtype is not torch.bfloat16:
            execution_input = execution_input.to(dtype=torch.bfloat16)
        original_shape = execution_input.shape
        flat = execution_input.reshape(-1, original_shape[-1])
        try:
            output = _direct_kitchen_fp8_linear(
                flat,
                self.weight,
                self.bias,
                input_scale=self.input_scale,
            )
        except BaseException as exc:
            self.rejected_dispatch_count += 1
            self.last_dispatch_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "LTX direct Kitchen FP8 dispatch failed; fallback is forbidden"
            ) from exc
        self.native_dispatch_count += 1
        self.last_dispatch_error = None
        output = output.reshape(*original_shape[:-1], self.weight.shape[0])
        return self._apply_lora(execution_input, output)

    def add_lora_adapter(
        self,
        name: str,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        alpha_over_rank: float,
    ) -> None:
        _add_lora_adapter(self, name, down, up, alpha_over_rank=alpha_over_rank)

    def set_lora_strength(self, name: str, strength: float) -> None:
        _set_lora_strength(self, name, strength)

    def delete_lora_adapter(self, name: str) -> None:
        if name in self._lora_adapters:
            self._lora_adapters.pop(name)

    def _apply_lora(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        dispatched = False
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            output = output + adapter(input)
            dispatched = True
        if dispatched:
            self.lora_dispatch_count += 1
        return output

    def dispatch_evidence(self) -> dict[str, Any]:
        return {
            "backend": "comfy_kitchen.tensorcore_fp8",
            "native_dispatch_count": self.native_dispatch_count,
            "rejected_dispatch_count": self.rejected_dispatch_count,
            "dense_fallback_count": self.dense_fallback_count,
            "last_dispatch_error": self.last_dispatch_error,
        }

    def move_stored_storage(self, device: torch.device | str) -> None:
        from comfy_kitchen.tensor import QuantizedTensor

        target = torch.device(device)
        weight = self.weight
        params = dataclass_replace(weight.params, scale=weight.params.scale.to(device=target))
        restored = QuantizedTensor(weight._qdata.to(device=target), weight._layout_cls, params)
        self._parameters["weight"] = nn.Parameter(restored, requires_grad=False)
        self._parameters["bias"] = nn.Parameter(self.bias.to(device=target), requires_grad=False)


class _LTX23ConnectorProjection(nn.Linear):
    """BF16 projection seam after Diffusers' FP32 LTX 2.3 text normalization.

    The pinned LTX2 connector computes per-token RMS normalization and the
    modality rescale in the incoming hidden-state dtype.  Gemma's mixed text
    path can deliberately retain that intermediate as FP32, while the exact
    connector projection weights are BF16.  Cast only at this projection
    boundary: moving it earlier changes the normalization numerics, and
    letting ``nn.Linear`` infer it fails when its input and weights differ.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype is not self.weight.dtype:
            input = input.to(dtype=self.weight.dtype)
        return torch.nn.functional.linear(input, self.weight, self.bias)


class _LTX23LoraAdapter(nn.Module):
    def __init__(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        alpha_over_rank: float,
    ) -> None:
        super().__init__()
        if down.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
            raise ValueError("LTX model LoRA weights must remain BF16")
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError("LTX model LoRA geometry is invalid")
        if not math.isfinite(alpha_over_rank) or alpha_over_rank <= 0:
            raise ValueError("LTX model LoRA alpha/rank must be positive and finite")
        self.down = nn.Parameter(down, requires_grad=False)
        self.up = nn.Parameter(up, requires_grad=False)
        self.alpha_over_rank = float(alpha_over_rank)
        self.strength = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Diffusers' LTX 2.3 blocks can retain a FP32 intermediate through
        # normalization.  The official model LoRA remains BF16, so the LoRA
        # boundary—not the upstream normalization—is where we match the
        # activation to the stored branch weights.
        if input.dtype is not self.down.dtype:
            input = input.to(dtype=self.down.dtype)
        branch = torch.nn.functional.linear(input, self.down)
        branch = torch.nn.functional.linear(branch, self.up)
        return branch * (self.alpha_over_rank * self.strength)


class LTX23DenseLoraLinear(nn.Module):
    """Retain one BF16 base linear and add independently switchable LoRA branches."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if type(base) is not nn.Linear or base.weight.is_meta:
            raise TypeError("LTX dense LoRA wrapper requires a materialized nn.Linear")
        self.base = base
        self._lora_adapters = nn.ModuleDict()
        self.lora_dispatch_count = 0

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Preserve FP32 work performed before this linear, then hand off once
        # to the exact BF16 base/LoRA family.  Both the base projection and
        # each additive LoRA branch must see the same converted activation.
        branch_input = input
        if branch_input.dtype is not self.base.weight.dtype:
            branch_input = branch_input.to(dtype=self.base.weight.dtype)
        output = self.base(branch_input)
        dispatched = False
        for adapter in self._lora_adapters.values():
            if adapter.strength == 0.0:
                continue
            output = output + adapter(branch_input)
            dispatched = True
        if dispatched:
            self.lora_dispatch_count += 1
        return output

    def add_lora_adapter(
        self,
        name: str,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        alpha_over_rank: float,
    ) -> None:
        _add_lora_adapter(self, name, down, up, alpha_over_rank=alpha_over_rank)

    def set_lora_strength(self, name: str, strength: float) -> None:
        _set_lora_strength(self, name, strength)

    def delete_lora_adapter(self, name: str) -> None:
        if name in self._lora_adapters:
            self._lora_adapters.pop(name)


def inspect_ltx23_av_artifact(
    path: Path,
    *,
    expected_variant: LTX23AVVariant | None = None,
) -> LTX23AVArtifactContract:
    """Prove the exact AV state and stored-linear contract without reading payloads."""

    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("LTX AV artifact must be one SafeTensors file")
    header = _read_safetensors_header(resolved)
    return _contract_from_header(
        resolved,
        header,
        artifact_signature=path_signature(resolved),
        expected_variant=expected_variant,
    )


def build_ltx23_av_meta_shell(contract: LTX23AVArtifactContract) -> nn.Module:
    """Build the pinned Diffusers LTX 2.3 transformer on the meta device."""

    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel

    with init_empty_weights():
        shell = LTX2VideoTransformer3DModel.from_config(_DIFFUSERS_LTX23_CONFIG)
    _validate_diffusers_shell(shell, contract)
    shell._latentslate_ltx23_av_interface = contract.header_fingerprint
    return shell


def plan_ltx23_av_materialization(
    shell: nn.Module,
    path: Path,
    *,
    expected_variant: LTX23AVVariant | None = None,
) -> LTX23AVMaterializationPlan:
    contract = inspect_ltx23_av_artifact(path, expected_variant=expected_variant)
    shell_state = shell.state_dict()
    expected = {item.target_key: item for item in contract.transformer_state}
    if set(shell_state) != set(expected):
        missing = sorted(set(expected) - set(shell_state))[:3]
        extra = sorted(set(shell_state) - set(expected))[:3]
        raise ValueError(f"LTX AV shell state differs: missing={missing}, extra={extra}")
    for key, tensor in shell_state.items():
        if tuple(tensor.shape) != expected[key].shape:
            raise ValueError(f"LTX AV shell shape differs for {key!r}")
        if not tensor.is_meta:
            raise ValueError("LTX AV materialization requires a fully meta-device shell")
    for spec in contract.linears:
        if type(shell.get_submodule(spec.module_name)) is not nn.Linear:
            raise TypeError(f"LTX AV linear target changed for {spec.module_name!r}")
    fingerprint = _fingerprint(
        {
            "header": contract.header_fingerprint,
            "diffusers_runtime_revision": LTX23_AV_DIFFUSERS_RUNTIME_REVISION,
            "mapping_revision": LTX23_AV_DIFFUSERS_MAPPING_REVISION,
            "shell": f"{type(shell).__module__}.{type(shell).__qualname__}",
            "state": [
                (item.source_key, item.target_key, item.dtype, item.shape)
                for item in contract.transformer_state
            ],
        }
    )
    return LTX23AVMaterializationPlan(
        contract=contract,
        shell_type=f"{type(shell).__module__}.{type(shell).__qualname__}",
        plan_fingerprint=fingerprint,
    )


def materialize_ltx23_av(
    shell: nn.Module,
    plan: LTX23AVMaterializationPlan,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Stream exact state into a validated shell without dequantizing FP8 bases."""

    if compute_dtype is not torch.bfloat16:
        raise ValueError("LTX 2.3 stored AV materialization requires BF16 compute")
    if f"{type(shell).__module__}.{type(shell).__qualname__}" != plan.shell_type:
        raise TypeError("LTX AV shell type differs from its plan")
    rebound = inspect_ltx23_av_artifact(plan.contract.path, expected_variant=plan.contract.variant)
    if (
        rebound.artifact_signature != plan.contract.artifact_signature
        or rebound.header_fingerprint != plan.contract.header_fingerprint
    ):
        raise RuntimeError("LTX AV artifact changed after planning")
    # Revalidate topology immediately before the first mutation.
    rebound_plan = plan_ltx23_av_materialization(
        shell, plan.contract.path, expected_variant=plan.contract.variant
    )
    if rebound_plan.plan_fingerprint != plan.plan_fingerprint:
        raise RuntimeError("LTX AV shell changed after planning")
    linear_names = {item.module_name for item in plan.contract.linears}
    try:
        with safe_open(str(plan.contract.path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(_read_safetensors_header(plan.contract.path)) - {
                "__metadata__"
            }:
                raise ValueError("LTX AV artifact key set changed while opening payloads")
            for spec in plan.contract.linears:
                bias = handle.get_tensor(spec.source_bias_key)
                if spec.quantized:
                    qdata = handle.get_tensor(spec.source_weight_key)
                    weight_scale = handle.get_tensor(spec.source_weight_scale_key)
                    input_scale = handle.get_tensor(spec.source_input_scale_key)
                    weight = _restore_fp8(qdata, weight_scale, compute_dtype)
                    replacement: nn.Module = LTX23StoredFP8Linear(
                        weight, bias, input_scale=input_scale
                    )
                else:
                    weight = handle.get_tensor(spec.source_weight_key)
                    replacement = nn.Linear(
                        spec.weight.shape[1], spec.weight.shape[0], bias=True, device="meta"
                    )
                    replacement._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
                    replacement._parameters["bias"] = nn.Parameter(bias, requires_grad=False)
                _replace_module(shell, spec.module_name, replacement)
            for state in plan.contract.transformer_state:
                module_name, _, _leaf = state.target_key.rpartition(".")
                if module_name in linear_names:
                    continue
                _assign_state_tensor(shell, state.target_key, handle.get_tensor(state.source_key))
        if any(tensor.is_meta for tensor in shell.state_dict().values()):
            raise RuntimeError("LTX AV materialization left meta-device state")
        shell._latentslate_ltx23_av_materialization = {
            "variant": plan.contract.variant,
            "plan_fingerprint": plan.plan_fingerprint,
            "source_state_keys": len(plan.contract.state),
            "transformer_state_keys": len(plan.contract.transformer_state),
            "linears": len(plan.contract.linears),
            "stored_fp8_linears": plan.contract.quantized_linear_count,
            "dense_base_dequantizations": 0,
        }
        return shell
    except BaseException as exc:
        shell._latentslate_ltx23_av_poisoned = f"{type(exc).__name__}: {exc}"
        raise


def build_ltx23_connector_meta_shell(contract: LTX23AVArtifactContract) -> nn.Module:
    """Build the exact pinned Diffusers LTX 2.3 text connector on meta."""

    from accelerate import init_empty_weights
    from diffusers.pipelines.ltx2 import LTX2TextConnectors

    with init_empty_weights():
        shell = LTX2TextConnectors.from_config(_DIFFUSERS_LTX23_CONNECTOR_CONFIG)
    # Keep the pinned connector topology and its FP32 per-token normalization,
    # but make the two BF16 projection boundaries explicit.  Both replacements
    # preserve the state-dict keys exactly, so the checkpoint closure below
    # continues to validate the original Diffusers module layout.
    for name in ("video_text_proj_in", "audio_text_proj_in"):
        projection = getattr(shell, name, None)
        if type(projection) is not nn.Linear:
            raise RuntimeError(f"Pinned Diffusers LTX 2.3 connector lacks {name}")
        replacement = _LTX23ConnectorProjection(
            projection.in_features,
            projection.out_features,
            bias=projection.bias is not None,
            device="meta",
        )
        _replace_module(shell, name, replacement)
    _validate_connector_shell(shell, contract)
    shell._latentslate_ltx23_connector_interface = contract.header_fingerprint
    return shell


def plan_ltx23_connector_materialization(
    shell: nn.Module,
    path: Path,
    *,
    expected_variant: LTX23AVVariant | None = None,
) -> LTX23ConnectorMaterializationPlan:
    contract = inspect_ltx23_av_artifact(path, expected_variant=expected_variant)
    _validate_connector_shell(shell, contract)
    fingerprint = _fingerprint(
        {
            "header": contract.header_fingerprint,
            "diffusers_runtime_revision": LTX23_AV_DIFFUSERS_RUNTIME_REVISION,
            "mapping_revision": LTX23_AV_DIFFUSERS_MAPPING_REVISION,
            "shell": f"{type(shell).__module__}.{type(shell).__qualname__}",
            "state": [
                (item.source_key, item.target_key, item.dtype, item.shape)
                for item in (*contract.connector_state, *contract.external_connector_state)
            ],
        }
    )
    return LTX23ConnectorMaterializationPlan(
        contract=contract,
        shell_type=f"{type(shell).__module__}.{type(shell).__qualname__}",
        plan_fingerprint=fingerprint,
    )


def materialize_ltx23_connectors(
    shell: nn.Module,
    plan: LTX23ConnectorMaterializationPlan,
) -> nn.Module:
    """Stream the exact 262-key BF16 connector closure into its meta shell."""

    if f"{type(shell).__module__}.{type(shell).__qualname__}" != plan.shell_type:
        raise TypeError("LTX connector shell type differs from its plan")
    rebound = inspect_ltx23_av_artifact(plan.contract.path, expected_variant=plan.contract.variant)
    if (
        rebound.artifact_signature != plan.contract.artifact_signature
        or rebound.header_fingerprint != plan.contract.header_fingerprint
    ):
        raise RuntimeError("LTX connector artifact changed after planning")
    rebound_plan = plan_ltx23_connector_materialization(
        shell, plan.contract.path, expected_variant=plan.contract.variant
    )
    if rebound_plan.plan_fingerprint != plan.plan_fingerprint:
        raise RuntimeError("LTX connector shell changed after planning")
    try:
        with safe_open(str(plan.contract.path), framework="pt", device="cpu") as handle:
            for state in (*plan.contract.connector_state, *plan.contract.external_connector_state):
                _assign_state_tensor(shell, state.target_key, handle.get_tensor(state.source_key))
        if any(tensor.is_meta for tensor in shell.state_dict().values()):
            raise RuntimeError("LTX connector materialization left meta-device state")
        shell._latentslate_ltx23_connector_materialization = {
            "variant": plan.contract.variant,
            "plan_fingerprint": plan.plan_fingerprint,
            "state_keys": len(shell.state_dict()),
        }
        return shell
    except BaseException as exc:
        shell._latentslate_ltx23_connector_poisoned = f"{type(exc).__name__}: {exc}"
        raise


def inspect_ltx23_model_lora(
    base: LTX23AVArtifactContract,
    lora_path: Path,
) -> LTX23AVLoraContract:
    """Map the exact official distilled model LoRA onto the Dev AV linears."""

    if base.variant != "dev":
        raise ValueError("The LTX 2.3 distilled model LoRA targets the Dev AV artifact")
    resolved = Path(lora_path).resolve(strict=True)
    header = _read_safetensors_header(resolved)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("LTX model LoRA metadata must be an object")
    linear_by_name = {item.module_name: item for item in base.linears}
    roles: dict[str, dict[str, str]] = {}
    for key in sorted(header):
        if key.endswith(".lora_A.weight"):
            roles.setdefault(key[: -len(".lora_A.weight")], {})["down"] = key
        elif key.endswith(".lora_B.weight"):
            roles.setdefault(key[: -len(".lora_B.weight")], {})["up"] = key
        elif key.endswith(".alpha"):
            roles.setdefault(key[: -len(".alpha")], {})["alpha"] = key
        else:
            raise ValueError(f"LTX model LoRA contains unsupported tensor {key!r}")
    if len(roles) != _EXPECTED_LORA_TARGETS:
        raise ValueError("LTX model LoRA target count changed")
    targets: list[LTX23AVLoraTargetSpec] = []
    missing_alpha: list[str] = []
    for source_stem, role in sorted(roles.items()):
        if set(role) not in ({"down", "up"}, {"down", "up", "alpha"}):
            raise ValueError(f"LTX model LoRA pair is incomplete for {source_stem!r}")
        source_module_name = source_stem.removeprefix("diffusion_model.")
        module_name = _map_transformer_key(source_module_name + ".weight").removesuffix(".weight")
        if source_stem == source_module_name or module_name not in linear_by_name:
            raise ValueError(f"LTX model LoRA target is unsupported: {source_stem!r}")
        linear = linear_by_name[module_name]
        down = _entry(header, role["down"])
        up = _entry(header, role["up"])
        if down.dtype != "BF16" or up.dtype != "BF16":
            raise ValueError("LTX model LoRA weights must remain BF16")
        if (
            len(down.shape) != 2
            or len(up.shape) != 2
            or up.shape[1] != down.shape[0]
            or down.shape[1] != linear.weight.shape[1]
            or up.shape[0] != linear.weight.shape[0]
        ):
            raise ValueError(f"LTX model LoRA geometry changed for {source_stem!r}")
        rank = down.shape[0]
        alpha_key = role.get("alpha")
        alpha_over_rank = 1.0
        if alpha_key is None:
            missing_alpha.append(module_name)
        else:
            alpha = _entry(header, alpha_key)
            if alpha.dtype != "BF16" or alpha.shape != ():
                raise ValueError(f"LTX model LoRA alpha changed for {source_stem!r}")
            # Header-only inspection cannot read the scalar value. Runtime loading
            # must bind and validate it before installing the additive branch.
            alpha_over_rank = math.nan
        targets.append(
            LTX23AVLoraTargetSpec(
                module_name, role["down"], role["up"], alpha_key, rank, alpha_over_rank
            )
        )
    if missing_alpha != [_EXPECTED_MISSING_ALPHA]:
        raise ValueError(f"LTX model LoRA missing-alpha contract changed: {missing_alpha}")
    return LTX23AVLoraContract(
        path=resolved,
        artifact_signature=path_signature(resolved),
        header_fingerprint=_fingerprint(header),
        targets=tuple(targets),
    )


def install_ltx23_model_lora(
    transformer: nn.Module,
    contract: LTX23AVLoraContract,
    *,
    adapter_name: str,
    strength: float = 0.5,
) -> LTX23AVLoraInstallation:
    """Install the model LoRA as additive branches; never merge the FP8 base."""

    if not _ADAPTER_NAME.fullmatch(adapter_name):
        raise ValueError("LTX model LoRA adapter name is unsafe")
    if not math.isfinite(strength):
        raise ValueError("LTX model LoRA strength must be finite")
    rebound_header = _read_safetensors_header(contract.path)
    rebound_header.pop("__metadata__", None)
    if (
        path_signature(contract.path) != contract.artifact_signature
        or _fingerprint(rebound_header) != contract.header_fingerprint
    ):
        raise RuntimeError("LTX model LoRA changed after inspection")
    promoted: list[tuple[str, nn.Linear]] = []
    installed: list[str] = []
    try:
        with safe_open(str(contract.path), framework="pt", device="cpu") as handle:
            for target in contract.targets:
                module = transformer.get_submodule(target.module_name)
                if type(module) is nn.Linear:
                    original = module
                    module = LTX23DenseLoraLinear(original)
                    _replace_module(transformer, target.module_name, module)
                    promoted.append((target.module_name, original))
                if not isinstance(module, (LTX23StoredFP8Linear, LTX23DenseLoraLinear)):
                    raise TypeError(f"LTX model LoRA target changed: {target.module_name!r}")
                if adapter_name in module._lora_adapters:
                    raise ValueError(
                        f"LTX model LoRA adapter {adapter_name!r} is already installed"
                    )
                down = handle.get_tensor(target.down_key)
                up = handle.get_tensor(target.up_key)
                alpha_over_rank = 1.0
                if target.alpha_key is not None:
                    alpha = handle.get_tensor(target.alpha_key)
                    if (
                        alpha.dtype is not torch.bfloat16
                        or alpha.ndim != 0
                        or not bool(torch.isfinite(alpha))
                        or not bool(alpha > 0)
                    ):
                        raise ValueError(
                            f"LTX model LoRA alpha is invalid for {target.module_name!r}"
                        )
                    alpha_over_rank = float(alpha.float().item()) / target.rank
                target_device = (
                    module.bias.device if module.bias is not None else module.weight.device
                )
                module.add_lora_adapter(
                    adapter_name,
                    down.to(device=target_device),
                    up.to(device=target_device),
                    alpha_over_rank=alpha_over_rank,
                )
                module.set_lora_strength(adapter_name, strength)
                installed.append(target.module_name)
        if len(installed) != len(contract.targets):
            raise RuntimeError("LTX model LoRA installation count changed")
        return LTX23AVLoraInstallation(
            adapter_name=adapter_name,
            contract_fingerprint=contract.header_fingerprint,
            modules=tuple(installed),
        )
    except BaseException:
        for module_name in reversed(installed):
            module = transformer.get_submodule(module_name)
            if isinstance(module, (LTX23StoredFP8Linear, LTX23DenseLoraLinear)):
                module.delete_lora_adapter(adapter_name)
        for module_name, original in reversed(promoted):
            current = transformer.get_submodule(module_name)
            if isinstance(current, LTX23DenseLoraLinear) and not current._lora_adapters:
                _replace_module(transformer, module_name, original)
        raise


def set_ltx23_model_lora_strength(
    transformer: nn.Module,
    installation: LTX23AVLoraInstallation,
    strength: float,
) -> None:
    if not math.isfinite(strength):
        raise ValueError("LTX model LoRA strength must be finite")
    modules: list[LTX23StoredFP8Linear | LTX23DenseLoraLinear] = []
    for module_name in installation.modules:
        module = transformer.get_submodule(module_name)
        if not isinstance(module, (LTX23StoredFP8Linear, LTX23DenseLoraLinear)):
            raise TypeError(f"LTX model LoRA target changed: {module_name!r}")
        if installation.adapter_name not in module._lora_adapters:
            raise RuntimeError(f"LTX model LoRA adapter disappeared: {module_name!r}")
        modules.append(module)
    for module in modules:
        module.set_lora_strength(installation.adapter_name, strength)


def remove_ltx23_model_lora(
    transformer: nn.Module,
    installation: LTX23AVLoraInstallation,
) -> None:
    modules: list[tuple[str, LTX23StoredFP8Linear | LTX23DenseLoraLinear]] = []
    for module_name in installation.modules:
        module = transformer.get_submodule(module_name)
        if not isinstance(module, (LTX23StoredFP8Linear, LTX23DenseLoraLinear)):
            raise TypeError(f"LTX model LoRA target changed: {module_name!r}")
        if installation.adapter_name not in module._lora_adapters:
            raise RuntimeError(f"LTX model LoRA adapter disappeared: {module_name!r}")
        modules.append((module_name, module))
    for module_name, module in modules:
        module.delete_lora_adapter(installation.adapter_name)
        if isinstance(module, LTX23DenseLoraLinear) and not module._lora_adapters:
            _replace_module(transformer, module_name, module.base)


def ltx23_model_lora_dispatch_evidence(
    transformer: nn.Module,
    installation: LTX23AVLoraInstallation,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    modules: list[LTX23StoredFP8Linear | LTX23DenseLoraLinear] = []
    for module_name in installation.modules:
        module = transformer.get_submodule(module_name)
        if not isinstance(module, (LTX23StoredFP8Linear, LTX23DenseLoraLinear)):
            raise TypeError(f"LTX model LoRA target changed: {module_name!r}")
        if installation.adapter_name not in module._lora_adapters:
            raise RuntimeError(f"LTX model LoRA adapter disappeared: {module_name!r}")
        modules.append(module)
    dispatched = 0
    for module in modules:
        if module.lora_dispatch_count > 0:
            dispatched += 1
        if reset:
            module.lora_dispatch_count = 0
    return {
        "adapter_name": installation.adapter_name,
        "selected_targets": len(installation.modules),
        "dispatched_targets": dispatched,
        "complete": dispatched == len(installation.modules),
    }


def aggregate_ltx23_fp8_dispatch(shell: nn.Module) -> dict[str, Any]:
    modules = [item for item in shell.modules() if isinstance(item, LTX23StoredFP8Linear)]
    return {
        "modules": len(modules),
        "native_dispatch_count": sum(item.native_dispatch_count for item in modules),
        "rejected_dispatch_count": sum(item.rejected_dispatch_count for item in modules),
        "dense_fallback_count": sum(item.dense_fallback_count for item in modules),
    }


def _validate_diffusers_shell(shell: nn.Module, contract: LTX23AVArtifactContract) -> None:
    expected = {item.target_key: item for item in contract.transformer_state}
    actual = shell.state_dict()
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise RuntimeError(
            f"Pinned Diffusers LTX 2.3 shell closure changed: missing={missing}, extra={extra}"
        )
    for key, tensor in actual.items():
        if not tensor.is_meta or tuple(tensor.shape) != expected[key].shape:
            raise RuntimeError(f"Pinned Diffusers LTX 2.3 shell differs for {key!r}")
    if sum(type(item) is nn.Linear for item in shell.modules()) != _EXPECTED_LINEARS:
        raise RuntimeError("Pinned Diffusers LTX 2.3 linear topology changed")


def _validate_connector_shell(shell: nn.Module, contract: LTX23AVArtifactContract) -> None:
    expected = {
        item.target_key: item
        for item in (*contract.connector_state, *contract.external_connector_state)
    }
    actual = shell.state_dict()
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise RuntimeError(
            f"Pinned Diffusers LTX 2.3 connector closure changed: missing={missing}, extra={extra}"
        )
    for key, tensor in actual.items():
        if not tensor.is_meta or tuple(tensor.shape) != expected[key].shape:
            raise RuntimeError(f"Pinned Diffusers LTX 2.3 connector differs for {key!r}")


def _replace_all(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def _map_transformer_key(source_key: str) -> str:
    target = _replace_all(source_key, _TRANSFORMER_RENAMES)
    if target.endswith((".weight", ".bias")):
        if target.startswith("adaln_single."):
            target = target.replace("adaln_single.", "time_embed.", 1)
        if target.startswith("audio_adaln_single."):
            target = target.replace("audio_adaln_single.", "audio_time_embed.", 1)
    return target


def _map_connector_key(source_key: str) -> str:
    return _replace_all(source_key, _CONNECTOR_RENAMES)


def _contract_from_header(
    path: Path,
    raw_header: dict[str, Any],
    *,
    artifact_signature: dict[str, Any],
    expected_variant: LTX23AVVariant | None,
) -> LTX23AVArtifactContract:
    header = dict(raw_header)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("LTX AV SafeTensors metadata must be an object")
    model = {key: value for key, value in header.items() if key.startswith(_PREFIX)}
    state_raw = {
        key.removeprefix(_PREFIX): value
        for key, value in model.items()
        if not key.endswith(_SCALE_SUFFIXES)
    }
    if len(state_raw) != _EXPECTED_STATE_KEYS:
        raise ValueError(f"LTX AV logical state count changed: {len(state_raw)}")
    state = tuple(_entry(state_raw, key, logical_key=key) for key in sorted(state_raw))
    transformer_state: list[LTX23AVMappedStateSpec] = []
    connector_state: list[LTX23AVMappedStateSpec] = []
    for item in state:
        connector = item.key.startswith(_CONNECTOR_PREFIXES)
        target_key = _map_connector_key(item.key) if connector else _map_transformer_key(item.key)
        mapped = LTX23AVMappedStateSpec(
            source_key=_PREFIX + item.key,
            target_key=target_key,
            dtype=item.dtype,
            shape=item.shape,
        )
        (connector_state if connector else transformer_state).append(mapped)
    if len(transformer_state) != _EXPECTED_TRANSFORMER_STATE_KEYS:
        raise ValueError("LTX AV Diffusers transformer split count changed")
    if len(connector_state) != _EXPECTED_CONNECTOR_STATE_KEYS:
        raise ValueError("LTX AV Diffusers connector split count changed")
    if len({item.target_key for item in transformer_state}) != len(transformer_state):
        raise ValueError("LTX AV transformer key mapping is not one-to-one")
    if len({item.target_key for item in connector_state}) != len(connector_state):
        raise ValueError("LTX AV connector key mapping is not one-to-one")
    external_connector_state = tuple(
        LTX23AVMappedStateSpec(
            source_key=key,
            target_key=_map_connector_key(key),
            dtype=_entry(header, key).dtype,
            shape=_entry(header, key).shape,
        )
        for key in sorted(header)
        if key.startswith("text_embedding_projection.")
    )
    if len(external_connector_state) != _EXPECTED_EXTERNAL_CONNECTOR_STATE_KEYS:
        raise ValueError("LTX AV external connector projection closure changed")
    if any(item.dtype != "BF16" for item in (*connector_state, *external_connector_state)):
        raise ValueError("LTX AV connector precision changed")
    rank2_weights = [
        item
        for item in transformer_state
        if item.target_key.endswith(".weight") and len(item.shape) == 2
    ]
    if len(rank2_weights) != _EXPECTED_LINEARS:
        raise ValueError(f"LTX AV linear count changed: {len(rank2_weights)}")
    fp8_count = sum(item.dtype == "F8_E4M3" for item in rank2_weights)
    variants = [name for name, count in _EXPECTED_FP8.items() if count == fp8_count]
    if len(variants) != 1:
        raise ValueError(f"LTX AV stored FP8 linear count is unsupported: {fp8_count}")
    variant: LTX23AVVariant = variants[0]  # type: ignore[assignment]
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"LTX AV artifact is {variant}, expected {expected_variant}")
    dtype_counts: dict[str, int] = {}
    for item in state:
        dtype_counts[item.dtype] = dtype_counts.get(item.dtype, 0) + 1
    if dtype_counts != _EXPECTED_STATE_DTYPES[variant]:
        raise ValueError(f"LTX AV state dtype contract changed: {dtype_counts}")
    state_by_key = {item.key: item for item in state}
    linears: list[LTX23AVLinearSpec] = []
    expected_model_keys = {_PREFIX + item.key for item in state}
    for weight in rank2_weights:
        module_name = weight.target_key.removesuffix(".weight")
        source_module_name = weight.source_key.removeprefix(_PREFIX).removesuffix(".weight")
        source_bias_key = source_module_name + ".bias"
        bias = state_by_key.get(source_bias_key)
        if bias is None or bias.dtype != "BF16" or bias.shape != (weight.shape[0],):
            raise ValueError(f"LTX AV BF16 linear bias contract changed for {module_name!r}")
        if weight.dtype not in {"BF16", "F8_E4M3"}:
            raise ValueError(f"LTX AV linear precision changed for {module_name!r}")
        quantized = weight.dtype == "F8_E4M3"
        weight_scale_key = _PREFIX + source_module_name + ".weight_scale" if quantized else None
        input_scale_key = _PREFIX + source_module_name + ".input_scale" if quantized else None
        if quantized:
            for scale_key in (weight_scale_key, input_scale_key):
                scale = _entry(model, scale_key)
                if scale.dtype != "F32" or scale.shape != ():
                    raise ValueError(f"LTX AV scalar FP8 sidecar changed for {module_name!r}")
                expected_model_keys.add(scale_key)
        linears.append(
            LTX23AVLinearSpec(
                module_name=module_name,
                weight=LTX23AVStateSpec(weight.target_key, weight.dtype, weight.shape),
                bias=bias,
                quantized=quantized,
                source_weight_key=weight.source_key,
                source_bias_key=_PREFIX + bias.key,
                source_weight_scale_key=weight_scale_key,
                source_input_scale_key=input_scale_key,
            )
        )
    if set(model) != expected_model_keys:
        extra = sorted(set(model) - expected_model_keys)[:3]
        missing = sorted(expected_model_keys - set(model))[:3]
        raise ValueError(f"LTX AV model key closure changed: missing={missing}, extra={extra}")
    return LTX23AVArtifactContract(
        path=path,
        variant=variant,
        artifact_signature=artifact_signature,
        header_fingerprint=_fingerprint(
            {
                **model,
                **{item.source_key: header[item.source_key] for item in external_connector_state},
            }
        ),
        state=state,
        transformer_state=tuple(transformer_state),
        connector_state=tuple(connector_state),
        external_connector_state=external_connector_state,
        linears=tuple(linears),
    )


def _direct_kitchen_fp8_linear(
    input: torch.Tensor,
    weight: Any,
    bias: torch.Tensor,
    *,
    input_scale: float,
) -> torch.Tensor:
    if input.device.type != "cuda":
        raise RuntimeError("LTX native FP8 dispatch requires CUDA input")
    import comfy_kitchen as ck
    from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

    scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    with ck.use_backend("cuda"):
        quantize = ck.registry.get_implementation("quantize_per_tensor_fp8", backend="cuda")
        qdata = quantize(input, scale, torch.float8_e4m3fn)
        return scaled_mm_v2(
            qdata,
            weight._qdata.t(),
            scale,
            weight.params.scale,
            bias=bias,
            out_dtype=input.dtype,
        )


def _add_lora_adapter(
    module: LTX23StoredFP8Linear | LTX23DenseLoraLinear,
    name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha_over_rank: float,
) -> None:
    if not _ADAPTER_NAME.fullmatch(name):
        raise ValueError("LTX model LoRA adapter name is unsafe")
    if name in module._lora_adapters:
        raise ValueError(f"LTX model LoRA adapter {name!r} is already installed")
    if down.shape[1] != module.weight.shape[1] or up.shape[0] != module.weight.shape[0]:
        raise ValueError("LTX model LoRA geometry differs from target linear")
    module._lora_adapters[name] = _LTX23LoraAdapter(down, up, alpha_over_rank=alpha_over_rank)


def _set_lora_strength(
    module: LTX23StoredFP8Linear | LTX23DenseLoraLinear,
    name: str,
    strength: float,
) -> None:
    if not math.isfinite(strength):
        raise ValueError("LTX model LoRA strength must be finite")
    try:
        adapter = module._lora_adapters[name]
    except KeyError as exc:
        raise KeyError(f"LTX model LoRA adapter {name!r} is not installed") from exc
    adapter.strength = float(strength)


def _restore_fp8(qdata: torch.Tensor, scale: torch.Tensor, compute_dtype: torch.dtype) -> Any:
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    if qdata.dtype is not torch.float8_e4m3fn or qdata.ndim != 2:
        raise ValueError("LTX AV stored qdata must remain 2D FP8 E4M3")
    _validate_positive_scalar(scale, "weight_scale")
    params = TensorCoreFP8Layout.Params(
        scale=scale, orig_dtype=compute_dtype, orig_shape=tuple(qdata.shape)
    )
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _validate_positive_scalar(value: torch.Tensor, name: str) -> None:
    if (
        value.dtype is not torch.float32
        or value.ndim != 0
        or not bool(torch.isfinite(value))
        or not bool(value > 0)
    ):
        raise ValueError(f"LTX AV {name} must be one positive finite F32 scalar")


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if leaf not in parent._modules:
        raise AttributeError(f"LTX AV module disappeared: {path!r}")
    parent._modules[leaf] = replacement


def _assign_state_tensor(root: nn.Module, key: str, value: torch.Tensor) -> None:
    parent_path, _, leaf = key.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        raise AttributeError(f"LTX AV state target disappeared: {key!r}")


def _entry(
    values: dict[str, Any],
    key: str,
    *,
    logical_key: str | None = None,
) -> LTX23AVStateSpec:
    value = values.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
        raise TypeError(f"LTX AV invalid header entry for {key!r}")
    shape = value.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape
    ):
        raise TypeError(f"LTX AV invalid header shape for {key!r}")
    return LTX23AVStateSpec(logical_key or key, value["dtype"], tuple(shape))


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("LTX AV SafeTensors header is truncated")
        length = struct.unpack("<Q", raw_length)[0]
        if length > _MAX_HEADER_BYTES or length > size - 8:
            raise ValueError("LTX AV SafeTensors header exceeds bounds")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("LTX AV SafeTensors header is truncated")
    parsed = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(parsed, dict):
        raise TypeError("LTX AV SafeTensors header must be an object")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"LTX AV duplicate JSON key {key!r}")
        result[key] = value
    return result


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(encoded.encode()).hexdigest()
