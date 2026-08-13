"""CPU-only, Engine-owned LTX 2.3 media-component materialization seam.

The LTX checkpoint embeds three independently useful media components.  This
module validates their source-to-Diffusers conversion against a meta-device
shell before it reads a tensor payload, then streams each tensor directly to
CPU. It has no workflow-runtime dependency.

The key transforms and configurations are transcribed from Diffusers commit
``f53d552``'s ``scripts/convert_ltx2_to_diffusers.py`` (the LTX 2.3 branch).
They are kept here rather than importing that script so the Engine has one
small, pinned runtime boundary and no conversion-script dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn

from .ltx23_kitchen_contracts import LTX23StoredArtifactPlan

LTX23MediaComponent = Literal["video_vae", "audio_vae", "vocoder", "latent_upsampler"]

_EXPECTED_SOURCE_COUNTS: Mapping[LTX23MediaComponent, int] = MappingProxyType(
    {"video_vae": 170, "audio_vae": 102, "vocoder": 1227, "latent_upsampler": 72}
)
# Preserve the source script's replacement order.  In particular, replacing
# ``up_blocks.1`` must not then be transformed again as ``up_blocks.0``.
_VIDEO_RENAMES = (
    ("down_blocks.0", "down_blocks.0"),
    ("down_blocks.1", "down_blocks.0.downsamplers.0"),
    ("down_blocks.2", "down_blocks.1"),
    ("down_blocks.3", "down_blocks.1.downsamplers.0"),
    ("down_blocks.4", "down_blocks.2"),
    ("down_blocks.5", "down_blocks.2.downsamplers.0"),
    ("down_blocks.6", "down_blocks.3"),
    ("down_blocks.7", "down_blocks.3.downsamplers.0"),
    ("down_blocks.8", "mid_block"),
    ("up_blocks.0", "mid_block"),
    ("up_blocks.1", "up_blocks.0.upsamplers.0"),
    ("up_blocks.2", "up_blocks.0"),
    ("up_blocks.3", "up_blocks.1.upsamplers.0"),
    ("up_blocks.4", "up_blocks.1"),
    ("up_blocks.5", "up_blocks.2.upsamplers.0"),
    ("up_blocks.6", "up_blocks.2"),
    ("up_blocks.7", "up_blocks.3.upsamplers.0"),
    ("up_blocks.8", "up_blocks.3"),
    ("last_time_embedder", "time_embedder"),
    ("last_scale_shift_table", "scale_shift_table"),
    ("res_blocks", "resnets"),
    ("per_channel_statistics.mean-of-means", "latents_mean"),
    ("per_channel_statistics.std-of-means", "latents_std"),
)
_AUDIO_RENAMES = (
    ("per_channel_statistics.mean-of-means", "latents_mean"),
    ("per_channel_statistics.std-of-means", "latents_std"),
)
_VOCODER_RENAMES = (
    ("resblocks", "resnets"),
    ("conv_pre", "conv_in"),
    ("conv_post", "conv_out"),
    ("act_post", "act_out"),
    ("downsample.lowpass", "downsample"),
)

_VIDEO_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "block_out_channels": (256, 512, 1024, 1024),
        "down_block_types": ("LTX2VideoDownBlock3D",) * 4,
        "decoder_block_out_channels": (256, 512, 512, 1024),
        "layers_per_block": (4, 6, 4, 2, 2),
        "decoder_layers_per_block": (4, 6, 4, 2, 2),
        "spatio_temporal_scaling": (True,) * 4,
        "decoder_spatio_temporal_scaling": (True,) * 4,
        "decoder_inject_noise": (False,) * 5,
        "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
        "upsample_residual": (False,) * 4,
        "upsample_factor": (2, 2, 1, 2),
        "timestep_conditioning": False,
        "patch_size": 4,
        "patch_size_t": 1,
        "resnet_norm_eps": 1e-6,
        "encoder_causal": True,
        "decoder_causal": False,
        "encoder_spatial_padding_mode": "zeros",
        "decoder_spatial_padding_mode": "zeros",
        "spatial_compression_ratio": 32,
        "temporal_compression_ratio": 8,
    }
)
_AUDIO_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "base_channels": 128, "output_channels": 2, "ch_mult": (1, 2, 4),
        "num_res_blocks": 2, "attn_resolutions": None, "in_channels": 2,
        "resolution": 256, "latent_channels": 8, "norm_type": "pixel",
        "causality_axis": "height", "dropout": 0.0, "mid_block_add_attention": False,
        "sample_rate": 16000, "mel_hop_length": 160, "is_causal": True,
        "mel_bins": 64, "double_z": True,
    }
)
_VOCODER_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "in_channels": 128, "hidden_channels": 1536, "out_channels": 2,
        "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4], "upsample_factors": [5, 2, 2, 2, 2, 2],
        "resnet_kernel_sizes": [3, 7, 11], "resnet_dilations": [[1, 3, 5]] * 3,
        "act_fn": "snakebeta", "leaky_relu_negative_slope": 0.1, "antialias": True,
        "antialias_ratio": 2, "antialias_kernel_size": 12, "final_act_fn": None,
        "final_bias": False, "bwe_in_channels": 128, "bwe_hidden_channels": 512,
        "bwe_out_channels": 2, "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
        "bwe_upsample_factors": [6, 5, 2, 2, 2], "bwe_resnet_kernel_sizes": [3, 7, 11],
        "bwe_resnet_dilations": [[1, 3, 5]] * 3, "bwe_act_fn": "snakebeta",
        "bwe_leaky_relu_negative_slope": 0.1, "bwe_antialias": True,
        "bwe_antialias_ratio": 2, "bwe_antialias_kernel_size": 12,
        "bwe_final_act_fn": None, "bwe_final_bias": False, "filter_length": 512,
        "hop_length": 80, "window_length": 512, "num_mel_channels": 64,
        "input_sampling_rate": 16000, "output_sampling_rate": 48000,
    }
)
_UPSCALER_CONFIG: Mapping[str, Any] = MappingProxyType(
    {"in_channels": 128, "mid_channels": 1024, "num_blocks_per_stage": 4, "dims": 3,
     "spatial_upsample": True, "temporal_upsample": False, "rational_spatial_scale": 2.0,
     "use_rational_resampler": False}
)


@dataclass(frozen=True, slots=True)
class LTX23MediaTensorPlan:
    """One header-proven source tensor and its exact shell target."""

    source: str
    target: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class LTX23MediaComponentPlan:
    """Immutable plan for one independently materializable LTX media component."""

    stored_artifact: LTX23StoredArtifactPlan
    component: LTX23MediaComponent
    shell_type: str
    tensors: tuple[LTX23MediaTensorPlan, ...]
    ignored_sources: tuple[str, ...]
    fingerprint: str

    @property
    def source_count(self) -> int:
        return len(self.tensors) + len(self.ignored_sources)


def build_ltx23_media_shell(component: LTX23MediaComponent) -> nn.Module:
    """Create one pinned Diffusers LTX 2.3 shell on the meta device only."""

    from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2VocoderWithBWE

    with init_empty_weights():
        if component == "video_vae":
            shell: nn.Module = AutoencoderKLLTX2Video.from_config(dict(_VIDEO_CONFIG))
        elif component == "audio_vae":
            shell = AutoencoderKLLTX2Audio.from_config(dict(_AUDIO_CONFIG))
        elif component == "vocoder":
            shell = LTX2VocoderWithBWE.from_config(dict(_VOCODER_CONFIG))
        elif component == "latent_upsampler":
            shell = LTX2LatentUpsamplerModel(**dict(_UPSCALER_CONFIG))
        else:
            raise ValueError(f"unsupported LTX 2.3 media component {component!r}")
    # Diffusers initializes a few deterministic buffers outside Accelerate's
    # meta context.  Discard those initial values before planning so no
    # duplicate checkpoint residency exists and every target is materialized.
    shell.to_empty(device="meta")
    return shell


def plan_ltx23_media_component(
    stored_artifact: LTX23StoredArtifactPlan,
    component: LTX23MediaComponent,
    shell: nn.Module,
) -> LTX23MediaComponentPlan:
    """Derive an exact source-to-shell plan using SafeTensors headers only."""

    stored_artifact.require_available()
    expected_role = {
        "video_vae": "vae",
        "audio_vae": "audio_vae",
        "vocoder": "vocoder",
        "latent_upsampler": "latent_upscaler",
    }[component]
    sources = sorted(key for key, role in stored_artifact.roles.items() if role.startswith(expected_role + "/"))
    if len(sources) != _EXPECTED_SOURCE_COUNTS[component]:
        raise ValueError(f"LTX {component} source count changed: {len(sources)}")
    if not stored_artifact.revalidate():
        raise ValueError("LTX media artifact changed after its stored-artifact plan")
    shell_state = shell.state_dict()
    if any(not value.is_meta for value in shell_state.values()):
        raise ValueError("LTX media planning requires a fully meta-device shell")

    ignored: set[str] = set()
    converted: dict[str, str] = {}
    for source in sources:
        target = _convert_source(component, source, ignored)
        if target is not None:
            if target in converted.values():
                raise ValueError(f"LTX {component} conversion collides at {target!r}")
            converted[source] = target
    if set(ignored) | set(converted) != set(sources):
        raise ValueError(f"LTX {component} source-role closure changed")

    tensors: list[LTX23MediaTensorPlan] = []
    with safe_open(str(stored_artifact.identity.path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(stored_artifact.roles):
            raise ValueError("LTX media artifact key set changed while opening header")
        for source, target in sorted(converted.items()):
            if target not in shell_state:
                raise ValueError(f"LTX {component} shell is missing target {target!r}")
            view = handle.get_slice(source)
            shape = tuple(view.get_shape())
            dtype = view.get_dtype()
            target_tensor = shell_state[target]
            if shape != tuple(target_tensor.shape):
                raise ValueError(f"LTX {component} shell tensor differs for {source!r} -> {target!r}")
            tensors.append(LTX23MediaTensorPlan(source, target, shape, dtype))
    targets = {item.target for item in tensors}
    if targets != set(shell_state):
        raise ValueError(
            f"LTX {component} shell closure differs: "
            f"missing={sorted(set(shell_state) - targets)[:3]}, extra={sorted(targets - set(shell_state))[:3]}"
        )
    fingerprint = _fingerprint(
        {"stored": stored_artifact.fingerprint, "component": component,
         "shell": _shell_type(shell), "tensors": [(x.source, x.target, x.shape, x.dtype) for x in tensors],
         "ignored": sorted(ignored)}
    )
    return LTX23MediaComponentPlan(stored_artifact, component, _shell_type(shell), tuple(tensors), tuple(sorted(ignored)), fingerprint)


def materialize_ltx23_media_component(shell: nn.Module, plan: LTX23MediaComponentPlan) -> nn.Module:
    """Stream exactly one planned component into its CPU shell, without copies."""

    _validate_materialization_shell(shell, plan)
    if not plan.stored_artifact.revalidate():
        raise ValueError("LTX media artifact changed before materialization")
    try:
        with safe_open(str(plan.stored_artifact.identity.path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(plan.stored_artifact.roles):
                raise ValueError("LTX media artifact key set changed during materialization")
            for item in plan.tensors:
                value = handle.get_tensor(item.source)
                if tuple(value.shape) != item.shape or _dtype_name(value.dtype) != item.dtype:
                    raise ValueError(f"LTX media payload differs from header for {item.source!r}")
                _assign_payload_tensor(shell, item.target, value)
        unresolved = [name for name, value in shell.state_dict().items() if value.is_meta]
        if unresolved:
            raise RuntimeError(f"LTX {plan.component} materialization left meta state: {unresolved[:3]}")
        wrong_dtype = [
            item.target for item in plan.tensors if _dtype_name(shell.state_dict()[item.target].dtype) != item.dtype
        ]
        if wrong_dtype:
            raise RuntimeError(f"LTX {plan.component} materialization changed stored dtypes: {wrong_dtype[:3]}")
        shell._latentslate_ltx23_media = {  # type: ignore[attr-defined]
            "component": plan.component, "plan_fingerprint": plan.fingerprint,
            "source_count": plan.source_count, "device": "cpu",
        }
        return shell
    except BaseException as exc:
        shell._latentslate_ltx23_media_poisoned = f"{type(exc).__name__}: {exc}"  # type: ignore[attr-defined]
        raise


def ltx23_media_component_residency(shell: nn.Module) -> str:
    """Report a component's sole parameter/buffer residency, failing on mixed state."""

    devices = {value.device.type for value in shell.state_dict().values()}
    if not devices:
        return "empty"
    if len(devices) != 1:
        raise RuntimeError(f"LTX media component has mixed residency: {sorted(devices)}")
    return devices.pop()


def unload_ltx23_media_component(shell: nn.Module) -> None:
    """Release one independently-loaded component by returning it to meta state."""

    if ltx23_media_component_residency(shell) == "meta":
        return
    shell.to_empty(device="meta")
    if ltx23_media_component_residency(shell) != "meta":
        raise RuntimeError("LTX media component did not unload to meta")


def _convert_source(component: LTX23MediaComponent, source: str, ignored: set[str]) -> str | None:
    if component == "latent_upsampler":
        return source
    prefix = {"video_vae": "vae.", "audio_vae": "audio_vae.", "vocoder": "vocoder."}[component]
    if not source.startswith(prefix):
        raise ValueError(f"LTX {component} source prefix changed: {source!r}")
    key = source.removeprefix(prefix)
    renames = _VIDEO_RENAMES if component == "video_vae" else _AUDIO_RENAMES if component == "audio_vae" else _VOCODER_RENAMES
    for before, after in renames:
        key = key.replace(before, after)
    if component == "vocoder" and ".ups." in key:
        key = key.replace(".ups.", ".upsamplers.")
    return key


def _validate_materialization_shell(shell: nn.Module, plan: LTX23MediaComponentPlan) -> None:
    if _shell_type(shell) != plan.shell_type:
        raise TypeError("LTX media shell type differs from its plan")
    state = shell.state_dict()
    expected = {item.target: item for item in plan.tensors}
    if set(state) != set(expected):
        raise ValueError("LTX media shell topology differs from its plan")
    for target, item in expected.items():
        value = state[target]
        if not value.is_meta or tuple(value.shape) != item.shape:
            raise ValueError(f"LTX media shell state differs for {target!r}")


def _assign_payload_tensor(shell: nn.Module, target: str, value: torch.Tensor) -> None:
    """Assign one payload without the shell dtype coercion or a second copy."""

    parent_path, _, leaf = target.rpartition(".")
    parent = shell.get_submodule(parent_path) if parent_path else shell
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        raise AttributeError(f"LTX media target disappeared: {target!r}")


def _dtype_name(dtype: torch.dtype) -> str:
    values = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16", torch.float64: "F64"}
    try:
        return values[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported LTX media dtype {dtype}") from exc


def _shell_type(shell: nn.Module) -> str:
    return f"{type(shell).__module__}.{type(shell).__qualname__}"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(encoded.encode()).hexdigest()
