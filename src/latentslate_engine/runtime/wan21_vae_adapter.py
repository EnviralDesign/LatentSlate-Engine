"""Exact BF16 Comfy Wan 2.1 VAE header plan and CPU materializer."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_safetensors, revalidate_artifact
from .wan22_stored_adapter import _read_safetensors_header

WAN21_VAE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "base_dim": 96,
        "decoder_base_dim": None,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "z_dim": 16,
        "in_channels": 3,
        "out_channels": 3,
        "scale_factor_temporal": 4,
        "scale_factor_spatial": 8,
        "latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.916,
        ],
    }
)


@dataclass(frozen=True, slots=True)
class WanVaeSemantics:
    latent_channels: int = 16
    temporal_ratio: int = 4
    spatial_ratio: int = 8
    rgb_channels: int = 3
    normalization: str = "diffusers_autoencoderklwan_mean_std"
    mean: tuple[float, ...] = ()
    std_values: tuple[float, ...] = ()

    def normalize(
        self, latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        _validate_latent_statistics(latents, mean, std, self)
        mean, std = _broadcast_statistics(latents, mean, std)
        return (latents - mean) / std

    def denormalize(
        self, latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        _validate_latent_statistics(latents, mean, std, self)
        mean, std = _broadcast_statistics(latents, mean, std)
        return latents * std + mean


@dataclass(frozen=True, slots=True)
class WanVaePlan:
    identity: ArtifactIdentity
    config_fingerprint: str
    mapping_fingerprint: str
    source_to_target: Mapping[str, str]
    missing: tuple[str, ...]
    duplicate_targets: tuple[str, ...]
    extras: tuple[str, ...]
    mismatches: tuple[str, ...]
    semantics: WanVaeSemantics = WanVaeSemantics()

    @property
    def available(self) -> bool:
        return not (self.missing or self.duplicate_targets or self.extras or self.mismatches)

    def require_available(self) -> None:
        if not self.available:
            raise ValueError(
                f"Wan VAE unavailable: missing={len(self.missing)}, duplicate={len(self.duplicate_targets)}, extras={len(self.extras)}, mismatches={len(self.mismatches)}"
            )


def build_wan21_vae_skeleton(config: Mapping[str, Any] = WAN21_VAE_CONFIG) -> nn.Module:
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLWan

    with init_empty_weights():
        return AutoencoderKLWan(**dict(config))


def map_comfy_wan21_vae_key(key: str) -> str | None:
    root = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }
    if key in root:
        return root[key]
    for side in ("encoder", "decoder"):
        if key == f"{side}.conv1.weight":
            return f"{side}.conv_in.weight"
        if key == f"{side}.conv1.bias":
            return f"{side}.conv_in.bias"
        if key.startswith(f"{side}.head.0."):
            return key.replace(f"{side}.head.0.", f"{side}.norm_out.")
        if key.startswith(f"{side}.head.2."):
            return key.replace(f"{side}.head.2.", f"{side}.conv_out.")
        if key.startswith(f"{side}.middle.0.residual."):
            return _map_residual(key.replace(f"{side}.middle.0.", f"{side}.mid_block.resnets.0."))
        if key.startswith(f"{side}.middle.2.residual."):
            return _map_residual(key.replace(f"{side}.middle.2.", f"{side}.mid_block.resnets.1."))
        if key.startswith(f"{side}.middle.1."):
            return key.replace(f"{side}.middle.1.", f"{side}.mid_block.attentions.0.")
    if key.startswith("encoder.downsamples."):
        prefix, rest = key.split(".", 3)[0:3], key.split(".", 3)[3]
        return _map_residual(f"encoder.down_blocks.{prefix[2]}.{rest}")
    if key.startswith("decoder.upsamples."):
        parts = key.split(".", 3)
        index = int(parts[2])
        rest = parts[3]
        group, pos = divmod(index, 4)
        base = f"decoder.up_blocks.{group}."
        return (
            _map_residual(base + f"resnets.{pos}.{rest}")
            if pos < 3
            else base + f"upsamplers.0.{rest}"
        )
    return None


def _map_residual(key: str) -> str:
    return (
        key.replace("residual.0.", "norm1.")
        .replace("residual.2.", "conv1.")
        .replace("residual.3.", "norm2.")
        .replace("residual.6.", "conv2.")
        .replace(".shortcut.", ".conv_shortcut.")
    )


def comfy_key_for_wan21_vae_target(target: str) -> str | None:
    """Inverse canonical map used only to construct bounded synthetic fixtures."""
    root = {
        "quant_conv.weight": "conv1.weight",
        "quant_conv.bias": "conv1.bias",
        "post_quant_conv.weight": "conv2.weight",
        "post_quant_conv.bias": "conv2.bias",
    }
    if target in root:
        return root[target]
    for side in ("encoder", "decoder"):
        if target.startswith(f"{side}.conv_in."):
            return target.replace(f"{side}.conv_in.", f"{side}.conv1.")
        if target.startswith(f"{side}.norm_out."):
            return target.replace(f"{side}.norm_out.", f"{side}.head.0.")
        if target.startswith(f"{side}.conv_out."):
            return target.replace(f"{side}.conv_out.", f"{side}.head.2.")
        for mid, res in ((0, 0), (1, 2)):
            p = f"{side}.mid_block.resnets.{mid}."
            if target.startswith(p):
                return _unmap_residual(target.replace(p, f"{side}.middle.{res}."))
        p = f"{side}.mid_block.attentions.0."
        if target.startswith(p):
            return target.replace(p, f"{side}.middle.1.")
    if target.startswith("encoder.down_blocks."):
        parts = target.split(".", 3)
        return _unmap_residual(f"encoder.downsamples.{parts[2]}.{parts[3]}")
    if target.startswith("decoder.up_blocks."):
        parts = target.split(".", 4)
        group = int(parts[2])
        kind = parts[3]
        rest = parts[4]
        if kind == "resnets":
            return _unmap_residual(
                f"decoder.upsamples.{group * 4 + int(rest.split('.', 1)[0])}.{rest.split('.', 1)[1]}"
            )
        if kind == "upsamplers":
            return f"decoder.upsamples.{group * 4 + 3}.{rest.split('.', 1)[1]}"
    return None


def _unmap_residual(key: str) -> str:
    return (
        key.replace("norm1.", "residual.0.")
        .replace("conv1.", "residual.2.")
        .replace("norm2.", "residual.3.")
        .replace("conv2.", "residual.6.")
        .replace(".conv_shortcut.", ".shortcut.")
    )


def plan_comfy_wan21_vae(path: Path, config: Mapping[str, Any] = WAN21_VAE_CONFIG) -> WanVaePlan:
    path = Path(path).resolve(strict=True)
    probe = probe_safetensors(path)
    if probe.architecture_signals != ("wan_vae_2_1",) or probe.tensor_dtypes != ("BF16",):
        raise ValueError("Wan VAE requires the exact BF16 wan_vae_2_1 artifact")
    header = _read_safetensors_header(path, probe.identity.size_bytes)
    shell = build_wan21_vae_skeleton(config)
    expected = {k: tuple(v.shape) for k, v in shell.state_dict().items()}
    mapping = {k: map_comfy_wan21_vae_key(k) for k in header if k != "__metadata__"}
    extras = tuple(sorted(k for k, v in mapping.items() if v is None))
    mapping = {k: v for k, v in mapping.items() if v is not None}
    mismatches = tuple(
        sorted(
            k
            for k, v in mapping.items()
            if v not in expected
            or tuple(header[k].get("shape", ())) != expected[v]
            or header[k].get("dtype") != "BF16"
        )
    )
    targets = {}
    for source, target in mapping.items():
        targets.setdefault(target, []).append(source)
    duplicate = tuple(sorted(target for target, sources in targets.items() if len(sources) != 1))
    missing = tuple(sorted(set(expected) - set(mapping.values())))
    fp = _fingerprint(
        config,
        {
            **mapping,
            "__duplicates__": "|".join(duplicate),
            "__semantics__": str(_derive_semantics(config)),
        },
    )
    return WanVaePlan(
        probe.identity,
        _fingerprint(config, {}),
        fp,
        MappingProxyType(dict(sorted(mapping.items()))),
        missing,
        duplicate,
        extras,
        mismatches,
        _derive_semantics(config),
    )


def materialize_wan21_vae(
    plan: WanVaePlan, config: Mapping[str, Any], *, compute_dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    from safetensors import safe_open

    plan.require_available()
    expected_fp = _fingerprint(
        config,
        {
            **plan.source_to_target,
            "__duplicates__": "|".join(plan.duplicate_targets),
            "__semantics__": str(_derive_semantics(config)),
        },
    )
    if (
        compute_dtype != torch.bfloat16
        or plan.config_fingerprint != _fingerprint(config, {})
        or plan.mapping_fingerprint != expected_fp
        or plan.semantics != _derive_semantics(config)
    ):
        raise ValueError("Wan VAE materializer requires its validated BF16 plan/config")
    vae = build_wan21_vae_skeleton(config)
    consumed = set()
    tensor = None
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as h:
            if not revalidate_artifact(plan.identity):
                raise ValueError("Wan VAE identity changed before materialization")
            if set(h.keys()) != set(plan.source_to_target):
                raise ValueError("Wan VAE source schema changed")
            for source, target in plan.source_to_target.items():
                tensor = h.get_tensor(source)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError("Wan VAE requires stored BF16 tensors")
                parent_path, _, attr = target.rpartition(".")
                parent = vae.get_submodule(parent_path) if parent_path else vae
                current = getattr(parent, attr, None)
                if not isinstance(current, torch.Tensor) or tuple(current.shape) != tuple(
                    tensor.shape
                ):
                    raise ValueError("Wan VAE target shape changed")
                if attr in parent._parameters:
                    parent._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
                elif attr in parent._buffers:
                    parent._buffers[attr] = tensor
                else:
                    raise ValueError("Wan VAE target is not state")
                consumed.add(source)
        if consumed != set(plan.source_to_target) or any(x.is_meta for x in vae.parameters()):
            raise ValueError("Wan VAE materialization incomplete")
        vae._latentslate_vae_config_fingerprint = plan.config_fingerprint
        vae._latentslate_vae_mapping_fingerprint = plan.mapping_fingerprint
        return vae
    except BaseException:
        for module in vae.modules():
            for name, p in tuple(module._parameters.items()):
                if p is not None:
                    module._parameters[name] = nn.Parameter(
                        torch.empty(tuple(p.shape), dtype=p.dtype, device="meta"),
                        requires_grad=False,
                    )
        tensor = None
        raise


class WanVaeResidencySession:
    def __init__(
        self,
        vae: nn.Module,
        plan: WanVaePlan,
        *,
        onload_device: torch.device | str,
        offload_device: torch.device | str = "cpu",
    ):
        from . import wan22_stored_adapter as wan

        self.vae = vae
        plan.require_available()
        if (
            getattr(vae, "_latentslate_vae_config_fingerprint", None) != plan.config_fingerprint
            or getattr(vae, "_latentslate_vae_mapping_fingerprint", None)
            != plan.mapping_fingerprint
        ):
            raise ValueError("Wan VAE residency plan does not match the materialized VAE")
        self._semantics = plan.semantics
        self._wan = wan
        self.onload = wan._canonicalize_residency_device(torch.device(onload_device))
        self.offload = torch.device(offload_device)
        if self.offload.type != "cpu":
            raise ValueError("Wan VAE residency requires CPU offload")
        self.snapshot = {
            n: x.dtype
            for n, x in (dict(vae.named_parameters()) | dict(vae.named_buffers())).items()
        }
        self._owner = None
        self._closed = False
        self._entered = False
        self._executing = False
        self._lock = threading.RLock()
        if not self.snapshot or any(
            x.is_meta for x in (dict(vae.named_parameters()) | dict(vae.named_buffers())).values()
        ):
            raise ValueError("Wan VAE residency requires materialized state")

    def __enter__(self):
        with self._lock:
            if self._closed or self._entered:
                raise RuntimeError("Wan VAE residency is one-shot")
            with self._wan._WAN_SESSION_GUARD_LOCK:
                if self._wan._ACTIVE_WAN_SESSION is not None:
                    raise RuntimeError("a Wan/VAE residency session is already active process-wide")
                self._wan._ACTIVE_WAN_SESSION = self
            self._owner = threading.get_ident()
            try:
                self._move(self.onload)
                self._assert(self.onload)
                self._entered = True
                return self
            except BaseException:
                self._teardown(True)
                raise

    def __exit__(self, *_):
        self.close()
        return False

    def close(self):
        with self._lock:
            if self._closed:
                return
            if threading.get_ident() != self._owner:
                raise RuntimeError("Wan VAE residency close must run on owning thread")
            if self._executing:
                raise RuntimeError("cannot close Wan VAE residency while encode/decode is active")
            self._teardown(False)

    @property
    def active(self):
        return self._entered and not self._closed

    def encode(self, video):
        return self._run(lambda: encode_wan21_latents(self.vae, video, self._semantics))

    def decode(self, latents):
        return self._run(lambda: decode_wan21_latents(self.vae, latents, self._semantics))

    def _run(self, operation):
        with self._lock:
            if not self.active or threading.get_ident() != self._owner:
                raise RuntimeError("Wan VAE encode/decode requires active owning residency")
            if self._executing:
                raise RuntimeError("Wan VAE encode/decode is non-reentrant")
            self._executing = True
        try:
            return operation()
        except BaseException:
            self._teardown(True)
            raise
        finally:
            with self._lock:
                self._executing = False

    def _move(self, device):
        self.vae.to(device=device)

    def _assert(self, device):
        state = dict(self.vae.named_parameters()) | dict(self.vae.named_buffers())
        if set(state) != set(self.snapshot) or any(
            x.is_meta or x.dtype != self.snapshot[n] or x.device != device for n, x in state.items()
        ):
            raise RuntimeError("Wan VAE residency state/device changed")

    def _teardown(self, suppress):
        error = None
        try:
            self._move(self.offload)
            self._assert(self.offload)
        except BaseException as exc:  # noqa: BLE001 - cleanup must release the global guard
            error = exc
        finally:
            self._entered = False
            self._closed = True
            with self._wan._WAN_SESSION_GUARD_LOCK:
                if self._wan._ACTIVE_WAN_SESSION is self:
                    self._wan._ACTIVE_WAN_SESSION = None
        if error and not suppress:
            raise RuntimeError("Wan VAE residency teardown failed") from error


def configure_wan21_vae_memory(
    vae: nn.Module, *, tiling: bool = False, slicing: bool = False
) -> None:
    """Use only the pinned Diffusers VAE public tiling/slicing controls."""
    if not isinstance(tiling, bool) or not isinstance(slicing, bool):
        raise TypeError("Wan VAE tiling and slicing must be bool")
    for enabled, name in ((tiling, "tiling"), (slicing, "slicing")):
        enable = getattr(vae, f"enable_{name}", None)
        disable = getattr(vae, f"disable_{name}", None)
        if not callable(enable) or not callable(disable):
            raise TypeError(f"Wan VAE does not expose public {name} controls")
        (enable if enabled else disable)()


def encode_wan21_latents(vae, video, semantics):
    """Causal VAE encode boundary: raw video to normalized 16-channel latents."""
    if video.ndim != 5 or video.shape[1] != semantics.rgb_channels:
        raise ValueError("Wan VAE video must be [B,3,T,H,W]")
    if video.shape[2] < 1 or video.shape[3] < 1 or video.shape[4] < 1:
        raise ValueError("Wan VAE video dimensions must be positive")
    if (video.shape[2] - 1) % semantics.temporal_ratio:
        raise ValueError(
            f"Wan VAE video frame count must be {semantics.temporal_ratio}k+1 for causal round-trip"
        )
    if video.shape[3] % semantics.spatial_ratio or video.shape[4] % semantics.spatial_ratio:
        raise ValueError("Wan VAE video height and width must match the spatial ratio")
    latent = vae.encode(video).latent_dist.mode()
    expected = (
        video.shape[0],
        semantics.latent_channels,
        ((video.shape[2] - 1) // semantics.temporal_ratio) + 1,
        video.shape[3] // semantics.spatial_ratio,
        video.shape[4] // semantics.spatial_ratio,
    )
    if tuple(latent.shape) != expected:
        raise RuntimeError(
            f"Wan VAE encoded an unexpected latent shape: {tuple(latent.shape)} != {expected}"
        )
    return semantics.normalize(
        latent,
        torch.tensor(semantics.mean),
        torch.tensor(semantics.std_values),
    )


def decode_wan21_latents(vae, latents, semantics):
    """Causal VAE decode boundary: normalized latents back to RGB video."""
    _validate_latent_statistics(
        latents,
        torch.tensor(semantics.mean),
        torch.tensor(semantics.std_values),
        semantics,
    )
    video = vae.decode(
        semantics.denormalize(
            latents,
            torch.tensor(semantics.mean),
            torch.tensor(semantics.std_values),
        )
    ).sample
    expected = (
        latents.shape[0],
        semantics.rgb_channels,
        ((latents.shape[2] - 1) * semantics.temporal_ratio) + 1,
        latents.shape[3] * semantics.spatial_ratio,
        latents.shape[4] * semantics.spatial_ratio,
    )
    if tuple(video.shape) != expected:
        raise RuntimeError(
            f"Wan VAE decoded an unexpected video shape: {tuple(video.shape)} != {expected}"
        )
    return video


def _validate_latent_statistics(latents, mean, std, semantics):
    if latents.ndim != 5 or latents.shape[1] != semantics.latent_channels:
        raise ValueError("Wan VAE latents must be [B,16,T,H,W]")
    if (
        mean.numel() != semantics.latent_channels
        or std.numel() != semantics.latent_channels
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(std).all())
        or bool((std <= 0).any())
    ):
        raise ValueError(
            "Wan VAE latent mean/std must be finite 16-channel values with positive std"
        )


def _broadcast_statistics(latents, mean, std):
    return mean.to(device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1, 1), std.to(
        device=latents.device, dtype=latents.dtype
    ).reshape(1, -1, 1, 1, 1)


def _derive_semantics(config):
    z = int(config.get("z_dim", 0))
    dims = config.get("dim_mult")
    temporal = config.get("temperal_downsample")
    if (
        z != 16
        or config.get("in_channels") != 3
        or config.get("out_channels") != 3
        or not isinstance(dims, list)
        or len(dims) < 1
        or not isinstance(temporal, list)
        or len(temporal) != len(dims) - 1
        or any(type(value) is not bool for value in temporal)
    ):
        raise ValueError("Wan VAE config is incompatible with 16-channel Wan 2.1 semantics")
    mean, std = config.get("latents_mean"), config.get("latents_std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 16
        or len(std) != 16
        or not all(isinstance(value, (int, float)) for value in mean + std)
        or not all(value > 0 for value in std)
    ):
        raise ValueError("Wan VAE requires finite positive 16-channel latent statistics")
    temporal_ratio, spatial_ratio = 2 ** sum(temporal), 2 ** (len(dims) - 1)
    if (
        config.get("scale_factor_temporal") != temporal_ratio
        or config.get("scale_factor_spatial") != spatial_ratio
    ):
        raise ValueError("Wan VAE scale factors must equal architecture ratios")
    return WanVaeSemantics(
        z,
        temporal_ratio,
        spatial_ratio,
        3,
        "diffusers_autoencoderklwan_mean_std",
        tuple(mean),
        tuple(std),
    )


def _fingerprint(config: Mapping[str, Any], mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"config": dict(config), "mapping": sorted(mapping.items())},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
