"""Exact CPU/header contracts for the Engine-native Wan 2.2 TI2V 5B path.

The split artifacts originate from pinned source repositories.  Pinned workflow
and node code define behavior, but this module only builds ordinary Diffusers
shells and direct Engine materializers.  It never imports or invokes a workflow
runtime.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from ..stored_quant import read_safetensors_header
from .kit import stable_fingerprint
from .wan21_vae_adapter import build_wan21_vae_skeleton
from .wan22_stored_adapter import (
    build_wan_transformer_skeleton,
    map_stored_wan_parameter_key,
)

Wan5StoredRole = Literal["transformer", "vae"]

WAN5_TRANSFORMER_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "in_channels": 48,
        "out_channels": 48,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 14336,
        "num_layers": 30,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
    }
)

WAN5_VAE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "base_dim": 160,
        "decoder_base_dim": 256,
        "z_dim": 48,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "latents_mean": [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.157,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.123,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.052,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ],
        "latents_std": [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.499,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.06,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ],
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_temporal": 4,
        "scale_factor_spatial": 16,
    }
)


@dataclass(frozen=True, slots=True)
class Wan5ArtifactContract:
    role: Wan5StoredRole
    filename: str
    size_bytes: int
    source_sha256: str
    header_sha256: str
    schema_sha256: str
    tensor_count: int
    architecture: str


WAN5_TRANSFORMER = Wan5ArtifactContract(
    "transformer",
    "wan2.2_ti2v_5B_fp16.safetensors",
    9_999_658_848,
    "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e",
    "7b99ff54748ad753007022b026bb8a1b0ddea568ca3a97b40ef6b12c6d16ae52",
    "5317bf88f8ab6a8acdc58e697c954a43aceecc7b658735e81dccc308af59ef90",
    825,
    "wan22_ti2v_5b_48ch_30block",
)
WAN5_VAE = Wan5ArtifactContract(
    "vae",
    "wan2.2_vae.safetensors",
    1_409_400_960,
    "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
    "9f909558eb58ea4d75c9aefe3cef7e54ecd636e02d555f3cbc6225654b29587c",
    "f01c9f6cada88c48a74a8b14f129bc75c3d1b7e36a3c3aeaf45ff4f9b1b1b8e9",
    196,
    "wan_vae_2_2_48ch",
)


@dataclass(frozen=True, slots=True)
class Wan5StoredPlan:
    contract: Wan5ArtifactContract
    identity: ArtifactIdentity
    config_fingerprint: str
    mapping_fingerprint: str
    source_to_target: Mapping[str, str]
    errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.errors

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            f"wan5-{self.contract.role}-plan",
            {
                "contract": self.contract,
                "identity": self.identity,
                "config": self.config_fingerprint,
                "mapping": self.mapping_fingerprint,
            },
        )

    def require_available(self) -> None:
        if self.errors:
            raise ValueError(
                f"Wan 5B {self.contract.role} contract is unavailable: " + "; ".join(self.errors)
            )

    def revalidate(self) -> bool:
        if not revalidate_artifact(self.identity):
            return False
        try:
            refreshed = plan_wan5_stored_artifact(self.identity.path, self.contract)
        except (OSError, TypeError, ValueError):
            return False
        return refreshed == self


def plan_wan5_stored_artifact(path: Path, contract: Wan5ArtifactContract) -> Wan5StoredPlan:
    source = Path(path).resolve(strict=True)
    probe = probe_artifact(source)
    errors = list(_identity_errors(source, probe, contract))
    header = read_safetensors_header(source, probe.identity.size_bytes)
    if contract.role == "transformer":
        config = WAN5_TRANSFORMER_CONFIG
        shell = build_wan_transformer_skeleton(config)
        mapper = map_stored_wan_parameter_key
    elif contract.role == "vae":
        config = WAN5_VAE_CONFIG
        shell = build_wan21_vae_skeleton(config)
        mapper = map_wan5_vae_key
    else:
        raise ValueError(f"unsupported Wan 5B stored role {contract.role!r}")

    expected = {name: tuple(value.shape) for name, value in shell.state_dict().items()}
    mapping: dict[str, str] = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        target = mapper(key)
        if target is None:
            errors.append(f"unmapped source tensor {key!r}")
            continue
        mapping[key] = target
        if target not in expected:
            errors.append(f"mapped target is absent: {target!r}")
            continue
        if entry.get("dtype") != "F16" or tuple(entry.get("shape", ())) != expected[target]:
            errors.append(f"stored dtype/shape mismatch for {key!r}")

    target_counts = Counter(mapping.values())
    missing = sorted(set(expected) - set(target_counts))
    duplicates = sorted(target for target, count in target_counts.items() if count != 1)
    if missing:
        errors.append(f"missing shell targets: {len(missing)}")
    if duplicates:
        errors.append(f"duplicate shell targets: {len(duplicates)}")
    config_fingerprint = stable_fingerprint(f"wan5-{contract.role}-config", dict(config))
    mapping_fingerprint = stable_fingerprint(
        f"wan5-{contract.role}-mapping", sorted(mapping.items())
    )
    return Wan5StoredPlan(
        contract,
        probe.identity,
        config_fingerprint,
        mapping_fingerprint,
        MappingProxyType(dict(sorted(mapping.items()))),
        tuple(errors),
    )


def materialize_wan5_stored_artifact(
    plan: Wan5StoredPlan, *, compute_dtype: torch.dtype = torch.float16
) -> nn.Module:
    """Direct-assign one exact FP16 payload into its validated meta shell."""

    from safetensors import safe_open

    plan.require_available()
    if compute_dtype is not torch.float16:
        raise ValueError("Wan 5B stored artifacts require their native FP16 compute dtype")
    config = WAN5_TRANSFORMER_CONFIG if plan.contract.role == "transformer" else WAN5_VAE_CONFIG
    expected_config = stable_fingerprint(f"wan5-{plan.contract.role}-config", dict(config))
    expected_mapping = stable_fingerprint(
        f"wan5-{plan.contract.role}-mapping", sorted(plan.source_to_target.items())
    )
    if plan.config_fingerprint != expected_config or plan.mapping_fingerprint != expected_mapping:
        raise ValueError("Wan 5B stored plan fingerprint changed before materialization")
    module = (
        build_wan_transformer_skeleton(config)
        if plan.contract.role == "transformer"
        else build_wan21_vae_skeleton(config)
    )
    consumed: set[str] = set()
    tensor = None
    try:
        with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
            if not revalidate_artifact(plan.identity) or set(handle.keys()) != set(
                plan.source_to_target
            ):
                raise ValueError("Wan 5B stored artifact changed before payload read")
            for source, target in plan.source_to_target.items():
                tensor = handle.get_tensor(source)
                if tensor.dtype is not torch.float16:
                    raise ValueError("Wan 5B stored payload is not FP16")
                _assign_tensor(module, target, tensor)
                consumed.add(source)
        if consumed != set(plan.source_to_target):
            raise ValueError("Wan 5B stored payload consumption is incomplete")
        state = dict(module.named_parameters()) | dict(module.named_buffers())
        if any(value.is_meta for value in state.values()):
            raise ValueError("Wan 5B materializer left meta state")
        module._latentslate_wan5_contract = plan.contract.role
        module._latentslate_wan5_plan_fingerprint = plan.fingerprint
        module._latentslate_wan5_artifact_identity = plan.identity
        module.eval()
        module.requires_grad_(False)
        return module
    except BaseException:
        _dematerialize(module)
        tensor = None
        raise


def map_wan5_vae_key(key: str) -> str | None:
    """Map one pinned Wan 2.2 VAE source key to the Diffusers shell."""

    static = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
        "encoder.conv1.weight": "encoder.conv_in.weight",
        "encoder.conv1.bias": "encoder.conv_in.bias",
        "decoder.conv1.weight": "decoder.conv_in.weight",
        "decoder.conv1.bias": "decoder.conv_in.bias",
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.weight": "decoder.conv_out.weight",
        "decoder.head.2.bias": "decoder.conv_out.bias",
    }
    if key in static:
        return static[key]
    for side in ("encoder", "decoder"):
        if key.startswith(f"{side}.middle.0.residual."):
            return _map_vae_residual(
                key.replace(f"{side}.middle.0.", f"{side}.mid_block.resnets.0.")
            )
        if key.startswith(f"{side}.middle.2.residual."):
            return _map_vae_residual(
                key.replace(f"{side}.middle.2.", f"{side}.mid_block.resnets.1.")
            )
        if key.startswith(f"{side}.middle.1."):
            return key.replace(f"{side}.middle.1.", f"{side}.mid_block.attentions.0.").replace(
                ".norm.", ".norm."
            )
    if key.startswith("encoder.downsamples."):
        parts = key.split(".")
        if len(parts) < 6 or not parts[2].isdigit() or not parts[4].isdigit():
            return None
        group = int(parts[2])
        item = int(parts[4])
        suffix = ".".join(parts[5:])
        if suffix.startswith(("residual.", "shortcut.")):
            return _map_vae_residual(f"encoder.down_blocks.{group}.resnets.{item}.{suffix}")
        if item == 2 and suffix.startswith(("resample.", "time_conv.")):
            return f"encoder.down_blocks.{group}.downsampler.{suffix}"
        return None
    if not key.startswith("decoder.upsamples."):
        return None
    parts = key.split(".")
    if len(parts) < 6 or not parts[2].isdigit() or not parts[4].isdigit():
        return None
    group = int(parts[2])
    item = int(parts[4])
    suffix = ".".join(parts[5:])
    if suffix.startswith(("residual.", "shortcut.")):
        return _map_vae_residual(f"decoder.up_blocks.{group}.resnets.{item}.{suffix}")
    if item == 3 and suffix.startswith(("resample.", "time_conv.")):
        return f"decoder.up_blocks.{group}.upsampler.{suffix}"
    return None


def _map_vae_residual(key: str) -> str:
    return (
        key.replace("residual.0.", "norm1.")
        .replace("residual.2.", "conv1.")
        .replace("residual.3.", "norm2.")
        .replace("residual.6.", "conv2.")
        .replace(".shortcut.", ".conv_shortcut.")
    )


def _identity_errors(source: Path, probe: Any, contract: Wan5ArtifactContract) -> tuple[str, ...]:
    errors: list[str] = []
    checks = (
        (source.name == contract.filename, "filename"),
        (probe.identity.size_bytes == contract.size_bytes, "size"),
        (probe.identity.header_sha256 == contract.header_sha256, "header"),
        (probe.schema_sha256 == contract.schema_sha256, "schema"),
        (probe.tensor_count == contract.tensor_count, "tensor count"),
        (probe.quantization_contract == "native/fp16", "precision"),
        (probe.tensor_dtypes == ("F16",), "tensor dtype"),
        (probe.architecture_signals == (contract.architecture,), "architecture"),
        (probe.component_signals == (contract.role,), "component"),
    )
    for valid, label in checks:
        if not valid:
            errors.append(f"{label} differs from the pinned {contract.role} contract")
    return tuple(errors)


def _assign_tensor(root: nn.Module, target: str, tensor: torch.Tensor) -> None:
    parent_path, separator, name = target.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    current = getattr(parent, name, None)
    if not isinstance(current, torch.Tensor) or tuple(current.shape) != tuple(tensor.shape):
        raise ValueError(f"Wan 5B target changed: {target!r}")
    if name in parent._parameters:
        parent._parameters[name] = nn.Parameter(tensor, requires_grad=False)
    elif name in parent._buffers:
        parent._buffers[name] = tensor
    else:
        raise ValueError(f"Wan 5B target is not registered state: {target!r}")


def _dematerialize(root: nn.Module) -> None:
    for module in root.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is not None:
                module._parameters[name] = nn.Parameter(
                    torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
                    requires_grad=False,
                )
        for name, buffer in tuple(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = torch.empty(
                    tuple(buffer.shape), dtype=buffer.dtype, device="meta"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
