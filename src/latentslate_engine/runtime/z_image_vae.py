"""Exact Flux.1 AE plan and materializer for Z-Image Turbo."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from ..z_image_turbo_recipe import _read_z_safetensors_header

Z_IMAGE_FLUX_AE_CONFIG = MappingProxyType(
    {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,
        "latent_channels": 16,
        "norm_num_groups": 32,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }
)
_Z_IMAGE_VAE_HEADER_SHA256 = "6753860d781c5040a82e9aee0726719966ae774c1513d38789b264b30c496a39"
_Z_IMAGE_VAE_TENSOR_COUNT = 244
_Z_IMAGE_ATTENTION_1X1_LINEAR_SOURCES = frozenset(
    f"{stage}.mid.attn_1.{projection}.weight"
    for stage in ("encoder", "decoder")
    for projection in ("q", "k", "v", "proj_out")
)


@dataclass(frozen=True, slots=True)
class ZImageVaePlan:
    identity: ArtifactIdentity
    header_sha256: str
    schema_sha256: str
    source_to_target: Mapping[str, str]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ZImagePngArtifact:
    path: Path
    width: int
    height: int
    size_bytes: int
    sha256: str


class ZImageDecodeCancelled(RuntimeError):
    pass


class ZImagePngPublicationCancelled(RuntimeError):
    pass


def build_z_image_flux_ae_shell():
    """Return a meta-device Diffusers AutoencoderKL shell with Flux.1 facts."""

    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL

    with init_empty_weights():
        return AutoencoderKL(**dict(Z_IMAGE_FLUX_AE_CONFIG))


def plan_z_image_flux_ae(path: Path) -> ZImageVaePlan:
    probe = probe_artifact(path)
    if probe.format != "safetensors":
        raise ValueError("Z-Image AE is not SafeTensors")
    raw, header = _read_z_safetensors_header(probe.identity.path, probe.identity.size_bytes)
    header_sha256 = hashlib.sha256(raw).hexdigest()
    if header_sha256 != _Z_IMAGE_VAE_HEADER_SHA256:
        raise ValueError("Z-Image AE header differs from the exact Flux.1 mapping")
    sources = {
        key: value
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }
    if len(sources) != _Z_IMAGE_VAE_TENSOR_COUNT or {v.get("dtype") for v in sources.values()} != {
        "F32"
    }:
        raise ValueError("Z-Image AE must contain exactly 244 F32 tensors")
    shell = build_z_image_flux_ae_shell()
    target = shell.state_dict()
    mapping = {source: _map_flux_ae_key(source) for source in sources}
    if (
        None in mapping.values()
        or set(mapping.values()) != set(target)
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError(
            "Z-Image Flux.1 AE does not exactly cover its Diffusers AutoencoderKL shell"
        )
    for source, destination in mapping.items():
        assert destination is not None
        _validate_z_image_flux_ae_shape(
            source,
            tuple(sources[source]["shape"]),
            tuple(target[destination].shape),
        )
    frozen = MappingProxyType(
        {source: destination for source, destination in mapping.items() if destination}
    )
    fingerprint = hashlib.sha256(repr(tuple(sorted(frozen.items()))).encode()).hexdigest()
    return ZImageVaePlan(probe.identity, header_sha256, probe.schema_sha256, frozen, fingerprint)


def revalidate_z_image_flux_ae(plan: ZImageVaePlan) -> bool:
    try:
        return (
            revalidate_artifact(plan.identity) and plan_z_image_flux_ae(plan.identity.path) == plan
        )
    except (OSError, TypeError, ValueError):
        return False


def materialize_z_image_flux_ae(
    plan: ZImageVaePlan,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
):
    """Materialize only exact F32 source tensors into the matching Diffusers shell."""

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    if not revalidate_z_image_flux_ae(plan):
        raise ValueError("Z-Image Flux.1 AE changed after planning")
    model = build_z_image_flux_ae_shell()
    target = model.state_dict()
    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("Z-Image Flux.1 AE changed before materialization")
        total = len(plan.source_to_target)
        for index, (source, destination) in enumerate(plan.source_to_target.items(), 1):
            if cancelled():
                raise RuntimeError("Z-Image AE materialization canceled")
            value = handle.get_tensor(source)
            if value.dtype is not torch.float32:
                raise ValueError(f"Z-Image Flux.1 AE source changed: {source}")
            _validate_z_image_flux_ae_shape(
                source, tuple(value.shape), tuple(target[destination].shape)
            )
            value = _squeeze_z_image_attention_1x1_weight(source, value)
            set_module_tensor_to_device(model, destination, "cpu", value=value, dtype=torch.float32)
            if progress is not None and (index == total or index % 8 == 0):
                progress(index, total)
    if any(value.is_meta for value in model.state_dict().values()):
        raise ValueError("Z-Image Flux.1 AE retains meta parameters")
    model.eval()
    return model


@torch.no_grad()
def decode_z_image_flux_ae(
    model: torch.nn.Module,
    latents: torch.Tensor,
    *,
    cancelled: Callable[[], bool] = lambda: False,
):
    """Decode the exact Flux.1 latent convention into one or more RGB images."""

    from PIL import Image

    if cancelled():
        raise ZImageDecodeCancelled("Z-Image decode canceled before VAE execution")
    if latents.ndim != 4 or latents.shape[1] != 16 or not torch.isfinite(latents).all():
        raise ValueError("Z-Image VAE requires finite BCHW 16-channel latents")
    parameter = next(model.parameters())
    scaled = latents.to(device=parameter.device, dtype=torch.float32)
    scaled = scaled / float(Z_IMAGE_FLUX_AE_CONFIG["scaling_factor"])
    scaled = scaled + float(Z_IMAGE_FLUX_AE_CONFIG["shift_factor"])
    decoded = model.decode(scaled, return_dict=False)[0]
    if cancelled():
        raise ZImageDecodeCancelled("Z-Image decode canceled after VAE execution")
    if decoded.ndim != 4 or decoded.shape[1] != 3 or not torch.isfinite(decoded).all():
        raise ValueError("Z-Image VAE returned invalid RGB samples")
    pixels = ((decoded / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    arrays = pixels.cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def write_z_image_png_atomic(
    image: object,
    output_path: Path,
    *,
    expected_size: tuple[int, int] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> ZImagePngArtifact:
    """Publish one PNG atomically, then report facts observed from final bytes."""

    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("Z-Image PNG output must be a Pillow image")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.convert("RGB").save(temporary, format="PNG")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        temporary_size = temporary.stat().st_size
        temporary_digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        with Image.open(temporary) as observed:
            observed.load()
            width, height = observed.size
            if observed.format != "PNG" or observed.mode != "RGB":
                raise ValueError("Z-Image temporary artifact is not an RGB PNG")
            if expected_size is not None and (width, height) != expected_size:
                raise ValueError("Z-Image temporary PNG dimensions differ from the request")
        if cancelled():
            raise ZImagePngPublicationCancelled(
                "Z-Image generation canceled before atomic PNG publication"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    size = output_path.stat().st_size
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if size != temporary_size or digest != temporary_digest:
        raise OSError("Z-Image published PNG bytes changed during atomic replacement")
    return ZImagePngArtifact(output_path, width, height, size, digest)


def _map_flux_ae_key(source: str) -> str | None:
    """Map Z-Image's Flux.1 AE keys without borrowing another family's private API."""

    direct = {
        "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
        "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
        "encoder.quant_conv.weight": "quant_conv.weight",
        "encoder.quant_conv.bias": "quant_conv.bias",
        "decoder.post_quant_conv.weight": "post_quant_conv.weight",
        "decoder.post_quant_conv.bias": "post_quant_conv.bias",
    }
    if source in direct:
        return direct[source]
    if source.startswith(("encoder.conv_", "decoder.conv_", "bn.")):
        return source

    value = source.replace("nin_shortcut", "conv_shortcut")
    patterns = (
        (
            r"encoder\.down\.(\d+)\.block\.(\d+)\.(.+)",
            lambda m: f"encoder.down_blocks.{m[1]}.resnets.{m[2]}.{m[3]}",
        ),
        (
            r"encoder\.down\.(\d+)\.downsample\.conv\.(.+)",
            lambda m: f"encoder.down_blocks.{m[1]}.downsamplers.0.conv.{m[2]}",
        ),
        (
            r"encoder\.mid\.block_(\d+)\.(.+)",
            lambda m: f"encoder.mid_block.resnets.{int(m[1]) - 1}.{m[2]}",
        ),
        (
            r"decoder\.up\.(\d+)\.block\.(\d+)\.(.+)",
            lambda m: f"decoder.up_blocks.{3 - int(m[1])}.resnets.{m[2]}.{m[3]}",
        ),
        (
            r"decoder\.up\.(\d+)\.upsample\.conv\.(.+)",
            lambda m: f"decoder.up_blocks.{3 - int(m[1])}.upsamplers.0.conv.{m[2]}",
        ),
        (
            r"decoder\.mid\.block_(\d+)\.(.+)",
            lambda m: f"decoder.mid_block.resnets.{int(m[1]) - 1}.{m[2]}",
        ),
    )
    for pattern, replacement in patterns:
        if match := re.fullmatch(pattern, value):
            return replacement(match)

    for prefix in ("encoder", "decoder"):
        marker = f"{prefix}.mid.attn_1."
        if value.startswith(marker):
            suffix = value.removeprefix(marker)
            attention = {
                "norm.weight": "group_norm.weight",
                "norm.bias": "group_norm.bias",
                "q.weight": "to_q.weight",
                "q.bias": "to_q.bias",
                "k.weight": "to_k.weight",
                "k.bias": "to_k.bias",
                "v.weight": "to_v.weight",
                "v.bias": "to_v.bias",
                "proj_out.weight": "to_out.0.weight",
                "proj_out.bias": "to_out.0.bias",
            }.get(suffix)
            return f"{prefix}.mid_block.attentions.0.{attention}" if attention else None
    return None


def _validate_z_image_flux_ae_shape(
    source: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    """Allow only the eight known Flux attention 1x1-convolution projections."""

    if source in _Z_IMAGE_ATTENTION_1X1_LINEAR_SOURCES:
        if (
            len(source_shape) == 4
            and source_shape[2:] == (1, 1)
            and len(target_shape) == 2
            and source_shape[:2] == target_shape
        ):
            return
        raise ValueError(f"Z-Image Flux.1 AE attention 1x1 mapping mismatch: {source}")
    if source_shape != target_shape:
        raise ValueError(f"Z-Image Flux.1 AE shape mismatch: {source}")


def _squeeze_z_image_attention_1x1_weight(source: str, value: torch.Tensor) -> torch.Tensor:
    """Squeeze only the source-pinned 1x1 Conv2d projections into Linear weights."""

    if source not in _Z_IMAGE_ATTENTION_1X1_LINEAR_SOURCES:
        return value
    if value.ndim != 4 or tuple(value.shape[2:]) != (1, 1):
        raise ValueError(f"Z-Image Flux.1 AE attention 1x1 payload changed: {source}")
    return value[:, :, 0, 0]
