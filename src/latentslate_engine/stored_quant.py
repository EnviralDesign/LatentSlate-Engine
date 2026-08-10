"""Lazy reconstruction of stored Comfy quantized tensors (CPU-only).

This module only restores existing stored layouts. It never calls a quantizer.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import _MAX_HEADER_BYTES, ArtifactIdentity, probe_artifact, revalidate_artifact

_MAX_MARKER_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class StoredQuantizedLayer:
    path: Path
    identity: ArtifactIdentity
    key: str
    contract: str
    scale_key: str
    marker_key: str | None
    group_size: int | None = None

    def materialize(self, compute_dtype):
        """Load this one stored layer and reconstruct its existing layout."""

        import torch
        from safetensors import safe_open

        if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("stored quant: unsupported compute dtype")
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            # Opening first binds this materialization to the existing file handle.
            # Revalidating the directory entry afterwards rejects a replacement before
            # any tensor payload is read (including on Windows, where the old handle
            # can remain readable after a rename).
            if not revalidate_artifact(self.identity):
                raise ValueError("stored quant: artifact identity changed before materialization")
            return restore_stored_quantized_tensor(handle, self, compute_dtype)

    def dequantize_cpu(self, compute_dtype):
        """Materialize one layer and apply its stored scale on CPU for verification."""

        return self.materialize(compute_dtype).dequantize()


def discover_stored_layer(path: Path, key: str, contract: str) -> StoredQuantizedLayer:
    """Read header metadata only; no tensor payload is loaded until materialize()."""

    return discover_stored_layers(path, (key,), contract)[key]


def discover_stored_layers(
    path: Path,
    keys: tuple[str, ...],
    contract: str,
) -> dict[str, StoredQuantizedLayer]:
    """Describe several stored layers from one bounded SafeTensors header pass."""

    if not keys or len(set(keys)) != len(keys):
        raise ValueError("stored quant: layer discovery requires distinct weight keys")

    source = Path(path).resolve(strict=True)
    artifact = probe_artifact(source)
    if artifact.format != "safetensors":
        raise ValueError("stored quant: only SafeTensors artifacts are supported")
    if artifact.quantization_contract != contract:
        raise ValueError("stored quant: artifact contract does not match requested contract")
    identity = artifact.identity
    with source.open("rb") as stream:
        prefix_bytes = stream.read(8)
        if len(prefix_bytes) != 8:
            raise ValueError("stored quant: SafeTensors header is truncated")
        length = struct.unpack("<Q", prefix_bytes)[0]
        if length > _MAX_HEADER_BYTES or length > identity.size_bytes - 8:
            raise ValueError("stored quant: SafeTensors header exceeds bounds")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("stored quant: SafeTensors header is truncated")
        header = json.loads(raw_header)
    return _describe_stored_layers(
        identity=identity,
        keys=keys,
        contract=contract,
        available_keys=set(header),
        metadata=header.get("__metadata__"),
    )


def describe_stored_layers_from_handle(
    handle,
    *,
    identity: ArtifactIdentity,
    keys: tuple[str, ...],
    contract: str,
) -> dict[str, StoredQuantizedLayer]:
    """Describe requested stored layers through one already-bound SafeTensors handle.

    The caller must revalidate ``identity`` *after* opening ``handle`` and before
    calling this helper. No path, header, or tensor payload is read here.
    """

    return _describe_stored_layers(
        identity=identity,
        keys=keys,
        contract=contract,
        available_keys=set(handle.keys()),
        metadata=handle.metadata(),
    )


def _describe_stored_layers(
    *,
    identity: ArtifactIdentity,
    keys: tuple[str, ...],
    contract: str,
    available_keys: set[str],
    metadata: Any,
) -> dict[str, StoredQuantizedLayer]:
    """Validate auxiliary topology and build descriptors without tensor reads."""

    if not keys or len(set(keys)) != len(keys):
        raise ValueError("stored quant: layer discovery requires distinct weight keys")
    if contract not in {
        "comfy_quant/float8_e4m3fn",
        "comfy_legacy/scaled_fp8_e4m3fn",
        "comfy_quant/int8_tensorwise_convrot",
    }:
        raise ValueError(f"stored quant: unsupported contract {contract!r}")
    layers: dict[str, Any] = {}
    if contract == "comfy_quant/int8_tensorwise_convrot":
        raw = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError as exc:
            raise ValueError("stored quant: invalid global quantization metadata") from exc
        layers = parsed.get("layers", {}) if isinstance(parsed, dict) else {}
        if not isinstance(layers, dict):
            raise ValueError("stored quant: missing global ConvRot layer metadata")

    result: dict[str, StoredQuantizedLayer] = {}
    prefix = "model.diffusion_model."
    for key in keys:
        actual_key = prefix + key if prefix + key in available_keys else key
        stem = actual_key.removesuffix(".weight")
        if actual_key not in available_keys:
            raise ValueError(f"stored quant: weight key is absent: {key!r}")
        if contract == "comfy_legacy/scaled_fp8_e4m3fn":
            scale_key = stem + ".scale_weight"
            marker_key = None
        else:
            scale_key = stem + ".weight_scale"
            marker_key = stem + ".comfy_quant"
        group_size = None
        if contract == "comfy_quant/int8_tensorwise_convrot":
            normalized_stem = actual_key.removeprefix(prefix).removesuffix(".weight")
            layer = layers.get(normalized_stem, {})
            if not isinstance(layer, dict):
                raise ValueError("stored quant: missing layer quantization metadata")
            nested = layer.get("params", {}) if isinstance(layer.get("params"), dict) else {}
            group_size = (
                layer.get("convrot_groupsize")
                or layer.get("convrot_group_size")
                or layer.get("group_size")
                or nested.get("convrot_groupsize")
                or nested.get("convrot_group_size")
                or nested.get("group_size")
            )
            if not isinstance(group_size, int) or isinstance(group_size, bool):
                raise ValueError("stored quant: missing ConvRot group size")
        result[key] = StoredQuantizedLayer(identity.path, identity, actual_key, contract, scale_key, marker_key, group_size)
    return result


def restore_stored_quantized_tensor(handle, layer: StoredQuantizedLayer, compute_dtype):
    """Restore one pre-quantized layer from an already-open SafeTensors handle.

    The caller owns handle lifetime and artifact identity binding. This function
    reads only the layer, scale, and bounded marker tensors; it never quantizes or
    converts stored weights.
    """

    import torch
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorWiseINT8Layout

    if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("stored quant: unsupported compute dtype")
    expected = (layer.key, layer.scale_key) + ((layer.marker_key,) if layer.marker_key else ())
    _require_keys(handle.keys(), expected)
    qdata = handle.get_tensor(layer.key)
    scale = handle.get_tensor(layer.scale_key)
    marker = _marker_json(handle.get_tensor(layer.marker_key)) if layer.marker_key else {}
    if layer.contract == "comfy_quant/float8_e4m3fn":
        _validate_fp8_geometry(qdata, scale)
        if qdata.dtype != torch.float8_e4m3fn or _marker_field(marker, "format") != "float8_e4m3fn":
            raise ValueError("stored quant: invalid Comfy FP8 weight or scale")
        params = TensorCoreFP8Layout.Params(scale=scale, orig_dtype=compute_dtype, orig_shape=tuple(qdata.shape))
        return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)
    if layer.contract == "comfy_legacy/scaled_fp8_e4m3fn":
        _validate_fp8_geometry(qdata, scale)
        if qdata.dtype != torch.float8_e4m3fn:
            raise ValueError("stored quant: invalid legacy FP8 weight or scale")
        params = TensorCoreFP8Layout.Params(scale=scale, orig_dtype=compute_dtype, orig_shape=tuple(qdata.shape))
        return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)
    if layer.contract == "comfy_quant/int8_tensorwise_convrot":
        _validate_convrot_geometry(qdata, scale)
        if qdata.dtype != torch.int8:
            raise ValueError("stored quant: invalid INT8 weight or scale")
        if (
            _marker_field(marker, "format") != "int8_tensorwise"
            or _marker_field(marker, "convrot") is not True
            or _marker_field(marker, "convrot_groupsize") != layer.group_size
            or (_marker_field(marker, "per_row") is not None and _marker_field(marker, "per_row") is not True)
            or layer.group_size is None
            or layer.group_size <= 0
        ):
            raise ValueError("stored quant: missing ConvRot marker/group size")
        if qdata.shape[1] % layer.group_size:
            raise ValueError("stored quant: ConvRot group size does not divide K")
        params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=compute_dtype, orig_shape=tuple(qdata.shape), is_weight=True, convrot=True, convrot_groupsize=layer.group_size)
        return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    raise ValueError(f"stored quant: unsupported contract {layer.contract!r}")


def _require_keys(actual: list[str], expected: tuple[str, ...]) -> None:
    missing = set(expected) - set(actual)
    if missing:
        raise ValueError("stored quant: required tensors are absent")


def _marker_json(tensor) -> dict[str, Any]:
    import torch

    if tensor.ndim != 1 or tensor.numel() > _MAX_MARKER_BYTES or tensor.dtype != torch.uint8:
        raise ValueError("stored quant: marker must be bounded U8 JSON")
    raw = bytes(tensor.detach().cpu().flatten().tolist())
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("stored quant: marker is not JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError("stored quant: marker must be an object")
    return parsed


def _marker_field(marker: dict[str, Any], field: str) -> Any:
    """Read one marker field from top-level or ``params``, rejecting conflicts."""

    nested = marker.get("params")
    if nested is not None and not isinstance(nested, dict):
        raise ValueError("stored quant: marker params must be an object")
    top_value = marker.get(field)
    nested_value = nested.get(field) if isinstance(nested, dict) else None
    if top_value is not None and nested_value is not None and top_value != nested_value:
        raise ValueError(f"stored quant: conflicting marker {field}")
    return top_value if top_value is not None else nested_value


def _validate_fp8_geometry(qdata, scale) -> None:
    import torch

    if qdata.ndim != 2 or scale.dtype != torch.float32:
        raise ValueError("stored quant: weight must be 2D with F32 scale")
    if tuple(scale.shape) != () or not bool(torch.isfinite(scale).all()):
        raise ValueError("stored quant: FP8 scale must be one finite F32 scalar")


def _validate_convrot_geometry(qdata, scale) -> None:
    import torch

    if qdata.ndim != 2 or scale.dtype != torch.float32:
        raise ValueError("stored quant: weight must be 2D with F32 scale")
    if tuple(scale.shape) != (qdata.shape[0], 1) or not bool(torch.isfinite(scale).all()):
        raise ValueError("stored quant: ConvRot scale must be finite F32 [rows, 1]")
