"""CPU-only, structural model-artifact inspection.

Probes read SafeTensors headers and GGUF metadata/tensor tables only. They never
read a tensor payload, import torch, instantiate a model, or convert weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 100_000
_MAX_STRING_BYTES = 16 * 1024 * 1024
_SAFE_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
_GGUF_TYPE_LAYOUT = {
    # Verified against the locally staged Q5_K_M Wan tensors: F32/F16, Q5_K
    # (176/256), Q6_K (210/256), and the current GGML BF16 enum (30).
    # Unknown encodings are rejected rather than guessed.
    0: (1, 4), 1: (1, 2), 13: (256, 176), 14: (256, 210), 30: (1, 2),
}


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: Path
    size_bytes: int
    mtime_ns: int
    header_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactProbe:
    identity: ArtifactIdentity
    format: str
    family_signals: tuple[str, ...]
    architecture_signals: tuple[str, ...]
    component_signals: tuple[str, ...]
    quantization_contract: str | None
    tensor_count: int
    tensor_dtypes: tuple[str, ...]
    key_prefix: str | None
    marker_counts: dict[str, int]
    key_shape_signals: dict[str, str | int]
    metadata: dict[str, Any]
    schema_sha256: str

    @property
    def path(self) -> Path:
        return self.identity.path


def probe_artifact(path: Path) -> ArtifactProbe:
    """Validate and inspect a supported artifact without reading payload bytes."""

    resolved = Path(path).resolve(strict=True)
    if resolved.suffix.lower() == ".safetensors":
        return probe_safetensors(resolved)
    if resolved.suffix.lower() == ".gguf":
        return probe_gguf(resolved)
    raise ValueError(f"artifact: unsupported format {resolved.suffix!r}")


def revalidate_artifact(identity: ArtifactIdentity) -> bool:
    """Check an artifact identity again immediately before a future load."""

    try:
        if identity.path.suffix.lower() == ".gguf":
            return probe_gguf(identity.path).identity == identity
        stat = identity.path.stat()
        if stat.st_size != identity.size_bytes or stat.st_mtime_ns != identity.mtime_ns:
            return False
        with identity.path.open("rb") as stream:
            header_size = _read_u64(stream)
            if header_size > _MAX_HEADER_BYTES or header_size > stat.st_size - 8:
                return False
            raw = _read_exact(stream, header_size)
    except (OSError, ValueError):
        return False
    return hashlib.sha256(raw).hexdigest() == identity.header_sha256


def probe_safetensors(path: Path) -> ArtifactProbe:
    before = path.stat()
    with path.open("rb") as stream:
        header_size = _read_u64(stream)
        if header_size > _MAX_HEADER_BYTES:
            raise ValueError("artifact: SafeTensors header exceeds inspection limit")
        raw_header = _read_exact(stream, header_size)
    after = path.stat()
    _unchanged_during_probe(before, after)
    header = json.loads(raw_header, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise TypeError("artifact: SafeTensors header must be an object")
    payload_size = after.st_size - 8 - header_size
    if payload_size < 0:
        raise ValueError("artifact: SafeTensors header exceeds file bounds")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise TypeError("artifact: SafeTensors metadata must be an object")
    entries = {key: value for key, value in header.items() if key != "__metadata__"}
    if not entries or len(entries) > _MAX_ENTRIES:
        raise ValueError("artifact: SafeTensors tensor count is invalid")
    _validate_safetensors_entries(entries, payload_size)
    normalized, prefix = _normalized_safetensors_header(entries)
    keys = sorted(normalized)
    dtypes = sorted({str(value["dtype"]) for value in entries.values()})
    markers = _marker_counts(keys)
    contract = _safetensors_contract(metadata, dtypes, markers, normalized)
    family, architectures, components = _safetensors_signals(keys, normalized)
    return ArtifactProbe(
        identity=_identity(path, after, raw_header), format="safetensors",
        family_signals=family, architecture_signals=architectures, component_signals=components,
        quantization_contract=contract, tensor_count=len(entries), tensor_dtypes=tuple(dtypes),
        key_prefix=prefix, marker_counts=markers,
        key_shape_signals=_shape_signals(normalized, keys), metadata=metadata,
        schema_sha256=_schema_fingerprint(normalized),
    )


def probe_gguf(path: Path) -> ArtifactProbe:
    before = path.stat()
    with path.open("rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError("artifact: GGUF magic is missing")
        version = _read_u32(stream)
        if version != 3:
            raise ValueError(f"artifact: unsupported GGUF version {version}")
        tensor_count, metadata_count = _read_u64(stream), _read_u64(stream)
        if not 0 < tensor_count <= _MAX_ENTRIES or metadata_count > _MAX_ENTRIES:
            raise ValueError("artifact: GGUF entry count exceeds inspection limit")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_gguf_string(stream)
            if key in metadata:
                raise ValueError("artifact: duplicate GGUF metadata key")
            metadata[key] = _read_gguf_value(stream)
            if stream.tell() > _MAX_HEADER_BYTES:
                raise ValueError("artifact: GGUF metadata exceeds aggregate inspection budget")
        tensors = []
        for _ in range(tensor_count):
            tensors.append(_read_gguf_tensor(stream))
            if stream.tell() > _MAX_HEADER_BYTES:
                raise ValueError("artifact: GGUF tensor table exceeds aggregate inspection budget")
        table_end = stream.tell()
        stream.seek(0)
        raw_table = _read_exact(stream, table_end)
    after = path.stat()
    _unchanged_during_probe(before, after)
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("artifact: GGUF alignment is invalid")
    data_start = _align(table_end, alignment)
    _validate_gguf_tensors(tensors, after.st_size - data_start, alignment)
    shapes = {name: tuple(reversed(shape)) for name, shape, _, _ in tensors}
    keys = sorted(shapes)
    family, architectures, components = _wan_signals(keys, shapes)
    type_counts: dict[int, int] = {}
    for _, _, type_id, _ in tensors:
        type_counts[type_id] = type_counts.get(type_id, 0) + 1
    return ArtifactProbe(
        identity=_identity(path, after, raw_table), format="gguf", family_signals=family,
        architecture_signals=architectures, component_signals=components,
        quantization_contract=_gguf_contract(metadata, tensors), tensor_count=len(tensors), tensor_dtypes=(),
        key_prefix=None, marker_counts={}, key_shape_signals=_shape_signals(shapes, keys),
        metadata={"gguf_version": version, "tensor_type_counts": type_counts, **metadata},
        schema_sha256=_schema_fingerprint(shapes),
    )


def _validate_safetensors_entries(entries: dict[str, Any], payload_size: int) -> None:
    regions: list[tuple[int, int]] = []
    for key, value in entries.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise ValueError("artifact: invalid SafeTensors tensor entry")
        dtype = value.get("dtype")
        if dtype not in _SAFE_DTYPE_BYTES:
            raise ValueError(f"artifact: unsupported SafeTensors dtype for {key!r}")
        raw_shape = value.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) > 8
            or not all(isinstance(item, int) and item >= 0 for item in raw_shape)
        ):
            raise ValueError(f"artifact: invalid SafeTensors shape for {key!r}")
        offsets = value.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(item, int) for item in offsets):
            raise ValueError(f"artifact: invalid SafeTensors offsets for {key!r}")
        start, end = offsets
        expected = math.prod(raw_shape) * _SAFE_DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected or end > payload_size:
            raise ValueError(f"artifact: SafeTensors bounds mismatch for {key!r}")
        regions.append((start, end))
    cursor = 0
    for start, end in sorted(regions):
        if start != cursor:
            raise ValueError("artifact: SafeTensors payload offsets are not contiguous")
        cursor = end
    if cursor != payload_size:
        raise ValueError("artifact: SafeTensors payload does not exactly match tensor offsets")


def _read_gguf_tensor(stream: BinaryIO) -> tuple[str, tuple[int, ...], int, int]:
    name = _read_gguf_string(stream)
    dimensions = _read_u32(stream)
    if not name or dimensions > 8:
        raise ValueError("artifact: invalid GGUF tensor info")
    shape = tuple(_read_u64(stream) for _ in range(dimensions))
    if not shape or any(item == 0 for item in shape):
        raise ValueError("artifact: invalid GGUF tensor shape")
    type_id, offset = _read_u32(stream), _read_u64(stream)
    if type_id not in _GGUF_TYPE_LAYOUT:
        raise ValueError(f"artifact: unsupported GGUF tensor type {type_id}")
    return name, shape, type_id, offset


def _validate_gguf_tensors(tensors: list[tuple[str, tuple[int, ...], int, int]], data_size: int, alignment: int) -> None:
    if data_size < 0:
        raise ValueError("artifact: GGUF tensor table exceeds file bounds")
    names: set[str] = set()
    regions: list[tuple[int, int]] = []
    for name, shape, type_id, offset in tensors:
        if name in names:
            raise ValueError("artifact: duplicate GGUF tensor name")
        names.add(name)
        block, bytes_per_block = _GGUF_TYPE_LAYOUT[type_id]
        elements = math.prod(shape)
        if elements % block:
            raise ValueError(f"artifact: GGUF tensor {name!r} does not fit its quantization block")
        end = offset + elements // block * bytes_per_block
        if offset % alignment or end > data_size:
            raise ValueError(f"artifact: GGUF bounds mismatch for {name!r}")
        regions.append((offset, end))
    cursor = 0
    for start, end in sorted(regions):
        if start < cursor:
            raise ValueError("artifact: GGUF tensor offsets overlap")
        cursor = max(cursor, end)
    if cursor != data_size:
        raise ValueError("artifact: GGUF tensor offsets do not reach exact file bound")


def _safetensors_contract(metadata: dict[str, Any], dtypes: list[str], markers: dict[str, int], entries: dict[str, Any]) -> str | None:
    raw = metadata.get("_quantization_metadata")
    if isinstance(raw, str):
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
        layers = parsed.get("layers") if isinstance(parsed, dict) else None
        if isinstance(layers, dict):
            formats = {value.get("format") for value in layers.values() if isinstance(value, dict)}
            layer_names = {name for name, value in layers.items() if isinstance(name, str) and isinstance(value, dict)}
            valid_layers = all(
                value.get("format") == "int8_tensorwise"
                and bool(value.get("convrot"))
                and entries.get(f"{name}.weight", {}).get("dtype") == "I8"
                and f"{name}.weight_scale" in entries
                and f"{name}.comfy_quant" in entries
                for name, value in layers.items()
                if isinstance(name, str) and isinstance(value, dict)
            )
            quantizable = _quantizable_weights(entries)
            int8_stems = {key.removesuffix(".weight") for key in quantizable if entries[key].get("dtype") == "I8"}
            artifact_coverage = len(int8_stems) * 10 >= len(quantizable) * 9
            metadata_coverage = len(layer_names & int8_stems) * 10 >= len(int8_stems) * 9
            if formats == {"int8_tensorwise"} and "I8" in dtypes and valid_layers and int8_stems and artifact_coverage and metadata_coverage and layer_names <= int8_stems:
                return "comfy_quant/int8_tensorwise_convrot"
    quantizable = _quantizable_weights(entries)
    fp8_weights = [key for key in quantizable if entries[key].get("dtype") == "F8_E4M3"]
    comfy_fp8 = fp8_weights and len(fp8_weights) * 10 >= len(quantizable) * 9 and all(
        key.removesuffix(".weight") + ".comfy_quant" in entries
        and key.removesuffix(".weight") + ".weight_scale" in entries
        for key in fp8_weights
    )
    legacy_fp8 = fp8_weights and len(fp8_weights) * 10 >= len(quantizable) * 9 and all(
        key.removesuffix(".weight") + ".scale_weight" in entries for key in fp8_weights
    )
    if "F8_E4M3" in dtypes and comfy_fp8:
        return "comfy_quant/float8_e4m3fn"
    if "F8_E4M3" in dtypes and legacy_fp8:
        return "comfy_legacy/scaled_fp8_e4m3fn"
    weight_dtypes = {value.get("dtype") for key, value in entries.items() if key.endswith(".weight")}
    if weight_dtypes == {"BF16"}:
        return "native/bf16"
    if weight_dtypes == {"F16"}:
        return "native/fp16"
    if weight_dtypes == {"F32"}:
        return "native/fp32"
    return None


def _normalized_safetensors_header(entries: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    prefix = "model.diffusion_model."
    normalized: dict[str, Any] = {}
    has_prefix = any(key.startswith(prefix) for key in entries)
    if has_prefix and any(key.startswith(("blocks.", "patch_embedding.", "head.")) for key in entries):
        raise ValueError("artifact: mixed prefixed/root transformer namespaces")
    for key, value in entries.items():
        normalized_key = key.removeprefix(prefix) if has_prefix else key
        if normalized_key in normalized:
            raise ValueError("artifact: duplicate normalized SafeTensors key")
        normalized[normalized_key] = value
    return normalized, prefix if has_prefix else ("" if any(key.startswith("blocks.") for key in entries) else None)


def _marker_counts(keys: list[str]) -> dict[str, int]:
    markers = (".comfy_quant", ".weight_scale", ".scale_weight", ".self_attn", ".cross_attn", ".ffn", "spiece_model")
    return {marker: sum(marker in key for key in keys) for marker in markers}


def _safetensors_signals(keys: list[str], entries: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    family, architectures, components = _wan_signals(keys, {key: _tensor_shape(value) for key, value in entries.items()})
    encoder_indices = _indices(keys, r"encoder\.block\.(\d+)\.")
    umt5 = len(encoder_indices) == 24 and encoder_indices == set(range(24)) and _tensor_shape(entries.get("spiece_model")) == (4548313,) and _tensor_shape(entries.get("encoder.final_layer_norm.weight")) == (4096,) and _tensor_shape(entries.get("encoder.block.0.layer.0.SelfAttention.q.weight")) == (4096, 4096) and any(".DenseReluDense." in key for key in keys)
    if umt5:
        components = (*components, "text_encoder")
        architectures = (*architectures, "umt5_xxl")
    vae = _tensor_shape(entries.get("decoder.middle.0.residual.0.gamma")) == (384, 1, 1, 1) and "decoder.upsamples.0.upsamples.0.residual.2.weight" not in entries and _tensor_shape(entries.get("decoder.conv1.weight")) == (384, 16, 3, 3, 3) and _tensor_shape(entries.get("encoder.head.2.weight")) == (32, 384, 3, 3, 3)
    if vae:
        components = (*components, "vae")
        architectures = (*architectures, "wan_vae_2_1")
    return family, architectures, components


def _wan_signals(keys: list[str], shapes: dict[str, tuple[int, ...]]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    blocks = _indices(keys, r"blocks\.(\d+)\.")
    signature = blocks == set(range(40)) and shapes.get("patch_embedding.weight") == (5120, 36, 1, 2, 2) and shapes.get("head.modulation") == (1, 2, 5120) and shapes.get("head.head.weight") == (64, 5120) and any(".self_attn." in key for key in keys) and any(".cross_attn." in key for key in keys)
    return (("wan22",), ("wan22_14b_36ch_40block_out16",), ("transformer",)) if signature else ((), (), ())


def _indices(keys: list[str], pattern: str) -> set[int]:
    return {int(match.group(1)) for key in keys if (match := re.match(pattern, key))}


def _shape_signals(entries: dict[str, Any], keys: list[str]) -> dict[str, str | int]:
    shapes = entries if all(isinstance(value, tuple) for value in entries.values()) else {key: _tensor_shape(value) for key, value in entries.items()}
    signals: dict[str, str | int] = {}
    for key in ("patch_embedding.weight", "head.modulation", "head.head.weight", "decoder.conv1.weight", "encoder.head.2.weight"):
        if shape := shapes.get(key):
            signals[key] = "x".join(str(item) for item in shape)
    for name, pattern in (("transformer_block_count", r"blocks\.(\d+)\."), ("text_encoder_block_count", r"encoder\.block\.(\d+)\.")):
        if indices := _indices(keys, pattern):
            signals[name] = len(indices)
    return signals


def _tensor_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
        return ()
    shape = value["shape"]
    return tuple(shape) if all(isinstance(item, int) and item > 0 for item in shape) else ()


def _gguf_contract(metadata: dict[str, Any], tensors: list[tuple[str, tuple[int, ...], int, int]]) -> str | None:
    quantizable = [type_id for name, shape, type_id, _ in tensors if name.endswith(".weight") and len(shape) >= 2]
    compatible = sum(type_id in {13, 14} for type_id in quantizable)
    return "gguf/q5_k_m" if metadata.get("general.file_type") == 17 and quantizable and compatible * 10 >= len(quantizable) * 9 and 13 in quantizable and 14 in quantizable else None


def _schema_fingerprint(entries: dict[str, Any]) -> str:
    def schema(value: Any) -> Any:
        if isinstance(value, dict):
            return (value.get("dtype"), tuple(value.get("shape", [])))
        return tuple(value)
    value = "\n".join(f"{key}:{schema(entries[key])}" for key in sorted(entries))
    return hashlib.sha256(value.encode()).hexdigest()


def _quantizable_weights(entries: dict[str, Any]) -> list[str]:
    return [key for key, value in entries.items() if key.endswith(".weight") and "relative_attention_bias" not in key and len(_tensor_shape(value)) >= 2 and key.startswith(("blocks.", "encoder.block."))]


def _identity(path: Path, stat: Any, raw_header: bytes) -> ArtifactIdentity:
    return ArtifactIdentity(path.resolve(), stat.st_size, stat.st_mtime_ns, hashlib.sha256(raw_header).hexdigest())


def _unchanged_during_probe(before: Any, after: Any) -> None:
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError("artifact: changed while being inspected")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("artifact: duplicate JSON key")
        result[key] = value
    return result


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


_GGUF_ARRAY = 9
_GGUF_STRING = 8
_GGUF_TYPES: dict[int, tuple[str, int]] = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def _read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("artifact: truncated header or table")
    return value


def _read_gguf_string(stream: BinaryIO) -> str:
    length = _read_u64(stream)
    if length > _MAX_STRING_BYTES:
        raise ValueError("artifact: GGUF string exceeds inspection limit")
    return _read_exact(stream, length).decode("utf-8")


def _read_gguf_value(stream: BinaryIO) -> Any:
    value_type = _read_u32(stream)
    if value_type == _GGUF_STRING:
        return _read_gguf_string(stream)
    if value_type == _GGUF_ARRAY:
        element_type, count = _read_u32(stream), _read_u64(stream)
        if count > _MAX_ENTRIES:
            raise ValueError("artifact: GGUF array exceeds inspection limit")
        return [_read_gguf_value_of_type(stream, element_type) for _ in range(count)]
    return _read_gguf_value_of_type(stream, value_type)


def _read_gguf_value_of_type(stream: BinaryIO, value_type: int) -> Any:
    if value_type == _GGUF_STRING:
        return _read_gguf_string(stream)
    try:
        format_char, size = _GGUF_TYPES[value_type]
    except KeyError as exc:
        raise ValueError(f"artifact: unsupported GGUF metadata type {value_type}") from exc
    return struct.unpack("<" + format_char, _read_exact(stream, size))[0]
