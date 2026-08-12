from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import struct
from pathlib import Path
from typing import Any

from ..resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceFormat,
)
from .inspection_errors import SourceInspectionError
from .models import (
    ArtifactFacts,
    ResourceInspectionResult,
    ResourceInspectRequest,
    SafeTensorsFacts,
)

_MAX_SAFETENSORS_HEADER = 8 * 1024 * 1024
_MAX_TENSOR_KEYS = 1024


def _inspect_local_file(path: Path, *, filename: str | None = None) -> ArtifactFacts:
    inspected_name = filename or path.name
    size = path.stat().st_size
    digest = hashlib.sha256()
    prefix = bytearray()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            if len(prefix) < _MAX_SAFETENSORS_HEADER + 8:
                remaining = _MAX_SAFETENSORS_HEADER + 8 - len(prefix)
                prefix.extend(chunk[:remaining])
    safetensors = (
        _parse_safetensors_bytes(bytes(prefix))
        if Path(inspected_name).suffix.casefold() == ".safetensors"
        else None
    )
    precision = _precision_from_safetensors(safetensors) or _precision_from_name(inspected_name)
    quantization = _quantization_from_name(inspected_name)
    return ArtifactFacts(
        filename=inspected_name,
        size_bytes=size,
        sha256=digest.hexdigest(),
        format=_format_from_name(inspected_name),
        precision=precision,
        quantization=quantization,
        safetensors=safetensors,
    )


def _parse_safetensors_bytes(raw: bytes) -> SafeTensorsFacts | None:
    if len(raw) < 8:
        return None
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER:
        return None
    if len(raw) < 8 + header_size:
        return None
    try:
        header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    metadata_raw = header.get("__metadata__")
    metadata = (
        {str(key): str(value) for key, value in metadata_raw.items()}
        if isinstance(metadata_raw, dict)
        else {}
    )
    tensor_items = [(str(key), value) for key, value in header.items() if key != "__metadata__"]
    if any(not isinstance(value, dict) for _, value in tensor_items):
        return None
    tensor_items.sort(key=lambda item: item[0])
    schema = []
    shapes: dict[str, list[int]] = {}
    dtypes: set[str] = set()
    keys: list[str] = []
    try:
        for key, value in tensor_items:
            dtype = value.get("dtype")
            raw_shape = value.get("shape")
            offsets = value.get("data_offsets")
            if not isinstance(dtype, str) or not dtype:
                return None
            if not isinstance(raw_shape, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in raw_shape
            ):
                return None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in offsets
                )
                or offsets[1] < offsets[0]
            ):
                return None
            shape = list(raw_shape)
            schema.append({"key": key, "dtype": dtype, "shape": shape})
            dtypes.add(dtype)
            if len(keys) < _MAX_TENSOR_KEYS:
                keys.append(key)
                shapes[key] = shape
    except (TypeError, ValueError):
        return None
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return SafeTensorsFacts(
        tensor_count=len(tensor_items),
        tensor_keys=keys,
        dtypes=sorted(dtypes),
        shapes=shapes,
        metadata=metadata,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
        truncated=len(tensor_items) > _MAX_TENSOR_KEYS,
    )


def _detected_from_facts(facts: ArtifactFacts) -> dict[str, Any]:
    detected: dict[str, Any] = {
        "format": facts.format.value,
        "precision": facts.precision.value,
        "quantization": facts.quantization.value,
    }
    if facts.safetensors:
        detected.update(
            {
                "tensor_count": facts.safetensors.tensor_count,
                "tensor_dtypes": facts.safetensors.dtypes,
                "schema_sha256": facts.safetensors.schema_sha256,
            }
        )
    return detected


def _recommendations(filename: str, context: str) -> dict[str, Any]:
    haystack = f"{filename} {context}".casefold()
    family = "custom"
    for token, candidate in (
        ("klein-9", "klein9b"),
        ("klein9", "klein9b"),
        ("klein-4", "klein4b"),
        ("klein4", "klein4b"),
        ("wan2.2", "wan22"),
        ("wan22", "wan22"),
        ("ltx-2.3", "ltx23"),
        ("ltx23", "ltx23"),
        ("minimax-h3", "h3"),
    ):
        if token in haystack:
            family = candidate
            break
    component = None
    for token, candidate in (
        ("text_encoder", "text_encoder"),
        ("text-encoder", "text_encoder"),
        ("transformer", "transformer"),
        ("vae", "vae"),
        ("lora", "lora"),
    ):
        if token in haystack:
            component = candidate
            break
    name = Path(filename).stem.replace("_", " ").replace("-", " ").strip().title() or filename
    result: dict[str, Any] = {"name": name, "family": family}
    if component:
        result["component"] = component
    return result


def _safetensors_recommendations(
    facts: ArtifactFacts,
    recommendations: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add only high-confidence adapter and family hints from SafeTensors structure."""

    safetensors = facts.safetensors
    if safetensors is None:
        return recommendations, {}
    detected: dict[str, Any] = {}
    if _has_paired_lora_tensors(safetensors.tensor_keys):
        recommendations["component"] = "lora"
        detected["lora_tensor_pairs"] = True
    base_model = _recognized_base_model(safetensors.metadata)
    if base_model is not None:
        family, value = base_model
        recommendations["family"] = family
        recommendations["base_model"] = value
        detected["base_model"] = value
    return recommendations, detected


def _has_paired_lora_tensors(keys: list[str]) -> bool:
    paired_suffixes = (
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
    )
    for left_suffix, right_suffix in paired_suffixes:
        left = {key.removesuffix(left_suffix) for key in keys if key.endswith(left_suffix)}
        right = {key.removesuffix(right_suffix) for key in keys if key.endswith(right_suffix)}
        if left & right:
            return True
    return False


def _recognized_base_model(metadata: dict[str, str]) -> tuple[str, str] | None:
    raw = metadata.get("ss_base_model_version")
    if raw is None:
        return None
    normalized = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized == "flux2_klein_9b":
        return "klein9b", "black-forest-labs/FLUX.2-klein-9B"
    if normalized == "flux2_klein_4b":
        return "klein4b", "black-forest-labs/FLUX.2-klein-4B"
    return None


def _format_from_name(filename: str) -> ResourceFormat:
    lowered = filename.casefold()
    if lowered.endswith(".safetensors"):
        return ResourceFormat.SAFETENSORS
    if lowered.endswith(".gguf"):
        return ResourceFormat.GGUF
    if lowered.endswith((".ckpt", ".pt", ".pth", ".bin")):
        return ResourceFormat.CHECKPOINT
    return ResourceFormat.UNKNOWN


def _precision_from_safetensors(facts: SafeTensorsFacts | None) -> ArtifactPrecision | None:
    if facts is None or not facts.dtypes:
        return None
    normalized = {item.upper() for item in facts.dtypes}
    if normalized <= {"BF16"}:
        return ArtifactPrecision.BF16
    if normalized <= {"F16", "FLOAT16"}:
        return ArtifactPrecision.FP16
    if normalized <= {"F32", "FLOAT32"}:
        return ArtifactPrecision.FP32
    if normalized and all(item.startswith("F8") for item in normalized):
        return ArtifactPrecision.FP8
    return None


def _precision_from_name(filename: str) -> ArtifactPrecision:
    lowered = filename.casefold()
    if "bf16" in lowered:
        return ArtifactPrecision.BF16
    if "fp16" in lowered or "float16" in lowered:
        return ArtifactPrecision.FP16
    if "fp8" in lowered or "float8" in lowered:
        return ArtifactPrecision.FP8
    if "fp32" in lowered or "float32" in lowered:
        return ArtifactPrecision.FP32
    return ArtifactPrecision.UNKNOWN


def _quantization_from_name(filename: str) -> ArtifactQuantization:
    lowered = filename.casefold()
    if lowered.endswith(".gguf"):
        return ArtifactQuantization.GGUF
    if "nvfp4" in lowered:
        return ArtifactQuantization.NVFP4
    if "int8" in lowered:
        return ArtifactQuantization.INT8
    return ArtifactQuantization.UNKNOWN


def _apply_assertions(
    request: ResourceInspectRequest,
    result: ResourceInspectionResult,
) -> ResourceInspectionResult:
    facts = result.facts
    if request.expected_size_bytes is not None:
        if facts.size_bytes is not None and facts.size_bytes != request.expected_size_bytes:
            raise SourceInspectionError(
                f"source size {facts.size_bytes} does not match asserted size "
                f"{request.expected_size_bytes}"
            )
        facts = facts.model_copy(update={"size_bytes": request.expected_size_bytes})
    if request.expected_sha256 is not None:
        digest = request.expected_sha256.casefold()
        if facts.sha256 is not None and facts.sha256.casefold() != digest:
            raise SourceInspectionError("source SHA-256 does not match the asserted digest")
        facts = facts.model_copy(update={"sha256": digest})
        exact = result.exact_source
        if exact is not None and exact.sha256 is None:
            exact = exact.model_copy(update={"sha256": digest})
            result = result.model_copy(update={"exact_source": exact})
    return result.model_copy(update={"facts": facts})


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    return None


def _sha256(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value):
        return value.casefold()
    return None


def _reject_non_public_literal(hostname: str) -> None:
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return
    if not address.is_global:
        raise SourceInspectionError("direct HTTPS URL cannot target a private or local address")
