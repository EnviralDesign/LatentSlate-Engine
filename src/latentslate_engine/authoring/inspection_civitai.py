from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..resources import ResourceSource, ResourceSourceKind
from .inspection_artifacts import (
    _detected_from_facts,
    _format_from_name,
    _positive_int,
    _precision_from_name,
    _quantization_from_name,
    _recommendations,
    _sha256,
)
from .inspection_errors import SourceInspectionError
from .models import (
    ArtifactFacts,
    AuthoringSourceType,
    ResourceInspectionResult,
    ResourceInspectRequest,
    SourceCandidate,
)


def inspect_civitai(
    request: ResourceInspectRequest,
    *,
    read_remote_json: Callable[[str, str | None], dict[str, Any]],
) -> ResourceInspectionResult:
    version_id = _civitai_version_id(request.source)
    if version_id is None:
        raise SourceInspectionError(
            "CivitAI source must identify a model version, for example civitai://version/123"
        )
    token_name = request.token_env or "CIVITAI_TOKEN"
    token = os.environ.get(token_name, "").strip() or None
    try:
        metadata = read_remote_json(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            token,
        )
    except Exception as exc:
        raise SourceInspectionError("CivitAI metadata lookup failed") from exc
    raw_files = metadata.get("files") if isinstance(metadata, dict) else None
    candidates = [
        _civitai_candidate(item)
        for item in (raw_files or [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    ]
    if not candidates:
        raise SourceInspectionError("CivitAI model version contains no downloadable files")
    selected: SourceCandidate | None = None
    if request.file_id is not None:
        selected = next((item for item in candidates if item.id == str(request.file_id)), None)
        if selected is None:
            raise SourceInspectionError(
                f"CivitAI model version does not contain file_id {request.file_id}"
            )
    elif len(candidates) == 1:
        selected = candidates[0]

    if selected is None:
        return ResourceInspectionResult(
            source_type=AuthoringSourceType.CIVITAI,
            canonical_source=f"civitai://version/{version_id}",
            facts=ArtifactFacts(),
            candidates=candidates,
            detected={"model_version_id": version_id},
            recommended={},
            warnings=["several candidate files exist; select one exact file_id"],
        )

    digest = _sha256(selected.sha256)
    source = ResourceSource(
        type=ResourceSourceKind.CIVITAI,
        model_version_id=version_id,
        file_id=int(selected.id),
        sha256=digest,
        token_env=request.token_env,
        requires_auth=request.requires_auth,
    )
    facts = ArtifactFacts(
        filename=selected.filename,
        size_bytes=selected.size_bytes,
        sha256=digest,
        format=_format_from_name(selected.filename or ""),
        precision=_precision_from_name(selected.filename or ""),
        quantization=_quantization_from_name(selected.filename or ""),
    )
    warnings: list[str] = []
    if facts.size_bytes is None:
        warnings.append("CivitAI did not expose an exact byte size; assert one explicitly")
    detected = _detected_from_facts(facts)
    detected.update(
        {
            "model_version_id": version_id,
            "file_id": int(selected.id),
            "base_model": metadata.get("baseModel") if isinstance(metadata, dict) else None,
        }
    )
    return ResourceInspectionResult(
        source_type=AuthoringSourceType.CIVITAI,
        canonical_source=f"civitai://version/{version_id}/file/{selected.id}",
        facts=facts,
        exact_source=source,
        candidates=candidates,
        detected={key: value for key, value in detected.items() if value is not None},
        recommended=_recommendations(selected.filename or selected.label, json.dumps(metadata)[:2000]),
        warnings=warnings,
    )


def _civitai_version_id(source: str) -> int | None:
    raw = source.strip()
    if raw.casefold().startswith("civitai://"):
        parts = [part for part in raw[10:].split("/") if part]
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return None
    parsed = urlsplit(raw)
    query = parse_qs(parsed.query)
    values = query.get("modelVersionId") or query.get("modelversionid")
    if values and values[0].isdigit():
        return int(values[0])
    parts = [part for part in parsed.path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "model-versions" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _civitai_candidate(item: dict[str, Any]) -> SourceCandidate:
    file_id = int(item["id"])
    name = str(item.get("name") or f"file-{file_id}")
    hashes = item.get("hashes")
    digest = hashes.get("SHA256") if isinstance(hashes, dict) else None
    size = _positive_int(item.get("sizeBytes"))
    if size is None:
        size_kb = item.get("sizeKB")
        if isinstance(size_kb, (int, float)) and size_kb > 0:
            size = round(float(size_kb) * 1024)
    metadata = {
        key: item[key]
        for key in ("type", "primary", "pickleScanResult", "virusScanResult")
        if key in item
    }
    return SourceCandidate(
        id=str(file_id),
        label=name,
        filename=name,
        size_bytes=size,
        sha256=_sha256(digest),
        metadata=metadata,
    )
