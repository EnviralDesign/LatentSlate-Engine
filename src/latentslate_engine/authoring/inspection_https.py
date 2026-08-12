from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from ..resources import ResourceSource, ResourceSourceKind
from .inspection_artifacts import (
    _detected_from_facts,
    _format_from_name,
    _parse_safetensors_bytes,
    _precision_from_name,
    _quantization_from_name,
    _recommendations,
    _reject_non_public_literal,
)
from .inspection_errors import SourceInspectionError
from .models import ArtifactFacts, AuthoringSourceType, ResourceInspectRequest, ResourceInspectionResult

_MAX_SAFETENSORS_HEADER = 8 * 1024 * 1024

def inspect_https(
    request: ResourceInspectRequest,
    *,
    open_remote_request: Callable[..., Any],
) -> ResourceInspectionResult:
    parsed = urlsplit(request.source.strip())
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise SourceInspectionError("direct resources must use HTTPS")
    if parsed.username or parsed.password:
        raise SourceInspectionError("direct HTTPS URLs cannot contain userinfo")
    if parsed.query or parsed.fragment:
        raise SourceInspectionError(
            "direct HTTPS URLs cannot contain query strings or fragments; use a stable file URL"
        )
    if request.requires_auth or request.token_env:
        raise SourceInspectionError("direct HTTPS sources do not support credentials")
    _reject_non_public_literal(parsed.hostname)
    canonical = parsed.geturl()
    size, header = _probe_remote_file(canonical, open_remote_request=open_remote_request)
    filename = PurePosixPath(unquote(parsed.path)).name or None
    facts = ArtifactFacts(
        filename=filename,
        size_bytes=size,
        sha256=request.expected_sha256.casefold() if request.expected_sha256 else None,
        format=_format_from_name(filename or ""),
        precision=_precision_from_name(filename or ""),
        quantization=_quantization_from_name(filename or ""),
        safetensors=_parse_safetensors_bytes(header) if filename and filename.casefold().endswith(".safetensors") else None,
    )
    exact_source = None
    if request.expected_sha256:
        exact_source = ResourceSource(
            type=ResourceSourceKind.HTTPS,
            url=canonical,
            sha256=request.expected_sha256.casefold(),
        )
    warnings = [
        "direct HTTPS authoring is available only to the trusted local CLI; "
        "the server API does not accept arbitrary URLs"
    ]
    if size is None:
        warnings.append("remote server did not expose an exact byte size; assert one explicitly")
    if request.expected_sha256 is None:
        warnings.append("an exact SHA-256 is required before the declaration can be published")
    return ResourceInspectionResult(
        source_type=AuthoringSourceType.HTTPS,
        canonical_source=canonical,
        facts=facts,
        exact_source=exact_source,
        detected=_detected_from_facts(facts),
        recommended=_recommendations(filename or parsed.hostname, canonical),
        warnings=warnings,
    )


def _probe_remote_file(
    url: str,
    *,
    open_remote_request: Callable[..., Any],
) -> tuple[int | None, bytes]:
    response = open_remote_request(
        url,
        None,
        headers={"Range": f"bytes=0-{_MAX_SAFETENSORS_HEADER + 7}"},
    )
    try:
        status = response.getcode()
        size = _response_size(response, status)
        raw = bytearray()
        while len(raw) < _MAX_SAFETENSORS_HEADER + 8:
            chunk = response.read(min(1024 * 1024, _MAX_SAFETENSORS_HEADER + 8 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        return size, bytes(raw)
    finally:
        response.close()


def _response_size(response: Any, status: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if status == 206 and isinstance(content_range, str) and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit() and int(total) > 0:
            return int(total)
    length = response.headers.get("Content-Length")
    if isinstance(length, str) and length.isdigit() and int(length) > 0:
        return int(length)
    return None
