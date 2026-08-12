from __future__ import annotations

import os
import re
import struct
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from huggingface_hub import HfApi

from ..resources import (
    ResourceFormat,
    ResourceSource,
    ResourceSourceKind,
    _reject_non_public_literal,
)
from .inspection_artifacts import (
    _MAX_SAFETENSORS_HEADER,
    _detected_from_facts,
    _format_from_name,
    _parse_safetensors_bytes,
    _positive_int,
    _precision_from_name,
    _precision_from_safetensors,
    _quantization_from_name,
    _recommendations,
    _safetensors_recommendations,
    _sha256,
    _value,
)
from .inspection_errors import SourceInspectionError
from .models import (
    ArtifactFacts,
    AuthoringSourceType,
    ResourceInspectionResult,
    ResourceInspectRequest,
)

_HF_REVISION = re.compile(r"^[a-fA-F0-9]{40}$")
_MAX_REDIRECTS = 5


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_hf_urlopen = build_opener(_NoRedirect()).open


def inspect_huggingface(
    request: ResourceInspectRequest,
    *,
    hf_api_factory: Callable[..., Any] = HfApi,
    probe_safetensors_header: Callable[[str, str | None], tuple[bytes | None, str | None]]
    | None = None,
) -> ResourceInspectionResult:
    repo_id, parsed_revision, parsed_filename = _parse_huggingface_locator(request.source)
    revision = request.revision or parsed_revision
    filename = request.filename or parsed_filename
    token_name = request.token_env or "HF_TOKEN"
    token = os.environ.get(token_name, "").strip() or None
    probe_safetensors_header = probe_safetensors_header or _probe_safetensors_header
    api = hf_api_factory(token=token)
    try:
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=token,
        )
    except Exception as exc:
        raise SourceInspectionError("Hugging Face metadata lookup failed") from exc
    immutable_revision = str(_value(info, "sha") or "")
    if not _HF_REVISION.fullmatch(immutable_revision):
        raise SourceInspectionError("Hugging Face did not resolve an immutable commit revision")
    siblings = list(_value(info, "siblings") or [])
    files = [_hf_file(item) for item in siblings]
    files = [item for item in files if item["filename"]]

    warnings: list[str] = []
    if filename:
        selected = next((item for item in files if item["filename"] == filename), None)
        if selected is None:
            raise SourceInspectionError(
                f"Hugging Face repository does not contain exact file {filename!r}"
            )
        size = _positive_int(selected.get("size"))
        digest = _sha256(selected.get("sha256"))
        if size is None:
            warnings.append("Hugging Face did not expose an exact byte size; assert one explicitly")
        source = ResourceSource(
            type=ResourceSourceKind.HUGGINGFACE,
            repo_id=repo_id,
            revision=immutable_revision,
            filename=filename,
            sha256=digest,
            token_env=request.token_env,
            requires_auth=request.requires_auth,
        )
        facts = ArtifactFacts(
            filename=PurePosixPath(filename).name,
            size_bytes=size,
            sha256=digest,
            format=_format_from_name(filename),
            precision=_precision_from_name(filename),
            quantization=_quantization_from_name(filename),
        )
        if facts.format == ResourceFormat.SAFETENSORS:
            raw_header, header_warning = probe_safetensors_header(
                _huggingface_resolve_url(repo_id, immutable_revision, filename),
                token,
            )
            if raw_header is not None:
                safetensors = _parse_safetensors_bytes(raw_header)
                if safetensors is None:
                    warnings.append("Hugging Face SafeTensors header is malformed")
                else:
                    facts = facts.model_copy(update={"safetensors": safetensors})
                    facts = facts.model_copy(
                        update={
                            "precision": _precision_from_safetensors(facts.safetensors)
                            or facts.precision
                        }
                    )
            if header_warning:
                warnings.append(header_warning)
        canonical = f"hf://{repo_id}/{filename}@{immutable_revision}"
    else:
        selected = _select_snapshot_files(
            files,
            request.allow_patterns,
            request.ignore_patterns,
        )
        if not selected:
            raise SourceInspectionError("Hugging Face snapshot patterns select no files")
        unknown_sizes = [
            item["filename"] for item in selected if _positive_int(item["size"]) is None
        ]
        size = None if unknown_sizes else sum(int(item["size"]) for item in selected)
        if unknown_sizes:
            warnings.append(
                "Hugging Face did not expose exact sizes for: " + ", ".join(unknown_sizes[:5])
            )
        selected_names = [str(item["filename"]) for item in selected]
        detected_format = (
            ResourceFormat.DIFFUSERS
            if "model_index.json" in selected_names
            else ResourceFormat.DIRECTORY
        )
        source = ResourceSource(
            type=ResourceSourceKind.HUGGINGFACE,
            repo_id=repo_id,
            revision=immutable_revision,
            allow_patterns=tuple(request.allow_patterns),
            ignore_patterns=tuple(request.ignore_patterns),
            token_env=request.token_env,
            requires_auth=request.requires_auth,
        )
        facts = ArtifactFacts(
            filename=None,
            size_bytes=size,
            format=detected_format,
        )
        canonical = f"hf://{repo_id}@{immutable_revision}"
        if request.allow_patterns:
            canonical += "#" + ",".join(request.allow_patterns)

    recommended_name = facts.filename or repo_id.rsplit("/", 1)[-1]
    recommendations = _recommendations(recommended_name, repo_id)
    recommendations, safetensors_detected = _safetensors_recommendations(facts, recommendations)
    detected = _detected_from_facts(facts)
    detected.update(safetensors_detected)
    detected["immutable_revision"] = immutable_revision
    detected["selected_file_count"] = 1 if filename else len(selected)
    return ResourceInspectionResult(
        source_type=AuthoringSourceType.HUGGINGFACE,
        canonical_source=canonical,
        facts=facts,
        exact_source=source,
        detected=detected,
        recommended=recommendations,
        warnings=warnings,
    )


def _huggingface_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return "https://huggingface.co/" + "/".join(
        (
            quote(repo_id, safe="/"),
            "resolve",
            quote(revision, safe=""),
            quote(filename, safe="/"),
        )
    )


def _probe_safetensors_header(url: str, token: str | None) -> tuple[bytes | None, str | None]:
    """Read only a declared SafeTensors header, retaining HF auth only at its origin."""

    prefix, warning = _read_huggingface_range(url, token, "bytes=0-7")
    if prefix is None:
        return None, warning
    if len(prefix) < 8:
        return None, "Hugging Face file did not return a complete SafeTensors header length"
    header_size = struct.unpack("<Q", prefix[:8])[0]
    if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER:
        return None, "Hugging Face SafeTensors header length is invalid or exceeds the 8 MiB limit"
    raw, warning = _read_huggingface_range(
        url,
        token,
        f"bytes=0-{header_size + 7}",
    )
    if raw is None:
        return None, warning
    if len(raw) < header_size + 8:
        return None, "Hugging Face file returned an incomplete SafeTensors header"
    return raw[: header_size + 8], None


def _read_huggingface_range(
    url: str,
    token: str | None,
    byte_range: str,
) -> tuple[bytes | None, str | None]:
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        parsed = urlsplit(current)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            return None, "Hugging Face header probe redirect is not HTTPS"
        try:
            _reject_non_public_literal(parsed.hostname)
        except ValueError:
            return None, "Hugging Face header probe redirect targets a private or local address"
        headers = {"Range": byte_range, "User-Agent": "LatentSlate-Engine"}
        if token and _is_huggingface_origin(parsed):
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = _hf_urlopen(Request(current, headers=headers), timeout=30)
        except HTTPError as exc:
            response = exc
        except OSError:
            return None, "Hugging Face header probe could not open the remote file"
        status = response.getcode()
        if status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                return None, "Hugging Face header probe redirect lacks Location"
            current = urljoin(current, location)
            continue
        try:
            if status not in {200, 206}:
                return None, f"Hugging Face header probe returned HTTP {status}"
            limit = _range_limit(byte_range)
            return response.read(limit), None
        finally:
            response.close()
    return None, "Hugging Face header probe exceeded redirect limit"


def _range_limit(byte_range: str) -> int:
    return int(byte_range.rsplit("-", 1)[-1]) + 1


def _is_huggingface_origin(parsed: Any) -> bool:
    return parsed.hostname.casefold() in {
        "huggingface.co",
        "www.huggingface.co",
    } and parsed.port in {
        None,
        443,
    }


def _parse_huggingface_locator(source: str) -> tuple[str, str | None, str | None]:
    raw = source.strip()
    if raw.casefold().startswith("hf://"):
        body = raw[5:]
        path_part, separator, revision = body.partition("@")
        parts = [part for part in path_part.split("/") if part]
        if len(parts) < 2:
            raise SourceInspectionError("Hugging Face source must contain owner/repository")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:]) or None
        return repo_id, revision if separator else None, filename
    parsed = urlsplit(raw)
    if parsed.scheme.casefold() != "https" or (parsed.hostname or "").casefold() not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise SourceInspectionError("invalid Hugging Face source locator")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SourceInspectionError(
            "Hugging Face URLs cannot contain userinfo, query strings, or fragments"
        )
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise SourceInspectionError("Hugging Face URL must contain owner/repository")
    repo_id = "/".join(parts[:2])
    if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
        return repo_id, parts[3], "/".join(parts[4:])
    return repo_id, None, None


def _hf_file(item: Any) -> dict[str, Any]:
    filename = _value(item, "rfilename") or _value(item, "path") or _value(item, "filename")
    lfs = _value(item, "lfs")
    size = _value(item, "size")
    digest = None
    if lfs:
        size = size or _value(lfs, "size")
        digest = _value(lfs, "sha256") or _value(lfs, "oid")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            digest = digest.removeprefix("sha256:")
    return {"filename": str(filename or ""), "size": size, "sha256": digest}


def _select_snapshot_files(
    files: list[dict[str, Any]],
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> list[dict[str, Any]]:
    selected = []
    for item in files:
        filename = str(item["filename"])
        if allow_patterns and not any(fnmatchcase(filename, pattern) for pattern in allow_patterns):
            continue
        if ignore_patterns and any(fnmatchcase(filename, pattern) for pattern in ignore_patterns):
            continue
        selected.append(item)
    return selected
