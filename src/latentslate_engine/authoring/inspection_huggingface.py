from __future__ import annotations

import os
import re
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from huggingface_hub import HfApi

from ..resources import ResourceFormat, ResourceSource, ResourceSourceKind
from .inspection_artifacts import (
    _detected_from_facts,
    _format_from_name,
    _positive_int,
    _precision_from_name,
    _quantization_from_name,
    _recommendations,
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

def inspect_huggingface(
    request: ResourceInspectRequest,
    *,
    hf_api_factory: Callable[..., Any] = HfApi,
) -> ResourceInspectionResult:
    repo_id, parsed_revision, parsed_filename = _parse_huggingface_locator(request.source)
    revision = request.revision or parsed_revision
    filename = request.filename or parsed_filename
    token_name = request.token_env or "HF_TOKEN"
    token = os.environ.get(token_name, "").strip() or None
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
        canonical = f"hf://{repo_id}/{filename}@{immutable_revision}"
    else:
        selected = _select_snapshot_files(
            files,
            request.allow_patterns,
            request.ignore_patterns,
        )
        if not selected:
            raise SourceInspectionError("Hugging Face snapshot patterns select no files")
        unknown_sizes = [item["filename"] for item in selected if _positive_int(item["size"]) is None]
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

    detected = _detected_from_facts(facts)
    detected["immutable_revision"] = immutable_revision
    detected["selected_file_count"] = 1 if filename else len(selected)
    recommended_name = facts.filename or repo_id.rsplit("/", 1)[-1]
    return ResourceInspectionResult(
        source_type=AuthoringSourceType.HUGGINGFACE,
        canonical_source=canonical,
        facts=facts,
        exact_source=source,
        detected=detected,
        recommended=_recommendations(recommended_name, repo_id),
        warnings=warnings,
    )


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
