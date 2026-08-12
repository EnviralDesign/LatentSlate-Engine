from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import HfApi

from ..acquisition import deployment_install as installer
from ..config import Settings
from ..resources import ResourceSource, ResourceSourceKind
from .inspection_artifacts import (
    _apply_assertions,
    _detected_from_facts,
    _inspect_local_file,
    _recommendations,
)
from .inspection_civitai import _civitai_version_id, inspect_civitai
from .inspection_errors import SourceInspectionError
from .inspection_https import inspect_https
from .inspection_huggingface import inspect_huggingface
from .models import (
    AuthoringSourceType,
    ResourceInspectionResult,
    ResourceInspectRequest,
)

# Replaceable seams keep external lookups fully mocked in model-free tests.
hf_api_factory = HfApi
read_remote_json = installer._read_json
open_remote_request = installer._open_request

def inspect_source(
    request: ResourceInspectRequest,
    settings: Settings,
    *,
    allow_local: bool = True,
    allow_direct_https: bool = True,
) -> ResourceInspectionResult:
    source_type = _resolve_source_type(request)
    if source_type == AuthoringSourceType.LOCAL:
        if not allow_local:
            raise SourceInspectionError(
                "local filesystem inspection is available only to the local CLI"
            )
        result = _inspect_local(request, settings)
    elif source_type == AuthoringSourceType.HUGGINGFACE:
        result = inspect_huggingface(request, hf_api_factory=hf_api_factory)
    elif source_type == AuthoringSourceType.CIVITAI:
        result = inspect_civitai(request, read_remote_json=read_remote_json)
    elif source_type == AuthoringSourceType.HTTPS:
        if not allow_direct_https:
            raise SourceInspectionError(
                "direct HTTPS inspection is disabled on the server; use the local CLI"
            )
        result = inspect_https(request, open_remote_request=open_remote_request)
    else:  # pragma: no cover - enum exhaustiveness
        raise SourceInspectionError(f"unsupported source type {source_type!r}")
    return _apply_assertions(request, result)


def stage_import(
    request: ResourceInspectRequest,
    inspection: ResourceInspectionResult,
    destination: Path,
) -> ResourceInspectionResult:
    """Materialize one local source into an Engine-owned staging file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if inspection.source_type != AuthoringSourceType.LOCAL:
        raise SourceInspectionError(
            f"source type {inspection.source_type.value!r} is declared remotely, not imported"
        )
    source_input = Path(request.source).expanduser()
    if installer._is_reparse(source_input):
        raise SourceInspectionError("local imports cannot be symbolic links or reparse aliases")
    source = source_input.resolve()
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)

    facts = _inspect_local_file(
        destination,
        filename=inspection.facts.filename,
    )
    if inspection.facts.size_bytes is not None and facts.size_bytes != inspection.facts.size_bytes:
        raise SourceInspectionError("imported file size changed after inspection")
    if request.expected_sha256 and facts.sha256 != request.expected_sha256.casefold():
        raise SourceInspectionError("imported file SHA-256 does not match the asserted digest")
    manual = ResourceSource(
        type=ResourceSourceKind.MANUAL,
        sha256=facts.sha256,
        label=f"Imported local file {facts.filename}",
    )
    detected = dict(inspection.detected)
    detected.update(_detected_from_facts(facts))
    return inspection.model_copy(
        update={
            "facts": facts,
            "exact_source": manual,
            "detected": detected,
        }
    )


def _resolve_source_type(request: ResourceInspectRequest) -> AuthoringSourceType:
    if request.source_type != AuthoringSourceType.AUTO:
        return request.source_type
    raw = request.source.strip()
    lowered = raw.casefold()
    if lowered.startswith("hf://"):
        return AuthoringSourceType.HUGGINGFACE
    if lowered.startswith("civitai://"):
        return AuthoringSourceType.CIVITAI
    if lowered.startswith("https://"):
        host = (urlsplit(raw).hostname or "").casefold()
        if host in {"huggingface.co", "www.huggingface.co"}:
            return AuthoringSourceType.HUGGINGFACE
        if host in {"civitai.com", "www.civitai.com"} and _civitai_version_id(raw):
            return AuthoringSourceType.CIVITAI
        return AuthoringSourceType.HTTPS
    return AuthoringSourceType.LOCAL


def _inspect_local(
    request: ResourceInspectRequest,
    settings: Settings,
) -> ResourceInspectionResult:
    source_input = Path(request.source).expanduser()
    if installer._is_reparse(source_input):
        raise SourceInspectionError("local imports cannot be symbolic links or reparse aliases")
    path = source_input.resolve()
    if not path.is_file():
        raise SourceInspectionError(f"local source is not an individual file: {path}")
    if path.stat().st_size > settings.max_upload_bytes:
        raise SourceInspectionError(
            "local source exceeds LATENTSLATE_ENGINE_MAX_UPLOAD_BYTES"
        )
    facts = _inspect_local_file(path)
    exact = ResourceSource(
        type=ResourceSourceKind.MANUAL,
        sha256=facts.sha256,
        label=f"Imported local file {path.name}",
    )
    return ResourceInspectionResult(
        source_type=AuthoringSourceType.LOCAL,
        canonical_source=str(path),
        facts=facts,
        exact_source=exact,
        detected=_detected_from_facts(facts),
        recommended=_recommendations(path.name, str(path)),
    )
