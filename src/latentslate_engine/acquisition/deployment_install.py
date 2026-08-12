"""Fail-closed acquisition of an exact deployment-profile resource closure."""

from __future__ import annotations

import ctypes
import hashlib
import ipaddress
import json
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..recipes import (
    DeploymentLock,
    DeploymentPlan,
    build_deployment_lock,
    build_deployment_plan,
    build_recipe_selection_lock,
    build_recipe_selection_plan,
)
from ..resources import (
    ResourceDescriptor,
    ResourceInventory,
    ResourceSource,
    ResourceSourceKind,
    discover_resources,
)

_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_REDIRECTS = 5
_REPARSE_POINT = 0x400
_USER_AGENT = "LatentSlate-Engine/0.1 (+https://github.com/EnviralDesign/LatentSlate-Engine)"
InstallProgressCallback = Callable[[str, dict[str, Any]], None]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


# Kept as a module global so tests can replace every network call.  Redirects
# are processed explicitly below; urllib must never silently forward headers.
urlopen = build_opener(_NoRedirect()).open


class DeploymentInstallError(ValueError):
    """An install failed a security or integrity precondition."""


class DeploymentInstallResourceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    relative_path: str


class DeploymentInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_key: str
    resources: list[DeploymentInstallResourceResult] = Field(default_factory=list)
    installed_resource_ids: list[str] = Field(default_factory=list)
    skipped_resource_ids: list[str] = Field(default_factory=list)
    deployment_plan: DeploymentPlan
    deployment_lock: DeploymentLock


@dataclass(frozen=True, slots=True)
class _Acquisition:
    resource: ResourceDescriptor
    source: ResourceSource
    token: str | None
    target: Path
    stage: Path
    progress: InstallProgressCallback | None = None


def snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download as download

    return str(download(**kwargs))


def hf_hub_download(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download as download

    return str(download(**kwargs))


def install_deployment_profile(
    settings: Settings,
    registry: Any,
    profile_key: str,
    *,
    progress: InstallProgressCallback | None = None,
) -> DeploymentInstallResult:
    """Install a saved profile only after its entire closure has passed preflight."""

    return _install_recipe_selection(
        settings,
        registry,
        plan_builder=lambda candidate: build_deployment_plan(settings, candidate, profile_key),
        lock_builder=lambda candidate: build_deployment_lock(settings, candidate, profile_key),
        selection_label="deployment profile",
        progress=progress,
    )


def install_recipe_selection(
    settings: Settings,
    registry: Any,
    recipe_keys: list[str] | tuple[str, ...],
    *,
    progress: InstallProgressCallback | None = None,
) -> DeploymentInstallResult:
    """Install one explicit recipe selection through the profile-grade pipeline."""

    canonical_keys = tuple(sorted(recipe_keys))
    return _install_recipe_selection(
        settings,
        registry,
        plan_builder=lambda candidate: build_recipe_selection_plan(
            settings, candidate, canonical_keys
        ),
        lock_builder=lambda candidate: build_recipe_selection_lock(
            settings, candidate, canonical_keys
        ),
        selection_label="recipe selection",
        progress=progress,
    )


def _install_recipe_selection(
    settings: Settings,
    registry: Any,
    *,
    plan_builder: Callable[[Any], DeploymentPlan],
    lock_builder: Callable[[Any], DeploymentLock],
    selection_label: str,
    progress: InstallProgressCallback | None = None,
) -> DeploymentInstallResult:
    """Install a closure only after its lock and all local preflight checks pass."""

    plan = plan_builder(registry)
    if not plan.remote_provisionable:
        raise DeploymentInstallError(
            "deployment install refused: "
            f"{selection_label} is not remotely provisionable for automatic installation; "
            "inspect its plan"
        )
    initial_lock = lock_builder(registry)
    descriptors = registry.resources.by_id()
    resources: list[ResourceDescriptor] = []
    for item in initial_lock.resources:
        if item.id not in {resource.id for resource in resources}:
            try:
                resources.append(descriptors[item.id])
            except KeyError as exc:
                raise DeploymentInstallError(
                    f"locked resource {item.id!r} is absent from inventory"
                ) from exc

    acquisitions: list[_Acquisition] = []
    skipped: list[ResourceDescriptor] = []
    for resource in resources:
        target = _target_path(settings, resource)
        _mkdir_safe(target.parent, settings.home.resolve())
        _reject_reparse_components(target.parent, settings.home.resolve())
        if _exists(target):
            if _is_reparse(target):
                raise DeploymentInstallError(f"resource target is a link/reparse point: {target}")
            if ResourceInventory(resources=[resource], paths={resource.id: target}).is_installed(
                resource.id
            ):
                skipped.append(resource)
                continue
            raise DeploymentInstallError(
                f"target for {resource.id!r} already exists but is incomplete: {target}; remove or repair it"
            )
        source, token = _select_source(resource)
        stage = _stage_directory(settings, resource)
        _preflight_stage(stage, resource, source)
        acquisitions.append(_Acquisition(resource, source, token, target, stage, progress))
    _validate_secrets(acquisitions)
    _prepare_temp_capacity(settings, sum(item.resource.size_bytes for item in acquisitions))
    _emit_progress(progress, "preflight", resource_count=len(resources))

    # No network has occurred before this point.  A bad later resource cannot
    # produce a partial profile caused by an avoidable local precondition.
    results = [
        DeploymentInstallResourceResult(
            id=resource.id, status="skipped_installed", relative_path=resource.relative_path
        )
        for resource in skipped
    ]
    for resource in skipped:
        _emit_progress(progress, "skipped", resource_id=resource.id)
    for acquisition in acquisitions:
        _install_resource(acquisition)
        results.append(
            DeploymentInstallResourceResult(
                id=acquisition.resource.id,
                status="installed",
                relative_path=acquisition.resource.relative_path,
            )
        )

    rediscovered = discover_resources(settings)
    incomplete = [
        resource.id for resource in resources if not rediscovered.is_installed(resource.id)
    ]
    if incomplete:
        raise DeploymentInstallError("final discovery rejected resources: " + ", ".join(incomplete))
    # Rebuild the full registry as well as resource discovery: recipe
    # availability is derived from its resource inventory at catalog time.
    from ..tools import default_registry

    final_registry = default_registry(settings, emit_warnings=False)
    final_plan = plan_builder(final_registry)
    final_lock = lock_builder(final_registry)
    installed = sorted(item.id for item in results if item.status == "installed")
    skipped_ids = sorted(item.id for item in results if item.status == "skipped_installed")
    return DeploymentInstallResult(
        profile_key=final_plan.profile_key,
        resources=sorted(results, key=lambda item: item.id),
        installed_resource_ids=installed,
        skipped_resource_ids=skipped_ids,
        deployment_plan=final_plan,
        deployment_lock=final_lock,
    )


def _select_source(resource: ResourceDescriptor) -> tuple[ResourceSource, str | None]:
    source = next((source for source in resource.sources if source.is_exact()), None)
    if source is None:
        raise DeploymentInstallError(
            f"resource {resource.id!r} has no immutable acquisition source"
        )
    secret = source.required_secret()
    token = os.environ.get(secret, "").strip() if secret else None
    return source, token or None


def _validate_secrets(acquisitions: list[_Acquisition]) -> None:
    missing = sorted(
        {
            acquisition.source.required_secret()
            for acquisition in acquisitions
            if acquisition.source.required_secret() and not acquisition.token
        }
    )
    if missing:
        raise DeploymentInstallError("missing required environment secrets: " + ", ".join(missing))


def _install_resource(acquisition: _Acquisition) -> None:
    _emit_progress(
        acquisition.progress,
        "resource_start",
        resource_id=acquisition.resource.id,
        total_bytes=acquisition.resource.size_bytes,
    )
    _prepare_stage(acquisition.stage, acquisition.resource, acquisition.source)
    try:
        _emit_progress(
            acquisition.progress, "phase", resource_id=acquisition.resource.id, phase="downloading"
        )
        staged = _download_to_stage(acquisition)
        _emit_progress(
            acquisition.progress, "phase", resource_id=acquisition.resource.id, phase="verifying"
        )
        _assert_complete(staged, acquisition.resource, acquisition.source)
        _emit_progress(
            acquisition.progress, "phase", resource_id=acquisition.resource.id, phase="publishing"
        )
        _publish_no_clobber(staged, acquisition.target, acquisition.resource)
    except _IntegrityError:
        _emit_progress(acquisition.progress, "failed", resource_id=acquisition.resource.id)
        _cleanup_stage(acquisition.stage)
        raise
    except Exception:
        _emit_progress(acquisition.progress, "failed", resource_id=acquisition.resource.id)
        raise
    if _exists(acquisition.stage):
        _cleanup_stage(acquisition.stage)
    _emit_progress(acquisition.progress, "complete", resource_id=acquisition.resource.id)


def _target_path(settings: Settings, resource: ResourceDescriptor) -> Path:
    relative = Path(resource.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DeploymentInstallError("resource relative_path must stay within Engine data")
    home = settings.home.resolve()
    target = home / relative
    family_root = (
        settings.model_root if resource.kind.value == "model" else settings.lora_root
    ) / resource.family
    try:
        target.relative_to(home)
        target.relative_to(family_root)
    except ValueError as exc:
        raise DeploymentInstallError(
            f"resource {resource.id!r} target escapes family directory"
        ) from exc
    return target


def _stage_directory(settings: Settings, resource: ResourceDescriptor) -> Path:
    root = settings.temp_dir / "deployment-installs"
    try:
        root.relative_to(settings.home)
    except ValueError as exc:
        raise DeploymentInstallError("settings.temp_dir must stay inside Engine home") from exc
    return root / hashlib.sha256(resource.id.encode()).hexdigest()


def _preflight_stage(stage: Path, resource: ResourceDescriptor, source: ResourceSource) -> None:
    _reject_reparse_components(stage.parent, stage.parents[2])
    if not _exists(stage):
        return
    if _is_reparse(stage) or not stage.is_dir():
        raise DeploymentInstallError(f"staging path is not a safe directory: {stage}")
    _assert_no_reparse_tree(stage)
    _validate_stage_manifest(stage, resource, source)
    for name in ("manifest.json", "payload", "payload.part"):
        candidate = stage / name
        if _exists(candidate) and _is_reparse(candidate):
            raise DeploymentInstallError(f"staging entry is a link/reparse point: {candidate}")
    cache = stage / "payload" / ".cache"
    if _exists(cache) and (_is_reparse(cache) or not cache.is_dir()):
        raise DeploymentInstallError(f"Hugging Face cache is unsafe: {cache}")


def _prepare_stage(stage: Path, resource: ResourceDescriptor, source: ResourceSource) -> None:
    _mkdir_safe(stage.parent, stage.parents[2])
    if not _exists(stage):
        stage.mkdir()
    _preflight_stage(stage, resource, source)
    manifest = stage / "manifest.json"
    if not _exists(manifest):
        manifest.write_text(
            json.dumps(_stage_identity(resource, source), sort_keys=True), encoding="utf-8"
        )


def _stage_identity(resource: ResourceDescriptor, source: ResourceSource) -> dict[str, Any]:
    serialized_source = source.model_dump(mode="json")
    # Preserve stage reuse for pre-pattern declarations while incorporating a
    # non-empty subset contract into the resumable staging identity.
    for name in ("allow_patterns", "ignore_patterns"):
        if not serialized_source[name]:
            del serialized_source[name]
    return {"resource_id": resource.id, "source": serialized_source}


def _validate_stage_manifest(
    stage: Path, resource: ResourceDescriptor, source: ResourceSource
) -> None:
    manifest = stage / "manifest.json"
    if not _exists(manifest):
        # A non-empty unowned stage must be remediated by an operator, never reused.
        if any(stage.iterdir()):
            raise DeploymentInstallError(f"staging directory is unowned: {stage}")
        return
    if _is_reparse(manifest) or not manifest.is_file():
        raise DeploymentInstallError(f"staging manifest is unsafe: {manifest}")
    try:
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentInstallError(f"staging manifest is corrupt: {stage}") from exc
    if existing != _stage_identity(resource, source):
        raise DeploymentInstallError(f"staging directory belongs to a different source: {stage}")


def _download_to_stage(acquisition: _Acquisition) -> Path:
    if acquisition.source.type == ResourceSourceKind.HUGGINGFACE:
        return _download_huggingface(acquisition)
    if acquisition.source.type == ResourceSourceKind.CIVITAI:
        return _download_civitai(acquisition)
    if acquisition.source.type == ResourceSourceKind.HTTPS:
        return _download_https(acquisition)
    raise DeploymentInstallError(
        f"manual source cannot be installed remotely: {acquisition.resource.id}"
    )


def _download_huggingface(acquisition: _Acquisition) -> Path:
    source, resource, stage = acquisition.source, acquisition.resource, acquisition.stage
    assert source.repo_id is not None
    payload = stage / "payload"
    _mkdir_safe(payload, stage)
    if resource.format.value in {"diffusers", "directory"}:
        if not _stage_complete(payload, resource):
            download_kwargs: dict[str, Any] = {
                "repo_id": source.repo_id,
                "revision": source.revision,
                "local_dir": str(payload),
                "token": acquisition.token,
            }
            if source.allow_patterns:
                download_kwargs["allow_patterns"] = list(source.allow_patterns)
            if source.ignore_patterns:
                download_kwargs["ignore_patterns"] = list(source.ignore_patterns)
            tqdm_class = _tqdm_class(acquisition.progress)
            if tqdm_class is not None:
                download_kwargs["tqdm_class"] = tqdm_class
            snapshot_download(**download_kwargs)
        _remove_hf_cache(payload, stage)
        return payload
    if not source.filename:
        raise DeploymentInstallError(f"Hugging Face file source for {resource.id!r} lacks filename")
    candidate = _safe_child(payload, source.filename)
    if not _stage_complete(candidate, resource):
        download_kwargs = {
            "repo_id": source.repo_id,
            "filename": source.filename,
            "revision": source.revision,
            "local_dir": str(payload),
            "token": acquisition.token,
        }
        tqdm_class = _tqdm_class(acquisition.progress)
        if tqdm_class is not None:
            download_kwargs["tqdm_class"] = tqdm_class
        hf_hub_download(**download_kwargs)
    _remove_hf_cache(payload, stage)
    return candidate


def _download_https(acquisition: _Acquisition) -> Path:
    source, resource, stage = acquisition.source, acquisition.resource, acquisition.stage
    if not source.url or not source.sha256:
        raise DeploymentInstallError("direct HTTPS source must declare url and sha256")
    payload = stage / "payload"
    if not _stage_complete(payload, resource):
        _download_http_file(
            source.url,
            payload,
            None,
            resource.size_bytes,
            source.sha256,
            progress=acquisition.progress,
            resource_id=resource.id,
        )
    return payload


def _download_civitai(acquisition: _Acquisition) -> Path:
    source, resource, stage = acquisition.source, acquisition.resource, acquisition.stage
    url, api_hash = _civitai_file_url_and_hash(source, acquisition.token)
    if source.sha256 and api_hash and source.sha256.casefold() != api_hash.casefold():
        raise _IntegrityError("Civitai metadata SHA256 conflicts with declared SHA256")
    payload = stage / "payload"
    if not _stage_complete(payload, resource):
        _download_http_file(
            url,
            payload,
            acquisition.token,
            resource.size_bytes,
            source.sha256 or api_hash,
            progress=acquisition.progress,
            resource_id=resource.id,
        )
    if api_hash and not _sha256_matches(payload, api_hash):
        raise _IntegrityError("downloaded Civitai file failed version metadata SHA256")
    return payload


def _civitai_file_url_and_hash(source: ResourceSource, token: str | None) -> tuple[str, str | None]:
    if source.model_version_id is None:
        if not source.url or not source.sha256:
            raise DeploymentInstallError("Civitai URL source must declare url and sha256")
        return source.url, None
    metadata = _read_json(
        f"https://civitai.com/api/v1/model-versions/{source.model_version_id}", token
    )
    files = metadata.get("files") if isinstance(metadata, dict) else None
    selected = next(
        (
            item
            for item in files or []
            if isinstance(item, dict) and item.get("id") == source.file_id
        ),
        None,
    )
    if selected is None:
        raise DeploymentInstallError(
            "Civitai version metadata does not contain the declared file_id"
        )
    url = selected.get("downloadUrl")
    if not isinstance(url, str) or not _is_https(url):
        raise DeploymentInstallError("Civitai selected file lacks a safe HTTPS download URL")
    hashes = selected.get("hashes")
    digest = hashes.get("SHA256") if isinstance(hashes, dict) else None
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        raise DeploymentInstallError("Civitai selected file exposes an invalid SHA256")
    return url, digest


def _read_json(url: str, token: str | None) -> dict[str, Any]:
    response = _open_request(url, token)
    try:
        if response.getcode() != 200:
            raise DeploymentInstallError(f"Civitai metadata returned HTTP {response.getcode()}")
        raw = _read_limited(response, _MAX_METADATA_BYTES)
    finally:
        response.close()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentInstallError("Civitai returned invalid version metadata") from exc
    if not isinstance(payload, dict):
        raise DeploymentInstallError("Civitai returned invalid version metadata")
    return payload


def _download_http_file(
    url: str,
    destination: Path,
    token: str | None,
    expected: int,
    digest: str | None,
    *,
    progress: InstallProgressCallback | None = None,
    resource_id: str | None = None,
) -> None:
    if expected <= 0:
        raise _IntegrityError("Civitai download requires a positive declared size")
    _reject_reparse_components(destination.parent, destination.parent.parent)
    part = destination.with_name(destination.name + ".part")
    if _exists(part) and (_is_reparse(part) or not part.is_file()):
        raise DeploymentInstallError(f"staging partial payload is unsafe: {part}")
    if _exists(destination) and (_is_reparse(destination) or not destination.is_file()):
        raise DeploymentInstallError(f"staging payload is unsafe: {destination}")
    if _exists(destination):
        _safe_unlink(destination, destination.parent)
    for attempt in range(2):
        offset = part.stat().st_size if _exists(part) else 0
        if offset > expected:
            _safe_unlink(part, destination.parent)
            offset = 0
        if offset == expected:
            if digest and not _sha256_matches(part, digest):
                _safe_unlink(part, destination.parent)
                offset = 0
            else:
                _publish_file_no_clobber(part, destination, "staging payload")
                return
        if offset:
            _emit_download_progress(progress, resource_id, offset, expected)
        _require_free_space(destination.parent, expected - offset)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = _open_request(url, token, headers=headers)
        try:
            status = response.getcode()
            if status == 416 and offset and attempt == 0:
                _safe_unlink(part, destination.parent)
                continue
            mode = _validate_download_headers(response, status, offset, expected)
            with part.open(mode) as handle:
                written = offset if mode == "ab" else 0
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected:
                        raise _IntegrityError("Civitai response exceeded declared resource size")
                    handle.write(chunk)
                    _emit_download_progress(progress, resource_id, written, expected)
        finally:
            response.close()
        if part.stat().st_size != expected:
            raise _IntegrityError("Civitai response size does not match declared resource size")
        if digest and not _sha256_matches(part, digest):
            raise _IntegrityError("Civitai response failed expected SHA256")
        _publish_file_no_clobber(part, destination, "staging payload")
        return
    raise DeploymentInstallError("Civitai refused safe range recovery")


def _emit_progress(progress: InstallProgressCallback | None, event: str, **data: Any) -> None:
    if progress is not None:
        progress(event, data)


def _emit_download_progress(
    progress: InstallProgressCallback | None,
    resource_id: str | None,
    completed: int,
    total: int,
) -> None:
    """Emit byte-level transfer state only when a caller supplied a presenter."""

    if resource_id is not None:
        _emit_progress(
            progress,
            "download_progress",
            resource_id=resource_id,
            completed=completed,
            total=total,
        )


def _tqdm_class(progress: InstallProgressCallback | None) -> type[Any] | None:
    """Read an optional presenter hook without coupling acquisition to Rich."""

    candidate = getattr(progress, "tqdm_class", None)
    return candidate if isinstance(candidate, type) else None


def _validate_download_headers(response: Any, status: int, offset: int, expected: int) -> str:
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            declared_length = int(length)
        except ValueError as exc:
            raise _IntegrityError("Civitai returned invalid Content-Length") from exc
        if declared_length < 0 or declared_length > expected - offset:
            raise _IntegrityError("Civitai Content-Length exceeds declared resource size")
    if offset:
        content_range = response.headers.get("Content-Range", "")
        if (
            status != 206
            or not content_range.startswith(f"bytes {offset}-")
            or not content_range.endswith(f"/{expected}")
        ):
            raise _IntegrityError("Civitai returned an unsafe resume range")
        return "ab"
    if status != 200:
        raise DeploymentInstallError(f"Civitai download returned HTTP {status}")
    return "wb"


def _open_request(url: str, token: str | None, *, headers: dict[str, str] | None = None) -> Any:
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        if not _is_https(current):
            raise DeploymentInstallError("remote acquisition redirect is not HTTPS")
        _reject_non_public_literal_url(current)
        request_headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
        request_headers.update(headers or {})
        # A credential is added only at the trusted API origin.  A delivery CDN,
        # arbitrary declared URL, and every cross-origin redirect receive none.
        if token and _is_civitai_api_origin(current):
            request_headers["Authorization"] = f"Bearer {token}"
        try:
            response = urlopen(Request(current, headers=request_headers), timeout=60)
        except HTTPError as exc:
            response = exc
        except OSError as exc:
            raise DeploymentInstallError("remote acquisition request failed") from exc
        status = response.getcode()
        if status not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise DeploymentInstallError("remote acquisition redirect lacks Location")
        current = urljoin(current, location)
    raise DeploymentInstallError("remote acquisition exceeded redirect limit")


def _read_limited(response: Any, limit: int) -> bytes:
    raw = bytearray()
    while chunk := response.read(min(1024 * 1024, limit + 1 - len(raw))):
        raw.extend(chunk)
        if len(raw) > limit:
            raise DeploymentInstallError("remote metadata exceeds size limit")
    return bytes(raw)


def _reject_non_public_literal_url(url: str) -> None:
    hostname = urlsplit(url).hostname
    if hostname is None:
        raise DeploymentInstallError("remote acquisition URL lacks a hostname")
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return
    if not address.is_global:
        raise DeploymentInstallError(
            "remote acquisition cannot target a private or local address"
        )


def _is_https(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme.lower() == "https"
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def _is_civitai_api_origin(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme.lower() == "https"
        and parsed.hostname == "civitai.com"
        and parsed.port in {None, 443}
    )


def _assert_complete(path: Path, resource: ResourceDescriptor, source: ResourceSource) -> None:
    _assert_no_reparse_tree(path)
    if not _stage_complete(path, resource):
        raise _IntegrityError(f"downloaded resource {resource.id!r} failed completeness validation")
    if source.sha256 and not _sha256_matches(path, source.sha256):
        raise _IntegrityError(f"downloaded resource {resource.id!r} failed source SHA256")


def _stage_complete(path: Path, resource: ResourceDescriptor) -> bool:
    if _is_reparse(path):
        return False
    return ResourceInventory(resources=[resource], paths={resource.id: path}).is_installed(
        resource.id
    )


def _sha256_matches(path: Path, expected: str) -> bool:
    if _is_reparse(path) or not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().casefold() == expected.casefold()


def _safe_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise DeploymentInstallError("source filename escapes staging directory")
    candidate = root / child
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DeploymentInstallError("source filename escapes staging directory") from exc
    _reject_reparse_components(candidate.parent, root)
    return candidate


def _assert_no_reparse_tree(path: Path) -> None:
    if _is_reparse(path):
        raise _IntegrityError(f"staged payload contains a link/reparse point: {path}")
    if not path.is_dir():
        return
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            candidate = current_path / name
            if _is_reparse(candidate):
                raise _IntegrityError(f"staged payload contains a link/reparse point: {candidate}")


def _remove_hf_cache(payload: Path, stage: Path) -> None:
    cache = payload / ".cache"
    if _exists(cache):
        _reject_reparse_components(cache, stage)
        if _is_reparse(cache) or not cache.is_dir():
            raise DeploymentInstallError(f"Hugging Face cache is unsafe: {cache}")
        shutil.rmtree(cache)


def _publish_no_clobber(staged: Path, target: Path, resource: ResourceDescriptor) -> None:
    if _is_reparse(staged):
        raise DeploymentInstallError(f"staged resource is a link/reparse point: {staged}")
    if staged.is_file():
        _publish_file_no_clobber(staged, target, resource.id)
        return
    if staged.is_dir():
        _publish_directory_no_clobber(staged, target, resource.id)
        return
    raise DeploymentInstallError(f"staged resource disappeared before publication: {staged}")


def _publish_file_no_clobber(source: Path, target: Path, label: str) -> None:
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise DeploymentInstallError(
            f"target appeared during publication for {label}: {target}"
        ) from exc
    except OSError as exc:
        raise DeploymentInstallError(
            f"atomic no-clobber file publication unavailable for {label}"
        ) from exc
    _safe_unlink(source, source.parent)


def _publish_directory_no_clobber(source: Path, target: Path, label: str) -> None:
    if os.name == "nt":
        result = ctypes.windll.kernel32.MoveFileW(str(source), str(target))
        if result:
            return
        raise DeploymentInstallError(f"atomic no-clobber directory publication failed for {label}")
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (AttributeError, OSError) as exc:
            raise DeploymentInstallError(
                "atomic no-clobber directory publication is unavailable"
            ) from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
            return
        raise DeploymentInstallError(f"atomic no-clobber directory publication failed for {label}")
    raise DeploymentInstallError(
        "atomic no-clobber directory publication is unsupported on this platform"
    )


def _mkdir_safe(path: Path, boundary: Path) -> None:
    relative = path.relative_to(boundary)
    current = boundary
    if _exists(current) and (_is_reparse(current) or not current.is_dir()):
        raise DeploymentInstallError(f"unsafe staging parent: {current}")
    for part in relative.parts:
        current = current / part
        if _exists(current):
            if _is_reparse(current) or not current.is_dir():
                raise DeploymentInstallError(f"unsafe staging component: {current}")
        else:
            current.mkdir()


def _reject_reparse_components(path: Path, boundary: Path) -> None:
    try:
        relative = path.relative_to(boundary)
    except ValueError as exc:
        raise DeploymentInstallError("staging path escapes Engine-owned boundary") from exc
    current = boundary
    for part in relative.parts:
        if _is_reparse(current):
            raise DeploymentInstallError(f"staging path contains a link/reparse point: {current}")
        current = current / part
    if _exists(current) and _is_reparse(current):
        raise DeploymentInstallError(f"staging path contains a link/reparse point: {current}")


def _safe_unlink(path: Path, boundary: Path) -> None:
    _reject_reparse_components(path.parent, boundary)
    if _is_reparse(path) or not path.is_file():
        raise DeploymentInstallError(f"refusing to remove unsafe staging file: {path}")
    path.unlink()


def _cleanup_stage(stage: Path) -> None:
    _reject_reparse_components(stage.parent, stage.parents[2])
    if _exists(stage):
        if _is_reparse(stage) or not stage.is_dir():
            raise DeploymentInstallError(f"refusing to remove unsafe staging directory: {stage}")
        shutil.rmtree(stage)


def _require_free_space(directory: Path, required: int) -> None:
    try:
        if shutil.disk_usage(directory).free < required:
            raise DeploymentInstallError("insufficient free space for declared resource")
    except OSError as exc:
        raise DeploymentInstallError("unable to inspect free space for Engine staging") from exc


def _prepare_temp_capacity(settings: Settings, required: int) -> None:
    """Create the trusted Engine temp root before fail-closed capacity checks."""

    home = settings.home.resolve()
    temp = settings.temp_dir
    try:
        temp.relative_to(home)
    except ValueError as exc:
        raise DeploymentInstallError("settings.temp_dir must stay inside Engine home") from exc
    _mkdir_safe(temp, home)
    _require_free_space(temp, required)


def _exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(mode) or bool(attributes & _REPARSE_POINT)


class _IntegrityError(DeploymentInstallError):
    pass
