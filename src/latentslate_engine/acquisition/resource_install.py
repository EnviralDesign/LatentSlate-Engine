"""Single-resource acquisition through the deployment install safety pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import Settings
from ..resources import ResourceDescriptor, ResourceInventory, discover_resources
from . import deployment_install as pipeline


class ResourceInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource: ResourceDescriptor
    status: str
    artifact_path: str
    installed: bool


def install_resource(
    settings: Settings,
    registry: Any,
    resource_id: str,
    *,
    progress: pipeline.InstallProgressCallback | None = None,
) -> ResourceInstallResult:
    """Fetch one declared resource without inventing a second downloader."""

    try:
        resource = registry.resources.resolve(resource_id, include_components=True)
    except (KeyError, ValueError) as exc:
        raise pipeline.DeploymentInstallError(str(exc)) from exc

    target = pipeline._target_path(settings, resource)
    home = settings.home.resolve()
    pipeline._mkdir_safe(target.parent, home)
    pipeline._reject_reparse_components(target.parent, home)
    if pipeline._exists(target):
        if pipeline._is_reparse(target):
            raise pipeline.DeploymentInstallError(
                f"resource target is a link/reparse point: {target}"
            )
        if ResourceInventory(resources=[resource], paths={resource.id: target}).is_installed(
            resource.id
        ):
            return ResourceInstallResult(
                resource=resource.model_copy(
                    update={"available": True, "unavailable_reason": None}
                ),
                status="skipped_installed",
                artifact_path=str(target),
                installed=True,
            )
        raise pipeline.DeploymentInstallError(
            f"target for {resource.id!r} already exists but is incomplete: {target}; "
            "remove or repair it"
        )

    source, token = pipeline._select_source(resource)
    stage = pipeline._stage_directory(settings, resource)
    pipeline._preflight_stage(stage, resource, source)
    acquisition = pipeline._Acquisition(
        resource=resource,
        source=source,
        token=token,
        target=target,
        stage=stage,
        progress=progress,
    )
    pipeline._validate_secrets([acquisition])
    pipeline._prepare_temp_capacity(settings, resource.size_bytes)
    pipeline._emit_progress(progress, "preflight", resource_count=1)
    pipeline._install_resource(acquisition)

    rediscovered = discover_resources(settings)
    if not rediscovered.is_installed(resource.id):
        raise pipeline.DeploymentInstallError(
            f"final discovery rejected resource {resource.id!r}"
        )
    installed_resource = rediscovered.resolve(resource.id, include_components=True)
    return ResourceInstallResult(
        resource=installed_resource,
        status="installed",
        artifact_path=str(rediscovered.path_for(resource.id)),
        installed=True,
    )
