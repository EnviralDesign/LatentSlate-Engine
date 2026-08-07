from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .bundles import descriptors as bundle_descriptors
from .config import Settings
from .jobs import JobManager, JobSubmissionError
from .protocol import (
    AssetResponse,
    CatalogResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    JobCreateRequest,
    JobResponse,
    RuntimeStatusResponse,
)
from .runtime.manager import RUNTIME_MANAGER
from .storage import Storage
from .tools import ToolRegistry, default_registry


def create_app(
    settings: Settings | None = None,
    registry: ToolRegistry | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    registry = registry or default_registry()
    storage = Storage(settings)
    jobs = JobManager(settings, registry, storage)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await jobs.start()
        try:
            yield
        finally:
            await jobs.stop()
            RUNTIME_MANAGER.clear()

    app = FastAPI(
        title="LatentSlate Engine",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.storage = storage
    app.state.jobs = jobs

    async def authenticate(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.token is None:
            return
        expected = f"Bearer {settings.token}"
        if authorization != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    auth = Depends(authenticate)

    @app.exception_handler(JobSubmissionError)
    async def job_submission_error(_: Request, exc: JobSubmissionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT
            if exc.error.code == "schema_mismatch"
            else status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error=exc.error).model_dump(mode="json"),
        )

    @app.exception_handler(KeyError)
    async def missing_entity(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error=ErrorBody(code="not_found", message=str(exc), retryable=False)
            ).model_dump(mode="json"),
        )

    @app.get("/v1/health", response_model=HealthResponse, dependencies=[auth])
    async def health() -> HealthResponse:
        queued, running = jobs.counts()
        return HealthResponse(
            engine_version=__version__, queued_jobs=queued, running_jobs=running
        )

    @app.get("/v1/catalog", response_model=CatalogResponse, dependencies=[auth])
    async def catalog() -> CatalogResponse:
        return CatalogResponse(
            engine_version=__version__,
            tools=registry.descriptors(),
            bundles=bundle_descriptors(),
        )

    @app.get("/v1/bundles", dependencies=[auth])
    async def bundles():
        return {"bundles": [bundle.model_dump(mode="json") for bundle in bundle_descriptors()]}

    @app.get(
        "/v1/runtime",
        response_model=RuntimeStatusResponse,
        dependencies=[auth],
    )
    async def runtime_status() -> RuntimeStatusResponse:
        return RuntimeStatusResponse.model_validate(RUNTIME_MANAGER.status())

    @app.delete(
        "/v1/runtime/cache",
        response_model=RuntimeStatusResponse,
        dependencies=[auth],
    )
    async def clear_runtime_cache() -> RuntimeStatusResponse:
        return RuntimeStatusResponse.model_validate(RUNTIME_MANAGER.clear_caches())

    @app.post(
        "/v1/assets",
        response_model=AssetResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[auth],
    )
    async def upload_asset(file: Annotated[UploadFile, File()]) -> AssetResponse:
        try:
            asset = storage.store_asset(
                file.file,
                file.filename,
                file.content_type,
                settings.max_upload_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
        finally:
            await file.close()
        return AssetResponse(
            id=asset.id,
            filename=asset.filename,
            content_type=asset.content_type,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
        )

    @app.post(
        "/v1/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[auth],
    )
    async def create_job(request: JobCreateRequest) -> JobResponse:
        return await jobs.submit(request)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[auth])
    async def get_job(job_id: UUID) -> JobResponse:
        return await jobs.get(job_id)

    @app.delete("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[auth])
    async def cancel_job(job_id: UUID) -> JobResponse:
        return await jobs.cancel(job_id)

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_id}", dependencies=[auth])
    async def download_artifact(job_id: UUID, artifact_id: UUID) -> FileResponse:
        try:
            artifact = await jobs.artifact(job_id, artifact_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found") from exc
        return FileResponse(
            artifact.path,
            media_type=artifact.content_type,
            filename=artifact.filename,
        )

    return app


app = create_app()
