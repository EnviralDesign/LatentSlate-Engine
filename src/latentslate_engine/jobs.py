from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event
from typing import Any
from uuid import UUID, uuid4

from .config import Settings
from .protocol import (
    ArtifactDescriptor,
    ErrorBody,
    JobCreateRequest,
    JobResponse,
    JobStatus,
    MediaType,
)
from .storage import Storage, StoredArtifact
from .tools import ToolRegistry
from .tools.base import ToolCancelled, ToolContext


@dataclass(slots=True)
class JobState:
    id: UUID
    request: JobCreateRequest
    status: JobStatus = JobStatus.QUEUED
    progress: float | None = 0.0
    message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifacts: list[StoredArtifact] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    error: ErrorBody | None = None
    cancel_event: Event = field(default_factory=Event)


class JobManager:
    def __init__(self, settings: Settings, registry: ToolRegistry, storage: Storage):
        self.settings = settings
        self.registry = registry
        self.storage = storage
        self._jobs: dict[UUID, JobState] = {}
        self._queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_worker(), name="latentslate-engine-worker")

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def submit(self, request: JobCreateRequest) -> JobResponse:
        tool = self.registry.get(request.tool_id)
        descriptor = tool.descriptor
        if not descriptor.available:
            raise JobSubmissionError(
                ErrorBody(
                    code="tool_unavailable",
                    message=descriptor.unavailable_reason or "Tool is unavailable",
                    retryable=False,
                )
            )
        if (
            request.schema_revision != descriptor.schema_revision
            or request.schema_hash != descriptor.schema_hash
        ):
            raise JobSubmissionError(
                ErrorBody(
                    code="schema_mismatch",
                    message="The tool schema changed; refresh the engine catalog.",
                    retryable=False,
                    details={
                        "expected_revision": descriptor.schema_revision,
                        "expected_hash": descriptor.schema_hash,
                    },
                )
            )
        self._validate_inputs(descriptor.inputs, request.inputs)
        state = JobState(id=uuid4(), request=request)
        async with self._lock:
            self._jobs[state.id] = state
        await self._queue.put(state.id)
        return self.response(state)

    async def get(self, job_id: UUID) -> JobResponse:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                raise KeyError(job_id)
            return self.response(state)

    async def cancel(self, job_id: UUID) -> JobResponse:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                raise KeyError(job_id)
            if state.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED):
                return self.response(state)
            state.cancel_event.set()
            if state.status == JobStatus.QUEUED:
                state.status = JobStatus.CANCELED
                state.progress = None
                state.message = "Canceled before execution"
                state.completed_at = datetime.now(UTC)
            else:
                state.message = "Cancellation requested"
            return self.response(state)

    async def artifact(self, job_id: UUID, artifact_id: UUID) -> StoredArtifact:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                raise KeyError(job_id)
            for artifact in state.artifacts:
                if artifact.id == artifact_id:
                    return artifact
            raise FileNotFoundError(artifact_id)

    def counts(self) -> tuple[int, int]:
        queued = sum(state.status == JobStatus.QUEUED for state in self._jobs.values())
        running = sum(state.status == JobStatus.RUNNING for state in self._jobs.values())
        return queued, running

    async def _run_worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                async with self._lock:
                    state = self._jobs[job_id]
                    if state.status == JobStatus.CANCELED:
                        continue
                    state.status = JobStatus.RUNNING
                    state.started_at = datetime.now(UTC)
                    state.progress = 0.0
                    state.message = "Starting"
                await asyncio.to_thread(self._execute, state)
            finally:
                self._queue.task_done()

    def _execute(self, state: JobState) -> None:
        tool = self.registry.get(state.request.tool_id)

        def progress(value: float, message: str | None) -> None:
            state.progress = min(1.0, max(0.0, value))
            state.message = message

        context = ToolContext(
            job_id=state.id,
            settings=self.settings,
            storage=self.storage,
            cancel_event=state.cancel_event,
            progress=progress,
        )
        try:
            context.check_cancelled()
            artifacts = tool.run(context, state.request.inputs)
            context.check_cancelled()
            state.artifacts = artifacts
            state.provenance = {
                "engine_tool_key": tool.descriptor.key,
                "schema_revision": tool.descriptor.schema_revision,
                "schema_hash": tool.descriptor.schema_hash,
                **tool.provenance(),
            }
            state.status = JobStatus.SUCCEEDED
            state.progress = 1.0
            state.message = "Complete"
        except ToolCancelled as exc:
            state.status = JobStatus.CANCELED
            state.progress = None
            state.message = str(exc)
        except Exception as exc:  # noqa: BLE001 - job boundary must capture provider failures
            state.status = JobStatus.FAILED
            state.progress = None
            state.message = "Generation failed"
            state.error = ErrorBody(code="generation_failed", message=str(exc), retryable=False)
        finally:
            state.completed_at = datetime.now(UTC)

    def response(self, state: JobState) -> JobResponse:
        artifacts = [
            ArtifactDescriptor(
                id=artifact.id,
                role=artifact.role,
                media_type=MediaType(artifact.media_type),
                filename=artifact.filename,
                content_type=artifact.content_type,
                size_bytes=artifact.size_bytes,
                download_url=f"/v1/jobs/{state.id}/artifacts/{artifact.id}",
                metadata=artifact.metadata,
            )
            for artifact in state.artifacts
        ]
        return JobResponse(
            id=state.id,
            tool_id=state.request.tool_id,
            status=state.status,
            progress=state.progress,
            message=state.message,
            created_at=state.created_at,
            started_at=state.started_at,
            completed_at=state.completed_at,
            artifacts=artifacts,
            provenance=state.provenance,
            error=state.error,
        )

    @staticmethod
    def _validate_inputs(descriptors, values: dict[str, Any]) -> None:
        known = {descriptor.key: descriptor for descriptor in descriptors}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise JobSubmissionError(
                ErrorBody(
                    code="validation_failed",
                    message="Unknown tool inputs",
                    details={"inputs": unknown},
                )
            )
        missing = [
            descriptor.key
            for descriptor in descriptors
            if descriptor.required and descriptor.key not in values
        ]
        if missing:
            raise JobSubmissionError(
                ErrorBody(
                    code="validation_failed",
                    message="Missing required tool inputs",
                    details={"inputs": missing},
                )
            )
        for descriptor in descriptors:
            if descriptor.key not in values:
                continue
            value = values[descriptor.key]
            if descriptor.type.value in {"image", "video", "audio"}:
                try:
                    from .protocol import AssetInput

                    AssetInput.model_validate(value)
                except Exception as exc:  # noqa: BLE001
                    raise JobSubmissionError(
                        ErrorBody(
                            code="validation_failed",
                            message=f"{descriptor.key} must reference an uploaded asset",
                        )
                    ) from exc
            elif descriptor.type.value == "choice":
                allowed = {option.value for option in descriptor.options}
                if value not in allowed:
                    raise JobSubmissionError(
                        ErrorBody(
                            code="validation_failed",
                            message=f"Invalid value for {descriptor.key}",
                            details={"allowed": sorted(allowed)},
                        )
                    )


class JobSubmissionError(ValueError):
    def __init__(self, error: ErrorBody):
        super().__init__(error.message)
        self.error = error
