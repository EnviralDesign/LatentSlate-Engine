from __future__ import annotations

import asyncio
import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event
from typing import Any
from uuid import UUID, uuid4

from .config import Settings
from .protocol import (
    ArtifactDescriptor,
    AssetInput,
    ErrorBody,
    InputType,
    JobCreateRequest,
    JobResponse,
    JobStatus,
    MediaType,
    ToolInput,
)
from .runtime.kit import cleanup_accelerator_memory, is_cuda_oom
from .runtime.manager import RUNTIME_MANAGER
from .safe_errors import SafeJobFailure
from .storage import Storage, StoredArtifact
from .tools import ToolRegistry
from .tools.base import ToolCancelled, ToolContext

LOGGER = logging.getLogger(__name__)


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
            self._worker = asyncio.create_task(
                self._run_worker(),
                name="latentslate-engine-worker",
            )

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

        normalized_inputs = self._validate_inputs(descriptor.inputs, request.inputs)
        semantic_errors = tool.validate_inputs(normalized_inputs)
        if semantic_errors:
            error = semantic_errors[0]
            raise JobSubmissionError(
                ErrorBody(
                    code="validation_failed",
                    message=error.message,
                    retryable=False,
                    details={"input": error.input_key, **dict(error.details)},
                )
            )
        request = request.model_copy(update={"inputs": normalized_inputs})
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
            if state.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELED,
            ):
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
        base_provenance = {
            "engine_tool_key": tool.descriptor.key,
            "schema_revision": tool.descriptor.schema_revision,
            "schema_hash": tool.descriptor.schema_hash,
            **tool.provenance(),
        }
        state.provenance = dict(base_provenance)
        try:
            context.check_cancelled()
            artifacts = tool.run(context, state.request.inputs)
            context.check_cancelled()
            state.artifacts = artifacts
            state.status = JobStatus.SUCCEEDED
            state.progress = 1.0
            state.message = "Complete"
        except ToolCancelled as exc:
            try:
                self.storage.remove_job_folder(state.id)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not replace cancellation state
                LOGGER.error(
                    "Job %s cancellation cleanup failed: code=cancellation_cleanup_failed "
                    "type=%s",
                    state.id,
                    type(cleanup_error).__name__,
                )
                context.record_provenance(
                    cancellation_cleanup={
                        "status": "failed",
                        "error_type": type(cleanup_error).__name__,
                    },
                    cancellation={"requested": True, "completed": False},
                )
                state.status = JobStatus.FAILED
                state.progress = None
                state.message = "Canceled job cleanup failed"
                state.error = ErrorBody(
                    code="cancellation_cleanup_failed",
                    message=(
                        "Generation was canceled, but Engine could not remove partial outputs."
                    ),
                    retryable=False,
                    details={"cleanup_error_type": type(cleanup_error).__name__},
                )
            else:
                context.record_provenance(
                    cancellation_cleanup={"status": "complete"},
                    cancellation={"requested": True, "completed": True},
                )
                state.status = JobStatus.CANCELED
                state.progress = None
                state.message = str(exc)
        except SafeJobFailure as exc:
            # This is intentionally narrower than the generic provider-failure
            # path below.  A private worker has authenticated and sanitized
            # every label, so do not attach traceback exception formatting that
            # could recover hostile library/request text.
            diagnostic = f" diagnostic={exc.diagnostic}" if exc.diagnostic else ""
            LOGGER.error(
                "Job %s failed: tool=%s code=%s type=%s message=%s%s",
                state.id,
                tool.descriptor.key,
                exc.code,
                exc.error_type,
                exc.public_message,
                diagnostic,
            )
            state.status = JobStatus.FAILED
            state.progress = None
            state.message = exc.public_message
            state.error = ErrorBody(
                code=exc.code,
                message=exc.public_message,
                retryable=False,
                details={},
            )
        except Exception as exc:
            cuda_oom = is_cuda_oom(exc)
            error_code = "cuda_out_of_memory" if cuda_oom else "generation_failed"
            LOGGER.exception(
                "Job %s failed: tool=%s code=%s",
                state.id,
                tool.descriptor.key,
                error_code,
            )
            evicted_runtime = (
                RUNTIME_MANAGER.evict_active(clear_cache=True) if cuda_oom else None
            )
            if cuda_oom:
                cleanup_accelerator_memory()
                context.record_provenance(
                    runtime_failure={
                        "kind": "cuda_out_of_memory",
                        "active_runtime_evicted": evicted_runtime,
                    }
                )
            state.status = JobStatus.FAILED
            state.progress = None
            state.message = (
                "CUDA out of memory; active runtime evicted"
                if cuda_oom
                else "Generation failed"
            )
            state.error = ErrorBody(
                code=error_code,
                message=str(exc),
                retryable=False,
                details={
                    "active_runtime_evicted": evicted_runtime,
                }
                if cuda_oom
                else {},
            )
        finally:
            state.provenance = {
                **base_provenance,
                **context.runtime_provenance,
            }
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

    def _validate_inputs(
        self,
        descriptors: list[ToolInput],
        values: dict[str, Any],
    ) -> dict[str, Any]:
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

        normalized = dict(values)
        missing = []
        for descriptor in descriptors:
            if descriptor.key in normalized:
                continue
            if descriptor.default is not None:
                normalized[descriptor.key] = copy.deepcopy(descriptor.default)
            elif descriptor.required:
                missing.append(descriptor.key)
        if missing:
            raise JobSubmissionError(
                ErrorBody(
                    code="validation_failed",
                    message="Missing required tool inputs",
                    details={"inputs": missing},
                )
            )

        for descriptor in descriptors:
            if descriptor.key not in normalized:
                continue
            value = normalized[descriptor.key]
            if value is None:
                if descriptor.required:
                    raise self._input_error(
                        descriptor,
                        f"{descriptor.key} is required",
                        expected=descriptor.type.value,
                    )
                continue

            if descriptor.multiple:
                if not isinstance(value, list):
                    raise self._input_error(
                        descriptor,
                        f"{descriptor.key} must be a list",
                        expected=f"list[{descriptor.type.value}]",
                        received_type=type(value).__name__,
                    )
                if descriptor.required and not value:
                    raise self._input_error(
                        descriptor,
                        f"{descriptor.key} must not be empty",
                    )
                normalized[descriptor.key] = [
                    self._validate_input_value(descriptor, item, index=index)
                    for index, item in enumerate(value)
                ]
            else:
                normalized[descriptor.key] = self._validate_input_value(
                    descriptor,
                    value,
                )
        return normalized

    def _validate_input_value(
        self,
        descriptor: ToolInput,
        value: Any,
        *,
        index: int | None = None,
    ) -> Any:
        if value is None:
            raise self._input_error(
                descriptor,
                f"{descriptor.key} contains a null value",
                index=index,
            )

        if descriptor.type == InputType.TEXT:
            if not isinstance(value, str):
                raise self._type_error(descriptor, value, "text", index=index)
            if descriptor.required and not value.strip():
                raise self._input_error(
                    descriptor,
                    f"{descriptor.key} must not be empty",
                    index=index,
                )
            return value

        if descriptor.type == InputType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise self._type_error(descriptor, value, "number", index=index)
            if not math.isfinite(float(value)):
                raise self._input_error(
                    descriptor,
                    f"{descriptor.key} must be finite",
                    index=index,
                )
            self._validate_numeric_bounds(descriptor, float(value), index=index)
            return value

        if descriptor.type == InputType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise self._type_error(descriptor, value, "integer", index=index)
            self._validate_numeric_bounds(descriptor, float(value), index=index)
            return value

        if descriptor.type == InputType.BOOLEAN:
            if not isinstance(value, bool):
                raise self._type_error(descriptor, value, "boolean", index=index)
            return value

        if descriptor.type == InputType.CHOICE:
            if not isinstance(value, str):
                raise self._type_error(descriptor, value, "choice", index=index)
            allowed = {option.value for option in descriptor.options}
            if value not in allowed:
                raise self._input_error(
                    descriptor,
                    f"Invalid value for {descriptor.key}",
                    index=index,
                    allowed=sorted(allowed),
                )
            return value

        if descriptor.type in {
            InputType.IMAGE,
            InputType.VIDEO,
            InputType.AUDIO,
        }:
            try:
                asset = AssetInput.model_validate(value)
            except Exception as exc:
                raise self._input_error(
                    descriptor,
                    f"{descriptor.key} must reference an uploaded asset",
                    index=index,
                    expected="asset",
                ) from exc
            try:
                self.storage.resolve_asset(asset.asset_id)
            except FileNotFoundError as exc:
                raise self._input_error(
                    descriptor,
                    f"{descriptor.key} references a missing uploaded asset",
                    index=index,
                    asset_id=str(asset.asset_id),
                ) from exc
            return asset.model_dump(mode="json")

        if descriptor.type == InputType.RESOURCE:
            if not isinstance(value, str):
                raise self._type_error(descriptor, value, "resource ID", index=index)
            if not value.strip():
                raise self._input_error(
                    descriptor,
                    f"{descriptor.key} must not be empty",
                    index=index,
                )
            return value

        raise self._input_error(
            descriptor,
            f"Unsupported input type for {descriptor.key}",
            index=index,
            expected=descriptor.type.value,
        )

    def _validate_numeric_bounds(
        self,
        descriptor: ToolInput,
        value: float,
        *,
        index: int | None,
    ) -> None:
        if descriptor.ui is None:
            return
        if descriptor.ui.min is not None and value < descriptor.ui.min:
            raise self._input_error(
                descriptor,
                f"{descriptor.key} is below its minimum",
                index=index,
                min=descriptor.ui.min,
            )
        if descriptor.ui.max is not None and value > descriptor.ui.max:
            raise self._input_error(
                descriptor,
                f"{descriptor.key} exceeds its maximum",
                index=index,
                max=descriptor.ui.max,
            )

    def _type_error(
        self,
        descriptor: ToolInput,
        value: Any,
        expected: str,
        *,
        index: int | None,
    ) -> JobSubmissionError:
        return self._input_error(
            descriptor,
            f"{descriptor.key} must be {expected}",
            index=index,
            expected=expected,
            received_type=type(value).__name__,
        )

    @staticmethod
    def _input_error(
        descriptor: ToolInput,
        message: str,
        *,
        index: int | None = None,
        **details: Any,
    ) -> JobSubmissionError:
        payload: dict[str, Any] = {"input": descriptor.key, **details}
        if index is not None:
            payload["index"] = index
        return JobSubmissionError(
            ErrorBody(
                code="validation_failed",
                message=message,
                retryable=False,
                details=payload,
            )
        )


class JobSubmissionError(ValueError):
    def __init__(self, error: ErrorBody):
        super().__init__(error.message)
        self.error = error
