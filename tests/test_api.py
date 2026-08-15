from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from latentslate_engine.app import create_app
from latentslate_engine.config import Settings
from latentslate_engine.protocol import (
    ChoiceOption,
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.storage import StoredArtifact
from latentslate_engine.tools import ToolRegistry
from latentslate_engine.tools.base import InputValidationError, Tool, ToolContext

TEST_TOOL_ID = UUID("dd7ff56c-1684-4b4d-bd1d-fdd96abc1535")
VALIDATION_TOOL_ID = UUID("b90c0f45-5b88-4a89-bf7d-5c57734ddcaf")
OOM_TOOL_ID = UUID("cf26772a-595d-4f9f-83e4-b6ce2f984bbc")
FAILURE_TOOL_ID = UUID("935701c9-a519-4cdb-876c-dbf3741216d7")
PREFLIGHT_TOOL_ID = UUID("aac75f7b-1b77-4252-afaa-b7d3e8a58582")
CANCELLABLE_TOOL_ID = UUID("279781ed-5485-4f37-ab78-8a46fa0c0c29")
LATE_CANCEL_TOOL_ID = UUID("29773d34-8eea-4a2f-90a2-a3aee6f3f02e")


class CopyTool(Tool):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=TEST_TOOL_ID,
            key="test.copy_image",
            schema_revision=1,
            name="Copy Image",
            workflow_kind=WorkflowKind.IMAGE_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(
                    key="source_image",
                    label="Source",
                    type=InputType.IMAGE,
                    role=InputRole.SOURCE_IMAGE,
                    required=True,
                    ui=InputUi(group="Input"),
                )
            ],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        from latentslate_engine.protocol import AssetInput

        source = AssetInput.model_validate(inputs["source_image"])
        source_path = context.resolve_asset(source.asset_id)
        output = context.storage.artifact_path(context.job_id, "copy.png")
        output.write_bytes(source_path.read_bytes())
        context.progress(1.0, "Complete")
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output.name,
                content_type="image/png",
                path=output,
                role="primary",
                media_type="image",
                metadata={"copied": True},
            )
        ]


class FakeManagedRuntime:
    def __init__(self) -> None:
        self.unload_count = 0
        self.clear_count = 0

    def unload(self) -> None:
        self.unload_count += 1

    def clear_cache(self) -> None:
        self.clear_count += 1


def test_runtime_reset_unloads_and_evicts_every_wrapper(tmp_path: Path):
    first = FakeManagedRuntime()
    second = FakeManagedRuntime()
    RUNTIME_MANAGER.clear()
    RUNTIME_MANAGER.activate(("test", "first"), lambda: first)
    RUNTIME_MANAGER.activate(("test", "second"), lambda: second)
    try:
        app = create_app(settings(tmp_path), ToolRegistry([CopyTool()]))
        with TestClient(app) as client:
            before = client.get("/v1/runtime")
            assert before.status_code == 200
            assert len(before.json()["runtimes"]) == 2
            before_counts = {
                id(first): (first.unload_count, first.clear_count),
                id(second): (second.unload_count, second.clear_count),
            }

            reset = client.delete("/v1/runtime")

        assert reset.status_code == 200
        assert reset.json()["active_runtime"] is None
        assert reset.json()["runtimes"] == []
        for runtime in (first, second):
            unloads, clears = before_counts[id(runtime)]
            assert runtime.unload_count == unloads + 1
            assert runtime.clear_count == clears + 1
    finally:
        RUNTIME_MANAGER.clear()


def test_runtime_reset_refuses_to_unload_while_a_job_is_active(tmp_path: Path):
    runtime = FakeManagedRuntime()
    RUNTIME_MANAGER.clear()
    RUNTIME_MANAGER.activate(("test", "busy"), lambda: runtime)
    try:
        app = create_app(settings(tmp_path), ToolRegistry([CopyTool()]))
        app.state.jobs.counts = lambda: (0, 1)
        with TestClient(app) as client:
            reset = client.delete("/v1/runtime")
            assert reset.status_code == 409
            assert "requires an idle Engine" in reset.json()["detail"]
            assert RUNTIME_MANAGER.status()["active_runtime"] == "test:busy"
            assert runtime.unload_count == 0
            assert runtime.clear_count == 0
    finally:
        RUNTIME_MANAGER.clear()


class CudaOomTool(Tool):
    def __init__(self) -> None:
        self.runtime: FakeManagedRuntime | None = None

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=OOM_TOOL_ID,
            key="test.cuda_oom",
            schema_revision=1,
            name="CUDA OOM",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del inputs
        self.runtime = RUNTIME_MANAGER.activate(("test", "cuda_oom"), FakeManagedRuntime)
        context.record_provenance(test_runtime_entered=True)
        raise RuntimeError("CUDA out of memory. Tried to allocate 10 MiB")


class FailureTool(Tool):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=FAILURE_TOOL_ID,
            key="test.failure",
            schema_revision=1,
            name="Failure",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del context, inputs
        raise RuntimeError("deliberate provider failure")


class SafeWorkerFailureTool(FailureTool):
    """Injects a hostile chained cause while exercising the safe worker seam."""

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del context, inputs
        from latentslate_engine.runtime.z_image_turbo_managed import ZImageWorkerFailure

        hostile = RuntimeError("C:\\private\\hostile-path\nPROMPT=do not disclose")
        raise ZImageWorkerFailure(
            "RuntimeError", "sampling", "z_image_turbo.generate", "a" * 64
        ) from hostile


class ValidationTool(Tool):
    def __init__(self) -> None:
        self.received_inputs: dict[str, Any] | None = None

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=VALIDATION_TOOL_ID,
            key="test.validation",
            schema_revision=1,
            name="Validation",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(
                    key="prompt",
                    label="Prompt",
                    type=InputType.TEXT,
                    required=True,
                ),
                ToolInput(
                    key="count",
                    label="Count",
                    type=InputType.INTEGER,
                    required=True,
                    default=4,
                    ui=InputUi(min=1, max=8, step=1),
                ),
                ToolInput(
                    key="ratio",
                    label="Ratio",
                    type=InputType.NUMBER,
                    default=0.5,
                    ui=InputUi(min=0, max=1),
                ),
                ToolInput(
                    key="enabled",
                    label="Enabled",
                    type=InputType.BOOLEAN,
                    default=False,
                ),
                ToolInput(
                    key="mode",
                    label="Mode",
                    type=InputType.CHOICE,
                    default="fast",
                    options=[
                        ChoiceOption(value="fast", label="Fast"),
                        ChoiceOption(value="slow", label="Slow"),
                    ],
                ),
                ToolInput(
                    key="tags",
                    label="Tags",
                    type=InputType.TEXT,
                    default=[],
                    multiple=True,
                ),
            ],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        self.received_inputs = inputs
        output = context.storage.artifact_path(context.job_id, "validated.bin")
        output.write_bytes(b"validated")
        context.progress(1.0, "Complete")
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output.name,
                content_type="application/octet-stream",
                path=output,
                role="primary",
                media_type="image",
                metadata={"inputs": inputs},
            )
        ]


class PreflightTool(Tool):
    def __init__(self) -> None:
        self.ran = False

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=PREFLIGHT_TOOL_ID,
            key="test.preflight",
            schema_revision=1,
            name="Preflight",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(key="width", label="Width", type=InputType.INTEGER, required=True),
                ToolInput(key="height", label="Height", type=InputType.INTEGER, required=True),
            ],
        ).with_schema_hash()

    def validate_inputs(self, inputs: dict[str, Any]) -> list[InputValidationError]:
        if inputs["height"] % 64:
            return [
                InputValidationError(
                    input_key="height",
                    message="height must be divisible by 64",
                    details={"alignment": 64},
                )
            ]
        return []

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del context, inputs
        self.ran = True
        return []


class CancellableTool(Tool):
    """A deterministic cooperative worker for exercising the public cancel API."""

    def __init__(self, sentinel_dir: Path) -> None:
        self.sentinel_dir = sentinel_dir
        self.started = Event()
        self.allow_success = Event()
        self.invoked_job_ids: list[UUID] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=CANCELLABLE_TOOL_ID,
            key="test.cancellable",
            schema_revision=1,
            name="Cancellable",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del inputs
        self.invoked_job_ids.append(context.job_id)
        self.started.set()
        while not self.allow_success.wait(timeout=0.005):
            context.check_cancelled()
        context.check_cancelled()
        (self.sentinel_dir / f"{context.job_id}.success").write_text("success")
        return []


class LateCancelTool(Tool):
    """Writes a job artifact, then lets the manager observe a late cancellation."""

    def __init__(self) -> None:
        self.wrote_artifact = Event()
        self.allow_return = Event()

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=LATE_CANCEL_TOOL_ID,
            key="test.late_cancel",
            schema_revision=1,
            name="Late Cancel",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[],
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        del inputs
        output = context.storage.artifact_path(context.job_id, "late-cancel.bin")
        output.write_bytes(b"must not publish")
        self.wrote_artifact.set()
        if not self.allow_return.wait(timeout=2.0):
            raise RuntimeError("late-cancel test tool was not released")
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output.name,
                content_type="application/octet-stream",
                path=output,
                role="primary",
                media_type="image",
                metadata={},
            )
        ]


def settings(tmp_path: Path, token: str | None = None) -> Settings:
    return Settings(
        home=tmp_path,
        token=token,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )


def await_job(
    client: TestClient,
    job_id: str,
    headers: dict[str, str] | None = None,
    *,
    timeout_seconds: float = 10.0,
):
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        last_payload = response.json()
        if last_payload["status"] in {"succeeded", "failed", "canceled"}:
            return last_payload
        time.sleep(0.02)
    raise AssertionError(
        f"job did not finish within {timeout_seconds:.1f}s; last payload={last_payload!r}"
    )


def await_status(
    client: TestClient,
    job_id: str,
    statuses: set[str],
    headers: dict[str, str] | None = None,
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in statuses:
            return last_payload
        time.sleep(0.005)
    raise AssertionError(
        f"job did not reach {sorted(statuses)!r} within {timeout_seconds:.1f}s; "
        f"last payload={last_payload!r}"
    )


def catalog_tool(
    client: TestClient, key: str, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    tools = client.get("/v1/catalog", headers=headers).json()["tools"]
    return next(tool for tool in tools if tool["key"] == key)


def create_job_payload(
    tool: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tool_id": tool["id"],
        "schema_revision": tool["schema_revision"],
        "schema_hash": tool["schema_hash"],
        "inputs": inputs,
    }


def test_catalog_upload_job_and_download(tmp_path: Path):
    app = create_app(settings(tmp_path), ToolRegistry([CopyTool()]))
    with TestClient(app) as client:
        catalog = client.get("/v1/catalog")
        assert catalog.status_code == 200
        tool = catalog.json()["tools"][0]
        assert tool["key"] == "test.copy_image"
        assert tool["schema_hash"].startswith("sha256:")

        uploaded = client.post(
            "/v1/assets",
            files={"file": ("source.png", b"not-a-real-png", "image/png")},
        )
        assert uploaded.status_code == 201
        asset_id = uploaded.json()["id"]

        created = client.post(
            "/v1/jobs",
            json=create_job_payload(
                tool,
                {"source_image": {"type": "asset", "asset_id": asset_id}},
            ),
        )
        assert created.status_code == 202
        job = await_job(client, created.json()["id"])
        assert job["status"] == "succeeded"
        assert job["artifacts"][0]["metadata"]["copied"] is True

        artifact = client.get(job["artifacts"][0]["download_url"])
        assert artifact.status_code == 200
        assert artifact.content == b"not-a-real-png"


def test_cuda_oom_evicts_active_runtime_and_preserves_failure_provenance(
    tmp_path: Path,
):
    tool_impl = CudaOomTool()
    RUNTIME_MANAGER.clear()
    try:
        app = create_app(settings(tmp_path), ToolRegistry([tool_impl]))
        with TestClient(app) as client:
            tool = catalog_tool(client, "test.cuda_oom")
            created = client.post(
                "/v1/jobs",
                json=create_job_payload(tool, {}),
            )
            assert created.status_code == 202
            job = await_job(client, created.json()["id"])

        assert job["status"] == "failed"
        assert job["error"]["code"] == "cuda_out_of_memory"
        assert job["error"]["details"]["active_runtime_evicted"] == "test:cuda_oom"
        assert job["provenance"]["test_runtime_entered"] is True
        assert job["provenance"]["runtime_failure"] == {
            "kind": "cuda_out_of_memory",
            "active_runtime_evicted": "test:cuda_oom",
        }
        assert tool_impl.runtime is not None
        assert tool_impl.runtime.unload_count == 1
        assert tool_impl.runtime.clear_count == 1
        assert RUNTIME_MANAGER.status()["active_runtime"] is None
        assert RUNTIME_MANAGER.status()["runtimes"] == []
    finally:
        RUNTIME_MANAGER.clear()


def test_failed_job_logs_tool_code_message_and_traceback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger="latentslate_engine.jobs")
    app = create_app(settings(tmp_path), ToolRegistry([FailureTool()]))

    with TestClient(app) as client:
        tool = catalog_tool(client, "test.failure")
        created = client.post("/v1/jobs", json=create_job_payload(tool, {}))
        assert created.status_code == 202
        job = await_job(client, created.json()["id"])

    assert job["status"] == "failed"
    assert job["error"]["code"] == "generation_failed"
    record = next(record for record in caplog.records if record.name == "latentslate_engine.jobs")
    assert str(job["id"]) in record.getMessage()
    assert "tool=test.failure" in record.getMessage()
    assert "code=generation_failed" in record.getMessage()
    assert record.exc_info is not None
    assert "deliberate provider failure" in str(record.exc_info[1])


def test_safe_worker_failure_never_logs_traceback_or_hostile_cause(tmp_path: Path, caplog):
    caplog.set_level(logging.ERROR, logger="latentslate_engine.jobs")
    app = create_app(settings(tmp_path), ToolRegistry([SafeWorkerFailureTool()]))

    with TestClient(app) as client:
        tool = catalog_tool(client, "test.failure")
        created = client.post("/v1/jobs", json=create_job_payload(tool, {}))
        assert created.status_code == 202
        job = await_job(client, created.json()["id"])

    hostile = "C:\\private\\hostile-path\nPROMPT=do not disclose"
    assert job["status"] == "failed"
    assert job["error"] == {
        "code": "generation_failed",
        "message": (
            "Z-Image worker failed (RuntimeError during sampling at z_image_turbo.generate; "
            "diagnostic aaaaaaaaaaaa)"
        ),
        "retryable": False,
        "details": {},
    }
    records = [record for record in caplog.records if record.name == "latentslate_engine.jobs"]
    assert len(records) == 1
    record = records[0]
    assert record.exc_info is None
    assert "Traceback" not in caplog.text
    assert hostile not in caplog.text and "hostile-path" not in caplog.text
    assert "code=generation_failed" in record.getMessage()
    assert "type=RuntimeError" in record.getMessage()
    assert "diagnostic=aaaaaaaaaaaa" in record.getMessage()


def test_schema_mismatch_is_explicit(tmp_path: Path):
    app = create_app(settings(tmp_path), ToolRegistry([CopyTool()]))
    with TestClient(app) as client:
        tool = client.get("/v1/catalog").json()["tools"][0]
        response = client.post(
            "/v1/jobs",
            json={
                "tool_id": tool["id"],
                "schema_revision": tool["schema_revision"],
                "schema_hash": "sha256:stale",
                "inputs": {},
            },
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "schema_mismatch"


def test_optional_bearer_auth(tmp_path: Path):
    app = create_app(settings(tmp_path, token="secret"), ToolRegistry([CopyTool()]))
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 401
        response = client.get(
            "/v1/health",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200


def test_authenticated_cancel_running_job_is_terminal_and_publishes_nothing(
    tmp_path: Path,
):
    tool_impl = CancellableTool(tmp_path)
    headers = {"Authorization": "Bearer test-token"}
    app = create_app(
        settings(tmp_path, token="test-token"), ToolRegistry([tool_impl])
    )

    with TestClient(app) as client:
        tool = catalog_tool(client, "test.cancellable", headers)
        created = client.post(
            "/v1/jobs", json=create_job_payload(tool, {}), headers=headers
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        running = await_status(client, job_id, {"running"}, headers)
        assert running["message"] == "Starting"
        assert tool_impl.started.wait(timeout=2.0)

        canceled = client.delete(f"/v1/jobs/{job_id}", headers=headers)
        assert canceled.status_code == 200
        cancel_payload = canceled.json()
        assert cancel_payload["status"] in {"running", "canceled"}
        if cancel_payload["status"] == "running":
            assert cancel_payload["message"] == "Cancellation requested"

        terminal = await_job(client, job_id, headers)

    assert terminal["status"] == "canceled"
    assert terminal["artifacts"] == []
    assert not (tmp_path / f"{job_id}.success").exists()


def test_late_cancel_removes_owned_job_folder_before_reporting_canceled(
    tmp_path: Path,
):
    tool_impl = LateCancelTool()
    headers = {"Authorization": "Bearer test-token"}
    app_settings = settings(tmp_path, token="test-token")
    app = create_app(app_settings, ToolRegistry([tool_impl]))

    with TestClient(app) as client:
        tool = catalog_tool(client, "test.late_cancel", headers)
        created = client.post(
            "/v1/jobs", json=create_job_payload(tool, {}), headers=headers
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        await_status(client, job_id, {"running"}, headers)
        assert tool_impl.wrote_artifact.wait(timeout=2.0)
        job_folder = app_settings.jobs_dir / job_id
        assert (job_folder / "late-cancel.bin").read_bytes() == b"must not publish"

        canceled = client.delete(f"/v1/jobs/{job_id}", headers=headers)
        assert canceled.status_code == 200
        assert canceled.json()["status"] == "running"
        assert canceled.json()["message"] == "Cancellation requested"
        tool_impl.allow_return.set()
        terminal = await_job(client, job_id, headers)

    assert terminal["status"] == "canceled"
    assert terminal["artifacts"] == []
    assert terminal["provenance"]["cancellation_cleanup"] == {"status": "complete"}
    assert not job_folder.exists()


def test_late_cancel_cleanup_failure_is_explicit_and_preserves_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    tool_impl = LateCancelTool()
    headers = {"Authorization": "Bearer test-token"}
    app_settings = settings(tmp_path, token="test-token")
    app = create_app(app_settings, ToolRegistry([tool_impl]))

    def fail_cleanup(_job_id: UUID) -> None:
        raise OSError("PRIVATE_PATH_SENTINEL\nPRIVATE_NEWLINE_SENTINEL")

    monkeypatch.setattr(app.state.storage, "remove_job_folder", fail_cleanup)
    caplog.set_level(logging.ERROR, logger="latentslate_engine.jobs")
    with TestClient(app) as client:
        tool = catalog_tool(client, "test.late_cancel", headers)
        created = client.post(
            "/v1/jobs", json=create_job_payload(tool, {}), headers=headers
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        await_status(client, job_id, {"running"}, headers)
        assert tool_impl.wrote_artifact.wait(timeout=2.0)
        job_folder = app_settings.jobs_dir / job_id

        canceled = client.delete(f"/v1/jobs/{job_id}", headers=headers)
        assert canceled.status_code == 200
        tool_impl.allow_return.set()
        terminal = await_job(client, job_id, headers)

    assert terminal["status"] == "failed"
    assert terminal["message"] == "Canceled job cleanup failed"
    assert terminal["error"] == {
        "code": "cancellation_cleanup_failed",
        "message": "Generation was canceled, but Engine could not remove partial outputs.",
        "retryable": False,
        "details": {"cleanup_error_type": "OSError"},
    }
    assert terminal["provenance"]["cancellation"] == {
        "requested": True,
        "completed": False,
    }
    assert terminal["provenance"]["cancellation_cleanup"] == {
        "status": "failed",
        "error_type": "OSError",
    }
    assert "PRIVATE_PATH_SENTINEL" not in str(terminal)
    assert "PRIVATE_NEWLINE_SENTINEL" not in str(terminal)
    assert (job_folder / "late-cancel.bin").read_bytes() == b"must not publish"
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "latentslate_engine.jobs"
    ]
    assert any(
        "code=cancellation_cleanup_failed" in message and "type=OSError" in message
        for message in messages
    )
    assert all("PRIVATE_PATH_SENTINEL" not in message for message in messages)
    assert all("PRIVATE_NEWLINE_SENTINEL" not in message for message in messages)


def test_authenticated_cancel_queued_job_never_invokes_the_tool(tmp_path: Path):
    tool_impl = CancellableTool(tmp_path)
    headers = {"Authorization": "Bearer test-token"}
    app = create_app(
        settings(tmp_path, token="test-token"), ToolRegistry([tool_impl])
    )

    with TestClient(app) as client:
        tool = catalog_tool(client, "test.cancellable", headers)
        first = client.post(
            "/v1/jobs", json=create_job_payload(tool, {}), headers=headers
        )
        assert first.status_code == 202
        first_job_id = first.json()["id"]
        await_status(client, first_job_id, {"running"}, headers)
        assert tool_impl.started.wait(timeout=2.0)

        queued = client.post(
            "/v1/jobs", json=create_job_payload(tool, {}), headers=headers
        )
        assert queued.status_code == 202
        queued_job_id = queued.json()["id"]
        queued_state = client.get(f"/v1/jobs/{queued_job_id}", headers=headers)
        assert queued_state.status_code == 200
        assert queued_state.json()["status"] == "queued"

        canceled = client.delete(f"/v1/jobs/{queued_job_id}", headers=headers)
        assert canceled.status_code == 200
        cancel_payload = canceled.json()
        assert cancel_payload["status"] == "canceled"
        assert cancel_payload["message"] == "Canceled before execution"
        assert cancel_payload["artifacts"] == []

        tool_impl.allow_success.set()
        assert await_job(client, first_job_id, headers)["status"] == "succeeded"
        terminal = await_job(client, queued_job_id, headers)

    assert terminal["status"] == "canceled"
    assert terminal["artifacts"] == []
    assert UUID(queued_job_id) not in tool_impl.invoked_job_ids
    assert not (tmp_path / f"{queued_job_id}.success").exists()


def test_schema_defaults_are_applied_before_execution(tmp_path: Path):
    tool_impl = ValidationTool()
    app = create_app(settings(tmp_path), ToolRegistry([tool_impl]))
    with TestClient(app) as client:
        tool = catalog_tool(client, "test.validation")
        created = client.post(
            "/v1/jobs",
            json=create_job_payload(tool, {"prompt": "A small red boat"}),
        )
        assert created.status_code == 202
        job = await_job(client, created.json()["id"])
        assert job["status"] == "succeeded"

    assert tool_impl.received_inputs == {
        "prompt": "A small red boat",
        "count": 4,
        "ratio": 0.5,
        "enabled": False,
        "mode": "fast",
        "tags": [],
    }


@pytest.mark.parametrize(
    ("inputs", "message_fragment"),
    [
        ({"prompt": "test", "count": True}, "must be integer"),
        ({"prompt": "test", "count": 9}, "exceeds its maximum"),
        ({"prompt": "test", "ratio": "0.5"}, "must be number"),
        ({"prompt": "test", "mode": "turbo"}, "Invalid value"),
        ({"prompt": "   "}, "must not be empty"),
        ({"prompt": "test", "tags": "one"}, "must be a list"),
        ({"prompt": "test", "unknown": 1}, "Unknown tool inputs"),
    ],
)
def test_invalid_inputs_are_rejected_before_queueing(
    tmp_path: Path,
    inputs: dict[str, Any],
    message_fragment: str,
):
    app = create_app(settings(tmp_path), ToolRegistry([ValidationTool()]))
    with TestClient(app) as client:
        tool = catalog_tool(client, "test.validation")
        response = client.post(
            "/v1/jobs",
            json=create_job_payload(tool, inputs),
        )
        assert response.status_code == 422
        payload = response.json()["error"]
        assert payload["code"] == "validation_failed"
        assert message_fragment in payload["message"]


def test_cross_field_tool_preflight_rejects_before_queueing(tmp_path: Path):
    tool_impl = PreflightTool()
    app = create_app(settings(tmp_path), ToolRegistry([tool_impl]))
    with TestClient(app) as client:
        tool = catalog_tool(client, "test.preflight")
        response = client.post(
            "/v1/jobs",
            json=create_job_payload(tool, {"width": 960, "height": 540}),
        )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "validation_failed",
        "message": "height must be divisible by 64",
        "retryable": False,
        "details": {"input": "height", "alignment": 64},
    }
    assert tool_impl.ran is False


def test_missing_uploaded_asset_is_rejected_before_queueing(tmp_path: Path):
    app = create_app(settings(tmp_path), ToolRegistry([CopyTool()]))
    with TestClient(app) as client:
        tool = catalog_tool(client, "test.copy_image")
        missing_asset_id = str(uuid4())
        response = client.post(
            "/v1/jobs",
            json=create_job_payload(
                tool,
                {
                    "source_image": {
                        "type": "asset",
                        "asset_id": missing_asset_id,
                    }
                },
            ),
        )
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_failed"
        assert error["details"]["input"] == "source_image"
        assert error["details"]["asset_id"] == missing_asset_id
