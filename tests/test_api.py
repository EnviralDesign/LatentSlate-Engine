from __future__ import annotations

import time
from pathlib import Path
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
from latentslate_engine.tools.base import Tool, ToolContext

TEST_TOOL_ID = UUID("dd7ff56c-1684-4b4d-bd1d-fdd96abc1535")
VALIDATION_TOOL_ID = UUID("b90c0f45-5b88-4a89-bf7d-5c57734ddcaf")
OOM_TOOL_ID = UUID("cf26772a-595d-4f9f-83e4-b6ce2f984bbc")


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


def settings(tmp_path: Path, token: str | None = None) -> Settings:
    return Settings(
        home=tmp_path,
        token=token,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )


def await_job(
    client: TestClient,
    job_id: str,
    headers: dict[str, str] | None = None,
):
    for _ in range(100):
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "canceled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def catalog_tool(client: TestClient, key: str) -> dict[str, Any]:
    tools = client.get("/v1/catalog").json()["tools"]
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
