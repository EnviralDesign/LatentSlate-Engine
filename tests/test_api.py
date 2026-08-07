from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from latentslate_engine.app import create_app
from latentslate_engine.config import Settings
from latentslate_engine.protocol import (
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from latentslate_engine.storage import StoredArtifact
from latentslate_engine.tools import ToolRegistry
from latentslate_engine.tools.base import Tool, ToolContext


TEST_TOOL_ID = UUID("dd7ff56c-1684-4b4d-bd1d-fdd96abc1535")


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


def settings(tmp_path: Path, token: str | None = None) -> Settings:
    return Settings(
        home=tmp_path,
        token=token,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )


def await_job(client: TestClient, job_id: str, headers: dict[str, str] | None = None):
    for _ in range(100):
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "canceled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


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
            json={
                "tool_id": tool["id"],
                "schema_revision": tool["schema_revision"],
                "schema_hash": tool["schema_hash"],
                "inputs": {"source_image": {"type": "asset", "asset_id": asset_id}},
            },
        )
        assert created.status_code == 202
        job = await_job(client, created.json()["id"])
        assert job["status"] == "succeeded"
        assert job["artifacts"][0]["metadata"]["copied"] is True

        artifact = client.get(job["artifacts"][0]["download_url"])
        assert artifact.status_code == 200
        assert artifact.content == b"not-a-real-png"


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
            "/v1/health", headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200
