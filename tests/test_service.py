from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from latentslate_engine.service import FLF_ID, I2V_ID, T2V_ID, TOOLS, create_app


class FakeRuntime:
    available = True

    def __init__(self, *, blocked: bool = False) -> None:
        self.blocked = blocked
        self.started = threading.Event()
        self.finish = threading.Event()
        self.operations: list[str] = []
        self.release_count = 0

    def generate(
        self, operation: str, inputs: dict[str, Any], output_path: Path
    ) -> None:
        self.operations.append(operation)
        self.started.set()
        if self.blocked and not self.finish.wait(5):
            raise RuntimeError("test runtime timed out")
        output_path.write_bytes(b"test-mp4")

    def release(self) -> None:
        self.release_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "operation": self.operations[-1] if self.operations else None,
            "generation_count": len(self.operations),
            "reuse_count": 0,
            "switch_count": 0,
            "release_count": self.release_count,
        }


def _png(width: int = 64, height: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(buffer, "PNG")
    return buffer.getvalue()


def _tool(tool_id: str) -> dict[str, Any]:
    return next(tool for tool in TOOLS if tool["id"] == tool_id)


def _job_body(tool_id: str, **inputs: Any) -> dict[str, Any]:
    tool = _tool(tool_id)
    defaults = {
        "prompt": "A small test scene",
        "width": 64,
        "height": 64,
        "duration_seconds": 1.0,
        "seed": 7,
    }
    defaults.update(inputs)
    return {
        "tool_id": tool_id,
        "schema_revision": tool["schema_revision"],
        "schema_hash": tool["schema_hash"],
        "inputs": defaults,
    }


def _wait_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_health_and_catalog_expose_only_stable_ltx_tools(tmp_path: Path) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        catalog = client.get("/v1/catalog").json()
        assert catalog["protocol_version"] == "1.0"
        assert [tool["id"] for tool in catalog["tools"]] == [T2V_ID, I2V_ID, FLF_ID]
        assert [tool["key"] for tool in catalog["tools"]] == [
            "ltx23.text_to_video",
            "ltx23.image_to_video",
            "ltx23.first_last_frame_to_video",
        ]
        assert all(tool["schema_revision"] == 2 for tool in catalog["tools"])
        assert [tool["schema_hash"] for tool in catalog["tools"]] == [
            "sha256:94f9397a5ff16d5101e81f62396c5c744f045799bcdbdf961b036ee8f0ac2c78",
            "sha256:8364fcc55ec44ae780d49d9c9404768c81a5680783106934f9a17bd990be7efa",
            "sha256:aa624d8d8fe060dcc39c15623e4b4b07eb405305051ebdd5fd2caf8368d8acd9",
        ]
        assert catalog["tools"][0]["canvas"] == {
            "alignment": 64,
            "min_side": 64,
            "max_pixels": 942080,
        }
        assert catalog["tools"][1]["canvas"]["alignment"] == 64
        assert catalog["tools"][2]["canvas"]["alignment"] == 32
        assert catalog == client.get("/v1/catalog").json()


def test_bearer_auth_protects_the_complete_v1_surface(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer unit-token"}
    with TestClient(
        create_app(home=tmp_path, token="unit-token", executor=FakeRuntime())
    ) as client:
        unauthorized = client.get("/v1/catalog")
        assert unauthorized.status_code == 401
        assert unauthorized.json() == {
            "error": {"message": "Bearer authentication required"}
        }
        assert client.get("/v1/health", headers=headers).status_code == 200
        assert (
            client.post(
                "/v1/assets",
                files={"file": ("source.png", _png(), "image/png")},
            ).status_code
            == 401
        )


def test_asset_job_poll_and_artifact_download(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        uploaded = client.post(
            "/v1/assets", files={"file": ("source.png", _png(), "image/png")}
        )
        assert uploaded.status_code == 200
        asset = {"type": "asset", "asset_id": uploaded.json()["id"]}
        submitted = client.post("/v1/jobs", json=_job_body(I2V_ID, start_image=asset))
        assert submitted.status_code == 200
        assert submitted.json()["status"] in {"queued", "running", "succeeded"}

        job = _wait_terminal(client, submitted.json()["id"])
        assert job["status"] == "succeeded"
        assert job["artifacts"] == [
            {
                "role": "primary",
                "filename": "output.mp4",
                "download_url": f"/v1/artifacts/{job['id']}/output.mp4",
            }
        ]
        artifact = client.get(job["artifacts"][0]["download_url"])
        assert artifact.status_code == 200
        assert artifact.content == b"test-mp4"
        assert runtime.operations == ["i2v"]


def test_job_contract_rejects_stale_schema_invalid_geometry_and_assets(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        stale_revision = _job_body(T2V_ID)
        stale_revision["schema_revision"] = 1
        assert client.post("/v1/jobs", json=stale_revision).status_code == 409

        stale_hash = _job_body(T2V_ID)
        stale_hash["schema_hash"] = "sha256:stale"
        assert client.post("/v1/jobs", json=stale_hash).status_code == 409

        invalid_geometry = _job_body(T2V_ID, width=96)
        invalid = client.post("/v1/jobs", json=invalid_geometry)
        assert invalid.status_code == 422
        assert "divisible by 64" in invalid.json()["error"]["message"]

        missing_asset = _job_body(
            I2V_ID,
            start_image={
                "type": "asset",
                "asset_id": "00000000-0000-0000-0000-000000000000",
            },
        )
        assert client.post("/v1/jobs", json=missing_asset).status_code == 422

        uploaded = client.post(
            "/v1/assets", files={"file": ("wrong.png", _png(128, 64), "image/png")}
        ).json()
        wrong_canvas = _job_body(
            I2V_ID,
            start_image={"type": "asset", "asset_id": uploaded["id"]},
        )
        response = client.post("/v1/jobs", json=wrong_canvas)
        assert response.status_code == 422
        assert "canvas dimensions" in response.json()["error"]["message"]


def test_runtime_release_is_visible(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        response = client.delete("/v1/runtime")
        assert response.status_code == 200
        assert response.json()["released"] is True
        assert response.json()["runtime"]["release_count"] == 1


def test_running_cancellation_waits_for_native_quiescence_and_discards_output(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(blocked=True)
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        submitted = client.post("/v1/jobs", json=_job_body(T2V_ID)).json()
        assert runtime.started.wait(1)

        canceled = client.delete(f"/v1/jobs/{submitted['id']}")
        assert canceled.status_code == 200
        assert canceled.json()["status"] == "running"
        assert canceled.json()["message"] == "Cancellation requested"
        assert client.get("/v1/health").status_code == 200
        assert client.get(f"/v1/jobs/{submitted['id']}").json()["status"] == "running"

        runtime.finish.set()
        terminal = _wait_terminal(client, submitted["id"])
        assert terminal["status"] == "canceled"
        assert terminal["artifacts"] == []
        assert (
            client.get(f"/v1/artifacts/{submitted['id']}/output.mp4").status_code == 404
        )
