from pathlib import Path

from fastapi.testclient import TestClient

from latentslate_engine.app import create_app
from latentslate_engine.config import Settings
from latentslate_engine.tools import ToolRegistry


def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )


def test_runtime_observability_and_cache_clear_endpoints(tmp_path: Path):
    app = create_app(settings(tmp_path), ToolRegistry([]))
    with TestClient(app) as client:
        status = client.get("/v1/runtime")
        assert status.status_code == 200
        assert status.json() == {
            "active_runtime": None,
            "max_wrappers": 8,
            "runtimes": [],
            "cleanup_errors": [],
        }

        cleared = client.delete("/v1/runtime/cache")
        assert cleared.status_code == 200
        assert cleared.json() == {
            "active_runtime": None,
            "max_wrappers": 8,
            "runtimes": [],
            "cleanup_errors": [],
        }


def test_asset_upload_response_exposes_content_hash_and_reuses_identity(tmp_path: Path):
    app = create_app(settings(tmp_path), ToolRegistry([]))
    with TestClient(app) as client:
        first = client.post(
            "/v1/assets",
            files={"file": ("first.png", b"identical", "image/png")},
        )
        second = client.post(
            "/v1/assets",
            files={"file": ("renamed.png", b"identical", "image/png")},
        )

        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["id"] == second.json()["id"]
        assert first.json()["sha256"] == second.json()["sha256"]
        assert len(first.json()["sha256"]) == 64
