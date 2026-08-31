from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from latentslate_engine.service import (
    FLF_ID,
    I2V_ID,
    KLEIN_T2I_ID,
    KLEIN_TWO_IMAGE_ID,
    MAX_ASSET_COUNT,
    MAX_JOB_COUNT,
    T2V_ID,
    TOOLS,
    ActiveRuntimeOwner,
    EngineService,
    KleinModelPaths,
    LtxModelPaths,
    create_app,
)


class FakeRuntime:
    def __init__(
        self,
        *,
        blocked: bool = False,
        available_families: set[str] | None = None,
    ) -> None:
        self.blocked = blocked
        self.available_families = available_families or {"ltx", "klein"}
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
        output_path.write_bytes(
            b"test-png" if output_path.suffix == ".png" else b"test-mp4"
        )

    def available(self, operation: str) -> bool:
        family = "klein" if operation.startswith("klein_") else "ltx"
        return family in self.available_families

    def unavailable_reason(self, operation: str) -> str:
        family = "Klein" if operation.startswith("klein_") else "LTX"
        return f"Required {family} model files are not installed"

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
    if tool_id in {KLEIN_T2I_ID, KLEIN_TWO_IMAGE_ID}:
        defaults = {
            "prompt": "A small test scene",
            "width": 256,
            "height": 256,
            "seed": 7,
        }
    else:
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


def test_health_and_catalog_expose_five_stable_tools(tmp_path: Path) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        catalog = client.get("/v1/catalog").json()
        assert catalog["protocol_version"] == "1.0"
        assert [tool["id"] for tool in catalog["tools"]] == [
            T2V_ID,
            I2V_ID,
            FLF_ID,
            KLEIN_T2I_ID,
            KLEIN_TWO_IMAGE_ID,
        ]
        assert [tool["key"] for tool in catalog["tools"]] == [
            "ltx23.text_to_video",
            "ltx23.image_to_video",
            "ltx23.first_last_frame_to_video",
            "flux2_klein9b.text_to_image",
            "flux2_klein9b.two_image_to_image",
        ]
        assert [tool["schema_revision"] for tool in catalog["tools"]] == [2, 2, 2, 1, 1]
        assert [tool["schema_hash"] for tool in catalog["tools"]] == [
            "sha256:94f9397a5ff16d5101e81f62396c5c744f045799bcdbdf961b036ee8f0ac2c78",
            "sha256:8364fcc55ec44ae780d49d9c9404768c81a5680783106934f9a17bd990be7efa",
            "sha256:aa624d8d8fe060dcc39c15623e4b4b07eb405305051ebdd5fd2caf8368d8acd9",
            "sha256:2e94d609c2db43e883da19fb0c73faa1bef7f3459c916760079f7cedd212c6b3",
            "sha256:d756bc62e593edd29f3c2c909f3c92fd22d10cb2fb44a2b51bdd93afdb605ed8",
        ]
        assert catalog["tools"][0]["canvas"] == {
            "alignment": 64,
            "min_side": 64,
            "max_pixels": 942080,
        }
        assert catalog["tools"][1]["canvas"]["alignment"] == 64
        assert catalog["tools"][2]["canvas"]["alignment"] == 32
        assert catalog["tools"][3]["canvas"] == {
            "alignment": 16,
            "min_side": 256,
            "max_pixels": 1048576,
            "max_aspect": 4.0,
        }
        assert [item["key"] for item in catalog["tools"][4]["inputs"]] == [
            "prompt",
            "image_1",
            "image_2",
            "width",
            "height",
            "seed",
        ]
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


def test_klein_two_image_preserves_source_geometry_and_publishes_png(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        first = client.post(
            "/v1/assets",
            files={"file": ("first.png", _png(320, 640), "image/png")},
        ).json()
        second = client.post(
            "/v1/assets",
            files={"file": ("second.png", _png(768, 384), "image/png")},
        ).json()
        body = _job_body(
            KLEIN_TWO_IMAGE_ID,
            width=512,
            height=256,
            image_1={"type": "asset", "asset_id": first["id"]},
            image_2={"type": "asset", "asset_id": second["id"]},
        )
        submitted = client.post("/v1/jobs", json=body)
        assert submitted.status_code == 200

        job = _wait_terminal(client, submitted.json()["id"])
        assert job["status"] == "succeeded"
        assert job["artifacts"] == [
            {
                "role": "primary",
                "filename": "output.png",
                "download_url": f"/v1/artifacts/{job['id']}/output.png",
            }
        ]
        artifact = client.get(job["artifacts"][0]["download_url"])
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "image/png"
        assert artifact.content == b"test-png"
        assert runtime.operations == ["klein_two_image"]


def test_catalog_and_submission_use_independent_family_availability(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(available_families={"klein"})
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        catalog = client.get("/v1/catalog").json()["tools"]
        assert [tool["available"] for tool in catalog] == [
            False,
            False,
            False,
            True,
            True,
        ]
        assert "LTX" in catalog[0]["unavailable_reason"]
        assert "unavailable_reason" not in catalog[3]

        unavailable = client.post("/v1/jobs", json=_job_body(T2V_ID))
        assert unavailable.status_code == 503
        assert "LTX" in unavailable.json()["error"]["message"]
        available = client.post("/v1/jobs", json=_job_body(KLEIN_T2I_ID))
        assert available.status_code == 200
        assert _wait_terminal(client, available.json()["id"])["status"] == "succeeded"

    runtime = FakeRuntime(available_families={"ltx"})
    with TestClient(create_app(home=tmp_path / "ltx", executor=runtime)) as client:
        catalog = client.get("/v1/catalog").json()["tools"]
        assert [tool["available"] for tool in catalog] == [
            True,
            True,
            True,
            False,
            False,
        ]
        unavailable = client.post("/v1/jobs", json=_job_body(KLEIN_T2I_ID))
        assert unavailable.status_code == 503
        assert "Klein" in unavailable.json()["error"]["message"]
        available = client.post("/v1/jobs", json=_job_body(T2V_ID))
        assert available.status_code == 200
        assert _wait_terminal(client, available.json()["id"])["status"] == "succeeded"


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


def test_klein_request_domain_is_explicit_and_has_no_duration_input(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        accepted = _job_body(
            KLEIN_T2I_ID,
            width=2048,
            height=512,
            seed=(1 << 64) - 1,
        )
        response = client.post("/v1/jobs", json=accepted)
        assert response.status_code == 200
        assert _wait_terminal(client, response.json()["id"])["status"] == "succeeded"

        invalid_cases = [
            (_job_body(KLEIN_T2I_ID, width=255), "divisible by 16"),
            (_job_body(KLEIN_T2I_ID, width=240), "at least 256"),
            (_job_body(KLEIN_T2I_ID, width=2064, height=512), "must not exceed"),
            (_job_body(KLEIN_T2I_ID, width=1280, height=256), "must not exceed 4:1"),
            (_job_body(KLEIN_T2I_ID, seed=1 << 64), "Klein seed"),
        ]
        for body, message in invalid_cases:
            rejected = client.post("/v1/jobs", json=body)
            assert rejected.status_code == 422
            assert message in rejected.json()["error"]["message"]

        duration = _job_body(KLEIN_T2I_ID)
        duration["inputs"]["duration_seconds"] = 1.0
        rejected = client.post("/v1/jobs", json=duration)
        assert rejected.status_code == 422
        assert "unknown inputs" in rejected.json()["error"]["message"]


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


def test_completed_job_capacity_reclaims_oldest_terminal_history(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        first_job_id = None
        for _ in range(MAX_JOB_COUNT + 2):
            response = client.post("/v1/jobs", json=_job_body(T2V_ID))
            assert response.status_code == 200
            job_id = response.json()["id"]
            first_job_id = first_job_id or job_id
            assert _wait_terminal(client, job_id)["status"] == "succeeded"

        service = client.app.state.engine_service
        assert len(service._jobs) == MAX_JOB_COUNT
        assert len(list(service.job_root.iterdir())) == MAX_JOB_COUNT
        assert client.get(f"/v1/jobs/{first_job_id}").status_code == 404


def test_completed_media_jobs_reclaim_request_assets(tmp_path: Path) -> None:
    with TestClient(create_app(home=tmp_path, executor=FakeRuntime())) as client:
        for _ in range(MAX_ASSET_COUNT + 2):
            uploaded = client.post(
                "/v1/assets", files={"file": ("source.png", _png(), "image/png")}
            )
            assert uploaded.status_code == 200
            asset = {"type": "asset", "asset_id": uploaded.json()["id"]}
            submitted = client.post(
                "/v1/jobs", json=_job_body(I2V_ID, start_image=asset)
            )
            assert submitted.status_code == 200
            assert _wait_terminal(client, submitted.json()["id"])["status"] == (
                "succeeded"
            )

        service = client.app.state.engine_service
        assert service._assets == {}
        assert service._asset_bytes == 0
        assert list(service.asset_root.iterdir()) == []
        assert len(service._jobs) == MAX_JOB_COUNT


def test_shared_asset_stays_until_running_job_reaches_quiescence(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(blocked=True)
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        uploaded = client.post(
            "/v1/assets", files={"file": ("source.png", _png(), "image/png")}
        ).json()
        asset = {"type": "asset", "asset_id": uploaded["id"]}
        running = client.post(
            "/v1/jobs", json=_job_body(I2V_ID, start_image=asset)
        ).json()
        assert runtime.started.wait(1)
        queued = client.post(
            "/v1/jobs", json=_job_body(I2V_ID, start_image=asset)
        ).json()

        assert client.delete(f"/v1/jobs/{queued['id']}").json()["status"] == "canceled"
        service = client.app.state.engine_service
        assert len(service._assets) == 1
        assert next(iter(service._assets.values())).path.is_file()

        runtime.finish.set()
        assert _wait_terminal(client, running["id"])["status"] == "succeeded"
        assert service._assets == {}
        assert service._asset_bytes == 0


def test_asset_validation_and_job_admission_are_atomic_with_reclamation(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(blocked=True)
    with TestClient(create_app(home=tmp_path, executor=runtime)) as client:
        uploaded = client.post(
            "/v1/assets", files={"file": ("source.png", _png(), "image/png")}
        ).json()
        asset = {"type": "asset", "asset_id": uploaded["id"]}
        service = client.app.state.engine_service
        first = service.submit(_job_body(I2V_ID, start_image=asset))
        assert runtime.started.wait(1)

        original_resolve = service._resolve_asset
        validation_entered = threading.Event()
        admit_second = threading.Event()

        def gated_resolve(value: Any, expected_size: tuple[int, int] | None):
            resolved = original_resolve(value, expected_size)
            validation_entered.set()
            assert admit_second.wait(2)
            return resolved

        service._resolve_asset = gated_resolve
        submission: dict[str, Any] = {}

        def submit_second() -> None:
            try:
                submission["job"] = service.submit(_job_body(I2V_ID, start_image=asset))
            except Exception as error:  # noqa: BLE001 - asserted below
                submission["error"] = error

        submitter = threading.Thread(target=submit_second)
        submitter.start()
        assert validation_entered.wait(1)
        acquired_during_validation = service._lock.acquire(blocking=False)
        if acquired_during_validation:
            service._lock.release()

        runtime.finish.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not first.output_path.exists():
            time.sleep(0.01)
        runtime.finish.clear()
        admit_second.set()
        submitter.join(2)

        second = submission.get("job")
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and (
            first.status == "running" or len(runtime.operations) < 2
        ):
            time.sleep(0.01)
        asset_exists_for_second = (
            len(service._assets) == 1
            and next(iter(service._assets.values())).path.is_file()
        )
        runtime.finish.set()
        if second is not None:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and second.status in {
                "queued",
                "running",
            }:
                time.sleep(0.01)

        assert "error" not in submission
        assert not acquired_during_validation
        assert first.status == "succeeded"
        assert second is not None and second.status == "succeeded"
        assert asset_exists_for_second
        assert service._assets == {}


def test_shutdown_cancels_queued_work_and_waits_only_for_running_native_call(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(blocked=True)
    service = EngineService(tmp_path, runtime)
    jobs = [service.submit(_job_body(T2V_ID))]
    assert runtime.started.wait(1)
    jobs.extend(service.submit(_job_body(T2V_ID)) for _ in range(3))

    closer = threading.Thread(target=service.close)
    closer.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not all(job.cancel_requested for job in jobs):
        time.sleep(0.01)

    assert closer.is_alive()
    assert jobs[0].status == "running"
    assert all(job.status == "canceled" for job in jobs[1:])
    runtime.finish.set()
    closer.join(2)

    assert not closer.is_alive()
    assert runtime.operations == ["t2v"]
    assert all(job.status == "canceled" for job in jobs)
    assert all(not job.output_path.exists() for job in jobs)
    assert runtime.release_count == 1


def test_active_owner_reuses_one_klein_worker_and_replaces_cross_family(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    owner = ActiveRuntimeOwner(
        LtxModelPaths(missing, missing, missing, missing, missing),
        KleinModelPaths(missing, missing, missing, missing),
    )
    owner._availability = {"ltx": True, "klein": True}
    events: list[str] = []
    processes: list[Any] = []

    class FakeProcess:
        def __init__(self, name: str, pid: int) -> None:
            self.name = name
            self.pid = pid
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: int) -> None:
            events.append(f"join:{self.name}")
            self.alive = False

        def terminate(self) -> None:
            events.append(f"terminate:{self.name}")
            self.alive = False

    class FakeConnection:
        def __init__(self, process: FakeProcess) -> None:
            self.process = process
            self.operation = ""

        def send(self, message: dict[str, Any]) -> None:
            if message["type"] == "close":
                events.append(f"close:{self.process.name}")
            else:
                self.operation = message["operation"]
                events.append(f"generate:{self.operation}:{self.process.pid}")

        def recv(self) -> dict[str, Any]:
            return {
                "ok": True,
                "details": {"operation": self.operation},
            }

        def close(self) -> None:
            events.append(f"connection-close:{self.process.name}")

    def start_worker(family: str, operation: str) -> None:
        assert all(not process.alive for process in processes)
        process = FakeProcess(f"{family}:{operation}", len(processes) + 1)
        processes.append(process)
        owner._process = process
        owner._connection = FakeConnection(process)
        owner._family = family
        owner._worker_operation = operation if family == "ltx" else None
        owner._last_operation = None
        owner._last_generation = None
        events.append(f"start:{process.name}")

    owner._start_worker = start_worker  # type: ignore[method-assign]
    output = tmp_path / "output.bin"
    owner.generate("klein_t2i", {}, output)
    owner.generate("klein_two_image", {}, output)
    owner.generate("klein_t2i", {}, output)
    assert len(processes) == 1
    assert owner.snapshot()["reuse_count"] == 2
    assert owner.snapshot()["switch_count"] == 0

    owner.generate("t2v", {}, output)
    owner.generate("i2v", {}, output)
    owner.generate("klein_two_image", {}, output)
    assert [process.name for process in processes] == [
        "klein:klein_t2i",
        "ltx:t2v",
        "ltx:i2v",
        "klein:klein_two_image",
    ]
    assert owner.snapshot()["switch_count"] == 3
    assert all(not process.alive for process in processes[:-1])
    assert processes[-1].alive
    owner.release()
    assert all(not process.alive for process in processes)
    assert owner.snapshot()["family"] is None
