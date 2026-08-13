from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import latentslate_engine.runtime.wan22_native_managed as managed_module
from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.runtime.wan22_native_managed import ManagedNativeWanI2VRuntime


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="native request validation requires the locked runtime group",
)
def test_native_wan_cancellation_is_checked_before_worker_start(tmp_path: Path):
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest

    managed = ManagedNativeWanI2VRuntime(SimpleNamespace(fingerprint="recipe:test"))  # type: ignore[arg-type]
    request = WanI2VRequest(
        image=None,
        prompt="move",
        num_frames=5,
        height=64,
        width=64,
        steps=4,
    )

    with pytest.raises(asyncio.CancelledError):
        managed.generate(
            request,
            source_image_path=tmp_path / "missing.png",
            output_path=tmp_path / "output.mp4",
            device="cpu",
            fps=16,
            cancelled=lambda: True,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="native request validation requires the locked runtime group",
)
def test_invalid_request_is_rejected_before_worker_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest

    managed = ManagedNativeWanI2VRuntime(SimpleNamespace(fingerprint="recipe:stale"))  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr(
        managed_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )
    with pytest.raises(ValueError, match=r"4k\+1"):
        managed.generate(
            WanI2VRequest(
                image=None,
                prompt="move",
                num_frames=6,
                height=64,
                width=64,
                steps=4,
            ),
            source_image_path=tmp_path / "missing.png",
            output_path=tmp_path / "output.mp4",
            device="cpu",
            fps=16,
        )


def test_supervisor_accepts_only_a_clean_exited_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    recipe = SimpleNamespace(
        fingerprint="recipe:test",
        to_json_dict=lambda: {"recipe": "canonical"},
        public_component_manifest=dict,
    )
    managed = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_i2v_runtime.validate_wan_i2v_request",
        lambda _request: None,
    )

    class _Process:
        pid = 123

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    closed = []

    class _Tree:
        def __init__(self, process):
            self.process = process

        def close(self):
            closed.append(True)

        def terminate(self):
            raise AssertionError("clean worker must not be terminated")

        def wait_for_empty(self, timeout=15.0):
            assert timeout == 15.0

    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(managed_module, "_wait_for_worker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        managed_module,
        "_validate_worker_provenance_against_request",
        lambda *_args, **_kwargs: None,
    )
    paths = {
        "request": tmp_path / "request.json",
        "result": tmp_path / "result.json",
        "progress": tmp_path / "progress.jsonl",
        "gate": tmp_path / "gate",
    }
    monkeypatch.setattr(managed_module, "_worker_paths", lambda _output: paths)

    def write_result(_path, *, expected_output):
        expected_output.write_bytes(b"mp4")
        return {
            "output_size_bytes": 3,
            "provenance": {"sampler": "euler", "shift": 5.0},
        }

    monkeypatch.setattr(managed_module, "_read_success_result", write_result)
    result = managed.generate(
        SimpleNamespace(
            prompt="move",
            negative_prompt="",
            num_frames=5,
            height=64,
            width=64,
            steps=4,
            seed=1,
            stage_policy="comfy_split",
            high_guidance=3.5,
            low_guidance=3.5,
        ),
        source_image_path=source,
        output_path=output,
        device="cpu",
        fps=16,
    )

    assert result.worker_pid == 123
    assert result.worker_exit_code == 0
    assert result.output_size_bytes == 3
    assert closed == [True]
    assert managed.status()["loaded"] is False
    assert managed.status()["last_worker"]["memory_boundary"] == "disposable_process_exit"
    assert all(not path.exists() for path in paths.values())


def test_worker_result_rejects_unexpected_output_path(tmp_path: Path):
    expected = tmp_path / "expected.mp4"
    actual = tmp_path / "actual.mp4"
    expected.write_bytes(b"a")
    actual.write_bytes(b"b")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "output_path": str(actual),
                "output_size_bytes": 1,
                "provenance": {},
            }
        )
    )
    with pytest.raises(RuntimeError, match="unexpected output"):
        managed_module._read_success_result(result, expected_output=expected)


def test_worker_provenance_rejects_an_unstructured_mapping():
    with pytest.raises(RuntimeError, match="provenance schema"):
        managed_module._validate_worker_provenance({"sampler": "euler"})


def test_worker_provenance_is_bound_to_fixed_semantics_and_expected_artifacts(tmp_path: Path):
    roles = {
        "transformer_high_noise": "transformer_high",
        "transformer_low_noise": "transformer_low",
        "text_encoder": "text_encoder",
        "vae": "vae",
    }
    identities = {
        role: ArtifactIdentity(tmp_path / f"{prefix}.safetensors", 100 + index, 200 + index, prefix)
        for index, (role, prefix) in enumerate(roles.items())
    }
    request = SimpleNamespace(
        support_plan=SimpleNamespace(fingerprint="support", tokenizer_sha256="tokenizer"),
        identities=identities,
        components={
            role: {"quantization_contract": f"contract:{prefix}"} for role, prefix in roles.items()
        },
        operation="comfy_i2v_base",
        configured_loras=(),
        active_loras=(),
    )
    provenance = {
        "support_fingerprint": "support",
        "tokenizer_sha256": "tokenizer",
        "transformer_high_header_sha256": "transformer_high",
        "transformer_low_header_sha256": "transformer_low",
        "text_encoder_header_sha256": "text_encoder",
        "vae_header_sha256": "vae",
        "transformer_high_contract": "contract:transformer_high",
        "transformer_low_contract": "contract:transformer_low",
        "text_encoder_contract": "contract:text_encoder",
        "stage_policy": "comfy_split",
        "steps": 20,
        "seed": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "shift": 5.0,
        "transformer_high_size_bytes": 100,
        "transformer_low_size_bytes": 101,
        "text_encoder_size_bytes": 102,
        "vae_size_bytes": 103,
        "transformer_high_mtime_ns": 200,
        "transformer_low_mtime_ns": 201,
        "text_encoder_mtime_ns": 202,
        "vae_mtime_ns": 203,
        "configured_loras": [],
        "active_loras": [],
    }
    managed_module._validate_worker_provenance_against_request(provenance, request, expected_seed=1)
    for key, changed in (
        ("steps", 4),
        ("sampler", "heun"),
        ("transformer_high_header_sha256", "forged"),
        ("transformer_high_size_bytes", 0),
        ("seed", 999),
        ("configured_loras", [{"forged": True}]),
        ("active_loras", [{"forged": True}]),
    ):
        forged = dict(provenance)
        forged[key] = changed
        with pytest.raises(RuntimeError):
            managed_module._validate_worker_provenance_against_request(
                forged, request, expected_seed=1
            )


def test_supervisor_rejects_preexisting_private_ipc_paths(tmp_path: Path):
    paths = {
        "request": tmp_path / "request.json",
        "result": tmp_path / "result.json",
        "progress": tmp_path / "progress.jsonl",
        "gate": tmp_path / "gate",
    }
    paths["progress"].write_text("stale")
    with pytest.raises(RuntimeError, match="already exist: progress"):
        managed_module._require_fresh_ipc_paths(paths)


def test_cancellation_refuses_to_report_terminal_before_worker_exit():
    class _Process:
        def __init__(self):
            self.terminations = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise managed_module.subprocess.TimeoutExpired("worker", timeout)

        def terminate(self):
            self.terminations += 1

        def kill(self):
            self.terminations += 1

    process = _Process()
    with pytest.raises(RuntimeError, match="did not terminate"):
        managed_module._terminate_worker(None, process)
    assert process.terminations == 2


def test_termination_kills_a_live_descendant_after_the_worker_root_exits():
    class _Process:
        def poll(self):
            return 0

    class _Tree:
        process = _Process()

        def __init__(self):
            self.terminated = 0
            self.empty_waits = 0

        def active_processes(self):
            return 1

        def terminate(self):
            self.terminated += 1

        def wait_for_empty(self, timeout=15.0):
            assert timeout == 15.0
            self.empty_waits += 1

    tree = _Tree()
    managed_module._terminate_worker(tree, tree.process)
    assert tree.terminated == 1
    assert tree.empty_waits == 1


def test_unload_proves_tree_empty_before_closing_it():
    managed = ManagedNativeWanI2VRuntime(
        SimpleNamespace(fingerprint="recipe:test", public_component_manifest=dict)
    )  # type: ignore[arg-type]

    class _Process:
        def poll(self):
            return 0

    events = []

    class _Tree:
        process = _Process()

        def active_processes(self):
            return 0

        def wait_for_empty(self, timeout=15.0):
            events.append("empty")

        def terminate(self):
            events.append("terminate")

        def close(self):
            events.append("close")

    managed._active_tree = _Tree()  # type: ignore[assignment]
    managed.unload()
    assert events == ["empty", "close"]
    assert managed.status()["active_worker"] is False


def test_unload_keeps_tree_empty_failure_primary_over_close_failure():
    managed = ManagedNativeWanI2VRuntime(
        SimpleNamespace(fingerprint="recipe:test", public_component_manifest=dict)
    )  # type: ignore[arg-type]

    class _Tree:
        process = SimpleNamespace()

        def active_processes(self):
            raise RuntimeError("tree accounting failed")

        def close(self):
            raise OSError("close failed")

    managed._active_tree = _Tree()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="tree accounting failed") as caught:
        managed.unload()
    assert "close failed" in "\n".join(caught.value.__notes__)
    assert managed._active_tree is None
