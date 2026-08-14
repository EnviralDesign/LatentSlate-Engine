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


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="native request validation requires the locked runtime group",
)
def test_flf_requires_two_distinct_paths_before_worker_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from latentslate_engine.runtime.wan22_flf_runtime import WanFLFRequest

    path = tmp_path / "endpoint.png"
    path.write_bytes(b"endpoint")
    managed = ManagedNativeWanI2VRuntime(
        SimpleNamespace(fingerprint="recipe:flf", operation="wan22_flf_base")
    )  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr(
        managed_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )
    with pytest.raises(ValueError, match="distinct paths"):
        managed.generate(
            WanFLFRequest(start_image=None, end_image=None, prompt="move"),
            source_image_path=path,
            end_image_path=path,
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
        operation="wan22_i2v_base",
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
            return None

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
    monkeypatch.setattr(managed_module, "_wait_for_persistent_result", lambda *_args, **_kwargs: None)
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
        "command": tmp_path / "command.json",
    }
    monkeypatch.setattr(managed_module, "_worker_paths", lambda _output: paths)

    def write_result(_path, *, expected_output, expected_binding):
        expected_output.write_bytes(b"mp4")
        return {
            "output_size_bytes": 3,
            "stream_metadata": {
                "width": 64, "height": 64, "frame_count": 5, "fps": 16,
                "duration_seconds": 5 / 16, "has_audio": False,
                "codec_name": "h264", "pixel_format": "yuv420p",
            },
            "provenance": {"sampler": "euler", "shift": 5.0},
        }

    monkeypatch.setattr(managed_module, "_read_persistent_result", write_result)
    result = managed.generate(
        SimpleNamespace(
            prompt="move",
            negative_prompt="",
            num_frames=5,
            height=64,
            width=64,
            steps=4,
            seed=1,
            stage_policy="expert_split",
            high_guidance=3.5,
            low_guidance=3.5,
        ),
        source_image_path=source,
        output_path=output,
        device="cpu",
        fps=16,
    )

    assert result.worker_pid == 123
    assert result.worker_exit_code is None
    assert result.output_size_bytes == 3
    assert closed == []
    assert managed.status()["loaded"] is True
    assert managed.status()["last_worker"]["memory_boundary"] == "persistent_exact_recipe_worker"
    assert not paths["result"].exists()
    assert not paths["progress"].exists()
    assert not paths["command"].exists()


def test_persistent_session_reuses_one_pid_and_keeps_models_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two compatible jobs must cause one materialization/session spawn."""

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    recipe = SimpleNamespace(
        fingerprint="recipe:warm", operation="wan22_i2v_base",
        to_json_dict=lambda: {"recipe": "canonical"}, public_component_manifest=dict,
    )
    runtime = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_i2v_runtime.validate_wan_i2v_request",
        lambda _request: None,
    )
    paths = {
        "request": tmp_path / "request.json", "result": tmp_path / "result.json",
        "progress": tmp_path / "progress.jsonl", "gate": tmp_path / "gate",
        "command": tmp_path / "command.json",
    }
    spawns: list[object] = []

    class _Process:
        pid = 4242
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def wait(self, timeout=None):
            self.stopped = True
            return 0

    class _Tree:
        def __init__(self, process):
            self.process = process

        def active_processes(self):
            return 0 if self.process.stopped else 1

        def terminate(self):
            self.process.stopped = True

        def wait_for_empty(self, timeout=15.0):
            pass

        def close(self):
            pass

    monkeypatch.setattr(managed_module, "_worker_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module.subprocess, "Popen", lambda *_args, **_kwargs: (spawns.append(object()) or _Process())
    )
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(managed_module, "_wait_for_persistent_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(managed_module, "_validate_worker_provenance_against_request", lambda *_args, **_kwargs: None)
    calls = 0

    def result(_path, *, expected_output, expected_binding):
        nonlocal calls
        expected_output.write_bytes(b"mp4")
        calls += 1
        return {
            "output_size_bytes": 3,
            "stream_metadata": {
                "width": 64, "height": 64, "frame_count": 5, "fps": 16,
                "duration_seconds": 5 / 16, "has_audio": False,
                "codec_name": "h264", "pixel_format": "yuv420p",
            },
            "provenance": {},
        }

    monkeypatch.setattr(managed_module, "_read_persistent_result", result)
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest

    warm_flags = []
    for index in range(2):
        output = tmp_path / f"output-{index}.mp4"
        produced = runtime.generate(
            WanI2VRequest(image=None, prompt="move", num_frames=5, height=64, width=64, steps=4),
            source_image_path=source, output_path=output, device="cuda", fps=16,
        )
        assert produced.worker_pid == 4242
        assert produced.worker_exit_code is None
        warm_flags.append(produced.pipeline_warm)
    assert len(spawns) == 1
    assert calls == 2
    assert warm_flags == [False, True]
    assert runtime.status()["loaded"] is True
    assert runtime.status()["active_worker"] is False
    assert runtime.status()["cache"] == {"pipeline_warm": True}
    runtime.clear_cache()
    assert runtime.status()["loaded"] is True
    runtime.unload()
    assert runtime.status()["loaded"] is False


def test_persistent_payload_is_private_and_rejects_preexisting_output(tmp_path: Path) -> None:
    root = managed_module._worker_paths(tmp_path / "visible.mp4")
    assert root["request"].parent != tmp_path
    assert root["request"].parent.name.startswith("latentslate-wan14-")
    assert set(root) == {"request", "result", "progress", "gate", "command"}
    managed_module._cleanup_persistent_session(root)


def test_windows_ipc_directory_requires_protected_owner_system_dacl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    class _Advapi:
        def ConvertStringSecurityDescriptorToSecurityDescriptorW(self, *args):
            calls.append(("convert", *args[:2]))
            return 1

        def SetFileSecurityW(self, *args):
            calls.append(("set", args[0], args[1]))
            return 1

    class _Kernel:
        def LocalFree(self, _descriptor):
            calls.append(("free",))

    monkeypatch.setattr(managed_module.os, "name", "nt")
    monkeypatch.setattr(
        managed_module.ctypes,
        "WinDLL",
        lambda name, **_kwargs: _Advapi() if name == "advapi32" else _Kernel(),
    )
    managed_module._secure_ipc_directory(tmp_path)
    assert calls[0][:3] == ("convert", "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)", 1)
    assert calls[1][2] == 0x00000004 | 0x80000000


def test_persistent_failure_evicts_session_and_only_removes_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    target = tmp_path / "target.mp4"
    recipe = SimpleNamespace(
        fingerprint="recipe:failure", operation="wan22_i2v_base",
        to_json_dict=lambda: {"recipe": "canonical"}, public_component_manifest=dict,
    )
    runtime = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_i2v_runtime.validate_wan_i2v_request", lambda _request: None
    )
    session_root = tmp_path / "session"
    session_root.mkdir()
    paths = {key: session_root / name for key, name in {
        "request": "request.json", "result": "result.json", "progress": "progress.jsonl",
        "gate": "gate", "command": "command.json",
    }.items()}

    class _Process:
        pid = 55
        stopped = False

        def poll(self):
            return 1 if self.stopped else None

        def wait(self, timeout=None):
            self.stopped = True
            return 1

    class _Tree:
        def __init__(self, process):
            self.process = process

        def active_processes(self):
            return 1 if not self.process.stopped else 0

        def terminate(self):
            self.process.stopped = True

        def wait_for_empty(self, timeout=15.0):
            assert self.process.stopped

        def close(self):
            pass

    monkeypatch.setattr(managed_module, "_worker_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(
        managed_module, "_wait_for_persistent_result", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed"))
    )
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest

    with pytest.raises(RuntimeError, match="worker failed"):
        runtime.generate(
            WanI2VRequest(image=None, prompt="move", num_frames=5, height=64, width=64, steps=4),
            source_image_path=source, output_path=target, device="cuda", fps=16,
        )
    assert runtime.status()["loaded"] is False
    assert runtime.status()["last_worker"]["terminated"] is True
    assert not target.exists()
    assert not session_root.exists()


def test_dead_idle_session_is_not_reported_warm_and_next_use_would_respawn(tmp_path: Path) -> None:
    runtime = ManagedNativeWanI2VRuntime(
        SimpleNamespace(fingerprint="recipe:dead", public_component_manifest=dict)
    )  # type: ignore[arg-type]
    root = tmp_path / "session"
    root.mkdir()
    paths = {key: root / name for key, name in {
        "request": "request.json", "result": "result.json", "progress": "progress.jsonl",
        "gate": "gate", "command": "command.json",
    }.items()}

    class _Process:
        pid = 99

        def poll(self):
            return 1

    class _Tree:
        process = _Process()

        def active_processes(self):
            return 0

        def terminate(self):
            raise AssertionError("dead idle worker should not be terminated twice")

        def wait_for_empty(self, timeout=15.0):
            pass

        def close(self):
            pass

    runtime._session = managed_module._WanWorkerSession(  # type: ignore[assignment]
        _Tree.process, _Tree(), paths, "cuda", "binding", b"x" * 32, successful_jobs=1
    )
    runtime._active_tree = runtime._session.tree
    status = runtime.status()
    assert status["loaded"] is False
    assert status["cache"] == {"pipeline_warm": False}
    assert not root.exists()


def test_poison_preserves_primary_error_when_terminate_and_close_fail(tmp_path: Path) -> None:
    runtime = ManagedNativeWanI2VRuntime(
        SimpleNamespace(fingerprint="recipe:cleanup", public_component_manifest=dict)
    )  # type: ignore[arg-type]
    root = tmp_path / "session"
    root.mkdir()
    paths = {key: root / name for key, name in {
        "request": "request.json", "result": "result.json", "progress": "progress.jsonl",
        "gate": "gate", "command": "command.json",
    }.items()}

    class _Process:
        pid = 17

        def poll(self):
            return None

    class _Tree:
        process = _Process()

        def active_processes(self):
            raise RuntimeError("terminate accounting failed")

        def close(self):
            raise OSError("close failed")

    session = managed_module._WanWorkerSession(_Tree.process, _Tree(), paths, "cuda", "bind", b"x" * 32)
    primary = RuntimeError("generation primary")
    runtime._session = session
    runtime._active_tree = session.tree
    runtime._poison_session(session, primary)
    assert runtime.status()["loaded"] is False
    notes = "\n".join(primary.__notes__)
    assert "terminate accounting failed" in notes and "close failed" in notes


def test_late_cancellation_after_worker_result_removes_owned_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    target = tmp_path / "target.mp4"
    recipe = SimpleNamespace(
        fingerprint="recipe:late-cancel", operation="wan22_i2v_base",
        to_json_dict=lambda: {"recipe": "canonical"}, public_component_manifest=dict,
    )
    runtime = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(managed_module, "revalidate_runtime_request", lambda _request: True)
    monkeypatch.setattr("latentslate_engine.runtime.wan22_i2v_runtime.validate_wan_i2v_request", lambda _request: None)
    root = tmp_path / "session"
    root.mkdir()
    paths = {key: root / name for key, name in {
        "request": "request.json", "result": "result.json", "progress": "progress.jsonl",
        "gate": "gate", "command": "command.json",
    }.items()}

    class _Process:
        pid = 23
        stopped = False

        def poll(self):
            return 1 if self.stopped else None

        def wait(self, timeout=None):
            self.stopped = True
            return 1

    class _Tree:
        process = _Process()

        def active_processes(self):
            return 0 if self.process.stopped else 1

        def terminate(self):
            self.process.stopped = True

        def wait_for_empty(self, timeout=15.0):
            assert self.process.stopped

        def close(self):
            pass

    secret = b"x" * 32
    binding = managed_module._binding(
        {"recipe": recipe.to_json_dict(), "operation": recipe.operation, "device": "cuda"}, secret
    )
    runtime._session = managed_module._WanWorkerSession(_Tree.process, _Tree(), paths, "cuda", binding, secret)
    runtime._active_tree = runtime._session.tree
    monkeypatch.setattr(
        managed_module, "_wait_for_persistent_result", lambda *_args, **_kwargs: target.write_bytes(b"mp4")
    )
    checks = iter((False, True))
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest

    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            WanI2VRequest(image=None, prompt="move", num_frames=5, height=64, width=64, steps=4),
            source_image_path=source, output_path=target, device="cuda", fps=16,
            cancelled=lambda: next(checks),
        )
    assert not target.exists()
    assert runtime.status()["loaded"] is False


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
                "stream_metadata": {},
                "provenance": {},
            }
        )
    )
    with pytest.raises(RuntimeError, match="unexpected output"):
        managed_module._read_success_result(result, expected_output=expected)


def test_worker_provenance_rejects_an_unstructured_mapping():
    with pytest.raises(RuntimeError, match="provenance schema"):
        managed_module._validate_worker_provenance({"sampler": "euler"})


def test_worker_stream_metadata_rejects_duration_not_bound_to_frames_and_fps():
    stream = {
        "width": 64,
        "height": 64,
        "frame_count": 5,
        "fps": 16,
        "duration_seconds": 999.0,
        "has_audio": False,
        "codec_name": "h264",
        "pixel_format": "yuv420p",
    }
    with pytest.raises(RuntimeError, match="duration"):
        managed_module._validate_stream_metadata(stream)


@pytest.mark.parametrize("field", ("rejected_delta", "dense_fallback_delta"))
def test_transformer_dispatch_rejects_nonzero_rejection_or_dense_fallback(field: str):
    proof = {
        stage: {
            "fp8_module_count": 1,
            "fp8_modules": {
                "blocks.0.layer": {
                    "native_dispatch_delta": 1,
                    "rejected_delta": 0,
                    "dense_fallback_delta": 0,
                }
            },
            "int8_module_count": 0,
            "int8_modules": {},
            "dense_fallback_count": 0,
            "rejected_count": 0,
        }
        for stage in ("high", "low")
    }
    proof["high"]["fp8_modules"]["blocks.0.layer"][field] = 1
    with pytest.raises(RuntimeError, match="not clean"):
        managed_module._validate_transformer_dispatch(proof)


def test_transformer_dispatch_keeps_int8_evidence_distinct_from_fp8():
    proof = {
        stage: {
            "fp8_module_count": 0,
            "fp8_modules": {},
            "int8_module_count": 1,
            "int8_modules": {
                "blocks.0.layer": {
                    "int8_dispatch_delta": 1,
                    "rejected_delta": 0,
                    "dense_fallback_delta": 0,
                }
            },
            "dense_fallback_count": 0,
            "rejected_count": 0,
        }
        for stage in ("high", "low")
    }
    managed_module._validate_transformer_dispatch(proof)


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
        operation="wan22_i2v_base",
        configured_loras=(),
        active_loras=(),
        adapter_plans={
            role: SimpleNamespace(
                artifact_contract="contract:" + prefix,
                source_to_target={"layer.weight": "blocks.0.layer.weight"},
                quant_auxiliary=("layer.weight_scale", "layer.comfy_quant"),
            )
            for role, prefix in roles.items()
            if role.startswith("transformer")
        },
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
        "stage_policy": "expert_split",
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
        "lora_dispatch": {
            "high": {"target_module_count": 0, "dispatch_call_count": 0},
            "low": {"target_module_count": 0, "dispatch_call_count": 0},
        },
        "transformer_dispatch": {
            stage: {
                "fp8_module_count": 1,
                "fp8_modules": {
                    "blocks.0.layer": {
                        "native_dispatch_delta": 1,
                        "rejected_delta": 0,
                        "dense_fallback_delta": 0,
                    }
                },
                "int8_module_count": 0,
                "int8_modules": {},
                "dense_fallback_count": 0,
                "rejected_count": 0,
            }
            for stage in ("high", "low")
        },
    }
    managed_module._validate_worker_provenance_against_request(provenance, request, expected_seed=1)
    int8_request = SimpleNamespace(
        **{
            **request.__dict__,
            "adapter_plans": {
                **request.adapter_plans,
                "transformer_high_noise": SimpleNamespace(
                    artifact_contract="comfy_quant/int8_tensorwise_convrot",
                    source_to_target={"layer.weight": "blocks.0.layer.weight"},
                    quant_auxiliary=("layer.weight_scale", "layer.comfy_quant"),
                ),
            },
        }
    )
    int8_provenance = {
        **provenance,
        "transformer_dispatch": {
            **provenance["transformer_dispatch"],
            "high": {
                "fp8_module_count": 0,
                "fp8_modules": {},
                "int8_module_count": 1,
                "int8_modules": {
                    "blocks.0.layer": {
                        "int8_dispatch_delta": 1,
                        "rejected_delta": 0,
                        "dense_fallback_delta": 0,
                    }
                },
                "dense_fallback_count": 0,
                "rejected_count": 0,
            },
        },
    }
    managed_module._validate_worker_provenance_against_request(
        int8_provenance, int8_request, expected_seed=1
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        managed_module._validate_worker_provenance_against_request(
            provenance, int8_request, expected_seed=1
        )
    flf_request = SimpleNamespace(**{**request.__dict__, "operation": "wan22_flf_base"})
    flf_provenance = {**provenance, "shift": 8.0}
    managed_module._validate_worker_provenance_against_request(
        flf_provenance, flf_request, expected_seed=1
    )
    with pytest.raises(RuntimeError, match="shift does not match recipe"):
        managed_module._validate_worker_provenance_against_request(
            provenance, flf_request, expected_seed=1
        )
    for key, changed in (
        ("steps", 4),
        ("sampler", "heun"),
        ("transformer_high_header_sha256", "forged"),
        ("transformer_high_size_bytes", 0),
        ("seed", 999),
        ("configured_loras", [{"forged": True}]),
        ("active_loras", [{"forged": True}]),
        ("lora_dispatch", {"high": {}, "low": {}}),
        ("transformer_dispatch", {"high": {}, "low": {}}),
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
