from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

import latentslate_engine.runtime.wan5_kitchen_managed as managed_module
import latentslate_engine.runtime.wan5_kitchen_worker as worker_module
from latentslate_engine.runtime.wan5_kitchen_managed import ManagedWan5KitchenRuntime


class _Request:
    operation = "wan5_t2v"
    fingerprint = "request-fingerprint"
    component_fingerprint = "component-fingerprint"

    def to_json_dict(self):
        return {
            "schema_version": 1,
            "family": "wan22",
            "operation": self.operation,
            "base_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "execution_contract": {},
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {},
        }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        **{name: tmp_path / name for name in ("request", "result", "progress", "gate")},
        "staging": tmp_path / "staging.mp4",
    }


def _metadata(request: _Request) -> dict[str, object]:
    return {
        "family": "wan22",
        "runtime": "engine-native/wan5-kitchen",
        "operation": request.operation,
        "request_fingerprint": request.fingerprint,
        "component_fingerprint": request.component_fingerprint,
        "seed": 7,
        "width": 1280,
        "height": 704,
        "frame_count": 25,
        "fps": 24.0,
        "duration_seconds": 25 / 24,
        "sampling": {
            "steps": 30,
            "source_schedule_steps": 31,
            "discard_penultimate_sigma": True,
            "guidance_scale": 5.0,
            "sampler_runtime": "diffusers/UniPCMultistepScheduler",
            "solver_bridge": "comfy/uni_pc-vp-flow-v1",
            "terminal_vp_sigma": 0.001,
        },
        "kitchen_dispatch": {
            "proven": True,
            "fallback_calls": 0,
            "native_modules": 168,
            "expected_modules": 168,
        },
        "output": {
            "fps": 24.0,
            "frame_count": 25,
            "time_base": {"numerator": 1, "denominator": 12288},
            "duration": 12800,
            "duration_seconds": 25 / 24,
            "has_audio": False,
        },
    }


def test_worker_rejects_tamper_before_recipe_rehydration(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    generation = managed_module._generation(
        request.operation,
        prompt="scene",
        output_path=tmp_path / "output.mp4",
        width=1280,
        height=704,
        num_frames=25,
        seed=7,
        start_image_path=None,
        staging_output_path=tmp_path / "staging.mp4",
    )
    payload = managed_module._payload(request, generation)  # type: ignore[arg-type]
    tampered = copy.deepcopy(payload)
    tampered["generation"]["seed"] = 8
    monkeypatch.setattr(
        worker_module,
        "_validate_generation",
        lambda *_args: (_ for _ in ()).throw(AssertionError("parsed before binding")),
    )

    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_binding(tampered)


def test_supervisor_recomputes_published_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "request_binding": "binding",
                "output_path": str(output),
                "output_size_bytes": 3,
                "metadata": {"output_sha256": "0" * 64},
                "allocator_policy": "expandable_segments:True",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="output hash"):
        managed_module._read_success(result, output, "binding")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sampling", "solver_bridge", "diffusers/native-flow"),
        ("sampling", "terminal_vp_sigma", 0.0),
        ("output", "duration_seconds", 1.0),
        ("output", "duration", 0),
        ("output", "duration", 12_801),
        ("time_base", "numerator", 2),
        ("time_base", "denominator", 0),
        ("time_base", "denominator", 12_289),
        ("top", "duration_seconds", 1.0),
    ],
)
def test_managed_metadata_rejects_solver_or_observed_timing_tamper(
    section: str, field: str, value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    if section == "top":
        metadata[field] = value
    elif section == "time_base":
        metadata["output"]["time_base"][field] = value  # type: ignore[index]
    else:
        metadata[section][field] = value  # type: ignore[index]
    generation = {
        "seed": 7,
        "width": 1280,
        "height": 704,
        "num_frames": 25,
    }

    with pytest.raises(RuntimeError, match="bound request"):
        managed_module._validate_metadata(metadata, request, generation)  # type: ignore[arg-type]


def test_worker_failure_publishes_safe_diagnostics_without_logging_child_detail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    result = tmp_path / "result.json"
    fingerprint = "a" * 64
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "request_binding": "expected",
                "error_type": "ImportError",
                "failure_stage": "encode_mp4",
                "error_fingerprint": fingerprint,
                "failure_location": "export_utils.export_to_video",
            }
        ),
        encoding="utf-8",
    )

    failure = managed_module._worker_failure(result, 1, "expected")
    assert failure == {
        "message": (
            "Wan 5B worker failed (ImportError during encode_mp4 at "
            "export_utils.export_to_video; diagnostic aaaaaaaaaaaa)"
        ),
        "error_type": "ImportError",
        "stage": "encode_mp4",
        "location": "export_utils.export_to_video",
        "fingerprint": fingerprint,
    }
    assert "private" not in failure["message"]
    assert "imageio" not in failure["message"]
    hostile_detail = "prompt=private prompt\nM:/private/path"
    with caplog.at_level("ERROR"):
        managed_module._log_worker_failure({**failure, "log_detail": hostile_detail})
    assert "ImportError" in caplog.text
    assert "encode_mp4" in caplog.text
    assert hostile_detail not in caplog.text
    assert "private prompt" not in caplog.text
    assert "M:/private/path" not in caplog.text


def test_worker_failure_surfaces_closed_numerical_boundary(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "request_binding": "expected",
                "error_type": "_Wan5NumericalError",
                "failure_stage": "generate",
                "error_fingerprint": "b" * 64,
                "failure_location": "wan5_kitchen._require_finite_tensor",
                "numerical_boundary": "scheduler_output_latents",
                "denoise_step": 29,
                "transformer_pass": None,
            }
        ),
        encoding="utf-8",
    )

    failure = managed_module._worker_failure(result, 1, "expected")

    assert failure["boundary"] == "scheduler_output_latents"
    assert failure["step"] == 29
    assert "boundary scheduler_output_latents step 29" in failure["message"]


def test_worker_serializes_only_closed_numerical_diagnostic_fields() -> None:
    error = ValueError("hostile prompt and private path")
    error.numerical_boundary = "transformer_noise_prediction"  # type: ignore[attr-defined]
    error.denoise_step = 7  # type: ignore[attr-defined]
    error.transformer_pass = "unconditional"  # type: ignore[attr-defined]

    diagnostic = worker_module._failure_diagnostic(
        error, worker_module._FailureContext(stage="generate")
    )

    assert diagnostic["numerical_boundary"] == "transformer_noise_prediction"
    assert diagnostic["denoise_step"] == 7
    assert diagnostic["transformer_pass"] == "unconditional"
    assert "hostile" not in json.dumps(diagnostic)
    assert "private" not in json.dumps(diagnostic)


def test_worker_progress_maps_output_encoding_to_a_stable_failure_stage(tmp_path: Path) -> None:
    failure = worker_module._FailureContext()
    worker_module._append_progress(
        tmp_path / "progress.jsonl",
        {"progress": 0.93, "message": "Encoding Wan 2.2 MP4"},
        failure,
    )

    assert failure.stage == "encode_mp4"


@pytest.mark.parametrize(
    ("message", "stage"),
    [
        ("Preprocessing Wan 2.2 guide image", "guide_preprocess"),
        ("Encoding Wan 2.2 guide image", "guide_vae_encode"),
        ("Prepared Wan 2.2 guide latent", "guide_latent_ready"),
    ],
)
def test_worker_progress_distinguishes_i2v_guide_boundaries(
    tmp_path: Path, message: str, stage: str
) -> None:
    failure = worker_module._FailureContext()
    worker_module._append_progress(
        tmp_path / "progress.jsonl",
        {"progress": 0.11, "message": message},
        failure,
    )

    assert failure.stage == stage


def test_managed_success_proves_empty_worker_tree(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    runtime = ManagedWan5KitchenRuntime(request)  # type: ignore[arg-type]
    paths, events = _paths(tmp_path), []

    class _Process:
        pid = 42

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class _Tree:
        def __init__(self, process):
            self.process = process

        def wait_for_empty(self):
            events.append("empty")

        def close(self):
            events.append("close")

        def terminate(self):
            raise AssertionError("successful worker was terminated")

    output = tmp_path / "output.mp4"
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_wan5_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)
    monkeypatch.setattr(managed_module, "_wait", lambda *_args: None)

    def result(_path, _expected, binding):
        output.write_bytes(b"mp4")
        return {
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": _metadata(request),
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_success", result)
    generated = runtime.generate(
        prompt="scene",
        output_path=output,
        width=1280,
        height=704,
        num_frames=25,
        seed=7,
        start_image_path=None,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )

    assert generated.output_path == output.resolve()
    assert events == ["empty", "close"]
    assert runtime.status()["last_worker"]["outcome"] == "succeeded"
    assert runtime.status()["last_worker"]["tree_empty"] is True
    assert all(not path.exists() for path in paths.values())


def test_cancellation_terminates_worker_and_removes_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = ManagedWan5KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths, events, output = _paths(tmp_path), [], tmp_path / "output.mp4"

    class _Process:
        pid = 99

        def poll(self):
            return None

    class _Tree:
        def __init__(self, process):
            self.process = process

        def terminate(self):
            events.append("terminate")

        def wait_for_empty(self):
            events.append("empty")

        def close(self):
            events.append("close")

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_wan5_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(managed_module, "DisposableProcessTree", _Tree)

    def cancel(*_args):
        output.write_bytes(b"partial")
        paths["staging"].write_bytes(b"partial staging")
        raise asyncio.CancelledError

    monkeypatch.setattr(managed_module, "_wait", cancel)
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=1280,
            height=704,
            num_frames=25,
            seed=7,
            start_image_path=None,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert events == ["terminate", "empty", "close"]
    assert not output.exists()
    assert not paths["staging"].exists()
    assert runtime.status()["last_worker"]["outcome"] == "canceled"
