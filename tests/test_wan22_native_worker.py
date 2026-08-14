from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.runtime import wan22_native_worker as worker


def test_worker_rejects_bypassed_fixed_recipe_operation_values() -> None:
    canonical = {
        "steps": 20,
        "stage_policy": "expert_split",
        "high_guidance": 3.5,
        "low_guidance": 3.5,
    }
    worker._validate_fixed_operation(canonical)
    for key, changed in (
        ("steps", 19),
        ("stage_policy", "diffusers_boundary"),
        ("high_guidance", 4.0),
        ("low_guidance", 4.0),
    ):
        tampered = dict(canonical)
        tampered[key] = changed
        with pytest.raises(ValueError, match=key):
            worker._validate_fixed_operation(tampered)


def test_worker_keeps_lightx_as_a_separate_pinned_four_step_operation() -> None:
    lightx = {
        "steps": 4,
        "stage_policy": "expert_split",
        "high_guidance": 1.0,
        "low_guidance": 1.0,
    }

    worker._validate_fixed_operation(lightx, operation="wan22_i2v_lightx2v_4step")
    with pytest.raises(ValueError, match="steps"):
        worker._validate_fixed_operation(lightx)


def test_worker_keeps_t2v_lightx_as_a_separate_pinned_four_step_operation() -> None:
    lightx = {
        "steps": 4,
        "stage_policy": "expert_split",
        "high_guidance": 1.0,
        "low_guidance": 1.0,
    }

    worker._validate_fixed_operation(lightx, operation="wan22_t2v_lightx2v_4step")
    with pytest.raises(ValueError, match="low_guidance"):
        worker._validate_fixed_operation(
            {**lightx, "low_guidance": 3.5},
            operation="wan22_t2v_lightx2v_4step",
        )


def test_worker_keeps_flf_as_a_distinct_shift8_cfg4_operation() -> None:
    flf = {
        "steps": 20,
        "stage_policy": "expert_split",
        "high_guidance": 4.0,
        "low_guidance": 4.0,
    }
    worker._validate_fixed_operation(flf, operation="wan22_flf_base")
    with pytest.raises(ValueError, match="high_guidance"):
        worker._validate_fixed_operation(
            {**flf, "high_guidance": 3.5}, operation="wan22_flf_base"
        )


def test_worker_keeps_flf_lightx_as_a_distinct_shift5_cfg1_operation() -> None:
    lightx = {
        "steps": 4,
        "stage_policy": "expert_split",
        "high_guidance": 1.0,
        "low_guidance": 1.0,
    }
    worker._validate_fixed_operation(lightx, operation="wan22_flf_lightx2v_4step")
    with pytest.raises(ValueError, match="low_guidance"):
        worker._validate_fixed_operation(
            {**lightx, "low_guidance": 4.0},
            operation="wan22_flf_lightx2v_4step",
        )


def test_supervisor_owned_encoder_cleanup_leaves_other_targets_untouched(tmp_path: Path) -> None:
    import latentslate_engine.runtime.wan22_native_managed as managed

    target = tmp_path / "output.mp4"
    owned = tmp_path / ".output.mp4.random.tmp.mp4"
    other = tmp_path / ".other.mp4.random.tmp.mp4"
    owned.write_bytes(b"partial")
    other.write_bytes(b"other")

    managed._cleanup_owned_encoder_temps(target)

    assert not owned.exists()
    assert other.exists()


def test_supervisor_encoder_cleanup_never_replaces_a_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentslate_engine.runtime.wan22_native_managed as managed

    primary = RuntimeError("generation failed")
    monkeypatch.setattr(
        Path, "glob", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("glob"))
    )

    managed._cleanup_owned_encoder_temps(tmp_path / "output.mp4", primary=primary)

    assert "staging cleanup also failed: glob" in "\n".join(primary.__notes__)


def test_persistent_command_requires_exact_session_device_and_endpoint_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image-bytes")
    stat = source.stat()
    import hashlib

    endpoint = {
        "path": str(source.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    recipe = {"canonical": "recipe"}
    secret = b"s" * 32
    command = {
        "schema_version": 1, "recipe": recipe, "operation": "wan22_i2v_base", "device": "cuda",
        "session_binding": worker._session_binding(recipe, "wan22_i2v_base", "cuda", secret),
        "source_endpoint": endpoint, "end_endpoint": None, "output_path": str(tmp_path / "out.mp4"),
        "fps": 16,
        "generation": {
            "prompt": "move", "negative_prompt": "", "num_frames": 5, "height": 64,
            "width": 64, "steps": 20, "seed": 1, "stage_policy": "expert_split",
            "high_guidance": 3.5, "low_guidance": 3.5,
        },
    }
    command["request_binding"] = worker._command_binding(command, secret)
    generation, binding, actual_source, actual_end, output = worker._validate_persistent_payload(command, secret)
    assert generation["prompt"] == "move"
    assert binding == command["request_binding"]
    assert actual_source == source.resolve() and actual_end is None and output.name == "out.mp4"

    forged_device = {**command, "device": "cpu"}
    forged_device["request_binding"] = worker._command_binding(forged_device, secret)
    with pytest.raises(ValueError, match="session binding"):
        worker._validate_persistent_payload(forged_device, secret)

    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after dispatch"):
        worker._validate_persistent_payload(command, secret)


def test_persistent_worker_loads_once_for_two_commands_and_releases_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child loop itself, not only its parent mock, proves warm reuse."""

    import hashlib

    import latentslate_engine.runtime.video_output as output_module
    import latentslate_engine.wan22_recipe as recipe_module

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    stat = source.stat()
    endpoint = {"path": str(source.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    secret = b"x" * 32

    def command(output: Path, seed: int) -> dict[str, object]:
        recipe = {"fixture": "exact"}
        value: dict[str, object] = {
            "schema_version": 1, "recipe": recipe, "operation": "wan22_i2v_base", "device": "cuda",
            "session_binding": worker._session_binding(recipe, "wan22_i2v_base", "cuda", secret),
            "source_endpoint": endpoint, "end_endpoint": None, "output_path": str(output), "fps": 16,
            "generation": {"prompt": "move", "negative_prompt": "", "num_frames": 5, "height": 64, "width": 64, "steps": 20, "seed": seed, "stage_policy": "expert_split", "high_guidance": 3.5, "low_guidance": 3.5},
        }
        value["request_binding"] = worker._command_binding(value, secret)
        return value

    first, second = command(tmp_path / "first.mp4", 1), command(tmp_path / "second.mp4", 2)
    events: list[str] = []

    class _Runtime:
        @classmethod
        def load(cls, *_args, **_kwargs):
            events.append("load")
            return cls()

        def generate(self, request, *, device, progress):
            events.append(f"generate:{request.seed}:{device}")
            progress(1, 1, "high")
            return SimpleNamespace(video="video", provenance=SimpleNamespace())

        def release(self):
            events.append("release")

    fake_recipe = SimpleNamespace(
        operation="wan22_i2v_base", support_plan=SimpleNamespace(root=tmp_path),
        identities={role: SimpleNamespace(path=tmp_path / f"{role}.bin") for role in ("transformer_high_noise", "transformer_low_noise", "text_encoder", "vae")},
        adapter_plans={}, configured_loras=(), active_loras=(),
    )
    for identity in fake_recipe.identities.values():
        identity.path.write_bytes(b"x")
    monkeypatch.setattr(recipe_module, "rehydrate_native_wan22_i2v_14b_runtime_request", lambda _value: fake_recipe)
    monkeypatch.setattr(worker, "_runtime_type", lambda _operation: _Runtime)
    monkeypatch.setattr(worker, "_load_rgb", lambda _path: "image")
    monkeypatch.setattr(worker, "_public_provenance", lambda _value: {"proof": "ok"})
    monkeypatch.setattr(output_module, "encode_rgb_video_tensor", lambda _video, *, fps, output_path: output_path.write_bytes(b"mp4"))
    monkeypatch.setattr(output_module, "validate_encoded_video_stream", lambda *_args, **_kwargs: {"ok": True})
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(worker, "_write_json", lambda _path, value: writes.append(dict(value)))
    commands = [second]

    def wait(_path: Path):
        if commands:
            return commands.pop()
        raise RuntimeError("stop test loop")

    monkeypatch.setattr(worker, "_wait_command", wait)
    assert worker._run_persistent_session(first, tmp_path / "result.json", tmp_path / "progress.jsonl", tmp_path / "command.json", secret.hex()) == 1
    assert events == ["load", "generate:1:cuda", "generate:2:cuda", "release"]
    assert sum(value.get("ok") is True for value in writes) == 2
    assert writes[-1]["ok"] is False
