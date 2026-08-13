from __future__ import annotations

from pathlib import Path

import pytest

from latentslate_engine.runtime import wan22_native_worker as worker


def test_worker_rejects_bypassed_fixed_recipe_operation_values() -> None:
    canonical = {
        "steps": 20,
        "stage_policy": "comfy_split",
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
