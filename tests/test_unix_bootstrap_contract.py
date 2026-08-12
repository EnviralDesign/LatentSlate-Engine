from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_unix_bootstrap_uses_shared_selection_policy_and_locked_tiers():
    script = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")

    assert "runtime_bootstrap.py" in script
    assert "select --mode" in script
    assert "sync --locked --extra" in script
    assert "--group runtime" in script
    assert "torch_cuda_unavailable" in script
    assert "torch_cuda_mismatch" in script
    assert "kitchen_cuda_backend_unavailable" in script
    assert "network_error" not in script
    assert "runtime-selection.json" in script
    assert 'if "$python_path" -m latentslate_engine doctor; then' in script
    assert '"$selected_tier" != protocol' in script


def test_unix_engine_wrapper_reuses_recorded_locked_tier():
    script = (ROOT / "scripts" / "engine.sh").read_text(encoding="utf-8")

    assert "runtime-selection.json" in script
    assert "uv_args=(run --locked --extra" in script
    assert "--group runtime" in script
    assert "nvidia-cu130|nvidia-cu128|protocol" in script


def test_windows_bootstrap_writes_bomless_state_and_handles_doctor_exit_by_tier():
    script = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")

    assert "UTF8Encoding]::new($false)" in script
    assert "$Selection.selected_tier -ne \"protocol\"" in script
    assert "$Selection.selected_tier -eq \"protocol\"" in script
