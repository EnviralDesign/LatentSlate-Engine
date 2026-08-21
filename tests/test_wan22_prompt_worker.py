from __future__ import annotations

from pathlib import Path

import pytest

from latentslate_engine.runtime.framework.worker import (
    DisposableChildContext,
    DisposableChildPaths,
    sha256_fingerprint,
)
from latentslate_engine.runtime.wan22_prompt_worker import _WanPromptHandler


def _context(tmp_path: Path) -> DisposableChildContext:
    return DisposableChildContext(
        paths=DisposableChildPaths(
            request=tmp_path / "request.json",
            result=tmp_path / "result.json",
            progress=tmp_path / "progress.jsonl",
            start_gate=tmp_path / "start-gate",
        ),
        maximum_progress_bytes=1024 * 1024,
        stage_for_progress=lambda _message: "encode_prompt",
        protocol_error=lambda reason: ValueError(reason),
    )


def _payload(tmp_path: Path) -> dict[str, object]:
    model = tmp_path / "model"
    model.mkdir()
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "model_path": str(model.resolve()),
        "prompt": "safe prompt",
        "negative_prompt": "safe negative",
        "max_sequence_length": 512,
        "output_path": str((tmp_path / "conditioning.safetensors").resolve()),
    }
    return {**unsigned, "request_binding": sha256_fingerprint(unsigned)}


def test_prompt_worker_binds_every_request_field(tmp_path: Path):
    payload = _payload(tmp_path)

    bound = _WanPromptHandler().bind_request(payload, _context(tmp_path))

    assert bound.binding == payload["request_binding"]
    assert bound.prompt == "safe prompt"
    assert bound.output_path == tmp_path / "conditioning.safetensors"


def test_prompt_worker_rejects_tampering_after_binding(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["prompt"] = "tampered"

    with pytest.raises(ValueError, match="binding"):
        _WanPromptHandler().bind_request(payload, _context(tmp_path))


def test_prompt_worker_failure_schema_never_exposes_exception_text(tmp_path: Path):
    context = _context(tmp_path)
    context.binding = "bound"
    context.stage = "encode_prompt"

    result = _WanPromptHandler().failure_result(
        RuntimeError("private model path and traceback detail"), context
    )

    assert result == {
        "schema_version": 1,
        "ok": False,
        "request_binding": "bound",
        "error_type": "RuntimeError",
        "failure_stage": "encode_prompt",
    }
