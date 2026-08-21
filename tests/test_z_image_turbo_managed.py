"""No-GPU contract coverage for the Z-Image private worker boundary."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from PIL import Image
from torch import nn

from latentslate_engine.runtime import z_image_turbo as core_module
from latentslate_engine.runtime import z_image_turbo_managed as managed
from latentslate_engine.runtime import z_image_turbo_worker as worker
from latentslate_engine.runtime import z_image_worker_protocol as worker_protocol
from latentslate_engine.runtime.framework.worker import (
    PersistentChildContext,
    PersistentChildPaths,
    PersistentWorkerFailedStart,
    PersistentWorkerPaths,
    PersistentWorkerSession,
    PersistentWorkerSupervisor,
)
from latentslate_engine.runtime.framework.worker import persistent as persistent_module
from latentslate_engine.runtime.framework.worker import persistent_child as persistent_child_module


def _persistent_paths(tmp_path: Path) -> PersistentWorkerPaths:
    return PersistentWorkerPaths(
        request=tmp_path / "request.json",
        result=tmp_path / "result.json",
        progress=tmp_path / "progress.jsonl",
        heartbeat=tmp_path / "heartbeat.jsonl",
        start_gate=tmp_path / "start-gate",
        command=tmp_path / "command.json",
        cancel=tmp_path / "cancel-requested",
    )


def _child_context(tmp_path: Path) -> PersistentChildContext:
    paths = _persistent_paths(tmp_path)
    return PersistentChildContext(
        paths=PersistentChildPaths(
            request=paths.request,
            result=paths.result,
            progress=paths.progress,
            heartbeat=paths.heartbeat,
            start_gate=paths.start_gate,
            command=paths.command,
            cancel=paths.cancel,
        ),
        maximum_bytes=1024 * 1024,
        heartbeat_seconds=0.01,
        protocol_error=worker._ZImageHandler("").protocol_error,
    )


def _inert_supervisor(tmp_path: Path) -> PersistentWorkerSupervisor:
    paths = _persistent_paths(tmp_path)
    supervisor = PersistentWorkerSupervisor(command=("fixed-test-worker",), paths=paths)
    supervisor.session = PersistentWorkerSession(
        process=SimpleNamespace(poll=lambda: None),
        tree=SimpleNamespace(),
        paths=paths,
    )
    return supervisor


def _cuda_health_proof() -> dict[str, str]:
    return {phase: "pass" for phase in worker._CUDA_HEALTH_PHASES}


def test_cuda_health_phase_and_error_code_contract_matches_parent_and_child():
    assert worker._CUDA_HEALTH_PHASES is worker_protocol.CUDA_HEALTH_PHASES
    assert managed._CUDA_HEALTH_PHASES is worker_protocol.CUDA_HEALTH_PHASES
    assert managed._CUDA_HEALTH_STAGES is worker_protocol.CUDA_HEALTH_STAGES
    assert worker._CUDA_ERROR_CODES is worker_protocol.CUDA_ERROR_CODES
    assert managed._CUDA_ERROR_CODES is worker_protocol.CUDA_ERROR_CODES
    assert worker._FAILURE_STAGES is worker_protocol.FAILURE_STAGES
    assert managed._FAILURE_STAGES is worker_protocol.FAILURE_STAGES


def test_worker_and_qwen_preflight_share_one_cuda_health_implementation():
    from latentslate_engine.runtime import z_image_qwen_runtime as qwen_runtime

    assert worker._cuda_health is qwen_runtime._cuda_health
    assert "_cuda_health.z_image_cuda_health_check" in inspect.getsource(
        worker._z_image_cuda_health_check
    )
    assert "_cuda_health.z_image_cuda_health_check" in inspect.getsource(
        qwen_runtime._preflight_z_image_full_precision_linear
    )
    facts = worker._cuda_health.z_image_cuda_health_check(torch, "cpu")
    assert facts == {
        "source_device": "cpu",
        "target_device": "cpu",
        "dtype": "uint8",
        "numel": 16,
        "contiguous": True,
        "storage_offset": 0,
        "blocking_copy": True,
        "readback_equal": True,
    }


def test_worker_resolves_bare_cuda_once_to_current_index_without_gpu_use():
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 2,
            device_count=lambda: 4,
        ),
    )

    assert worker._resolve_worker_cuda_device(fake_torch, "cuda") == torch.device("cuda:2")
    assert worker._resolve_worker_cuda_device(fake_torch, "cuda:2") == torch.device("cuda:2")
    assert worker._resolve_worker_cuda_device(fake_torch, "cuda:1") == torch.device("cuda:1")


def test_worker_preserves_explicit_other_index_without_consulting_current_device():
    current_calls: list[None] = []
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: current_calls.append(None),
            device_count=lambda: 4,
        ),
    )

    assert worker._resolve_worker_cuda_device(fake_torch, "cuda:1") == torch.device("cuda:1")
    assert current_calls == []


def test_loaded_session_binding_and_runtime_key_use_only_the_indexed_device(tmp_path: Path):
    secret = bytes(range(32))
    payload = _payload(tmp_path, secret)
    recipe = payload["recipe"]

    loaded_binding = worker._execution_session_binding(recipe, "cuda:2", secret)
    explicit_session = {
        "recipe": recipe,
        "device": "cuda:2",
        "dtype": "bfloat16",
        "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    }

    assert loaded_binding == worker._binding(explicit_session, secret)
    assert worker._execution_runtime_key(recipe, "cuda:2")[1] == "cuda:2"
    assert "cuda" not in worker._execution_runtime_key(recipe, "cuda:2")


@pytest.mark.parametrize("requested", ["cpu", "cuda:4", "not-a-device"])
def test_worker_rejects_non_cuda_and_invalid_devices_without_health_classification(
    requested,
):
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 2,
            device_count=lambda: 4,
        ),
    )

    with pytest.raises(ValueError, match="Z-Image worker"):
        worker._resolve_worker_cuda_device(fake_torch, requested)


def test_worker_rejects_cuda_when_no_gpu_is_available_without_querying_current_device():
    current_calls: list[None] = []
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: current_calls.append(None),
            device_count=lambda: 0,
        ),
    )

    with pytest.raises(ValueError, match="unavailable"):
        worker._resolve_worker_cuda_device(fake_torch, "cuda")
    assert current_calls == []


def _payload(tmp_path: Path, secret: bytes) -> dict[str, object]:
    components = {
        role: {"path": str(tmp_path / f"{role}.bin"), "role": role}
        for role in ("pipeline_support", "transformer", "text_encoder", "vae")
    }
    fingerprint_payload = {
        "schema_version": 1,
        "base_model": "pinned",
        "operation": "zimage_turbo_t2i_int8_convrot",
        "schedule": {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guider": "basic",
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
        },
        "components": {role: dict(component) for role, component in sorted(components.items())},
    }
    recipe = {
        "schema_version": 1,
        "family": "zimage",
        "operation": "zimage_turbo_t2i_int8_convrot",
        "base_model": "pinned",
        "execution_contract": {
            "guider": "basic_positive_only",
            "noise": "cpu_fp32_manual_seed_then_transfer",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "shift": 3,
            "output": "png_rgb_1024",
        },
        "schedule": {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guider": "basic",
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
        },
        "components": components,
        "fingerprint": "z-image-turbo:sha256:"
        + hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    session = {
        "recipe": recipe,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
    }
    unsigned: dict[str, object] = {
        "schema_version": 1,
        **session,
        "session_binding": worker._binding(session, secret),
        "output_path": str(tmp_path / "result.png"),
        "generation": {"prompt": "safe prompt", "seed": 7},
    }
    return {**unsigned, "request_binding": worker._binding(unsigned, secret)}


def _payload_named(tmp_path: Path, secret: bytes, name: str) -> dict[str, object]:
    value = _payload(tmp_path, secret)
    unsigned = {key: item for key, item in value.items() if key != "request_binding"}
    unsigned["output_path"] = str(tmp_path / name)
    return {**unsigned, "request_binding": worker._binding(unsigned, secret)}


def _payload_with_fixed_lora(tmp_path: Path, secret: bytes) -> dict[str, object]:
    value = _payload(tmp_path, secret)
    recipe = json.loads(json.dumps(value["recipe"]))
    recipe["components"]["style_lora"] = {
        "path": str(tmp_path / "fixed-lora.safetensors"),
        "role": "style_lora",
    }
    fingerprint_payload = {
        key: recipe[key]
        for key in ("schema_version", "base_model", "operation", "schedule", "components")
    }
    recipe["fingerprint"] = "z-image-turbo:sha256:" + hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    session = {
        "recipe": recipe,
        "device": value["device"],
        "dtype": value["dtype"],
        "execution": value["execution"],
    }
    unsigned = {
        "schema_version": 1,
        **session,
        "session_binding": worker._binding(session, secret),
        "output_path": value["output_path"],
        "generation": value["generation"],
    }
    return {**unsigned, "request_binding": worker._binding(unsigned, secret)}


def test_private_worker_hmac_binding_rejects_tampering_before_rehydrate(tmp_path: Path):
    secret = bytes(range(32))
    payload = _payload(tmp_path, secret)
    recipe, device, generation, output, binding, session = worker._validate(payload, secret)
    assert recipe["family"] == "zimage"
    assert device == "cuda:0" and generation == {"prompt": "safe prompt", "seed": 7}
    assert output.name == "result.png" and binding == payload["request_binding"]
    assert session == payload["session_binding"]
    altered = dict(payload)
    altered["generation"] = {"prompt": "other", "seed": 7}
    with pytest.raises(ValueError, match="command binding"):
        worker._validate(altered, secret)


def test_fixed_lora_manifest_is_fingerprinted_and_hmac_bound(tmp_path: Path):
    secret = bytes(range(32))
    payload = _payload_with_fixed_lora(tmp_path, secret)

    recipe, *_rest = worker._validate(payload, secret)
    assert set(recipe["components"]) == {
        "pipeline_support",
        "transformer",
        "text_encoder",
        "vae",
        "style_lora",
    }

    tampered = json.loads(json.dumps(payload))
    tampered["recipe"]["components"]["style_lora"]["path"] += ".changed"
    session = {
        key: tampered[key] for key in ("recipe", "device", "dtype", "execution")
    }
    tampered["session_binding"] = worker._binding(session, secret)
    unsigned = {key: value for key, value in tampered.items() if key != "request_binding"}
    tampered["request_binding"] = worker._binding(unsigned, secret)
    with pytest.raises(ValueError, match="fingerprint"):
        worker._validate(tampered, secret)


def test_fixed_lora_install_failure_is_closed_and_parent_accepted():
    secret = bytes(range(32))
    failure = worker._FailureContext(
        "lora_install", "z_image_turbo_worker._load_core", "binding"
    )
    result = worker._failure_result(RuntimeError("private adapter path"), failure, "", secret)

    accepted = managed._validate_failure_result(result, "binding")

    assert accepted is not None
    assert accepted["failure_stage"] == "lora_install"
    assert "private" not in json.dumps(result)


def test_worker_rejects_existing_output_before_heavy_imports(tmp_path: Path):
    secret = bytes(range(32))
    payload = _payload(tmp_path, secret)
    Path(str(payload["output_path"])).touch()
    with pytest.raises(ValueError, match="fresh PNG"):
        worker._validate(payload, secret)


def test_worker_failure_envelope_is_deterministic_and_never_serializes_hostile_text(tmp_path: Path):
    """The disposable IPC error is useful without becoming a prompt/path channel."""

    secret = bytes(range(32))
    payload = _payload(tmp_path, secret)
    hostile = "C:\\private\\secret-token.txt\nPROMPT=do not disclose"
    failure = worker._FailureContext(
        "qwen_materialize", "z_image_turbo_worker._load_core", payload["request_binding"]
    )
    first = worker._failure_result(RuntimeError(hostile), failure, "")
    second = worker._failure_result(RuntimeError(hostile), failure, "")
    assert first == second
    assert set(first) == {
        "schema_version",
        "ok",
        "request_binding",
        "result_binding",
        "error_type",
        "failure_stage",
        "failure_location",
        "error_fingerprint",
    }
    encoded = json.dumps(first)
    assert hostile not in encoded
    assert "private" not in encoded and "PROMPT" not in encoded and "secret-token" not in encoded
    assert first["failure_stage"] == "qwen_materialize"
    assert first["failure_location"] == "z_image_turbo_worker._load_core"

    hostile_type = type("SecretPromptType", (RuntimeError,), {})
    class_name_result = worker._failure_result(hostile_type(hostile), failure, "")
    assert class_name_result["error_type"] == "Exception"
    assert "SecretPromptType" not in json.dumps(class_name_result)


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("CUDA out of memory. private-path", "cuda_oom"),
        ("an illegal memory access was encountered private-path", "illegal_memory_access"),
        ("CUDA error: invalid argument private-path", "invalid_argument"),
        ("operation not supported private-path", "operation_not_supported"),
        ("CUDA driver initialization failed private-path", "driver_error"),
        ("unclassified private-path", "unknown_runtime"),
    ),
)
def test_synthetic_cuda_error_classifier_is_closed(message, expected):
    assert worker._classify_synthetic_cuda_error(RuntimeError(message)) == expected


def test_synthetic_cuda_classifier_recognizes_torch_oom_by_exact_type():
    assert (
        worker._classify_synthetic_cuda_error(torch.OutOfMemoryError("private"))
        == "cuda_oom"
    )


def test_synthetic_cuda_classifier_never_stringifies_hostile_subclasses():
    class HostileRuntime(RuntimeError):
        def __str__(self):
            raise AssertionError("must not stringify hostile exception")

    assert worker._classify_synthetic_cuda_error(HostileRuntime("secret")) == "unknown_runtime"


def test_device_contract_value_error_remains_a_plain_authenticated_failure():
    secret = bytes(range(32))
    failure = worker._FailureContext(
        "device_contract", "z_image_turbo_worker._resolve_worker_cuda_device", "binding"
    )

    result = worker._failure_result(ValueError("private device detail"), failure, "", secret)

    assert result["error_type"] == "ValueError"
    assert result["failure_stage"] == "device_contract"
    assert result["failure_location"] == "z_image_turbo_worker._resolve_worker_cuda_device"
    assert "cuda_error_code" not in result
    assert managed._validate_failure_result(result, "binding") is not None


def test_synthetic_cuda_failure_code_is_hmac_bound_and_never_leaks_text():
    secret = bytes(range(32))
    hostile = "illegal memory access C:\\private\\prompt.txt PROMPT=secret"
    code = worker._classify_synthetic_cuda_error(RuntimeError(hostile))
    failure = worker._FailureContext(
        "cuda_health_post_qwen", "z_image_turbo_worker._load_core", "binding"
    )
    safe = worker._CudaHealthFailure(
        "post_qwen", "copy", code, "RuntimeError", ("pre_import", "post_tokenizer")
    )
    result = worker._failure_result(safe, failure, "", secret)
    assert result["cuda_error_code"] == "illegal_memory_access"
    assert set(result) == {
        "schema_version",
        "ok",
        "request_binding",
        "error_type",
        "failure_stage",
        "failure_location",
        "error_fingerprint",
        "cuda_error_code",
        "cuda_health_completed",
        "cuda_health_phase",
        "cuda_health_substage",
        "result_binding",
    }
    assert hostile not in json.dumps(result)
    assert "prompt.txt" not in json.dumps(result) and "PROMPT" not in json.dumps(result)
    assert managed._validate_failure_result(result, "binding")["cuda_error_code"] == (
        "illegal_memory_access"
    )
    assert result["cuda_health_completed"] == ["pre_import", "post_tokenizer"]
    assert result["cuda_health_phase"] == "post_qwen"
    assert result["cuda_health_substage"] == "copy"
    tampered = dict(result)
    tampered["cuda_error_code"] = "driver_error"
    unsigned = {key: value for key, value in tampered.items() if key != "result_binding"}
    assert not hmac.compare_digest(
        tampered["result_binding"], managed._result_binding(unsigned, secret)
    )
    prefix_tampered = dict(result)
    prefix_tampered["cuda_health_completed"] = ["pre_import"]
    prefix_unsigned = {
        key: value for key, value in prefix_tampered.items() if key != "result_binding"
    }
    prefix_tampered["result_binding"] = managed._result_binding(prefix_unsigned, secret)
    assert managed._validate_failure_result(prefix_tampered, "binding") is None


@pytest.mark.parametrize("phase", worker._CUDA_HEALTH_PHASES[:-1])
def test_cold_load_core_attributes_each_cuda_health_phase_exactly(
    tmp_path: Path, monkeypatch, phase
):
    from latentslate_engine.runtime import z_image_conditioning as conditioning
    from latentslate_engine.runtime import z_image_nextdit as nextdit
    from latentslate_engine.runtime import z_image_qwen_runtime as qwen_runtime
    from latentslate_engine.runtime import z_image_vae as vae

    monkeypatch.setattr(conditioning, "build_z_image_qwen_tokenizer", lambda *_args: object())
    monkeypatch.setattr(qwen_runtime, "build_z_image_mixed_qwen_shell", lambda *_args: object())
    monkeypatch.setattr(
        qwen_runtime,
        "materialize_z_image_mixed_qwen",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        nextdit, "materialize_z_image_nextdit", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        vae, "materialize_z_image_flux_ae", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        core_module,
        "ZImageTurboCore",
        lambda *_args, **_kwargs: SimpleNamespace(execution_device=torch.device("cpu")),
    )
    monkeypatch.setattr(
        worker,
        "_resolve_worker_cuda_device",
        lambda _torch, requested: (
            torch.device("cuda:3") if requested == "cuda" else torch.device(requested)
        ),
    )
    seen: list[tuple[torch.device, str]] = []

    def fail_selected(_torch, resolved_device, current):
        seen.append((resolved_device, current))
        if current == phase:
            raise worker._CudaHealthFailure(
                current, "copy", "invalid_argument", "RuntimeError"
            )

    monkeypatch.setattr(worker, "_z_image_cuda_health_check", fail_selected)
    recipe = SimpleNamespace(
        fingerprint="test-fingerprint",
        plans={
            "pipeline_support": object(),
            "text_encoder": object(),
            "transformer": object(),
            "vae": object(),
        }
    )
    failure = worker._FailureContext()
    context = _child_context(tmp_path)
    with pytest.raises(worker._CudaHealthFailure) as raised:
        worker._load_core(
            recipe,
            "cuda",
            context,
            failure,
        )
    assert seen[-1] == (torch.device("cuda:3"), phase)
    assert all(resolved == torch.device("cuda:3") for resolved, _phase in seen)
    assert raised.value.completed == worker._CUDA_HEALTH_PHASES[
        : worker._CUDA_HEALTH_PHASES.index(phase)
    ]
    assert failure.stage == f"cuda_health_{phase}"
    assert failure.location == "z_image_turbo_worker._load_core"


def test_execute_attributes_immediate_pre_qwen_cuda_health_phase(
    tmp_path: Path, monkeypatch
):
    core = SimpleNamespace(
        execution_device=torch.device("cuda:0"),
        _latentslate_cuda_health_passes=worker._CUDA_HEALTH_PHASES[:-1],
    )

    def fail(_torch, _device, phase):
        raise worker._CudaHealthFailure(
            phase, "sync_before", "driver_error", "RuntimeError"
        )

    monkeypatch.setattr(worker, "_z_image_cuda_health_check", fail)
    failure = worker._FailureContext()
    context = _child_context(tmp_path)
    with pytest.raises(worker._CudaHealthFailure) as raised:
        worker._execute(
            core,
            SimpleNamespace(),
            {"prompt": "safe", "seed": 1},
            tmp_path / "output.png",
            "binding",
            context,
            True,
            failure,
        )
    assert failure.stage == "cuda_health_pre_qwen_preflight"
    assert failure.location == "z_image_turbo_worker._execute"
    assert raised.value.completed == worker._CUDA_HEALTH_PHASES[:-1]


def test_parent_rejects_tampered_failure_and_logs_only_safe_fields(tmp_path: Path, caplog):
    output = tmp_path / "output.png"
    hostile = "C:\\private\\secret-token.txt\nPROMPT=do not disclose"
    result = worker._failure_result(
        RuntimeError(hostile),
        worker._FailureContext("sampling", "z_image_turbo.generate", "binding"),
        "",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with caplog.at_level(logging.ERROR), pytest.raises(managed.ZImageWorkerFailure) as raised:
        managed._read_result(result_path, output, "binding", b"")
    public = str(raised.value)
    assert public == (
        "Z-Image worker failed (RuntimeError during sampling at z_image_turbo.generate; "
        f"diagnostic {result['error_fingerprint'][:12]})"
    )
    assert hostile not in public and hostile not in caplog.text
    assert "private" not in caplog.text and "PROMPT" not in caplog.text

    result["failure_stage"] = "hostile\nPROMPT"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind"):
        managed._read_result(result_path, output, "binding", b"")


def test_worker_main_preserves_bound_request_binding_without_exception_text(tmp_path: Path, monkeypatch):
    secret = bytes(range(32))
    payload = _payload(tmp_path, secret)
    request, result, progress, heartbeat, gate, command, cancel = (
        tmp_path / name
        for name in ("request.json", "result.json", "progress.jsonl", "heartbeat.jsonl", "gate", "command.json", "cancel")
    )
    request.write_text(json.dumps(payload), encoding="utf-8")
    gate.touch()
    hostile = "C:\\private\\secret-token.txt\nPROMPT=do not disclose"

    def fail(*_args, **_kwargs):
        raise RuntimeError(hostile)

    monkeypatch.setenv("LATENTSLATE_ZIMAGE_IPC_SECRET", secret.hex())
    monkeypatch.setattr(worker._ZImageHandler, "load", fail)
    assert worker.main(
        [
            "--request", str(request), "--result", str(result), "--progress", str(progress),
            "--heartbeat", str(heartbeat), "--start-gate", str(gate), "--command", str(command),
            "--cancel", str(cancel),
        ]
    ) == 1
    value = json.loads(result.read_text(encoding="utf-8"))
    assert value["request_binding"] == payload["request_binding"]
    assert hostile not in json.dumps(value)


def test_parent_validates_png_identity_dimensions_and_exact_dispatch(tmp_path: Path):
    output = tmp_path / "output.png"
    Image.new("RGB", (1024, 1024), "black").save(output, format="PNG")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    unsigned = {
        "schema_version": 1,
        "ok": True,
        "request_binding": "binding",
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": digest,
        "metadata": {},
    }
    result = {**unsigned, "result_binding": managed._result_binding(unsigned, b"")}
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert managed._read_result(result_path, output, "binding", b"")["output_sha256"] == digest
    result["output_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind"):
        managed._read_result(result_path, output, "binding", b"")


def test_parent_rejects_every_authenticated_result_tamper_and_wrong_secret(tmp_path: Path):
    """A valid-looking field cannot be changed without the session capability."""

    secret = bytes(range(32))
    output = tmp_path / "output.png"
    failure = worker._failure_result(
        RuntimeError("untrusted text"),
        worker._FailureContext("sampling", "z_image_turbo.generate", "binding"),
        "",
        secret,
    )
    result_path = tmp_path / "result.json"
    alternatives = {
        "schema_version": 2,
        "ok": True,
        "request_binding": "a" * 64,
        "error_type": "ValueError",
        "failure_stage": "decode",
        "failure_location": "z_image_turbo_worker._execute",
        "error_fingerprint": "f" * 64,
    }
    for field, value in alternatives.items():
        tampered = dict(failure)
        tampered[field] = value
        result_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not bind"):
            managed._read_result(result_path, output, "binding", secret)

    result_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind"):
        managed._read_result(result_path, output, "binding", b"wrong-secret")

    signed = {key: value for key, value in failure.items() if key != "result_binding"}
    for malformed in (
        {**signed, "unexpected": "field"},
        {key: value for key, value in signed.items() if key != "error_fingerprint"},
    ):
        malformed["result_binding"] = managed._result_binding(malformed, secret)
        result_path.write_text(json.dumps(malformed), encoding="utf-8")
        with pytest.raises(RuntimeError, match="failure result is invalid"):
            managed._read_result(result_path, output, "binding", secret)


def test_parent_rejects_success_tampering_before_output_or_metadata_trust(tmp_path: Path):
    secret = bytes(range(32))
    output = tmp_path / "output.png"
    Image.new("RGB", (1024, 1024), "black").save(output, format="PNG")
    unsigned = {
        "schema_version": 1,
        "ok": True,
        "request_binding": "binding",
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "metadata": {"safe": True},
    }
    result = {**unsigned, "result_binding": managed._result_binding(unsigned, secret)}
    result_path = tmp_path / "result.json"
    for field, value in {
        "output_path": str(tmp_path / "other.png"),
        "output_size_bytes": output.stat().st_size + 1,
        "output_sha256": "0" * 64,
        "metadata": {"safe": False},
    }.items():
        tampered = dict(result)
        tampered[field] = value
        result_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not bind"):
            managed._read_result(result_path, output, "binding", secret)


def test_hostile_exception_methods_cannot_break_or_influence_failure_serialization():
    class HostileException(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("must not stringify hostile exception")

        def __repr__(self) -> str:
            raise AssertionError("must not repr hostile exception")

        @property
        def args(self):  # type: ignore[override]
            raise AssertionError("must not read hostile exception args")

    result = worker._failure_result(
        HostileException(),
        worker._FailureContext("sampling", "z_image_turbo.generate", "binding"),
        "",
        bytes(range(32)),
    )
    assert result["error_type"] == "Exception"
    assert result["error_fingerprint"] == hashlib.sha256(
        b"zimage-worker-failure-v1:Exception:sampling:z_image_turbo.generate"
    ).hexdigest()


def test_transformer_onload_callback_propagates_before_gpu_stage_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The core observer must classify GPU staging before ``onload`` can fail."""

    request = SimpleNamespace(fingerprint="request")
    failure = worker._FailureContext()
    monkeypatch.setattr(core_module, "revalidate_z_image_turbo_runtime_request", lambda _: True)
    monkeypatch.setattr(
        core_module,
        "encode_z_image_prompt",
        lambda *_args, **_kwargs: SimpleNamespace(
            positive=torch.zeros((1, 1, 1)), attention_mask=torch.ones((1, 1), dtype=torch.bool)
        ),
    )

    class TextStage:
        def __init__(self, *_args):
            pass

        def onload(self):
            pass

        def offload(self):
            pass

        def verify_dispatch(self):
            return {}

    class TransformerStage:
        def __init__(self, *_args):
            pass

        def onload(self):
            assert failure.stage == "transformer_onload"
            assert failure.location == "z_image_turbo.generate"
            raise RuntimeError("synthetic onload failure")

        def offload(self):
            pass

        def verify_dispatch(self):
            return {}

    monkeypatch.setattr(core_module, "ZImageMixedQwenStage", TextStage)
    monkeypatch.setattr(core_module, "ZImageNextDiTStage", TransformerStage)
    core = core_module.ZImageTurboCore(
        request,
        tokenizer=object(),
        text_encoder=nn.Linear(1, 1),
        transformer=nn.Linear(1, 1),
        vae=nn.Linear(1, 1),
        execution_device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="synthetic onload failure"):
        core.generate(
            prompt="safe",
            seed=1,
            output_path=tmp_path / "unused.png",
            failure_stage=lambda stage: worker._set_core_failure_stage(failure, stage),
        )
    assert failure.stage == "transformer_onload"
    worker._set_core_failure_stage(failure, "conditioning.linear_117")
    assert failure.stage == "conditioning.linear_117"
    worker._set_core_failure_stage(failure, "conditioning.block_34")
    assert failure.stage == "conditioning.block_34"
    worker._set_core_failure_stage(failure, "conditioning.edge_07")
    assert failure.stage == "conditioning.edge_07"
    with pytest.raises(ValueError, match="invalid diagnostic stage"):
        worker._set_core_failure_stage(failure, "untrusted-stage")
    with pytest.raises(ValueError, match="invalid diagnostic stage"):
        worker._set_core_failure_stage(failure, "conditioning.linear_252")


def test_all_ordered_qwen_edges_are_authenticated_worker_to_parent():
    expected = {
        *(f"conditioning.edge_{index:02d}" for index in range(7, 21)),
        *(
            f"conditioning.preflight_{kind}_{substage}"
            for kind in ("fp8", "nvfp4")
            for substage in (
                "cuda_sync",
                "uint8_allocate",
                "ordinary_uint8_copy",
                "ordinary_uint8_sync",
                "ordinary_uint8_readback",
                "origin_flat_prepare",
                "origin_uint8_copy",
                "flat_dtype_view",
                "shape_restore",
                "scale_move",
                "bit_verify",
                "direct_fp32_dequant",
                "f_linear",
                "validate",
            )
        ),
    }
    assert expected <= worker._QWEN_FAILURE_STAGES
    assert expected <= worker._FAILURE_STAGES
    assert expected <= managed._QWEN_FAILURE_STAGES
    assert expected <= managed._FAILURE_STAGES

    for edge in sorted(expected):
        failure = worker._FailureContext(stage="conditioning", location="z_image_turbo.generate")
        worker._set_core_failure_stage(failure, edge)
        result = worker._failure_result(
            RuntimeError(r"private prompt C:\secret\model.safetensors"), failure, ""
        )
        assert result["failure_stage"] == edge
        accepted = managed._validate_failure_result(result, "")
        assert accepted is not None and accepted["failure_stage"] == edge
        assert "private" not in json.dumps(result)


def test_parent_provenance_requires_exact_mixed_and_convrot_closures():
    request = type(
        "Request",
        (),
        {
            "fingerprint": "fingerprint",
            "schedule": {"width": 1024},
            "components": {},
            "public_component_manifest": lambda _self: {},
        },
    )()
    metadata = {
        "family": "zimage",
        "runtime": "engine-native/z-image-turbo",
        "requested_device": "cuda:0",
        "execution_device": "cuda:0",
        "request_fingerprint": "fingerprint",
        "seed": 1,
        "schedule": {"width": 1024},
        "components": {},
        "phases": ["planning", "text_encoder", "transformer", "vae", "complete"],
        "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
        "lora_dispatch": None,
        "cuda_health": _cuda_health_proof(),
        "qwen_dispatch": {
            "contract": "full_precision_mm",
            "backend": "comfy-kitchen/public-direct-fp32-dequant+torch/f.linear",
            "module_count": 189,
            "fp8_modules": 177,
            "nvfp4_modules": 12,
            "dequantized_modules": 189,
            "f_linear_modules": 189,
            "complete": True,
            "min_module_dequant_delta": 1,
            "max_module_dequant_delta": 1,
            "total_dequantizations": 189,
            "total_f_linear_calls": 189,
            "dense_checkpoint_fallback_count": 0,
            "rejected_dispatch_count": 0,
            "activation_quantized": False,
            "scaled_mm_calls": 0,
            "per_op_residency": True,
            "stored_transport": "source-backed-raw-byte/vbar-equivalent",
            "full_module_cuda_onload": False,
            "cpu_master_retained": True,
            "first_linear_preflight": True,
            "first_linear_format": "fp8",
            "first_linear_logical_shape": "4096x2560",
            "first_linear_storage_dtype": "float8_e4m3fn",
            "first_linear_compute_dtype": "float32",
            "first_linear_output_shape": "1x1x4096",
            "first_linear_backend": (
                "comfy_kitchen.dequantize_per_tensor_fp8+torch/f.linear"
            ),
            "first_linear_layout_registered": True,
            "first_linear_transfer": "source-backed-raw-byte/current-stream/blocking",
            "first_linear_transport_equivalence": "vbar-output-equivalent",
            "first_linear_bit_identity": True,
            "first_linear_byte_count": 10_485_760,
            "first_linear_logical_wrapper_cast": False,
            "first_linear_dequant_contract": "public-direct-fp32",
        },
        "transformer_dispatch": {
            "module_count": 202,
            "dense_fallback_count": 0,
            "rejected_dispatch_count": 0,
        },
    }
    managed._validate_metadata(metadata, request, 1, "cuda:0")
    lora_request = type(
        "Request",
        (),
        {
            "fingerprint": "fingerprint",
            "schedule": {"width": 1024},
            "components": {"style_lora": {}},
            "public_component_manifest": lambda _self: {},
        },
    )()
    metadata["lora_dispatch"] = {
        "status": "proven",
        "backend": "engine-native/bf16-additive-bypass",
        "resource_id": "lora:zimage:kutches--imagezv2/70s-horror-movie-b",
        "strength": 1.0,
        "rank": 16,
        "target_count": 240,
        "qkv_row_slice_targets": 90,
        "direct_targets": 150,
        "total_dispatch_delta": 1920,
        "min_target_dispatch_delta": 8,
        "max_target_dispatch_delta": 8,
        "complete": True,
        "base_merged_or_dequantized": False,
    }
    managed._validate_metadata(metadata, lora_request, 1, "cuda:0")
    valid_lora = dict(metadata["lora_dispatch"])
    for field, invalid in (
        ("target_count", 239),
        ("total_dispatch_delta", 1919),
        ("total_dispatch_delta", 1921),
        ("min_target_dispatch_delta", True),
        ("max_target_dispatch_delta", True),
        ("total_dispatch_delta", True),
        ("rank", True),
        ("strength", True),
    ):
        metadata["lora_dispatch"] = {**valid_lora, field: invalid}
        with pytest.raises(RuntimeError, match="provenance"):
            managed._validate_metadata(metadata, lora_request, 1, "cuda:0")
    metadata["lora_dispatch"] = None
    metadata["requested_device"] = "cuda"
    metadata["execution_device"] = "cuda:2"
    managed._validate_metadata(metadata, request, 1, "cuda")
    metadata["requested_device"] = "cuda:0"
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")
    metadata["execution_device"] = "cuda:0"
    metadata["qwen_dispatch"]["fp8_modules"] = 176
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")
    metadata["qwen_dispatch"]["fp8_modules"] = 177
    metadata["qwen_dispatch"]["dequantized_modules"] = 188
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")
    metadata["qwen_dispatch"]["dequantized_modules"] = 189
    metadata["qwen_dispatch"]["total_f_linear_calls"] = 188
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")
    metadata["qwen_dispatch"]["total_f_linear_calls"] = 189
    metadata["qwen_dispatch"]["total_dequantizations"] = 190
    metadata["qwen_dispatch"]["total_f_linear_calls"] = 190
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")
    metadata["qwen_dispatch"]["total_dequantizations"] = 189
    metadata["qwen_dispatch"]["total_f_linear_calls"] = 189
    metadata["transformer_dispatch"] = {
        "module_count": 202,
        "dense_fallback_count": 1,
        "rejected_dispatch_count": 0,
    }
    with pytest.raises(RuntimeError, match="provenance"):
        managed._validate_metadata(metadata, request, 1, "cuda:0")


def test_child_persistent_loop_loads_once_and_executes_two_bound_commands(
    tmp_path: Path, monkeypatch
):
    secret = bytes(range(32))
    first = _payload_named(tmp_path, secret, "first.png")
    second = _payload_named(tmp_path, secret, "second.png")
    recipe = SimpleNamespace(
        fingerprint=first["recipe"]["fingerprint"],
        schedule=first["recipe"]["schedule"],
        public_component_manifest=lambda: {"safe": "manifest"},
    )
    calls: list[str] = []

    class FakeCore:
        execution_device = torch.device("cpu")
        _latentslate_cuda_health_passes = worker._CUDA_HEALTH_PHASES[:-1]

        def generate(self, *, output_path: Path, **_kwargs):
            calls.append(output_path.name)
            Image.new("RGB", (1024, 1024), "black").save(output_path, format="PNG")
            return SimpleNamespace(
                artifact=SimpleNamespace(
                    path=output_path,
                    size_bytes=output_path.stat().st_size,
                    sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                ),
                qwen_dispatch={"module_count": 189},
                transformer_dispatch={"module_count": 202},
                phases=("complete",),
            )

    monkeypatch.setattr(
        "latentslate_engine.z_image_turbo_recipe.rehydrate_z_image_turbo_runtime_request",
        lambda _value: recipe,
    )
    monkeypatch.setattr(worker, "_load_core", lambda *_args: FakeCore())
    monkeypatch.setattr(worker, "_z_image_cuda_health_check", lambda *_args: None)
    context = _child_context(tmp_path)
    handler = worker._ZImageHandler(secret.hex())
    first_command = handler.bind_initial(first, context)
    session = handler.load(first_command, context)
    handler.execute(session, first_command, context, cold=True)
    second_command = handler.bind_command(second, session, context)
    context.reset_progress()
    handler.execute(session, second_command, context, cold=False)
    assert calls == ["first.png", "second.png"]


def test_supervisor_deadlines_and_cancel_marker_are_bounded(tmp_path: Path):
    supervisor = _inert_supervisor(tmp_path)
    with pytest.raises(managed.ZImageWorkerTimeout, match="deadline"):
        managed._wait_for_z_result(
            supervisor,
            lambda *_args: None,
            lambda: None,
            generation_timeout_seconds=0.00001,
            stage_timeout_seconds=1,
            cancel_grace_seconds=0.00001,
        )

    class ToolCancelled(RuntimeError):
        pass

    with pytest.raises(ToolCancelled):
        managed._wait_for_z_result(
            supervisor,
            lambda *_args: None,
            lambda: (_ for _ in ()).throw(ToolCancelled()),
            generation_timeout_seconds=1,
            stage_timeout_seconds=1,
            cancel_grace_seconds=0.00001,
        )
    assert supervisor.paths.cancel.is_file()
    assert managed._outcome(ToolCancelled()) == "canceled"


def test_timeout_boundary_gives_atomic_result_one_ordered_last_chance(tmp_path: Path, monkeypatch):
    """A result written exactly at the timeout yield wins over process destruction."""

    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))

    def yield_once(seconds: float) -> None:
        if seconds == 0:
            paths.result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(persistent_module.time, "sleep", yield_once)
    managed._wait_for_z_result(
        supervisor,
        lambda *_args: None,
        lambda: None,
        generation_timeout_seconds=1,
        stage_timeout_seconds=100,
        cancel_grace_seconds=0.001,
    )
    assert paths.result.is_file()
    assert not paths.cancel.exists()


def test_heartbeat_boundary_record_renews_only_heartbeat_liveness(tmp_path: Path, monkeypatch):
    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    clock = iter((0.0, 46.0, 46.0, 46.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))

    def race_writer(seconds: float) -> None:
        if seconds == 0:
            paths.heartbeat.write_text('{"heartbeat":1}\n', encoding="utf-8")
        elif seconds == managed._POLL:
            paths.result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(persistent_module.time, "sleep", race_writer)
    managed._wait_for_z_result(
        supervisor,
        lambda *_args: None,
        lambda: None,
        generation_timeout_seconds=100,
        stage_timeout_seconds=100,
        cancel_grace_seconds=0.001,
    )
    assert paths.result.is_file()
    assert not paths.cancel.exists()


def test_hard_generation_deadline_is_not_extended_by_a_boundary_heartbeat(
    tmp_path: Path, monkeypatch
):
    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    clock = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        persistent_module.time,
        "sleep",
        lambda seconds: paths.heartbeat.write_text('{"heartbeat":1}\n', encoding="utf-8")
        if seconds == 0
        else None,
    )
    monkeypatch.setattr(
        supervisor,
        "request_cancel_and_grace",
        lambda _seconds: paths.cancel.touch(exist_ok=True),
    )
    with pytest.raises(managed.ZImageWorkerTimeout, match="generation exceeded"):
        managed._wait_for_z_result(
            supervisor,
            lambda *_args: None,
            lambda: None,
            generation_timeout_seconds=1,
            stage_timeout_seconds=100,
            cancel_grace_seconds=0.001,
        )
    assert paths.heartbeat.is_file() and paths.cancel.is_file()


def test_progress_boundary_record_renews_only_stage_liveness(tmp_path: Path, monkeypatch):
    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    paths.heartbeat.write_text('{"heartbeat":1}\n', encoding="utf-8")
    clock = iter((0.0, 0.0, 11.0, 11.0, 11.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))

    def race_writer(seconds: float) -> None:
        if seconds == 0:
            paths.progress.write_text(
                '{"progress":0.5,"message":"Sampling Z-Image latents"}\n', encoding="utf-8"
            )
        elif seconds == managed._POLL:
            paths.result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(persistent_module.time, "sleep", race_writer)
    seen: list[float] = []
    managed._wait_for_z_result(
        supervisor,
        lambda value, _message: seen.append(value),
        lambda: None,
        generation_timeout_seconds=100,
        stage_timeout_seconds=10,
        cancel_grace_seconds=0.001,
    )
    assert seen == [0.5]
    assert paths.result.is_file()
    assert not paths.cancel.exists()


def test_stale_heartbeat_is_a_terminal_worker_timeout(tmp_path: Path, monkeypatch):
    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    clock = iter((0.0, 46.0, 46.0, 46.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(persistent_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        supervisor,
        "request_cancel_and_grace",
        lambda _seconds: paths.cancel.touch(exist_ok=True),
    )
    with pytest.raises(managed.ZImageWorkerTimeout, match="heartbeat became stale"):
        managed._wait_for_z_result(
            supervisor,
            lambda *_args: None,
            lambda: None,
            generation_timeout_seconds=100,
            stage_timeout_seconds=100,
            cancel_grace_seconds=0.001,
        )
    assert paths.cancel.is_file()


def test_stale_progress_remains_a_terminal_worker_timeout(tmp_path: Path, monkeypatch):
    supervisor = _inert_supervisor(tmp_path)
    paths = supervisor.paths
    paths.heartbeat.write_text('{"heartbeat":1}\n', encoding="utf-8")
    clock = iter((0.0, 0.0, 11.0))
    monkeypatch.setattr(persistent_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(persistent_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        supervisor,
        "request_cancel_and_grace",
        lambda _seconds: paths.cancel.touch(exist_ok=True),
    )
    with pytest.raises(managed.ZImageWorkerTimeout, match="stage exceeded"):
        managed._wait_for_z_result(
            supervisor,
            lambda *_args: None,
            lambda: None,
            generation_timeout_seconds=100,
            stage_timeout_seconds=10,
            cancel_grace_seconds=0.001,
        )
    assert paths.cancel.is_file()


def test_worker_stops_heartbeat_before_success_result_and_maps_cold_and_warm_progress(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "image.png"
    publications: list[bool] = []
    context = _child_context(tmp_path)
    progress_path = context.paths.progress

    class Core:
        execution_device = torch.device("cpu")
        _latentslate_cuda_health_passes = worker._CUDA_HEALTH_PHASES[:-1]

        def generate(self, *, output_path: Path, progress, **_kwargs):
            for value, message in (
                (0.02, "Encoding Z-Image prompt"),
                (0.12, "Sampling Z-Image latents"),
                (0.82, "Z-Image step 8/8"),
                (0.86, "Decoding Z-Image PNG"),
                (1.0, "Complete"),
            ):
                progress(value, message)
            Image.new("RGB", (1024, 1024), "black").save(output_path, format="PNG")
            return SimpleNamespace(
                artifact=SimpleNamespace(
                    path=output_path,
                    size_bytes=output_path.stat().st_size,
                    sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                ),
                qwen_dispatch={"module_count": 189},
                transformer_dispatch={"module_count": 202},
                phases=("complete",),
            )

    recipe = SimpleNamespace(
        fingerprint="request", schedule={"width": 1024}, public_component_manifest=dict
    )

    def write_result(path: Path, value: dict[str, object]) -> None:
        publications.append(context._heartbeat is None)
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(persistent_child_module, "atomic_write_json", write_result)
    monkeypatch.setattr(worker, "_z_image_cuda_health_check", lambda *_args: None)
    context.publish_progress(0.01, "Validating Z-Image request")
    context.publish_progress(0.12, "Core ready")
    context.start_heartbeat()
    result = worker._execute(
        Core(),
        recipe,
        {"prompt": "p", "seed": 1},
        output,
        "binding",
        context,
        cold=True,
    )
    context.publish_result(result)
    records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    values = [record["progress"] for record in records]
    assert values == sorted(values)
    assert 0.12 in values and 0.92 in values and values[-1] == 1.0
    assert publications == [True]

    # A warm command has no fictional loading range, but follows the exact
    # same inference/decode scale once it starts doing useful work.
    warm_context = _child_context(tmp_path / "warm")
    warm_progress = warm_context.paths.progress
    warm_progress.parent.mkdir()
    worker._execution_progress(warm_context, 0.12, "Sampling Z-Image latents", cold=False)
    worker._execution_progress(warm_context, 0.82, "Z-Image step 8/8", cold=False)
    worker._execution_progress(warm_context, 0.86, "Decoding Z-Image PNG", cold=False)
    warm_values = [
        json.loads(line)["progress"] for line in warm_progress.read_text(encoding="utf-8").splitlines()
    ]
    assert warm_values == [0.12, 0.92, 0.92]


def test_heartbeat_stop_requires_a_real_join(tmp_path: Path):
    context = _child_context(tmp_path)
    context.start_heartbeat()
    assert context._heartbeat is not None
    thread = context._heartbeat[1]
    context.stop_heartbeat()
    assert not thread.is_alive()


def test_tool_records_post_unload_status_without_a_stale_worker_pid(tmp_path: Path, monkeypatch):
    """``keep_pipeline_loaded=false`` must describe the state after eviction."""

    from latentslate_engine.config import Settings
    from latentslate_engine.runtime.manager import RUNTIME_MANAGER
    from latentslate_engine.storage import Storage
    from latentslate_engine.tools.base import ExecutionPlan, ToolContext
    from latentslate_engine.tools.z_image_turbo import ZImageTurboTextToImageTool

    recipe = SimpleNamespace(
        fingerprint="request",
        components={"transformer": {"header_sha256": "header"}},
        public_component_manifest=lambda: {"exact": True},
    )
    requested_devices: list[str] = []

    class Runtime:
        def __init__(self, _recipe):
            self.loaded = True

        def generate(self, *, output_path: Path, device: str, **_kwargs):
            requested_devices.append(device)
            Image.new("RGB", (1024, 1024), "black").save(output_path, format="PNG")
            return SimpleNamespace(
                metadata={"family": "zimage"}, worker_pid=4242, pipeline_warm=False
            )

        def unload(self):
            self.loaded = False

        def clear_cache(self):
            pass

        def status(self):
            return {"loaded": self.loaded, "worker_pid": 4242 if self.loaded else None, "cleanup_errors": []}

    settings = Settings(
        tmp_path,
        None,
        1024 * 1024,
        "h3",
        "profile",
        "cuda:0",
        execution_device="cuda:3",
        execution_device_source="LATENTSLATE_EXECUTION_DEVICE",
    )
    context = ToolContext(
        job_id=uuid4(),
        settings=settings,
        storage=Storage(settings),
        cancel_event=SimpleNamespace(is_set=lambda: False),
        progress=lambda *_args: None,
        execution=ExecutionPlan(
            variant_key="z-test",
            family="zimage",
            optimizations={"keep_pipeline_loaded": False},
            recipe=recipe,
        ),
    )
    import latentslate_engine.tools.z_image_turbo as tool_module

    monkeypatch.setattr(tool_module, "ZImageTurboRuntimeRequest", SimpleNamespace)
    monkeypatch.setattr(tool_module, "revalidate_z_image_turbo_runtime_request", lambda _recipe: True)
    monkeypatch.setattr(ZImageTurboTextToImageTool, "variant_base_availability", lambda _self: (True, None))
    monkeypatch.setattr(
        "latentslate_engine.runtime.z_image_turbo_managed.ManagedZImageTurboRuntime", Runtime
    )
    RUNTIME_MANAGER.clear()
    try:
        artifacts = ZImageTurboTextToImageTool().run(context, {"prompt": "p", "seed": 1})
    finally:
        RUNTIME_MANAGER.clear()
    assert len(artifacts) == 1
    assert requested_devices == ["cuda:3"]
    assert context.runtime_provenance["runtime_plan"]["requested_device"] == "cuda:3"
    assert (
        context.runtime_provenance["runtime_plan"]["execution_device_source"]
        == "LATENTSLATE_EXECUTION_DEVICE"
    )
    status = context.runtime_provenance["runtime_result"]["runtime_status"]
    assert status["loaded"] is False and status["worker_pid"] is None


def test_managed_session_reuses_one_child_then_unloads_once(tmp_path: Path, monkeypatch):
    request = SimpleNamespace(
        fingerprint="request",
        schedule={"width": 1024},
        components={"transformer": {"header_sha256": "header"}},
        to_json_dict=lambda: {"recipe": "bound"},
        public_component_manifest=lambda: {"safe": "components"},
    )
    runtime = managed.ManagedZImageTurboRuntime(request)
    paths = _persistent_paths(tmp_path)
    terminated: list[str] = []

    class Process:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    class Tree:
        def terminate(self):
            terminated.append("terminate")

        def wait_for_empty(self):
            terminated.append("empty")

        def close(self):
            terminated.append("close")

    process, tree = Process(), Tree()

    class Supervisor:
        def __init__(self, secret: bytes):
            self.paths = paths
            self.secret = secret
            self.session = None

        def start(self, payload):
            self.paths.request.write_text(json.dumps(payload), encoding="utf-8")
            self.session = SimpleNamespace(process=process)

        def send(self, payload):
            self.paths.command.write_text(json.dumps(payload), encoding="utf-8")

        def terminate(self):
            tree.terminate()
            process.wait(timeout=15)
            tree.wait_for_empty()

        def close(self):
            tree.close()

        def cleanup_job(self):
            for path in (
                self.paths.command,
                self.paths.result,
                self.paths.progress,
                self.paths.heartbeat,
            ):
                path.unlink(missing_ok=True)
            return []

        def cleanup_session(self):
            self.cleanup_job()
            for path in (self.paths.request, self.paths.start_gate, self.paths.cancel):
                path.unlink(missing_ok=True)
            return []

    monkeypatch.setattr(managed, "revalidate_z_image_turbo_runtime_request", lambda _request: True)
    monkeypatch.setattr(managed, "_paths", lambda: paths)
    monkeypatch.setattr(managed, "_supervisor", lambda _paths, secret: Supervisor(secret))

    def fake_wait(session, _progress, _cancelled, **_timeouts):
        payload = (
            json.loads(session.paths.request.read_text(encoding="utf-8"))
            if not session.paths.command.is_file()
            else json.loads(session.paths.command.read_text(encoding="utf-8"))
        )
        output = Path(payload["output_path"])
        Image.new("RGB", (1024, 1024), "black").save(output, format="PNG")
        metadata = {
            "family": "zimage",
            "runtime": "engine-native/z-image-turbo",
            "requested_device": payload["device"],
            "execution_device": "cuda:0",
            "request_fingerprint": "request",
            "seed": payload["generation"]["seed"],
            "schedule": {"width": 1024},
            "components": {"safe": "components"},
            "phases": ["planning", "text_encoder", "transformer", "vae", "complete"],
            "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
            "lora_dispatch": None,
            "cuda_health": _cuda_health_proof(),
            "qwen_dispatch": {
                "contract": "full_precision_mm",
                "backend": "comfy-kitchen/public-direct-fp32-dequant+torch/f.linear",
                "module_count": 189,
                "fp8_modules": 177,
                "nvfp4_modules": 12,
                "dequantized_modules": 189,
                "f_linear_modules": 189,
                "complete": True,
                "min_module_dequant_delta": 1,
                "max_module_dequant_delta": 1,
                "total_dequantizations": 189,
                "total_f_linear_calls": 189,
                "dense_checkpoint_fallback_count": 0,
                "rejected_dispatch_count": 0,
                "activation_quantized": False,
                "scaled_mm_calls": 0,
                "per_op_residency": True,
                "stored_transport": "source-backed-raw-byte/vbar-equivalent",
                "full_module_cuda_onload": False,
                "cpu_master_retained": True,
                "first_linear_preflight": True,
                "first_linear_format": "fp8",
                "first_linear_logical_shape": "4096x2560",
                "first_linear_storage_dtype": "float8_e4m3fn",
                "first_linear_compute_dtype": "float32",
                "first_linear_output_shape": "1x1x4096",
                "first_linear_backend": (
                    "comfy_kitchen.dequantize_per_tensor_fp8+torch/f.linear"
                ),
                "first_linear_layout_registered": True,
                "first_linear_transfer": "source-backed-raw-byte/current-stream/blocking",
                "first_linear_transport_equivalence": "vbar-output-equivalent",
                "first_linear_bit_identity": True,
                "first_linear_byte_count": 10_485_760,
                "first_linear_logical_wrapper_cast": False,
                "first_linear_dequant_contract": "public-direct-fp32",
            },
            "transformer_dispatch": {
                "module_count": 202,
                "dense_fallback_count": 0,
                "rejected_dispatch_count": 0,
            },
        }
        unsigned = {
            "schema_version": 1,
            "ok": True,
            "request_binding": payload["request_binding"],
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "metadata": metadata,
        }
        session.paths.result.write_text(
            json.dumps(
                {
                    **unsigned,
                    "result_binding": managed._result_binding(unsigned, session.secret),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(managed, "_wait_for_z_result", fake_wait)
    first = runtime.generate(
        prompt="one",
        seed=1,
        output_path=tmp_path / "one.png",
        device="cuda:0",
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    second = runtime.generate(
        prompt="two",
        seed=2,
        output_path=tmp_path / "two.png",
        device="cuda:0",
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    assert first.worker_pid == second.worker_pid == 4242
    assert not first.pipeline_warm and second.pipeline_warm
    runtime.unload()
    assert terminated == ["terminate", "empty", "close"]
    assert runtime.status()["loaded"] is False


def test_managed_failed_start_retains_tree_proof_without_masking_gate_error(
    tmp_path: Path, monkeypatch
):
    request = SimpleNamespace(
        fingerprint="request",
        schedule={"width": 1024},
        components={"transformer": {"header_sha256": "header"}},
        to_json_dict=lambda: {"recipe": "bound"},
        public_component_manifest=lambda: {"safe": "components"},
    )
    runtime = managed.ManagedZImageTurboRuntime(request)
    paths = _persistent_paths(tmp_path)

    class Supervisor:
        failed_start = PersistentWorkerFailedStart(
            pid=4545,
            exit_code=1,
            terminated=True,
            tree_empty=True,
            cleanup_errors=("root:OSError",),
        )

        def start(self, _payload):
            raise FileExistsError("original gate publication failure")

    monkeypatch.setattr(managed, "revalidate_z_image_turbo_runtime_request", lambda _request: True)
    monkeypatch.setattr(managed, "_paths", lambda: paths)
    monkeypatch.setattr(managed, "_supervisor", lambda _paths, _secret: Supervisor())

    with pytest.raises(FileExistsError, match="original gate publication failure"):
        runtime.generate(
            prompt="safe prompt",
            seed=1,
            output_path=tmp_path / "never-published.png",
            device="cuda:0",
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    status = runtime.status()
    assert status["loaded"] is False and status["worker_pid"] is None
    assert status["last_worker"] == {
        "pid": 4545,
        "exit_code": 1,
        "terminated": True,
        "tree_empty": True,
        "outcome": "failed",
        "timeout": False,
        "pipeline_warm": False,
        "memory_boundary": "persistent_exact_recipe_worker",
    }
    assert status["cleanup_errors"] == ["root:OSError"]


def test_failure_result_poisoning_keeps_tree_cleanup_and_safe_runtime_status(tmp_path: Path, monkeypatch):
    request = SimpleNamespace(
        fingerprint="request",
        schedule={"width": 1024},
        components={"transformer": {"header_sha256": "header"}},
        to_json_dict=lambda: {"recipe": "bound"},
        public_component_manifest=lambda: {"safe": "components"},
    )
    runtime = managed.ManagedZImageTurboRuntime(request)
    paths = _persistent_paths(tmp_path)
    terminated: list[str] = []

    class Process:
        pid = 4343

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 1

    class Tree:
        def terminate(self):
            terminated.append("terminate")

        def wait_for_empty(self):
            terminated.append("empty")

        def close(self):
            terminated.append("close")

    process, tree = Process(), Tree()

    class Supervisor:
        def __init__(self, secret: bytes):
            self.paths = paths
            self.secret = secret
            self.session = None

        def start(self, payload):
            self.paths.request.write_text(json.dumps(payload), encoding="utf-8")
            self.session = SimpleNamespace(process=process)

        def terminate(self):
            tree.terminate()
            process.wait(timeout=15)
            tree.wait_for_empty()

        def close(self):
            tree.close()

        def cleanup_job(self):
            for path in (
                self.paths.command,
                self.paths.result,
                self.paths.progress,
                self.paths.heartbeat,
            ):
                path.unlink(missing_ok=True)
            return []

        def cleanup_session(self):
            self.cleanup_job()
            for path in (self.paths.request, self.paths.start_gate, self.paths.cancel):
                path.unlink(missing_ok=True)
            return []

    monkeypatch.setattr(managed, "revalidate_z_image_turbo_runtime_request", lambda _request: True)
    monkeypatch.setattr(managed, "_paths", lambda: paths)
    monkeypatch.setattr(managed, "_supervisor", lambda _paths, secret: Supervisor(secret))

    def fail_wait(session, _progress, _cancelled, **_timeouts):
        payload = json.loads(session.paths.request.read_text(encoding="utf-8"))
        session.paths.result.write_text(
            json.dumps(
                worker._failure_result(
                    RuntimeError("C:\\private\\secret-token.txt\nPROMPT=do not disclose"),
                    worker._FailureContext(
                        "nextdit_materialize",
                        "z_image_turbo_worker._load_core",
                        payload["request_binding"],
                    ),
                    "",
                    session.secret,
                )
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(managed, "_wait_for_z_result", fail_wait)
    with pytest.raises(managed.ZImageWorkerFailure, match="nextdit_materialize"):
        runtime.generate(
            prompt="safe prompt",
            seed=1,
            output_path=tmp_path / "failed.png",
            device="cuda:0",
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert terminated == ["terminate", "empty", "close"]
    status = runtime.status()
    assert status["loaded"] is False
    assert status["last_worker"] == {
        "pid": 4343,
        "exit_code": None,
        "terminated": True,
        "outcome": "failed",
        "timeout": False,
        "pipeline_warm": False,
        "memory_boundary": "persistent_exact_recipe_worker",
        "failure_stage": "nextdit_materialize",
        "failure_location": "z_image_turbo_worker._load_core",
        "diagnostic": status["last_worker"]["diagnostic"],
    }
    assert len(status["last_worker"]["diagnostic"]) == 12
    assert not tmp_path.exists() or not any(tmp_path.iterdir())
