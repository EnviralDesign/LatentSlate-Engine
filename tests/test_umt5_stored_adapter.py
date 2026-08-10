from __future__ import annotations

import gc
import json
import os
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import latentslate_engine.runtime.umt5_stored_adapter as umt5_adapter
from latentslate_engine.runtime.umt5_stored_adapter import (
    UMT5EncoderResidencySession,
    build_umt5_encoder_skeleton,
    materialize_umt5_encoder,
    plan_comfy_umt5_encoder,
)
from latentslate_engine.runtime.wan22_stored_adapter import NativeStoredLinear

_CONFIG = {
    "vocab_size": 16,
    "d_model": 4,
    "d_kv": 2,
    "d_ff": 16,
    "num_layers": 1,
    "num_heads": 2,
    "dropout_rate": 0.0,
    "feed_forward_proj": "gated-gelu",
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "layer_norm_epsilon": 1e-6,
    "tie_word_embeddings": True,
}


def _marker(value: dict[str, object]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value).encode("utf-8")), dtype=torch.uint8)


def _write_encoder(path: Path, contract: str) -> None:
    skeleton = build_umt5_encoder_skeleton(_CONFIG)
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    seen_shared = False
    for target, value in skeleton.state_dict().items():
        if target == "encoder.embed_tokens.weight":
            continue
        source = target
        if source == "shared.weight":
            seen_shared = True
        parent_path, _, _ = target.rpartition(".")
        parent = skeleton.get_submodule(parent_path) if parent_path else skeleton
        shape = tuple(value.shape)
        quantized = source.endswith(".weight") and isinstance(parent, torch.nn.Linear)
        if not quantized:
            tensors[source] = torch.zeros(shape, dtype=torch.float16)
            continue
        stem = source.removesuffix(".weight")
        if contract == "comfy_legacy/scaled_fp8_e4m3fn":
            tensors[source] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
            tensors[stem + ".scale_weight"] = torch.tensor(0.25, dtype=torch.float32)
        else:
            tensors[source] = torch.zeros(shape, dtype=torch.int8)
            tensors[stem + ".weight_scale"] = torch.full((shape[0], 1), 0.25, dtype=torch.float32)
            tensors[stem + ".input_scale"] = torch.full((shape[0], 1), 0.25, dtype=torch.float32)
            tensors[stem + ".comfy_quant"] = _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": shape[1]}
            )
            layers[stem] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": shape[1]}
    assert seen_shared
    tensors["spiece_model"] = torch.tensor([1], dtype=torch.uint8)
    if contract == "comfy_legacy/scaled_fp8_e4m3fn":
        tensors["scaled_fp8"] = torch.empty((0,), dtype=torch.float8_e4m3fn)
        metadata = None
    else:
        metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    save_file(tensors, path, metadata=metadata)


@pytest.mark.parametrize("contract", ["comfy_legacy/scaled_fp8_e4m3fn", "comfy_quant/int8_tensorwise_convrot"])
def test_complete_stored_umt5_encoder_plan_and_forward(tmp_path: Path, contract: str):
    path = tmp_path / "umt5.safetensors"
    _write_encoder(path, contract)

    plan = plan_comfy_umt5_encoder(path, _CONFIG)
    assert plan.available, plan.errors
    assert plan.source_to_targets["shared.weight"] == ("shared.weight", "encoder.embed_tokens.weight")
    encoder = materialize_umt5_encoder(plan, _CONFIG, compute_dtype=torch.float16)

    assert not any(parameter.is_meta for parameter in encoder.parameters())
    assert isinstance(encoder.encoder.block[0].layer[0].SelfAttention.q, NativeStoredLinear)
    assert encoder.shared.weight is encoder.encoder.embed_tokens.weight
    output = encoder(input_ids=torch.tensor([[1, 2]], dtype=torch.long)).last_hidden_state
    assert output.shape == (1, 2, 4)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_convrot_input_scale_must_be_exact_weight_scale_alias(tmp_path: Path):
    path = tmp_path / "umt5-convrot.safetensors"
    _write_encoder(path, "comfy_quant/int8_tensorwise_convrot")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors["encoder.block.0.layer.0.SelfAttention.q.input_scale"][0, 0] = 0.5
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="input_scale is not the proven stored scale alias"):
        materialize_umt5_encoder(
            plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16
        )


@pytest.mark.parametrize(
    "source",
    ["shared.weight", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
)
def test_plan_rejects_stored_quant_sidecars_on_embedding_weights(tmp_path: Path, source: str):
    path = tmp_path / "invalid-embedding-quant.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors[source.removesuffix(".weight") + ".scale_weight"] = torch.tensor(0.25, dtype=torch.float32)
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_umt5_encoder(path, _CONFIG)

    assert not plan.available
    assert any("nn.Linear" in error or "tied/shared" in error for error in plan.errors)


def test_plan_rejects_orphan_quant_sidecar(tmp_path: Path):
    path = tmp_path / "orphan-sidecar.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors["encoder.block.0.layer.0.SelfAttention.orphan.scale_weight"] = torch.tensor(0.25, dtype=torch.float32)
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_umt5_encoder(path, _CONFIG)

    assert not plan.available
    assert plan.unexpected_extras == ("encoder.block.0.layer.0.SelfAttention.orphan.scale_weight",)


def test_materializer_late_failure_dematerializes_and_releases_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "late-failure.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    plan = plan_comfy_umt5_encoder(path, _CONFIG)
    original_build = umt5_adapter.build_umt5_encoder_skeleton
    original_dematerialize = umt5_adapter._dematerialize
    references: list[weakref.ReferenceType[torch.nn.Module]] = []
    dematerialized: list[bool] = []

    def record_build(config):
        encoder = original_build(config)
        references.append(weakref.ref(encoder))
        return encoder

    def record_dematerialize(encoder):
        original_dematerialize(encoder)
        dematerialized.append(all(parameter.is_meta for parameter in encoder.parameters()))

    monkeypatch.setattr(umt5_adapter, "build_umt5_encoder_skeleton", record_build)
    monkeypatch.setattr(umt5_adapter, "_dematerialize", record_dematerialize)
    monkeypatch.setattr(
        umt5_adapter,
        "_assign_alias_targets",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("late dense assignment failure")),
    )

    with pytest.raises(RuntimeError, match="late dense assignment failure"):
        materialize_umt5_encoder(plan, _CONFIG, compute_dtype=torch.float16)

    assert dematerialized == [True]
    gc.collect()
    assert len(references) == 1
    assert references[0]() is None


@pytest.mark.parametrize("contract", ["comfy_legacy/scaled_fp8_e4m3fn", "comfy_quant/int8_tensorwise_convrot"])
def test_umt5_residency_encodes_masks_and_offloads_cpu(tmp_path: Path, contract: str):
    path = tmp_path / "umt5-residency.safetensors"
    _write_encoder(path, contract)
    encoder = materialize_umt5_encoder(plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16)
    session = UMT5EncoderResidencySession(encoder, onload_device="cpu")

    with session:
        output = session.encode(torch.tensor([[1, 2]]), torch.tensor([[1, 0]]), sequence_length=4)
        assert output.shape == (1, 4, 4)
        assert output.dtype == torch.float16
        assert torch.equal(output[:, 1:], torch.zeros_like(output[:, 1:]))

    assert not session.active
    assert all(value.device.type == "cpu" for value in encoder.parameters())
    quant = encoder.encoder.block[0].layer[0].SelfAttention.q
    assert quant.weight._qdata.device.type == "cpu"
    assert quant.weight.params.scale.device.type == "cpu"


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF") != "1" or not torch.cuda.is_available(),
    reason="set LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF=1 on a CUDA host",
)
@pytest.mark.parametrize("contract", ["comfy_legacy/scaled_fp8_e4m3fn", "comfy_quant/int8_tensorwise_convrot"])
def test_opt_in_cuda_umt5_residency_round_trip(tmp_path: Path, contract: str):
    path = tmp_path / "umt5-cuda-residency.safetensors"
    _write_encoder(path, contract)
    encoder = materialize_umt5_encoder(plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16)
    session = UMT5EncoderResidencySession(encoder, onload_device="cuda")

    with session:
        output = session.encode(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), sequence_length=4)
        assert output.device.type == "cuda"
        quant = encoder.encoder.block[0].layer[0].SelfAttention.q
        assert quant.weight._qdata.device.type == "cuda"
        assert quant.weight.params.scale.device.type == "cuda"

    quant = encoder.encoder.block[0].layer[0].SelfAttention.q
    assert quant.weight._qdata.device.type == "cpu"
    assert quant.weight.params.scale.device.type == "cpu"


def test_umt5_residency_is_process_wide_cross_thread_safe_and_baseexception_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "umt5-session-safety.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    encoder = materialize_umt5_encoder(plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16)
    first = UMT5EncoderResidencySession(encoder, onload_device="cpu")
    second = UMT5EncoderResidencySession(encoder, onload_device="cpu")
    errors: list[BaseException] = []

    with first:
        with pytest.raises(RuntimeError, match="process-wide"):
            second.__enter__()
        thread = threading.Thread(target=lambda: errors.append(_close_error(first)))
        thread.start()
        thread.join()
        assert errors and isinstance(errors[0], RuntimeError)
        assert first.active
        monkeypatch.setattr(encoder, "forward", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            first.encode(torch.tensor([[1]]), torch.tensor([[1]]), sequence_length=4)

    assert not first.active
    assert all(value.device.type == "cpu" for value in encoder.parameters())


@pytest.mark.parametrize(
    ("input_ids", "mask", "sequence_length", "message"),
    [
        (torch.tensor([[1.0]]), None, 4, "int32 or int64"),
        (torch.tensor([[-1]]), None, 4, "nonnegative"),
        (torch.tensor([[16]]), None, 4, "within the encoder vocabulary"),
        (torch.tensor([[1, 2]]), None, True, "positive non-bool"),
        (torch.tensor([[1, 2]]), None, 1, "exceed"),
        (torch.tensor([[1]]), torch.tensor([[0.5]]), 4, "boolean or integer"),
        (torch.tensor([[1]]), torch.tensor([[2]]), 4, "binary"),
    ],
)
def test_umt5_encode_rejects_invalid_prompt_contract(
    tmp_path: Path, input_ids: torch.Tensor, mask: torch.Tensor | None, sequence_length: int, message: str
):
    path = tmp_path / "invalid-prompt.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    encoder = materialize_umt5_encoder(plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16)

    with UMT5EncoderResidencySession(encoder, onload_device="cpu") as session, pytest.raises(
        ValueError, match=message
    ):
        session.encode(input_ids, mask, sequence_length=sequence_length)


def test_umt5_encode_masked_nan_output_is_bitwise_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "masked-nan.safetensors"
    _write_encoder(path, "comfy_legacy/scaled_fp8_e4m3fn")
    encoder = materialize_umt5_encoder(plan_comfy_umt5_encoder(path, _CONFIG), _CONFIG, compute_dtype=torch.float16)

    def nan_forward(*_args, **kwargs):
        batch, sequence = kwargs["input_ids"].shape
        return SimpleNamespace(last_hidden_state=torch.full((batch, sequence, 4), float("nan"), dtype=torch.float16))

    monkeypatch.setattr(encoder, "forward", nan_forward)
    with UMT5EncoderResidencySession(encoder, onload_device="cpu") as session:
        output = session.encode(torch.tensor([[1]]), torch.tensor([[0]]), sequence_length=4)

    assert torch.equal(output, torch.zeros_like(output))


def _close_error(session: UMT5EncoderResidencySession) -> BaseException:
    try:
        session.close()
    except BaseException as exc:  # noqa: BLE001 - test captures cross-thread refusal
        return exc
    raise AssertionError("cross-thread close unexpectedly succeeded")
