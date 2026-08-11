from __future__ import annotations

import gc
import json
import threading
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from latentslate_engine.runtime import klein_stored_adapter as adapter
from latentslate_engine.runtime.klein_stored_adapter import (
    KLEIN_STORED_FP8_CONTRACT,
    KleinStoredLinear,
    KleinTransformerResidencySession,
    build_klein_transformer_skeleton,
    comfy_flux2_source_for_target,
    map_comfy_flux2_parameter,
    materialize_klein_transformer,
    move_klein_transformer_storage,
    plan_comfy_klein_transformer,
)

_SMALL_CONFIG = {
    "patch_size": 1,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 1,
    "joint_attention_dim": 8,
    "axes_dims_rope": (2, 2, 2, 2),
    "rope_theta": 2000,
    "timestep_guidance_channels": 4,
    "guidance_embeds": False,
    "mlp_ratio": 2.0,
    "eps": 1e-6,
}


def _small_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    skeleton = build_klein_transformer_skeleton(_SMALL_CONFIG)
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for target, value in skeleton.state_dict().items():
        source = comfy_flux2_source_for_target(target)
        assert source is not None, target
        grouped.setdefault(source, []).append((target, value))

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, str]] = {}
    for source, targets in grouped.items():
        ordered = sorted(targets, key=lambda item: map_comfy_flux2_parameter(source).index(item[0]))
        shape = tuple(ordered[0][1].shape)
        quantized = (
            source.startswith(("double_blocks.", "single_blocks."))
            and source.endswith(".weight")
            and len(shape) == 2
            and ".norm." not in source
        )
        if len(ordered) == 1:
            source_shape = shape
        else:
            source_shape = (sum(item[1].shape[0] for item in ordered), shape[1])
        if quantized:
            tensors[source] = torch.zeros(source_shape, dtype=torch.float8_e4m3fn)
            stem = source.removesuffix(".weight")
            tensors[stem + ".weight_scale"] = torch.tensor(0.25, dtype=torch.float32)
            tensors[stem + ".input_scale"] = torch.tensor(0.5, dtype=torch.float32)
            layers[stem] = {"format": "float8_e4m3fn"}
        else:
            tensors[source] = torch.zeros(source_shape, dtype=torch.bfloat16)
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    return tensors, metadata


def test_complete_klein_fp8_header_maps_exact_diffusers_shell(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)

    assert plan.available
    assert plan.errors == ()
    assert plan.artifact_contract == KLEIN_STORED_FP8_CONTRACT
    assert len(plan.quantized_sources) == 10
    assert len(plan.source_to_targets) == len(tensors) - 20
    assert len(
        {target for targets in plan.source_to_targets.values() for target in targets}
    ) == len(build_klein_transformer_skeleton(_SMALL_CONFIG).state_dict())


def test_complete_klein_fp8_materializer_preserves_qdata_scales_and_adaln_order(
    tmp_path: Path,
):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    fused_key = "double_blocks.0.img_attn.qkv.weight"
    fused = tensors[fused_key]
    fused[:8] = 1
    fused[8:16] = 2
    fused[16:] = 3
    adaln_key = "final_layer.adaLN_modulation.1.weight"
    tensors[adaln_key][:8] = 4
    tensors[adaln_key][8:] = 5
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)
    transformer = materialize_klein_transformer(plan, _SMALL_CONFIG)

    q = transformer.transformer_blocks[0].attn.to_q
    k = transformer.transformer_blocks[0].attn.to_k
    v = transformer.transformer_blocks[0].attn.to_v
    assert all(isinstance(layer, KleinStoredLinear) for layer in (q, k, v))
    assert all(layer.weight._qdata.dtype == torch.float8_e4m3fn for layer in (q, k, v))
    assert torch.equal(q.weight._qdata.float(), torch.ones((8, 8)))
    assert torch.equal(k.weight._qdata.float(), torch.full((8, 8), 2.0))
    assert torch.equal(v.weight._qdata.float(), torch.full((8, 8), 3.0))
    assert tuple(layer.weight.params.scale.item() for layer in (q, k, v)) == (
        0.25,
        0.25,
        0.25,
    )
    assert (q.input_scale, k.input_scale, v.input_scale) == (0.5, 0.5, 0.5)
    assert torch.equal(
        transformer.norm_out.linear.weight[:8],
        torch.full((8, 8), 5.0, dtype=torch.bfloat16),
    )
    assert torch.equal(
        transformer.norm_out.linear.weight[8:],
        torch.full((8, 8), 4.0, dtype=torch.bfloat16),
    )
    assert not any(parameter.is_meta for parameter in transformer.parameters())
    assert transformer._latentslate_klein_artifact_identity == plan.identity


def test_complete_small_klein_fp8_transformer_runs_forward(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)
    transformer = materialize_klein_transformer(
        plan_comfy_klein_transformer(path, _SMALL_CONFIG),
        _SMALL_CONFIG,
    )

    with torch.no_grad():
        output = transformer(
            hidden_states=torch.zeros((1, 2, 4), dtype=torch.bfloat16),
            encoder_hidden_states=torch.zeros((1, 1, 8), dtype=torch.bfloat16),
            timestep=torch.zeros((1,), dtype=torch.bfloat16),
            img_ids=torch.zeros((2, 4), dtype=torch.float32),
            txt_ids=torch.zeros((1, 4), dtype=torch.float32),
            return_dict=False,
        )[0]

    assert output.shape == (1, 2, 4)
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output).all())


def _materialized_small_transformer(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    quant_index = 0
    for key, tensor in tensors.items():
        if tensor.dtype is torch.float8_e4m3fn and key.endswith(".weight"):
            tensor.fill_(float((quant_index % 4) + 1))
            stem = key.removesuffix(".weight")
            tensors[stem + ".weight_scale"].fill_(0.125 * ((quant_index % 3) + 1))
            quant_index += 1
        elif tensor.dtype is torch.bfloat16:
            tensor.fill_(0.125)
    save_file(tensors, path, metadata=metadata)
    return materialize_klein_transformer(
        plan_comfy_klein_transformer(path, _SMALL_CONFIG),
        _SMALL_CONFIG,
    )


def _stored_snapshot(transformer):
    stored = [module for module in transformer.modules() if isinstance(module, KleinStoredLinear)]
    return (
        stored,
        [module.weight._qdata.detach().clone() for module in stored],
        [module.weight.params.scale.detach().clone() for module in stored],
        {
            name: value.detach().clone()
            for name, value in transformer.named_parameters()
            if not hasattr(value, "_qdata")
        },
    )


def _small_forward(transformer, *, device: torch.device | None = None):
    device = device or torch.device("cpu")
    with torch.no_grad():
        return transformer(
            hidden_states=torch.zeros((1, 2, 4), dtype=torch.bfloat16, device=device),
            encoder_hidden_states=torch.zeros((1, 1, 8), dtype=torch.bfloat16, device=device),
            timestep=torch.zeros((1,), dtype=torch.bfloat16, device=device),
            img_ids=torch.zeros((2, 4), dtype=torch.float32, device=device),
            txt_ids=torch.zeros((1, 4), dtype=torch.float32, device=device),
            return_dict=False,
        )[0]


def _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before):
    assert all(module.weight._qdata.device.type == "cpu" for module in stored)
    assert all(module.weight.params.scale.device.type == "cpu" for module in stored)
    assert all(
        torch.equal(module.weight._qdata, expected)
        for module, expected in zip(stored, qdata_before, strict=True)
    )
    assert all(
        torch.equal(module.weight.params.scale, expected)
        for module, expected in zip(stored, scales_before, strict=True)
    )
    assert all(
        torch.equal(dict(transformer.named_parameters())[name], expected)
        for name, expected in dense_before.items()
    )


def test_klein_residency_cpu_lifecycle_tracks_outer_forward_and_preserves_storage(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    session = KleinTransformerResidencySession(transformer, onload_device="cpu")

    with session:
        assert session.active
        assert session.device == torch.device("cpu")
        output = _small_forward(transformer)
        assert output.device.type == "cpu"
        assert bool(torch.isfinite(output).all())

    assert not session.active
    _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before)
    with pytest.raises(RuntimeError, match="one-shot"):
        session.__enter__()


def test_klein_residency_base_exception_exit_returns_storage_to_cpu(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    session = KleinTransformerResidencySession(transformer, onload_device="cpu")

    with pytest.raises(BaseException, match="synthetic abort"), session:
        raise KeyboardInterrupt("synthetic abort")

    assert not session.active
    _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before)


def test_klein_residency_process_guard_and_cross_thread_close_rejection(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    first = KleinTransformerResidencySession(transformer, onload_device="cpu")
    second = KleinTransformerResidencySession(transformer, onload_device="cpu")

    with first:
        with pytest.raises(RuntimeError, match="already active process-wide"):
            second.__enter__()
        errors: list[BaseException] = []

        def close_from_other_thread() -> None:
            try:
                first.close()
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        worker = threading.Thread(target=close_from_other_thread)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert "owning context thread" in str(errors[0])
        assert first.active

    with second:
        assert second.active


def test_klein_lazy_residency_onloads_only_at_first_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transformer = _materialized_small_transformer(tmp_path)
    calls: list[torch.device] = []
    original_move = adapter.move_klein_transformer_storage

    def record_move(module, device):
        calls.append(torch.device(device))
        return original_move(module, device)

    monkeypatch.setattr(adapter, "move_klein_transformer_storage", record_move)
    session = KleinTransformerResidencySession(
        transformer,
        onload_device="cpu",
        lazy_onload=True,
    )

    with session:
        assert calls == []
        output = _small_forward(transformer)
        assert bool(torch.isfinite(output).all())
        assert calls == [torch.device("cpu")]

    assert calls == [torch.device("cpu"), torch.device("cpu")]


def test_klein_residency_rejects_close_while_outer_forward_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transformer = _materialized_small_transformer(tmp_path)
    session = KleinTransformerResidencySession(transformer, onload_device="cpu")
    original_forward = transformer.forward
    close_errors: list[BaseException] = []

    def forward_that_attempts_close(*args, **kwargs):
        try:
            session.close()
        except BaseException as exc:  # noqa: BLE001 - asserted below
            close_errors.append(exc)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(transformer, "forward", forward_that_attempts_close)
    with session:
        output = _small_forward(transformer)
        assert bool(torch.isfinite(output).all())
        assert session.active

    assert len(close_errors) == 1
    assert "while a forward is active" in str(close_errors[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_fp8_storage_moves_to_exact_cuda_device_and_back(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    quant_index = 0
    for key, tensor in tensors.items():
        if tensor.dtype is torch.float8_e4m3fn and key.endswith(".weight"):
            tensor.fill_(float((quant_index % 4) + 1))
            stem = key.removesuffix(".weight")
            tensors[stem + ".weight_scale"].fill_(0.125 * ((quant_index % 3) + 1))
            quant_index += 1
        elif tensor.dtype is torch.bfloat16:
            tensor.fill_(0.125)
    save_file(tensors, path, metadata=metadata)
    transformer = materialize_klein_transformer(
        plan_comfy_klein_transformer(path, _SMALL_CONFIG),
        _SMALL_CONFIG,
    )
    target = torch.device("cuda", torch.cuda.current_device())
    stored = [module for module in transformer.modules() if isinstance(module, KleinStoredLinear)]
    qdata_before = [module.weight._qdata.detach().clone() for module in stored]
    scales_before = [module.weight.params.scale.detach().clone() for module in stored]
    dense_before = {
        name: parameter.detach().clone()
        for name, parameter in transformer.named_parameters()
        if not hasattr(parameter, "_qdata")
    }

    move_klein_transformer_storage(transformer, torch.device("cuda"))
    assert stored
    assert all(module.weight._qdata.device == target for module in stored)
    assert all(module.weight.params.scale.device == target for module in stored)
    assert all(module.weight._qdata.dtype is torch.float8_e4m3fn for module in stored)
    assert all(
        torch.equal(module.weight._qdata.cpu(), expected)
        for module, expected in zip(stored, qdata_before, strict=True)
    )
    assert all(
        torch.equal(module.weight.params.scale.cpu(), expected)
        for module, expected in zip(stored, scales_before, strict=True)
    )
    with torch.no_grad():
        output = transformer(
            hidden_states=torch.zeros((1, 2, 4), dtype=torch.bfloat16, device=target),
            encoder_hidden_states=torch.zeros((1, 1, 8), dtype=torch.bfloat16, device=target),
            timestep=torch.zeros((1,), dtype=torch.bfloat16, device=target),
            img_ids=torch.zeros((2, 4), dtype=torch.float32, device=target),
            txt_ids=torch.zeros((1, 4), dtype=torch.float32, device=target),
            return_dict=False,
        )[0]
    assert output.device == target
    assert bool(torch.isfinite(output).all())

    torch.cuda.synchronize(target)
    move_klein_transformer_storage(transformer, "cpu")
    assert all(module.weight._qdata.device.type == "cpu" for module in stored)
    assert all(module.weight.params.scale.device.type == "cpu" for module in stored)
    assert all(
        torch.equal(module.weight._qdata, expected)
        for module, expected in zip(stored, qdata_before, strict=True)
    )
    assert all(
        torch.equal(module.weight.params.scale, expected)
        for module, expected in zip(stored, scales_before, strict=True)
    )
    assert all(
        torch.equal(dict(transformer.named_parameters())[name], expected)
        for name, expected in dense_before.items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_residency_uses_canonical_cuda_ordinal_and_restores_exact_storage(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    target = torch.device("cuda", torch.cuda.current_device())
    session = KleinTransformerResidencySession(transformer, onload_device=torch.device("cuda"))

    with session:
        assert session.device == target
        assert all(module.weight._qdata.device == target for module in stored)
        assert all(module.weight.params.scale.device == target for module in stored)
        output = _small_forward(transformer, device=target)
        assert output.device == target
        assert bool(torch.isfinite(output).all())

    _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_residency_cuda_barrier_failure_poisoned_without_cpu_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transformer = _materialized_small_transformer(tmp_path)
    stored, _, _, _ = _stored_snapshot(transformer)
    target = torch.device("cuda", torch.cuda.current_device())
    session = KleinTransformerResidencySession(transformer, onload_device=target)

    def fail_barrier(_device: torch.device) -> None:
        raise RuntimeError("synthetic CUDA barrier loss")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_barrier)
    with pytest.raises(RuntimeError, match="teardown failed"), session:
        assert all(module.weight._qdata.device == target for module in stored)

    assert not session.active
    assert "barrier failed" in transformer._latentslate_klein_residency_poisoned
    # The failed barrier deliberately leaves the original CUDA allocations in
    # place rather than rebuilding wrapper storage on CPU while kernels may run.
    assert all(module.weight._qdata.device == target for module in stored)
    assert all(module.weight.params.scale.device == target for module in stored)


def test_klein_materializer_opens_exactly_one_safetensors_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)
    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)

    import safetensors

    original = safetensors.safe_open
    calls = 0

    def counted_safe_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counted_safe_open)
    materialize_klein_transformer(plan, _SMALL_CONFIG)

    assert calls == 1


def test_klein_materializer_rejects_replaced_artifact(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)
    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)
    tensors["unknown.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    path.unlink()
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="artifact identity changed"):
        materialize_klein_transformer(plan, _SMALL_CONFIG)


def test_klein_materializer_rejects_forged_orphan_auxiliary_plan(tmp_path: Path):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)
    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)
    forged = replace(
        plan,
        auxiliary_sources=plan.auxiliary_sources + ("orphan.input_scale",),
    )

    with pytest.raises(ValueError, match="quant auxiliary roles differ"):
        materialize_klein_transformer(forged, _SMALL_CONFIG)


def test_klein_materializer_late_failure_releases_partial_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "klein-fp8.safetensors"
    tensors, metadata = _small_checkpoint()
    save_file(tensors, path, metadata=metadata)
    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)
    original_build = adapter.build_klein_transformer_skeleton
    original_assign = adapter._assign_dense_target
    original_restore = adapter._restore_global_fp8_tensor
    captured = []
    restored_weights: list[weakref.ReferenceType] = []
    assigned = 0

    def capture_build(config):
        transformer = original_build(config)
        captured.append(transformer)
        return transformer

    def fail_after_one_dense(root, target, tensor):
        nonlocal assigned
        original_assign(root, target, tensor)
        assigned += 1
        if assigned == 1:
            raise RuntimeError("synthetic late failure")

    def capture_restore(*args, **kwargs):
        weight = original_restore(*args, **kwargs)
        restored_weights.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(adapter, "build_klein_transformer_skeleton", capture_build)
    monkeypatch.setattr(adapter, "_assign_dense_target", fail_after_one_dense)
    monkeypatch.setattr(adapter, "_restore_global_fp8_tensor", capture_restore)

    with pytest.raises(RuntimeError, match="synthetic late failure"):
        materialize_klein_transformer(plan, _SMALL_CONFIG)

    assert len(captured) == 1
    assert all(parameter.is_meta for parameter in captured[0].parameters())
    assert not hasattr(captured[0], "_latentslate_klein_artifact_identity")
    gc.collect()
    assert restored_weights and all(reference() is None for reference in restored_weights)


def test_klein_fp8_plan_requires_exact_global_layer_metadata(tmp_path: Path):
    path = tmp_path / "missing-layer.safetensors"
    tensors, metadata = _small_checkpoint()
    parsed = json.loads(metadata["_quantization_metadata"])
    parsed["layers"].pop(next(iter(parsed["layers"])))
    metadata["_quantization_metadata"] = json.dumps(parsed)
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)

    assert not plan.available
    assert plan.artifact_contract is None
    assert "global FP8 metadata does not exactly match quantized layers" in plan.errors


def test_klein_fp8_plan_rejects_dense_quant_payload(tmp_path: Path):
    path = tmp_path / "wrong-dense-dtype.safetensors"
    tensors, metadata = _small_checkpoint()
    tensors["img_in.weight"] = tensors["img_in.weight"].to(torch.float16)
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)

    assert not plan.available
    assert any(error.startswith("dense source must remain BF16") for error in plan.errors)


def test_klein_fp8_plan_rejects_orphan_quant_auxiliary(tmp_path: Path):
    path = tmp_path / "orphan-sidecar.safetensors"
    tensors, metadata = _small_checkpoint()
    tensors["unknown.weight_scale"] = torch.tensor(1.0, dtype=torch.float32)
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)

    assert not plan.available
    assert "unknown.weight_scale" in plan.unexpected_sources
