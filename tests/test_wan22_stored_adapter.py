from __future__ import annotations

import asyncio
import gc
import json
import os
import threading
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import latentslate_engine.runtime.wan22_stored_adapter as adapter
from latentslate_engine.runtime.wan22_stored_adapter import (
    NativeStoredLinear,
    StoredPrecisionConv3d,
    SynchronousBlockResidencyManager,
    WanTransformerResidencySession,
    attach_native_stored_linear,
    build_wan_transformer_skeleton,
    comfy_source_key_for_diffusers_parameter,
    map_comfy_wan_parameter_key,
    materialize_wan_transformer,
    plan_comfy_wan_transformer,
    plan_wan_root_residency,
    validate_stored_quant_offload_mode,
)

_SMALL_WAN_CONFIG = {
    "patch_size": (1, 1, 1),
    "num_attention_heads": 1,
    # Kitchen's CUDA FP8 kernel requires all stored-linear contraction
    # dimensions to be divisible by 16.  These remain synthetic, one-block
    # fixtures; only their tensor-core dimensions are widened.
    "attention_head_dim": 16,
    "in_channels": 4,
    "out_channels": 16,
    "text_dim": 16,
    "freq_dim": 16,
    "ffn_dim": 16,
    "num_layers": 1,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 8,
    "pos_embed_seq_len": None,
}


class _RecordingBlock(torch.nn.Module):
    def __init__(self, *, fail_forward: bool = False, reenter: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.moves: list[str] = []
        self.fail_forward = fail_forward
        self.reenter = reenter
        self.on_forward = None

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(torch.device(device)))
        return super().to(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.on_forward is not None:
            self.on_forward()
        if self.fail_forward:
            raise RuntimeError("intentional block failure")
        if self.reenter:
            self.reenter = False
            return self(input)
        return input * self.weight


def _fp8_weight():
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    qdata = torch.zeros((16, 16), dtype=torch.float8_e4m3fn)
    qdata[0, :2] = torch.tensor([1, 2], dtype=torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(scale=torch.tensor(0.5), orig_dtype=torch.float32, orig_shape=(16, 16))
    return QuantizedTensor(qdata, "TensorCoreFP8Layout", params)


def _int8_weight():
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    qdata = torch.tensor([[2, -2, 0, 0]], dtype=torch.int8)
    params = TensorWiseINT8Layout.Params(
        scale=torch.tensor([[0.25]]),
        orig_dtype=torch.float32,
        orig_shape=(1, 4),
        is_weight=True,
        convrot=True,
        convrot_groupsize=4,
    )
    return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)


def _marker(value: dict[str, object]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value).encode("utf-8")), dtype=torch.uint8)


def _write_complete_small_wan_checkpoint(path: Path, contract: str) -> None:
    skeleton = build_wan_transformer_skeleton(_SMALL_WAN_CONFIG)
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    for target, value in skeleton.state_dict().items():
        source = comfy_source_key_for_diffusers_parameter(target)
        assert source is not None, target
        shape = tuple(value.shape)
        parent_path, separator, _ = target.rpartition(".")
        parent = skeleton.get_submodule(parent_path) if separator else skeleton
        quantized = (
            source.endswith(".weight")
            and isinstance(parent, torch.nn.Linear)
            and (contract != "comfy_quant/int8_tensorwise_convrot" or source.startswith("blocks."))
        )
        if not quantized:
            patch_embedding = target in {"patch_embedding.weight", "patch_embedding.bias"}
            dtype = (
                torch.float32
                if contract == "comfy_quant/float8_e4m3fn" and patch_embedding
                else torch.float16
            )
            tensors[source] = torch.zeros(shape, dtype=dtype)
            continue
        stem = source.removesuffix(".weight")
        if contract == "comfy_quant/int8_tensorwise_convrot":
            tensors[source] = torch.zeros(shape, dtype=torch.int8)
            tensors[stem + ".weight_scale"] = torch.full((shape[0], 1), 0.25, dtype=torch.float32)
            tensors[stem + ".comfy_quant"] = _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": shape[1], "per_row": True}
            )
            layers[stem] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": shape[1]}
        else:
            tensors[source] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
            suffix = ".scale_weight" if contract == "comfy_legacy/scaled_fp8_e4m3fn" else ".weight_scale"
            tensors[stem + suffix] = torch.tensor(0.25, dtype=torch.float32)
            if contract == "comfy_legacy/scaled_fp8_e4m3fn":
                tensors[stem + ".scale_input"] = torch.tensor(0.25, dtype=torch.float32)
            else:
                tensors[stem + ".comfy_quant"] = _marker({"format": "float8_e4m3fn"})
    if contract == "comfy_legacy/scaled_fp8_e4m3fn":
        tensors["scaled_fp8"] = torch.tensor([1], dtype=torch.uint8)
    metadata = {"_quantization_metadata": json.dumps({"layers": layers})} if layers else None
    save_file(tensors, path, metadata=metadata)


@pytest.mark.parametrize(
    "contract",
    ["comfy_quant/float8_e4m3fn", "comfy_legacy/scaled_fp8_e4m3fn", "comfy_quant/int8_tensorwise_convrot"],
)
def test_complete_small_wan_materializer_restores_stored_linear_contracts(tmp_path: Path, contract: str):
    path = tmp_path / f"{contract.rsplit('/', 1)[-1]}.safetensors"
    _write_complete_small_wan_checkpoint(path, contract)
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)

    transformer = materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)

    assert not any(parameter.is_meta for parameter in transformer.parameters())
    assert not any(buffer.is_meta for buffer in transformer.buffers())
    assert isinstance(transformer.blocks[0].attn1.to_q, NativeStoredLinear)
    assert isinstance(transformer.proj_out, NativeStoredLinear) == (contract != "comfy_quant/int8_tensorwise_convrot")
    assert sum(isinstance(module, NativeStoredLinear) for module in transformer.blocks[0].modules()) == 10
    assert not any(isinstance(module, torch.nn.Linear) for module in transformer.blocks[0].modules())
    assert transformer.patch_embedding.weight.dtype == (
        torch.float32 if contract == "comfy_quant/float8_e4m3fn" else torch.float16
    )
    assert isinstance(transformer.patch_embedding, StoredPrecisionConv3d) == (
        contract == "comfy_quant/float8_e4m3fn"
    )
    patch_output = transformer.patch_embedding(torch.zeros((1, 4, 1, 1, 1), dtype=torch.float16))
    assert patch_output.dtype == torch.float16
    assert torch.isfinite(
        transformer.blocks[0].attn1.to_q(torch.zeros((1, 1, 16), dtype=torch.float16))
    ).all()
    output = transformer(
        torch.zeros((1, 4, 1, 1, 1), dtype=torch.float16),
        torch.tensor([1]),
        torch.zeros((1, 1, 16), dtype=torch.float16),
        return_dict=False,
    )[0]
    assert output.shape == (1, 16, 1, 1, 1)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    if contract == "comfy_legacy/scaled_fp8_e4m3fn":
        assert transformer.blocks[0].attn1.to_q.input_scale == 0.25
    roots = plan_wan_root_residency(transformer)
    assert roots.root_components == ("rope", "patch_embedding", "condition_embedder", "norm_out", "proj_out")
    assert roots.blocks == ("blocks.0",)
    assert {"scale_shift_table", "rope.freqs_cos", "rope.freqs_sin"} <= set(roots.root_state)
    classified = set(roots.root_state)
    classified.update(name for names in roots.block_state.values() for name in names)
    all_state = set(dict(transformer.named_parameters())) | set(dict(transformer.named_buffers()))
    assert classified == all_state


def test_official_legacy_top_level_proj_out_mapping_materializes(tmp_path: Path):
    path = tmp_path / "official-legacy-layout.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_legacy/scaled_fp8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)

    assert plan.source_to_target["head.head.weight"] == "proj_out.weight"
    transformer = materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)

    assert isinstance(transformer.proj_out, NativeStoredLinear)


def _small_wan_inputs(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((1, 4, 1, 1, 1), dtype=torch.float16, device=device),
        torch.tensor([1], device=device),
        torch.zeros((1, 1, 16), dtype=torch.float16, device=device),
    )


def test_wan_transformer_residency_session_runs_complete_forward_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "residency-current.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    plan = plan_wan_root_residency(transformer)
    session = WanTransformerResidencySession(transformer, plan, onload_device="cpu")
    monkeypatch.setattr(
        transformer,
        "to",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("whole-transformer move")),
    )

    with session:
        output = transformer(*_small_wan_inputs(), return_dict=False)[0]
        assert output.device.type == "cpu"
        assert session.active
        assert session._block_residency.attached
        assert transformer.scale_shift_table.device.type == "cpu"
        assert transformer.rope.freqs_cos.device.type == "cpu"
        assert transformer.rope.freqs_sin.device.type == "cpu"

    assert not session.active
    assert not session._block_residency.attached
    assert all(value.device.type == "cpu" for value in dict(transformer.named_parameters()).values())
    assert all(value.device.type == "cpu" for value in dict(transformer.named_buffers()).values())
    root_text_linear = transformer.condition_embedder.text_embedder.linear_1
    assert root_text_linear.weight._qdata.device.type == "cpu"
    assert root_text_linear.weight.params.scale.device.type == "cpu"
    assert root_text_linear.bias.device.type == "cpu"
    assert transformer.patch_embedding.weight.dtype == torch.float32
    assert transformer.patch_embedding.bias.dtype == torch.float32


def test_wan_transformer_residency_session_cancellation_cleans_roots_and_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "residency-cancel.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    session = WanTransformerResidencySession(transformer, plan_wan_root_residency(transformer), onload_device="cpu")
    monkeypatch.setattr(
        transformer.blocks[0],
        "forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError), session:
        transformer(*_small_wan_inputs(), return_dict=False)

    assert not session.active
    assert not session._block_residency.attached
    assert all(value.device.type == "cpu" for value in dict(transformer.named_parameters()).values())
    assert all(value.device.type == "cpu" for value in dict(transformer.named_buffers()).values())


def test_wan_transformer_residency_poisoned_when_cuda_barrier_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "residency-poison.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG),
        _SMALL_WAN_CONFIG,
        compute_dtype=torch.float16,
    )
    plan = plan_wan_root_residency(transformer)
    session = WanTransformerResidencySession(transformer, plan, onload_device="cpu")
    session.__enter__()
    session.onload_device = torch.device("cuda:0")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("illegal access")),
    )

    with pytest.raises(RuntimeError, match="teardown failed"):
        session.close()

    assert not session.active
    assert not session._block_residency.attached
    assert "barrier failed" in transformer._latentslate_residency_poisoned
    with pytest.raises(RuntimeError, match="residency is poisoned"):
        WanTransformerResidencySession(transformer, plan, onload_device="cpu")


def test_wan_transformer_residency_session_rejects_concurrent_or_reentrant_use(tmp_path: Path):
    path = tmp_path / "residency-guard.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    plan = plan_wan_root_residency(transformer)
    first = WanTransformerResidencySession(transformer, plan, onload_device="cpu")
    second = WanTransformerResidencySession(transformer, plan, onload_device="cpu")

    with first:
        with pytest.raises(RuntimeError, match="already active process-wide"):
            second.__enter__()
        with pytest.raises(RuntimeError, match="cannot be re-entered"):
            first.__enter__()


def test_wan_transformer_residency_session_rejects_stale_or_duplicate_block_plan(tmp_path: Path):
    path = tmp_path / "residency-plan.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    plan = plan_wan_root_residency(transformer)

    with pytest.raises(ValueError, match="duplicate blocks"):
        WanTransformerResidencySession(transformer, replace(plan, blocks=("blocks.0", "blocks.0")), onload_device="cpu")
    with pytest.raises(ValueError, match="stale or incomplete"):
        WanTransformerResidencySession(transformer, replace(plan, block_state={"blocks.0": ()}), onload_device="cpu")


def test_wan_transformer_residency_session_rejects_cross_thread_close_during_active_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "residency-thread-close.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    session = WanTransformerResidencySession(transformer, plan_wan_root_residency(transformer), onload_device="cpu")
    started = threading.Event()
    release = threading.Event()
    worker_errors: list[BaseException] = []
    original_forward = transformer.blocks[0].forward

    def paused_forward(*args, **kwargs):
        started.set()
        if not release.wait(timeout=10):
            raise RuntimeError("test forward release timed out")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(transformer.blocks[0], "forward", paused_forward)

    def run_forward() -> None:
        try:
            with session:
                transformer(*_small_wan_inputs(), return_dict=False)
        except BaseException as exc:  # noqa: BLE001 - record thread failure for the assertion below
            worker_errors.append(exc)

    worker = threading.Thread(target=run_forward)
    worker.start()
    try:
        assert started.wait(timeout=10)
        with pytest.raises(RuntimeError, match="owning context thread"):
            session.close()
        assert session.active
        assert session._block_residency.attached
    finally:
        release.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert not worker_errors
    assert not session.active
    assert all(value.device.type == "cpu" for value in dict(transformer.named_parameters()).values())


def test_wan_transformer_residency_session_is_process_wide_across_transformers(tmp_path: Path):
    first_path = tmp_path / "residency-first.safetensors"
    second_path = tmp_path / "residency-second.safetensors"
    _write_complete_small_wan_checkpoint(first_path, "comfy_quant/float8_e4m3fn")
    _write_complete_small_wan_checkpoint(second_path, "comfy_quant/float8_e4m3fn")
    first_transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(first_path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    second_transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(second_path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    first = WanTransformerResidencySession(first_transformer, plan_wan_root_residency(first_transformer), onload_device="cpu")
    second = WanTransformerResidencySession(second_transformer, plan_wan_root_residency(second_transformer), onload_device="cpu")

    with first, pytest.raises(RuntimeError, match="process-wide"):
        second.__enter__()


def test_materializer_rejects_plan_artifact_replacement(tmp_path: Path):
    path = tmp_path / "replacement.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    save_file({"opaque": torch.tensor([1], dtype=torch.uint8)}, path)

    with pytest.raises(ValueError):
        materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)


def test_materializer_opens_one_safetensors_tensor_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "one-handle.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    import safetensors

    original = safetensors.safe_open
    opens = 0

    def counted_safe_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counted_safe_open)
    materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)

    assert opens == 1


def test_materializer_derives_descriptors_from_bound_handle_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "bound-handle.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    from latentslate_engine import stored_quant

    monkeypatch.setattr(
        stored_quant,
        "probe_artifact",
        lambda _path: (_ for _ in ()).throw(AssertionError("unbound probe during materialization")),
    )
    materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)


def test_materializer_rejects_identity_swap_during_descriptor_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "descriptor-swap.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    original_revalidate = adapter.revalidate_artifact
    calls = 0

    def swapped_revalidate(identity):
        nonlocal calls
        calls += 1
        return original_revalidate(identity) if calls == 1 else False

    monkeypatch.setattr(adapter, "revalidate_artifact", swapped_revalidate)
    with pytest.raises(ValueError, match="changed during quant descriptor discovery"):
        materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)


def test_materializer_dematerializes_partial_model_after_late_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "cleanup.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    original_build = adapter.build_wan_transformer_skeleton
    original_restore = adapter.restore_stored_quantized_tensor
    observed: list[torch.nn.Module] = []
    restored_weights: list[weakref.ReferenceType] = []

    def capture_build(config):
        transformer = original_build(config)
        observed.append(transformer)
        return transformer

    monkeypatch.setattr(adapter, "build_wan_transformer_skeleton", capture_build)

    def capture_restore(*args, **kwargs):
        weight = original_restore(*args, **kwargs)
        restored_weights.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(adapter, "restore_stored_quantized_tensor", capture_restore)
    monkeypatch.setattr(
        adapter,
        "_assign_dense_target",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("late dense assignment failure")),
    )
    with pytest.raises(RuntimeError, match="late dense assignment failure"):
        materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)

    assert len(observed) == 1
    assert all(parameter.is_meta for parameter in observed[0].parameters())
    assert all(buffer.is_meta for buffer in observed[0].buffers())
    gc.collect()
    assert restored_weights and all(reference() is None for reference in restored_weights)


def test_materializer_rejects_duplicate_or_missing_bias_plan_entries(tmp_path: Path):
    path = tmp_path / "invalid-plan.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    duplicate = replace(plan, duplicate_targets=("blocks.0.attn1.to_q.weight",))
    with pytest.raises(ValueError, match="duplicate"):
        materialize_wan_transformer(duplicate, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)

    missing_bias_map = dict(plan.source_to_target)
    del missing_bias_map["blocks.0.self_attn.q.bias"]
    missing_bias = replace(plan, source_to_target=missing_bias_map)
    with pytest.raises(ValueError, match="plan targets"):
        materialize_wan_transformer(missing_bias, _SMALL_WAN_CONFIG, compute_dtype=torch.float16)


def test_materializer_rejects_same_shape_different_behavior_config(tmp_path: Path):
    path = tmp_path / "different-rope-behavior.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)
    changed = {**_SMALL_WAN_CONFIG, "rope_max_seq_len": 16}

    with pytest.raises(ValueError, match="config does not match"):
        materialize_wan_transformer(plan, changed, compute_dtype=torch.float16)


def test_materializer_requires_authoritative_dense_compute_dtype(tmp_path: Path):
    path = tmp_path / "authoritative-dtype.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)

    with pytest.raises(ValueError, match="compute dtype must exactly match"):
        materialize_wan_transformer(plan, _SMALL_WAN_CONFIG, compute_dtype=torch.float32)


@pytest.mark.parametrize("unexpected_dtype", [torch.bfloat16, torch.float32])
def test_materializer_rejects_unapproved_current_fp8_dense_precision(
    tmp_path: Path, unexpected_dtype: torch.dtype
):
    path = tmp_path / "mixed-dense-dtypes.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    # F32 is reserved for patch_embedding weight+bias; BF16 is never accepted
    # by the currently proven SmoothMix stored artifact contract.
    tensors["head.modulation"] = tensors["head.modulation"].to(unexpected_dtype)
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="stored dense precision contract mismatch"):
        materialize_wan_transformer(
            plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
        )


def test_plan_rejects_current_fp8_sidecars_on_f32_patch_embedding(tmp_path: Path):
    path = tmp_path / "current-fp8-quantized-patch.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors["patch_embedding.weight_scale"] = torch.tensor(0.25, dtype=torch.float32)
    tensors["patch_embedding.comfy_quant"] = _marker({"format": "float8_e4m3fn"})
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG)

    assert not plan.available
    assert plan.dense_precision_contract is None
    assert plan.invalid_non_linear_quant_sources == ("patch_embedding.weight",)
    assert any("patch embedding weight and bias must be explicit unquantized dense sources" in error for error in plan.errors)
    assert any("non-linear weights" in error for error in plan.errors)


def test_materializer_rejects_input_scale_on_current_fp8(tmp_path: Path):
    path = tmp_path / "current-with-legacy-input-scale.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors["blocks.0.self_attn.q.scale_input"] = torch.tensor(0.25, dtype=torch.float32)
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="only supported by the legacy FP8 contract"):
        materialize_wan_transformer(
            plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
        )


def test_materializer_rejects_orphan_quant_sidecar(tmp_path: Path):
    path = tmp_path / "orphan-sidecar.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
        metadata = handle.metadata()
    tensors["blocks.0.self_attn.norm_q.scale_input"] = torch.tensor(0.25, dtype=torch.float32)
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="unconsumed quant auxiliaries"):
        materialize_wan_transformer(
            plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
        )


@pytest.mark.parametrize("input_scale", [None, torch.tensor(0.25)], ids=["current", "legacy"])
def test_native_stored_linear_matches_current_and_legacy_fp8_cpu_reference(input_scale: torch.Tensor | None):
    bias = torch.zeros(16)
    bias[0] = 0.25
    linear = NativeStoredLinear(_fp8_weight(), bias=bias, input_scale=input_scale)
    output = linear(torch.tensor([[1.0, 2.0, *([0.0] * 14)]]))

    assert output.shape == (1, 16)
    assert torch.allclose(output[:, :1], torch.tensor([[2.75]]), atol=0.01, rtol=0.01)
    assert torch.equal(output[:, 1:], torch.zeros((1, 15)))


def test_native_stored_linear_runs_convrot_int8_cpu():
    weight = _int8_weight()
    linear = NativeStoredLinear(weight)
    input = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(linear(input), torch.nn.functional.linear(input, weight.dequantize()))


@pytest.mark.parametrize("contract", ["current", "legacy", "convrot"])
def test_native_stored_linear_flattens_rank_three_without_dense_fallback(
    contract: str, monkeypatch: pytest.MonkeyPatch
):
    from comfy_kitchen.tensor import QuantizedTensor

    if contract == "convrot":
        weight = _int8_weight()
        input_scale = None
    else:
        weight = _fp8_weight()
        input_scale = torch.tensor(0.25) if contract == "legacy" else None
    monkeypatch.setattr(
        QuantizedTensor,
        "dequantize",
        lambda _self: (_ for _ in ()).throw(AssertionError("dense fallback")),
    )
    native_calls: list[tuple[torch.Tensor, QuantizedTensor]] = []

    def native_linear(input: torch.Tensor, stored_weight: QuantizedTensor, bias=None) -> torch.Tensor:
        assert isinstance(stored_weight, QuantizedTensor)
        assert bias is None
        native_calls.append((input, stored_weight))
        return torch.zeros(
            (input.shape[0], stored_weight.shape[0]),
            device=input.device,
            dtype=input.dtype,
        )

    monkeypatch.setattr(adapter.F, "linear", native_linear)

    output = NativeStoredLinear(weight, input_scale=input_scale)(torch.ones((2, 3, weight.shape[1])))

    assert output.shape == (2, 3, weight.shape[0])
    assert torch.isfinite(output).all()
    assert len(native_calls) == 1
    assert native_calls[0][0].shape == (6, weight.shape[1])
    assert native_calls[0][1]._qdata.data_ptr() == weight._qdata.data_ptr()


def test_attach_native_stored_linear_registers_parameters_and_preserves_scale_precision():
    parent = torch.nn.Module()
    parent.linear = torch.nn.Linear(16, 16)
    scale = torch.tensor(0.00390625, dtype=torch.float32)
    bias = torch.zeros(16)
    bias[0] = 0.25
    attached = attach_native_stored_linear(parent, "linear", _fp8_weight(), bias, scale)

    assert parent.linear is attached
    assert {"weight", "bias"} <= set(dict(attached.named_parameters()))
    assert not dict(attached.named_buffers())
    assert attached.weight.storage_dtype == torch.float8_e4m3fn
    parent.to(dtype=torch.float16)
    assert attached.weight.dtype == torch.float16
    assert attached.bias.dtype == torch.float16
    assert attached.input_scale == scale.item()


def test_stored_quant_offload_contract_allows_only_block_groups():
    assert validate_stored_quant_offload_mode("group_block") == "group_block"


@pytest.mark.parametrize("mode", ["sequential", "cpu_offload", "meta", "group_leaf", "model", "whole_model", "disk"])
def test_stored_quant_offload_contract_rejects_meta_reconstruction_and_nonblock_modes(mode: str):
    with pytest.raises(ValueError, match="block-level group offload"):
        validate_stored_quant_offload_mode(mode)


def test_engine_owned_block_residency_moves_whole_block_and_keeps_output_device():
    block = _RecordingBlock()
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cpu", offload_device="cpu")
    manager.attach()

    output = block(torch.tensor([2.0]))

    assert output.device.type == "cpu"
    assert block.moves.count("cpu") >= 2
    assert block.moves[-1] == "cpu"
    assert manager.active_block is None
    manager.remove()
    assert not manager.attached
    assert block.moves == ["cpu", "cpu", "cpu"]


def test_engine_owned_block_residency_offloads_after_forward_exception():
    block = _RecordingBlock(fail_forward=True)
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cpu", offload_device="cpu")
    manager.attach()

    with pytest.raises(RuntimeError, match="intentional block failure"):
        block(torch.tensor([2.0]))

    assert block.moves == ["cpu", "cpu"]
    assert manager.active_block is None
    manager.remove()


def test_remove_during_active_forward_preserves_post_offload_then_later_succeeds():
    block = _RecordingBlock()
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cpu", offload_device="cpu")
    manager.attach()
    removal_errors: list[str] = []

    def attempt_remove() -> None:
        with pytest.raises(RuntimeError, match="while a block is active"):
            manager.remove()
        removal_errors.append("rejected")

    block.on_forward = attempt_remove
    output = block(torch.tensor([2.0]))

    assert output.device.type == "cpu"
    assert removal_errors == ["rejected"]
    assert block.moves == ["cpu", "cpu"]
    assert manager.attached
    assert manager.active_block is None
    manager.remove()
    assert not manager.attached


def test_engine_owned_block_residency_preserves_quantized_tensor_storage(monkeypatch: pytest.MonkeyPatch):
    from comfy_kitchen.tensor import QuantizedTensor

    block = NativeStoredLinear(_fp8_weight(), input_scale=torch.tensor(0.25))
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cpu", offload_device="cpu")
    manager.attach()
    monkeypatch.setattr(
        adapter.F,
        "linear",
        lambda input, stored_weight, bias=None: torch.zeros(
            (input.shape[0], stored_weight.shape[0]), device=input.device, dtype=input.dtype
        )
        if isinstance(stored_weight, QuantizedTensor) and bias is None
        else (_ for _ in ()).throw(AssertionError("stored native linear contract")),
    )
    try:
        output = block(torch.tensor([[1.0, 2.0, *([0.0] * 14)]], dtype=torch.float16))
        assert output.device.type == "cpu"
        assert block.weight.storage_dtype == torch.float8_e4m3fn
        assert block.weight._qdata.device.type == "cpu"
        assert block.weight.params.scale.device.type == "cpu"
    finally:
        manager.remove()


def test_engine_owned_block_residency_is_nonreentrant_and_fails_closed():
    block = _RecordingBlock(reenter=True)
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cpu", offload_device="cpu")
    manager.attach()

    with pytest.raises(RuntimeError, match="non-reentrant"):
        block(torch.tensor([2.0]))
    assert manager.active_block is None
    assert block.moves.count("cpu") >= 2
    assert block.moves[-1] == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        block(torch.tensor([2.0]))
    manager.remove()


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF") != "1" or not torch.cuda.is_available(),
    reason="set LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF=1 on a CUDA host",
)
def test_opt_in_cuda_native_stored_linear_block_residency_proof():
    block = NativeStoredLinear(_fp8_weight(), input_scale=torch.tensor(0.25))
    manager = SynchronousBlockResidencyManager({"block.0": block}, onload_device="cuda", offload_device="cpu")
    manager.attach()
    try:
        output = block(torch.tensor([[[1.0, 2.0, *([0.0] * 14)]]], device="cuda"))
        assert output.device.type == "cuda"
        assert output.shape == (1, 1, 16)
        assert block.weight._qdata.device.type == "cpu"
        assert block.weight.params.scale.device.type == "cpu"
    finally:
        manager.remove()


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF") != "1" or not torch.cuda.is_available(),
    reason="set LATENTSLATE_ENGINE_RUN_CUDA_RESIDENCY_PROOF=1 on a CUDA host",
)
def test_opt_in_cuda_full_tiny_wan_residency_session(tmp_path: Path):
    path = tmp_path / "cuda-full-wan.safetensors"
    _write_complete_small_wan_checkpoint(path, "comfy_quant/float8_e4m3fn")
    transformer = materialize_wan_transformer(
        plan_comfy_wan_transformer(path, _SMALL_WAN_CONFIG), _SMALL_WAN_CONFIG, compute_dtype=torch.float16
    )
    session = WanTransformerResidencySession(transformer, plan_wan_root_residency(transformer), onload_device="cuda")

    with session:
        output = transformer(*_small_wan_inputs("cuda"), return_dict=False)[0]
        assert output.device.type == "cuda"
        assert output.shape == (1, 16, 1, 1, 1)
        assert transformer.scale_shift_table.device.type == "cuda"
        assert transformer.rope.freqs_cos.device.type == "cuda"
        root_text_linear = transformer.condition_embedder.text_embedder.linear_1
        assert root_text_linear.weight._qdata.device.type == "cuda"
        assert root_text_linear.weight.params.scale.device.type == "cuda"
        assert root_text_linear.bias.device.type == "cuda"
        assert transformer.patch_embedding.weight.dtype == torch.float32
        assert transformer.patch_embedding.bias.dtype == torch.float32
        assert transformer.patch_embedding.weight.device.type == "cuda"
        assert transformer.patch_embedding.bias.device.type == "cuda"

    assert all(value.device.type == "cpu" for value in dict(transformer.named_parameters()).values())
    assert all(value.device.type == "cpu" for value in dict(transformer.named_buffers()).values())
    assert transformer.proj_out.weight._qdata.device.type == "cpu"
    assert transformer.proj_out.weight.params.scale.device.type == "cpu"
    assert transformer.blocks[0].attn1.to_q.weight._qdata.device.type == "cpu"
    assert transformer.blocks[0].attn1.to_q.weight.params.scale.device.type == "cpu"
    root_text_linear = transformer.condition_embedder.text_embedder.linear_1
    assert root_text_linear.weight._qdata.device.type == "cpu"
    assert root_text_linear.weight.params.scale.device.type == "cpu"
    assert root_text_linear.bias.device.type == "cpu"
    assert transformer.patch_embedding.weight.dtype == torch.float32
    assert transformer.patch_embedding.bias.dtype == torch.float32
    assert transformer.patch_embedding.weight.device.type == "cpu"
    assert transformer.patch_embedding.bias.device.type == "cpu"


@pytest.mark.parametrize("scale", [torch.tensor(0.0), torch.tensor(-0.25), torch.tensor(float("nan"))])
def test_native_stored_linear_rejects_nonpositive_or_nonfinite_input_scale(scale: torch.Tensor):
    with pytest.raises(ValueError, match="positive finite F32 scalar"):
        NativeStoredLinear(_fp8_weight(), input_scale=scale)


def test_comfy_key_mapping_covers_pinned_diffusers_layout():
    assert map_comfy_wan_parameter_key("model.diffusion_model.head.modulation") == "scale_shift_table"
    assert map_comfy_wan_parameter_key("blocks.3.self_attn.o.weight") == "blocks.3.attn1.to_out.0.weight"
    assert map_comfy_wan_parameter_key("blocks.3.cross_attn.norm_k.weight") == "blocks.3.attn2.norm_k.weight"
    assert map_comfy_wan_parameter_key("blocks.3.ffn.0.bias") == "blocks.3.ffn.net.0.proj.bias"
    assert map_comfy_wan_parameter_key("vae.decoder.conv1.weight") is None


def test_residency_cuda_device_is_canonicalized_and_requires_exact_ordinal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    requested = adapter._canonicalize_residency_device(torch.device("cuda"))

    assert requested == torch.device("cuda:0")
    assert adapter._matches_requested_device(torch.device("cuda:0"), requested)
    assert not adapter._matches_requested_device(torch.device("cuda:1"), requested)


def test_meta_skeleton_has_no_allocated_parameters():
    skeleton = build_wan_transformer_skeleton()

    assert all(parameter.is_meta for parameter in skeleton.parameters())
    assert len(skeleton.state_dict()) == 1095


def _force_supported_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    original_probe = adapter.probe_safetensors
    monkeypatch.setattr(
        adapter,
        "probe_safetensors",
        lambda path: replace(original_probe(path), quantization_contract="comfy_quant/float8_e4m3fn"),
    )


def test_plan_fails_closed_for_missing_meta_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "partial.safetensors"
    # The artifact probe validates this payload, but planning only reads its header.
    save_file({"patch_embedding.weight": torch.empty((5120, 36, 1, 2, 2), dtype=torch.float8_e4m3fn)}, path)

    _force_supported_probe(monkeypatch)
    plan = plan_comfy_wan_transformer(path)

    assert not plan.available
    assert "patch_embedding.weight" not in plan.missing_targets
    assert plan.missing_targets
    with pytest.raises(ValueError, match="missing Diffusers parameters"):
        plan.require_available()


def test_plan_reports_shape_mismatch_and_recognized_legacy_auxiliary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "mismatched.safetensors"
    save_file(
        {
            "patch_embedding.weight": torch.empty((1,), dtype=torch.float8_e4m3fn),
            "patch_embedding.scale_input": torch.tensor(0.25),
            "unknown.weight": torch.tensor([1.0]),
        },
        path,
    )

    _force_supported_probe(monkeypatch)
    plan = plan_comfy_wan_transformer(path)

    assert plan.shape_mismatches[0].target_key == "patch_embedding.weight"
    assert "patch_embedding.scale_input" in plan.quant_auxiliary
    assert plan.unexpected_extras == ("unknown.weight",)


def test_plan_rejects_bf16_and_unknown_artifact_contracts(tmp_path: Path):
    bf16 = tmp_path / "bf16.safetensors"
    unknown = tmp_path / "unknown.safetensors"
    save_file({"any.weight": torch.empty((1,), dtype=torch.bfloat16)}, bf16)
    save_file({"opaque": torch.tensor([1], dtype=torch.uint8)}, unknown)

    with pytest.raises(ValueError, match="'native/bf16'"):
        plan_comfy_wan_transformer(bf16)
    with pytest.raises(ValueError, match="None"):
        plan_comfy_wan_transformer(unknown)
