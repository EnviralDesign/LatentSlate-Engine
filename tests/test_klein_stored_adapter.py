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

from latentslate_engine.artifacts import probe_artifact
from latentslate_engine.runtime import klein_stored_adapter as adapter
from latentslate_engine.runtime.klein_stored_adapter import (
    KLEIN9B_CONFIG,
    KLEIN_STORED_FP8_CONTRACT,
    KLEIN_STORED_NVFP4_CONTRACT,
    KleinStoredLinear,
    KleinStoredNVFP4Linear,
    KleinTransformerResidencySession,
    build_klein_transformer_skeleton,
    comfy_flux2_source_for_target,
    map_comfy_flux2_parameter,
    materialize_klein_nvfp4_transformer,
    materialize_klein_transformer,
    move_klein_transformer_storage,
    plan_bfl_klein_nvfp4_transformer,
    plan_comfy_klein_transformer,
)

_SMALL_CONFIG = {
    "patch_size": 1,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    # Kitchen's CUDA FP8 kernel requires the contraction dimension to be a
    # multiple of 16.  Keep the fixture tiny while using a kernel-valid
    # transformer width; the real Klein model is wider still.
    "attention_head_dim": 16,
    "num_attention_heads": 1,
    "joint_attention_dim": 16,
    "axes_dims_rope": (4, 4, 4, 4),
    "rope_theta": 2000,
    "timestep_guidance_channels": 4,
    "guidance_embeds": False,
    "mlp_ratio": 2.0,
    "eps": 1e-6,
}


def test_klein9b_transformer_config_builds_exact_diffusers_shell_on_meta():
    transformer = build_klein_transformer_skeleton(KLEIN9B_CONFIG)

    assert len(transformer.transformer_blocks) == 8
    assert len(transformer.single_transformer_blocks) == 24
    assert transformer.config.num_attention_heads == 32
    assert transformer.config.attention_head_dim == 128
    assert transformer.config.joint_attention_dim == 12_288
    assert all(parameter.is_meta for parameter in transformer.parameters())


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


def _small_nvfp4_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors, _metadata = _small_checkpoint()
    layers: dict[str, dict[str, str]] = {}
    converted: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key.endswith((".weight_scale", ".input_scale")):
            continue
        if value.dtype is torch.float8_e4m3fn and key.endswith(".weight"):
            stem = key.removesuffix(".weight")
            rows, columns = value.shape
            converted[key] = torch.zeros((rows, columns // 2), dtype=torch.uint8)
            converted[stem + ".weight_scale"] = torch.ones(
                (rows, columns // 16), dtype=torch.float8_e4m3fn
            )
            converted[stem + ".weight_scale_2"] = torch.tensor(0.25, dtype=torch.float32)
            converted[stem + ".input_scale"] = torch.tensor(0.5, dtype=torch.float32)
            layers[stem] = {"format": "nvfp4"}
        else:
            converted[key] = value
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    return converted, metadata


def test_complete_klein_nvfp4_header_maps_exact_packed_layout(tmp_path: Path):
    path = tmp_path / "klein-nvfp4.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    save_file(tensors, path, metadata=metadata)

    plan = plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG)

    assert plan.available
    assert probe_artifact(path).quantization_contract == KLEIN_STORED_NVFP4_CONTRACT
    assert plan.artifact_contract == KLEIN_STORED_NVFP4_CONTRACT
    assert len(plan.quantized_sources) == 10
    assert all(source.endswith(".weight") for source in plan.quantized_sources)
    assert len(plan.auxiliary_sources) == 30


def test_klein_nvfp4_header_rejects_wrong_block_scale_shape(tmp_path: Path):
    path = tmp_path / "klein-nvfp4-invalid.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    key = "double_blocks.0.img_attn.proj.weight_scale"
    tensors[key] = torch.ones((16, 2), dtype=torch.float8_e4m3fn)
    save_file(tensors, path, metadata=metadata)

    plan = plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG)

    assert not plan.available
    assert any("invalid .weight_scale" in error for error in plan.contract_errors)


@pytest.mark.parametrize("metadata_value", ["[]", "null", '"nvfp4"'])
def test_klein_nvfp4_header_rejects_non_object_metadata_without_crashing(
    tmp_path: Path, metadata_value: str
):
    path = tmp_path / "klein-nvfp4-malformed-metadata.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    metadata["_quantization_metadata"] = metadata_value
    save_file(tensors, path, metadata=metadata)

    plan = plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG)

    assert not plan.available
    assert "global NVFP4 metadata must be an object" in plan.contract_errors


def test_klein_nvfp4_materializer_preserves_packed_storage(tmp_path: Path, monkeypatch):
    path = tmp_path / "klein-nvfp4.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    save_file(tensors, path, metadata=metadata)
    monkeypatch.setattr(adapter, "_require_nvfp4_cuda_backend", lambda _device: None)

    transformer = materialize_klein_nvfp4_transformer(
        plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG),
        _SMALL_CONFIG,
    )
    linears = [
        module for module in transformer.modules() if isinstance(module, KleinStoredNVFP4Linear)
    ]

    assert len(linears) == 14
    assert all(module.weight._qdata.dtype is torch.uint8 for module in linears)
    assert all(module.weight._layout_cls == "TensorCoreNVFP4Layout" for module in linears)
    assert all(module.weight.params.scale.dtype is torch.float32 for module in linears)
    assert all(
        module.weight.params.block_scale.dtype is torch.float8_e4m3fn for module in linears
    )
    assert len(transformer._latentslate_klein_nvfp4_modules) == len(linears)
    assert transformer._latentslate_klein_native_backend.endswith("scaled_mm_nvfp4")


def test_klein_nvfp4_materializer_splits_unquantized_fused_qkv(tmp_path: Path, monkeypatch):
    """First-party 9B retains selected QKV projections as dense BF16 tensors."""

    path = tmp_path / "klein-nvfp4-dense-qkv.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    parsed = json.loads(metadata["_quantization_metadata"])
    for source in (
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.txt_attn.qkv.weight",
    ):
        packed = tensors[source]
        tensors[source] = torch.zeros((packed.shape[0], packed.shape[1] * 2), dtype=torch.bfloat16)
        stem = source.removesuffix(".weight")
        for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
            del tensors[stem + suffix]
        del parsed["layers"][stem]
    metadata["_quantization_metadata"] = json.dumps(parsed)
    save_file(tensors, path, metadata=metadata)
    monkeypatch.setattr(adapter, "_require_nvfp4_cuda_backend", lambda _device: None)

    transformer = materialize_klein_nvfp4_transformer(
        plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG), _SMALL_CONFIG
    )

    assert not transformer.transformer_blocks[0].attn.to_q.weight.is_meta
    assert not transformer.transformer_blocks[0].attn.add_q_proj.weight.is_meta


def test_klein_nvfp4_residency_teardown_preserves_all_physical_storage(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "klein-nvfp4.safetensors"
    tensors, metadata = _small_nvfp4_checkpoint()
    save_file(tensors, path, metadata=metadata)
    monkeypatch.setattr(adapter, "_require_nvfp4_cuda_backend", lambda _device: None)
    transformer = materialize_klein_nvfp4_transformer(
        plan_bfl_klein_nvfp4_transformer(path, _SMALL_CONFIG), _SMALL_CONFIG
    )
    linears = [
        module for module in transformer.modules() if isinstance(module, KleinStoredNVFP4Linear)
    ]
    before = [
        (
            module.weight._qdata.clone(),
            module.weight.params.scale.clone(),
            module.weight.params.block_scale.clone(),
        )
        for module in linears
    ]

    with KleinTransformerResidencySession(transformer, onload_device="cpu") as session:
        assert session.policy["stored_bytes"] == adapter._physical_state_bytes(transformer)

    for module, (qdata, scale, block_scale) in zip(linears, before, strict=True):
        assert module.weight.device.type == "cpu"
        assert torch.equal(module.weight._qdata, qdata)
        assert torch.equal(module.weight.params.scale, scale)
        assert torch.equal(module.weight.params.block_scale, block_scale)


def test_klein_nvfp4_backend_gate_rejects_cpu():
    with pytest.raises(RuntimeError, match="configured CUDA"):
        adapter._require_nvfp4_cuda_backend("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_nvfp4_linear_dispatches_native_cuda_kernel():
    target = torch.device("cuda", torch.cuda.current_device())
    try:
        adapter._require_nvfp4_cuda_backend(target)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    qdata = torch.zeros((128, 32), dtype=torch.uint8, device=target)
    block_scale = torch.ones((128, 4), dtype=torch.float8_e4m3fn, device=target)
    tensor_scale = torch.tensor(0.25, dtype=torch.float32, device=target)
    weight = adapter._restore_nvfp4_tensor(
        qdata, block_scale, tensor_scale, (128, 64), torch.bfloat16
    )
    linear = KleinStoredNVFP4Linear(
        weight,
        input_scale=torch.tensor(0.5, dtype=torch.float32),
    )

    output = linear(torch.zeros((128, 64), dtype=torch.bfloat16, device=target))

    assert output.shape == (128, 128)
    assert output.dtype is torch.bfloat16
    assert linear.native_dispatch_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_dynamic_nvfp4_linear_dispatches_native_cuda_kernel():
    target = torch.device("cuda", torch.cuda.current_device())
    try:
        adapter._require_nvfp4_cuda_backend(target)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    weight = adapter._restore_nvfp4_tensor(
        torch.zeros((128, 32), dtype=torch.uint8, device=target),
        torch.ones((128, 4), dtype=torch.float8_e4m3fn, device=target),
        torch.tensor(0.25, dtype=torch.float32, device=target),
        (128, 64),
        torch.bfloat16,
    )
    linear = KleinStoredNVFP4Linear(weight, input_scale=None)

    output = linear(torch.zeros((128, 64), dtype=torch.bfloat16, device=target))

    assert output.shape == (128, 128)
    assert output.dtype is torch.bfloat16
    assert linear.native_dispatch_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_dynamic_fp8_linear_dispatches_native_cuda_kernel():
    target = torch.device("cuda", torch.cuda.current_device())
    qdata = torch.zeros((128, 64), dtype=torch.float8_e4m3fn, device=target)
    weight = adapter._restore_global_fp8_tensor(
        qdata,
        torch.tensor(0.25, dtype=torch.float32, device=target),
        torch.bfloat16,
    )
    linear = KleinStoredLinear(weight, input_scale=None)

    output = linear(torch.zeros((128, 64), dtype=torch.bfloat16, device=target))

    assert output.shape == (128, 128)
    assert output.dtype is torch.bfloat16
    assert linear.native_dispatch_count == 1


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
    fused[:16] = 1
    fused[16:32] = 2
    fused[32:] = 3
    adaln_key = "final_layer.adaLN_modulation.1.weight"
    tensors[adaln_key][:16] = 4
    tensors[adaln_key][16:] = 5
    save_file(tensors, path, metadata=metadata)

    plan = plan_comfy_klein_transformer(path, _SMALL_CONFIG)
    transformer = materialize_klein_transformer(plan, _SMALL_CONFIG)

    q = transformer.transformer_blocks[0].attn.to_q
    k = transformer.transformer_blocks[0].attn.to_k
    v = transformer.transformer_blocks[0].attn.to_v
    assert all(isinstance(layer, KleinStoredLinear) for layer in (q, k, v))
    assert all(layer.weight._qdata.dtype == torch.float8_e4m3fn for layer in (q, k, v))
    assert torch.equal(q.weight._qdata.float(), torch.ones((16, 16)))
    assert torch.equal(k.weight._qdata.float(), torch.full((16, 16), 2.0))
    assert torch.equal(v.weight._qdata.float(), torch.full((16, 16), 3.0))
    assert tuple(layer.weight.params.scale.item() for layer in (q, k, v)) == (
        0.25,
        0.25,
        0.25,
    )
    assert (q.input_scale, k.input_scale, v.input_scale) == (0.5, 0.5, 0.5)
    assert torch.equal(
        transformer.norm_out.linear.weight[:16],
        torch.full((16, 16), 5.0, dtype=torch.bfloat16),
    )
    assert torch.equal(
        transformer.norm_out.linear.weight[16:],
        torch.full((16, 16), 4.0, dtype=torch.bfloat16),
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
            encoder_hidden_states=torch.zeros((1, 1, 16), dtype=torch.bfloat16),
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
            encoder_hidden_states=torch.zeros((1, 1, 16), dtype=torch.bfloat16, device=device),
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


def test_klein_grouped_residency_streams_blocks_and_preserves_storage(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    session = KleinTransformerResidencySession(
        transformer, onload_device="cpu", residency_mode="grouped"
    )

    with session:
        first = _small_forward(transformer)
        second = _small_forward(transformer)
        assert bool(torch.isfinite(first).all())
        assert bool(torch.isfinite(second).all())
        policy = session.policy
        assert policy["mode"] == "grouped"
        assert policy["resident_block_count"] == 1
        assert policy["streamed_block_count"] == 1
        assert policy["resident_block_bytes"] + policy["streamed_block_bytes"] == sum(
            session._group_sizes.values()
        )
        assert len(session._group_handles) == 2 * policy["streamed_block_count"]
        expected_resident = min(
            session._group_sizes,
            key=lambda name: (-session._group_sizes[name], name),
        )
        assert session._resident_groups == (expected_resident,)

    assert not session._group_handles
    _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before)


@pytest.mark.parametrize(
    "failure_stage",
    ["root_move", "resident_move", "streamed_move", "pre_hook", "post_hook"],
)
def test_klein_grouped_setup_is_transactional_for_every_mutating_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    session = KleinTransformerResidencySession(
        transformer, onload_device="cpu", residency_mode="grouped"
    )
    decision = session._choose_policy()
    session._decision = decision
    groups = session._group_modules()
    ordered = sorted(groups, key=lambda name: (-session._group_sizes[name], name))
    resident_name = ordered[0]
    streamed_name = next(name for name in groups if name != resident_name)
    resident = groups[resident_name]
    streamed = groups[streamed_name]
    group_ids = {id(group) for group in groups.values()}
    root = next(
        child
        for child in transformer.children()
        if not (
            isinstance(child, torch.nn.ModuleList)
            and all(id(item) in group_ids for item in child)
        )
    )
    hook_counts_before = {
        name: (len(block._forward_pre_hooks), len(block._forward_hooks))
        for name, block in groups.items()
    }

    if failure_stage.endswith("move"):
        target = {
            "root_move": root,
            "resident_move": resident,
            "streamed_move": streamed,
        }[failure_stage]
        original_move = session._move_module
        failed = False

        def fail_once(module, device):
            nonlocal failed
            if module is target and not failed:
                failed = True
                raise RuntimeError(f"injected {failure_stage}")
            original_move(module, device)

        monkeypatch.setattr(session, "_move_module", fail_once)
    elif failure_stage == "pre_hook":
        monkeypatch.setattr(
            streamed,
            "register_forward_pre_hook",
            lambda _hook: (_ for _ in ()).throw(RuntimeError("injected pre_hook")),
        )
    else:
        monkeypatch.setattr(
            streamed,
            "register_forward_hook",
            lambda _hook, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected post_hook")
            ),
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_stage}"):
        session.__enter__()

    assert session._closed and not session.active
    assert session._group_handles == []
    assert session._execution_handles == []
    assert adapter._ACTIVE_KLEIN_SESSION is None
    assert not getattr(transformer, "_latentslate_klein_residency_poisoned", None)
    assert {
        name: (len(block._forward_pre_hooks), len(block._forward_hooks))
        for name, block in groups.items()
    } == hook_counts_before
    _assert_snapshot_preserved(transformer, stored, qdata_before, scales_before, dense_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_grouped_setup_barrier_uncertainty_poisons_without_cpu_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transformer = _materialized_small_transformer(tmp_path)
    target = torch.device("cuda", torch.cuda.current_device())
    session = KleinTransformerResidencySession(
        transformer, onload_device=target, residency_mode="grouped"
    )
    original_move = session._move_module
    original_synchronize = torch.cuda.synchronize
    moved_stateful_root = False

    def fail_second_root_move(module, device):
        nonlocal moved_stateful_root
        if not isinstance(module, torch.nn.ModuleList) and moved_stateful_root:
            raise RuntimeError("injected CUDA setup move")
        original_move(module, device)
        if not isinstance(module, torch.nn.ModuleList) and any(
            value.device == target for value in adapter._state_values(module).values()
        ):
            moved_stateful_root = True

    monkeypatch.setattr(session, "_move_module", fail_second_root_move)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(RuntimeError("injected setup barrier")),
    )

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        session.__enter__()

    assert "rollback barrier failed" in transformer._latentslate_klein_residency_poisoned
    assert session._group_handles == []
    assert session._execution_handles == []
    assert adapter._ACTIVE_KLEIN_SESSION is None
    assert any(value.device == target for value in adapter._state_values(transformer).values())

    # Test-owned recovery after proving fail-closed behavior; production evicts
    # the poisoned runtime instead of reconstructing across a failed barrier.
    monkeypatch.setattr(torch.cuda, "synchronize", original_synchronize)
    original_synchronize(target)
    monkeypatch.setattr(session, "_move_module", original_move)
    for child in transformer.children():
        original_move(child, torch.device("cpu"))
    del transformer._latentslate_klein_residency_poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_klein_grouped_residency_keeps_only_budgeted_subset_on_cuda(tmp_path: Path):
    transformer = _materialized_small_transformer(tmp_path)
    stored, qdata_before, scales_before, dense_before = _stored_snapshot(transformer)
    target = torch.device("cuda", torch.cuda.current_device())
    session = KleinTransformerResidencySession(
        transformer, onload_device=target, residency_mode="grouped"
    )

    with session:
        assert session._resident_groups
        assert session._streamed_groups
        for name in session._resident_groups:
            assert all(
                value.device == target
                for value in adapter._state_values(transformer.get_submodule(name)).values()
            )
        for name in session._streamed_groups:
            assert all(
                value.device.type == "cpu"
                for value in adapter._state_values(transformer.get_submodule(name)).values()
            )
        assert len(session._group_handles) == 2 * len(session._streamed_groups)

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
                encoder_hidden_states=torch.zeros((1, 1, 16), dtype=torch.bfloat16, device=target),
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
