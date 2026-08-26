from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from latentslate_engine.runtime import ltx23_av_stored_adapter as av

_ROOT = Path(r"M:\LatentSlateEngineData")
_DEV = _ROOT / "models/ltx23/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
_DISTILLED = _ROOT / "models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
_MODEL_LORA = (
    _ROOT
    / "loras/ltx23/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
)


def _stored_linear(*, in_features: int = 4, out_features: int = 3):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    qdata = torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(
        scale=torch.tensor(0.25, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape),
    )
    weight = QuantizedTensor(qdata, "TensorCoreFP8Layout", params)
    return av.LTX23StoredFP8Linear(
        weight,
        torch.zeros(out_features, dtype=torch.bfloat16),
        input_scale=torch.tensor(0.5, dtype=torch.float32),
    )


def test_module_storage_counts_quantized_physical_state_once_and_restores_cpu_objects():
    module = torch.nn.Module()
    module.stored = _stored_linear()
    module.dense = torch.nn.Linear(2, 2)
    originals = {
        (id(owner), name): value
        for owner in module.modules()
        for values in (owner._parameters, owner._buffers)
        for name, value in values.items()
        if value is not None
    }

    storage = av.capture_ltx23_module_storage(module)
    expected = (
        3 * 4  # FP8 qdata
        + 4  # F32 scale sidecar
        + 4  # F32 input-scale sidecar
        + 3 * 2  # BF16 bias
        + 2 * 2 * 4  # dense F32 weight
        + 2 * 4  # dense F32 bias
    )
    assert storage.physical_bytes == expected

    binding = storage.copy_to("cpu")
    binding.activate()

    def current(owner, name):
        return (
            owner._parameters[name]
            if name in owner._parameters
            else owner._buffers[name]
        )

    assert any(
        current(owner, name) is not original
        for owner in module.modules()
        for name, original in [
            (name, originals[(id(owner), name)])
            for name in (*owner._parameters.keys(), *owner._buffers.keys())
            if (id(owner), name) in originals
        ]
    )
    binding.restore_cpu()
    assert all(
        current(owner, name) is original
        for owner in module.modules()
        for name, original in [
            (name, originals[(id(owner), name)])
            for name in (*owner._parameters.keys(), *owner._buffers.keys())
            if (id(owner), name) in originals
        ]
    )


def test_leaf_storage_paths_merge_cross_group_aliases_and_force_residency() -> None:
    shared = nn.Parameter(torch.ones(8), requires_grad=False)
    model = nn.Module()
    model.root_alias = shared
    model.transformer_blocks = nn.ModuleList([nn.Module()])
    model.transformer_blocks[0].weight = shared
    model.transformer_blocks[0].other = nn.Parameter(
        torch.ones(5_000), requires_grad=False
    )

    leaves = av.capture_ltx23_leaf_storages(model)

    assert [leaf.path for leaf in leaves] == ["<root>"]
    alias = leaves[0]
    assert alias.schedule_groups == ("root", "transformer_blocks.0")
    assert alias.force_resident is True
    assert len(alias.storage.slots) == 3
    assert alias.storage.physical_bytes == (
        shared.numel() + 5_000
    ) * shared.element_size()


def test_leaf_capture_accepts_model_owned_schedule_resolver_without_av_drift() -> None:
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    default = av.capture_ltx23_leaf_storages(model)
    seen: list[str] = []

    def resolver(path, _slots, _sources):
        seen.append(path)
        return av.LTX23LeafSchedule("block.0", tiny_force_resident=False)

    custom = av.capture_ltx23_leaf_storages(model, schedule_resolver=resolver)

    assert [leaf.schedule_groups for leaf in default] == [("transformer_blocks.0",)]
    assert [leaf.schedule_groups for leaf in custom] == [("block.0",)]
    assert seen == ["transformer_blocks.0"]
    assert custom[0].force_resident is False


def test_stored_fp8_linear_records_direct_dispatch_without_dense_fallback(monkeypatch):
    linear = _stored_linear()
    seen = {}

    def fake_dispatch(input, weight, bias, *, input_scale):
        seen.update(input=input, weight=weight, bias=bias, input_scale=input_scale)
        return torch.ones((input.shape[0], weight.shape[0]), dtype=input.dtype)

    monkeypatch.setattr(av, "_direct_kitchen_fp8_linear", fake_dispatch)
    output = linear(torch.zeros((2, 5, 4), dtype=torch.bfloat16))

    assert output.shape == (2, 5, 3)
    assert seen["input"].shape == (10, 4)
    assert seen["input_scale"] == 0.5
    assert linear.dispatch_evidence() == {
        "backend": "comfy_kitchen.tensorcore_fp8",
        "native_dispatch_count": 1,
        "rejected_dispatch_count": 0,
        "dense_fallback_count": 0,
        "last_dispatch_error": None,
    }


def test_stored_fp8_linear_fails_closed_and_records_rejection(monkeypatch):
    linear = _stored_linear()

    def reject(*_args, **_kwargs):
        raise NotImplementedError("kernel unavailable")

    monkeypatch.setattr(av, "_direct_kitchen_fp8_linear", reject)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        linear(torch.zeros((1, 4), dtype=torch.bfloat16))

    evidence = linear.dispatch_evidence()
    assert evidence["native_dispatch_count"] == 0
    assert evidence["rejected_dispatch_count"] == 1
    assert evidence["dense_fallback_count"] == 0
    assert evidence["last_dispatch_error"] == "NotImplementedError: kernel unavailable"


def test_stored_fp8_linear_keeps_base_native_and_dispatches_additive_lora(monkeypatch):
    linear = _stored_linear()
    monkeypatch.setattr(
        av,
        "_direct_kitchen_fp8_linear",
        lambda input, weight, bias, *, input_scale: torch.zeros(
            (input.shape[0], weight.shape[0]), dtype=input.dtype
        ),
    )
    linear.add_lora_adapter(
        "distilled",
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.ones((3, 2), dtype=torch.bfloat16),
        alpha_over_rank=1.0,
    )
    linear.set_lora_strength("distilled", 0.5)

    output = linear(torch.ones((1, 4), dtype=torch.bfloat16))
    torch.testing.assert_close(output, torch.full((1, 3), 4.0, dtype=torch.bfloat16))
    assert linear.native_dispatch_count == 1
    assert linear.lora_dispatch_count == 1
    assert linear.dense_fallback_count == 0


def test_stored_fp8_linear_casts_fp32_once_for_native_and_lora_paths(monkeypatch):
    linear = _stored_linear()
    native_inputs: list[torch.Tensor] = []

    def fake_dispatch(input, weight, bias, *, input_scale):
        native_inputs.append(input)
        return torch.zeros((input.shape[0], weight.shape[0]), dtype=input.dtype)

    monkeypatch.setattr(av, "_direct_kitchen_fp8_linear", fake_dispatch)
    linear.add_lora_adapter(
        "distilled",
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16),
        alpha_over_rank=1.0,
    )
    linear.set_lora_strength("distilled", 0.5)
    adapter = linear._lora_adapters["distilled"]
    lora_inputs: list[torch.Tensor] = []
    original_lora_forward = adapter.forward

    def capture_lora(input):
        lora_inputs.append(input)
        return original_lora_forward(input)

    monkeypatch.setattr(adapter, "forward", capture_lora)
    input = torch.tensor([[1.125, -2.25, 3.5, 4.75]], dtype=torch.float32)
    output = linear(input)

    expected_activation = input.to(torch.bfloat16)
    assert native_inputs[0].dtype is torch.bfloat16
    assert lora_inputs[0].dtype is torch.bfloat16
    torch.testing.assert_close(native_inputs[0], expected_activation)
    torch.testing.assert_close(lora_inputs[0], expected_activation)
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(
        output,
        torch.tensor([[0.5625, 1.125, 1.6875]], dtype=torch.bfloat16),
    )


def test_stored_fp8_linear_rejects_nonfloating_activations():
    linear = _stored_linear()
    with pytest.raises(TypeError, match="floating-point activations"):
        linear(torch.zeros((1, 4), dtype=torch.int64))


def test_stored_fp8_linear_rejects_invalid_bias_and_input_scale():
    linear = _stored_linear()
    weight = linear.weight
    with pytest.raises(ValueError, match="BF16 bias"):
        av.LTX23StoredFP8Linear(
            weight,
            torch.zeros(3, dtype=torch.float32),
            input_scale=torch.tensor(0.5),
        )
    with pytest.raises(ValueError, match="positive finite F32 scalar"):
        av.LTX23StoredFP8Linear(
            weight,
            torch.zeros(3, dtype=torch.bfloat16),
            input_scale=torch.tensor(0.0),
        )


def test_file_backed_av_materialization_retains_only_meta_base_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tiny-av.safetensors"
    path.write_bytes(b"fixture")
    shell = torch.nn.Module()
    shell.proj = torch.nn.Linear(4, 3, bias=True, device="meta")
    spans = MappingProxyType(
        {
            "base.proj.weight": av.LTX23AVSafetensorSpan(
                "base.proj.weight", "F8_E4M3", (3, 4), 128, 12
            ),
            "base.proj.weight_scale": av.LTX23AVSafetensorSpan(
                "base.proj.weight_scale", "F32", (), 256, 4
            ),
            "base.proj.input_scale": av.LTX23AVSafetensorSpan(
                "base.proj.input_scale", "F32", (), 260, 4
            ),
            "base.proj.bias": av.LTX23AVSafetensorSpan(
                "base.proj.bias", "BF16", (3,), 264, 6
            ),
        }
    )
    linear = av.LTX23AVLinearSpec(
        module_name="proj",
        weight=av.LTX23AVStateSpec("proj.weight", "F8_E4M3", (3, 4)),
        bias=av.LTX23AVStateSpec("proj.bias", "BF16", (3,)),
        quantized=True,
        source_weight_key="base.proj.weight",
        source_bias_key="base.proj.bias",
        source_weight_scale_key="base.proj.weight_scale",
        source_input_scale_key="base.proj.input_scale",
    )
    mapped = (
        av.LTX23AVMappedStateSpec(
            "base.proj.weight", "proj.weight", "F8_E4M3", (3, 4)
        ),
        av.LTX23AVMappedStateSpec(
            "base.proj.bias", "proj.bias", "BF16", (3,)
        ),
    )
    contract = av.LTX23AVArtifactContract(
        path=path,
        variant="dev",
        artifact_signature={"fixture": True},
        header_fingerprint="f" * 64,
        state=(),
        transformer_state=mapped,
        connector_state=(),
        external_connector_state=(),
        linears=(linear,),
        transformer_base_spans=spans,
        header_size_bytes=64,
    )
    plan = av.LTX23AVMaterializationPlan(
        contract=contract,
        shell_type=f"{type(shell).__module__}.{type(shell).__qualname__}",
        plan_fingerprint="p" * 64,
    )

    class _Handle:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def keys(self):
            return spans.keys()

        def get_tensor(self, key: str):
            self.requested.append(key)
            if key != "base.proj.input_scale":
                raise AssertionError(f"base payload unexpectedly materialized: {key}")
            return torch.tensor(0.5, dtype=torch.float32)

    handle = _Handle()
    monkeypatch.setattr(av, "inspect_ltx23_av_artifact", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(av, "plan_ltx23_av_materialization", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(av, "_read_safetensors_header", lambda _path: dict.fromkeys(spans))

    result = av.materialize_ltx23_av(shell, plan, payload_handle=handle)
    assert result is shell
    assert handle.requested == ["base.proj.input_scale"]
    assert isinstance(shell.proj, av.LTX23StoredFP8Linear)
    assert all(value.is_meta for value in shell.state_dict().values())
    descriptors = shell._latentslate_ltx23_av_source_descriptors
    assert len(descriptors) == 3
    weight_descriptor = descriptors[id(shell.proj.weight)]
    assert [span.key for span in weight_descriptor.spans] == [
        "base.proj.weight",
        "base.proj.weight_scale",
    ]
    assert descriptors[id(shell.proj.bias)].spans[0].key == "base.proj.bias"
    assert descriptors[id(shell.proj.input_scale)].spans[0].key == (
        "base.proj.input_scale"
    )
    assert shell._latentslate_ltx23_av_materialization["base_cpu_tensor_bytes"] == 0


def test_file_backed_dense_bf16_spans_replace_fp32_meta_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency.aimdo import (
        AimdoDynamicResidency,
    )

    path = tmp_path / "dense-av.safetensors"
    path.write_bytes(b"fixture")
    shell = torch.nn.Module()
    shell.proj = torch.nn.Linear(4, 3, bias=True, device="meta")
    assert shell.proj.weight.dtype is torch.float32
    spans = MappingProxyType(
        {
            "base.proj.weight": av.LTX23AVSafetensorSpan(
                "base.proj.weight", "BF16", (3, 4), 128, 24
            ),
            "base.proj.bias": av.LTX23AVSafetensorSpan(
                "base.proj.bias", "BF16", (3,), 256, 6
            ),
        }
    )
    weight_state = av.LTX23AVStateSpec("proj.weight", "BF16", (3, 4))
    bias_state = av.LTX23AVStateSpec("proj.bias", "BF16", (3,))
    linear = av.LTX23AVLinearSpec(
        module_name="proj",
        weight=weight_state,
        bias=bias_state,
        quantized=False,
        source_weight_key="base.proj.weight",
        source_bias_key="base.proj.bias",
        source_weight_scale_key=None,
        source_input_scale_key=None,
    )
    mapped = (
        av.LTX23AVMappedStateSpec(
            "base.proj.weight", "proj.weight", "BF16", (3, 4)
        ),
        av.LTX23AVMappedStateSpec(
            "base.proj.bias", "proj.bias", "BF16", (3,)
        ),
    )
    contract = av.LTX23AVArtifactContract(
        path=path,
        variant="dev",
        artifact_signature={"fixture": True},
        header_fingerprint="f" * 64,
        state=(),
        transformer_state=mapped,
        connector_state=(),
        external_connector_state=(),
        linears=(linear,),
        transformer_base_spans=spans,
        header_size_bytes=64,
    )
    plan = av.LTX23AVMaterializationPlan(
        contract=contract,
        shell_type=f"{type(shell).__module__}.{type(shell).__qualname__}",
        plan_fingerprint="p" * 64,
    )

    class _Handle:
        def keys(self):
            return spans.keys()

        def get_tensor(self, key: str):
            raise AssertionError(f"dense base payload unexpectedly materialized: {key}")

    monkeypatch.setattr(av, "inspect_ltx23_av_artifact", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(av, "plan_ltx23_av_materialization", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(av, "_read_safetensors_header", lambda _path: dict.fromkeys(spans))

    result = av.materialize_ltx23_av(shell, plan, payload_handle=_Handle())

    assert result is shell
    assert shell.proj.weight.is_meta and shell.proj.bias.is_meta
    assert shell.proj.weight.dtype is shell.proj.bias.dtype is torch.bfloat16
    descriptors = shell._latentslate_ltx23_av_source_descriptors
    values = (descriptors[id(shell.proj.weight)], descriptors[id(shell.proj.bias)])
    assert AimdoDynamicResidency.group_bytes(values) == 2 * 1024


def test_authenticated_av_spans_reject_boolean_or_out_of_bounds_offsets() -> None:
    header = {
        "weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
    }
    span = av._authenticated_transformer_spans(
        header, physical_keys={"weight"}, payload_offset=32, file_size=36
    )["weight"]
    assert (span.offset, span.size, span.dtype, span.shape) == (32, 4, "BF16", (2,))

    invalid = {
        "weight": {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]}
    }
    with pytest.raises(ValueError, match="metadata is invalid"):
        av._authenticated_transformer_spans(
            invalid, physical_keys={"weight"}, payload_offset=32, file_size=34
        )
    with pytest.raises(ValueError, match="bounds are invalid"):
        av._authenticated_transformer_spans(
            header, physical_keys={"weight"}, payload_offset=32, file_size=35
        )
def test_connector_projection_preserves_fp32_normalization_then_casts_to_bf16(monkeypatch):
    projection = av._LTX23ConnectorProjection(4, 3, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [[1.0, -0.5, 0.25, 0.0], [0.0, 0.75, -1.0, 0.5], [-0.25, 0.0, 0.5, 1.0]],
                dtype=torch.bfloat16,
            )
        )
        projection.bias.copy_(torch.tensor([0.25, -0.5, 0.75], dtype=torch.bfloat16))

    hidden_states = torch.tensor(
        [[[1.1, -0.7, 0.3, 2.2], [-1.4, 0.6, 1.8, -0.2]]], dtype=torch.float32
    )
    # This is Diffusers' per-token RMS normalization and intentionally stays
    # FP32 until the projection handoff.
    normalized = hidden_states * torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + 1e-6)
    observed: list[torch.dtype] = []
    native_linear = torch.nn.functional.linear

    def capture_linear(input, weight, bias=None):
        observed.append(input.dtype)
        return native_linear(input, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", capture_linear)
    output = projection(normalized)

    expected = native_linear(
        normalized.to(torch.bfloat16), projection.weight, projection.bias
    )
    assert normalized.dtype is torch.float32
    assert observed == [torch.bfloat16]
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected)


def test_distilled_dense_bf16_boundary_accepts_adaln_fp32_without_changing_fp8_dispatch(monkeypatch):
    """Pure Distilled has no model-LoRA wrapper around its dense projections."""

    transformer = torch.nn.Module()
    transformer.dense = torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
    transformer.fp8 = _stored_linear()
    with torch.no_grad():
        transformer.dense.weight.copy_(torch.eye(4, dtype=torch.bfloat16))
        transformer.dense.bias.zero_()
    adaln_output = torch.tensor([[1.1, -0.7, 0.3, 2.2]], dtype=torch.float32)
    state_keys_before = tuple(transformer.state_dict())
    dense_weight_before = transformer.dense.weight

    with pytest.raises(RuntimeError, match="mat1 and mat2 must have the same dtype"):
        transformer.dense(adaln_output)

    dense = av.LTX23AVLinearSpec(
        module_name="dense",
        weight=av.LTX23AVStateSpec("dense.weight", "BF16", (4, 4)),
        bias=av.LTX23AVStateSpec("dense.bias", "BF16", (4,)),
        quantized=False,
        source_weight_key="dense.weight",
        source_bias_key="dense.bias",
        source_weight_scale_key=None,
        source_input_scale_key=None,
    )
    av._install_ltx23_distilled_dense_bf16_boundaries(transformer, (dense,))
    assert isinstance(transformer.dense, av._LTX23DenseBFloat16Linear)
    assert not isinstance(transformer.dense, av.LTX23DenseLoraLinear)
    assert tuple(transformer.state_dict()) == state_keys_before
    assert transformer.dense.weight is dense_weight_before
    residency_storage = av.capture_ltx23_module_storage(transformer)
    residency_binding = residency_storage.copy_to("cpu")
    residency_binding.activate()
    residency_binding.restore_cpu()

    dense_inputs: list[torch.dtype] = []
    native_linear = torch.nn.functional.linear

    def capture_dense_linear(input, weight, bias=None):
        dense_inputs.append(input.dtype)
        return native_linear(input, weight, bias)

    fp8_inputs: list[torch.dtype] = []

    def fake_fp8_dispatch(input, weight, bias, *, input_scale):
        fp8_inputs.append(input.dtype)
        return torch.zeros((input.shape[0], weight.shape[0]), dtype=input.dtype)

    monkeypatch.setattr(torch.nn.functional, "linear", capture_dense_linear)
    monkeypatch.setattr(av, "_direct_kitchen_fp8_linear", fake_fp8_dispatch)
    output = transformer.fp8(transformer.dense(adaln_output))

    assert dense_inputs == [torch.bfloat16]
    assert fp8_inputs == [torch.bfloat16]
    assert output.dtype is torch.bfloat16
    assert transformer.fp8.native_dispatch_count == 1
    assert transformer.fp8.dense_fallback_count == 0


@pytest.mark.skipif(
    not (_DEV.is_file() and _DISTILLED.is_file()), reason="LTX 2.3 artifacts absent"
)
def test_installed_ltx23_artifacts_have_exact_diffusers_split_and_fp8_contracts():
    dev = av.inspect_ltx23_av_artifact(_DEV, expected_variant="dev")
    distilled = av.inspect_ltx23_av_artifact(_DISTILLED, expected_variant="distilled")

    assert (
        len(dev.state),
        len(dev.transformer_state),
        len(dev.connector_state),
        len(dev.external_connector_state),
    ) == (
        4_444,
        4_186,
        258,
        4,
    )
    assert (len(dev.linears), dev.quantized_linear_count, dev.dense_linear_count) == (
        1_660,
        1_496,
        164,
    )
    assert (
        len(distilled.linears),
        distilled.quantized_linear_count,
        distilled.dense_linear_count,
    ) == (1_660, 1_462, 198)
    assert all(
        item.source_weight_scale_key and item.source_input_scale_key
        for item in dev.linears
        if item.quantized
    )
    assert all(item.bias.dtype == "BF16" for item in dev.linears)
    assert len(dev.transformer_base_spans) == 4_186 + 2 * 1_496
    # The previous 23,722,941,536-byte runtime master combined this immutable
    # 20.98 GB base with 2,740,286,368 bytes of active model-LoRA state.
    assert sum(span.size for span in dev.transformer_base_spans.values()) == 20_982_655_168
    assert all(
        span.offset >= 8 + dev.header_size_bytes and span.size > 0
        for span in dev.transformer_base_spans.values()
    )


@pytest.mark.skipif(not _DEV.is_file(), reason="LTX 2.3 Dev artifact absent")
def test_pinned_diffusers_meta_shell_is_exact_transformer_closure():
    from latentslate_engine.runtime.framework.residency.aimdo import (
        AimdoDynamicResidency,
    )

    contract = av.inspect_ltx23_av_artifact(_DEV, expected_variant="dev")
    shell = av.build_ltx23_av_meta_shell(contract)
    plan = av.plan_ltx23_av_materialization(shell, _DEV, expected_variant="dev")

    assert len(shell.state_dict()) == 4_186
    assert all(tensor.is_meta for tensor in shell.state_dict().values())
    assert len(plan.plan_fingerprint) == 64
    assert len([item for item in shell.modules() if type(item) is torch.nn.Linear]) == 1_660

    shell = av.materialize_ltx23_av(shell, plan, source_backed=True)
    descriptors = shell._latentslate_ltx23_av_source_descriptors
    assert all(value.template.is_meta for value in descriptors.values())
    assert all(AimdoDynamicResidency.group_bytes((value,)) > 0 for value in descriptors.values())
    root = av.capture_ltx23_module_storage(
        shell,
        exclude_children=frozenset({"transformer_blocks"}),
        source_values=descriptors,
    )
    root_values = tuple(
        descriptors.get(id(slot.cpu_value), slot.cpu_value) for slot in root.slots
    )
    assert AimdoDynamicResidency.group_bytes(root_values) == 854_947_840

    connectors = av.build_ltx23_connector_meta_shell(contract)
    connector_plan = av.plan_ltx23_connector_materialization(
        connectors, _DEV, expected_variant="dev"
    )
    assert len(connectors.state_dict()) == 262
    assert all(tensor.is_meta for tensor in connectors.state_dict().values())
    assert len(connector_plan.plan_fingerprint) == 64


@pytest.mark.skipif(
    not (_DEV.is_file() and _MODEL_LORA.is_file()), reason="LTX 2.3 model LoRA absent"
)
def test_installed_model_lora_maps_all_linears_and_identity_missing_alpha():
    base = av.inspect_ltx23_av_artifact(_DEV, expected_variant="dev")
    lora = av.inspect_ltx23_model_lora(base, _MODEL_LORA)

    assert len(lora.targets) == 1_660
    missing = [item for item in lora.targets if item.alpha_key is None]
    assert [(item.module_name, item.alpha_over_rank) for item in missing] == [
        ("time_embed.linear", 1.0)
    ]
    assert all(math.isnan(item.alpha_over_rank) for item in lora.targets if item.alpha_key)
    assert {item.module_name for item in lora.targets} == {
        item.module_name for item in base.linears
    }
    lora_header = av._read_safetensors_header(_MODEL_LORA)
    lora_header.pop("__metadata__", None)
    lora_bytes = sum(
        int(torch.Size(entry["shape"]).numel()) * 2 for entry in lora_header.values()
    )
    assert lora_bytes == 2_740_295_670
    file_backed_group_bytes = sum(
        span.size for span in base.transformer_base_spans.values()
    )
    lora_cpu_source_bytes = (
        lora_bytes
        - sum(item.alpha_key is not None for item in lora.targets) * 2
    )
    assert (
        file_backed_group_bytes + lora_cpu_source_bytes == 23_722_947_520
    )


def test_dense_model_lora_install_strength_dispatch_and_remove(tmp_path):
    path = tmp_path / "model-lora.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            "proj.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
            "proj.alpha": torch.tensor(2.0, dtype=torch.bfloat16),
        },
        path,
    )
    header = av._read_safetensors_header(path)
    header.pop("__metadata__", None)
    contract = av.LTX23AVLoraContract(
        path=path,
        artifact_signature=av.path_signature(path),
        header_fingerprint=av._fingerprint(header),
        targets=(
            av.LTX23AVLoraTargetSpec(
                module_name="proj",
                down_key="proj.lora_A.weight",
                up_key="proj.lora_B.weight",
                alpha_key="proj.alpha",
                rank=2,
                alpha_over_rank=math.nan,
            ),
        ),
    )
    transformer = torch.nn.Module()
    transformer.proj = torch.nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
    transformer.proj.weight.data.zero_()
    transformer.proj.bias.data.zero_()

    installed = av.install_ltx23_model_lora(
        transformer, contract, adapter_name="distilled", strength=0.5
    )
    assert isinstance(transformer.proj, av.LTX23DenseLoraLinear)
    output = transformer.proj(torch.ones((1, 4), dtype=torch.bfloat16))
    torch.testing.assert_close(output, torch.full((1, 3), 4.0, dtype=torch.bfloat16))
    assert av.ltx23_model_lora_dispatch_evidence(transformer, installed) == {
        "adapter_name": "distilled",
        "selected_targets": 1,
        "dispatched_targets": 1,
        "complete": True,
    }

    av.set_ltx23_model_lora_strength(transformer, installed, 0.0)
    torch.testing.assert_close(
        transformer.proj(torch.ones((1, 4), dtype=torch.bfloat16)),
        torch.zeros((1, 3), dtype=torch.bfloat16),
    )
    av.remove_ltx23_model_lora(transformer, installed)
    assert type(transformer.proj) is torch.nn.Linear


def test_file_backed_dense_base_keeps_meta_identity_across_lora_strength_transitions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-lora-meta.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            "proj.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
            "proj.alpha": torch.tensor(2.0, dtype=torch.bfloat16),
        },
        path,
    )
    header = av._read_safetensors_header(path)
    header.pop("__metadata__", None)
    contract = av.LTX23AVLoraContract(
        path=path,
        artifact_signature=av.path_signature(path),
        header_fingerprint=av._fingerprint(header),
        targets=(
            av.LTX23AVLoraTargetSpec(
                "proj",
                "proj.lora_A.weight",
                "proj.lora_B.weight",
                "proj.alpha",
                2,
                math.nan,
            ),
        ),
    )
    transformer = nn.Module()
    transformer.proj = nn.Linear(4, 3, bias=True, device="meta")
    base = transformer.proj
    base_weight = base.weight
    base_bias = base.bias

    installation = av.install_ltx23_model_lora(
        transformer, contract, adapter_name="distilled", strength=0.5
    )
    assert isinstance(transformer.proj, av.LTX23DenseLoraLinear)
    assert transformer.proj.base is base
    assert transformer.proj.base.weight is base_weight
    assert transformer.proj.base.bias is base_bias
    assert all(
        not parameter.is_meta
        for parameter in transformer.proj._lora_adapters.parameters()
    )
    av.set_ltx23_model_lora_strength(transformer, installation, 0.0)
    assert transformer.proj._lora_adapters["distilled"].strength == 0.0
    av.set_ltx23_model_lora_strength(transformer, installation, 0.5)
    assert transformer.proj._lora_adapters["distilled"].strength == 0.5
    assert transformer.proj.base.weight is base_weight

    av.remove_ltx23_model_lora(transformer, installation)
    assert transformer.proj is base
    assert transformer.proj.weight is base_weight


def test_dense_model_lora_casts_fp32_input_at_bf16_base_and_lora_boundary(tmp_path, monkeypatch):
    path = tmp_path / "model-lora.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor(
                [[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -1.0]],
                dtype=torch.bfloat16,
            ),
            "proj.lora_B.weight": torch.tensor(
                [[0.5, 1.0], [-1.0, 0.25], [0.75, -0.5]], dtype=torch.bfloat16
            ),
            "proj.alpha": torch.tensor(2.0, dtype=torch.bfloat16),
        },
        path,
    )
    header = av._read_safetensors_header(path)
    header.pop("__metadata__", None)
    contract = av.LTX23AVLoraContract(
        path=path,
        artifact_signature=av.path_signature(path),
        header_fingerprint=av._fingerprint(header),
        targets=(
            av.LTX23AVLoraTargetSpec(
                module_name="proj",
                down_key="proj.lora_A.weight",
                up_key="proj.lora_B.weight",
                alpha_key="proj.alpha",
                rank=2,
                alpha_over_rank=math.nan,
            ),
        ),
    )
    transformer = torch.nn.Module()
    transformer.proj = torch.nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        transformer.proj.weight.copy_(
            torch.tensor(
                [[1.0, 0.5, -0.25, 0.0], [0.0, -0.5, 0.75, 1.0], [-1.0, 0.25, 0.5, -0.75]],
                dtype=torch.bfloat16,
            )
        )
        transformer.proj.bias.copy_(torch.tensor([0.25, -0.5, 0.75], dtype=torch.bfloat16))

    av.install_ltx23_model_lora(transformer, contract, adapter_name="distilled", strength=0.5)
    assert isinstance(transformer.proj, av.LTX23DenseLoraLinear)
    input = torch.tensor([[1.1, -0.7, 0.3, 2.2]], dtype=torch.float32)
    expected_input = input.to(torch.bfloat16)
    base = transformer.proj.base
    adapter = transformer.proj._lora_adapters["distilled"]
    native_linear = torch.nn.functional.linear
    expected = native_linear(expected_input, base.weight, base.bias) + native_linear(
        native_linear(expected_input, adapter.down), adapter.up
    ) * (adapter.alpha_over_rank * adapter.strength)
    observed: list[torch.dtype] = []

    def capture_linear(value, weight, bias=None):
        observed.append(value.dtype)
        return native_linear(value, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", capture_linear)
    output = transformer.proj(input)

    assert input.dtype is torch.float32
    assert observed == [torch.bfloat16, torch.bfloat16, torch.bfloat16]
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected)
