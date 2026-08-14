from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

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


@pytest.mark.skipif(not _DEV.is_file(), reason="LTX 2.3 Dev artifact absent")
def test_pinned_diffusers_meta_shell_is_exact_transformer_closure():
    contract = av.inspect_ltx23_av_artifact(_DEV, expected_variant="dev")
    shell = av.build_ltx23_av_meta_shell(contract)
    plan = av.plan_ltx23_av_materialization(shell, _DEV, expected_variant="dev")

    assert len(shell.state_dict()) == 4_186
    assert all(tensor.is_meta for tensor in shell.state_dict().values())
    assert len(plan.plan_fingerprint) == 64
    assert len([item for item in shell.modules() if type(item) is torch.nn.Linear]) == 1_660

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
