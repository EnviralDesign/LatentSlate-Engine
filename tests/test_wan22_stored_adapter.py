from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import latentslate_engine.runtime.wan22_stored_adapter as adapter
from latentslate_engine.runtime.wan22_stored_adapter import (
    NativeStoredLinear,
    attach_native_stored_linear,
    build_wan_transformer_skeleton,
    map_comfy_wan_parameter_key,
    plan_comfy_wan_transformer,
    validate_stored_quant_offload_mode,
)


def _fp8_weight():
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    qdata = torch.tensor([[1, 2]], dtype=torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(scale=torch.tensor(0.5), orig_dtype=torch.float32, orig_shape=(1, 2))
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


@pytest.mark.parametrize("input_scale", [None, torch.tensor(0.25)], ids=["current", "legacy"])
def test_native_stored_linear_matches_current_and_legacy_fp8_cpu_reference(input_scale: torch.Tensor | None):
    linear = NativeStoredLinear(_fp8_weight(), bias=torch.tensor([0.25]), input_scale=input_scale)
    output = linear(torch.tensor([[1.0, 2.0]]))

    assert torch.allclose(output, torch.tensor([[2.75]]), atol=0.01, rtol=0.01)


def test_native_stored_linear_runs_convrot_int8_cpu():
    weight = _int8_weight()
    linear = NativeStoredLinear(weight)
    input = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(linear(input), torch.nn.functional.linear(input, weight.dequantize()))


def test_attach_native_stored_linear_registers_parameters_and_preserves_scale_precision():
    parent = torch.nn.Module()
    parent.linear = torch.nn.Linear(2, 1)
    scale = torch.tensor(0.00390625, dtype=torch.float32)
    attached = attach_native_stored_linear(parent, "linear", _fp8_weight(), torch.tensor([0.25]), scale)

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
