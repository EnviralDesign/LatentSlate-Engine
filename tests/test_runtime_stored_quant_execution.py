from __future__ import annotations

import pytest
import torch

from latentslate_engine.runtime.framework.stored_quant import StoredFP8Int8Linear


def _fp8_weight():
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    qdata = torch.zeros((16, 16), dtype=torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(
        scale=torch.tensor(0.5), orig_dtype=torch.float32, orig_shape=(16, 16)
    )
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


def test_fp8_dispatch_is_fail_closed_and_preserves_rank(monkeypatch: pytest.MonkeyPatch):
    linear = StoredFP8Int8Linear(_fp8_weight(), input_scale=torch.tensor(0.25))
    calls: list[tuple[int, ...]] = []

    def dispatch(flat_input: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(flat_input.shape))
        return torch.zeros((flat_input.shape[0], linear.weight.shape[0]))

    monkeypatch.setattr(linear, "_native_fp8_matmul", dispatch)
    output = linear(torch.ones((2, 3, 16)))

    assert output.shape == (2, 3, 16)
    assert calls == [(6, 16)]
    assert linear.native_dispatch_count == 1
    assert linear.native_rejection_count == 0
    assert linear.dense_fallback_count == 0


def test_fp8_dispatch_failure_is_counted_without_fallback(monkeypatch: pytest.MonkeyPatch):
    linear = StoredFP8Int8Linear(_fp8_weight())

    def fail(_input: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(linear, "_native_fp8_matmul", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        linear(torch.ones((1, 16)))

    assert linear.native_dispatch_count == 0
    assert linear.native_rejection_count == 1
    assert linear.dense_fallback_count == 0


def test_int8_dispatch_uses_stored_weight_without_dense_fallback():
    weight = _int8_weight()
    linear = StoredFP8Int8Linear(weight)
    input = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

    output = linear(input)

    expected = torch.nn.functional.linear(input, weight.dequantize())
    assert output.shape == (1, 1, 1)
    assert torch.allclose(output, expected)
    assert linear.int8_dispatch_count == 1
    assert linear.native_dispatch_count == 0
    assert linear.dense_fallback_count == 0


@pytest.mark.parametrize(
    "input_scale",
    [
        torch.tensor(0.0),
        torch.tensor(float("nan")),
        torch.tensor([0.25]),
        torch.tensor(0.25, dtype=torch.float16),
    ],
)
def test_invalid_input_scale_is_rejected(input_scale: torch.Tensor):
    with pytest.raises(ValueError, match="positive finite F32 scalar"):
        StoredFP8Int8Linear(_fp8_weight(), input_scale=input_scale)


def test_move_stored_storage_preserves_layout_and_scale():
    weight = _fp8_weight()
    original_qdata = weight._qdata.clone()
    original_scale = weight.params.scale.clone()
    linear = StoredFP8Int8Linear(weight, bias=torch.arange(16, dtype=torch.float32))

    linear.move_stored_storage("cpu")

    assert linear.weight._layout_cls == "TensorCoreFP8Layout"
    assert linear.weight.params.orig_shape == (16, 16)
    assert torch.equal(linear.weight._qdata.view(torch.uint8), original_qdata.view(torch.uint8))
    assert torch.equal(linear.weight.params.scale, original_scale)
    assert torch.equal(linear.bias, torch.arange(16, dtype=torch.float32))
