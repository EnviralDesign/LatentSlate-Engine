from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from latentslate_engine.stored_quant import discover_stored_layer


def _marker(value: dict[str, object]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value).encode("utf-8")), dtype=torch.uint8)


def _save_fp8(path: Path, *, marker: torch.Tensor | None = None, legacy: bool = False) -> None:
    tensors = {
        "model.diffusion_model.blocks.0.weight": torch.tensor([[1, 2]], dtype=torch.float8_e4m3fn),
        "model.diffusion_model.blocks.0.scale_weight" if legacy else "model.diffusion_model.blocks.0.weight_scale": torch.tensor(0.5),
    }
    if marker is not None:
        tensors["model.diffusion_model.blocks.0.comfy_quant"] = marker
    save_file(tensors, path)


def test_current_comfy_fp8_is_lazy_and_restores_with_requested_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "fp8.safetensors"
    _save_fp8(path, marker=_marker({"format": "float8_e4m3fn"}))

    import safetensors

    with monkeypatch.context() as scoped:
        scoped.setattr(safetensors, "safe_open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("payload read")))
        layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/float8_e4m3fn")

    restored = layer.materialize(torch.float32)
    assert restored.storage_dtype == torch.float8_e4m3fn
    assert restored.dtype == torch.float32
    assert torch.allclose(layer.dequantize_cpu(torch.float32), torch.tensor([[0.5, 1.0]]))


def test_legacy_fp8_without_marker_is_supported(tmp_path: Path):
    path = tmp_path / "legacy.safetensors"
    _save_fp8(path, legacy=True)

    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_legacy/scaled_fp8_e4m3fn")
    assert layer.marker_key is None
    assert torch.allclose(layer.dequantize_cpu(torch.float32), torch.tensor([[0.5, 1.0]]))


def test_int8_convrot_uses_actual_marker_and_nested_global_metadata(tmp_path: Path):
    path = tmp_path / "int8.safetensors"
    marker = _marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4, "per_row": True})
    save_file(
        {
            "blocks.0.weight": torch.tensor([[2, -2, 0, 0]], dtype=torch.int8),
            "blocks.0.weight_scale": torch.tensor([[0.25]]),
            "blocks.0.comfy_quant": marker,
        },
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.0": {"format": "int8_tensorwise", "convrot": True, "params": {"convrot_groupsize": 4}}}})},
    )

    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/int8_tensorwise_convrot")
    restored = layer.materialize(torch.float32)
    assert layer.group_size == 4
    assert restored.storage_dtype == torch.int8
    # This is deliberately the QuantizedTensor's layout dequantizer, not raw qdata * scale.
    assert torch.equal(layer.dequantize_cpu(torch.float32), restored.dequantize())
    assert torch.allclose(restored.dequantize(), torch.tensor([[0.0, 0.0, 0.5, -0.5]]))


def test_int8_convrot_accepts_marker_params_layout(tmp_path: Path):
    path = tmp_path / "nested-marker-int8.safetensors"
    save_file(
        {
            "blocks.0.weight": torch.tensor([[2, -2, 0, 0]], dtype=torch.int8),
            "blocks.0.weight_scale": torch.tensor([[0.25]]),
            "blocks.0.comfy_quant": _marker(
                {"params": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}}
            ),
        },
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}}})},
    )

    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/int8_tensorwise_convrot")
    assert layer.materialize(torch.float32).storage_dtype == torch.int8


def test_int8_convrot_rejects_explicit_non_per_row_marker(tmp_path: Path):
    path = tmp_path / "non-per-row-int8.safetensors"
    save_file(
        {
            "blocks.0.weight": torch.tensor([[2, -2, 0, 0]], dtype=torch.int8),
            "blocks.0.weight_scale": torch.tensor([[0.25]]),
            "blocks.0.comfy_quant": _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4, "per_row": False}
            ),
        },
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}}})},
    )

    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/int8_tensorwise_convrot")

    with pytest.raises(ValueError, match="missing ConvRot marker/group size"):
        layer.materialize(torch.float32)


@pytest.mark.parametrize(
    ("scale", "marker", "message"),
    [
        (torch.tensor(float("nan")), _marker({"format": "float8_e4m3fn"}), "FP8 scale must be one finite F32 scalar"),
        (torch.tensor([0.5], dtype=torch.float16), _marker({"format": "float8_e4m3fn"}), "weight must be 2D with F32 scale"),
        (torch.tensor([0.5]), _marker({"format": "float8_e4m3fn"}), "FP8 scale must be one finite F32 scalar"),
        (torch.tensor([0.5]), torch.tensor([1], dtype=torch.int8), "marker must be bounded U8 JSON"),
        (torch.tensor(0.5), _marker({"format": "wrong"}), "invalid Comfy FP8"),
    ],
)
def test_current_fp8_fails_closed_for_malformed_storage(tmp_path: Path, scale: torch.Tensor, marker: torch.Tensor, message: str):
    path = tmp_path / "malformed.safetensors"
    save_file(
        {
            "blocks.0.weight": torch.tensor([[1, 2]], dtype=torch.float8_e4m3fn),
            "blocks.0.weight_scale": scale,
            "blocks.0.comfy_quant": marker,
        },
        path,
    )
    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/float8_e4m3fn")
    with pytest.raises(ValueError, match=message):
        layer.materialize(torch.float32)


def test_materialization_rejects_replaced_artifact_after_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "replacement.safetensors"
    _save_fp8(path, marker=_marker({"format": "float8_e4m3fn"}))
    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/float8_e4m3fn")

    import safetensors

    real_safe_open = safetensors.safe_open

    @contextmanager
    def replace_after_open(*args, **kwargs):
        with real_safe_open(*args, **kwargs) as handle:
            _save_fp8(path, marker=_marker({"format": "float8_e4m3fn", "replacement": True}))
            yield handle

    monkeypatch.setattr(safetensors, "safe_open", replace_after_open)

    with pytest.raises(ValueError, match="identity changed"):
        layer.materialize(torch.float32)


def test_int8_convrot_rejects_marker_global_metadata_disagreement(tmp_path: Path):
    path = tmp_path / "mismatched-int8.safetensors"
    save_file(
        {
            "blocks.0.weight": torch.tensor([[2, -2, 0, 0]], dtype=torch.int8),
            "blocks.0.weight_scale": torch.tensor([[0.25]]),
            "blocks.0.comfy_quant": _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 8}
            ),
        },
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}}})},
    )

    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/int8_tensorwise_convrot")
    with pytest.raises(ValueError, match="missing ConvRot marker/group size"):
        layer.materialize(torch.float32)


def test_int8_convrot_requires_exact_row_scale_shape(tmp_path: Path):
    path = tmp_path / "bad-int8-scale.safetensors"
    save_file(
        {
            "blocks.0.weight": torch.tensor([[2, -2, 0, 0]], dtype=torch.int8),
            "blocks.0.weight_scale": torch.tensor(0.25),
            "blocks.0.comfy_quant": _marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
            ),
        },
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}}})},
    )
    layer = discover_stored_layer(path, "blocks.0.weight", "comfy_quant/int8_tensorwise_convrot")
    with pytest.raises(ValueError, match=r"ConvRot scale must be finite F32 \[rows, 1\]"):
        layer.materialize(torch.float32)
