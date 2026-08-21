from __future__ import annotations

import json
import struct
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from latentslate_engine.stored_quant import (
    discover_stored_layer,
    read_safetensors_header,
    restore_global_fp8_tensor,
    restore_nvfp4_tensor,
)


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


def test_shared_header_reader_returns_exact_object_without_payload_read(tmp_path: Path):
    header = b'{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}'
    path = tmp_path / "header.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"payload")

    assert read_safetensors_header(path) == {
        "weight": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}
    }


def test_shared_header_reader_rejects_duplicate_keys_and_invalid_bounds(tmp_path: Path):
    duplicate = b'{"weight":{},"weight":{}}'
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(struct.pack("<Q", len(duplicate)) + duplicate)

    with pytest.raises(ValueError, match="duplicate SafeTensors header key"):
        read_safetensors_header(path)
    with pytest.raises(ValueError, match="exceeds bounds"):
        read_safetensors_header(path, 8)
    with pytest.raises(ValueError, match="invalid SafeTensors artifact size"):
        read_safetensors_header(path, True)


def test_shared_fp8_restore_preserves_stored_objects_bits_and_logical_contract():
    qdata = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.125, dtype=torch.float32)

    restored = restore_global_fp8_tensor(qdata, scale, torch.bfloat16)

    assert restored._qdata is qdata
    assert torch.equal(restored._qdata.view(torch.uint8), qdata.view(torch.uint8))
    assert restored.params.scale is scale
    assert restored.params.orig_shape == (3, 4)
    assert restored.params.orig_dtype is torch.bfloat16
    assert restored._layout_cls == "TensorCoreFP8Layout"


@pytest.mark.parametrize(
    ("qdata", "scale"),
    [
        (torch.zeros((2, 2), dtype=torch.float32), torch.tensor(0.5)),
        (torch.zeros((4,), dtype=torch.float8_e4m3fn), torch.tensor(0.5)),
        (torch.zeros((2, 2), dtype=torch.float8_e4m3fn), torch.tensor(0.0)),
        (torch.zeros((2, 2), dtype=torch.float8_e4m3fn), torch.tensor(float("nan"))),
        (torch.zeros((2, 2), dtype=torch.float8_e4m3fn), torch.tensor(0.5, dtype=torch.float16)),
        (torch.zeros((2, 2), dtype=torch.float8_e4m3fn), torch.tensor([0.5])),
    ],
)
def test_shared_fp8_restore_rejects_malformed_storage(qdata: torch.Tensor, scale: torch.Tensor):
    with pytest.raises(ValueError, match="stored quant"):
        restore_global_fp8_tensor(qdata, scale, torch.bfloat16)


def test_shared_nvfp4_restore_preserves_padded_storage_and_logical_contract():
    qdata = torch.arange(32 * 24, dtype=torch.int64).reshape(32, 24).to(torch.uint8)
    block_scale = torch.ones((128, 4), dtype=torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.25, dtype=torch.float32)

    restored = restore_nvfp4_tensor(
        qdata,
        block_scale,
        tensor_scale,
        (17, 33),
        torch.bfloat16,
    )

    assert restored._qdata is qdata
    assert torch.equal(restored._qdata, qdata)
    assert restored.params.scale is tensor_scale
    assert restored.params.block_scale is block_scale
    assert restored.params.orig_shape == (17, 33)
    assert restored.params.orig_dtype is torch.bfloat16
    assert restored._layout_cls == "TensorCoreNVFP4Layout"


@pytest.mark.parametrize(
    ("qdata", "block_scale", "tensor_scale", "logical_shape"),
    [
        (
            torch.zeros((4, 3), dtype=torch.float8_e4m3fn),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            (4, 6),
        ),
        (
            torch.zeros((12,), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            (4, 6),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float32),
            torch.tensor(0.5),
            (4, 6),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4,), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            (4, 6),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.0),
            (4, 6),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            (5, 6),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            (4, 7),
        ),
        (
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.5),
            [4, 6],
        ),
    ],
)
def test_shared_nvfp4_restore_rejects_malformed_storage(
    qdata: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    logical_shape,
):
    with pytest.raises(ValueError, match="stored quant"):
        restore_nvfp4_tensor(
            qdata,
            block_scale,
            tensor_scale,
            logical_shape,
            torch.bfloat16,
        )


def test_current_stored_fp8_is_lazy_and_restores_with_requested_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
