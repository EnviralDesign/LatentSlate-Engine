import unittest
from unittest.mock import patch

import torch

from latentslate_engine.ltx23.fp8_linear import (
    Ltx23Fp8Linear,
    Ltx23Int8Linear,
    Ltx23Nvfp4Linear,
)
from latentslate_engine.ltx23.ops import (
    _quantize_fp8_input,
    _requantize_patched_nvfp4,
    _stochastic_quantize_nvfp4,
)


class _Checkpoint:
    def __init__(self) -> None:
        self.tensor_names = (
            "layer.weight",
            "layer.weight_scale",
            "layer.bias",
        )
        self.tensors = {
            "layer.weight": torch.zeros((2, 2), dtype=torch.float8_e4m3fn),
            "layer.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer.bias": torch.zeros(2, dtype=torch.bfloat16),
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class _Nvfp4Checkpoint:
    def __init__(self) -> None:
        self.tensor_names = (
            "layer.weight",
            "layer.weight_scale",
            "layer.weight_scale_2",
            "layer.bias",
        )
        self.tensors = {
            "layer.weight": torch.zeros((16, 8), dtype=torch.uint8),
            "layer.weight_scale": torch.zeros((16, 1), dtype=torch.float8_e4m3fn),
            "layer.weight_scale_2": torch.tensor(1.0, dtype=torch.float32),
            "layer.bias": torch.zeros(16, dtype=torch.bfloat16),
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class _Int8Checkpoint:
    def __init__(self) -> None:
        self.tensor_names = (
            "layer.weight",
            "layer.weight_scale",
            "layer.bias",
            "layer.comfy_quant",
        )
        self.tensors = {
            "layer.weight": torch.zeros((2, 2), dtype=torch.int8),
            "layer.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer.bias": torch.zeros(2, dtype=torch.bfloat16),
            "layer.comfy_quant": torch.tensor(
                list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8
            ),
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class Ltx23Fp8LinearTests(unittest.TestCase):
    def test_weight_only_fp8_format_does_not_require_an_input_scale(self) -> None:
        binding = Ltx23Fp8Linear(_Checkpoint(), "layer")

        self.assertIsNone(binding._input_scale)
        self.assertEqual(
            binding.source_size,
            sum(value.nbytes for value in _Checkpoint().tensors.values()),
        )

    def test_missing_input_scale_uses_comfy_unit_activation_scale(self) -> None:
        value = torch.ones((2, 2), dtype=torch.bfloat16)

        with patch(
            "latentslate_engine.ltx23.ops.ck.quantize_per_tensor_fp8",
            return_value=torch.zeros((2, 2), dtype=torch.float8_e4m3fn),
        ) as quantize:
            _quantize_fp8_input(value, None)

        self.assertTrue(
            torch.equal(
                quantize.call_args.args[1],
                torch.ones((), device=value.device, dtype=torch.float32),
            )
        )

    def test_nvfp4_binding_keeps_both_weight_scales_mapped(self) -> None:
        binding = Ltx23Nvfp4Linear(_Nvfp4Checkpoint(), "layer", (16, 16))

        self.assertIsNone(binding._input_scale)
        self.assertEqual(binding._logical_shape, (16, 16))
        self.assertEqual(
            binding.source_size,
            sum(value.nbytes for value in _Nvfp4Checkpoint().tensors.values()),
        )

    def test_patched_nvfp4_requantization_is_seeded_and_uses_float32_scale(self) -> None:
        weight = torch.full((16, 16), 1.5, dtype=torch.bfloat16)

        first = _requantize_patched_nvfp4(weight, "model.layer")
        second = _requantize_patched_nvfp4(weight, "model.layer")

        self.assertEqual(first._params.scale.dtype, torch.float32)
        self.assertTrue(torch.equal(first._qdata, second._qdata))
        self.assertTrue(
            torch.equal(first._params.block_scale, second._params.block_scale)
        )

    def test_nvfp4_large_matrix_requantization_preserves_source_slice_order(self) -> None:
        slice_rows: list[int] = []

        def quantize_block(
            value: torch.Tensor, _: torch.Tensor, __: torch.Generator
        ) -> tuple[torch.Tensor, torch.Tensor]:
            slice_rows.append(value.shape[0])
            marker = len(slice_rows)
            return (
                torch.full(
                    (value.shape[0], value.shape[1] // 2),
                    marker,
                    dtype=torch.uint8,
                ),
                torch.full(
                    (value.shape[0], value.shape[1] // 16),
                    marker,
                    dtype=torch.float8_e4m3fn,
                ),
            )

        with patch(
            "latentslate_engine.ltx23.ops._stochastic_quantize_nvfp4_block",
            side_effect=quantize_block,
        ):
            qdata, _ = _stochastic_quantize_nvfp4(
                torch.empty((4112, 4096), dtype=torch.bfloat16),
                torch.tensor(1.0, dtype=torch.float32),
                seed=42,
            )

        self.assertEqual(slice_rows, [4096, 16])
        self.assertTrue(torch.all(qdata[:4096] == 1))
        self.assertTrue(torch.all(qdata[4096:] == 2))

    def test_int8_binding_reads_the_checkpoint_quantization_metadata(self) -> None:
        binding = Ltx23Int8Linear(_Int8Checkpoint(), "layer")

        self.assertFalse(binding._convrot)
        self.assertEqual(binding._convrot_groupsize, 256)
        self.assertEqual(
            binding.source_size,
            sum(
                value.nbytes
                for name, value in _Int8Checkpoint().tensors.items()
                if name != "layer.comfy_quant"
            ),
        )


if __name__ == "__main__":
    unittest.main()
