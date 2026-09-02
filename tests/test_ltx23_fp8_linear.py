import unittest
from unittest.mock import patch

import torch

from latentslate_engine.ltx23.fp8_linear import Ltx23Fp8Linear
from latentslate_engine.ltx23.ops import _quantize_fp8_input


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


if __name__ == "__main__":
    unittest.main()
