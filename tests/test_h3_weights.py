"""Synthetic checkpoint coverage for H3's Kitchen weight layouts."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.native


@pytest.mark.parametrize("group_size,codebook", [(16, True), (32, False)])
def test_w4a8_checkpoint_preserves_kitchen_linear_output(
    tmp_path, group_size, codebook
):
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from safetensors.torch import save_file

    from latentslate_engine.h3.weights import H3Weight, Linear, swiglu_linear
    from latentslate_engine.mapped_checkpoint import MappedCheckpoint

    generator = torch.Generator().manual_seed(17)
    source = torch.randn(32, 256, generator=generator).to(torch.bfloat16)
    expected_weight = QuantizedTensor.from_float(
        source, "AsymW4A8Int8Layout", group_size=group_size, codebook=codebook
    )
    tensors = expected_weight.state_dict("projection.weight")
    # Both SafeTensors FP8 and its byte representation occur in checkpoints.
    if codebook:
        tensors["projection.weight_s_rel"] = tensors["projection.weight_s_rel"].view(
            torch.uint8
        )
    bias = torch.randn(32, generator=generator).to(torch.bfloat16)
    tensors["projection.bias"] = bias
    path = tmp_path / "synthetic.safetensors"
    save_file(tensors, path)
    owner = SimpleNamespace(
        checkpoint=MappedCheckpoint(path), compute_dtype=torch.bfloat16, updates={}
    )
    linear = Linear(256, 32)
    binding = H3Weight(
        owner,
        "projection",
        linear,
        {"format": "asym_w4a8_int8", "params": {"group_size": group_size}},
    )
    # CPU replay isolates storage/dispatch from the separately exercised VBAR owner.
    binding.materialize = lambda: binding.tensors
    binding.unpin = lambda: None
    linear.binding = binding
    inputs = torch.randn(2, 3, 256, generator=generator).to(torch.bfloat16)
    expected = torch.nn.functional.linear(inputs, expected_weight, bias)
    assert torch.equal(linear(inputs), expected)
    activated = torch.cat((inputs, inputs.neg()), dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(inputs) * inputs.neg(), expected_weight, bias
    )
    assert torch.equal(swiglu_linear(linear, activated), expected)


@pytest.mark.parametrize("sidecar", ["weight_s_rel", "weight_s_channel"])
def test_w4a8_missing_scale_is_rejected_during_loading(tmp_path, sidecar):
    import torch
    from safetensors.torch import save_file

    from latentslate_engine.h3.weights import H3Weight, Linear
    from latentslate_engine.mapped_checkpoint import MappedCheckpoint

    tensors = {
        "weight": torch.zeros(32, 128, dtype=torch.int8),
        "weight_s_rel": torch.ones(32, 16),
        "weight_s_channel": torch.ones(32),
    }
    del tensors[sidecar]
    path = tmp_path / "synthetic.safetensors"
    save_file({f"projection.{key}": value for key, value in tensors.items()}, path)
    owner = SimpleNamespace(
        checkpoint=MappedCheckpoint(path), compute_dtype=torch.bfloat16, updates={}
    )
    with pytest.raises(ValueError, match="Missing H3 W4A8 metadata"):
        H3Weight(owner, "projection", Linear(256, 32), {"format": "asym_w4a8_int8"})


def test_lora_patches_fresh_weights_once_and_zero_strength_keeps_base(tmp_path):
    import torch
    from safetensors.torch import save_file

    from latentslate_engine.h3.adapters import load_updates
    from latentslate_engine.h3.weights import H3Weight, Linear
    from latentslate_engine.mapped_checkpoint import MappedCheckpoint

    base = torch.eye(4, dtype=torch.bfloat16)
    path = tmp_path / "synthetic-base.safetensors"
    save_file({"projection.weight": base}, path)
    adapter = tmp_path / "synthetic-adapter.safetensors"
    save_file(
        {
            "diffusion_model.projection.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.projection.lora_B.weight": torch.ones(4, 2),
            "diffusion_model.projection.alpha": torch.tensor(4.0),
        },
        adapter,
    )
    linear = Linear(4, 4, bias=False)
    modules = {"projection": linear}
    owner = SimpleNamespace(
        checkpoint=MappedCheckpoint(path),
        compute_dtype=torch.bfloat16,
        updates=load_updates([(str(adapter), 0.5)], modules),
    )
    binding = H3Weight(owner, "projection", linear, {})
    binding.signature = object()
    resident = base.clone()
    patched = binding.patch_weight(resident, storage=resident)
    assert torch.equal(patched, base + 2)
    binding.resident = True
    assert torch.equal(binding.patch_weight(resident, storage=resident), base + 2)
    binding.resident = False
    reloaded = base.clone()
    assert torch.equal(binding.patch_weight(reloaded, storage=reloaded), patched)
    owner.updates = load_updates([(str(adapter), 0.0)], modules)
    restored = base.clone()
    assert torch.equal(binding.patch_weight(restored, storage=restored), base)


def test_lora_does_not_silently_accept_unknown_or_incomplete_pairs(tmp_path):
    import torch
    from safetensors.torch import save_file

    from latentslate_engine.h3.adapters import load_updates
    from latentslate_engine.h3.weights import Linear

    adapter = tmp_path / "synthetic.safetensors"
    save_file({"projection.lora_A.weight": torch.ones(2, 4)}, adapter)
    with pytest.raises(ValueError, match="incomplete pair"):
        load_updates([(str(adapter), 1.0)], {"projection": Linear(4, 4)})
