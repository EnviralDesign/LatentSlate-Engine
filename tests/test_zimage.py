"""Synthetic regressions for Z-Image's generation and recipe boundaries."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from latentslate_engine.zimage.adapters import apply_updates, load_updates
from latentslate_engine.zimage.contracts import validate_request
from latentslate_engine.zimage.recipes import resolve_zimage_request, zimage_t2i_recipe
from latentslate_engine.zimage.sampling import noise, res_multistep, sigmas

pytestmark = pytest.mark.native


def test_seed_replay_preserves_global_random_state():
    before = torch.random.get_rng_state().clone()
    first = noise(42, 256, 256)
    assert torch.equal(first, noise(42, 256, 256))
    assert not torch.equal(first, noise(43, 256, 256))
    assert torch.equal(before, torch.random.get_rng_state())


def test_solver_finishes_at_last_denoised_prediction():
    predictions = []

    def denoise(x, sigma):
        result = x * 0.25 + sigma.reshape(-1, 1, 1, 1) * 0.125
        predictions.append(result)
        return result

    output = res_multistep(denoise, noise(42, 64, 64), sigmas())
    assert len(predictions) == 8
    torch.testing.assert_close(output, predictions[-1], rtol=0, atol=1e-7)


def test_recipe_surface_resolves_seed_and_canvas():
    recipe = zimage_t2i_recipe(
        diffusion="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    values = resolve_zimage_request(recipe, {"prompt": "A geometric shape", "seed": 43})
    assert values == {
        "prompt": "A geometric shape",
        "seed": 43,
        "width": 1024,
        "height": 1024,
    }
    with pytest.raises((TypeError, ValueError)):
        resolve_zimage_request(recipe, {"prompt": "A geometric shape", "seed": -1})


@pytest.mark.parametrize(
    "width,height,seed",
    [(1025, 1024, 0), (2048, 1024, 0), (1024, 1024, -1), (1024, 1024, True)],
)
def test_request_rejects_invalid_canvas_or_seed(width, height, seed):
    with pytest.raises((TypeError, ValueError)):
        validate_request(width, height, seed)


def test_lora_split_qkv_alpha_order_and_zero_strength(tmp_path):
    path = tmp_path / "adapter.safetensors"
    prefix = "diffusion_model.layers.0.attention."
    save_file(
        {
            prefix + "to_q.lora_A.weight": torch.ones(1, 2),
            prefix + "to_q.lora_B.weight": torch.ones(2, 1),
            prefix + "to_q.alpha": torch.tensor(2.0),
            prefix + "to_v.lora_A.weight": torch.ones(1, 2),
            prefix + "to_v.lora_B.weight": torch.ones(2, 1),
        },
        path,
    )
    artifact = SimpleNamespace(path=path)
    modules = {"layers.0.attention.qkv": SimpleNamespace(in_features=2, out_features=6)}
    updates = load_updates(((artifact, 1.0), (artifact, -0.5)), modules, "cpu")
    result = apply_updates(
        torch.zeros(6, 2, dtype=torch.bfloat16), updates["layers.0.attention.qkv"]
    )
    assert torch.equal(result[:2], torch.ones(2, 2))
    assert torch.count_nonzero(result[2:4]) == 0
    assert torch.equal(result[4:], torch.full((2, 2), 0.5))
    assert load_updates(((artifact, 0.0),), modules, "cpu") == {}


def test_lora_rejects_unconsumed_weights(tmp_path):
    path = tmp_path / "adapter.safetensors"
    save_file({"unsupported.weight": torch.ones(2, 2)}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_updates(((SimpleNamespace(path=path), 1.0),), {}, "cpu")
