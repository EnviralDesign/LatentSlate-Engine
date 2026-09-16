"""Portable recipe behavior for the dual-transformer image family."""

import pytest

from latentslate_engine.ideogram4.recipes import (
    ideogram4_t2i_recipe,
    resolve_ideogram4_request,
)


def test_structured_prompt_is_preserved_and_fixed_defaults_resolve():
    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    prompt = '{"compositional_deconstruction":{"background":"white","elements":[{"type":"obj","bbox":[100,200,300,400],"desc":"blue circle"}]}}'
    assert resolve_ideogram4_request(recipe, {"prompt": prompt, "seed": 9}) == {
        "prompt": prompt,
        "seed": 9,
        "width": 1024,
        "height": 1024,
    }
    with pytest.raises(ValueError):
        resolve_ideogram4_request(recipe, {"prompt": prompt, "seed": -1})
    with pytest.raises(ValueError):
        resolve_ideogram4_request(recipe, {"prompt": prompt, "width": 2048})


def test_negative_checkpoint_is_a_required_separate_binding():
    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    values = recipe.resolve({"prompt": "A geometric shape"})
    assert values["negative_diffusion"].path != values["diffusion"].path
