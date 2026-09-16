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


def test_negative_checkpoint_is_a_separate_binding():
    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    values = recipe.resolve({"prompt": "A geometric shape"})
    assert values["negative_diffusion"].path != values["diffusion"].path


def test_single_model_recipe_does_not_resolve_a_negative_artifact(tmp_path):
    from latentslate_engine.authoring import (
        artifact_dependencies,
        compile_document,
        document_from_recipe,
        validate_document,
    )
    from latentslate_engine.ideogram4.contracts import TOKENIZER_FILES
    from latentslate_engine.ideogram4.recipes import resolve_ideogram4_fixed_identity

    for name in ("model", "text", "vae", *TOKENIZER_FILES):
        (tmp_path / name).write_bytes(b"synthetic")
    recipe = ideogram4_t2i_recipe(
        diffusion=tmp_path / "model",
        negative_diffusion=None,
        text_encoder=tmp_path / "text",
        vae=tmp_path / "vae",
        tokenizer=tmp_path,
    )
    identity = resolve_ideogram4_fixed_identity(recipe)
    assert identity.negative_diffusion is None
    assert identity.adapters == ()
    document = document_from_recipe(
        recipe,
        name="Synthetic recipe",
        recipe_id="00000000-0000-4000-8000-000000000002",
    )
    assert (
        compile_document(document).resolve({"prompt": "A shape"})["negative_diffusion"]
        is None
    )
    assert "negative_diffusion" not in {
        d["key"] for d in artifact_dependencies(document)
    }
    assert validate_document(document)["artifact_resolution"]["status"] == "resolved"


def test_saved_recipe_without_adapters_keeps_empty_composition():
    from latentslate_engine.authoring import compile_document, document_from_recipe

    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    document = document_from_recipe(
        recipe,
        name="Synthetic recipe",
        recipe_id="00000000-0000-4000-8000-000000000001",
    )
    document["fields"] = [f for f in document["fields"] if f["key"] != "adapters"]
    restored = compile_document(document)
    assert restored.resolve({"prompt": "A geometric shape"})["adapters"] == ()


@pytest.mark.native
def test_native_lora_alpha_order_zero_and_unknown_tensor(tmp_path):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    from latentslate_engine.ideogram4.adapters import apply_updates, load_updates

    path = tmp_path / "adapter.safetensors"
    prefix = "diffusion_model.layers.0.attention.qkv"
    tensors = {
        prefix + ".lora_A.weight": torch.ones(1, 2),
        prefix + ".lora_B.weight": torch.ones(6, 1),
        prefix + ".alpha": torch.tensor(2.0),
    }
    save_file(tensors, path)
    artifact = SimpleNamespace(path=path)
    modules = {"layers.0.attention.qkv": SimpleNamespace(in_features=2, out_features=6)}
    updates = load_updates(((artifact, 1.0), (artifact, -0.5)), modules, "cpu")
    result = apply_updates(
        torch.zeros(6, 2, dtype=torch.bfloat16), updates["layers.0.attention.qkv"]
    )
    assert torch.equal(result, torch.ones(6, 2))
    assert load_updates(((artifact, 0.0),), modules, "cpu") == {}
    tensors["unknown.weight"] = torch.ones(1)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="unconsumed"):
        load_updates(((artifact, 1.0),), modules, "cpu")
