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
        "background": "",
        "seed": 9,
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "mu": 0.0,
        "std": 1.75,
        "sampler": "euler",
    }
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "quality": "turbo"}
    )["steps"] == 12
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "quality": "quality"}
    ) == {
        "prompt": prompt,
        "background": "",
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "steps": 48,
        "mu": 0.0,
        "std": 1.5,
        "sampler": "euler",
    }
    with pytest.raises(ValueError):
        resolve_ideogram4_request(recipe, {"prompt": prompt, "seed": -1})
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "width": 2048, "height": 1024}
    )["width"] == 2048
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "width": 8192, "height": 512}
    )["width"] == 8192
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "width": 8192, "height": 256}
    ) == {
        "prompt": prompt,
        "background": "",
        "seed": 0,
        "width": 8192,
        "height": 256,
        "steps": 20,
        "mu": 0.0,
        "std": 1.75,
        "sampler": "euler",
    }
    assert resolve_ideogram4_request(
        recipe, {"prompt": prompt, "width": 2048, "height": 1040}
    )["height"] == 1040
    with pytest.raises(ValueError):
        resolve_ideogram4_request(
            recipe, {"prompt": prompt, "width": 2048, "height": 1050}
        )
    with pytest.raises(ValueError):
        resolve_ideogram4_request(
            recipe, {"prompt": prompt, "width": 2048, "height": 2064}
        )
    with pytest.raises(ValueError):
        resolve_ideogram4_request(
            recipe, {"prompt": prompt, "width": 16384, "height": 256}
        )


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


@pytest.mark.native
def test_full_lokr_uses_kronecker_product_without_alpha_rescaling(tmp_path):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    from latentslate_engine.ideogram4.adapters import apply_updates, load_updates

    path = tmp_path / "lokr.safetensors"
    prefix = "diffusion_model.layers.0.attention.qkv"
    w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w2 = torch.tensor([[5.0, 6.0, 7.0]])
    tensors = {
        prefix + ".lokr_w1": w1,
        prefix + ".lokr_w2": w2,
        prefix + ".alpha": torch.tensor(64.0),
    }
    save_file(tensors, path)
    artifact = SimpleNamespace(path=path)
    modules = {"layers.0.attention.qkv": SimpleNamespace(in_features=6, out_features=2)}
    updates = load_updates(((artifact, 0.5), (artifact, -0.25)), modules, "cpu")
    base = torch.ones(2, 6, dtype=torch.bfloat16)
    delta = torch.tensor(
        [[5.0, 6.0, 7.0, 10.0, 12.0, 14.0], [15.0, 18.0, 21.0, 20.0, 24.0, 28.0]],
        dtype=torch.bfloat16,
    )
    expected = (base + 0.5 * delta) - 0.25 * delta
    assert torch.equal(apply_updates(base.clone(), updates["layers.0.attention.qkv"]), expected)
    assert load_updates(((artifact, 0.0),), modules, "cpu") == {}
    tensors[prefix + ".lokr_w2"] = torch.ones(2, 3)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="LoKR factor dimensions"):
        load_updates(((artifact, 1.0),), modules, "cpu")
    del tensors[prefix + ".lokr_w2"]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="incomplete pair"):
        load_updates(((artifact, 1.0),), modules, "cpu")
    tensors[prefix + ".lokr_w2"] = w2
    tensors["unknown.weight"] = torch.ones(1)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="unconsumed"):
        load_updates(((artifact, 1.0),), modules, "cpu")


def test_quality_schedules_and_granular_recipe_reach_sigmas():
    from latentslate_engine.ideogram4.sampling import sigmas

    assert len(sigmas(1024, 1024, 20, 0.0, 1.75)) == 21
    assert len(sigmas(1024, 1024, 48, 0.0, 1.5)) == 49
    assert len(sigmas(1024, 1024, 12, 0.5, 1.75)) == 13
    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    assert [item["key"] for item in recipe.surface()] == [
        "prompt",
        "background",
        "width",
        "height",
        "seed",
        "quality",
    ]
    from latentslate_engine.authoring import compile_document, document_from_recipe

    document = document_from_recipe(
        recipe,
        name="Granular",
        recipe_id="00000000-0000-4000-8000-000000000003",
    )
    del document["presets"]
    next(field for field in document["fields"] if field["key"] == "steps")["mode"] = (
        "exposed"
    )
    granular = compile_document(document)
    resolved = resolve_ideogram4_request(granular, {"prompt": "A shape", "steps": 33})
    assert resolved["steps"] == 33
    assert resolved["background"] == ""
    assert len(sigmas(resolved["width"], resolved["height"], resolved["steps"], resolved["mu"], resolved["std"])) == 34


def test_background_is_optional_and_composed_into_the_caption():
    from latentslate_engine.authoring import compile_document, document_from_recipe
    from latentslate_engine.ideogram4.recipes import compose_caption

    recipe = ideogram4_t2i_recipe(
        diffusion="positive.safetensors",
        negative_diffusion="negative.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        tokenizer="tokenizer",
    )
    assert resolve_ideogram4_request(
        recipe, {"prompt": "A golden retriever on a skateboard", "background": "A sunny sidewalk"}
    )["background"] == "A sunny sidewalk"
    document = document_from_recipe(
        recipe,
        name="Synthetic recipe",
        recipe_id="00000000-0000-4000-8000-000000000004",
    )
    document["fields"] = [field for field in document["fields"] if field["key"] != "background"]
    restored = compile_document(document)
    assert restored.resolve({"prompt": "A shape"})["background"] == ""
    assert compose_caption("A golden retriever on a skateboard", "") == "A golden retriever on a skateboard"
    assert compose_caption("A golden retriever on a skateboard", "  ") == "A golden retriever on a skateboard"
    assert (
        compose_caption("A golden retriever on a skateboard", "A sunny sidewalk lined with hedges.")
        == '{"high_level_description":"A golden retriever on a skateboard","compositional_deconstruction":{"background":"A sunny sidewalk lined with hedges.","elements":[]}}'
    )
    structured = '{"high_level_description":"A golden retriever on a skateboard.","compositional_deconstruction":{"background":"copied scene","elements":[{"type":"obj","bbox":[0,0,100,100],"desc":"the dog"}]}}'
    assert (
        compose_caption(structured, "A sunny sidewalk lined with hedges.")
        == '{"high_level_description":"A golden retriever on a skateboard.","compositional_deconstruction":{"background":"A sunny sidewalk lined with hedges.","elements":[{"type":"obj","bbox":[0,0,100,100],"desc":"the dog"}]}}'
    )
