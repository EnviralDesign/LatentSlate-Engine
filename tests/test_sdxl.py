"""Synthetic SDXL recipe, bootstrap and sampler regression coverage."""

from dataclasses import replace

import pytest

from latentslate_engine.sdxl.recipes import (
    resolve_sdxl_fixed_identity,
    resolve_sdxl_request,
    sdxl_t2i_recipe,
)


def test_embedded_vae_round_trip_and_override_identity(tmp_path):
    from latentslate_engine.authoring import (
        artifact_dependencies,
        compile_document,
        document_from_recipe,
    )
    from latentslate_engine.sdxl.contracts import TOKENIZER_FILES

    for name in ("checkpoint", "decoder", *TOKENIZER_FILES):
        (tmp_path / name).write_bytes(b"synthetic")
    recipe = sdxl_t2i_recipe(checkpoint=tmp_path / "checkpoint", tokenizer=tmp_path)
    embedded = resolve_sdxl_fixed_identity(recipe)
    assert embedded.vae is None
    document = document_from_recipe(
        recipe, name="Synthetic", recipe_id="00000000-0000-4000-8000-000000000001"
    )
    assert "vae" not in {d["key"] for d in artifact_dependencies(document)}
    assert resolve_sdxl_fixed_identity(compile_document(document)) == embedded
    override = sdxl_t2i_recipe(
        checkpoint=tmp_path / "checkpoint", tokenizer=tmp_path, vae=tmp_path / "decoder"
    )
    assert resolve_sdxl_fixed_identity(override) != embedded


def test_sampling_controls_can_be_fixed_without_changing_prompt_ownership():
    recipe = sdxl_t2i_recipe(checkpoint="checkpoint", tokenizer="tokenizer")
    recipe = replace(
        recipe,
        fields=tuple(
            replace(f, exposed=False, value=6.0) if f.capability.key == "cfg" else f
            for f in recipe.fields
        ),
    )
    resolved = resolve_sdxl_request(
        recipe, {"prompt": "A geometric shape", "negative_prompt": "text", "seed": 0}
    )
    assert resolved["cfg"] == 6.0 and resolved["negative_prompt"] == "text"
    with pytest.raises(ValueError, match="fixed"):
        resolve_sdxl_request(recipe, {"prompt": "A shape", "cfg": 7.0})


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"steps": 0},
        {"steps": True},
        {"cfg": float("nan")},
        {"cfg": 0},
        {"sampler": "unsupported"},
        {"scheduler": "unsupported"},
        {"width": 1025},
        {"width": 2048, "height": 1024},
        {"negative_prompt": None},
    ],
)
def test_invalid_controls_rejected_before_native_loading(overrides):
    recipe = sdxl_t2i_recipe(checkpoint="checkpoint", tokenizer="tokenizer")
    with pytest.raises((TypeError, ValueError)):
        resolve_sdxl_request(recipe, {"prompt": "A shape", **overrides})


def test_official_bootstrap_covers_every_builtin_dependency(tmp_path):
    from latentslate_engine.bootstrap import selected_assets
    from latentslate_engine.sdxl.contracts import TOKENIZER_FILES
    from latentslate_engine.service import SDXLModelPaths

    paths = SDXLModelPaths.from_root(tmp_path / "models")
    assets = selected_assets(["sdxl"])
    assert {tmp_path / a["path"] for a in assets} == {
        paths.checkpoint,
        *(paths.tokenizer / f for f in TOKENIZER_FILES),
    }
    assert all(a["reference"]["source"] == "huggingface" for a in assets)


@pytest.mark.native
def test_karras_schedule_preserves_scalar_endpoint_rounding():
    import torch

    from latentslate_engine.sdxl.sampling import schedule

    sigmas, _ = schedule(25, "karras")
    # Frozen reference values at the first divergence caused by tensor endpoints.
    assert sigmas[8].item() == 3.168604850769043
    assert sigmas[-1].item() == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])
