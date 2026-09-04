from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from latentslate_engine.klein9b.contracts import TOKENIZER_FILES
from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    Recipe,
    exposed,
    fixed,
    klein9b_two_image_recipe,
    ltx23_t2v_recipe,
    resolve_klein9b_two_image,
    resolve_ltx23_t2v,
    resolve_wan2214b_t2v,
    wan2214b_t2v_recipe,
)


def _file(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return path


def _klein_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    diffusion = _file(root, "diffusion.safetensors")
    text_encoder = _file(root, "text-encoder.safetensors")
    vae = _file(root, "vae.safetensors")
    tokenizer = root / "support" / "tokenizer"
    for name in TOKENIZER_FILES:
        _file(tokenizer, name)
    _file(tokenizer.parent / "text_encoder", "config.json")
    return diffusion, text_encoder, vae, tokenizer


def test_generic_policy_supports_fixed_exposed_choice_and_optional_values() -> None:
    definition = Recipe(
        "small.policy",
        (
            fixed(Capability("revision", "integer"), 1),
            exposed(
                Capability("mode", "choice"),
                default="fast",
                choices=("fast", "quality"),
            ),
            exposed(Capability("note", "text", optional=True), default=None),
        ),
    )

    assert definition.resolve({}) == {"revision": 1, "mode": "fast", "note": None}
    assert definition.resolve({"mode": "quality", "note": "keep detail"}) == {
        "revision": 1,
        "mode": "quality",
        "note": "keep detail",
    }
    with pytest.raises(ValueError, match="fixed"):
        definition.resolve({"revision": 2})
    with pytest.raises(ValueError, match="one of"):
        definition.resolve({"mode": "unknown"})


def test_ltx_recipe_resolves_defaults_constraints_and_ordered_adapters(
    tmp_path: Path,
) -> None:
    first = Adapter(Artifact(tmp_path / "style.safetensors"), 0.35)
    second = Adapter(Artifact(tmp_path / "motion.safetensors"), 0.8)
    definition = ltx23_t2v_recipe(
        checkpoint=tmp_path / "model.safetensors",
        text_checkpoint=tmp_path / "text.safetensors",
        upsampler=tmp_path / "upsampler.safetensors",
        transformer_adapters=(first, second),
    )

    identity, request = resolve_ltx23_t2v(definition, {"prompt": "A glass city"})

    assert identity.transformer_lora_path is None
    assert identity.transformer_loras == (
        (str(first.artifact.path), 0.35),
        (str(second.artifact.path), 0.8),
    )
    assert request == {
        "prompt": "A glass city",
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 0,
    }
    assert [field["key"] for field in definition.surface()] == [
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]

    single_definition = ltx23_t2v_recipe(
        checkpoint=tmp_path / "model.safetensors",
        text_checkpoint=tmp_path / "text.safetensors",
        upsampler=tmp_path / "upsampler.safetensors",
        transformer_adapters=(first,),
    )
    single_identity, _ = resolve_ltx23_t2v(
        single_definition, {"prompt": "A glass city"}
    )
    assert single_identity.transformer_lora_path == str(first.artifact.path)
    assert single_identity.lora_strength == 0.35
    assert single_identity.transformer_loras == ()

    _, changed = resolve_ltx23_t2v(
        definition,
        {
            "prompt": "A glass city",
            "width": 768,
            "height": 512,
            "duration_seconds": 4.5,
            "seed": 9,
        },
    )
    assert changed["width"] == 768
    assert changed["duration_seconds"] == 4.5
    with pytest.raises(ValueError, match="fixed"):
        definition.resolve({"checkpoint": Artifact(tmp_path / "other")})
    with pytest.raises(ValueError, match="missing required"):
        definition.resolve({})
    with pytest.raises(ValueError, match="increments"):
        definition.resolve({"prompt": "x", "duration_seconds": 4.25})
    with pytest.raises(ValueError, match="must not exceed"):
        definition.resolve({"prompt": "x", "width": 14720, "height": 128})


def test_klein_two_image_recipe_preserves_reference_and_lora_order(
    tmp_path: Path,
) -> None:
    diffusion, text_encoder, vae, tokenizer = _klein_paths(tmp_path)
    first_lora = _file(tmp_path, "first-lora.safetensors")
    second_lora = _file(tmp_path, "second-lora.safetensors")
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    definition = klein9b_two_image_recipe(
        diffusion=diffusion,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        loras=(first_lora, second_lora),
    )

    identity, request = resolve_klein9b_two_image(
        definition,
        {
            "prompt": "Put both subjects at a table",
            "image_1": first_image,
            "image_2": second_image,
        },
    )

    assert tuple(item.path for item in identity.loras) == (
        first_lora.resolve(),
        second_lora.resolve(),
    )
    assert request["first_image"] == first_image
    assert request["second_image"] == second_image
    assert request["width"] is None
    assert request["height"] is None
    surface = {field["key"]: field for field in definition.surface()}
    assert surface["loras"]["ordered"] is True
    assert surface["loras"]["collection"] is True
    assert surface["width"]["nullable"] is True
    assert "steps" not in surface

    reversed_identity, changed = resolve_klein9b_two_image(
        definition,
        {
            "prompt": "Put both subjects at a table",
            "image_1": second_image,
            "image_2": first_image,
            "loras": (Artifact(second_lora), Artifact(first_lora)),
            "width": 512,
            "height": 1024,
            "seed": 7,
        },
    )
    assert tuple(item.path for item in reversed_identity.loras) == (
        second_lora.resolve(),
        first_lora.resolve(),
    )
    assert changed["first_image"] == second_image
    assert changed["second_image"] == first_image
    with pytest.raises(ValueError, match="both be provided"):
        definition.resolve(
            {
                "prompt": "x",
                "image_1": first_image,
                "image_2": second_image,
                "width": 512,
            }
        )
    with pytest.raises(ValueError, match="aspect ratio"):
        definition.resolve(
            {
                "prompt": "x",
                "image_1": first_image,
                "image_2": second_image,
                "width": 256,
                "height": 1280,
            }
        )


def test_wan_recipe_keeps_high_low_ownership_and_request_state_out_of_identity(
    tmp_path: Path,
) -> None:
    high_checkpoint = _file(tmp_path, "high.safetensors")
    high_primary = Adapter(Artifact(_file(tmp_path, "high-primary.safetensors")), 0.7)
    high_secondary = Adapter(
        Artifact(_file(tmp_path, "high-secondary.safetensors")), 0.2
    )
    low_checkpoint = _file(tmp_path, "low.safetensors")
    low_primary = Adapter(Artifact(_file(tmp_path, "low-primary.safetensors")), 0.9)
    text_encoder = _file(tmp_path, "umt5.safetensors")
    vae = _file(tmp_path, "wan-vae.safetensors")
    definition = wan2214b_t2v_recipe(
        high_checkpoint=high_checkpoint,
        high_adapters=(high_primary, high_secondary),
        low_checkpoint=low_checkpoint,
        low_adapters=(low_primary,),
        text_encoder=text_encoder,
        vae=vae,
        negative_prompt="fixed negative",
    )

    baseline, request = resolve_wan2214b_t2v(definition, {"prompt": "A robot"})
    assert baseline.high_lora == str(high_primary.artifact.path)
    assert baseline.high_secondary_lora == str(high_secondary.artifact.path)
    assert baseline.high_lora_strength == 0.7
    assert baseline.high_secondary_lora_strength == 0.2
    assert baseline.low_lora == str(low_primary.artifact.path)
    assert baseline.low_secondary_lora is None
    assert request["frame_count"] == 81
    assert [field["key"] for field in definition.surface()] == [
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]

    changed, changed_request = resolve_wan2214b_t2v(
        definition,
        {
            "prompt": "A different robot",
            "width": 832,
            "height": 480,
            "duration_seconds": 1.0,
            "seed": 19,
        },
    )
    assert changed.identity == baseline.identity
    assert changed_request != request
    assert changed_request["frame_count"] == 17
    with pytest.raises(ValueError, match="fixed"):
        definition.resolve({"steps": 6})
    with pytest.raises(ValueError, match="increments"):
        definition.resolve({"prompt": "x", "duration_seconds": 1.1})
    with pytest.raises(ValueError, match="aspect ratio"):
        definition.resolve({"prompt": "x", "width": 1280, "height": 480})


def test_recipe_import_and_resolution_do_not_import_torch() -> None:
    script = """
import sys
from latentslate_engine.recipe import ltx23_t2v_recipe, resolve_ltx23_t2v
recipe = ltx23_t2v_recipe(checkpoint='model', text_checkpoint='text', upsampler='up')
resolve_ltx23_t2v(recipe, {'prompt': 'test'})
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
