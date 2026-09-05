from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from latentslate_engine.klein9b.contracts import TOKENIZER_FILES
from latentslate_engine.klein9b.recipes import (
    klein9b_two_image_recipe,
    resolve_klein9b_two_image,
)
from latentslate_engine.ltx23.contracts import Ltx23FlfIdentity, Ltx23T2VIdentity
from latentslate_engine.ltx23.recipes import (
    LTX23_FLF_CAPABILITIES,
    LTX23_T2V_CAPABILITIES,
    ltx23_flf_recipe,
    ltx23_t2v_locked_recipe,
    ltx23_t2v_recipe,
    ltx23_t2v_tunable_recipe,
    resolve_ltx23_flf,
    resolve_ltx23_t2v,
)
from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    CapabilitySet,
    Recipe,
    exposed,
    fixed,
)
from latentslate_engine.wan2214b.flf import WanFLFRecipe
from latentslate_engine.wan2214b.i2v import WanI2VRecipe
from latentslate_engine.wan2214b.recipes import (
    WAN2214B_FLF_CAPABILITIES,
    WAN2214B_I2V_CAPABILITIES,
    WAN2214B_T2V_CAPABILITIES,
    resolve_wan2214b_flf,
    resolve_wan2214b_i2v,
    resolve_wan2214b_t2v,
    wan2214b_flf_recipe,
    wan2214b_i2v_recipe,
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
    revision = Capability("revision", "integer")
    mode = Capability("mode", "choice", choices=("fast", "quality"))
    note = Capability("note", "text", optional=True)
    capabilities = CapabilitySet("small", (revision, mode, note))
    definition = Recipe(
        "small.policy",
        capabilities,
        (
            fixed(revision, 1),
            exposed(mode, default="fast"),
            exposed(note, default=None),
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


def _ltx_product_recipes(
    tmp_path: Path,
) -> tuple[Recipe, Recipe, tuple[Adapter, Adapter]]:
    adapters = (
        Adapter(Artifact(tmp_path / "style.safetensors"), 0.35),
        Adapter(Artifact(tmp_path / "motion.safetensors"), 0.8),
    )
    artifacts = {
        "checkpoint": tmp_path / "model.safetensors",
        "text_checkpoint": tmp_path / "text.safetensors",
        "upsampler": tmp_path / "upsampler.safetensors",
        "transformer_adapters": adapters,
    }
    return (
        ltx23_t2v_locked_recipe(**artifacts),
        ltx23_t2v_tunable_recipe(**artifacts),
        adapters,
    )


def test_ltx_products_reuse_one_capability_set_but_derive_different_surfaces(
    tmp_path: Path,
) -> None:
    locked, tunable, _ = _ltx_product_recipes(tmp_path)

    assert locked.capabilities is LTX23_T2V_CAPABILITIES
    assert tunable.capabilities is LTX23_T2V_CAPABILITIES
    locked_capabilities = {
        field.capability.key: field.capability for field in locked.fields
    }
    tunable_capabilities = {
        field.capability.key: field.capability for field in tunable.fields
    }
    assert all(
        locked_capabilities[key] is tunable_capabilities[key]
        for key in locked_capabilities
    )

    assert [field["key"] for field in locked.surface()] == ["prompt", "seed"]
    assert [field["key"] for field in tunable.surface()] == [
        "transformer_adapter_strengths",
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]
    strength_surface = tunable.surface()[0]
    assert strength_surface["ordered"] is True
    assert strength_surface["collection"] is True
    assert strength_surface["constraints"] == {"min": 0.0, "max": 1.0}
    assert "transformer_adapter_artifacts" not in {
        field["key"] for field in tunable.surface()
    }


def test_ltx_locked_and_tunable_products_resolve_to_existing_family_inputs(
    tmp_path: Path,
) -> None:
    locked, tunable, adapters = _ltx_product_recipes(tmp_path)

    locked_identity, locked_request = resolve_ltx23_t2v(
        locked, {"prompt": "A locked glass city", "seed": 3}
    )
    tunable_identity, tunable_request = resolve_ltx23_t2v(
        tunable,
        {
            "prompt": "A tunable glass city",
            "width": 1024,
            "height": 512,
            "duration_seconds": 4.0,
            "seed": 7,
            "transformer_adapter_strengths": (0.6, 0.2),
        },
    )

    assert isinstance(locked_identity, Ltx23T2VIdentity)
    assert isinstance(tunable_identity, Ltx23T2VIdentity)
    assert locked_request == {
        "prompt": "A locked glass city",
        "width": 768,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 3,
    }
    assert tunable_request == {
        "prompt": "A tunable glass city",
        "width": 1024,
        "height": 512,
        "duration_seconds": 4.0,
        "seed": 7,
    }
    assert locked_identity.transformer_loras == (
        (str(adapters[0].artifact.path), 0.35),
        (str(adapters[1].artifact.path), 0.8),
    )
    assert tunable_identity.transformer_loras == (
        (str(adapters[0].artifact.path), 0.6),
        (str(adapters[1].artifact.path), 0.2),
    )

    with pytest.raises(ValueError, match="fixed"):
        locked.resolve({"prompt": "x", "width": 512})
    with pytest.raises(ValueError, match="fixed"):
        tunable.resolve(
            {
                "prompt": "x",
                "transformer_adapter_artifacts": (
                    Artifact(tmp_path / "replacement.safetensors"),
                ),
            }
        )


def test_ltx_recipe_bounds_narrow_family_domain_and_adapter_controls(
    tmp_path: Path,
) -> None:
    _, tunable, _ = _ltx_product_recipes(tmp_path)
    duration = LTX23_T2V_CAPABILITIES["duration_seconds"]

    assert duration.normalize(1.0) == 1.0
    with pytest.raises(ValueError, match="at least 2.0"):
        tunable.resolve({"prompt": "x", "duration_seconds": 1.0})
    with pytest.raises(ValueError, match="at most 1.0"):
        tunable.resolve({"prompt": "x", "transformer_adapter_strengths": (0.5, 1.1)})
    with pytest.raises(ValueError, match="matching order and length"):
        tunable.resolve({"prompt": "x", "transformer_adapter_strengths": (0.5,)})
    with pytest.raises(TypeError, match="ordered collection"):
        tunable.resolve({"prompt": "x", "transformer_adapter_strengths": {0.5, 0.8}})
    with pytest.raises(ValueError, match="cannot be lower"):
        exposed(duration, default=5.0, minimum=0.5)
    with pytest.raises(ValueError, match="cannot exceed"):
        exposed(duration, default=5.0, maximum=10.5)


def _ltx_flf_product(
    tmp_path: Path,
    *,
    checkpoint: Path | None = None,
    text_checkpoint: Path | None = None,
    device_index: int = 0,
) -> Recipe:
    return ltx23_flf_recipe(
        checkpoint=checkpoint or tmp_path / "flf-model.safetensors",
        text_checkpoint=text_checkpoint or tmp_path / "flf-text.safetensors",
        device_index=device_index,
    )


def test_ltx_flf_reuses_only_semantically_identical_t2v_capabilities(
    tmp_path: Path,
) -> None:
    definition = _ltx_flf_product(tmp_path)

    assert definition.capabilities is LTX23_FLF_CAPABILITIES
    assert {id(field.capability) for field in definition.fields} == {
        id(capability) for capability in LTX23_FLF_CAPABILITIES.capabilities
    }
    for key in (
        "checkpoint",
        "text_checkpoint",
        "device_index",
        "prompt",
        "duration_seconds",
        "seed",
    ):
        assert LTX23_FLF_CAPABILITIES[key] is LTX23_T2V_CAPABILITIES[key]

    flf_width = LTX23_FLF_CAPABILITIES["width"]
    flf_height = LTX23_FLF_CAPABILITIES["height"]
    t2v_width = LTX23_T2V_CAPABILITIES["width"]
    t2v_height = LTX23_T2V_CAPABILITIES["height"]
    assert flf_width is not t2v_width
    assert flf_height is not t2v_height
    assert (flf_width.step, flf_height.step) == (32, 32)
    assert (t2v_width.step, t2v_height.step) == (64, 64)

    _, request = resolve_ltx23_flf(
        definition,
        {
            "prompt": "A narrow 32-pixel-lattice shot",
            "start_image": tmp_path / "first.png",
            "end_image": tmp_path / "last.png",
            "width": 544,
        },
    )
    assert request["width"] == 544
    with pytest.raises(ValueError, match="increments of 64"):
        t2v_width.normalize(544)


def test_ltx_flf_surface_and_resolution_preserve_endpoint_and_identity_boundaries(
    tmp_path: Path,
) -> None:
    definition = _ltx_flf_product(tmp_path)
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    inputs = {
        "prompt": "The subject crosses the frame",
        "start_image": first,
        "end_image": last,
    }

    baseline, request = resolve_ltx23_flf(definition, inputs)
    swapped, swapped_request = resolve_ltx23_flf(
        definition,
        {**inputs, "start_image": last, "end_image": first},
    )

    assert isinstance(baseline, Ltx23FlfIdentity)
    assert request["first_image_path"] == first
    assert request["last_image_path"] == last
    assert swapped_request["first_image_path"] == last
    assert swapped_request["last_image_path"] == first
    assert swapped == baseline

    surface = {field["key"]: field for field in definition.surface()}
    assert list(surface) == [
        "prompt",
        "start_image",
        "end_image",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]
    assert surface["start_image"]["required"] is True
    assert surface["start_image"]["role"] == "start_image"
    assert surface["end_image"]["required"] is True
    assert surface["end_image"]["role"] == "end_image"
    assert not {"checkpoint", "text_checkpoint", "device_index"} & surface.keys()

    changes = (
        {"start_image": tmp_path / "other-first.png"},
        {"end_image": tmp_path / "other-last.png"},
        {"prompt": "A different prompt"},
        {"width": 544},
        {"height": 544},
        {"duration_seconds": 4.5},
        {"seed": 19},
    )
    for change in changes:
        changed, changed_request = resolve_ltx23_flf(definition, {**inputs, **change})
        assert changed == baseline
        assert changed_request != request

    for changed_product in (
        _ltx_flf_product(tmp_path, checkpoint=tmp_path / "other-model.safetensors"),
        _ltx_flf_product(tmp_path, text_checkpoint=tmp_path / "other-text.safetensors"),
        _ltx_flf_product(tmp_path, device_index=1),
    ):
        changed, _ = resolve_ltx23_flf(changed_product, inputs)
        assert changed != baseline


def test_ltx_flf_family_validation_rejects_invalid_request_values(
    tmp_path: Path,
) -> None:
    definition = _ltx_flf_product(tmp_path)
    inputs = {
        "prompt": "The subject crosses the frame",
        "start_image": tmp_path / "first.png",
        "end_image": tmp_path / "last.png",
    }

    with pytest.raises(ValueError, match="increments of 32"):
        definition.resolve({**inputs, "width": 528})
    with pytest.raises(ValueError, match="at least 1.0"):
        definition.resolve({**inputs, "duration_seconds": 0.5})
    with pytest.raises(ValueError, match="at least 0"):
        definition.resolve({**inputs, "seed": -1})


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


def _wan_i2v_product(tmp_path: Path) -> Recipe:
    return wan2214b_i2v_recipe(
        high_checkpoint=_file(tmp_path, "i2v-high.safetensors"),
        high_adapters=(
            Adapter(Artifact(_file(tmp_path, "i2v-high-primary.safetensors")), 0.7),
            Adapter(Artifact(_file(tmp_path, "i2v-high-secondary.safetensors")), 0.2),
        ),
        low_checkpoint=_file(tmp_path, "i2v-low.safetensors"),
        low_adapters=(
            Adapter(Artifact(_file(tmp_path, "i2v-low-primary.safetensors")), 0.9),
        ),
        text_encoder=_file(tmp_path, "i2v-umt5.safetensors"),
        vae=_file(tmp_path, "i2v-vae.safetensors"),
        negative_prompt="fixed I2V negative",
    )


def test_wan_i2v_reuses_family_capabilities_and_adds_one_source() -> None:
    common = (
        "high_checkpoint",
        "high_adapters",
        "low_checkpoint",
        "low_adapters",
        "text_encoder",
        "vae",
        "negative_prompt",
        "shift",
        "steps",
        "split_step",
        "cfg",
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    )
    for key in common:
        assert WAN2214B_I2V_CAPABILITIES[key] is WAN2214B_T2V_CAPABILITIES[key]
    assert (
        WAN2214B_I2V_CAPABILITIES["start_image"]
        is WAN2214B_FLF_CAPABILITIES["start_image"]
    )
    with pytest.raises(KeyError):
        WAN2214B_I2V_CAPABILITIES["end_image"]


def test_wan_turbo_capabilities_express_singleton_family_domains() -> None:
    expected = {
        "shift": 5.000000000000001,
        "steps": 4,
        "split_step": 2,
        "cfg": 1.0,
    }
    invalid = {"shift": 6.0, "steps": 6, "split_step": 3, "cfg": 2.0}
    for key, value in expected.items():
        capability = WAN2214B_T2V_CAPABILITIES[key]
        assert capability.choices == (value,)
        assert capability.normalize(value) == value
        with pytest.raises(ValueError, match="must be one of"):
            capability.normalize(invalid[key])


def test_wan_i2v_resolution_maps_source_and_preserves_request_identity(
    tmp_path: Path,
) -> None:
    definition = _wan_i2v_product(tmp_path)
    source = tmp_path / "source.png"
    inputs = {"prompt": "The subject waves", "start_image": source}

    baseline, request = resolve_wan2214b_i2v(definition, inputs)

    assert isinstance(baseline, WanI2VRecipe)
    assert definition.capabilities is WAN2214B_I2V_CAPABILITIES
    assert baseline.high_checkpoint == str(tmp_path / "i2v-high.safetensors")
    assert baseline.high_lora == str(tmp_path / "i2v-high-primary.safetensors")
    assert baseline.high_secondary_lora == str(
        tmp_path / "i2v-high-secondary.safetensors"
    )
    assert baseline.low_checkpoint == str(tmp_path / "i2v-low.safetensors")
    assert baseline.low_lora == str(tmp_path / "i2v-low-primary.safetensors")
    assert baseline.low_secondary_lora is None
    assert request["source_path"] == source
    assert [field["key"] for field in definition.surface()] == [
        "prompt",
        "start_image",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]
    changes = (
        {"start_image": tmp_path / "other.png"},
        {"prompt": "A different prompt"},
        {"width": 832},
        {"height": 768},
        {"duration_seconds": 1.0},
        {"seed": 19},
    )
    for change in changes:
        changed, changed_request = resolve_wan2214b_i2v(
            definition, {**inputs, **change}
        )
        assert changed.identity == baseline.identity
        assert changed_request != request


def _wan_flf_product(
    tmp_path: Path,
    *,
    high_checkpoint: Path | None = None,
    high_adapters: tuple[Adapter, ...] | None = None,
) -> tuple[Recipe, dict[str, object]]:
    high_primary = Adapter(
        Artifact(_file(tmp_path, "flf-high-primary.safetensors")), 0.7
    )
    high_secondary = Adapter(
        Artifact(_file(tmp_path, "flf-high-secondary.safetensors")), 0.2
    )
    low_primary = Adapter(Artifact(_file(tmp_path, "flf-low-primary.safetensors")), 0.9)
    values: dict[str, object] = {
        "high_checkpoint": high_checkpoint or _file(tmp_path, "flf-high.safetensors"),
        "high_adapters": high_adapters or (high_primary, high_secondary),
        "low_checkpoint": _file(tmp_path, "flf-low.safetensors"),
        "low_adapters": (low_primary,),
        "text_encoder": _file(tmp_path, "flf-umt5.safetensors"),
        "vae": _file(tmp_path, "flf-vae.safetensors"),
        "negative_prompt": "fixed FLF negative",
    }
    return wan2214b_flf_recipe(**values), values  # type: ignore[arg-type]


def test_wan_flf_recipe_uses_declared_capabilities_and_distinct_endpoint_surface(
    tmp_path: Path,
) -> None:
    definition, _ = _wan_flf_product(tmp_path)

    assert definition.capabilities is WAN2214B_FLF_CAPABILITIES
    assert {id(field.capability) for field in definition.fields} == {
        id(capability) for capability in WAN2214B_FLF_CAPABILITIES.capabilities
    }
    for key in (
        "high_checkpoint",
        "high_adapters",
        "low_checkpoint",
        "low_adapters",
        "text_encoder",
        "vae",
        "negative_prompt",
        "shift",
        "steps",
        "split_step",
        "cfg",
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ):
        assert WAN2214B_FLF_CAPABILITIES[key] is WAN2214B_T2V_CAPABILITIES[key]

    surface = {field["key"]: field for field in definition.surface()}
    assert list(surface) == [
        "prompt",
        "start_image",
        "end_image",
        "width",
        "height",
        "duration_seconds",
        "seed",
    ]
    assert surface["start_image"] == {
        "key": "start_image",
        "type": "image",
        "required": True,
        "role": "start_image",
    }
    assert surface["end_image"] == {
        "key": "end_image",
        "type": "image",
        "required": True,
        "role": "end_image",
    }
    assert (
        not {
            "high_checkpoint",
            "high_adapters",
            "low_checkpoint",
            "low_adapters",
            "shift",
            "steps",
            "split_step",
            "cfg",
        }
        & surface.keys()
    )


def test_wan_flf_resolution_preserves_endpoint_order_and_request_identity_boundary(
    tmp_path: Path,
) -> None:
    definition, _ = _wan_flf_product(tmp_path)
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    inputs = {
        "prompt": "The subject turns",
        "start_image": first,
        "end_image": last,
    }

    baseline, request = resolve_wan2214b_flf(definition, inputs)
    swapped, swapped_request = resolve_wan2214b_flf(
        definition,
        {**inputs, "start_image": last, "end_image": first},
    )

    assert isinstance(baseline, WanFLFRecipe)
    assert request["first_path"] == first
    assert request["last_path"] == last
    assert swapped_request["first_path"] == last
    assert swapped_request["last_path"] == first
    assert swapped.identity == baseline.identity

    changes = (
        {"start_image": tmp_path / "other-first.png"},
        {"end_image": tmp_path / "other-last.png"},
        {"prompt": "A different prompt"},
        {"width": 832},
        {"height": 768},
        {"duration_seconds": 1.0},
        {"seed": 19},
    )
    for change in changes:
        changed, changed_request = resolve_wan2214b_flf(
            definition, {**inputs, **change}
        )
        assert changed.identity == baseline.identity
        assert changed_request != request

    timed, timed_request = resolve_wan2214b_flf(
        definition, {**inputs, "duration_seconds": 2.5}
    )
    assert timed.identity == baseline.identity
    assert timed_request["frame_count"] == 41


def test_wan_flf_resolution_preserves_model_and_adapter_ownership(
    tmp_path: Path,
) -> None:
    definition, values = _wan_flf_product(tmp_path)
    inputs = {
        "prompt": "The subject turns",
        "start_image": tmp_path / "first.png",
        "end_image": tmp_path / "last.png",
    }

    baseline, _ = resolve_wan2214b_flf(definition, inputs)
    high_adapters = values["high_adapters"]
    low_adapters = values["low_adapters"]
    assert isinstance(high_adapters, tuple) and isinstance(low_adapters, tuple)
    assert baseline.high_checkpoint == str(values["high_checkpoint"])
    assert baseline.low_checkpoint == str(values["low_checkpoint"])
    assert baseline.high_lora == str(high_adapters[0].artifact.path)
    assert baseline.high_secondary_lora == str(high_adapters[1].artifact.path)
    assert baseline.high_lora_strength == 0.7
    assert baseline.high_secondary_lora_strength == 0.2
    assert baseline.low_lora == str(low_adapters[0].artifact.path)
    assert baseline.low_secondary_lora is None

    changed_artifact, _ = _wan_flf_product(
        tmp_path,
        high_checkpoint=_file(tmp_path, "different-flf-high.safetensors"),
    )
    artifact_identity, _ = resolve_wan2214b_flf(changed_artifact, inputs)
    changed_adapters, _ = _wan_flf_product(
        tmp_path,
        high_adapters=tuple(reversed(high_adapters)),
    )
    adapter_identity, _ = resolve_wan2214b_flf(changed_adapters, inputs)
    assert artifact_identity.identity != baseline.identity
    assert adapter_identity.identity != baseline.identity


def test_wan_flf_fixed_settings_and_family_domains_are_enforced(
    tmp_path: Path,
) -> None:
    definition, _ = _wan_flf_product(tmp_path)
    inputs = {
        "prompt": "The subject turns",
        "start_image": tmp_path / "first.png",
        "end_image": tmp_path / "last.png",
    }

    with pytest.raises(ValueError, match="fixed"):
        definition.resolve({**inputs, "steps": 6})
    with pytest.raises(ValueError, match="aspect ratio"):
        definition.resolve({**inputs, "width": 1280, "height": 480})
    with pytest.raises(ValueError, match="at least 1.0"):
        definition.resolve({**inputs, "duration_seconds": 0.75})
    with pytest.raises(ValueError, match="at least 0"):
        definition.resolve({**inputs, "seed": -1})


def test_generic_recipe_import_has_no_family_torch_or_allocator_side_effects() -> None:
    script = """
import os
import sys
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
import latentslate_engine.recipe
assert 'torch' not in sys.modules
for family in ('ltx23', 'klein9b', 'wan2214b'):
    prefix = f'latentslate_engine.{family}'
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
