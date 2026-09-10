from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from latentslate_engine.klein9b.contracts import TOKENIZER_FILES
from latentslate_engine.klein9b.recipes import (
    klein9b_two_image_recipe,
    resolve_klein9b_two_image,
)
from latentslate_engine.ltx23.contracts import (
    Ltx23FlfIdentity,
    Ltx23I2VIdentity,
    Ltx23T2VIdentity,
)
from latentslate_engine.ltx23.recipes import (
    LTX23_FLF_CAPABILITIES,
    LTX23_FLF_POLICY,
    LTX23_I2V_CAPABILITIES,
    LTX23_I2V_POLICY,
    LTX23_T2V_CAPABILITIES,
    LTX23_T2V_POLICY,
    ltx23_flf_recipe,
    ltx23_i2v_recipe,
    ltx23_t2v_locked_recipe,
    ltx23_t2v_recipe,
    ltx23_t2v_tunable_recipe,
    resolve_ltx23_flf,
    resolve_ltx23_flf_identity,
    resolve_ltx23_i2v,
    resolve_ltx23_i2v_identity,
    resolve_ltx23_t2v,
    resolve_ltx23_t2v_identity,
)
from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    Recipe,
    exposed,
    fixed,
)
from latentslate_engine.wan2214b.flf import WanFLFRecipe
from latentslate_engine.wan2214b.i2v import WanI2VRecipe
from latentslate_engine.wan2214b.recipes import (
    WAN2214B_FLF_CAPABILITIES,
    WAN2214B_FLF_POLICY,
    WAN2214B_I2V_CAPABILITIES,
    WAN2214B_I2V_POLICY,
    WAN2214B_T2V_CAPABILITIES,
    WAN2214B_T2V_POLICY,
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


def _ltx_i2v_product(tmp_path: Path, **changes: object) -> Recipe:
    values = {
        "checkpoint": tmp_path / "model.safetensors",
        "text_checkpoint": tmp_path / "text.safetensors",
        "upsampler": tmp_path / "upsampler.safetensors",
        **changes,
    }
    return ltx23_i2v_recipe(**values)  # type: ignore[arg-type]


def test_ltx_i2v_reuses_t2v_capabilities_and_flf_start_image(tmp_path: Path) -> None:
    definition = _ltx_i2v_product(tmp_path)
    assert definition.capabilities is LTX23_I2V_CAPABILITIES
    for field in definition.fields:
        assert field.capability is LTX23_I2V_CAPABILITIES[field.capability.key]
    common = (
        "checkpoint",
        "text_checkpoint",
        "upsampler",
        "transformer_adapter_artifacts",
        "transformer_adapter_strengths",
        "device_index",
        "prompt",
        "width",
        "height",
        "duration_seconds",
        "seed",
    )
    assert {cap.key for cap in LTX23_I2V_CAPABILITIES.capabilities} == {
        *common,
        "start_image",
    }
    for key in common:
        assert LTX23_I2V_CAPABILITIES[key] is LTX23_T2V_CAPABILITIES[key]
    assert (
        LTX23_I2V_CAPABILITIES["start_image"] is LTX23_FLF_CAPABILITIES["start_image"]
    )
    for key in ("width", "height"):
        assert LTX23_I2V_CAPABILITIES[key].step == 64
        assert LTX23_T2V_CAPABILITIES[key].step == 64
        assert LTX23_FLF_CAPABILITIES[key].step == 32
        assert LTX23_I2V_CAPABILITIES[key] is not LTX23_FLF_CAPABILITIES[key]
    surface = {field["key"]: field for field in definition.surface()}
    assert list(surface) == [
        "prompt",
        "start_image",
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
    with pytest.raises(ValueError, match="missing required.*start_image"):
        definition.resolve({"prompt": "A bird"})


def test_ltx_i2v_preserves_source_request_and_model_identity(tmp_path: Path) -> None:
    definition = _ltx_i2v_product(tmp_path)
    source = tmp_path / "unopened-source.png"
    inputs = {"prompt": "A bird", "start_image": source}
    baseline, request = resolve_ltx23_i2v(definition, inputs)
    assert baseline == Ltx23I2VIdentity(
        checkpoint_path=str(tmp_path / "model.safetensors"),
        text_checkpoint_path=str(tmp_path / "text.safetensors"),
        transformer_lora_path=None,
        upsampler_path=str(tmp_path / "upsampler.safetensors"),
    )
    assert request == {
        "prompt": "A bird",
        "image_path": source,
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 0,
    }
    for change in (
        {"start_image": tmp_path / "other.png"},
        {"prompt": "A fox"},
        {"width": 768},
        {"height": 768},
        {"duration_seconds": 4.5},
        {"seed": 9},
    ):
        identity, changed = resolve_ltx23_i2v(definition, {**inputs, **change})
        assert identity == baseline
        assert changed != request
        assert changed["image_path"] == change.get("start_image", source)
    for key, value in (
        ("checkpoint", tmp_path / "other-model"),
        ("text_checkpoint", tmp_path / "other-text"),
        ("upsampler", tmp_path / "other-upsampler"),
        ("device_index", 1),
    ):
        changed, _ = resolve_ltx23_i2v(
            _ltx_i2v_product(tmp_path, **{key: value}), inputs
        )
        assert changed != baseline


def test_ltx_i2v_preserves_native_adapter_representations(tmp_path: Path) -> None:
    first = Adapter(Artifact(tmp_path / "style"), 0.35)
    second = Adapter(Artifact(tmp_path / "motion"), 0.8)
    inputs = {"prompt": "A bird", "start_image": tmp_path / "source.png"}
    for adapters in ((), (first,), (first, second), (second, first)):
        definition = _ltx_i2v_product(tmp_path, transformer_adapters=adapters)
        identity, _ = resolve_ltx23_i2v(definition, inputs)
        assert resolve_ltx23_i2v_identity(definition) == identity
        if len(adapters) == 1:
            assert identity.transformer_lora_path == str(first.artifact.path)
            assert identity.lora_strength == 0.35
            assert identity.transformer_loras == ()
        else:
            assert identity.transformer_lora_path is None
            assert identity.lora_strength == 0.5
            assert identity.transformer_loras == tuple(
                (str(adapter.artifact.path), adapter.strength) for adapter in adapters
            )
        with pytest.raises(ValueError, match="fixed"):
            definition.resolve({**inputs, "transformer_adapter_strengths": (0.9,)})
    baseline, _ = resolve_ltx23_i2v(
        _ltx_i2v_product(tmp_path, transformer_adapters=(first, second)),
        inputs,
    )
    for adapters in (
        (second, first),
        (Adapter(Artifact(tmp_path / "replacement"), 0.35), second),
        (Adapter(first.artifact, 0.6), second),
    ):
        changed, _ = resolve_ltx23_i2v(
            _ltx_i2v_product(tmp_path, transformer_adapters=adapters),
            inputs,
        )
        assert changed != baseline
    malformed = Recipe(
        "ltx23.i2v.mismatched",
        LTX23_I2V_CAPABILITIES,
        tuple(
            fixed(field.capability, (0.5,))
            if field.capability.key == "transformer_adapter_strengths"
            else field
            for field in _ltx_i2v_product(tmp_path).fields
        ),
    )
    with pytest.raises(ValueError, match="matching order and length"):
        resolve_ltx23_i2v(malformed, inputs)
    with pytest.raises(ValueError):
        resolve_ltx23_i2v_identity(malformed)


def test_ltx_i2v_pre_request_identity_requires_fixed_model_fields(
    tmp_path: Path,
) -> None:
    definition = _ltx_i2v_product(tmp_path, device_index=1)
    assert resolve_ltx23_i2v_identity(definition).device_index == 1
    for key in (
        "checkpoint",
        "text_checkpoint",
        "upsampler",
        "device_index",
        "transformer_adapter_artifacts",
        "transformer_adapter_strengths",
    ):
        caller_model = replace(
            definition,
            fields=tuple(
                exposed(field.capability, default=field.value)
                if field.capability.key == key
                else field
                for field in definition.fields
            ),
        )
        with pytest.raises(ValueError, match=f"requires fixed {key}"):
            resolve_ltx23_i2v_identity(caller_model)
    with pytest.raises(TypeError, match="I2V capability set"):
        resolve_ltx23_i2v_identity(
            ltx23_t2v_recipe(checkpoint="model", text_checkpoint="text", upsampler="up")
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"width": 544}, "increments of 64"),
        ({"height": 32}, "at least 64"),
        ({"width": 1024, "height": 1024}, "must not exceed"),
        ({"duration_seconds": 0.5}, "at least 1.0"),
        ({"duration_seconds": 4.25}, "increments of 0.5"),
        ({"seed": -1}, "at least 0"),
        ({"seed": 2**64}, "at most"),
    ],
)
def test_ltx_i2v_enforces_family_domains(
    tmp_path: Path, change: dict, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        resolve_ltx23_i2v(
            _ltx_i2v_product(tmp_path),
            {
                "prompt": "A bird",
                "start_image": tmp_path / "source.png",
                **change,
            },
        )


def test_ltx_resolution_is_torch_free() -> None:
    script = """
import os
import sys
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
from latentslate_engine.ltx23.recipes import ltx23_i2v_recipe, resolve_ltx23_i2v, resolve_ltx23_i2v_identity
from latentslate_engine.ltx23.recipes import ltx23_t2v_recipe, resolve_ltx23_t2v, resolve_ltx23_t2v_identity
from latentslate_engine.ltx23.recipes import ltx23_flf_recipe, resolve_ltx23_flf, resolve_ltx23_flf_identity
recipe = ltx23_i2v_recipe(checkpoint='model', text_checkpoint='text', upsampler='up')
pre_request_identity = resolve_ltx23_i2v_identity(recipe)
identity, request = resolve_ltx23_i2v(recipe, {'prompt': 'A bird', 'start_image': 'absent.png'})
assert pre_request_identity == identity
assert request['image_path'] == 'absent.png'
recipe = ltx23_t2v_recipe(checkpoint='model', text_checkpoint='text', upsampler='up')
assert resolve_ltx23_t2v_identity(recipe) == resolve_ltx23_t2v(recipe, {'prompt': 'A bird'})[0]
recipe = ltx23_flf_recipe(checkpoint='model', text_checkpoint='text')
assert resolve_ltx23_flf_identity(recipe) == resolve_ltx23_flf(recipe, {
    'prompt': 'A bird', 'start_image': 'absent-first.png', 'end_image': 'absent-last.png'
})[0]
assert 'torch' not in sys.modules
assert 'latentslate_engine.ltx23.i2v' not in sys.modules
assert 'latentslate_engine.ltx23.t2v' not in sys.modules
assert 'latentslate_engine.ltx23.flf' not in sys.modules
assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


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


@pytest.mark.parametrize("custom_request", (False, True))
def test_ltx_i2v_complete_bound_contract(tmp_path: Path, custom_request: bool) -> None:
    """Lock the complete builder result against the pre-policy 4d7b329 behavior."""
    adapters = (
        Adapter(Artifact(tmp_path / "first-adapter"), 0.35),
        Adapter(Artifact(tmp_path / "second-adapter"), 0.8),
    )
    definition = _ltx_i2v_product(
        tmp_path, transformer_adapters=adapters, device_index=1
    )
    inputs = {"prompt": "A bird", "start_image": tmp_path / "source.png"}
    request_values = {"width": 512, "height": 512, "duration_seconds": 5.0, "seed": 0}
    if custom_request:
        request_values = {
            "width": 768,
            "height": 512,
            "duration_seconds": 4.5,
            "seed": 19,
        }
        inputs.update(request_values)
    assert definition.key == "ltx23.i2v.v1_1"
    assert definition.resolve(inputs) == {
        "checkpoint": Artifact(tmp_path / "model.safetensors"),
        "text_checkpoint": Artifact(tmp_path / "text.safetensors"),
        "upsampler": Artifact(tmp_path / "upsampler.safetensors"),
        "transformer_adapter_artifacts": tuple(
            adapter.artifact for adapter in adapters
        ),
        "transformer_adapter_strengths": (0.35, 0.8),
        "device_index": 1,
        "prompt": "A bird",
        "start_image": tmp_path / "source.png",
        **request_values,
    }
    identity, request = resolve_ltx23_i2v(definition, inputs)
    assert identity == Ltx23I2VIdentity(
        checkpoint_path=str(tmp_path / "model.safetensors"),
        text_checkpoint_path=str(tmp_path / "text.safetensors"),
        transformer_lora_path=None,
        upsampler_path=str(tmp_path / "upsampler.safetensors"),
        lora_strength=0.5,
        device_index=1,
        transformer_loras=(
            (str(tmp_path / "first-adapter"), 0.35),
            (str(tmp_path / "second-adapter"), 0.8),
        ),
    )
    assert request == {
        "prompt": "A bird",
        "image_path": tmp_path / "source.png",
        **request_values,
    }
    assert definition.surface() == _expected_video_surface("ltx")


@pytest.mark.parametrize("custom_request", (False, True))
def test_wan_flf_complete_bound_contract(tmp_path: Path, custom_request: bool) -> None:
    """Keep high/low binding, fixed turbo policy and request conversion explicit."""
    definition, values = _wan_flf_product(tmp_path)
    inputs = {
        "prompt": "A turn",
        "start_image": tmp_path / "first.png",
        "end_image": tmp_path / "last.png",
    }
    request_values = {"width": 512, "height": 512, "duration_seconds": 5.0, "seed": 0}
    frames = 81
    if custom_request:
        request_values = {
            "width": 832,
            "height": 480,
            "duration_seconds": 2.5,
            "seed": 19,
        }
        inputs.update(request_values)
        frames = 41
    assert definition.key == "wan2214b.flf.v1_1"
    assert definition.resolve(inputs) == {
        **{
            key: Artifact(values[key])
            for key in ("high_checkpoint", "low_checkpoint", "text_encoder", "vae")
        },
        "high_adapters": values["high_adapters"],
        "low_adapters": values["low_adapters"],
        "negative_prompt": "fixed FLF negative",
        "shift": 5.000000000000001,
        "steps": 4,
        "split_step": 2,
        "cfg": 1.0,
        "prompt": "A turn",
        "start_image": tmp_path / "first.png",
        "end_image": tmp_path / "last.png",
        **request_values,
    }
    expected = WanFLFRecipe(
        high_checkpoint=str(tmp_path / "flf-high.safetensors"),
        high_lora=str(tmp_path / "flf-high-primary.safetensors"),
        low_checkpoint=str(tmp_path / "flf-low.safetensors"),
        low_lora=str(tmp_path / "flf-low-primary.safetensors"),
        text_encoder=str(tmp_path / "flf-umt5.safetensors"),
        vae=str(tmp_path / "flf-vae.safetensors"),
        high_secondary_lora=str(tmp_path / "flf-high-secondary.safetensors"),
        low_secondary_lora=None,
        high_lora_strength=0.7,
        low_lora_strength=0.9,
        high_secondary_lora_strength=0.2,
        low_secondary_lora_strength=1.0,
        shift=5.000000000000001,
        steps=4,
        split_step=2,
        cfg=1.0,
        width=request_values["width"],
        height=request_values["height"],
        frame_count=frames,
        positive="A turn",
        negative="fixed FLF negative",
    )
    family_recipe, request = resolve_wan2214b_flf(definition, inputs)
    assert family_recipe == expected
    assert family_recipe.identity == expected.identity
    assert request == {
        "first_path": tmp_path / "first.png",
        "last_path": tmp_path / "last.png",
        "seed": request_values["seed"],
        "width": request_values["width"],
        "height": request_values["height"],
        "frame_count": frames,
        "positive_prompt": "A turn",
        "negative_prompt": "fixed FLF negative",
    }
    assert definition.surface() == _expected_video_surface("wan")


def _expected_video_surface(
    family: str, *, images: tuple[str, ...] | None = None, alignment: int | None = None
) -> tuple[dict[str, object], ...]:
    """Literal pre-change semantics, independent of the family policy declarations."""
    if images is None:
        images = ("start_image",) if family == "ltx" else ("start_image", "end_image")
    if alignment is None:
        alignment = 64 if family == "ltx" else 16
    return (
        {"key": "prompt", "type": "text", "required": True},
        *(
            {"key": key, "type": "image", "required": True, "role": key}
            for key in images
        ),
        *(
            {
                "key": key,
                "type": "integer",
                "required": False,
                "default": 512,
                "role": key,
                "constraints": {
                    "min": 64 if family == "ltx" else 480,
                    "max": 14720 if family == "ltx" else 1920,
                    "step": alignment,
                },
            }
            for key in ("width", "height")
        ),
        {
            "key": "duration_seconds",
            "type": "number",
            "required": False,
            "default": 5.0,
            "role": "duration_seconds",
            "constraints": {
                "min": 1.0,
                "max": 10.0 if family == "ltx" else 5.0,
                "step": 0.5 if family == "ltx" else 0.25,
            },
        },
        {
            "key": "seed",
            "type": "integer",
            "required": False,
            "default": 0,
            "role": "seed",
            "constraints": {"min": 0, "max": 18446744073709551615},
        },
    )


@pytest.fixture(params=("ltx", "wan"))
def policy_binding(request, tmp_path: Path):
    if request.param == "ltx":
        policy = LTX23_I2V_POLICY
        built = _ltx_i2v_product(tmp_path)
        bindings = {
            "checkpoint": Artifact(tmp_path / "model.safetensors"),
            "text_checkpoint": Artifact(tmp_path / "text.safetensors"),
            "upsampler": Artifact(tmp_path / "upsampler.safetensors"),
            "transformer_adapter_artifacts": (),
            "transformer_adapter_strengths": (),
            "device_index": 0,
        }
    else:
        policy = WAN2214B_FLF_POLICY
        built, values = _wan_flf_product(tmp_path)
        bindings = {
            **values,
            **{
                key: Artifact(values[key])
                for key in ("high_checkpoint", "low_checkpoint", "text_encoder", "vae")
            },
        }
    return policy, bindings, built


def test_unbound_policy_binds_to_existing_builder_result(policy_binding) -> None:
    policy, bindings, built = policy_binding
    bound = policy.bind(bindings)
    assert bound == built
    assert bound.capabilities is policy.capabilities
    assert bound.surface() == policy.surface()
    assert policy.surface() == _expected_video_surface(
        "ltx" if policy is LTX23_I2V_POLICY else "wan"
    )


def test_binding_rejects_missing_unknown_exposed_and_invalid_values(
    policy_binding,
) -> None:
    policy, bindings, _ = policy_binding
    for key in bindings:
        with pytest.raises(ValueError, match=f"missing hidden product bindings.*{key}"):
            policy.bind(
                {name: value for name, value in bindings.items() if name != key}
            )
    with pytest.raises(ValueError, match="unknown product bindings.*unknown"):
        policy.bind({**bindings, "unknown": 1})
    with pytest.raises(ValueError, match="cannot bind caller-exposed fields.*seed"):
        policy.bind({**bindings, "seed": 0})
    artifact_key = "checkpoint" if policy is LTX23_I2V_POLICY else "high_checkpoint"
    with pytest.raises(TypeError, match=f"{artifact_key} must be artifact"):
        policy.bind({**bindings, artifact_key: "not-an-Artifact"})
    if policy is WAN2214B_FLF_POLICY:
        with pytest.raises(
            ValueError, match="cannot rebind policy-fixed fields.*steps"
        ):
            policy.bind({**bindings, "steps": 4})


def test_policy_narrowing_drives_bound_recipe_and_existing_builder(
    policy_binding, tmp_path: Path, monkeypatch
) -> None:
    policy, bindings, _ = policy_binding
    narrowed = replace(
        policy,
        fields=tuple(
            exposed(field.capability, default=768, minimum=512, maximum=1024, step=128)
            if field.capability.key == "width"
            else field
            for field in policy.fields
        ),
    )
    inputs = {"prompt": "A shot", "start_image": tmp_path / "first.png"}
    if policy is LTX23_I2V_POLICY:
        monkeypatch.setattr(
            "latentslate_engine.ltx23.recipes.LTX23_I2V_POLICY", narrowed
        )
        built = _ltx_i2v_product(tmp_path)
    else:
        monkeypatch.setattr(
            "latentslate_engine.wan2214b.recipes.WAN2214B_FLF_POLICY", narrowed
        )
        built, _ = _wan_flf_product(tmp_path)
        inputs["end_image"] = tmp_path / "last.png"
    bound = narrowed.bind(bindings)
    assert built == bound
    assert bound.surface() == narrowed.surface()
    assert bound.resolve(inputs)["width"] == 768
    with pytest.raises(ValueError, match="at least 512"):
        bound.resolve({**inputs, "width": 256 if policy is LTX23_I2V_POLICY else 480})
    with pytest.raises(ValueError, match="increments of 128"):
        bound.resolve({**inputs, "width": 576})
    # Scalar values are individually legal, but the family pixel budget is not.
    with pytest.raises(ValueError, match="must not exceed"):
        bound.resolve({**inputs, "width": 1024, "height": 1024})


def test_cross_field_binding_validation_stays_at_family_resolution(
    policy_binding, tmp_path: Path
) -> None:
    policy, bindings, _ = policy_binding
    inputs = {"prompt": "A shot", "start_image": tmp_path / "first.png"}
    if policy is LTX23_I2V_POLICY:
        invalid = {**bindings, "transformer_adapter_strengths": (0.5,)}
        message = "matching order and length"
    else:
        invalid = {
            **bindings,
            "high_adapters": (Adapter(Artifact(tmp_path / "adapter")),) * 3,
        }
        inputs["end_image"] = tmp_path / "last.png"
        message = "at most primary and secondary"
    bound = policy.bind(invalid)
    with pytest.raises(ValueError, match=message):
        bound.resolve(inputs)


def test_product_policy_preserves_choices_nullability_and_collection_order() -> None:
    revision = Capability("revision", "integer", minimum=0)
    mode = Capability("mode", "choice", choices=("fast", "quality", "draft"))
    note = Capability("note", "text", optional=True)
    order = Capability("order", "integer", ordered=True, minimum=0, maximum=8, step=2)
    capabilities = CapabilitySet("small", (revision, mode, note, order))
    policy = ProductPolicy(
        "small.policy",
        capabilities,
        (
            exposed(order, default=(4, 2), maximum=6),
            exposed(note, default=None),
            exposed(mode, choices=("fast", "quality")),
        ),
    )
    bound = policy.bind({"revision": 1})
    assert (
        policy.surface()
        == bound.surface()
        == (
            {
                "key": "order",
                "type": "integer",
                "required": False,
                "default": [4, 2],
                "collection": True,
                "ordered": True,
                "constraints": {"min": 0, "max": 6, "step": 2},
            },
            {
                "key": "note",
                "type": "text",
                "required": False,
                "default": None,
                "nullable": True,
            },
            {
                "key": "mode",
                "type": "choice",
                "required": True,
                "constraints": {"choices": ["fast", "quality"]},
            },
        )
    )
    assert bound.resolve({"mode": "fast"}) == {
        "revision": 1,
        "order": (4, 2),
        "note": None,
        "mode": "fast",
    }
    assert bound.resolve({"mode": "quality", "order": (2, 4)})["order"] == (2, 4)
    with pytest.raises(ValueError, match="missing required.*mode"):
        bound.resolve({})
    with pytest.raises(ValueError, match="one of"):
        bound.resolve({"mode": "draft"})
    with pytest.raises(ValueError, match="at most 6"):
        bound.resolve({"mode": "fast", "order": (8,)})
    with pytest.raises(ValueError, match="at least 0"):
        policy.bind({"revision": -1})
    with pytest.raises(ValueError, match="reuse declared capability objects"):
        ProductPolicy("invalid", capabilities, (exposed(Capability("mode", "choice")),))
    with pytest.raises(ValueError, match="keys must be unique"):
        replace(policy, fields=policy.fields + (policy.fields[0],))


def test_unbound_family_policies_are_pure_before_configuration() -> None:
    script = """
import builtins
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from latentslate_engine.recipe import Artifact, ProductPolicy, Recipe

os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
os.environ.pop('LATENTSLATE_ENGINE_HOME', None)
os.environ.pop('LATENTSLATE_WAN_MODEL_ROOT', None)
environment = dict(os.environ)
def forbidden(*args, **kwargs):
    raise AssertionError('unbound policy must not perform IO, configure or bind')
original_import = builtins.__import__
def pure_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'dotenv'} or name == 'latentslate_engine.service':
        forbidden()
    return original_import(name, *args, **kwargs)

# Python's module loader may read source/bytecode; product code may do no IO.
with ExitStack() as stack:
    for name in ('open', 'read_text', 'read_bytes', 'stat', 'exists', 'is_file', 'is_dir', 'resolve'):
        stack.enter_context(patch.object(Path, name, forbidden))
    for name in ('open', 'stat', 'listdir', 'scandir'):
        stack.enter_context(patch.object(os, name, forbidden))
    stack.enter_context(patch.object(builtins, 'open', forbidden))
    stack.enter_context(patch.object(io, 'open', forbidden))
    stack.enter_context(patch.object(builtins, '__import__', pure_import))
    stack.enter_context(patch.object(Artifact, '__init__', forbidden))
    stack.enter_context(patch.object(ProductPolicy, 'bind', forbidden))
    stack.enter_context(patch.object(Recipe, '__post_init__', forbidden))
    from latentslate_engine.ltx23.recipes import LTX23_T2V_POLICY, LTX23_I2V_POLICY, LTX23_FLF_POLICY
    from latentslate_engine.wan2214b.recipes import WAN2214B_T2V_POLICY, WAN2214B_I2V_POLICY, WAN2214B_FLF_POLICY
    surfaces = [replace(policy).surface() for policy in (
        LTX23_T2V_POLICY, LTX23_I2V_POLICY, LTX23_FLF_POLICY,
        WAN2214B_T2V_POLICY, WAN2214B_I2V_POLICY, WAN2214B_FLF_POLICY)]
    assert dict(os.environ) == environment
    assert 'torch' not in sys.modules and 'dotenv' not in sys.modules
    assert 'latentslate_engine.service' not in sys.modules
print(json.dumps(surfaces))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    import json

    assert json.loads(completed.stdout) == [
        list(_expected_video_surface("ltx", images=())),
        list(_expected_video_surface("ltx")),
        list(
            _expected_video_surface(
                "ltx", images=("start_image", "end_image"), alignment=32
            )
        ),
        list(_expected_video_surface("wan", images=())),
        list(_expected_video_surface("wan", images=("start_image",))),
        list(_expected_video_surface("wan")),
    ]


@pytest.mark.parametrize("operation", ("ltx_t2v", "ltx_flf", "wan_t2v", "wan_i2v"))
def test_remaining_video_builders_preserve_bound_contract(
    tmp_path, monkeypatch, operation
):
    from latentslate_engine.ltx23 import recipes as ltx
    from latentslate_engine.wan2214b import recipes as wan

    inputs = {"prompt": "A shot"}
    if operation.startswith("ltx"):
        kwargs = {
            "checkpoint": tmp_path / "model",
            "text_checkpoint": tmp_path / "text",
            "device_index": 1,
        }
        bindings = {
            "checkpoint": Artifact(kwargs["checkpoint"]),
            "text_checkpoint": Artifact(kwargs["text_checkpoint"]),
        }
        if operation == "ltx_t2v":
            policy, builder, key = LTX23_T2V_POLICY, ltx23_t2v_recipe, "ltx23.t2v.v1"
            adapters = (
                Adapter(Artifact(tmp_path / "first"), 0.35),
                Adapter(Artifact(tmp_path / "second"), 0.8),
            )
            kwargs.update(upsampler=tmp_path / "up", transformer_adapters=adapters)
            bindings.update(
                upsampler=Artifact(kwargs["upsampler"]),
                transformer_adapter_artifacts=tuple(a.artifact for a in adapters),
                transformer_adapter_strengths=(0.35, 0.8),
            )
            surface = _expected_video_surface("ltx", images=())
        else:
            policy, builder, key = LTX23_FLF_POLICY, ltx23_flf_recipe, "ltx23.flf.v1_1"
            inputs.update(
                start_image=tmp_path / "first.png", end_image=tmp_path / "last.png"
            )
            surface = _expected_video_surface(
                "ltx", images=("start_image", "end_image"), alignment=32
            )
        bindings["device_index"] = 1
        fixed_settings = {}
        module = ltx
    else:
        kwargs = {
            "high_checkpoint": tmp_path / "high",
            "low_checkpoint": tmp_path / "low",
            "high_adapters": (
                Adapter(Artifact(tmp_path / "high-primary"), 0.7),
                Adapter(Artifact(tmp_path / "high-secondary"), 0.2),
            ),
            "low_adapters": (Adapter(Artifact(tmp_path / "low-primary"), 0.9),),
            "text_encoder": tmp_path / "text",
            "vae": tmp_path / "vae",
            "negative_prompt": "fixed negative",
        }
        bindings = {
            "high_checkpoint": Artifact(kwargs["high_checkpoint"]),
            "high_adapters": kwargs["high_adapters"],
            "low_checkpoint": Artifact(kwargs["low_checkpoint"]),
            "low_adapters": kwargs["low_adapters"],
            "text_encoder": Artifact(kwargs["text_encoder"]),
            "vae": Artifact(kwargs["vae"]),
            "negative_prompt": "fixed negative",
        }
        fixed_settings = {
            "shift": 5.000000000000001,
            "steps": 4,
            "split_step": 2,
            "cfg": 1.0,
        }
        if operation == "wan_t2v":
            policy, builder, key = (
                WAN2214B_T2V_POLICY,
                wan2214b_t2v_recipe,
                "wan2214b.t2v.v1",
            )
            surface = _expected_video_surface("wan", images=())
        else:
            policy, builder, key = (
                WAN2214B_I2V_POLICY,
                wan2214b_i2v_recipe,
                "wan2214b.i2v.v1_1",
            )
            inputs["start_image"] = tmp_path / "source.png"
            surface = _expected_video_surface("wan", images=("start_image",))
        module = wan
    definition = builder(**kwargs)
    expected = {
        **bindings,
        **fixed_settings,
        **inputs,
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 0,
    }
    assert definition.key == key
    assert definition.resolve(inputs) == expected
    assert [field.capability.key for field in definition.fields] == list(expected)
    assert definition == policy.bind(bindings)
    assert definition.surface() == policy.surface() == surface
    assert all(
        field.capability is policy.capabilities[field.capability.key]
        for field in definition.fields
    )
    changed = replace(
        policy,
        fields=tuple(
            exposed(field.capability, default=768, minimum=512, maximum=1024, step=128)
            if field.capability.key == "width"
            else field
            for field in policy.fields
        ),
    )
    policy_name = {
        "ltx_t2v": "LTX23_T2V_POLICY",
        "ltx_flf": "LTX23_FLF_POLICY",
        "wan_t2v": "WAN2214B_T2V_POLICY",
        "wan_i2v": "WAN2214B_I2V_POLICY",
    }[operation]
    monkeypatch.setattr(module, policy_name, changed)
    rebuilt = builder(**kwargs)
    assert rebuilt.surface() == changed.surface()
    assert rebuilt.resolve(inputs) == {**expected, "width": 768}
    with pytest.raises(ValueError, match="increments of 128"):
        rebuilt.resolve({**inputs, "width": 576})


@pytest.mark.parametrize("operation", ("t2v", "flf"))
def test_ltx_remaining_pre_request_identities_use_only_fixed_models(
    tmp_path, operation
):
    if operation == "t2v":
        definition = ltx23_t2v_recipe(
            checkpoint=tmp_path / "model",
            text_checkpoint=tmp_path / "text",
            upsampler=tmp_path / "up",
            device_index=1,
        )
        resolve_identity, resolve_request = (
            resolve_ltx23_t2v_identity,
            resolve_ltx23_t2v,
        )
        inputs = {"prompt": "A shot"}
        other = ltx23_flf_recipe(checkpoint="model", text_checkpoint="text")
    else:
        definition = _ltx_flf_product(tmp_path, device_index=1)
        resolve_identity, resolve_request = (
            resolve_ltx23_flf_identity,
            resolve_ltx23_flf,
        )
        inputs = {
            "prompt": "A shot",
            "start_image": tmp_path / "first.png",
            "end_image": tmp_path / "last.png",
        }
        other = ltx23_t2v_recipe(
            checkpoint="model", text_checkpoint="text", upsampler="up"
        )
    identity = resolve_identity(definition)
    assert identity.device_index == 1
    assert identity == resolve_request(definition, inputs)[0]
    for fixed_field in (field for field in definition.fields if not field.exposed):
        caller_model = replace(
            definition,
            fields=tuple(
                exposed(field.capability, default=field.value)
                if field is fixed_field
                else field
                for field in definition.fields
            ),
        )
        with pytest.raises(
            ValueError, match=f"requires fixed {fixed_field.capability.key}"
        ):
            resolve_identity(caller_model)
        assert resolve_request(caller_model, inputs)[0] == identity
    with pytest.raises(TypeError, match="capability set"):
        resolve_identity(other)
    if operation == "t2v":
        locked, tunable, _ = _ltx_product_recipes(tmp_path)
        assert resolve_identity(locked) == resolve_request(locked, inputs)[0]
        with pytest.raises(
            ValueError, match="requires fixed transformer_adapter_strengths"
        ):
            resolve_identity(tunable)
