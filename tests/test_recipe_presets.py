"""Family-neutral recipe presets: exclusive named bundles over recipe-owned scalars."""

import pytest

from latentslate_engine.recipe import (
    Capability,
    CapabilitySet,
    PresetChoice,
    PresetGroup,
    ProductPolicy,
    exposed,
    fixed,
)


def _bundle():
    steps = Capability("steps", "integer", minimum=1, maximum=200)
    mu = Capability("mu", "number", minimum=-10.0, maximum=10.0, step=0.05)
    prompt = Capability("prompt", "text")
    capabilities = CapabilitySet("demo.op", (steps, mu, prompt))
    quality = PresetGroup(
        key="quality",
        choices=(
            PresetChoice("quality", "Quality", {"steps": 48, "mu": 0.0}),
            PresetChoice("default", "Default", {"steps": 20, "mu": 0.0}),
            PresetChoice("turbo", "Turbo", {"steps": 12, "mu": 0.5}),
        ),
        value="default",
        driven=("steps", "mu"),
        exposed=True,
    )
    policy = ProductPolicy(
        "demo.v1",
        capabilities,
        (
            exposed(prompt),
            fixed(steps, 20),
            fixed(mu, 0.0),
        ),
        (quality,),
    )
    return capabilities, policy, steps, mu, prompt, quality


def test_exposed_preset_is_on_the_surface_instead_of_driven_fields():
    _, policy, _, _, _, _ = _bundle()
    recipe = policy.bind({})
    assert recipe.surface() == policy.surface()
    assert [item["key"] for item in recipe.surface()] == ["prompt", "quality"]
    assert recipe.surface()[1]["constraints"]["choices"] == [
        "quality",
        "default",
        "turbo",
    ]
    assert recipe.surface()[1]["default"] == "default"


def test_resolve_fills_driven_values_and_rejects_mixed_overrides():
    _, policy, _, _, _, _ = _bundle()
    recipe = policy.bind({})
    assert recipe.resolve({"prompt": "scene"})["quality"] == "default"
    assert recipe.resolve({"prompt": "scene"})["steps"] == 20
    resolved = recipe.resolve({"prompt": "scene", "quality": "quality"})
    assert (resolved["steps"], resolved["mu"]) == (48, 0.0)
    turbo = recipe.resolve({"prompt": "scene", "quality": "turbo"})
    assert (turbo["steps"], turbo["mu"]) == (12, 0.5)
    with pytest.raises(ValueError, match="preset-driven"):
        recipe.resolve({"prompt": "scene", "steps": 12})
    with pytest.raises(ValueError, match="preset-driven"):
        recipe.resolve({"prompt": "scene", "quality": "turbo", "mu": 0.5})
    with pytest.raises(ValueError, match="one of"):
        recipe.resolve({"prompt": "scene", "quality": "draft"})


def test_fixed_preset_never_appears_on_the_surface():
    capabilities, _, steps, mu, prompt, quality = _bundle()
    locked = PresetGroup(
        key=quality.key,
        choices=quality.choices,
        value="turbo",
        driven=quality.driven,
        exposed=False,
    )
    policy = ProductPolicy(
        "demo.locked",
        capabilities,
        (exposed(prompt), fixed(steps, 12), fixed(mu, 0.5)),
        (locked,),
    )
    recipe = policy.bind({})
    assert [item["key"] for item in recipe.surface()] == ["prompt"]
    assert recipe.resolve({"prompt": "scene"})["steps"] == 12
    with pytest.raises(ValueError, match="presets are fixed"):
        recipe.resolve({"prompt": "scene", "quality": "default"})


def test_cake_rule_rejects_exposed_driven_fields_and_overlapping_groups():
    capabilities, _, steps, mu, prompt, quality = _bundle()
    with pytest.raises(ValueError, match="cannot be exposed"):
        ProductPolicy(
            "demo.invalid",
            capabilities,
            (exposed(prompt), exposed(steps, default=20), fixed(mu, 0.0)),
            (quality,),
        )
    other = PresetGroup(
        key="look",
        choices=(PresetChoice("a", "A", {"steps": 20}),),
        value="a",
        driven=("steps",),
        exposed=True,
    )
    with pytest.raises(ValueError, match="overlapping preset driven key"):
        ProductPolicy(
            "demo.overlap",
            capabilities,
            (exposed(prompt), fixed(steps, 20), fixed(mu, 0.0)),
            (quality, other),
        )
    with pytest.raises(ValueError, match="collides with capability"):
        ProductPolicy(
            "demo.collide",
            capabilities,
            (exposed(prompt), fixed(steps, 20), fixed(mu, 0.0)),
            (
                PresetGroup(
                    key="steps",
                    choices=(PresetChoice("default", "Default", {"mu": 0.0}),),
                    value="default",
                    driven=("mu",),
                    exposed=True,
                ),
            ),
        )


def test_selected_choice_must_match_stored_defaults():
    capabilities, _, steps, mu, prompt, quality = _bundle()
    with pytest.raises(ValueError, match="must match stored"):
        ProductPolicy(
            "demo.mismatch",
            capabilities,
            (exposed(prompt), fixed(steps, 12), fixed(mu, 0.0)),
            (quality,),
        )


def test_no_preset_group_leaves_fields_independently_exposed():
    capabilities, _, steps, mu, prompt, _ = _bundle()
    policy = ProductPolicy(
        "demo.granular",
        capabilities,
        (exposed(prompt), exposed(steps, default=20), exposed(mu, default=0.0)),
    )
    recipe = policy.bind({})
    assert [item["key"] for item in recipe.surface()] == ["prompt", "steps", "mu"]
    assert recipe.resolve({"prompt": "scene", "steps": 30})["steps"] == 30
