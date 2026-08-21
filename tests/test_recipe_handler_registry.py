from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from latentslate_engine import variants


class _FakeRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["fake_recipe"]
    value: int


@dataclass(frozen=True)
class _FakeRecipe:
    value: int


def _fake_handler() -> variants._RecipeHandler:
    return variants._RecipeHandler(
        frozenset({"fake_recipe"}),
        _FakeRecipeConfig,
        _FakeRecipe,
        lambda _config, _family, _base_tool: None,
        lambda config, _inventory: _FakeRecipe(config.value),
        lambda recipe, _inventory: recipe,
        lambda recipe, _inventory, _loras, _configured: recipe,
    )


def test_new_recipe_type_parses_through_registry_without_compiler_branch(monkeypatch):
    handlers = (*variants._RECIPE_HANDLERS, _fake_handler())
    by_name, by_config, by_recipe = variants._build_recipe_handler_registries(handlers)
    monkeypatch.setattr(variants, "_RECIPE_HANDLER_BY_TYPE_NAME", by_name)
    monkeypatch.setattr(variants, "_RECIPE_HANDLER_BY_CONFIG", by_config)
    monkeypatch.setattr(variants, "_RECIPE_HANDLER_BY_RECIPE", by_recipe)

    definition = variants.VariantDefinition.model_validate(
        {
            "key": "test.fake-recipe",
            "name": "Fake recipe",
            "family": "wan22",
            "base_tool": "wan22.text_to_video",
            "recipe": {"type": "fake_recipe", "value": 7},
        }
    )

    assert definition.recipe == _FakeRecipeConfig(type="fake_recipe", value=7)
    assert variants._recipe_handler_for_config(definition.recipe) is by_name["fake_recipe"]


def test_recipe_registry_rejects_duplicate_type_names():
    duplicate = _fake_handler()

    with pytest.raises(ValueError, match="duplicate recipe type registration"):
        variants._build_recipe_handler_registries((duplicate, duplicate))
