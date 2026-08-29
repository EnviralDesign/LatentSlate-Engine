from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch

import latentslate_engine.wan2214b.i2v as wan_i2v
import latentslate_engine.wan2214b.pipeline as wan_pipeline
from latentslate_engine.wan2214b.i2v import (
    ImageConditioning,
    SourceImageIdentity,
    WanI2VRecipe,
    WanI2VSession,
    _model_conditioning,
)


def test_canonical_i2v_recipe_uses_distinct_artifacts_and_prompt() -> None:
    recipe = WanI2VRecipe()

    assert "i2v_high_noise" in recipe.high_checkpoint
    assert "i2v_low_noise" in recipe.low_checkpoint
    assert "i2v_lightx2v" in recipe.high_lora
    assert "i2v_lightx2v" in recipe.low_lora
    assert recipe.frame_count == 81
    assert recipe.positive == wan_i2v.POSITIVE_PROMPT
    assert recipe.negative == wan_i2v.NEGATIVE_PROMPT


def test_source_identity_is_content_derived_across_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same image bytes")
    second.write_bytes(first.read_bytes())

    assert SourceImageIdentity.from_path(first) == SourceImageIdentity.from_path(second)


def _image_session() -> WanI2VSession:
    session = object.__new__(WanI2VSession)
    session.recipe = WanI2VRecipe()
    session.device = torch.device("cpu")
    session._image_conditioning = None
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._conditioning_key = ("positive", "negative")
    session.high_weights = object()
    session.low_weights = object()
    return session


def test_same_bytes_reuse_image_conditioning_and_changed_bytes_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    same = tmp_path / "same.png"
    changed = tmp_path / "changed.png"
    first.write_bytes(b"first image")
    same.write_bytes(first.read_bytes())
    changed.write_bytes(b"changed image")
    builds: list[SourceImageIdentity] = []

    def build(
        _path: str | Path,
        identity: SourceImageIdentity,
        _recipe: WanI2VRecipe,
        _device: torch.device,
    ) -> ImageConditioning:
        builds.append(identity)
        return ImageConditioning(
            identity,
            torch.full((1, 16, 1, 1, 1), float(len(builds))),
            torch.zeros((1, 1, 1, 1, 1)),
        )

    monkeypatch.setattr(wan_i2v, "_build_image_conditioning", build)
    session = _image_session()
    prompts = session._conditioning
    high = session.high_weights
    low = session.low_weights

    original = session._ensure_image_conditioning(first)
    reused = session._ensure_image_conditioning(same)
    rebuilt = session._ensure_image_conditioning(changed)

    assert original is reused
    assert rebuilt is not original
    assert len(builds) == 2
    assert session._conditioning is prompts
    assert session.high_weights is high
    assert session.low_weights is low


def test_model_conditioning_is_mask_then_normalized_latent() -> None:
    latent = (
        torch.tensor(wan_i2v.LATENT_MEAN).view(1, 16, 1, 1, 1).repeat(1, 1, 2, 1, 1)
    )
    mask = torch.ones((1, 1, 2, 1, 1))
    mask[:, :, 0] = 0
    image = ImageConditioning(SourceImageIdentity(1, "hash"), latent, mask)

    conditioning = _model_conditioning(image, torch.device("cpu"))

    assert conditioning.shape == (1, 20, 2, 1, 1)
    torch.testing.assert_close(
        conditioning[:, :4, :, 0, 0],
        torch.tensor([[[1.0, 0.0]] * 4], dtype=torch.float16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        conditioning[:, 4:, 0],
        torch.zeros((1, 16, 1, 1), dtype=torch.float16),
        rtol=0,
        atol=0,
    )


class _ClosableBase:
    def close(self) -> None:
        pass


class _TextWeights:
    def __init__(self) -> None:
        self.base = _ClosableBase()


class _RecordingEncoder:
    calls: ClassVar[list[str]] = []

    def __init__(self, _weights: _TextWeights) -> None:
        pass

    def encode(
        self, prompt: str, _device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(prompt)
        value = torch.tensor([[[float(len(self.calls))]]])
        return torch.zeros(1, 1), torch.ones(1, 1), value


@pytest.mark.parametrize(
    ("changed_positive", "changed_negative"),
    [("new positive", "negative"), ("positive", "new negative")],
    ids=["positive", "negative"],
)
def test_changed_prompt_retains_image_and_model_state(
    monkeypatch: pytest.MonkeyPatch,
    changed_positive: str,
    changed_negative: str,
) -> None:
    _RecordingEncoder.calls = []
    monkeypatch.setattr(wan_pipeline, "Umt5Encoder", _RecordingEncoder)
    monkeypatch.setattr(
        wan_pipeline, "WanWeights", lambda *_args, **_kwargs: _TextWeights()
    )
    session = _image_session()
    session._conditioning = None
    session._conditioning_key = None
    session.text_weights = _TextWeights()
    image = ImageConditioning(
        SourceImageIdentity(1, "hash"), torch.zeros(1), torch.zeros(1)
    )
    session._image_conditioning = image
    high = session.high_weights
    low = session.low_weights

    first = session._ensure_conditioning("positive", "negative")
    second = session._ensure_conditioning(changed_positive, changed_negative)

    assert first is not second
    assert _RecordingEncoder.calls == [
        "positive",
        "negative",
        changed_positive,
        changed_negative,
    ]
    assert session._image_conditioning is image
    assert session.high_weights is high
    assert session.low_weights is low


def test_true_recipe_replacement_destroys_image_and_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _image_session()
    session._alive = True
    session._identity = ("model identity",)
    session._vae = object()
    session.text_weights = object()
    session._image_conditioning = ImageConditioning(
        SourceImageIdentity(1, "hash"), torch.zeros(1), torch.zeros(1)
    )

    def replacement_init(
        replacement: WanI2VSession,
        recipe: WanI2VRecipe,
        device: torch.device,
    ) -> None:
        replacement.recipe = recipe
        replacement.device = device

    monkeypatch.setattr(WanI2VSession, "__init__", replacement_init)
    changed = replace(session.recipe, high_lora_strength=0.5)

    replacement = session.replaced(changed)

    assert replacement.recipe is changed
    assert session._alive is False
    assert session._image_conditioning is None
    assert session._conditioning is None
    assert session._vae is None
    assert session.high_weights is None
    assert session.low_weights is None
