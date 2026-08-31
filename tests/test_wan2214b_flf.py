from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch

import latentslate_engine.wan2214b.flf as wan_flf
import latentslate_engine.wan2214b.pipeline as wan_pipeline
from latentslate_engine.wan2214b.flf import (
    FLFConditioning,
    OrderedSourceIdentity,
    WanFLFRecipe,
    WanFLFSession,
    _build_flf_conditioning,
    _model_conditioning,
)
from latentslate_engine.wan2214b.i2v import SourceImageIdentity


def test_canonical_flf_recipe_uses_i2v_artifacts_and_fixture_values() -> None:
    recipe = WanFLFRecipe()

    assert "i2v_high_noise" in recipe.high_checkpoint
    assert "i2v_low_noise" in recipe.low_checkpoint
    assert "i2v_lightx2v" in recipe.high_lora
    assert "i2v_lightx2v" in recipe.low_lora
    assert recipe.high_lora_strength == 1
    assert recipe.low_lora_strength == 1
    assert recipe.frame_count == 81
    assert recipe.positive == wan_flf.POSITIVE_PROMPT


def _flf_session() -> WanFLFSession:
    session = object.__new__(WanFLFSession)
    session.recipe = WanFLFRecipe()
    session.device = torch.device("cpu")
    session._flf_conditioning = None
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._conditioning_key = ("positive", "negative")
    session.high_weights = object()
    session.low_weights = object()
    return session


def _conditioning(identity: OrderedSourceIdentity, value: float) -> FLFConditioning:
    return FLFConditioning(
        identity=identity,
        latent=torch.full((1, 16, 21, 1, 1), value),
        mask=torch.zeros((1, 4, 21, 1, 1)),
    )


def test_same_endpoint_bytes_at_changed_paths_reuse_joint_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    first_alias = tmp_path / "first-alias.png"
    last = tmp_path / "last.png"
    last_alias = tmp_path / "last-alias.png"
    first.write_bytes(b"first image")
    first_alias.write_bytes(first.read_bytes())
    last.write_bytes(b"last image")
    last_alias.write_bytes(last.read_bytes())
    builds: list[OrderedSourceIdentity] = []

    def build(
        _first: str | Path,
        _last: str | Path,
        identity: OrderedSourceIdentity,
        _recipe: WanFLFRecipe,
        _device: torch.device,
    ) -> FLFConditioning:
        builds.append(identity)
        return _conditioning(identity, float(len(builds)))

    monkeypatch.setattr(wan_flf, "_build_flf_conditioning", build)
    session = _flf_session()

    original = session._ensure_flf_conditioning(first, last, 512, 512, 81)
    reused = session._ensure_flf_conditioning(first_alias, last_alias, 512, 512, 81)

    assert original is reused
    assert len(builds) == 1


@pytest.mark.parametrize("change", ["first", "last", "swap"])
def test_changed_ordered_pair_rebuilds_only_image_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    changed = tmp_path / "changed.png"
    first.write_bytes(b"first image")
    last.write_bytes(b"last image")
    changed.write_bytes(b"changed image")
    builds: list[OrderedSourceIdentity] = []

    def build(
        _first: str | Path,
        _last: str | Path,
        identity: OrderedSourceIdentity,
        _recipe: WanFLFRecipe,
        _device: torch.device,
    ) -> FLFConditioning:
        builds.append(identity)
        return _conditioning(identity, float(len(builds)))

    monkeypatch.setattr(wan_flf, "_build_flf_conditioning", build)
    session = _flf_session()
    prompt_state = session._conditioning
    high = session.high_weights
    low = session.low_weights
    original = session._ensure_flf_conditioning(first, last, 512, 512, 81)
    candidates = {
        "first": (changed, last),
        "last": (first, changed),
        "swap": (last, first),
    }

    rebuilt = session._ensure_flf_conditioning(*candidates[change], 512, 512, 81)

    assert rebuilt is not original
    assert len(builds) == 2
    assert session._conditioning is prompt_state
    assert session.high_weights is high
    assert session.low_weights is low


@pytest.mark.parametrize(
    ("width", "height", "frame_count"),
    [(832, 480, 81), (480, 832, 81), (512, 512, 17)],
)
def test_geometry_or_temporal_change_rebuilds_ordered_joint_conditioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    builds: list[OrderedSourceIdentity] = []

    def build(
        _first: str | Path,
        _last: str | Path,
        identity: OrderedSourceIdentity,
        _recipe: WanFLFRecipe,
        _device: torch.device,
    ) -> FLFConditioning:
        builds.append(identity)
        return _conditioning(identity, float(len(builds)))

    monkeypatch.setattr(wan_flf, "_build_flf_conditioning", build)
    session = _flf_session()
    prompt_state = session._conditioning
    high = session.high_weights
    low = session.low_weights
    original = session._ensure_flf_conditioning(first, last, 512, 512, 81)

    rebuilt = session._ensure_flf_conditioning(first, last, width, height, frame_count)

    assert rebuilt is not original
    assert len(builds) == 2
    assert session._conditioning is prompt_state
    assert session.high_weights is high
    assert session.low_weights is low


def test_flf_conditioning_consumes_target_geometry_and_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Encoder:
        encoder_input: torch.Tensor | None = None

        def encode(self, encoder_input: torch.Tensor) -> torch.Tensor:
            self.encoder_input = encoder_input.cpu()
            return torch.zeros((1, 16, 5, 2, 2))

    encoder = Encoder()
    sources = {
        "first.png": torch.zeros((1, 16, 16, 3)),
        "last.png": torch.ones((1, 16, 16, 3)),
    }
    monkeypatch.setattr(wan_flf, "_load_source_image", sources.__getitem__)
    monkeypatch.setattr(wan_flf, "_load_vae_encoder", lambda _path, _device: encoder)
    identity = OrderedSourceIdentity(
        SourceImageIdentity(Path("first.png"), 1, "first"),
        SourceImageIdentity(Path("last.png"), 1, "last"),
        16,
        16,
        17,
    )

    conditioning = _build_flf_conditioning(
        "first.png", "last.png", identity, WanFLFRecipe(), torch.device("cpu")
    )

    assert encoder.encoder_input is not None
    assert encoder.encoder_input.shape == (1, 3, 17, 16, 16)
    torch.testing.assert_close(
        encoder.encoder_input[:, :, 0],
        torch.full((1, 3, 16, 16), -1, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        encoder.encoder_input[:, :, 1:-1],
        torch.zeros((1, 3, 15, 16, 16), dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        encoder.encoder_input[:, :, -1],
        torch.ones((1, 3, 16, 16), dtype=torch.bfloat16),
    )
    assert conditioning.latent.shape == (1, 16, 5, 2, 2)
    assert conditioning.mask.shape == (1, 4, 5, 2, 2)
    assert torch.count_nonzero(conditioning.mask[:, :, 0]) == 0
    assert torch.all(conditioning.mask[:, :, 1:-1] == 1)
    torch.testing.assert_close(
        conditioning.mask[0, :, -1, 0, 0], torch.tensor([1.0, 1.0, 1.0, 0.0])
    )


def test_model_conditioning_uses_flf_mask_topology_then_normalized_latent() -> None:
    latent = (
        torch.tensor(wan_flf.LATENT_MEAN).view(1, 16, 1, 1, 1).repeat(1, 1, 21, 1, 1)
    )
    mask = torch.ones((1, 4, 21, 1, 1))
    mask[:, :, 0] = 0
    mask[:, 3, -1] = 0
    identity = OrderedSourceIdentity(
        SourceImageIdentity(Path("first.png"), 1, "first"),
        SourceImageIdentity(Path("last.png"), 1, "last"),
        512,
        512,
        81,
    )

    conditioning = _model_conditioning(
        FLFConditioning(identity, latent, mask), torch.device("cpu")
    )

    assert conditioning.shape == (1, 20, 21, 1, 1)
    torch.testing.assert_close(
        conditioning[0, :4, 0, 0, 0], torch.ones(4, dtype=torch.float16)
    )
    torch.testing.assert_close(
        conditioning[0, :4, 1, 0, 0], torch.zeros(4, dtype=torch.float16)
    )
    torch.testing.assert_close(
        conditioning[0, :4, -1, 0, 0],
        torch.tensor([0, 0, 0, 1], dtype=torch.float16),
    )
    torch.testing.assert_close(
        conditioning[:, 4:], torch.zeros((1, 16, 21, 1, 1), dtype=torch.float16)
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
def test_changed_prompt_retains_ordered_pair_and_model_state(
    monkeypatch: pytest.MonkeyPatch,
    changed_positive: str,
    changed_negative: str,
) -> None:
    _RecordingEncoder.calls = []
    monkeypatch.setattr(wan_pipeline, "Umt5Encoder", _RecordingEncoder)
    monkeypatch.setattr(
        wan_pipeline, "WanWeights", lambda *_args, **_kwargs: _TextWeights()
    )
    session = _flf_session()
    session._conditioning = None
    session._conditioning_key = None
    session.text_weights = _TextWeights()
    identity = OrderedSourceIdentity(
        SourceImageIdentity(Path("first.png"), 1, "first"),
        SourceImageIdentity(Path("last.png"), 1, "last"),
        512,
        512,
        81,
    )
    image = _conditioning(identity, 0)
    session._flf_conditioning = image
    high = session.high_weights
    low = session.low_weights

    first = session._ensure_conditioning("positive", "negative")
    second = session._ensure_conditioning(changed_positive, changed_negative)

    assert first is not second
    assert session._flf_conditioning is image
    assert session.high_weights is high
    assert session.low_weights is low


def test_true_recipe_replacement_destroys_all_flf_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _flf_session()
    session._alive = True
    session._identity = ("model identity",)
    session._vae = object()
    session.text_weights = object()
    identity = OrderedSourceIdentity(
        SourceImageIdentity(Path("first.png"), 1, "first"),
        SourceImageIdentity(Path("last.png"), 1, "last"),
        512,
        512,
        81,
    )
    session._flf_conditioning = _conditioning(identity, 0)

    def replacement_init(
        replacement: WanFLFSession,
        recipe: WanFLFRecipe,
        device: torch.device,
    ) -> None:
        replacement.recipe = recipe
        replacement.device = device

    monkeypatch.setattr(WanFLFSession, "__init__", replacement_init)
    changed = replace(session.recipe, high_lora_strength=0.5)

    replacement = session.replaced(changed)

    assert replacement.recipe is changed
    assert session._alive is False
    assert session._flf_conditioning is None
    assert session._conditioning is None
    assert session._vae is None
    assert session.high_weights is None
    assert session.low_weights is None
