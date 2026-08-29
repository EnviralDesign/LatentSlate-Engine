from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import av
import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor

import latentslate_engine.wan2214b.pipeline as wan_pipeline
from latentslate_engine.wan2214b.pipeline import (
    WanRecipe,
    WanSession,
    canonical_sigmas,
    save_video,
)
from latentslate_engine.wan2214b.weights import (
    ArtifactIdentity,
    WanWeights,
)


class _Store:
    def __init__(self, values: dict[str, torch.Tensor], name: str):
        self.values = values
        self.keys = frozenset(values)
        self.identity = ArtifactIdentity(name, 1, 1)

    def tensor(self, key: str) -> torch.Tensor:
        return self.values[key]

    def close(self) -> None:
        pass


def _small_weights(prefix: str, device: torch.device) -> WanWeights:
    base_float = torch.tensor([[0.25, -0.5], [0.75, 1.0]], dtype=torch.float16)
    scale = base_float.abs().max().float() / 448
    qdata = (base_float / scale.to(base_float.dtype)).to(torch.float8_e4m3fn)
    lora_prefix = f"diffusion_model.{prefix}"
    weights = object.__new__(WanWeights)
    weights.base = _Store(
        {f"{prefix}.weight": qdata, f"{prefix}.scale_weight": scale}, "base"
    )
    weights.lora = _Store(
        {
            f"{lora_prefix}.lora_up.weight": torch.tensor([[1.0], [2.0]]),
            f"{lora_prefix}.lora_down.weight": torch.tensor([[0.5, -0.25]]),
            f"{lora_prefix}.alpha": torch.tensor(1.0),
        },
        "lora",
    )
    weights.lora_strength = 1.0
    weights.native_fp8 = True
    weights._patched_weights = {}
    weights._active_weights = {}
    weights._active_qk_norms = {}
    weights._active_device = device
    weights._base_reopened = False
    weights._materialized_since_reopen = 0
    weights._reopen_before_next_access = False
    weights._prefetch_stream = None
    weights._prefetched_live = None
    return weights


def test_canonical_recipe_and_schedule_are_fixed() -> None:
    recipe = WanRecipe()
    assert recipe.frame_count == 81
    assert (recipe.steps, recipe.split_step, recipe.cfg, recipe.shift) == (
        4,
        2,
        1.0,
        5.000000000000001,
    )
    assert (recipe.width, recipe.height, recipe.duration, recipe.fps) == (
        512,
        512,
        5.0,
        16.0,
    )
    torch.testing.assert_close(
        canonical_sigmas(recipe.shift, recipe.steps),
        torch.tensor([1.0, 0.9375, 0.8333333134651184, 0.625, 0.0]),
        rtol=0,
        atol=0,
    )


def test_recipe_identity_consumes_both_models_loras_and_strengths(
    tmp_path: Path,
) -> None:
    paths = []
    for name in ("high", "high-lora", "low", "low-lora", "text", "vae"):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(name.encode())
        paths.append(str(path))
    recipe = WanRecipe(*paths)
    identity = recipe.identity

    changed = WanRecipe(*paths, high_lora_strength=0.5)
    assert changed.identity != identity
    assert replace(recipe, positive="another prompt").identity == identity
    assert replace(recipe, negative="another negative prompt").identity == identity

    Path(paths[3]).write_bytes(b"changed low lora")
    assert recipe.identity != identity


def test_live_lora_rebuilds_from_immutable_base_without_accumulation() -> None:
    prefix = "blocks.0.self_attn.q"
    weights = _small_weights(prefix, torch.device("cpu"))
    original = weights.base.tensor(f"{prefix}.weight").clone()

    first = weights._patched_weight(prefix, torch.device("cpu"), torch.float16)
    second = weights._patched_weight(prefix, torch.device("cpu"), torch.float16)

    assert isinstance(first, QuantizedTensor)
    assert isinstance(second, QuantizedTensor)
    assert first._qdata.data_ptr() != second._qdata.data_ptr()
    assert torch.equal(first._qdata, second._qdata)
    assert torch.equal(weights.base.tensor(f"{prefix}.weight"), original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP8 is required")
def test_materialized_lora_cache_is_stable_across_phase_reactivation() -> None:
    prefix = "blocks.3.ffn.0"
    device = torch.device("cuda")
    weights = _small_weights(prefix, device)
    weights._active_device = device

    first = weights._patched_weight(prefix, device, torch.float16)
    assert isinstance(first, QuantizedTensor)
    first_qdata = first._qdata.clone()
    weights.deactivate()
    weights.activate(device)
    second = weights._patched_weight(prefix, device, torch.float16)

    assert isinstance(second, QuantizedTensor)
    assert torch.equal(first_qdata, second._qdata)


def test_destroyed_session_is_unusable() -> None:
    session = object.__new__(WanSession)
    session._alive = True
    session._identity = ("recipe",)
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._conditioning_key = ("positive", "negative")
    session._vae = object()
    session.high_weights = object()
    session.low_weights = object()
    session.text_weights = object()

    session.destroy()

    assert session._conditioning is None
    assert session._conditioning_key is None
    assert session.high_weights is None
    with pytest.raises(RuntimeError, match="destructively replaced"):
        _ = session.identity


class _ClosableBase:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


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


def _conditioning_session() -> WanSession:
    session = object.__new__(WanSession)
    session.recipe = WanRecipe()
    session.device = torch.device("cpu")
    session._alive = True
    session._identity = ("model identity",)
    session._conditioning = None
    session._conditioning_key = None
    session.text_weights = _TextWeights()
    session.high_weights = object()
    session.low_weights = object()
    return session


def test_same_prompt_pair_reuses_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingEncoder.calls = []
    monkeypatch.setattr(wan_pipeline, "Umt5Encoder", _RecordingEncoder)
    session = _conditioning_session()

    first = session._ensure_conditioning("positive", "negative")
    second = session._ensure_conditioning("positive", "negative")

    assert first is second
    assert _RecordingEncoder.calls == ["positive", "negative"]


@pytest.mark.parametrize(
    ("changed_positive", "changed_negative"),
    [("new positive", "negative"), ("positive", "new negative")],
    ids=["positive", "negative"],
)
def test_changed_prompt_recomputes_only_conditioning(
    monkeypatch: pytest.MonkeyPatch,
    changed_positive: str,
    changed_negative: str,
) -> None:
    _RecordingEncoder.calls = []
    created: list[_TextWeights] = []

    def new_text_weights(*_args: object, **_kwargs: object) -> _TextWeights:
        weights = _TextWeights()
        created.append(weights)
        return weights

    monkeypatch.setattr(wan_pipeline, "Umt5Encoder", _RecordingEncoder)
    monkeypatch.setattr(wan_pipeline, "WanWeights", new_text_weights)
    session = _conditioning_session()
    high_weights = session.high_weights
    low_weights = session.low_weights
    identity = session.identity

    first = session._ensure_conditioning("positive", "negative")
    second = session._ensure_conditioning(changed_positive, changed_negative)

    assert first is not second
    assert len(_RecordingEncoder.calls) == 4
    assert len(created) == 1
    assert session.high_weights is high_weights
    assert session.low_weights is low_weights
    assert session.identity == identity


def test_model_identity_replacement_destroys_all_retained_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _conditioning_session()
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._conditioning_key = ("positive", "negative")
    session._vae = object()
    session.device = torch.device("cpu")

    def replacement_init(
        replacement: WanSession, recipe: WanRecipe, device: torch.device
    ) -> None:
        replacement.recipe = recipe
        replacement.device = device

    monkeypatch.setattr(WanSession, "__init__", replacement_init)
    changed_recipe = replace(session.recipe, high_lora_strength=0.5)

    replacement = session.replaced(changed_recipe)

    assert replacement.recipe is changed_recipe
    assert session._alive is False
    assert session._conditioning is None
    assert session._conditioning_key is None
    assert session._vae is None
    assert session.high_weights is None
    assert session.low_weights is None
    assert session.text_weights is None


def test_video_writer_emits_canonical_media_metadata(tmp_path: Path) -> None:
    path = tmp_path / "wan.mp4"
    images = torch.linspace(0, 1, 2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
    save_video(images, path, 16.0)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        assert stream.average_rate == 16
        assert stream.frames == 2
