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
    FRAME_RATE,
    MAX_FRAME_COUNT,
    MAX_PIXELS,
    MAX_SEED,
    MIN_FRAME_COUNT,
    WanRecipe,
    WanSession,
    canonical_sigmas,
    cpu_noise,
    half_open_delivery,
    latent_shape,
    save_half_open_video,
    save_video,
    transformer_token_count,
    validate_request,
)
from latentslate_engine.wan2214b.timing import (
    delivery_frame_count,
    native_frame_count,
    validate_duration_seconds,
)
from latentslate_engine.wan2214b.weights import (
    MATERIALIZED_PATCH_COUNT,
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
    weights._materialized_device_bytes = qdata.nbytes + scale.nbytes
    weights._retain_materialized_on_device = True
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
    assert (recipe.width, recipe.height, recipe.frame_count) == (
        512,
        512,
        81,
    )
    assert FRAME_RATE == 16
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
    assert replace(recipe, width=832, height=480, frame_count=17).identity == identity

    Path(paths[3]).write_bytes(b"changed low lora")
    assert recipe.identity != identity


def test_request_validation_accepts_the_complete_recovered_boundaries() -> None:
    accepted = (
        (480, 480, MIN_FRAME_COUNT, 0),
        (512, 512, 81, 923510416338945),
        (1024, 576, 81, 4),
        (480, 832, 81, 5),
        (832, 480, 49, MAX_SEED),
        (480, 832, 17, 1),
        (1280, 720, 17, 2),
        (720, 1280, 17, 3),
    )
    for request in accepted:
        validate_request(*request)


def test_duration_maps_to_native_lattice_and_half_open_delivery() -> None:
    assert native_frame_count(1.0) == 17
    assert native_frame_count(5.0) == 81
    assert delivery_frame_count(5.0) == 80
    images = torch.arange(81)
    delivered = half_open_delivery(images)
    assert delivered.shape[0] == 80
    assert delivered[-1].item() == 79
    assert images[-1].item() == 80


def test_family_delivery_boundary_retains_terminal_sample_until_encoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = torch.arange(81)
    encoded: list[torch.Tensor] = []
    monkeypatch.setattr(
        wan_pipeline,
        "save_video",
        lambda images, _path, _fps: encoded.append(images.clone()),
    )

    delivered = save_half_open_video(native, tmp_path / "wan.mp4", 16.0)

    assert native.shape[0] == 81
    assert native[-1].item() == 80
    assert delivered.shape[0] == 80
    assert encoded[0].shape[0] == 80


@pytest.mark.parametrize("duration", [0.75, 5.25, 1.1, float("inf"), float("nan")])
def test_invalid_wan_duration_is_rejected_early(duration: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_duration_seconds(duration)


@pytest.mark.parametrize(
    ("width", "height", "frames", "seed", "message"),
    [
        (479, 480, 17, 0, "multiples of 16"),
        (464, 480, 17, 0, "at least 480"),
        (1296, 720, 17, 0, "must not exceed"),
        (864, 480, 17, 0, "16:9"),
        (512, 512, 13, 0, "between 17 and 81"),
        (512, 512, 85, 0, "between 17 and 81"),
        (512, 512, 18, 0, r"4n\+1"),
        (512, 512, 17, -1, "between 0"),
        (512, 512, 17, MAX_SEED + 1, "between 0"),
    ],
)
def test_request_validation_rejects_without_coercion(
    width: int, height: int, frames: int, seed: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_request(width, height, frames, seed)


def test_request_formulas_cover_every_temporal_lattice_point() -> None:
    for frames in range(MIN_FRAME_COUNT, MAX_FRAME_COUNT + 1, 4):
        validate_request(512, 512, frames, 0)
        assert latent_shape(512, 512, frames) == (
            1,
            16,
            (frames - 1) // 4 + 1,
            64,
            64,
        )
        assert transformer_token_count(512, 512, frames) == (
            ((frames - 1) // 4 + 1) * 32 * 32
        )
    assert transformer_token_count(512, 512, 81) == 21_504
    assert transformer_token_count(480, 832, 49) == 20_280
    assert MAX_PIXELS == 921_600


def test_every_spatial_lattice_point_matches_the_recovered_envelope() -> None:
    for width in range(480, 1920 + 16, 16):
        for height in range(480, 1920 + 16, 16):
            expected = (
                width * height <= MAX_PIXELS
                and max(width, height) * 9 <= min(width, height) * 16
            )
            try:
                validate_request(width, height, 17, 0)
            except ValueError:
                accepted = False
            else:
                accepted = True
            assert accepted is expected


def test_public_seed_builds_canonical_cpu_noise_for_the_high_stage() -> None:
    shape = (1, 16, 21, 64, 64)
    expected = torch.randn(
        shape,
        dtype=torch.float32,
        generator=torch.manual_seed(923510416338945),
        device="cpu",
    )

    actual = cpu_noise(923510416338945, 512, 512, 81)

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, MAX_SEED])
def test_unsigned_seed_boundaries_build_exact_cpu_noise(seed: int) -> None:
    expected = torch.randn(
        (1, 16, 5, 60, 60),
        dtype=torch.float32,
        generator=torch.manual_seed(seed),
        device="cpu",
    )

    actual = cpu_noise(seed, 480, 480, 17)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_request_default_replacement_retains_warm_state(tmp_path: Path) -> None:
    paths = []
    for name in ("high", "high-lora", "low", "low-lora", "text", "vae"):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(name.encode())
        paths.append(str(path))
    recipe = WanRecipe(*paths)
    session = object.__new__(WanSession)
    session.recipe = recipe
    session.device = torch.device("cpu")
    session._alive = True
    session._identity = recipe.identity
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._conditioning_key = ("positive", "negative")
    session._vae = object()
    session.high_weights = object()
    session.low_weights = object()
    session.text_weights = object()
    warm_state = (
        session._conditioning,
        session._vae,
        session.high_weights,
        session.low_weights,
    )
    request_defaults = replace(recipe, width=832, height=480, frame_count=17)

    retained = session.replaced(request_defaults)

    assert retained is session
    assert session.recipe is request_defaults
    assert session._alive is True
    assert (
        session._conditioning,
        session._vae,
        session.high_weights,
        session.low_weights,
    ) == warm_state


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


def test_cold_patch_cache_streams_before_warm_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "blocks.3.ffn.0"
    device = torch.device("cpu")
    weights = _small_weights(prefix, device)
    weights._active_device = None
    weights._patched_weights = {}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1 << 40, 1 << 40))
    stream = type("Stream", (), {"synchronize": lambda self: None})()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: stream)

    weights.activate(device)
    assert weights._retain_materialized_on_device is False

    weights.deactivate()
    cached = _small_weights(prefix, device)._patched_weight(
        prefix, device, torch.float16
    )
    assert isinstance(cached, QuantizedTensor)
    weights._patched_weights = {
        f"cached-{index}": cached for index in range(MATERIALIZED_PATCH_COUNT)
    }
    weights.activate(device)
    assert weights._retain_materialized_on_device is True


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


def test_five_second_delivery_artifact_has_80_frames_at_16_fps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wan-five-seconds.mp4"
    save_video(torch.zeros((80, 16, 16, 3)), path, 16.0)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 16
        assert stream.frames == 80
        assert float(stream.duration * stream.time_base) == 5.0
