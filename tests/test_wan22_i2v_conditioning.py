from __future__ import annotations

import pytest
import torch

from latentslate_engine.runtime.wan22_i2v_conditioning import (
    build_wan_i2v_model_input,
    prepare_wan_i2v_conditioning,
    preprocess_wan_i2v_image,
)


class FakeVaeSession:
    def __init__(self):
        self.video = None

    def encode(self, video):
        self.video = video.clone()
        batch, _, frames, height, width = video.shape
        return torch.full(
            (batch, 16, ((frames - 1) // 4) + 1, height // 8, width // 8),
            0.25,
            dtype=torch.bfloat16,
            device=video.device,
        )


def test_prepare_conditioning_builds_exact_comfy_wan_shapes_and_mask():
    session = FakeVaeSession()
    image = torch.ones((1, 3, 704, 1280), dtype=torch.float32)
    state = prepare_wan_i2v_conditioning(
        session,
        image,
        num_frames=25,
        height=704,
        width=1280,
        seed=42,
        device="cpu",
    )

    assert state.noise_latents.shape == (1, 16, 7, 88, 160)
    assert state.condition.shape == (1, 20, 7, 88, 160)
    assert state.model_input_channels == 36
    assert build_wan_i2v_model_input(state).shape == (1, 36, 7, 88, 160)
    assert session.video.shape == (1, 3, 25, 704, 1280)
    assert torch.equal(session.video[:, :, 0], torch.ones_like(session.video[:, :, 0]))
    assert not session.video[:, :, 1:].any()
    mask = state.condition[:, :4]
    assert torch.equal(mask[:, :, 0], torch.ones_like(mask[:, :, 0]))
    assert not mask[:, :, 1:].any()
    assert torch.equal(state.condition[:, 4:], torch.full_like(state.condition[:, 4:], 0.25))


def test_seed_is_deterministic_and_changes_noise():
    image = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    first = prepare_wan_i2v_conditioning(
        FakeVaeSession(), image, num_frames=5, height=16, width=16, seed=7, device="cpu"
    )
    second = prepare_wan_i2v_conditioning(
        FakeVaeSession(), image, num_frames=5, height=16, width=16, seed=7, device="cpu"
    )
    third = prepare_wan_i2v_conditioning(
        FakeVaeSession(), image, num_frames=5, height=16, width=16, seed=8, device="cpu"
    )
    assert torch.equal(first.noise_latents, second.noise_latents)
    assert not torch.equal(first.noise_latents, third.noise_latents)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device canonicalization proof")
def test_unspecified_cuda_uses_current_ordinal_and_rejects_wrong_device():
    image = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    state = prepare_wan_i2v_conditioning(
        FakeVaeSession(), image, num_frames=5, height=16, width=16, seed=0, device="cuda"
    )
    assert state.noise_latents.device == torch.device("cuda", torch.cuda.current_device())

    class WrongDeviceVae:
        def encode(self, _video):
            return torch.zeros((1, 16, 2, 2, 2), dtype=torch.bfloat16, device="cpu")

    with pytest.raises(RuntimeError, match="incompatible"):
        prepare_wan_i2v_conditioning(
            WrongDeviceVae(), image, num_frames=5, height=16, width=16, seed=0, device="cuda"
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"height": 18}, "divisible by 16"),
        ({"width": 18}, "divisible by 16"),
        ({"num_frames": 4}, "4k\\+1"),
        ({"num_frames": 125}, "safety budget"),
        ({"height": 720, "width": 1280}, "safety budget"),
        ({"seed": -1}, "seed"),
    ],
)
def test_conditioning_rejects_invalid_dimensions_and_seed(kwargs, error):
    values = {"num_frames": 5, "height": 16, "width": 16, "seed": 0}
    values.update(kwargs)
    image = torch.zeros((1, 3, values["height"], values["width"]), dtype=torch.float32)
    with pytest.raises(ValueError, match=error):
        prepare_wan_i2v_conditioning(FakeVaeSession(), image, device="cpu", **values)


def test_preprocess_uses_pinned_video_processor_range_and_shape():
    image = torch.ones((3, 16, 16), dtype=torch.float32)
    processed = preprocess_wan_i2v_image(image, height=16, width=16)
    assert processed.shape == (1, 3, 16, 16)
    assert processed.dtype == torch.float32
    assert bool((processed <= 1).all()) and bool((processed >= -1).all())


def test_model_input_rejects_tampered_state():
    image = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    state = prepare_wan_i2v_conditioning(
        FakeVaeSession(), image, num_frames=5, height=16, width=16, seed=0, device="cpu"
    )
    state.condition[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="transformer input contract"):
        build_wan_i2v_model_input(state)
