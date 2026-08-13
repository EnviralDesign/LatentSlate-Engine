from __future__ import annotations

import torch

from latentslate_engine.runtime.wan22_flf_conditioning import (
    prepare_wan_flf_conditioning,
    preprocess_wan_flf_image,
)


class FakeVaeSession:
    def __init__(self) -> None:
        self.video = None

    def encode(self, video):
        self.video = video.clone()
        return torch.full(
            (1, 16, ((video.shape[2] - 1) // 4) + 1, video.shape[3] // 8, video.shape[4] // 8),
            0.25,
            dtype=torch.bfloat16,
            device=video.device,
        )


def test_flf_conditioning_matches_comfy_composite_and_temporal_mask():
    session = FakeVaeSession()
    start = torch.full((1, 3, 16, 16), -1.0, dtype=torch.float32)
    end = torch.full((1, 3, 16, 16), 1.0, dtype=torch.float32)
    state = prepare_wan_flf_conditioning(
        session, start, end, num_frames=9, height=16, width=16, seed=42, device="cpu"
    )

    assert state.noise_latents.shape == (1, 16, 3, 2, 2)
    assert state.condition.shape == (1, 20, 3, 2, 2)
    assert torch.equal(session.video[:, :, 0], start.to(dtype=torch.bfloat16))
    assert torch.equal(session.video[:, :, -1], end.to(dtype=torch.bfloat16))
    assert not session.video[:, :, 1:-1].any()
    mask = state.condition[:, :4]
    assert not mask[:, :, 0].any()  # start frame plus Comfy's three causal positions
    assert torch.equal(mask[:, :, 1], torch.ones_like(mask[:, :, 1]))
    assert torch.equal(mask[:, :3, -1], torch.ones_like(mask[:, :3, -1]))
    assert not mask[:, 3:4, -1].any()
    assert torch.equal(state.condition[:, 4:], torch.full_like(state.condition[:, 4:], 0.25))


def test_flf_endpoint_preprocess_matches_comfy_center_crop_then_bilinear():
    # A wide non-square input makes the template's horizontal center crop observable.
    source = torch.arange(8 * 16, dtype=torch.float32).reshape(1, 1, 8, 16) / 127.0
    source = source.repeat(1, 3, 1, 1)
    actual = preprocess_wan_flf_image(source, height=16, width=16)
    expected_raw = torch.nn.functional.interpolate(
        source[..., 4:12], size=(16, 16), mode="bilinear"
    )
    assert torch.allclose(actual, expected_raw * 2.0 - 1.0)
