from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers.hooks import HookRegistry

from latentslate_engine.runtime.wan22_flf_conditioning import (
    prepare_wan_flf_conditioning,
    preprocess_wan_flf_image,
)
from latentslate_engine.runtime.wan22_i2v_forward import WanI2VForward


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
    start = torch.full((1, 3, 64, 64), -1.0, dtype=torch.float32)
    end = torch.full((1, 3, 64, 64), 1.0, dtype=torch.float32)
    state = prepare_wan_flf_conditioning(
        session, start, end, num_frames=9, height=64, width=64, seed=42, device="cpu"
    )

    assert state.noise_latents.shape == (1, 16, 3, 8, 8)
    assert state.condition.shape == (1, 20, 3, 8, 8)
    assert torch.equal(session.video[:, :, 0], start.to(dtype=torch.bfloat16))
    assert torch.equal(session.video[:, :, -1], end.to(dtype=torch.bfloat16))
    assert not session.video[:, :, 1:-1].any()
    mask = state.condition[:, :4]
    # Engine forwards the post-``WAN21.concat_cond`` mask directly. Comfy's
    # node emits zero for endpoint positions, then its model wrapper inverts it.
    assert torch.equal(mask[:, :, 0], torch.ones_like(mask[:, :, 0]))
    assert not mask[:, :, 1].any()
    assert not mask[:, :3, -1].any()
    assert torch.equal(mask[:, 3:4, -1], torch.ones_like(mask[:, 3:4, -1]))
    assert torch.equal(state.condition[:, 4:], torch.full_like(state.condition[:, 4:], 0.25))


def test_flf_direct_mask_matches_pinned_comfy_wan21_post_wrapper_contract():
    """The direct engine tensor equals the active Comfy wrapper's inverted mask."""

    session = FakeVaeSession()
    endpoint = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    state = prepare_wan_flf_conditioning(
        session, endpoint, endpoint, num_frames=9, height=64, width=64, seed=0, device="cpu"
    )

    # Pinned ``WanFirstLastFrameToVideo`` raw node mask, followed by
    # ``WAN21.concat_cond``'s `mask = 1.0 - mask` before its direct concat.
    raw = torch.ones((1, 1, 12, 8, 8), dtype=torch.float32)
    raw[:, :, :4] = 0.0
    raw[:, :, -1:] = 0.0
    comfy_model_mask = 1.0 - raw.view(1, 3, 4, 8, 8).transpose(1, 2)
    assert torch.equal(state.condition[:, :4], comfy_model_mask)


def test_flf_forward_captures_exact_post_comfy_concat_order():
    """The direct transformer sees Comfy's `[noise, inverted-mask, latent]` order."""

    class CaptureTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._latentslate_compute_dtype = torch.float16
            self.hidden_states = None

        def forward(self, *, hidden_states, **_kwargs):
            self.hidden_states = hidden_states.detach().clone()
            return (torch.zeros_like(hidden_states[:, :16]),)

    session = FakeVaeSession()
    start = torch.full((1, 3, 64, 64), -1.0, dtype=torch.float32)
    end = torch.full((1, 3, 64, 64), 1.0, dtype=torch.float32)
    state = prepare_wan_flf_conditioning(
        session, start, end, num_frames=9, height=64, width=64, seed=42, device="cpu"
    )
    model = CaptureTransformer()
    registry = HookRegistry.check_if_exists_or_initialize(model)
    registry._set_context = lambda _identity: None
    latents = torch.full_like(state.noise_latents, 0.125)
    forward = WanI2VForward(state.condition)
    runtime_session = SimpleNamespace(
        transformer=model,
        active=True,
        onload_device=torch.device("cpu"),
    )
    forward(
        model,
        runtime_session,
        latents,
        999,
        torch.zeros((1, 512, 4096), dtype=torch.float16),
        "cond",
    )

    assert model.hidden_states is not None
    assert model.hidden_states.shape == (1, 36, 3, 8, 8)
    assert torch.equal(model.hidden_states[:, :16], latents.to(dtype=torch.float16))
    assert torch.equal(model.hidden_states[:, 16:20], state.condition[:, :4].to(torch.float16))
    assert torch.equal(model.hidden_states[:, 20:], state.condition[:, 4:].to(torch.float16))
    assert torch.equal(
        model.hidden_states[:, 16:20, :, :, :][:, :, 0],
        torch.ones_like(model.hidden_states[:, 16:20, 0]),
    )
    assert not model.hidden_states[:, 16:20, 1].any()


def test_flf_endpoint_preprocess_matches_comfy_center_crop_then_bilinear():
    # A wide non-square input makes the template's horizontal center crop observable.
    source = torch.arange(8 * 16, dtype=torch.float32).reshape(1, 1, 8, 16) / 127.0
    source = source.repeat(1, 3, 1, 1)
    actual = preprocess_wan_flf_image(source, height=64, width=64)
    expected_raw = torch.nn.functional.interpolate(
        source[..., 4:12], size=(64, 64), mode="bilinear"
    )
    assert torch.allclose(actual, expected_raw * 2.0 - 1.0)
