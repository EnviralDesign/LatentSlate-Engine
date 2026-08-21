"""Deterministic Wan 2.2 I2V first-frame latent conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .dimensions import require_dimensions
from .framework.residency import canonical_device
from .wan22_canvas import WAN_I2V_CANVAS

WAN_I2V_MAX_FRAMES = 121


class WanVaeSessionLike(Protocol):
    def encode(self, video: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class WanI2VConditioning:
    noise_latents: torch.Tensor
    condition: torch.Tensor
    num_frames: int
    height: int
    width: int
    seed: int

    @property
    def model_input_channels(self) -> int:
        return self.noise_latents.shape[1] + self.condition.shape[1]


def preprocess_wan_i2v_image(image: Any, *, height: int, width: int) -> torch.Tensor:
    """Use the pinned Diffusers video processor for one normalized RGB frame."""

    _validate_dimensions(height=height, width=width, num_frames=1)
    from diffusers.video_processor import VideoProcessor

    processed = VideoProcessor(vae_scale_factor=8).preprocess(
        image,
        height=height,
        width=width,
    )
    if (
        not isinstance(processed, torch.Tensor)
        or processed.shape != (1, 3, height, width)
        or processed.dtype != torch.float32
        or not bool(torch.isfinite(processed).all())
        or bool((processed < -1.0).any())
        or bool((processed > 1.0).any())
    ):
        raise ValueError("Wan I2V image preprocessing did not produce finite [1,3,H,W] in [-1,1]")
    return processed


def prepare_wan_i2v_conditioning(
    vae_session: WanVaeSessionLike,
    image: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device | str,
) -> WanI2VConditioning:
    """Encode the first frame and build exact 16+4+16 channel Wan input state."""

    _validate_dimensions(height=height, width=width, num_frames=num_frames)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Wan I2V seed must be an integer in [0, 2^63)")
    target = canonical_device(torch.device(device))
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Wan I2V conditioning supports CPU or CUDA devices")
    if (
        not isinstance(image, torch.Tensor)
        or image.shape != (1, 3, height, width)
        or image.dtype != torch.float32
        or image.device.type != "cpu"
        or not bool(torch.isfinite(image).all())
        or bool((image < -1.0).any())
        or bool((image > 1.0).any())
    ):
        raise ValueError("Wan I2V image must be a CPU float32 [1,3,H,W] tensor in [-1,1]")

    latent_frames = ((num_frames - 1) // 4) + 1
    latent_height = height // 8
    latent_width = width // 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, 16, latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(target)

    video = torch.zeros((1, 3, num_frames, height, width), dtype=torch.bfloat16, device=target)
    video[:, :, 0] = image.to(device=target, dtype=torch.bfloat16)
    latent_condition = vae_session.encode(video)
    expected_latent_shape = (1, 16, latent_frames, latent_height, latent_width)
    if (
        not isinstance(latent_condition, torch.Tensor)
        or latent_condition.shape != expected_latent_shape
        or latent_condition.device != target
        or latent_condition.dtype not in {torch.float16, torch.bfloat16}
        or not bool(torch.isfinite(latent_condition).all())
    ):
        raise RuntimeError(
            "Wan I2V VAE condition is incompatible with the 16-channel latent contract"
        )

    frame_mask = torch.zeros(
        (1, 1, num_frames, latent_height, latent_width),
        dtype=torch.float32,
        device=target,
    )
    frame_mask[:, :, 0] = 1.0
    first_frame = frame_mask[:, :, 0:1].repeat_interleave(4, dim=2)
    expanded_mask = torch.cat((first_frame, frame_mask[:, :, 1:]), dim=2)
    mask = expanded_mask.view(1, latent_frames, 4, latent_height, latent_width).transpose(1, 2)
    condition = torch.cat((mask, latent_condition.to(dtype=torch.float32)), dim=1)
    if condition.shape != (1, 20, latent_frames, latent_height, latent_width):
        raise RuntimeError("Wan I2V condition did not produce the required 20 channels")
    return WanI2VConditioning(
        noise_latents=noise,
        condition=condition,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )


def build_wan_i2v_model_input(state: WanI2VConditioning) -> torch.Tensor:
    """Concatenate scheduler noise and immutable I2V condition for the transformer."""

    if (
        state.noise_latents.dtype != torch.float32
        or state.condition.dtype != torch.float32
        or state.noise_latents.device != state.condition.device
        or state.noise_latents.shape[0] != 1
        or state.noise_latents.shape[1] != 16
        or state.condition.shape[0] != 1
        or state.condition.shape[1] != 20
        or state.noise_latents.shape[2:] != state.condition.shape[2:]
        or not bool(torch.isfinite(state.noise_latents).all())
        or not bool(torch.isfinite(state.condition).all())
    ):
        raise ValueError("Wan I2V state is incompatible with the transformer input contract")
    result = torch.cat((state.noise_latents, state.condition), dim=1)
    if result.shape[1] != 36:
        raise RuntimeError("Wan I2V transformer input must have 36 channels")
    return result


def _validate_dimensions(*, height: int, width: int, num_frames: int) -> None:
    for name, value in (("height", height), ("width", width), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Wan I2V {name} must be a positive integer")
    try:
        require_dimensions(width, height, WAN_I2V_CANVAS)
    except ValueError as exc:
        raise ValueError(f"Wan I2V {exc}") from exc
    if (num_frames - 1) % 4:
        raise ValueError("Wan I2V num_frames must be 4k+1")
    if num_frames > WAN_I2V_MAX_FRAMES:
        raise ValueError("Wan I2V request exceeds the 121-frame / 1280x704-area safety budget")
