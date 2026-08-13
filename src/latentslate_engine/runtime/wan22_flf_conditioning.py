"""Exact Comfy Wan 2.2 first/last-frame latent conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .wan22_i2v_conditioning import WanVaeSessionLike, _validate_dimensions
from .wan22_stored_adapter import _canonicalize_residency_device


@dataclass(frozen=True, slots=True)
class WanFLFConditioning:
    """Noise and immutable 20-channel condition for the Comfy FLF graph."""

    noise_latents: torch.Tensor
    condition: torch.Tensor
    num_frames: int
    height: int
    width: int
    seed: int


def preprocess_wan_flf_image(image: Any, *, height: int, width: int) -> torch.Tensor:
    """Apply the active Comfy FLF node's center-crop/bilinear endpoint resize.

    ``WanFirstLastFrameToVideo`` calls ``common_upscale(..., "bilinear",
    "center")`` on raw Comfy [0,1] RGB image tensors. The stored Diffusers VAE
    boundary receives the equivalent [-1,1] representation, so conversion happens
    only after the exact crop/interpolation operation.
    """

    _validate_dimensions(height=height, width=width, num_frames=1)
    samples = _comfy_rgb_tensor(image)
    old_height, old_width = samples.shape[-2:]
    old_aspect, new_aspect = old_width / old_height, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    resized = torch.nn.functional.interpolate(cropped, size=(height, width), mode="bilinear")
    result = resized.mul(2.0).sub(1.0).to(dtype=torch.float32, device="cpu")
    if result.shape != (1, 3, height, width) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("Wan FLF Comfy endpoint preprocessing produced an invalid image")
    return result


def _comfy_rgb_tensor(image: Any) -> torch.Tensor:
    """Normalize one PIL/CHW/HWC RGB image to Comfy's CPU [0,1] BCHW form."""

    if isinstance(image, torch.Tensor):
        value = image.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim == 3 and value.shape[0] == 3:
            value = value.unsqueeze(0)
        elif value.ndim == 3 and value.shape[-1] >= 3:
            value = value[..., :3].movedim(-1, 0).unsqueeze(0)
        elif value.ndim == 4 and value.shape[0] == 1 and value.shape[1] == 3:
            pass
        elif value.ndim == 4 and value.shape[0] == 1 and value.shape[-1] >= 3:
            value = value[..., :3].movedim(-1, 1)
        else:
            raise ValueError("Wan FLF endpoint image must be one RGB image")
    else:
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("Wan FLF endpoint image must be PIL RGB or a CPU tensor")
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        value = torch.from_numpy(array).movedim(-1, 0).unsqueeze(0)
    if (
        value.shape[0] != 1
        or value.shape[1] != 3
        or value.shape[2] <= 0
        or value.shape[3] <= 0
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
        or bool((value > 1.0).any())
    ):
        raise ValueError("Wan FLF endpoint image must be finite RGB in [0,1]")
    return value


def prepare_wan_flf_conditioning(
    vae_session: WanVaeSessionLike,
    start_image: torch.Tensor,
    end_image: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device | str,
) -> WanFLFConditioning:
    """Mirror ``WanFirstLastFrameToVideo`` without optional CLIP-vision inputs.

    The current official template supplies exactly one start and one end image. Its
    node creates a mid-grey canvas (0.5 in Comfy's [0,1] image space; zero in the
    normalized Engine/Diffusers boundary), places the endpoint frames, VAE-encodes
    the composite once, and provides a 4-channel temporal concat mask.
    """

    _validate_dimensions(height=height, width=width, num_frames=num_frames)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Wan FLF seed must be an integer in [0, 2^63)")
    target = _canonicalize_residency_device(torch.device(device))
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Wan FLF conditioning supports CPU or CUDA devices")
    for label, image in (("start", start_image), ("end", end_image)):
        if (
            not isinstance(image, torch.Tensor)
            or image.shape != (1, 3, height, width)
            or image.dtype != torch.float32
            or image.device.type != "cpu"
            or not bool(torch.isfinite(image).all())
            or bool((image < -1.0).any())
            or bool((image > 1.0).any())
        ):
            raise ValueError(
                f"Wan FLF {label} image must be a CPU float32 [1,3,H,W] tensor in [-1,1]"
            )

    latent_frames = ((num_frames - 1) // 4) + 1
    latent_height, latent_width = height // 8, width // 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, 16, latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(target)

    # Normalized zero is Comfy's 0.5 canvas after its image preprocessing.
    video = torch.zeros((1, 3, num_frames, height, width), dtype=torch.bfloat16, device=target)
    video[:, :, 0] = start_image.to(device=target, dtype=torch.bfloat16)
    video[:, :, -1] = end_image.to(device=target, dtype=torch.bfloat16)
    latent_image = vae_session.encode(video)
    expected = (1, 16, latent_frames, latent_height, latent_width)
    if (
        not isinstance(latent_image, torch.Tensor)
        or latent_image.shape != expected
        or latent_image.device != target
        or latent_image.dtype not in {torch.float16, torch.bfloat16}
        or not bool(torch.isfinite(latent_image).all())
    ):
        raise RuntimeError(
            "Wan FLF VAE condition is incompatible with the 16-channel latent contract"
        )

    # Source: Comfy ``WanFirstLastFrameToVideo``: zero first start_len + 3
    # frame positions and the final end_len positions before temporal packing.
    # ``WAN21.concat_cond`` then inverts that user-facing ``concat_mask`` before
    # concatenating it to the noise. This direct stored-model boundary bypasses
    # that wrapper, so it must preserve the *post-inversion* model mask here.
    comfy_mask_frames = torch.ones(
        (1, 1, latent_frames * 4, latent_height, latent_width),
        dtype=torch.float32,
        device=target,
    )
    comfy_mask_frames[:, :, :4] = 0.0  # one start frame + three causal look-ahead positions
    comfy_mask_frames[:, :, -1:] = 0.0  # one final endpoint frame
    mask = 1.0 - comfy_mask_frames.view(1, latent_frames, 4, latent_height, latent_width).transpose(
        1, 2
    )
    condition = torch.cat((mask, latent_image.to(dtype=torch.float32)), dim=1)
    if condition.shape != (1, 20, latent_frames, latent_height, latent_width):
        raise RuntimeError("Wan FLF condition did not produce the required 20 channels")
    return WanFLFConditioning(noise, condition, num_frames, height, width, seed)
