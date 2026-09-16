"""LTX 2.5 diffusion decoder loading and reference tile blending.

Tile placement and feathering narrowly follow ComfyUI 1a14b82e
``comfy.utils.tiled_scale_multidim`` and ``comfy.sd.VAE.decode_tiled_3d``.
"""

from __future__ import annotations

import itertools
import json

import torch

from latentslate_engine.ltx23.checkpoint import Ltx23Checkpoint

from .diffusion_decoder import NADiffusionDecoder


class Ltx25VideoDecoder:
    """Own the decoder weights while returning display-space CPU video."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        checkpoint = Ltx23Checkpoint(checkpoint_path)
        config = json.loads(checkpoint.metadata.get("config", "{}")).get("vae", {})
        decoder_config = config.get("decoder", {})
        parameters = {
            key: decoder_config[key]
            for key in (
                "in_channels",
                "out_channels",
                "patch_size",
                "head_dim",
                "stage_channels",
                "stage_depths",
                "stage_kernels",
                "upsamples",
                "stage5_kernel",
                "t_emb_dim",
                "default_num_inference_steps",
                "timestep_scale_multiplier",
            )
            if key in decoder_config
        }
        parameters["model_output_type"] = config.get("model_output_type", "x0")
        with torch.device("meta"):
            self.model = NADiffusionDecoder(**parameters)
        # The shipped file includes an unused training type embedding; Comfy's
        # inference decoder has no corresponding operation or parameter.
        state = {
            name.removeprefix("decoder."): checkpoint.tensor(name)
            for name in checkpoint.tensor_names
            if name.startswith("decoder.") and name != "decoder.type_emb"
        }
        self.model.load_state_dict(state, assign=True, strict=True)
        steps = parameters.get("default_num_inference_steps", 1)
        self.model.default_inference_timesteps = torch.linspace(1.0, 1.0 / steps, steps)
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self._mean = (
            checkpoint.tensor("per_channel_statistics.mean-of-means")
            .to(device=device, dtype=torch.bfloat16)
            .view(1, 128, 1, 1, 1)
        )
        self._std = (
            checkpoint.tensor("per_channel_statistics.std-of-means")
            .to(device=device, dtype=torch.bfloat16)
            .view(1, 128, 1, 1, 1)
        )

    def _decode_tile(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.to(device=self._mean.device, dtype=torch.bfloat16)
        generator = torch.Generator(device=x.device).manual_seed(0)
        return self.model(x * self._std + self._mean, generator=generator).float().cpu()

    @torch.inference_mode()
    def decode(
        self,
        latents: torch.Tensor,
        *,
        tile: tuple[int, int, int] = (8, 16, 16),
        overlap: tuple[int, int, int] = (2, 2, 2),
    ) -> torch.Tensor:
        """Decode BCHTW latents to BCHTW RGB; tile sizes are latent units."""
        if (
            latents.ndim != 5
            or tuple(latents.shape[:2]) != (1, 128)
            or min(latents.shape[2:]) < 1
        ):
            raise ValueError("LTX video decode requires one valid 128-channel latent")
        if any(t <= o or o < 0 for t, o in zip(tile, overlap, strict=True)):
            raise ValueError(
                "decode tiles must be larger than their nonnegative overlaps"
            )
        shape = latents.shape[2:]
        if all(size <= limit for size, limit in zip(shape, tile, strict=True)):
            output = self._decode_tile(latents)
        else:
            output = torch.zeros(1, 3, shape[0] * 8 - 7, shape[1] * 32, shape[2] * 32)
            divisor = torch.zeros(1, 1, *output.shape[2:])
            positions = [
                range(0, size - edge, limit - edge) if size > limit else [0]
                for size, limit, edge in zip(shape, tile, overlap, strict=True)
            ]
            scale = (8, 32, 32)
            feathers = (max(0, overlap[0] * 8 - 7), overlap[1] * 32, overlap[2] * 32)
            for position in itertools.product(*positions):
                sample = latents
                starts = []
                for axis, pos in enumerate(position):
                    pos = max(0, min(shape[axis] - overlap[axis], pos))
                    sample = sample.narrow(
                        axis + 2, pos, min(tile[axis], shape[axis] - pos)
                    )
                    starts.append(pos * scale[axis])
                decoded = self._decode_tile(sample)
                mask = torch.ones(1, 1, *decoded.shape[2:])
                for axis, feather in enumerate(feathers, 2):
                    if feather >= mask.shape[axis]:
                        continue
                    for offset in range(feather):
                        weight = (offset + 1) / feather
                        mask.narrow(axis, offset, 1).mul_(weight)
                        mask.narrow(axis, mask.shape[axis] - 1 - offset, 1).mul_(weight)
                target, target_divisor = output, divisor
                for axis, start in enumerate(starts, 2):
                    length = min(decoded.shape[axis], target.shape[axis] - start)
                    target = target.narrow(axis, start, length)
                    target_divisor = target_divisor.narrow(axis, start, length)
                    decoded = decoded.narrow(axis, 0, length)
                    mask = mask.narrow(axis, 0, length)
                target.add_(decoded * mask)
                target_divisor.add_(mask)
            output.div_(divisor)
        return output.add_(1.0).div_(2.0).clamp_(0.0, 1.0)

    def close(self) -> None:
        self.model = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()
