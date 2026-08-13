"""Exact 16-channel stored-Wan forward boundary for T2V orchestration."""

from __future__ import annotations

from typing import Any

import torch

from .wan22_i2v_forward import _engine_cache_context, _timestep_tensor


class WanT2VForward:
    """Run an active 16-channel T2V expert without I2V conditioning."""

    compute_dtype = torch.float16

    def __call__(
        self,
        model: Any,
        session: Any,
        latents: torch.Tensor,
        timestep: object,
        prompt_embeds: torch.Tensor,
        cache_identity: str,
    ) -> torch.Tensor:
        if getattr(session, "transformer", None) is not model or not getattr(session, "active", False):
            raise RuntimeError("Wan T2V forward requires its active matching residency session")
        if (
            not isinstance(latents, torch.Tensor)
            or latents.ndim != 5
            or latents.shape[0:2] != (1, 16)
            or latents.dtype != torch.float32
            or latents.device != getattr(session, "onload_device", None)
            or not bool(torch.isfinite(latents).all())
        ):
            raise ValueError("Wan T2V scheduler latents must be finite FP32 [1,16,T,H,W]")
        if (
            not isinstance(prompt_embeds, torch.Tensor)
            or prompt_embeds.shape != (1, 512, 4096)
            or prompt_embeds.dtype not in {torch.float16, torch.bfloat16}
            or prompt_embeds.device != latents.device
            or not bool(torch.isfinite(prompt_embeds).all())
        ):
            raise ValueError("Wan T2V prompt conditioning is incompatible with the latent device")
        timestep_tensor = _timestep_tensor(timestep, device=latents.device)
        with torch.inference_mode(), _engine_cache_context(model, cache_identity):
            output = model(
                hidden_states=latents.to(dtype=self.compute_dtype),
                timestep=timestep_tensor.expand(latents.shape[0]),
                encoder_hidden_states=prompt_embeds.to(dtype=self.compute_dtype),
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
            )
        if (
            not isinstance(output, (tuple, list))
            or len(output) != 1
            or not isinstance(output[0], torch.Tensor)
            or output[0].shape != latents.shape
            or output[0].dtype != self.compute_dtype
            or output[0].device != latents.device
            or not bool(torch.isfinite(output[0]).all())
        ):
            raise RuntimeError("Wan T2V transformer returned an incompatible prediction")
        return output[0]
