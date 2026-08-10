"""Exact stored-Wan transformer forward boundary for Engine I2V orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class WanI2VForward:
    """Callable injected into the stage-aware denoise coordinator."""

    condition: torch.Tensor
    compute_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.compute_dtype != torch.float16:
            raise ValueError("Wan stored transformer forward currently requires proven F16 compute")
        if (
            not isinstance(self.condition, torch.Tensor)
            or self.condition.ndim != 5
            or self.condition.shape[0] != 1
            or self.condition.shape[1] != 20
            or self.condition.dtype != torch.float32
            or not bool(torch.isfinite(self.condition).all())
        ):
            raise ValueError("Wan I2V forward condition must be finite FP32 [1,20,T,H,W]")

    def __call__(
        self,
        model: Any,
        session: Any,
        latents: torch.Tensor,
        timestep: object,
        prompt_embeds: torch.Tensor,
        cache_identity: str,
    ) -> torch.Tensor:
        if (
            getattr(session, "transformer", None) is not model
            or getattr(session, "active", False) is not True
        ):
            raise RuntimeError(
                "Wan transformer forward requires its active matching residency session"
            )
        if (
            getattr(model, "_latentslate_compute_dtype", None) != self.compute_dtype
            or getattr(session, "onload_device", None) != latents.device
        ):
            raise RuntimeError("Wan transformer compute dtype/device is not bound to this session")
        if cache_identity not in {"cond", "uncond"}:
            raise ValueError("Wan transformer cache identity must be cond or uncond")
        if (
            not isinstance(latents, torch.Tensor)
            or latents.ndim != 5
            or latents.shape[0] != 1
            or latents.shape[1] != 16
            or latents.dtype != torch.float32
            or latents.device != self.condition.device
            or latents.shape[2:] != self.condition.shape[2:]
            or not bool(torch.isfinite(latents).all())
        ):
            raise ValueError("Wan scheduler latents are incompatible with the I2V condition")
        if (
            not isinstance(prompt_embeds, torch.Tensor)
            or prompt_embeds.shape != (1, 512, 4096)
            or prompt_embeds.dtype not in {torch.float16, torch.bfloat16}
            or prompt_embeds.device != latents.device
            or not bool(torch.isfinite(prompt_embeds).all())
        ):
            raise ValueError(
                "Wan prompt conditioning must be finite [1,512,4096] on the latent device"
            )

        timestep_tensor = _timestep_tensor(timestep, device=latents.device)
        model_input = torch.cat((latents, self.condition), dim=1).to(dtype=self.compute_dtype)
        with torch.inference_mode(), _engine_cache_context(model, cache_identity):
            output = model(
                hidden_states=model_input,
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
            or output[0].device != latents.device
            or output[0].dtype != self.compute_dtype
            or not bool(torch.isfinite(output[0]).all())
        ):
            raise RuntimeError("Wan transformer returned an incompatible stored-model prediction")
        return output[0]


def _timestep_tensor(timestep: object, *, device: torch.device) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        if timestep.ndim != 0 or not bool(torch.isfinite(timestep)):
            raise ValueError("Wan transformer timestep must be a finite scalar")
        return timestep.to(device=device)
    if isinstance(timestep, bool) or not isinstance(timestep, (int, float)):
        raise TypeError("Wan transformer timestep must be numeric")
    value = float(timestep)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("Wan transformer timestep must be finite")
    return torch.tensor(value, device=device)


@contextmanager
def _engine_cache_context(model: Any, identity: str) -> Iterator[None]:
    """Always clear the pinned Diffusers hook context after a model call.

    The pinned ``CacheMixin.cache_context`` lacks ``try/finally`` and leaves
    the registry stuck when a forward raises. Engine orchestration owns this
    lifecycle until the upstream context is exception-safe.
    """

    from diffusers.hooks import HookRegistry

    registry = HookRegistry.check_if_exists_or_initialize(model)
    set_context = getattr(registry, "_set_context", None)
    if not callable(set_context):
        raise TypeError("Wan transformer hook registry cannot set cache context")
    set_context(identity)
    try:
        yield
    finally:
        set_context(None)
