"""Native Wan 2.2 14B first/last-frame runtime composition."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from .umt5_stored_adapter import UMT5EncoderResidencySession
from .wan21_vae_adapter import WanVaeResidencySession
from .wan22_flf_conditioning import (
    prepare_wan_flf_conditioning,
    preprocess_wan_flf_image,
)
from .wan22_i2v_forward import WanI2VForward
from .wan22_i2v_orchestration import ComfyWanEulerScheduler, StagePolicy, coordinate_denoise
from .wan22_i2v_runtime import (
    NativeWanI2VRuntime,
    WanI2VArtifactPaths,
    WanI2VResult,
    _raise_if_cancelled,
    _validate_video,
)
from .wan22_prompt import encode_wan_prompt_pair
from .wan22_stored_adapter import WanTransformerResidencySession, _canonicalize_residency_device
from .wan22_stored_lora import verify_wan_lora_dispatch, wan_lora_dispatch_snapshot

COMFY_WAN_FLF_SHIFT = 8.0
_FLF_OPERATIONS = frozenset({"comfy_i2v_flf_base", "comfy_i2v_flf_lightx2v_4step"})


@dataclass(frozen=True, slots=True)
class WanFLFRequest:
    start_image: Any
    end_image: Any
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81
    height: int = 640
    width: int = 640
    steps: int = 20
    seed: int = 0
    stage_policy: str = "comfy_split"
    high_guidance: float = 4.0
    low_guidance: float = 4.0
    operation: str = "comfy_i2v_flf_base"


class NativeWanFLFRuntime:
    """Reuse the I2V stored components with FLF-only two-endpoint conditioning."""

    def __init__(self, core: NativeWanI2VRuntime) -> None:
        self._core = core

    @classmethod
    def load(cls, paths: WanI2VArtifactPaths, **kwargs: Any) -> NativeWanFLFRuntime:
        return cls(NativeWanI2VRuntime.load(paths, **kwargs))

    def generate(
        self, request: WanFLFRequest, *, device: torch.device | str, progress=None, cancelled=None
    ) -> WanI2VResult:
        validate_wan_flf_request(request)
        self._core._validate_component_binding()
        _raise_if_cancelled(cancelled)
        target = _canonicalize_residency_device(torch.device(device))
        start = preprocess_wan_flf_image(
            request.start_image, height=request.height, width=request.width
        )
        end = preprocess_wan_flf_image(
            request.end_image, height=request.height, width=request.width
        )
        _raise_if_cancelled(cancelled)
        tokenizer = self._core.support.load_tokenizer()
        tokens = tokenizer.tokenize_pair(request.prompt, request.negative_prompt)
        with UMT5EncoderResidencySession(self._core.text_encoder, onload_device=target) as session:
            prompt_pair = encode_wan_prompt_pair(session, tokens)
        _raise_if_cancelled(cancelled)
        with WanVaeResidencySession(
            self._core.vae, self._core.vae_plan, onload_device=target
        ) as session:
            latent_state = prepare_wan_flf_conditioning(
                session,
                start,
                end,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                seed=request.seed,
                device=target,
            )
        _raise_if_cancelled(cancelled)
        shift = float(_flf_operation(request.operation)["shift"])
        scheduler = ComfyWanEulerScheduler(shift=shift)
        scheduler.set_timesteps(request.steps, device=target)

        def session_factory(model: Any, stage: str):
            if stage == "high" and model is self._core.high_model:
                plan = self._core.high_residency
            elif stage == "low" and model is self._core.low_model:
                plan = self._core.low_residency
            else:
                raise ValueError("Wan FLF denoise stage/model binding is invalid")
            return WanTransformerResidencySession(model, plan, onload_device=target)

        before = {
            stage: wan_lora_dispatch_snapshot(model)
            for stage, model in (("high", self._core.high_model), ("low", self._core.low_model))
        }
        latents = coordinate_denoise(
            latents=latent_state.noise_latents,
            timesteps=tuple(scheduler.timesteps),
            policy=StagePolicy(
                request.stage_policy,
                boundary_ratio=self._core.support.boundary_ratio,
                num_train_timesteps=scheduler.multiplier,
            ),
            high_model=self._core.high_model,
            low_model=self._core.low_model,
            session_factory=session_factory,
            forward=WanI2VForward(latent_state.condition),
            scheduler=scheduler,
            conditional=prompt_pair.prompt_embeds,
            unconditional=prompt_pair.negative_prompt_embeds,
            high_guidance=request.high_guidance,
            low_guidance=request.low_guidance,
            progress=progress,
            cancelled=cancelled,
        )
        _raise_if_cancelled(cancelled)
        dispatch = {
            stage: verify_wan_lora_dispatch(model, before[stage])
            for stage, model in (("high", self._core.high_model), ("low", self._core.low_model))
        }
        with WanVaeResidencySession(
            self._core.vae, self._core.vae_plan, onload_device=target
        ) as session:
            video = session.decode(latents).detach().to(device="cpu")
        _raise_if_cancelled(cancelled)
        _validate_video(video, request)
        return WanI2VResult(
            video=video,
            provenance=replace(
                self._core._provenance(request, lora_dispatch=dispatch), shift=shift
            ),
        )

    def release(self) -> None:
        self._core.release()


def validate_wan_flf_request(request: WanFLFRequest) -> None:
    if not isinstance(request, WanFLFRequest):
        raise TypeError("native Wan FLF generation requires WanFLFRequest")
    expected = _flf_operation(request.operation)
    # The common validator owns bounded 4k+1 geometry and seed rules without
    # consulting the unused image field.
    from .wan22_i2v_conditioning import _validate_dimensions

    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("Wan FLF prompt must be non-empty")
    if not isinstance(request.negative_prompt, str):
        raise TypeError("Wan FLF negative_prompt must be text")
    if isinstance(request.steps, bool) or not isinstance(request.steps, int):
        raise TypeError("Wan FLF inference steps must be an integer")
    if not 2 <= request.steps <= 1000:
        raise ValueError("Wan FLF inference steps must be between 2 and 1000")
    if request.stage_policy != "comfy_split":
        raise ValueError("Wan FLF request must use comfy_split")
    for name, value in (("high", request.high_guidance), ("low", request.low_guidance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Wan FLF {name} guidance must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Wan FLF {name} guidance must be finite and nonnegative")
    _validate_dimensions(height=request.height, width=request.width, num_frames=request.num_frames)
    if (
        isinstance(request.seed, bool)
        or not isinstance(request.seed, int)
        or not 0 <= request.seed < 2**63
    ):
        raise ValueError("Wan FLF seed must be an integer in [0, 2^63)")
    for key in ("steps", "stage_policy", "high_guidance", "low_guidance"):
        if getattr(request, key) != expected[key]:
            raise ValueError(f"Wan FLF {request.operation} requires {key}={expected[key]!r}")


def _flf_operation(operation: str) -> dict[str, str | int | float]:
    if operation not in _FLF_OPERATIONS:
        raise ValueError("native Wan FLF operation is invalid")
    from ..wan22_recipe import wan22_i2v_operation

    return dict(wan22_i2v_operation(operation))
