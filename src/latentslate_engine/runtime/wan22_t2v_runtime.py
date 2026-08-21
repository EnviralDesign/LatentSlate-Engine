"""Engine-owned native Wan 2.2 T2V runtime composition."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import torch

from .framework.residency import canonical_device
from .umt5_stored_adapter import (
    UMT5_XXL_CONFIG,
    UMT5EncoderResidencySession,
    materialize_umt5_encoder,
    plan_stored_umt5_encoder,
)
from .wan21_vae_adapter import (
    WAN21_VAE_CONFIG,
    WanVaeResidencySession,
    materialize_wan21_vae,
    plan_stored_wan21_vae,
)
from .wan22_i2v_orchestration import (
    CancellationCheck,
    ProgressCallback,
    StagePolicy,
    WanEulerScheduler,
    coordinate_denoise,
)
from .wan22_i2v_runtime import (
    NativeWanI2VRuntime,
    WanI2VArtifactPaths,
    WanI2VResult,
    _raise_if_cancelled,
    _validate_video,
)
from .wan22_prompt import encode_wan_prompt_pair
from .wan22_stored_adapter import (
    WAN22_14B_T2V_CONFIG,
    WanTransformerResidencySession,
    materialize_wan_transformer,
    plan_stored_wan_transformer,
    plan_wan_root_residency,
    verify_wan_stored_dispatch,
    wan_stored_dispatch_snapshot,
)
from .wan22_t2v_conditioning import prepare_wan_t2v_latents
from .wan22_t2v_forward import WanT2VForward
from .wan22_t2v_support import WanT2VSupportPlan, plan_wan_t2v_support, revalidate_wan_t2v_support


@dataclass(frozen=True, slots=True)
class WanT2VRequest:
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81
    height: int = 640
    width: int = 640
    steps: int = 20
    seed: int = 0
    stage_policy: str = "expert_split"
    high_guidance: float = 3.5
    low_guidance: float = 3.5


def _report_load_progress(
    callback: ProgressCallback | None, completed: int, stage: str
) -> None:
    """Report fixed, non-sensitive cold-load boundaries below denoising progress."""

    if callback is not None:
        callback(completed, 1000, stage)


class NativeWanT2VRuntime:
    """T2V composition uses its own 16-channel topology, never I2V conditioning."""

    def __init__(self, core: NativeWanI2VRuntime, support: WanT2VSupportPlan) -> None:
        self._core = core
        self.support = support

    @classmethod
    def load(
        cls,
        paths: WanI2VArtifactPaths,
        *,
        support_plan: WanT2VSupportPlan | None = None,
        adapter_plans: Mapping[str, Any] | None = None,
        configured_loras: tuple[dict[str, object], ...] = (),
        active_loras: tuple[Any, ...] = (),
        load_progress: ProgressCallback | None = None,
    ) -> Self:
        support = support_plan or plan_wan_t2v_support(paths.support)
        if support.root != paths.support.resolve(strict=True) or not revalidate_wan_t2v_support(
            support
        ):
            raise ValueError("Wan T2V support path changed before materialization")
        plans = adapter_plans or {
            "transformer_high_noise": plan_stored_wan_transformer(
                paths.transformer_high, WAN22_14B_T2V_CONFIG
            ),
            "transformer_low_noise": plan_stored_wan_transformer(
                paths.transformer_low, WAN22_14B_T2V_CONFIG
            ),
            "text_encoder": plan_stored_umt5_encoder(paths.text_encoder),
            "vae": plan_stored_wan21_vae(paths.vae),
        }
        if set(plans) != {"transformer_high_noise", "transformer_low_noise", "text_encoder", "vae"}:
            raise ValueError("Wan T2V adapter plans do not cover every native role")
        expected_paths = {
            "transformer_high_noise": paths.transformer_high,
            "transformer_low_noise": paths.transformer_low,
            "text_encoder": paths.text_encoder,
            "vae": paths.vae,
        }
        for role, plan in plans.items():
            plan.require_available()
            if plan.identity.path != expected_paths[role].resolve(strict=True):
                raise ValueError(f"Wan T2V {role} path does not match its catalog plan")
        _report_load_progress(load_progress, 4, "Loading high-noise transformer")
        high = materialize_wan_transformer(
            plans["transformer_high_noise"], WAN22_14B_T2V_CONFIG, compute_dtype=torch.float16
        )
        _report_load_progress(load_progress, 5, "Loading low-noise transformer")
        low = materialize_wan_transformer(
            plans["transformer_low_noise"], WAN22_14B_T2V_CONFIG, compute_dtype=torch.float16
        )
        _report_load_progress(load_progress, 6, "Loading text encoder")
        text = materialize_umt5_encoder(
            plans["text_encoder"], UMT5_XXL_CONFIG, compute_dtype=torch.float16
        )
        _report_load_progress(load_progress, 7, "Loading VAE")
        vae = materialize_wan21_vae(plans["vae"], WAN21_VAE_CONFIG, compute_dtype=torch.bfloat16)
        from .wan22_stored_lora import apply_wan_stage_loras

        by_stage = {
            stage: tuple(item for item in active_loras if item.stage == stage)
            for stage in ("high", "low")
        }
        apply_wan_stage_loras(high, by_stage["high"])
        apply_wan_stage_loras(low, by_stage["low"])
        core = NativeWanI2VRuntime(
            support=support,
            high_plan=plans["transformer_high_noise"],
            low_plan=plans["transformer_low_noise"],
            text_plan=plans["text_encoder"],
            vae_plan=plans["vae"],
            high_model=high,
            low_model=low,
            text_encoder=text,
            vae=vae,
            high_residency=plan_wan_root_residency(high),
            low_residency=plan_wan_root_residency(low),
            configured_loras=tuple(dict(item) for item in configured_loras),
            active_loras=tuple(active_loras),
        )
        core._validate_component_binding(support_revalidator=revalidate_wan_t2v_support)
        _report_load_progress(load_progress, 8, "Native Wan ready")
        return cls(core, support)
    def generate(
        self,
        request: WanT2VRequest,
        *,
        device: torch.device | str,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> WanI2VResult:
        validate_wan_t2v_request(request)
        self._core._validate_component_binding(support_revalidator=revalidate_wan_t2v_support)
        _raise_if_cancelled(cancelled)
        target = canonical_device(torch.device(device))
        latents = prepare_wan_t2v_latents(
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            seed=request.seed,
            device=target,
        )
        _raise_if_cancelled(cancelled)
        tokenizer = self.support.load_tokenizer()
        tokens = tokenizer.tokenize_pair(request.prompt, request.negative_prompt)
        with UMT5EncoderResidencySession(self._core.text_encoder, onload_device=target) as session:
            conditioning = encode_wan_prompt_pair(session, tokens)
        _raise_if_cancelled(cancelled)
        scheduler = WanEulerScheduler()
        scheduler.set_timesteps(request.steps, device=target)

        def session_factory(model: Any, stage: str):
            plan = (
                self._core.high_residency
                if stage == "high" and model is self._core.high_model
                else self._core.low_residency
                if stage == "low" and model is self._core.low_model
                else None
            )
            if plan is None:
                raise ValueError("Wan T2V denoise stage/model binding is invalid")
            return WanTransformerResidencySession(model, plan, onload_device=target)

        from .wan22_stored_lora import verify_wan_lora_dispatch, wan_lora_dispatch_snapshot

        before = {
            stage: wan_lora_dispatch_snapshot(model)
            for stage, model in (("high", self._core.high_model), ("low", self._core.low_model))
        }
        transformer_before = {
            stage: wan_stored_dispatch_snapshot(model)
            for stage, model in (("high", self._core.high_model), ("low", self._core.low_model))
        }
        result_latents = coordinate_denoise(
            latents=latents.noise_latents,
            timesteps=tuple(scheduler.timesteps),
            policy=StagePolicy(
                request.stage_policy,
                boundary_ratio=self.support.boundary_ratio,
                num_train_timesteps=scheduler.multiplier,
            ),
            high_model=self._core.high_model,
            low_model=self._core.low_model,
            session_factory=session_factory,
            forward=WanT2VForward(),
            scheduler=scheduler,
            conditional=conditioning.prompt_embeds,
            unconditional=conditioning.negative_prompt_embeds,
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
        transformer_dispatch = {
            stage: verify_wan_stored_dispatch(model, transformer_before[stage])
            for stage, model in (("high", self._core.high_model), ("low", self._core.low_model))
        }
        with WanVaeResidencySession(
            self._core.vae, self._core.vae_plan, onload_device=target
        ) as session:
            video = session.decode(result_latents).detach().to(device="cpu")
        _raise_if_cancelled(cancelled)
        _validate_video(video, request)
        provenance = self._core._provenance(
            request,
            lora_dispatch=dispatch,
            transformer_dispatch=transformer_dispatch,
        )
        return WanI2VResult(video=video, provenance=provenance)

    def release(self) -> None:
        self._core.release()


def validate_wan_t2v_request(request: WanT2VRequest) -> None:
    if not isinstance(request, WanT2VRequest):
        raise TypeError("native Wan T2V generation requires WanT2VRequest")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("Wan T2V prompt must be non-empty")
    if not isinstance(request.negative_prompt, str):
        raise TypeError("Wan T2V negative_prompt must be text")
    if isinstance(request.steps, bool) or not isinstance(request.steps, int):
        raise TypeError("Wan T2V inference steps must be an integer")
    if not 2 <= request.steps <= 1000:
        raise ValueError("Wan T2V inference steps must be between 2 and 1000")
    if request.stage_policy != "expert_split":
        raise ValueError("Wan T2V request must use expert_split")
    for name, value in (("high", request.high_guidance), ("low", request.low_guidance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Wan T2V {name} guidance must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Wan T2V {name} guidance must be finite and nonnegative")
    # Reuse the bounded geometry/seed validator, then discard the small CPU tensor.
    prepare_wan_t2v_latents(
        num_frames=request.num_frames,
        height=request.height,
        width=request.width,
        seed=request.seed,
        device="cpu",
    )
