"""Engine-owned native Wan 2.2 I2V runtime composition.

This module composes the exact, independently proven stored-artifact adapters.
It never invokes a weight quantizer, Diffusers pipeline orchestration, or an
Accelerate offload hook.
"""

from __future__ import annotations

import asyncio
import gc
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch

from .framework.residency import canonical_device
from .umt5_stored_adapter import (
    UMT5_XXL_CONFIG,
    UMT5EncoderResidencySession,
    UMT5StoredAdapterPlan,
    materialize_umt5_encoder,
    plan_stored_umt5_encoder,
)
from .wan21_vae_adapter import (
    WAN21_VAE_CONFIG,
    WanVaePlan,
    WanVaeResidencySession,
    materialize_wan21_vae,
    plan_stored_wan21_vae,
)
from .wan22_i2v_conditioning import (
    _validate_dimensions,
    prepare_wan_i2v_conditioning,
    preprocess_wan_i2v_image,
)
from .wan22_i2v_forward import WanI2VForward
from .wan22_i2v_orchestration import (
    WAN_FLOW_SHIFT,
    CancellationCheck,
    ProgressCallback,
    StagePolicy,
    WanEulerScheduler,
    coordinate_denoise,
)
from .wan22_i2v_support import (
    WanI2VSupportPlan,
    plan_wan_i2v_support,
    revalidate_wan_i2v_support,
)
from .wan22_prompt import encode_wan_prompt_pair
from .wan22_stored_adapter import (
    WAN22_14B_I2V_CONFIG,
    WanRootResidencyPlan,
    WanStoredAdapterPlan,
    WanTransformerResidencySession,
    materialize_wan_transformer,
    plan_stored_wan_transformer,
    plan_wan_root_residency,
    verify_wan_stored_dispatch,
    wan_stored_dispatch_snapshot,
)
from .wan22_stored_lora import (
    apply_wan_stage_loras,
    verify_wan_lora_dispatch,
    wan_lora_dispatch_snapshot,
)


@dataclass(frozen=True, slots=True)
class WanI2VArtifactPaths:
    support: Path
    transformer_high: Path
    transformer_low: Path
    text_encoder: Path
    vae: Path


@dataclass(frozen=True, slots=True)
class WanI2VRequest:
    image: Any
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 5
    height: int = 64
    width: int = 64
    steps: int = 4
    seed: int = 0
    stage_policy: str = "expert_split"
    high_guidance: float = 1.0
    low_guidance: float = 1.0


@dataclass(frozen=True, slots=True)
class WanI2VRuntimeProvenance:
    support_fingerprint: str
    tokenizer_sha256: str
    transformer_high_header_sha256: str
    transformer_low_header_sha256: str
    text_encoder_header_sha256: str
    vae_header_sha256: str
    transformer_high_contract: str
    transformer_low_contract: str
    text_encoder_contract: str
    stage_policy: str
    steps: int
    seed: int
    sampler: str
    scheduler: str
    shift: float
    transformer_high_path: str
    transformer_low_path: str
    text_encoder_path: str
    vae_path: str
    transformer_high_size_bytes: int
    transformer_low_size_bytes: int
    text_encoder_size_bytes: int
    vae_size_bytes: int
    transformer_high_mtime_ns: int
    transformer_low_mtime_ns: int
    text_encoder_mtime_ns: int
    vae_mtime_ns: int
    configured_loras: tuple[dict[str, object], ...] = ()
    active_loras: tuple[dict[str, object], ...] = ()
    lora_dispatch: Mapping[str, dict[str, int]] | None = None
    transformer_dispatch: Mapping[str, dict[str, object]] | None = None


@dataclass(frozen=True, slots=True)
class WanI2VResult:
    video: torch.Tensor
    provenance: WanI2VRuntimeProvenance


def _report_load_progress(
    callback: ProgressCallback | None, completed: int, stage: str
) -> None:
    """Report fixed, non-sensitive cold-load boundaries below denoising progress."""

    if callback is not None:
        callback(completed, 1000, stage)


@dataclass(slots=True)
class NativeWanI2VRuntime:
    """Materialized native components kept on CPU between generation stages."""

    support: WanI2VSupportPlan
    high_plan: WanStoredAdapterPlan
    low_plan: WanStoredAdapterPlan
    text_plan: UMT5StoredAdapterPlan
    vae_plan: WanVaePlan
    high_model: torch.nn.Module
    low_model: torch.nn.Module
    text_encoder: torch.nn.Module
    vae: torch.nn.Module
    high_residency: WanRootResidencyPlan
    low_residency: WanRootResidencyPlan
    configured_loras: tuple[dict[str, object], ...] = ()
    active_loras: tuple[Any, ...] = ()
    _released: bool = False

    @classmethod
    def load(
        cls,
        paths: WanI2VArtifactPaths,
        *,
        support_plan: WanI2VSupportPlan | None = None,
        adapter_plans: Mapping[str, Any] | None = None,
        configured_loras: tuple[dict[str, object], ...] = (),
        active_loras: tuple[Any, ...] = (),
        load_progress: ProgressCallback | None = None,
    ) -> Self:
        """Materialize an exact native component set on CPU.

        Catalog execution passes the previously validated plans. Materializers then
        bind their opened handles to those identities, closing the revalidate/replan
        race. Direct callers may omit them and receive the same exact planners here.
        """

        runtime = None
        high_model = low_model = text_encoder = vae = None
        try:
            if support_plan is None:
                support = plan_wan_i2v_support(paths.support)
            else:
                support = support_plan
                if support.root != paths.support.resolve(strict=True):
                    raise ValueError("Wan support path does not match its catalog plan")
                if not revalidate_wan_i2v_support(support):
                    raise ValueError("Wan support bundle changed before materialization")

            if adapter_plans is None:
                high_plan = plan_stored_wan_transformer(paths.transformer_high)
                low_plan = plan_stored_wan_transformer(paths.transformer_low)
                text_plan = plan_stored_umt5_encoder(paths.text_encoder)
                vae_plan = plan_stored_wan21_vae(paths.vae)
            else:
                required = {
                    "transformer_high_noise",
                    "transformer_low_noise",
                    "text_encoder",
                    "vae",
                }
                if set(adapter_plans) != required:
                    raise ValueError("Wan catalog adapter plans do not cover every native role")
                high_plan = adapter_plans["transformer_high_noise"]
                low_plan = adapter_plans["transformer_low_noise"]
                text_plan = adapter_plans["text_encoder"]
                vae_plan = adapter_plans["vae"]
                expected_paths = {
                    "transformer_high_noise": paths.transformer_high,
                    "transformer_low_noise": paths.transformer_low,
                    "text_encoder": paths.text_encoder,
                    "vae": paths.vae,
                }
                for role, expected_path in expected_paths.items():
                    if adapter_plans[role].identity.path != expected_path.resolve(strict=True):
                        raise ValueError(f"Wan {role} path does not match its catalog plan")
            for plan in (high_plan, low_plan, text_plan, vae_plan):
                plan.require_available()
            if high_plan.identity.path == low_plan.identity.path:
                raise ValueError("Wan high/low stages require distinct transformer artifacts")
            if high_plan.artifact_contract != low_plan.artifact_contract:
                raise ValueError("Wan high/low transformer storage contracts must match")

            _report_load_progress(load_progress, 4, "Loading high-noise transformer")
            high_model = materialize_wan_transformer(
                high_plan,
                WAN22_14B_I2V_CONFIG,
                compute_dtype=torch.float16,
            )
            _report_load_progress(load_progress, 5, "Loading low-noise transformer")
            low_model = materialize_wan_transformer(
                low_plan,
                WAN22_14B_I2V_CONFIG,
                compute_dtype=torch.float16,
            )
            _report_load_progress(load_progress, 6, "Loading text encoder")
            text_encoder = materialize_umt5_encoder(
                text_plan,
                UMT5_XXL_CONFIG,
                compute_dtype=torch.float16,
            )
            _report_load_progress(load_progress, 7, "Loading VAE")
            vae = materialize_wan21_vae(
                vae_plan,
                WAN21_VAE_CONFIG,
                compute_dtype=torch.bfloat16,
            )
            by_stage = {
                stage: tuple(item for item in active_loras if item.stage == stage)
                for stage in ("high", "low")
            }
            apply_wan_stage_loras(high_model, by_stage["high"])
            apply_wan_stage_loras(low_model, by_stage["low"])
            runtime = cls(
                support=support,
                high_plan=high_plan,
                low_plan=low_plan,
                text_plan=text_plan,
                vae_plan=vae_plan,
                high_model=high_model,
                low_model=low_model,
                text_encoder=text_encoder,
                vae=vae,
                high_residency=plan_wan_root_residency(high_model),
                low_residency=plan_wan_root_residency(low_model),
                configured_loras=tuple(dict(item) for item in configured_loras),
                active_loras=tuple(active_loras),
            )
            runtime._validate_component_binding()
            _report_load_progress(load_progress, 8, "Native Wan ready")
            return runtime
        except BaseException:
            runtime = None
            high_model = low_model = text_encoder = vae = None
            gc.collect()
            raise
    def generate(
        self,
        request: WanI2VRequest,
        *,
        device: torch.device | str,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> WanI2VResult:
        """Generate one CPU-resident RGB video tensor with strict staged residency."""

        if self._released:
            raise RuntimeError("native Wan I2V runtime was released")
        self._validate_component_binding()
        validate_wan_i2v_request(request)
        _raise_if_cancelled(cancelled)
        target = canonical_device(torch.device(device))
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("native Wan I2V supports CPU or CUDA execution")
        processed = preprocess_wan_i2v_image(
            request.image,
            height=request.height,
            width=request.width,
        )
        _raise_if_cancelled(cancelled)
        tokenizer = self.support.load_tokenizer()
        tokens = tokenizer.tokenize_pair(request.prompt, request.negative_prompt)
        with UMT5EncoderResidencySession(
            self.text_encoder,
            onload_device=target,
        ) as text_session:
            conditioning = encode_wan_prompt_pair(text_session, tokens)
        _raise_if_cancelled(cancelled)

        with WanVaeResidencySession(
            self.vae,
            self.vae_plan,
            onload_device=target,
        ) as vae_session:
            latent_state = prepare_wan_i2v_conditioning(
                vae_session,
                processed,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                seed=request.seed,
                device=target,
            )
        _raise_if_cancelled(cancelled)

        scheduler = WanEulerScheduler()
        scheduler.set_timesteps(request.steps, device=target)
        timesteps = tuple(scheduler.timesteps)
        policy = StagePolicy(
            request.stage_policy,
            boundary_ratio=self.support.boundary_ratio,
            num_train_timesteps=scheduler.multiplier,
        )
        forward = WanI2VForward(latent_state.condition)

        def session_factory(model: Any, stage: str):
            if stage == "high" and model is self.high_model:
                plan = self.high_residency
            elif stage == "low" and model is self.low_model:
                plan = self.low_residency
            else:
                raise ValueError("Wan denoise stage/model binding is invalid")
            return WanTransformerResidencySession(model, plan, onload_device=target)

        lora_dispatch_before = {
            "high": wan_lora_dispatch_snapshot(self.high_model),
            "low": wan_lora_dispatch_snapshot(self.low_model),
        }
        transformer_dispatch_before = {
            "high": wan_stored_dispatch_snapshot(self.high_model),
            "low": wan_stored_dispatch_snapshot(self.low_model),
        }
        latents = coordinate_denoise(
            latents=latent_state.noise_latents,
            timesteps=timesteps,
            policy=policy,
            high_model=self.high_model,
            low_model=self.low_model,
            session_factory=session_factory,
            forward=forward,
            scheduler=scheduler,
            conditional=conditioning.prompt_embeds,
            unconditional=conditioning.negative_prompt_embeds,
            high_guidance=request.high_guidance,
            low_guidance=request.low_guidance,
            progress=progress,
            cancelled=cancelled,
        )
        _raise_if_cancelled(cancelled)
        lora_dispatch = {
            stage: verify_wan_lora_dispatch(model, lora_dispatch_before[stage])
            for stage, model in (("high", self.high_model), ("low", self.low_model))
        }
        transformer_dispatch = {
            stage: verify_wan_stored_dispatch(model, transformer_dispatch_before[stage])
            for stage, model in (("high", self.high_model), ("low", self.low_model))
        }
        with WanVaeResidencySession(
            self.vae,
            self.vae_plan,
            onload_device=target,
        ) as vae_session:
            video = vae_session.decode(latents).detach().to(device="cpu")
        _raise_if_cancelled(cancelled)
        _validate_video(video, request)
        return WanI2VResult(
            video=video,
            provenance=self._provenance(
                request,
                lora_dispatch=lora_dispatch,
                transformer_dispatch=transformer_dispatch,
            ),
        )

    def release(self) -> None:
        """Break module-owned tensor references during worker-local cleanup.

        This makes a released direct runtime terminal and removes module registry
        ownership without changing any stored artifact. It is *not* a Windows
        host-memory guarantee: PyTorch/native allocator pages may remain private
        to a live process. The managed 14B recipe therefore uses process exit as
        its real release boundary.
        """

        if self._released:
            return
        for module in (self.high_model, self.low_model, self.text_encoder, self.vae):
            _dematerialize_module(module)
        self._released = True
        gc.collect()

    def _validate_component_binding(
        self,
        *,
        support_revalidator: Any | None = None,
    ) -> None:
        if self._released:
            raise RuntimeError("native Wan I2V runtime was released")
        if support_revalidator is None:
            support_revalidator = revalidate_wan_i2v_support
        if not support_revalidator(self.support):
            raise ValueError("Wan support bundle changed after runtime loading")
        for plan in (self.high_plan, self.low_plan, self.text_plan, self.vae_plan):
            plan.require_available()
        tokenizer_sha = getattr(self.text_encoder, "_latentslate_tokenizer_sha256", None)
        if tokenizer_sha != self.support.tokenizer_sha256:
            raise ValueError("Wan support tokenizer does not match the materialized UMT5 artifact")
        if (
            getattr(self.high_model, "_latentslate_compute_dtype", None) != torch.float16
            or getattr(self.low_model, "_latentslate_compute_dtype", None) != torch.float16
        ):
            raise ValueError("Wan transformers lack the proven F16 materialization binding")
        for label, model, plan in (
            ("high", self.high_model, self.high_plan),
            ("low", self.low_model, self.low_plan),
        ):
            poisoned = getattr(model, "_latentslate_residency_poisoned", None)
            if poisoned:
                raise RuntimeError(f"Wan {label} transformer residency is poisoned: {poisoned}")
            if (
                getattr(model, "_latentslate_wan_config_fingerprint", None)
                != plan.config_fingerprint
                or getattr(model, "_latentslate_wan_mapping_fingerprint", None)
                != plan.mapping_fingerprint
                or getattr(model, "_latentslate_wan_artifact_identity", None) != plan.identity
            ):
                raise ValueError(f"Wan {label} transformer does not match its artifact plan")
        if (
            getattr(self.text_encoder, "_latentslate_umt5_config_fingerprint", None)
            != self.text_plan.config_fingerprint
            or getattr(self.text_encoder, "_latentslate_umt5_mapping_fingerprint", None)
            != self.text_plan.mapping_fingerprint
            or getattr(self.text_encoder, "_latentslate_umt5_artifact_identity", None)
            != self.text_plan.identity
        ):
            raise ValueError("Wan UMT5 encoder does not match its validated artifact plan")
        if (
            getattr(self.vae, "_latentslate_vae_config_fingerprint", None)
            != self.vae_plan.config_fingerprint
            or getattr(self.vae, "_latentslate_vae_mapping_fingerprint", None)
            != self.vae_plan.mapping_fingerprint
            or getattr(self.vae, "_latentslate_vae_artifact_identity", None)
            != self.vae_plan.identity
        ):
            raise ValueError("Wan VAE does not match its validated artifact plan")

    def _provenance(
        self,
        request: WanI2VRequest,
        *,
        lora_dispatch: Mapping[str, dict[str, int]],
        transformer_dispatch: Mapping[str, dict[str, object]],
    ) -> WanI2VRuntimeProvenance:
        return WanI2VRuntimeProvenance(
            support_fingerprint=self.support.fingerprint,
            tokenizer_sha256=self.support.tokenizer_sha256,
            transformer_high_header_sha256=self.high_plan.identity.header_sha256,
            transformer_low_header_sha256=self.low_plan.identity.header_sha256,
            text_encoder_header_sha256=self.text_plan.identity.header_sha256,
            vae_header_sha256=self.vae_plan.identity.header_sha256,
            transformer_high_contract=self.high_plan.artifact_contract,
            transformer_low_contract=self.low_plan.artifact_contract,
            text_encoder_contract=self.text_plan.artifact_contract,
            stage_policy=request.stage_policy,
            steps=request.steps,
            seed=request.seed,
            sampler="euler",
            scheduler="simple",
            shift=WAN_FLOW_SHIFT,
            transformer_high_path=str(self.high_plan.identity.path),
            transformer_low_path=str(self.low_plan.identity.path),
            text_encoder_path=str(self.text_plan.identity.path),
            vae_path=str(self.vae_plan.identity.path),
            transformer_high_size_bytes=self.high_plan.identity.size_bytes,
            transformer_low_size_bytes=self.low_plan.identity.size_bytes,
            text_encoder_size_bytes=self.text_plan.identity.size_bytes,
            vae_size_bytes=self.vae_plan.identity.size_bytes,
            transformer_high_mtime_ns=self.high_plan.identity.mtime_ns,
            transformer_low_mtime_ns=self.low_plan.identity.mtime_ns,
            text_encoder_mtime_ns=self.text_plan.identity.mtime_ns,
            vae_mtime_ns=self.vae_plan.identity.mtime_ns,
            configured_loras=self.configured_loras,
            active_loras=tuple(item.public_dict() for item in self.active_loras),
            lora_dispatch={stage: dict(value) for stage, value in lora_dispatch.items()},
            transformer_dispatch={
                stage: dict(value) for stage, value in transformer_dispatch.items()
            },
        )


def _dematerialize_module(module: torch.nn.Module) -> None:
    """Replace all state tensors with meta tensors to break module ownership.

    This mirrors the failure-cleanup paths in the individual stored-artifact
    materializers. It deliberately assigns through the module registries so
    custom stored tensor subclasses (including Comfy Kitchen quantized tensors)
    cannot retain their backing storage through a normal ``Module.to`` call. It
    does not promise that a live Windows allocator returns its pages to the OS.
    """

    for child in module.modules():
        for name, parameter in tuple(child._parameters.items()):
            if parameter is not None:
                child._parameters[name] = torch.nn.Parameter(
                    torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
                    requires_grad=False,
                )
        for name, buffer in tuple(child._buffers.items()):
            if buffer is not None:
                child._buffers[name] = torch.empty(
                    tuple(buffer.shape), dtype=buffer.dtype, device="meta"
                )


def validate_wan_i2v_request(request: WanI2VRequest) -> None:
    """Validate native I2V inputs without materializing any model component."""
    if not isinstance(request, WanI2VRequest):
        raise TypeError("native Wan generation requires WanI2VRequest")
    if isinstance(request.steps, bool) or not isinstance(request.steps, int):
        raise TypeError("Wan inference steps must be an integer")
    if not 2 <= request.steps <= 1000:
        raise ValueError("Wan inference steps must be between 2 and 1000")
    _validate_dimensions(
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
    )
    if (
        isinstance(request.seed, bool)
        or not isinstance(request.seed, int)
        or not 0 <= request.seed < 2**63
    ):
        raise ValueError("Wan I2V seed must be an integer in [0, 2^63)")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("Wan prompt must be a nonempty string")
    if not isinstance(request.negative_prompt, str):
        raise TypeError("Wan negative prompt must be a string")
    if request.stage_policy not in {"expert_split", "diffusers_boundary"}:
        raise ValueError("Wan stage policy must be expert_split or diffusers_boundary")
    for name, value in (
        ("high", request.high_guidance),
        ("low", request.low_guidance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Wan {name} guidance must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Wan {name} guidance must be finite and nonnegative")


def _raise_if_cancelled(cancelled: CancellationCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise asyncio.CancelledError


def _validate_video(video: torch.Tensor, request: WanI2VRequest) -> None:
    if (
        not isinstance(video, torch.Tensor)
        or video.shape != (1, 3, request.num_frames, request.height, request.width)
        or video.device.type != "cpu"
        or video.dtype not in {torch.float16, torch.bfloat16, torch.float32}
        or not bool(torch.isfinite(video).all())
        or bool((video < -1.0).any())
        or bool((video > 1.0).any())
    ):
        raise RuntimeError("native Wan I2V returned an incompatible CPU video tensor")
