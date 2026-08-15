"""CPU-safe lifecycle guard for the bounded Z-Image Turbo native path.

It intentionally does not implement a generic diffusion fallback.  This module
owns the pieces that can be proven without a GPU: immutable request identity,
ordered residency intent, cancellation checkpoints, and the requirement that
the stored ConvRot layers retain their exact source layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Event

import torch

from ..z_image_turbo_recipe import (
    ZImageTurboRuntimeRequest,
    revalidate_z_image_turbo_runtime_request,
)
from .z_image_conditioning import encode_z_image_prompt
from .z_image_mixed_qwen import ZImageMixedQwenStage
from .z_image_sampler import ZImageSamplerStep, ZImageSamplingCancelled, z_image_res_multistep
from .z_image_stored_adapter import ZImageNextDiTStage
from .z_image_vae import (
    ZImageDecodeCancelled,
    ZImagePngArtifact,
    ZImagePngPublicationCancelled,
    decode_z_image_flux_ae,
    write_z_image_png_atomic,
)


def z_image_initial_noise(
    seed: int,
    *,
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Create Comfy-compatible CPU F32 noise before one explicit device transfer."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, 16, height // 8, width // 8),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return noise.to(device=torch.device(device), dtype=torch.float32)


class ZImagePhase(StrEnum):
    PLANNING = "planning"
    TEXT_ENCODER = "text_encoder"
    TRANSFORMER = "transformer"
    VAE = "vae"
    COMPLETE = "complete"
    EJECTED = "ejected"


class ZImageTurboCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ZImageTurboCoreResult:
    artifact: ZImagePngArtifact
    seed: int
    qwen_dispatch: dict[str, int | str | bool]
    transformer_dispatch: dict[str, int | str | bool]
    phases: tuple[str, ...]


@dataclass(slots=True)
class ZImageTurboLifecycle:
    """One-job lifecycle.  Cancellation always ejects the planned runtime."""

    request: ZImageTurboRuntimeRequest
    phase: ZImagePhase = ZImagePhase.PLANNING
    events: list[ZImagePhase] = field(default_factory=lambda: [ZImagePhase.PLANNING])
    ejected: bool = False

    def checkpoint(self, phase: ZImagePhase, cancelled: Callable[[], bool] | Event) -> None:
        if self.ejected:
            raise RuntimeError("Z-Image runtime was ejected")
        expected = {
            ZImagePhase.PLANNING: ZImagePhase.TEXT_ENCODER,
            ZImagePhase.TEXT_ENCODER: ZImagePhase.TRANSFORMER,
            ZImagePhase.TRANSFORMER: ZImagePhase.VAE,
        }.get(self.phase)
        if phase is not expected:
            self.eject()
            raise ValueError(
                f"Z-Image lifecycle phase {phase.value!r} is invalid after {self.phase.value!r}"
            )
        is_cancelled = cancelled.is_set if isinstance(cancelled, Event) else cancelled
        if is_cancelled():
            self.eject()
            raise ZImageTurboCancelled(f"Z-Image generation canceled during {phase.value}")
        if not revalidate_z_image_turbo_runtime_request(self.request):
            self.eject()
            raise ValueError("Z-Image runtime request changed or lost native dispatch proof")
        self.phase = phase
        self.events.append(phase)

    def require_stored_transformer_plan(self) -> int:
        plan = self.request.plans["transformer"]
        # The recipe module deliberately owns the concrete type so this remains
        # a narrow runtime seam rather than another ad-hoc loader.
        require_stored_layout = getattr(plan, "require_stored_layout", None)
        if not callable(require_stored_layout):
            self.eject()
            raise TypeError("Z-Image transformer plan has no stored-layout contract")
        require_stored_layout()
        return int(getattr(plan, "stored_layer_count", 0))

    def complete(self) -> None:
        if self.ejected:
            raise RuntimeError("Z-Image runtime was ejected")
        if self.phase is not ZImagePhase.VAE:
            self.eject()
            raise ValueError("Z-Image lifecycle cannot complete before VAE decode")
        self.phase = ZImagePhase.COMPLETE
        self.events.append(ZImagePhase.COMPLETE)

    def eject(self) -> None:
        self.ejected = True
        self.phase = ZImagePhase.EJECTED
        if not self.events or self.events[-1] != ZImagePhase.EJECTED:
            self.events.append(ZImagePhase.EJECTED)

    def public_provenance(self) -> dict[str, object]:
        return {
            "runtime": "ZImageTurboNative",
            "request_fingerprint": self.request.fingerprint,
            "components": self.request.public_component_manifest(),
            "pipeline_warm": False,
            "execution_cache": {"supported": False, "hit": False, "mode": "fresh_contract"},
            "staging_order": ["text_encoder", "transformer", "vae"],
            "phases": [phase.value for phase in self.events],
            "ejected": self.ejected,
            "stored_transformer_layers_planned": self.require_stored_transformer_plan()
            if not self.ejected
            else 0,
            "native_transformer_dispatch": {
                "proven": False,
                "count": 0,
                "reason": "GPU execution has not been accepted",
            },
        }


class ZImageTurboCore:
    """In-process staged core; worker/session ownership is intentionally separate."""

    def __init__(
        self,
        request: ZImageTurboRuntimeRequest,
        *,
        tokenizer: object,
        text_encoder: torch.nn.Module,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        execution_device: torch.device | str,
    ) -> None:
        self.request = request
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae = vae
        self.execution_device = torch.device(execution_device)
        if self.execution_device.type != "cuda":
            raise ValueError("Z-Image stored runtime requires a CUDA execution device")
        self._require_cpu_master()

    def _require_cpu_master(self) -> None:
        for name, model in (
            ("text_encoder", self.text_encoder),
            ("transformer", self.transformer),
            ("vae", self.vae),
        ):
            parameters = tuple(model.parameters())
            if not parameters or any(
                value.is_meta or value.device.type != "cpu" for value in parameters
            ):
                raise ValueError(f"Z-Image {name} must be fully materialized in CPU-master storage")

    def residency_contract(self) -> dict[str, object]:
        self._require_cpu_master()
        return {
            "cpu_master_resident": True,
            "execution_device": str(self.execution_device),
            "staging_order": ["text_encoder", "transformer", "vae"],
            "simultaneous_gpu_weight_stages": 1,
            "qwen_per_operation_residency": True,
            "qwen_full_module_cuda_onload": False,
            "retained_between_runs": True,
            "dense_or_dequant_fallback": False,
        }

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[float, str], None] = lambda _value, _message: None,
        failure_stage: Callable[[str], None] = lambda _stage: None,
    ) -> ZImageTurboCoreResult:
        if width != 1024 or height != 1024:
            raise ValueError("Z-Image Turbo saved core currently requires 1024x1024")
        lifecycle = ZImageTurboLifecycle(self.request)
        text_stage = ZImageMixedQwenStage(
            self.text_encoder,
            self.execution_device,
            cancelled,
            failure_stage,
        )
        transformer_stage = ZImageNextDiTStage(self.transformer, self.execution_device)
        qwen_proof: dict[str, int | str | bool] = {}
        transformer_proof: dict[str, int | str | bool] = {}
        try:
            failure_stage("conditioning")
            lifecycle.checkpoint(ZImagePhase.TEXT_ENCODER, cancelled)
            progress(0.02, "Encoding Z-Image prompt")
            text_stage.onload()
            try:
                encoded = encode_z_image_prompt(
                    self.text_encoder,
                    self.tokenizer,
                    prompt,
                    device=self.execution_device,
                    cancelled=cancelled,
                    diagnostic=failure_stage,
                )
                qwen_proof = text_stage.verify_dispatch()
            finally:
                text_stage.offload()
            if cancelled():
                raise ZImageTurboCancelled("Z-Image generation canceled after text encoding")

            lifecycle.checkpoint(ZImagePhase.TRANSFORMER, cancelled)
            progress(0.12, "Sampling Z-Image latents")
            failure_stage("transformer_onload")
            transformer_stage.onload()
            try:
                failure_stage("noise")
                latents = z_image_initial_noise(
                    seed,
                    height=height,
                    width=width,
                    device=self.execution_device,
                )
                caption = encoded.positive[encoded.attention_mask.to(dtype=torch.bool)].unsqueeze(0)

                def denoise(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                    if cancelled():
                        raise ZImageTurboCancelled("Z-Image generation canceled before denoiser")
                    raw = self.transformer(
                        value.to(torch.bfloat16),
                        sigma,
                        caption.to(torch.bfloat16),
                    )
                    sigma_broadcast = sigma.reshape((-1,) + (1,) * (value.ndim - 1))
                    result = value - raw.float() * sigma_broadcast
                    if not torch.isfinite(result).all():
                        raise ValueError(
                            "Z-Image flow wrapper returned non-finite denoised latents"
                        )
                    return result

                def sampler_progress(step: ZImageSamplerStep) -> None:
                    progress(
                        0.12 + 0.70 * ((step.index + 1) / 8), f"Z-Image step {step.index + 1}/8"
                    )

                try:
                    failure_stage("sampling")
                    latents = z_image_res_multistep(
                        denoise, latents, cancelled=cancelled, progress=sampler_progress
                    )
                except ZImageSamplingCancelled as exc:
                    raise ZImageTurboCancelled(str(exc)) from exc
                transformer_proof = transformer_stage.verify_dispatch()
            finally:
                transformer_stage.offload()

            lifecycle.checkpoint(ZImagePhase.VAE, cancelled)
            progress(0.86, "Decoding Z-Image PNG")
            failure_stage("decode")
            self.vae.to(device=self.execution_device, dtype=torch.float32)
            try:
                try:
                    images = decode_z_image_flux_ae(self.vae, latents, cancelled=cancelled)
                except ZImageDecodeCancelled as exc:
                    raise ZImageTurboCancelled(str(exc)) from exc
            finally:
                self.vae.to(device="cpu", dtype=torch.float32)
            if len(images) != 1:
                raise ValueError("Z-Image single-image core returned an invalid batch")
            if cancelled():
                raise ZImageTurboCancelled("Z-Image generation canceled before PNG publication")
            try:
                failure_stage("publish")
                artifact = write_z_image_png_atomic(
                    images[0],
                    output_path,
                    expected_size=(width, height),
                    cancelled=cancelled,
                )
            except ZImagePngPublicationCancelled as exc:
                raise ZImageTurboCancelled(str(exc)) from exc
            lifecycle.complete()
            self._require_cpu_master()
            progress(1.0, "Complete")
            return ZImageTurboCoreResult(
                artifact,
                seed,
                qwen_proof,
                transformer_proof,
                tuple(phase.value for phase in lifecycle.events),
            )
        except BaseException as exc:
            lifecycle.eject()
            cleanup_errors = []
            for label, cleanup in (
                ("text_encoder", text_stage.offload),
                ("transformer", transformer_stage.offload),
                ("vae", lambda: self.vae.to(device="cpu", dtype=torch.float32)),
            ):
                try:
                    cleanup()
                except (OSError, RuntimeError, TypeError, ValueError) as cleanup_exc:
                    cleanup_errors.append(f"{label}:{type(cleanup_exc).__name__}")
            if cleanup_errors:
                raise RuntimeError(
                    "Z-Image runtime ejection failed: " + ",".join(cleanup_errors)
                ) from exc
            raise
