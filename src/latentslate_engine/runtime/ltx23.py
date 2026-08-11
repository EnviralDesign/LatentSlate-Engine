from __future__ import annotations

import gc
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache, materialize_cached
from .diffusers_repository import LTX23_REPOSITORY_CONTRACT, validate_diffusers_repository
from .kit import ResolvedRuntimePlan, RuntimeDefaults, resolve_runtime_plan

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan


@dataclass(frozen=True, slots=True)
class LTX23Size:
    width: int
    height: int


LTX23_SIZE_PRESETS: dict[str, LTX23Size] = {
    "768x512": LTX23Size(width=768, height=512),
    "512x768": LTX23Size(width=512, height=768),
    "512x512": LTX23Size(width=512, height=512),
}

LTX23_FPS = 24
LTX23_STEPS = 8
LTX23_GUIDANCE_SCALE = 1.0
LTX23_MIN_DURATION_SECONDS = 1.0
LTX23_MAX_DURATION_SECONDS = 10.0
LTX23_MIN_FRAMES = 25
LTX23_MAX_FRAMES = 241


def _profile_modes(profile: str) -> tuple[str, str]:
    """Return the stored-weight and residency modes for a supported LTX profile."""

    try:
        return {
            "bf16_sequential_offload": ("bf16", "sequential"),
            "bf16_model_offload": ("bf16", "model"),
            "bf16_cuda": ("bf16", "none"),
        }[profile]
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown LATENTSLATE_LTX23_PROFILE={profile!r}; expected "
            "bf16_sequential_offload, bf16_model_offload, or bf16_cuda. "
            "LatentSlate Engine does not convert model weights at runtime."
        ) from exc


def frames_for_duration(duration_seconds: float) -> int:
    """Return an LTX-2.3 frame count satisfying ``num_frames % 8 == 1``."""
    duration = min(
        LTX23_MAX_DURATION_SECONDS,
        max(LTX23_MIN_DURATION_SECONDS, duration_seconds),
    )
    requested = math.ceil(duration * LTX23_FPS)
    aligned = math.ceil(max(0, requested - 1) / 8) * 8 + 1
    return min(LTX23_MAX_FRAMES, max(LTX23_MIN_FRAMES, aligned))


def resolve_ltx23_runtime_plan(
    settings: Settings,
    execution: ExecutionPlan | None,
) -> ResolvedRuntimePlan:
    """Resolve one complete native-BF16 LTX 2.3 Diffusers repository.

    Resource selection is a load recipe, not a conversion request.  The current
    LTX adapter deliberately supports only a complete, native-BF16 Diffusers
    directory; file checkpoints and every stored quantization contract remain
    unavailable until a matching loader is implemented and validated.
    """

    quantization, offload = _profile_modes(settings.ltx23_profile)
    model_path = (
        execution.model_path
        if execution is not None and execution.model_path is not None
        else require_repository(
            settings.model_root,
            "ltx23-basic",
            settings.ltx23_model_id,
        )
    )
    resolved_path = Path(model_path).resolve(strict=True)
    validate_diffusers_repository(resolved_path, LTX23_REPOSITORY_CONTRACT)

    defaults = RuntimeDefaults(
        family="ltx23",
        model_id=settings.ltx23_model_id,
        model_path=resolved_path,
        model_format="diffusers",
        device=settings.ltx23_device,
        quantization=quantization,
        attention="native",
        offload=offload,
        artifact_precision="bf16",
        artifact_quantization="native",
        vae_tiling="on",
        vae_slicing="off",
        cache="prompt",
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
    )
    plan = resolve_runtime_plan(execution, defaults)
    if plan.model_format != "diffusers":
        raise ValueError(
            f"LTX 2.3 supports complete Diffusers directories only, not {plan.model_format!r}"
        )
    if (
        execution is not None
        and execution.model_path is not None
        and (
            execution.model_format is None
            or execution.model_precision is None
            or execution.model_quantization is None
        )
    ):
        raise ValueError(
            "A selected LTX 2.3 model must explicitly declare format='diffusers', "
            "precision='bf16', and quantization='native'"
        )
    if (
        plan.quantization != "bf16"
        or plan.model_precision != "bf16"
        or plan.model_quantization != "native"
    ):
        raise ValueError(
            "LTX 2.3 currently supports only a native BF16 artifact; "
            "no quantized LTX loader or runtime conversion is implemented"
        )
    validate_diffusers_repository(plan.model_path, LTX23_REPOSITORY_CONTRACT)
    if plan.attention != "native":
        raise ValueError(f"LTX 2.3 supports only native attention, not {plan.attention!r}")
    if plan.offload not in {"sequential", "model", "none"}:
        raise ValueError(f"LTX 2.3 does not implement offload mode {plan.offload!r}")
    if plan.vae_tiling != "on" or plan.vae_slicing != "off":
        raise ValueError("LTX 2.3 requires VAE tiling on and VAE slicing off")
    if plan.cache not in {"none", "prompt"}:
        raise ValueError(f"LTX 2.3 does not implement cache mode {plan.cache!r}")
    if plan.loras:
        raise ValueError("LTX 2.3 LoRAs are not implemented by this runtime")
    return plan


class LTX23Runtime:
    """Lazy wrapper around the distilled upstream Diffusers LTX-2.3 pipeline."""

    def __init__(self, settings: Settings, load_plan: ResolvedRuntimePlan):
        if load_plan.family != "ltx23":
            raise ValueError(f"LTX 2.3 runtime cannot load execution family {load_plan.family!r}")
        self.settings = settings
        self.load_plan = load_plan
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._cache = RuntimeCache(
            load_plan.pipeline_fingerprint,
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            max_entries=settings.cache_max_entries,
        )

    def generate(
        self,
        *,
        plan: ResolvedRuntimePlan,
        prompt: str,
        output_path: Path,
        size_name: str,
        duration_seconds: float,
        seed: int,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            check_cancelled()
            self.load_plan.assert_same_pipeline(plan)
            pipeline_warm = self._pipeline is not None
            progress(0.02, "Loading LTX 2.3")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.pipelines.ltx2.utils import (
                DEFAULT_NEGATIVE_PROMPT,
                DISTILLED_SIGMA_VALUES,
            )
            from diffusers.utils import encode_video

            try:
                size = LTX23_SIZE_PRESETS[size_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LTX 2.3 size {size_name!r}") from exc

            num_frames = frames_for_duration(duration_seconds)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            conditioning, prompt_cache_hit = self._prompt_conditioning(
                pipe,
                plan,
                prompt,
                DEFAULT_NEGATIVE_PROMPT,
            )

            call_kwargs: dict[str, Any] = {}
            if conditioning is None:
                call_kwargs.update(
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                )
            else:
                (
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                ) = conditioning
                call_kwargs.update(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                )

            progress(0.10, "Generating synchronized video and audio")
            video, audio = pipe(
                **call_kwargs,
                width=size.width,
                height=size.height,
                num_frames=num_frames,
                frame_rate=float(LTX23_FPS),
                num_inference_steps=LTX23_STEPS,
                sigmas=DISTILLED_SIGMA_VALUES,
                guidance_scale=LTX23_GUIDANCE_SCALE,
                generator=generator,
                output_type="np",
                return_dict=False,
            )
            check_cancelled()

            progress(0.94, "Encoding MP4")
            encode_video(
                video[0],
                fps=LTX23_FPS,
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                output_path=str(output_path),
            )
            progress(1.0, "Complete")
            return {
                "width": size.width,
                "height": size.height,
                "fps": LTX23_FPS,
                "frame_count": num_frames,
                "duration_seconds": num_frames / LTX23_FPS,
                "has_audio": True,
                "steps": LTX23_STEPS,
                "guidance_scale": LTX23_GUIDANCE_SCALE,
                "seed": seed,
                "size": size_name,
                "model_id": plan.model_resource_id or plan.model_id,
                "profile": self.settings.ltx23_profile,
                "pipeline_fingerprint": plan.pipeline_fingerprint,
                "pipeline_kit": {
                    "attention": plan.attention,
                    "offload": plan.offload,
                    "vae_tiling": plan.vae_tiling,
                    "cache": plan.cache,
                },
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": prompt_cache_hit,
                },
            }

    def _prompt_conditioning(
        self,
        pipe: Any,
        plan: ResolvedRuntimePlan,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[Any, bool]:
        if not hasattr(pipe, "encode_prompt"):
            return None, False
        if not plan.cache_prompt:
            return (
                pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    max_sequence_length=1024,
                    device=pipe._execution_device,
                ),
                False,
            )
        key = self._cache.key(
            "prompt",
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guidance": False,
                "max_sequence_length": 1024,
            },
        )
        cached = self._cache.prompt.get(key)
        device = pipe._execution_device
        if cached is not None:
            return materialize_cached(cached, device=device), True

        conditioning = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=1024,
            device=device,
        )
        self._cache.prompt.put(key, conditioning)
        return conditioning, False

    def status(self) -> dict[str, Any]:
        return {
            "family": "ltx23",
            "model_id": self.load_plan.model_resource_id or self.load_plan.model_id,
            "profile": self.settings.ltx23_profile,
            "device": self.settings.ltx23_device,
            "pipeline_fingerprint": self.load_plan.pipeline_fingerprint,
            "loaded": self._pipeline is not None,
            "cache_support": {"prompt": True, "media": False},
            "cache": self._cache.status(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is None:
                return
            try:
                pipeline.remove_all_hooks()
            except Exception:  # noqa: BLE001, S110 - third-party teardown is best effort
                pass
            del pipeline
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001, S110 - CUDA cleanup is best effort
                pass

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers.pipelines.ltx2 import LTX2Pipeline

        self.load_plan.revalidate_components()
        pipe = LTX2Pipeline.from_pretrained(
            self.load_plan.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=self.load_plan.low_cpu_mem_usage,
        )
        try:
            self.load_plan.revalidate_components()
        except BaseException:
            del pipe
            raise
        if self.load_plan.offload == "sequential":
            pipe.enable_sequential_cpu_offload(device=self.load_plan.device)
        elif self.load_plan.offload == "model":
            pipe.enable_model_cpu_offload(device=self.load_plan.device)
        elif self.load_plan.offload == "none":
            pipe.to(self.load_plan.device)
        else:  # resolve_ltx23_runtime_plan guards this; retain local fail-closed defense.
            raise RuntimeError(
                f"LTX 2.3 does not implement offload mode {self.load_plan.offload!r}"
            )

        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._pipeline = pipe
        return pipe
