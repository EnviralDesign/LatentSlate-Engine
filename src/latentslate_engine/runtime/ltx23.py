from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache, materialize_cached


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


def frames_for_duration(duration_seconds: float) -> int:
    """Return an LTX-2.3 frame count satisfying ``num_frames % 8 == 1``."""
    duration = min(
        LTX23_MAX_DURATION_SECONDS,
        max(LTX23_MIN_DURATION_SECONDS, duration_seconds),
    )
    requested = math.ceil(duration * LTX23_FPS)
    aligned = math.ceil(max(0, requested - 1) / 8) * 8 + 1
    return min(LTX23_MAX_FRAMES, max(LTX23_MIN_FRAMES, aligned))


class LTX23Runtime:
    """Lazy wrapper around the distilled upstream Diffusers LTX-2.3 pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._cache = RuntimeCache(
            f"ltx23:{settings.ltx23_model_id}:{settings.ltx23_profile}",
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            max_entries=settings.cache_max_entries,
        )

    def generate(
        self,
        *,
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
                "model_id": self.settings.ltx23_model_id,
                "profile": self.settings.ltx23_profile,
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": prompt_cache_hit,
                },
            }

    def _prompt_conditioning(
        self,
        pipe: Any,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[Any, bool]:
        if not hasattr(pipe, "encode_prompt"):
            return None, False
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
            "model_id": self.settings.ltx23_model_id,
            "profile": self.settings.ltx23_profile,
            "device": self.settings.ltx23_device,
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
            except Exception:
                pass
            del pipeline
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers.pipelines.ltx2 import LTX2Pipeline

        model_path = require_repository(
            self.settings.model_root,
            "ltx23-basic",
            self.settings.ltx23_model_id,
        )
        pipe = LTX2Pipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        profile = self.settings.ltx23_profile
        if profile == "bf16_sequential_offload":
            pipe.enable_sequential_cpu_offload(device=self.settings.ltx23_device)
        elif profile == "bf16_model_offload":
            pipe.enable_model_cpu_offload(device=self.settings.ltx23_device)
        elif profile == "bf16_cuda":
            pipe.to(self.settings.ltx23_device)
        else:
            raise RuntimeError(
                f"Unknown LATENTSLATE_LTX23_PROFILE={profile!r}; expected "
                "bf16_sequential_offload, bf16_model_offload, or bf16_cuda"
            )

        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._pipeline = pipe
        return pipe
