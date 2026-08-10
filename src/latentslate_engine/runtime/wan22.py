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
class Wan22Size:
    width: int
    height: int


WAN22_SIZE_PRESETS: dict[str, Wan22Size] = {
    "1280x704": Wan22Size(width=1280, height=704),
    "704x1280": Wan22Size(width=704, height=1280),
}

WAN22_FPS = 24
WAN22_STEPS = 50
WAN22_GUIDANCE_SCALE = 5.0
WAN22_MIN_DURATION_SECONDS = 1.0
WAN22_MAX_DURATION_SECONDS = 5.0
WAN22_MIN_FRAMES = 25
WAN22_MAX_FRAMES = 121
WAN22_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, still image, "
    "overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "messy background, three legs, many people in the background, walking backwards"
)


def frames_for_duration(duration_seconds: float) -> int:
    """Return a Wan frame count satisfying ``(num_frames - 1) % 4 == 0``."""
    duration = min(
        WAN22_MAX_DURATION_SECONDS,
        max(WAN22_MIN_DURATION_SECONDS, duration_seconds),
    )
    requested = math.ceil(duration * WAN22_FPS)
    aligned = math.ceil(max(0, requested - 1) / 4) * 4 + 1
    return min(WAN22_MAX_FRAMES, max(WAN22_MIN_FRAMES, aligned))


class Wan22Runtime:
    """Lazy wrapper around Wan 2.2 TI2V-5B used in text-only mode."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._cache = RuntimeCache(
            f"wan22:{settings.wan22_model_id}:{settings.wan22_profile}",
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
            progress(0.02, "Loading Wan 2.2")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils import encode_video

            try:
                size = WAN22_SIZE_PRESETS[size_name]
            except KeyError as exc:
                raise ValueError(f"Unknown Wan 2.2 size {size_name!r}") from exc

            num_frames = frames_for_duration(duration_seconds)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            conditioning, prompt_cache_hit = self._prompt_conditioning(pipe, prompt)

            call_kwargs: dict[str, Any] = {}
            if conditioning is None:
                call_kwargs.update(
                    prompt=prompt,
                    negative_prompt=WAN22_NEGATIVE_PROMPT,
                )
            else:
                prompt_embeds, negative_prompt_embeds = conditioning
                call_kwargs.update(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                )

            progress(0.10, "Generating video")
            output = pipe(
                **call_kwargs,
                width=size.width,
                height=size.height,
                num_frames=num_frames,
                num_inference_steps=WAN22_STEPS,
                guidance_scale=WAN22_GUIDANCE_SCALE,
                generator=generator,
                output_type="np",
            ).frames[0]
            check_cancelled()

            progress(0.94, "Encoding MP4")
            encode_video(
                output,
                fps=WAN22_FPS,
                output_path=str(output_path),
            )
            progress(1.0, "Complete")
            return {
                "width": size.width,
                "height": size.height,
                "fps": WAN22_FPS,
                "frame_count": num_frames,
                "duration_seconds": num_frames / WAN22_FPS,
                "has_audio": False,
                "steps": WAN22_STEPS,
                "guidance_scale": WAN22_GUIDANCE_SCALE,
                "seed": seed,
                "size": size_name,
                "model_id": self.settings.wan22_model_id,
                "profile": self.settings.wan22_profile,
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": prompt_cache_hit,
                },
            }

    def _prompt_conditioning(self, pipe: Any, prompt: str) -> tuple[Any, bool]:
        if not hasattr(pipe, "encode_prompt"):
            return None, False
        key = self._cache.key(
            "prompt",
            {
                "prompt": prompt,
                "negative_prompt": WAN22_NEGATIVE_PROMPT,
                "guidance": True,
                "max_sequence_length": 512,
            },
        )
        cached = self._cache.prompt.get(key)
        device = pipe._execution_device
        if cached is not None:
            return materialize_cached(cached, device=device), True

        conditioning = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=WAN22_NEGATIVE_PROMPT,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        self._cache.prompt.put(key, conditioning)
        return conditioning, False

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "model_id": self.settings.wan22_model_id,
            "profile": self.settings.wan22_profile,
            "device": self.settings.wan22_device,
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
        from diffusers import AutoencoderKLWan, WanPipeline

        model_path = require_repository(
            self.settings.model_root,
            "wan22-basic",
            self.settings.wan22_model_id,
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        pipe = WanPipeline.from_pretrained(
            model_path,
            vae=vae,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        profile = self.settings.wan22_profile
        if profile == "bf16_sequential_offload":
            pipe.enable_sequential_cpu_offload(device=self.settings.wan22_device)
        elif profile == "bf16_model_offload":
            pipe.enable_model_cpu_offload(device=self.settings.wan22_device)
        elif profile == "bf16_cuda":
            pipe.to(self.settings.wan22_device)
        else:
            raise RuntimeError(
                f"Unknown LATENTSLATE_WAN22_PROFILE={profile!r}; expected "
                "bf16_sequential_offload, bf16_model_offload, or bf16_cuda"
            )

        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._pipeline = pipe
        return pipe
