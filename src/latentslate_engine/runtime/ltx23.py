from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ..config import Settings


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
LTX23_STEPS = 40
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
    """Lazy wrapper around the upstream Diffusers LTX-2.3 text-to-video pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._lock = Lock()

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
            progress(0.02, "Loading LTX 2.3")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
            from diffusers.utils import encode_video

            try:
                size = LTX23_SIZE_PRESETS[size_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LTX 2.3 size {size_name!r}") from exc

            num_frames = frames_for_duration(duration_seconds)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            progress(0.10, "Generating synchronized video and audio")
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                width=size.width,
                height=size.height,
                num_frames=num_frames,
                frame_rate=float(LTX23_FPS),
                num_inference_steps=LTX23_STEPS,
                guidance_scale=3.0,
                audio_guidance_scale=7.0,
                use_cross_timestep=True,
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
                "seed": seed,
                "size": size_name,
                "model_id": self.settings.ltx23_model_id,
                "profile": self.settings.ltx23_profile,
            }

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

        pipe = LTX2Pipeline.from_pretrained(
            self.settings.ltx23_model_id,
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
