from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ..config import Settings
from .cache import RuntimeCache


@dataclass(frozen=True, slots=True)
class H3Preset:
    width: int
    height: int
    steps: int


PRESETS: dict[str, H3Preset] = {
    "draft": H3Preset(width=832, height=480, steps=16),
    "balanced": H3Preset(width=960, height=544, steps=20),
    "final": H3Preset(width=960, height=544, steps=30),
}

H3_FPS = 24
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_MIN_FRAMES = 124
H3_MAX_FRAMES = 345
H3_MIN_DURATION_SECONDS = H3_MIN_FRAMES / H3_FPS
H3_MAX_DURATION_SECONDS = H3_MAX_FRAMES / H3_FPS


def frames_for_duration(duration_seconds: float) -> int:
    """Return the next legal H3 frame count without crossing its 15-second ceiling."""
    duration = min(H3_MAX_DURATION_SECONDS, max(5.0, duration_seconds))
    requested = math.ceil(duration * H3_FPS)
    groups = max(
        1,
        math.ceil((requested - H3_LATENTS_PER_CHUNK) / H3_FRAMES_PER_CHUNK),
    )
    aligned = groups * H3_FRAMES_PER_CHUNK + H3_LATENTS_PER_CHUNK
    return min(H3_MAX_FRAMES, max(H3_MIN_FRAMES, aligned))


class H3Runtime:
    """Lazy, persistent wrapper around the upstream Diffusers H3 FL2VA workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._cache = RuntimeCache(
            f"h3:{settings.h3_model_id}:{settings.h3_profile}",
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            max_entries=settings.cache_max_entries,
        )

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        preset_name: str,
        duration_seconds: float,
        seed: int,
        image_path: Path | None,
        last_image_path: Path | None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            check_cancelled()
            pipeline_warm = self._pipeline is not None
            progress(0.02, "Loading MiniMax-H3")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils import load_image
            from diffusers.utils.export_utils import encode_video

            preset = PRESETS[preset_name]
            num_frames = frames_for_duration(duration_seconds)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = load_image(str(image_path)) if image_path else None
            last_image = load_image(str(last_image_path)) if last_image_path else None

            progress(0.10, "Generating video and audio")
            result = pipe(
                prompt=prompt,
                image=image,
                last_image=last_image,
                width=preset.width,
                height=preset.height,
                num_frames=num_frames,
                num_inference_steps=preset.steps,
                generator=generator,
                output=["videos", "audio", "sampling_rate"],
            )
            check_cancelled()
            progress(0.94, "Encoding MP4")
            encode_video(
                result["videos"][0],
                fps=H3_FPS,
                output_path=str(output_path),
                audio=result["audio"][0],
                audio_sample_rate=result["sampling_rate"],
            )
            progress(1.0, "Complete")
            return {
                "width": preset.width,
                "height": preset.height,
                "fps": H3_FPS,
                "frame_count": num_frames,
                "duration_seconds": num_frames / H3_FPS,
                "has_audio": True,
                "steps": preset.steps,
                "preset": preset_name,
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": False,
                    "reference_hits": 0,
                },
            }

    def status(self) -> dict[str, Any]:
        return {
            "family": "h3",
            "model_id": self.settings.h3_model_id,
            "profile": self.settings.h3_profile,
            "device": self.settings.h3_device,
            "loaded": self._pipeline is not None,
            "cache_support": {
                "prompt": False,
                "media": False,
                "reason": (
                    "H3 conditioning is a multi-stage Modular Diffusers state; cache "
                    "plumbing is reserved but remains disabled until base H3 inference "
                    "is validated on the target memory profile."
                ),
            },
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
        profile = self.settings.h3_profile
        if profile == "consumer_int8":
            self._pipeline = self._load_consumer_int8()
        elif profile == "bf16_auto_offload":
            self._pipeline = self._load_bf16_auto_offload()
        else:
            raise RuntimeError(
                f"Unknown LATENTSLATE_H3_PROFILE={profile!r}; "
                "expected consumer_int8 or bf16_auto_offload"
            )
        return self._pipeline

    def _load_bf16_auto_offload(self) -> Any:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=self.settings.h3_device)
        pipe = ModularPipeline.from_pretrained(
            self.settings.h3_model_id,
            workflow="fl2va",
            components_manager=manager,
        )
        pipe.load_components(dtype=torch.bfloat16)
        return pipe

    def _load_consumer_int8(self) -> Any:
        import torch
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
        from diffusers.hooks import apply_group_offloading
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        model_id = self.settings.h3_model_id
        pipe = ModularPipeline.from_pretrained(model_id)
        pipe.update_components(
            transformer=MiniMaxH3Transformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                dtype=torch.bfloat16,
                quantization_config=TorchAoConfig(
                    Int8WeightOnlyConfig(version=2),
                    modules_to_not_convert=[
                        "proj_in",
                        "audio_proj_in",
                        "context_embedder",
                        "time_embedder",
                        "time_proj",
                        "token_refiner",
                        "norm_out",
                        "proj_out",
                        "audio_proj_out",
                    ],
                ),
                low_cpu_mem_usage=False,
            ),
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                subfolder="text_encoder",
                dtype=torch.bfloat16,
                quantization_config=TransformersTorchAoConfig(
                    Int8WeightOnlyConfig(version=2),
                    modules_to_not_convert=[
                        "model.visual",
                        "model.language_model.embed_tokens",
                        "model.language_model.norm",
                        "lm_head",
                    ],
                ),
            ),
        )
        pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)

        onload = torch.device(self.settings.h3_device)
        offload = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "use_stream": True,
        }
        pipe.transformer.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=1,
            **offload,
        )
        apply_group_offloading(
            pipe.text_encoder.model,
            offload_type="leaf_level",
            **offload,
        )
        apply_group_offloading(
            pipe.vae,
            offload_type="leaf_level",
            onload_device=onload,
            offload_device=torch.device("cpu"),
            use_stream=False,
        )
        pipe.audio_vae.to(onload)
        return pipe
