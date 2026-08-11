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
from .cache import RuntimeCache
from .diffusers_repository import H3_REPOSITORY_CONTRACT, validate_diffusers_repository
from .kit import ResolvedRuntimePlan, RuntimeDefaults, resolve_runtime_plan

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan


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


def resolve_h3_runtime_plan(
    settings: Settings,
    execution: ExecutionPlan | None,
) -> ResolvedRuntimePlan:
    """Resolve one complete BF16 Diffusers folder for H3.

    H3 has one supported profile today: the upstream ModularPipeline with auto CPU
    offload.  The execution plan may replace the complete model folder, but it may
    not change the stored artifact format or precision.  Quantized loaders are
    intentionally absent until an exact H3 artifact contract is proven.
    """

    if settings.h3_profile != "bf16_auto_offload":
        raise RuntimeError(
            f"Unknown LATENTSLATE_H3_PROFILE={settings.h3_profile!r}; "
            "expected bf16_auto_offload. LatentSlate Engine does not convert "
            "model weights at runtime."
        )

    model_path = (
        execution.model_path
        if execution is not None and execution.model_path is not None
        else require_repository(settings.model_root, "h3-basic", settings.h3_model_id)
    )
    model_path = Path(model_path).resolve(strict=True)
    validate_diffusers_repository(model_path, H3_REPOSITORY_CONTRACT)

    defaults = RuntimeDefaults(
        family="h3",
        model_id=settings.h3_model_id,
        model_path=model_path,
        model_format="diffusers",
        device=settings.h3_device,
        quantization="bf16",
        attention="native",
        # H3 uses ComponentsManager's auto CPU offload rather than the generic
        # Diffusers offload helpers. Keep this in the fingerprint/provenance while
        # leaving application of the policy to _load_bf16_auto_offload.
        offload="auto",
        artifact_precision="bf16",
        artifact_quantization="native",
        vae_tiling="off",
        vae_slicing="off",
        cache="none",
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
    )
    plan = resolve_runtime_plan(execution, defaults)
    if plan.model_format != "diffusers":
        raise ValueError(
            f"H3 supports complete Diffusers directories only, not {plan.model_format!r}"
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
            "A selected H3 model must explicitly declare format='diffusers', "
            "precision='bf16', and quantization='native'"
        )
    if (
        plan.quantization != "bf16"
        or plan.model_precision != "bf16"
        or plan.model_quantization != "native"
    ):
        raise ValueError(
            "H3 currently supports only a native BF16 artifact; "
            "no quantized H3 loader or runtime conversion is implemented"
        )
    validate_diffusers_repository(plan.model_path, H3_REPOSITORY_CONTRACT)
    if plan.attention != "native":
        raise ValueError(f"H3 supports only native attention, not {plan.attention!r}")
    if plan.offload != "auto":
        raise ValueError(f"H3 requires auto CPU offload, not {plan.offload!r}")
    if plan.vae_tiling != "off" or plan.vae_slicing != "off":
        raise ValueError("H3 requires VAE tiling and slicing to remain off")
    if plan.cache != "none":
        raise ValueError("H3 does not implement a conditioning cache")
    if plan.loras:
        raise ValueError("H3 LoRAs are not implemented by this runtime")
    return plan


class H3Runtime:
    """Lazy, persistent wrapper around the upstream Diffusers H3 FL2VA workflow."""

    def __init__(self, settings: Settings, load_plan: ResolvedRuntimePlan):
        if load_plan.family != "h3":
            raise ValueError(f"H3 runtime cannot load execution family {load_plan.family!r}")
        self.settings = settings
        self.load_plan = load_plan
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
        plan: ResolvedRuntimePlan,
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
            self.load_plan.assert_same_pipeline(plan)
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
                "model_id": plan.model_resource_id or plan.model_id,
                "pipeline_fingerprint": plan.pipeline_fingerprint,
                "pipeline_kit": {
                    "attention": plan.attention,
                    "offload": plan.offload,
                    "cache": plan.cache,
                },
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": False,
                    "reference_hits": 0,
                },
            }

    def status(self) -> dict[str, Any]:
        return {
            "family": "h3",
            "model_id": self.load_plan.model_resource_id or self.load_plan.model_id,
            "profile": self.settings.h3_profile,
            "device": self.settings.h3_device,
            "pipeline_fingerprint": self.load_plan.pipeline_fingerprint,
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
        self._pipeline = self._load_bf16_auto_offload()
        return self._pipeline

    def _load_bf16_auto_offload(self) -> Any:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        self.load_plan.revalidate_components()
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=self.settings.h3_device)
        pipe = ModularPipeline.from_pretrained(
            self.load_plan.model_path,
            workflow="fl2va",
            components_manager=manager,
        )
        try:
            pipe.load_components(dtype=torch.bfloat16)
            self.load_plan.revalidate_components()
        except BaseException:
            del pipe
            raise
        return pipe
