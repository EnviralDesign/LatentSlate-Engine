from __future__ import annotations

import gc
import math
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache
from .diffusers_repository import H3_REPOSITORY_CONTRACT, validate_diffusers_repository
from .dimensions import Dimensions, align_dimensions
from .kit import ResolvedRuntimePlan, RuntimeDefaults, resolve_runtime_plan

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan


H3_FPS = 24
H3_AUDIO_SAMPLE_RATE = 32_000
H3_AUDIO_CHANNELS = 2
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_MIN_FRAMES = 124
H3_MAX_FRAMES = 345
H3_DIMENSION_ALIGNMENT = 32
H3_MIN_SIDE = 64
H3_MAX_PIXELS = 1_032_192
H3_MIN_STEPS = 1
H3_MAX_STEPS = 30
# The public quality preset was retired in schema revision 2.  Keep the prior
# product policy visible and reproducible through the dedicated steps input.
H3_LEGACY_PRESET_STEPS = {"draft": 16, "balanced": 20, "final": 30}
H3_DEFAULT_WIDTH = 960
H3_DEFAULT_HEIGHT = 544
H3_DEFAULT_STEPS = H3_LEGACY_PRESET_STEPS["balanced"]
H3_MIN_DURATION_SECONDS = H3_MIN_FRAMES / H3_FPS
H3_MAX_DURATION_SECONDS = H3_MAX_FRAMES / H3_FPS
H3_TEXT_WORKFLOW = "t2va"
H3_FIRST_LAST_WORKFLOW = "fl2va"
H3_WORKFLOWS = frozenset({H3_TEXT_WORKFLOW, H3_FIRST_LAST_WORKFLOW})
# This is the immutable upstream identity used to define the accepted direct
# Diffusers closure. It is deliberately a *validator* identity, not a claim
# about the origin of an override folder with matching verified contents.
H3_FL2VA_CONTRACT_REPOSITORY = "MiniMaxAI/MiniMax-H3"
H3_FL2VA_CONTRACT_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def resolve_h3_dimensions(width: int | None, height: int | None) -> Dimensions:
    """Normalize an explicit H3 canvas before any pipeline components are loaded."""

    if (width is None) != (height is None):
        raise ValueError("H3 width and height must be supplied together")
    if width is None or height is None:
        raise ValueError("H3 width and height are required")
    dimensions = align_dimensions(
        width,
        height,
        alignment=H3_DIMENSION_ALIGNMENT,
        min_side=H3_MIN_SIDE,
        max_pixels=H3_MAX_PIXELS,
    )
    if dimensions.width > dimensions.height * 4 or dimensions.height > dimensions.width * 4:
        raise ValueError("aligned H3 dimensions must stay within a 1:4 to 4:1 aspect ratio")
    return dimensions


def validate_h3_steps(steps: int) -> int:
    """Validate the deliberate H3 step budget exposed by the public schema."""

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("H3 steps must be an integer")
    if not H3_MIN_STEPS <= steps <= H3_MAX_STEPS:
        raise ValueError(f"H3 steps must be between {H3_MIN_STEPS} and {H3_MAX_STEPS}")
    return steps


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
    *,
    workflow: str = H3_TEXT_WORKFLOW,
) -> ResolvedRuntimePlan:
    """Resolve one complete BF16 Diffusers folder for H3.

    H3 uses the upstream ModularPipeline with auto CPU offload. The selected
    workflow is a pipeline-load parameter, so it is included in the plan's
    fingerprint and cannot collide in the runtime manager. The execution plan may
    replace the complete model folder, but it may not change the stored artifact
    format or precision. Quantized loaders are intentionally absent until an exact
    H3 artifact contract is proven.
    """

    if workflow not in H3_WORKFLOWS:
        raise ValueError(f"Unsupported H3 workflow {workflow!r}")
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
        pipeline_parameters=(
            ("workflow", workflow),
            ("contract_revision", H3_FL2VA_CONTRACT_REVISION),
        ),
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
    """Lazy, persistent wrapper around one upstream Diffusers H3 workflow."""

    def __init__(self, settings: Settings, load_plan: ResolvedRuntimePlan):
        if load_plan.family != "h3":
            raise ValueError(f"H3 runtime cannot load execution family {load_plan.family!r}")
        self.settings = settings
        self.load_plan = load_plan
        self.workflow = str(dict(load_plan.pipeline_parameters).get("workflow", ""))
        if self.workflow not in H3_WORKFLOWS:
            raise ValueError(f"H3 runtime has unsupported workflow {self.workflow!r}")
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._cache = RuntimeCache(
            f"h3:{settings.h3_model_id}:{settings.h3_profile}:{self.workflow}",
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
        width: int | None,
        height: int | None,
        steps: int,
        duration_seconds: float,
        seed: int,
        image_path: Path | None,
        last_image_path: Path | None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            check_cancelled()
            self.load_plan.assert_same_pipeline(plan)
            dimensions = resolve_h3_dimensions(width, height)
            steps = validate_h3_steps(steps)
            num_frames = frames_for_duration(duration_seconds)
            pipeline_warm = self._pipeline is not None
            progress(0.02, "Loading MiniMax-H3")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils.export_utils import encode_video

            generator = torch.Generator(device="cpu").manual_seed(seed)
            generation_args: dict[str, Any] = {
                "prompt": prompt,
                "width": dimensions.width,
                "height": dimensions.height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "generator": generator,
                "output": ["videos", "audio", "sampling_rate"],
            }
            if self.workflow == H3_FIRST_LAST_WORKFLOW:
                if image_path is None and last_image_path is None:
                    raise ValueError("H3 fl2va requires a first- or last-frame image")
                from diffusers.utils import load_image

                if image_path is not None:
                    generation_args["image"] = load_image(str(image_path))
                if last_image_path is not None:
                    generation_args["last_image"] = load_image(str(last_image_path))
            elif image_path is not None or last_image_path is not None:
                raise ValueError("H3 t2va does not accept keyframe images")

            check_cancelled()
            progress(0.10, "Generating video and audio")
            result = pipe(**generation_args)
            check_cancelled()
            audio = result["audio"][0]
            sampling_rate = result["sampling_rate"]
            _validate_h3_audio_output(audio, sampling_rate)
            progress(0.94, "Encoding MP4")
            encode_video(
                result["videos"][0],
                fps=H3_FPS,
                output_path=str(output_path),
                audio=audio,
                audio_sample_rate=sampling_rate,
            )
            check_cancelled()
            progress(1.0, "Complete")
            return {
                "width": dimensions.width,
                "height": dimensions.height,
                **dimensions.metadata(),
                "fps": H3_FPS,
                "audio_sample_rate": H3_AUDIO_SAMPLE_RATE,
                "audio_channels": H3_AUDIO_CHANNELS,
                "frame_count": num_frames,
                "duration_seconds": num_frames / H3_FPS,
                "has_audio": True,
                "steps": steps,
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
            "workflow": self.workflow,
            "device": self.settings.h3_device,
            "pipeline_fingerprint": self.load_plan.pipeline_fingerprint,
            "supported_closure": {
                "repository": H3_FL2VA_CONTRACT_REPOSITORY,
                "revision": H3_FL2VA_CONTRACT_REVISION,
            },
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
            workflow=self.workflow,
            components_manager=manager,
        )
        try:
            pipe.load_components(dtype=torch.bfloat16)
            self.load_plan.revalidate_components()
        except BaseException:
            del pipe
            raise
        return pipe


def _validate_h3_audio_output(audio: Any, sampling_rate: Any) -> None:
    """Reject a pipeline result that cannot be the released H3 stereo contract."""

    if isinstance(sampling_rate, bool) or sampling_rate != H3_AUDIO_SAMPLE_RATE:
        raise ValueError(
            "MiniMax-H3 must return its native 32 kHz stereo audio; "
            f"got sampling_rate={sampling_rate!r}"
        )
    shape = getattr(audio, "shape", None)
    if shape is None:
        try:
            shape = (len(audio), len(audio[0]))
        except (IndexError, TypeError):
            shape = None
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != H3_AUDIO_CHANNELS:
        raise ValueError(
            "MiniMax-H3 must return a channel-major stereo waveform with shape "
            f"(2, samples); got {shape!r}"
        )
