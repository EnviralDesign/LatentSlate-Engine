from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache, materialize_cached
from .kit import (
    ResolvedRuntimePlan,
    RuntimeDefaults,
    apply_pipeline_kit,
    cleanup_accelerator_memory,
    resolve_runtime_plan,
)

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan


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
WAN22_MAX_SEQUENCE_LENGTH = 512
WAN22_PROMPT_WORKER_TIMEOUT_SECONDS = 30 * 60
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


def _profile_modes(profile: str) -> tuple[str, str]:
    try:
        return {
            "bf16_sequential_offload": ("bf16", "sequential"),
            "bf16_group_leaf": ("bf16", "group_leaf"),
            # Preserved for existing explicit environment configurations. These
            # are not advertised as 16 GB recovery variants.
            "bf16_model_offload": ("bf16", "model"),
            "bf16_cuda": ("bf16", "none"),
        }[profile]
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown Wan 2.2 profile {profile!r}; expected "
            "bf16_sequential_offload, bf16_group_leaf, "
            "bf16_model_offload, or bf16_cuda"
        ) from exc


def resolve_wan22_runtime_plan(
    settings: Settings,
    execution: ExecutionPlan | None,
) -> ResolvedRuntimePlan:
    model_id = settings.wan22_model_id
    quantization, offload = _profile_modes(settings.wan22_profile)
    model_path = (
        execution.model_path
        if execution is not None and execution.model_path is not None
        else require_repository(settings.model_root, "wan22-basic", model_id)
    )
    defaults = RuntimeDefaults(
        family="wan22",
        model_id=model_id,
        model_path=Path(model_path),
        model_format="diffusers",
        device=settings.wan22_device,
        quantization=quantization,
        attention="native",
        offload=offload,
        vae_tiling="on",
        vae_slicing="off",
        cache="prompt",
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
        group_offload_blocks=1,
        group_offload_use_stream=False,
        group_offload_record_stream=False,
    )
    plan = resolve_runtime_plan(execution, defaults)
    return plan


class Wan22Runtime:
    """Staged Wan 2.2 TI2V-5B runtime for one exact pipeline fingerprint.

    The UMT5 encoder runs in a short-lived CPU subprocess. On a cache miss, any
    loaded generation pipeline is released before the subprocess starts, ensuring
    the text encoder and the 20 GB transformer are never resident together.
    """

    def __init__(self, settings: Settings, load_plan: ResolvedRuntimePlan):
        if load_plan.family != "wan22":
            raise ValueError(
                f"Wan 2.2 runtime cannot load execution family {load_plan.family!r}"
            )
        self.settings = settings
        self.load_plan = load_plan
        self._pipeline: Any | None = None
        self._pipeline_kit: dict[str, Any] = {}
        self._lock = RLock()
        self._prompt_worker_runs = 0
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
            self.load_plan.assert_same_pipeline(plan)
            check_cancelled()
            try:
                progress(0.01, "Preparing Wan 2.2 prompt conditioning")
                conditioning_cpu, prompt_cache_hit, prompt_stage = (
                    self._prompt_conditioning_cpu(
                        plan,
                        prompt,
                        progress=progress,
                        check_cancelled=check_cancelled,
                    )
                )
                check_cancelled()

                pipeline_warm = self._pipeline is not None
                progress(0.05, "Loading Wan 2.2 generation pipeline")
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
                prompt_embeds, negative_prompt_embeds = materialize_cached(
                    conditioning_cpu,
                    device=pipe._execution_device,
                )

                def callback_on_step_end(
                    _pipe: Any,
                    step_index: int,
                    _timestep: Any,
                    callback_kwargs: dict[str, Any],
                ) -> dict[str, Any]:
                    check_cancelled()
                    fraction = (step_index + 1) / WAN22_STEPS
                    progress(
                        0.12 + 0.76 * fraction,
                        f"Generating video ({step_index + 1}/{WAN22_STEPS})",
                    )
                    return callback_kwargs

                progress(0.10, "Generating video")
                output = pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    width=size.width,
                    height=size.height,
                    num_frames=num_frames,
                    num_inference_steps=WAN22_STEPS,
                    guidance_scale=WAN22_GUIDANCE_SCALE,
                    generator=generator,
                    output_type="np",
                    callback_on_step_end=callback_on_step_end,
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
                    "model_id": plan.model_resource_id or plan.model_id,
                    "profile": self.settings.wan22_profile,
                    "quantization": plan.quantization,
                    "offload": plan.offload,
                    "pipeline_fingerprint": plan.pipeline_fingerprint,
                    "pipeline_kit": dict(self._pipeline_kit),
                    "staged_text_encoder": True,
                    "prompt_stage": prompt_stage,
                    "cache": {
                        "policy": plan.cache,
                        "pipeline_warm": pipeline_warm,
                        "prompt_hit": prompt_cache_hit,
                    },
                }
            except BaseException:
                # Wan failures are treated as potentially poisoned even when they
                # are not classified as CUDA OOM at the generic job boundary.
                self._unload_pipeline_locked()
                raise

    def _prompt_conditioning_cpu(
        self,
        plan: ResolvedRuntimePlan,
        prompt: str,
        *,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> tuple[tuple[Any, Any], bool, dict[str, Any]]:
        key = self._cache.key(
            "prompt",
            {
                "pipeline": plan.pipeline_fingerprint,
                "prompt": prompt,
                "negative_prompt": WAN22_NEGATIVE_PROMPT,
                "max_sequence_length": WAN22_MAX_SEQUENCE_LENGTH,
            },
        )
        if plan.cache_prompt:
            cached = self._cache.prompt.get(key)
            if cached is not None:
                return materialize_cached(cached, device="cpu"), True, {
                    "mode": "isolated_cpu_subprocess",
                    "cache_hit": True,
                    "pipeline_unloaded_before_encode": False,
                    "worker_seconds": 0.0,
                }

        pipeline_unloaded = self._pipeline is not None
        if pipeline_unloaded:
            progress(0.015, "Unloading Wan pipeline before CPU prompt encoding")
            self._unload_pipeline_locked()

        conditioning, elapsed = self._run_prompt_worker(
            plan,
            prompt,
            check_cancelled=check_cancelled,
        )
        if plan.cache_prompt:
            self._cache.prompt.put(key, conditioning)
        return conditioning, False, {
            "mode": "isolated_cpu_subprocess",
            "cache_hit": False,
            "pipeline_unloaded_before_encode": pipeline_unloaded,
            "worker_seconds": elapsed,
        }

    def _run_prompt_worker(
        self,
        plan: ResolvedRuntimePlan,
        prompt: str,
        *,
        check_cancelled: Callable[[], None],
    ) -> tuple[tuple[Any, Any], float]:
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix="wan22-prompt-",
            dir=self.settings.temp_dir,
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            request_path = temporary / "request.json"
            output_path = temporary / "conditioning.safetensors"
            log_path = temporary / "worker.log"
            request_path.write_text(
                json.dumps(
                    {
                        "model_path": str(plan.model_path),
                        "prompt": prompt,
                        "negative_prompt": WAN22_NEGATIVE_PROMPT,
                        "max_sequence_length": WAN22_MAX_SEQUENCE_LENGTH,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-m",
                "latentslate_engine.runtime.wan22_prompt_worker",
                str(request_path),
                str(output_path),
            ]
            started = time.monotonic()
            self._prompt_worker_runs += 1
            with log_path.open("w", encoding="utf-8") as worker_log:
                process = subprocess.Popen(
                    command,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    while process.poll() is None:
                        check_cancelled()
                        if (
                            time.monotonic() - started
                            > WAN22_PROMPT_WORKER_TIMEOUT_SECONDS
                        ):
                            raise TimeoutError(
                                "Wan 2.2 isolated prompt encoding exceeded "
                                f"{WAN22_PROMPT_WORKER_TIMEOUT_SECONDS} seconds"
                            )
                        time.sleep(0.25)
                except BaseException:
                    self._terminate_worker(process)
                    raise
            elapsed = time.monotonic() - started
            if process.returncode != 0:
                log = log_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    "Wan 2.2 isolated prompt encoder failed"
                    + (f": {log[-8000:]}" if log else "")
                )
            if not output_path.is_file():
                raise RuntimeError(
                    "Wan 2.2 isolated prompt encoder completed without an output file"
                )

            conditioning = self._load_prompt_conditioning(output_path)
            return conditioning, elapsed

    @staticmethod
    def _load_prompt_conditioning(output_path: Path) -> tuple[Any, Any]:
        from safetensors.torch import load_file

        tensors = load_file(str(output_path), device="cpu")
        try:
            # Detach from safetensors' file-backed storage before the temporary
            # directory is removed. This is especially important on Windows.
            conditioning = (
                tensors["prompt_embeds"].clone(),
                tensors["negative_prompt_embeds"].clone(),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Wan 2.2 prompt worker output is missing {exc.args[0]!r}"
            ) from exc
        finally:
            del tensors
        return conditioning

    @staticmethod
    def _terminate_worker(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "model_id": self.load_plan.model_resource_id or self.load_plan.model_id,
            "profile": self.settings.wan22_profile,
            "quantization": self.load_plan.quantization,
            "offload": self.load_plan.offload,
            "device": self.load_plan.device,
            "pipeline_fingerprint": self.load_plan.pipeline_fingerprint,
            "loaded": self._pipeline is not None,
            "pipeline_kit": dict(self._pipeline_kit),
            "staged_text_encoder": True,
            "prompt_worker_runs": self._prompt_worker_runs,
            "cache_support": {"prompt": True, "media": False},
            "cache": self._cache.status(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def unload(self) -> None:
        with self._lock:
            self._unload_pipeline_locked()

    def _unload_pipeline_locked(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._pipeline_kit = {}
        if pipeline is not None:
            try:
                pipeline.remove_all_hooks()
            except Exception:  # noqa: BLE001 - best-effort teardown of third-party hooks
                cleanup_accelerator_memory()
            del pipeline
        cleanup_accelerator_memory()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        plan = self.load_plan
        transformer: Any | None = None
        vae: Any | None = None
        pipe: Any | None = None
        try:
            import torch
            from diffusers import (
                AutoencoderKLWan,
                UniPCMultistepScheduler,
                WanPipeline,
            )

            transformer = self._load_transformer(plan)
            vae = AutoencoderKLWan.from_pretrained(
                plan.model_path,
                subfolder="vae",
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            scheduler = UniPCMultistepScheduler.from_pretrained(
                plan.model_path,
                subfolder="scheduler",
                local_files_only=True,
            )
            model_index = json.loads(
                (plan.model_path / "model_index.json").read_text(encoding="utf-8")
            )
            pipe = WanPipeline(
                tokenizer=None,
                text_encoder=None,
                vae=vae,
                scheduler=scheduler,
                transformer=transformer,
                transformer_2=None,
                boundary_ratio=model_index.get("boundary_ratio"),
                expand_timesteps=bool(model_index.get("expand_timesteps", False)),
            )
            self._pipeline_kit = apply_pipeline_kit(pipe, plan)
            pipe.set_progress_bar_config(disable=True)
            self._pipeline = pipe
            return pipe
        except BaseException:
            self._pipeline = None
            self._pipeline_kit = {}
            if pipe is not None:
                try:
                    pipe.remove_all_hooks()
                except Exception:  # noqa: BLE001 - cleanup must preserve original failure
                    cleanup_accelerator_memory()
            del pipe
            del vae
            del transformer
            cleanup_accelerator_memory()
            raise

    @staticmethod
    def _load_transformer(plan: ResolvedRuntimePlan) -> Any:
        import torch
        from diffusers import WanTransformer3DModel

        kwargs: dict[str, Any] = {
            "subfolder": "transformer",
            "dtype": torch.bfloat16,
            "local_files_only": True,
        }
        if plan.quantization != "bf16":
            raise RuntimeError(
                f"Wan 2.2 has no proven loader for {plan.quantization!r} artifacts"
            )
        kwargs["low_cpu_mem_usage"] = plan.low_cpu_mem_usage
        transformer = WanTransformer3DModel.from_pretrained(plan.model_path, **kwargs)
        transformer.eval()
        transformer.requires_grad_(False)
        return transformer
