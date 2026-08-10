from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache, materialize_cached
from .kit import (
    LoraLifecycle,
    ResolvedRuntimePlan,
    RuntimeDefaults,
    apply_pipeline_kit,
    cleanup_accelerator_memory,
    resolve_runtime_plan,
)

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan

KleinVariant = Literal["klein4b", "klein9b"]
KLEIN_VARIANTS: tuple[KleinVariant, ...] = ("klein4b", "klein9b")


@dataclass(frozen=True, slots=True)
class KleinSize:
    width: int | None
    height: int | None


KLEIN_SIZE_PRESETS: dict[str, KleinSize] = {
    "source": KleinSize(width=None, height=None),
    "512x512": KleinSize(width=512, height=512),
    "768x768": KleinSize(width=768, height=768),
    "1024x1024": KleinSize(width=1024, height=1024),
    "1344x768": KleinSize(width=1344, height=768),
    "768x1344": KleinSize(width=768, height=1344),
    "1152x864": KleinSize(width=1152, height=864),
    "864x1152": KleinSize(width=864, height=1152),
}

KLEIN_DISTILLED_STEPS = 4


def _profile_modes(profile: str) -> tuple[str, str]:
    try:
        return {
            "bf16_model_offload": ("bf16", "model"),
            "bf16_cuda": ("bf16", "none"),
        }[profile]
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown Klein profile {profile!r}; expected bf16_model_offload or bf16_cuda"
        ) from exc


def resolve_klein_runtime_plan(
    settings: Settings,
    variant: KleinVariant,
    execution: ExecutionPlan | None,
) -> ResolvedRuntimePlan:
    if variant == "klein4b":
        model_id = settings.klein4b_model_id
        profile = settings.klein4b_profile
        device = settings.klein4b_device
        bundle_id = "klein4b-basic"
    elif variant == "klein9b":
        model_id = settings.klein_model_id
        profile = settings.klein_profile
        device = settings.klein_device
        bundle_id = "klein9b-basic"
    else:
        raise ValueError(f"Unknown Klein variant {variant!r}")

    quantization, offload = _profile_modes(profile)
    model_override = execution is not None and execution.model_path is not None
    model_path = (
        execution.model_path
        if model_override
        else require_repository(settings.model_root, bundle_id, model_id)
    )
    defaults = RuntimeDefaults(
        family=variant,
        model_id=model_id,
        model_path=Path(model_path),
        model_format="diffusers",
        device=device,
        quantization=quantization,
        attention="native",
        offload=offload,
        vae_tiling="off",
        vae_slicing="off",
        cache="both",
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
        group_offload_blocks=1,
        group_offload_use_stream=False,
        group_offload_record_stream=False,
    )
    return resolve_runtime_plan(execution, defaults)


class KleinRuntime:
    """Persistent FLUX.2 Klein wrapper for one exact pipeline-load fingerprint."""

    def __init__(
        self,
        settings: Settings,
        variant: KleinVariant,
        load_plan: ResolvedRuntimePlan,
    ) -> None:
        if variant not in KLEIN_VARIANTS:
            raise ValueError(f"Unknown Klein variant {variant!r}")
        if load_plan.family != variant:
            raise ValueError(
                f"Klein runtime family {variant!r} cannot load plan {load_plan.family!r}"
            )
        self.settings = settings
        self.variant = variant
        self.load_plan = load_plan
        self._pipeline: Any | None = None
        self._pipeline_kit: dict[str, Any] = {}
        self._lock = RLock()
        self._active_reference_keys: list[str] | None = None
        self._active_plan: ResolvedRuntimePlan | None = None
        self._call_media_hits = 0
        self._call_media_misses = 0
        self._lora = LoraLifecycle(max_loaded=8)
        self._cache = RuntimeCache(
            load_plan.pipeline_fingerprint,
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            max_entries=settings.cache_max_entries,
        )

    @property
    def model_id(self) -> str:
        return self.load_plan.model_id

    @property
    def profile(self) -> str:
        if self.variant == "klein4b":
            return self.settings.klein4b_profile
        return self.settings.klein_profile

    @property
    def device(self) -> str:
        return self.load_plan.device

    @property
    def bundle_id(self) -> str:
        return "klein4b-basic" if self.variant == "klein4b" else "klein9b-basic"

    @property
    def display_name(self) -> str:
        return "FLUX.2 Klein 4B" if self.variant == "klein4b" else "FLUX.2 Klein 9B"

    def _repository_path(self, repo_id: str) -> Path:
        return require_repository(self.settings.model_root, self.bundle_id, repo_id)

    def generate(
        self,
        *,
        plan: ResolvedRuntimePlan,
        prompt: str,
        output_path: Path,
        size_name: str,
        seed: int,
        image_paths: list[Path],
        reference_keys: list[str] | None = None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            self.load_plan.assert_same_pipeline(plan)
            check_cancelled()
            pipeline_warm = self._pipeline is not None
            progress(0.02, f"Loading {self.display_name}")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils import load_image

            try:
                size = KLEIN_SIZE_PRESETS[size_name]
            except KeyError as exc:
                raise ValueError(f"Unknown image size {size_name!r}") from exc
            if size_name == "source" and not image_paths:
                raise ValueError("The source size preset requires a source image")

            lora_status = self._lora.apply(
                pipe,
                plan.loras,
                low_cpu_mem_usage=plan.low_cpu_mem_usage,
            )
            source_images = [load_image(str(path)) for path in image_paths]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            prompt_embeds, prompt_cache_hit = self._prompt_conditioning(pipe, plan, prompt)

            def callback_on_step_end(
                _pipe: Any,
                step_index: int,
                _timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                check_cancelled()
                fraction = (step_index + 1) / KLEIN_DISTILLED_STEPS
                progress(
                    0.12 + 0.78 * fraction,
                    f"Generating image ({step_index + 1}/{KLEIN_DISTILLED_STEPS})",
                )
                return callback_kwargs

            kwargs: dict[str, Any] = {
                "prompt": None if prompt_embeds is not None else prompt,
                "prompt_embeds": prompt_embeds,
                "num_inference_steps": KLEIN_DISTILLED_STEPS,
                "guidance_scale": 1.0,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if source_images:
                kwargs["image"] = source_images[0] if len(source_images) == 1 else source_images
            if size.width is not None and size.height is not None:
                kwargs["width"] = size.width
                kwargs["height"] = size.height

            self._active_reference_keys = reference_keys or [str(path) for path in image_paths]
            self._active_plan = plan
            self._call_media_hits = 0
            self._call_media_misses = 0
            try:
                progress(0.10, "Generating image")
                result = pipe(**kwargs)
            finally:
                self._active_reference_keys = None
                self._active_plan = None
            check_cancelled()

            progress(0.94, "Saving PNG")
            image = result.images[0]
            image.save(output_path, format="PNG")
            progress(1.0, "Complete")
            return {
                "width": image.width,
                "height": image.height,
                "steps": KLEIN_DISTILLED_STEPS,
                "seed": seed,
                "size": size_name,
                "reference_count": len(image_paths),
                "model_variant": self.variant,
                "model_id": plan.model_resource_id or plan.model_id,
                "profile": self.profile,
                "pipeline_fingerprint": plan.pipeline_fingerprint,
                "pipeline_kit": dict(self._pipeline_kit),
                "loras": lora_status,
                "cache": {
                    "policy": plan.cache,
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": prompt_cache_hit,
                    "reference_hits": self._call_media_hits,
                    "reference_misses": self._call_media_misses,
                },
            }

    def _prompt_conditioning(
        self,
        pipe: Any,
        plan: ResolvedRuntimePlan,
        prompt: str,
    ) -> tuple[Any | None, bool]:
        if not hasattr(pipe, "encode_prompt"):
            return None, False
        device = pipe._execution_device
        if not plan.cache_prompt:
            return self._encode_prompt(pipe, prompt, device), False

        key = self._cache.key(
            "prompt",
            {
                "pipeline": plan.pipeline_fingerprint,
                "loras": plan.lora_signature,
                "prompt": prompt,
                "max_sequence_length": 512,
                "text_encoder_out_layers": [9, 18, 27],
            },
        )
        cached = self._cache.prompt.get(key)
        if cached is not None:
            return materialize_cached(cached, device=device), True

        prompt_embeds = self._encode_prompt(pipe, prompt, device)
        self._cache.prompt.put(key, prompt_embeds)
        return prompt_embeds, False

    @staticmethod
    def _encode_prompt(pipe: Any, prompt: str, device: Any) -> Any:
        encoded = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        return encoded[0] if isinstance(encoded, tuple) else encoded

    def _install_reference_cache(self, pipe: Any) -> None:
        if getattr(pipe, "_latentslate_reference_cache", False):
            return
        runtime = self

        def prepare_image_latents_cached(
            pipe_self: Any,
            images: list[Any],
            batch_size: int,
            generator: Any,
            device: Any,
            dtype: Any,
        ) -> tuple[Any, Any]:
            return runtime._prepare_image_latents_cached(
                pipe_self,
                images,
                batch_size,
                generator,
                device,
                dtype,
            )

        pipe.prepare_image_latents = MethodType(prepare_image_latents_cached, pipe)
        pipe._latentslate_reference_cache = True

    def _prepare_image_latents_cached(
        self,
        pipe: Any,
        images: list[Any],
        batch_size: int,
        generator: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[Any, Any]:
        import torch

        plan = self._active_plan or self.load_plan
        reference_keys = self._active_reference_keys or []
        image_latents = []
        for index, image in enumerate(images):
            reference_key = reference_keys[index] if index < len(reference_keys) else None
            cache_key = None
            if plan.cache_media and reference_key is not None:
                cache_key = self._cache.key(
                    "reference_vae",
                    {
                        "pipeline": plan.pipeline_fingerprint,
                        "reference": reference_key,
                        "shape": list(image.shape),
                        "dtype": str(dtype),
                    },
                )
            cached = self._cache.media.get(cache_key) if cache_key is not None else None
            if cached is not None:
                latent = materialize_cached(cached, device=device).to(dtype=dtype)
                self._call_media_hits += 1
            else:
                image = image.to(device=device, dtype=dtype)
                latent = pipe._encode_vae_image(image=image, generator=generator)
                if cache_key is not None:
                    self._cache.media.put(cache_key, latent)
                if plan.cache_media:
                    self._call_media_misses += 1
            image_latents.append(latent)

        image_latent_ids = pipe._prepare_image_ids(image_latents)
        packed_latents = [pipe._pack_latents(latent).squeeze(0) for latent in image_latents]
        image_latents = torch.cat(packed_latents, dim=0).unsqueeze(0)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
        return image_latents, image_latent_ids

    def status(self) -> dict[str, Any]:
        return {
            "family": self.variant,
            "model_id": self.load_plan.model_resource_id or self.load_plan.model_id,
            "profile": self.profile,
            "device": self.device,
            "pipeline_fingerprint": self.load_plan.pipeline_fingerprint,
            "loaded": self._pipeline is not None,
            "pipeline_kit": dict(self._pipeline_kit),
            "cache_support": {"prompt": True, "media": True},
            "cache": self._cache.status(),
            "loras": self._lora.status(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._pipeline_kit = {}
            self._active_reference_keys = None
            self._active_plan = None
            self._lora.reset()
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
        if plan.quantization in {"native", "bf16"}:
            pipe = self._load_standard(plan)
        else:
            raise RuntimeError(
                f"Klein quantization mode {plan.quantization!r} is not implemented"
            )

        self._pipeline_kit = apply_pipeline_kit(pipe, plan)
        self._install_reference_cache(pipe)
        self._pipeline = pipe
        return pipe

    def _load_standard(self, plan: ResolvedRuntimePlan) -> Any:
        import torch
        from diffusers import Flux2KleinPipeline

        kwargs: dict[str, Any] = {"low_cpu_mem_usage": plan.low_cpu_mem_usage}
        if plan.quantization == "bf16":
            kwargs["dtype"] = torch.bfloat16
        return Flux2KleinPipeline.from_pretrained(plan.model_path, **kwargs)
