from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MethodType
from typing import Any, Callable, Literal

from ..config import Settings
from ..hardware import capability_metadata, supports_nvfp4
from ..model_store import require_model_file, require_repository
from .cache import RuntimeCache, materialize_cached


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


class KleinRuntime:
    """Lazy, persistent wrapper around the upstream FLUX.2 Klein pipeline."""

    def __init__(self, settings: Settings, variant: KleinVariant):
        if variant not in KLEIN_VARIANTS:
            raise ValueError(f"Unknown Klein variant {variant!r}")
        self.settings = settings
        self.variant = variant
        self._pipeline: Any | None = None
        self._lock = Lock()
        self._active_reference_keys: list[str] | None = None
        self._call_media_hits = 0
        self._call_media_misses = 0
        self._cache = RuntimeCache(
            f"{variant}:{self.model_id}:{self.profile}",
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            max_entries=settings.cache_max_entries,
        )

    @property
    def model_id(self) -> str:
        if self.variant == "klein4b":
            return self.settings.klein4b_model_id
        return self.settings.klein_model_id

    @property
    def profile(self) -> str:
        if self.variant == "klein4b":
            return self.settings.klein4b_profile
        return self.settings.klein_profile

    @property
    def device(self) -> str:
        if self.variant == "klein4b":
            return self.settings.klein4b_device
        return self.settings.klein_device

    @property
    def bundle_id(self) -> str:
        return "klein4b-basic" if self.variant == "klein4b" else "klein9b-basic"

    def _repository_path(self, repo_id: str) -> Path:
        return require_repository(self.settings.model_root, self.bundle_id, repo_id)

    @property
    def display_name(self) -> str:
        return "FLUX.2 Klein 4B" if self.variant == "klein4b" else "FLUX.2 Klein 9B"

    def generate(
        self,
        *,
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

            source_images = [load_image(str(path)) for path in image_paths]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            prompt_embeds, prompt_cache_hit = self._prompt_conditioning(pipe, prompt)

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
            self._call_media_hits = 0
            self._call_media_misses = 0
            try:
                progress(0.10, "Generating image")
                result = pipe(**kwargs)
            finally:
                self._active_reference_keys = None
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
                "model_id": self.model_id,
                "profile": self.profile,
                "cache": {
                    "pipeline_warm": pipeline_warm,
                    "prompt_hit": prompt_cache_hit,
                    "reference_hits": self._call_media_hits,
                    "reference_misses": self._call_media_misses,
                },
                **(
                    {
                        "transformer_model_id": self.settings.klein_transformer_model_id,
                        "text_encoder_model_id": self.settings.klein_text_encoder_model_id,
                    }
                    if self.variant == "klein9b"
                    else {}
                ),
            }

    def _prompt_conditioning(self, pipe: Any, prompt: str) -> tuple[Any | None, bool]:
        if not hasattr(pipe, "encode_prompt"):
            return None, False
        key = self._cache.key(
            "prompt",
            {
                "prompt": prompt,
                "max_sequence_length": 512,
                "text_encoder_out_layers": [9, 18, 27],
            },
        )
        cached = self._cache.prompt.get(key)
        device = pipe._execution_device
        if cached is not None:
            return materialize_cached(cached, device=device), True

        encoded = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        prompt_embeds = encoded[0] if isinstance(encoded, tuple) else encoded
        self._cache.prompt.put(key, prompt_embeds)
        return prompt_embeds, False

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

        reference_keys = self._active_reference_keys or []
        image_latents = []
        for index, image in enumerate(images):
            reference_key = reference_keys[index] if index < len(reference_keys) else None
            cache_key = None
            if reference_key is not None:
                cache_key = self._cache.key(
                    "reference_vae",
                    {
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
                self._call_media_misses += 1
            image_latents.append(latent)

        image_latent_ids = pipe._prepare_image_ids(image_latents)
        packed_latents = []
        for latent in image_latents:
            packed_latents.append(pipe._pack_latents(latent).squeeze(0))
        image_latents = torch.cat(packed_latents, dim=0).unsqueeze(0)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
        return image_latents, image_latent_ids

    def status(self) -> dict[str, Any]:
        return {
            "family": self.variant,
            "model_id": self.model_id,
            "profile": self.profile,
            "device": self.device,
            "loaded": self._pipeline is not None,
            "cache_support": {"prompt": True, "media": True},
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
        profile = self.profile
        if profile == "consumer_nvfp4":
            if self.variant != "klein9b":
                raise RuntimeError(
                    "consumer_nvfp4 is currently implemented only for Klein 9B; "
                    "use bf16_model_offload for Klein 4B"
                )
            pipe = self._load_consumer_nvfp4()
        elif profile == "consumer_int8":
            if self.variant != "klein9b":
                raise RuntimeError(
                    "consumer_int8 is currently implemented only for Klein 9B; "
                    "use bf16_model_offload for Klein 4B"
                )
            pipe = self._load_consumer_int8()
        elif profile == "bf16_model_offload":
            pipe = self._load_bf16_model_offload()
        elif profile == "bf16_cuda":
            pipe = self._load_bf16_cuda()
        else:
            variable = (
                "LATENTSLATE_KLEIN4B_PROFILE"
                if self.variant == "klein4b"
                else "LATENTSLATE_KLEIN_PROFILE"
            )
            raise RuntimeError(
                f"Unknown {variable}={profile!r}; expected "
                "consumer_nvfp4, consumer_int8, bf16_model_offload, or bf16_cuda"
            )
        self._install_reference_cache(pipe)
        self._pipeline = pipe
        return pipe

    def _load_consumer_nvfp4(self) -> Any:
        try:
            import torch
            from diffusers import Flux2Transformer2DModel, NVIDIAModelOptConfig
            from modelopt.torch.opt import enable_huggingface_checkpointing
        except ImportError as exc:
            raise RuntimeError(
                "The consumer_nvfp4 Klein profile requires NVIDIA ModelOpt. Run "
                "`uv sync`, or set LATENTSLATE_KLEIN_PROFILE=consumer_int8 for the "
                "portable TorchAO fallback."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Klein 9B consumer_nvfp4 requires a visible CUDA GPU. Run "
                "`latentslate-engine doctor` to inspect the current PyTorch install."
            )
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        if not supports_nvfp4(capability):
            metadata = capability_metadata(capability)
            raise RuntimeError(
                "Klein 9B consumer_nvfp4 requires Blackwell-class SM100+ hardware; "
                f"detected {str(metadata['sm']).upper()} {metadata['architecture']}. "
                "Use Klein 4B or set LATENTSLATE_KLEIN_PROFILE=consumer_int8."
            )

        enable_huggingface_checkpointing()
        transformer_path = require_model_file(
            self.settings.model_root,
            self.bundle_id,
            self.settings.klein_transformer_model_id,
            self.settings.klein_transformer_filename,
        )
        model_path = self._repository_path(self.settings.klein_model_id)
        transformer = Flux2Transformer2DModel.from_single_file(
            transformer_path,
            config=model_path,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=NVIDIAModelOptConfig(quant_type="NVFP4"),
            low_cpu_mem_usage=True,
        )
        return self._build_consumer_pipeline(transformer)

    def _load_consumer_int8(self) -> Any:
        import torch
        from diffusers import Flux2Transformer2DModel, TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        model_path = self._repository_path(self.settings.klein_model_id)
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2)),
            low_cpu_mem_usage=False,
        )
        return self._build_consumer_pipeline(transformer)

    def _build_consumer_pipeline(self, transformer: Any) -> Any:
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer

        text_encoder_path = self._repository_path(self.settings.klein_text_encoder_model_id)
        model_path = self._repository_path(self.settings.klein_model_id)
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_path,
            torch_dtype=None,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        transformer.requires_grad_(False)
        text_encoder.requires_grad_(False)

        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.enable_model_cpu_offload(device=self.settings.klein_device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_bf16_model_offload(self) -> Any:
        import torch
        from diffusers import Flux2KleinPipeline

        model_path = self._repository_path(self.model_id)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.enable_model_cpu_offload(device=self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_bf16_cuda(self) -> Any:
        import torch
        from diffusers import Flux2KleinPipeline

        model_path = self._repository_path(self.model_id)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
