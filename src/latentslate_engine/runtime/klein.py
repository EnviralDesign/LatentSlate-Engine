from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal

from ..config import Settings
from ..model_store import require_repository
from .cache import RuntimeCache, materialize_cached
from .diffusers_repository import (
    KLEIN4B_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from .dimensions import Dimensions, align_dimensions, floor_source_dimensions
from .kit import (
    LoraLifecycle,
    ResolvedRuntimePlan,
    RuntimeComponent,
    RuntimeDefaults,
    apply_attention_backend,
    apply_pipeline_kit,
    apply_vae_policy,
    cleanup_accelerator_memory,
    resolve_runtime_plan,
)

if TYPE_CHECKING:
    from ..tools.base import ExecutionPlan

KleinVariant = Literal["klein4b", "klein9b"]
KLEIN_VARIANTS: tuple[KleinVariant, ...] = ("klein4b", "klein9b")


KLEIN_DISTILLED_STEPS = 4
KLEIN_DIMENSION_ALIGNMENT = 16
KLEIN_MIN_SIDE = 64
KLEIN_MAX_PIXELS = 1_048_576


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
    if execution is not None and execution.recipe is not None:
        from ..klein_recipe import Klein4RuntimeRequest, revalidate_klein4_runtime_request

        request = execution.recipe
        if variant != "klein4b" or not isinstance(request, Klein4RuntimeRequest):
            raise ValueError("Klein component recipes are supported only by the Klein 4B runtime")
        if execution.model_path is not None or execution.model_resource_id is not None:
            raise ValueError("Klein component recipes cannot also declare a model override")
        if not revalidate_klein4_runtime_request(request):
            raise ValueError("Klein component recipe changed after catalog validation")

        component = request.components
        defaults = RuntimeDefaults(
            family="klein4b",
            model_id=request.base_model,
            model_path=Path(str(component["transformer"]["path"])),
            model_format="safetensors",
            device=settings.klein4b_device,
            quantization="fp8",
            attention="native",
            offload="staged",
            artifact_precision="fp8",
            artifact_quantization="native",
            vae_tiling="off",
            vae_slicing="off",
            cache="both",
            low_cpu_mem_usage=True,
            keep_pipeline_loaded=True,
            component_paths=(
                ("pipeline_support", Path(str(component["pipeline_support"]["path"]))),
                ("text_encoder", Path(str(component["text_encoder"]["path"]))),
                ("vae", Path(str(component["vae"]["path"]))),
            ),
            pipeline_parameters=(
                ("recipe_fingerprint", request.fingerprint),
                ("recipe_mode", request.mode),
                ("steps", request.steps),
                ("guidance_scale", request.guidance_scale),
            ),
        )
        resolved = resolve_runtime_plan(execution, defaults)
        if (
            resolved.quantization != "fp8"
            or resolved.offload != "staged"
            or resolved.attention != "native"
            or resolved.compile
            or resolved.loras
        ):
            raise ValueError(
                "Comfy Klein recipes require exact native-attention FP8 staged execution"
            )
        return replace(
            resolved,
            model_resource_id=str(component["transformer"]["resource_id"]),
        )

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
    stored_fp8 = bool(
        model_override and execution is not None and execution.model_format == "safetensors"
    )
    if stored_fp8:
        if variant != "klein4b":
            raise ValueError("Stored Klein FP8 artifacts are currently supported only for Klein 4B")
        quantization = "fp8"
        offload = "staged"
    model_path = (
        execution.model_path
        if model_override
        else require_repository(settings.model_root, bundle_id, model_id)
    )
    defaults = RuntimeDefaults(
        family=variant,
        model_id=model_id,
        model_path=Path(model_path),
        model_format="safetensors" if stored_fp8 else "diffusers",
        device=device,
        quantization=quantization,
        attention="native",
        offload=offload,
        artifact_precision="fp8" if stored_fp8 else "bf16",
        artifact_quantization="native",
        vae_tiling="off",
        vae_slicing="off",
        cache="both",
        low_cpu_mem_usage=True,
        keep_pipeline_loaded=True,
        group_offload_blocks=1,
        group_offload_use_stream=False,
        group_offload_record_stream=False,
    )
    resolved = resolve_runtime_plan(execution, defaults)
    if not stored_fp8:
        return resolved

    if (
        resolved.model_format != "safetensors"
        or resolved.model_precision != "fp8"
        or resolved.model_quantization != "native"
        or resolved.quantization != "fp8"
    ):
        raise ValueError(
            "Klein stored FP8 requires a SafeTensors artifact explicitly annotated "
            "precision='fp8', quantization='native'"
        )
    if resolved.attention != "native":
        raise ValueError("Klein stored FP8 currently supports native attention only")
    if resolved.offload != "staged":
        raise ValueError("Klein stored FP8 requires Engine-owned staged residency")
    if resolved.compile:
        raise ValueError("Klein stored FP8 does not yet support torch.compile")
    if resolved.loras:
        raise ValueError("Klein stored FP8 LoRA execution is not yet implemented")

    from .klein_stored_adapter import plan_comfy_klein_transformer

    adapter_plan = plan_comfy_klein_transformer(resolved.model_path)
    adapter_plan.require_available()
    support_path = require_repository(settings.model_root, bundle_id, model_id)
    validate_diffusers_repository(support_path, KLEIN4B_REPOSITORY_CONTRACT)
    components = (
        *resolved.components,
        RuntimeComponent.capture("pipeline_support", support_path),
    )
    return replace(resolved, components=components)


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
        if load_plan.model_format == "safetensors" and (
            variant != "klein4b"
            or load_plan.model_precision != "fp8"
            or load_plan.model_quantization != "native"
            or load_plan.quantization != "fp8"
            or load_plan.offload != "staged"
            or load_plan.attention != "native"
            or load_plan.compile
            or load_plan.loras
        ):
            raise ValueError(
                "Klein stored FP8 runtime requires the exact proven 4B/native/staged plan"
            )
        self.settings = settings
        self.variant = variant
        self.load_plan = load_plan
        self._pipeline: Any | None = None
        self._pipeline_kit: dict[str, Any] = {}
        self._dense_offload_hooks: dict[str, Any] = {}
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
        if self.load_plan.model_format == "safetensors":
            return "stored_fp8_staged"
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
        width: int | None,
        height: int | None,
        seed: int,
        image_paths: list[Path],
        reference_keys: list[str] | None = None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            self.load_plan.assert_same_pipeline(plan)
            check_cancelled()
            dimensions = self._resolve_dimensions(
                width=width,
                height=height,
                image_paths=image_paths,
            )
            pipeline_warm = self._pipeline is not None
            progress(0.02, f"Loading {self.display_name}")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils import load_image

            lora_status = self._lora.apply(
                pipe,
                plan.loras,
                low_cpu_mem_usage=plan.low_cpu_mem_usage,
            )
            source_images = [load_image(str(path)) for path in image_paths]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            schedule = self._schedule(plan)
            steps = int(schedule["steps"])
            guidance_scale = float(schedule["guidance_scale"])
            negative_prompt_embeds = None
            try:
                prompt_embeds, prompt_cache_hit = self._prompt_conditioning(pipe, plan, prompt)
                if guidance_scale > 1:
                    negative_prompt_embeds, _negative_cache_hit = self._prompt_conditioning(
                        pipe, plan, ""
                    )
            finally:
                self._offload_stored_text_encoder()
            check_cancelled()

            residency_session = None
            if self._is_stored_fp8(plan):
                from .klein_stored_adapter import KleinTransformerResidencySession

                residency_session = KleinTransformerResidencySession(
                    pipe.transformer,
                    onload_device=plan.device,
                    lazy_onload=True,
                )

            def callback_on_step_end(
                _pipe: Any,
                step_index: int,
                _timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                check_cancelled()
                fraction = (step_index + 1) / steps
                progress(
                    0.12 + 0.78 * fraction,
                    f"Generating image ({step_index + 1}/{steps})",
                )
                if residency_session is not None and step_index + 1 == steps:
                    # The callback runs after the final transformer invocation and
                    # before VAE decode, so release FP8 residency before the VAE onloads.
                    residency_session.close()
                return callback_kwargs

            kwargs: dict[str, Any] = {
                "prompt": None if prompt_embeds is not None else prompt,
                "prompt_embeds": prompt_embeds,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if negative_prompt_embeds is not None:
                kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            if source_images:
                kwargs["image"] = source_images[0] if len(source_images) == 1 else source_images
            if width is not None:
                kwargs["width"] = dimensions.width
                kwargs["height"] = dimensions.height

            self._active_reference_keys = reference_keys or [str(path) for path in image_paths]
            self._active_plan = plan
            self._call_media_hits = 0
            self._call_media_misses = 0
            try:
                progress(0.10, "Generating image")
                try:
                    if residency_session is None:
                        result = pipe(**kwargs)
                    else:
                        with residency_session:
                            result = pipe(**kwargs)
                finally:
                    self._offload_stored_dense_components()
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
                **dimensions.metadata(),
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
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

    @staticmethod
    def _resolve_dimensions(
        *,
        width: int | None,
        height: int | None,
        image_paths: list[Path],
    ) -> Dimensions:
        if (width is None) != (height is None):
            raise ValueError("width and height must be provided together")
        if width is not None and height is not None:
            return align_dimensions(
                width,
                height,
                alignment=KLEIN_DIMENSION_ALIGNMENT,
                min_side=KLEIN_MIN_SIDE,
                max_pixels=KLEIN_MAX_PIXELS,
            )
        if not image_paths:
            raise ValueError("width and height are required for text-to-image generation")

        # Inspect the EXIF-oriented source before any pipeline/model work. The
        # pinned Diffusers processor floors this visible canvas to its 16px grid
        # when width/height are omitted, while the call itself still omits kwargs.
        from PIL import Image, ImageOps

        with Image.open(image_paths[0]) as source:
            oriented = ImageOps.exif_transpose(source)
            source_width, source_height = oriented.size
        return floor_source_dimensions(
            source_width,
            source_height,
            alignment=KLEIN_DIMENSION_ALIGNMENT,
            min_side=KLEIN_MIN_SIDE,
            max_pixels=KLEIN_MAX_PIXELS,
        )

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
            try:
                return runtime._prepare_image_latents_cached(
                    pipe_self,
                    images,
                    batch_size,
                    generator,
                    device,
                    dtype,
                )
            finally:
                # Reference VAE encoding precedes the first transformer call.
                # Model CPU offload normally leaves the VAE resident until that
                # transformer hook runs, which is too late for I2I on constrained
                # VRAM. Release this exact VAE hook before transformer onload.
                runtime._offload_reference_vae(pipe_self)

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
            dense_hooks = self._dense_offload_hooks
            self._dense_offload_hooks = {}
            self._active_reference_keys = None
            self._active_plan = None
            self._lora.reset()
            if pipeline is not None:
                for hook in dense_hooks.values():
                    try:
                        hook.offload()
                    except Exception:  # noqa: BLE001 - best-effort third-party hook teardown
                        cleanup_accelerator_memory()
                try:
                    from accelerate.hooks import remove_hook_from_module

                    for name in dense_hooks:
                        component = getattr(pipeline, name, None)
                        if component is not None:
                            remove_hook_from_module(component, recurse=True)
                except Exception:  # noqa: BLE001 - best-effort third-party hook teardown
                    cleanup_accelerator_memory()
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
        if self._is_stored_fp8(plan):
            pipe = self._load_stored_fp8(plan)
        elif plan.quantization in {"native", "bf16"}:
            pipe = self._load_standard(plan)
        else:
            raise RuntimeError(f"Klein quantization mode {plan.quantization!r} is not implemented")

        if not self._is_stored_fp8(plan):
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

    @staticmethod
    def _is_stored_fp8(plan: ResolvedRuntimePlan) -> bool:
        return plan.model_format == "safetensors" and plan.quantization == "fp8"

    def _load_stored_fp8(self, plan: ResolvedRuntimePlan) -> Any:
        import torch
        from accelerate import cpu_offload_with_hook
        from diffusers import Flux2KleinPipeline

        from .klein_stored_adapter import (
            materialize_klein_transformer,
            plan_comfy_klein_transformer,
        )

        plan.revalidate_components()
        adapter_plan = plan_comfy_klein_transformer(plan.model_path)
        adapter_plan.require_available()
        plan.revalidate_components()
        transformer = materialize_klein_transformer(adapter_plan)
        support_path = plan.component_path("pipeline_support")
        component_names = {component.name for component in plan.components}
        standalone_components = {"text_encoder", "vae"} <= component_names
        text_hook = vae_hook = None
        pipe = None
        text_encoder = vae = None
        try:
            # The transformer materializer is intentionally multi-GB. Rebind
            # every catalog component immediately before Diffusers opens the
            # support tree, then require SafeTensors-only component loading.
            plan.revalidate_components()
            if standalone_components:
                from .klein_components import (
                    load_klein_text_encoder,
                    load_klein_vae,
                    plan_klein_pipeline_support,
                    plan_klein_small_vae,
                    plan_klein_text_encoder,
                    plan_klein_vae,
                )

                recipe_mode = str(self._schedule(plan)["mode"])
                plan_klein_pipeline_support(support_path, recipe_mode)
                text_plan = plan_klein_text_encoder(plan.component_path("text_encoder"))
                vae_plan = (
                    plan_klein_small_vae(plan.component_path("vae"))
                    if recipe_mode == "base"
                    else plan_klein_vae(plan.component_path("vae"))
                )
                text_encoder = load_klein_text_encoder(text_plan, support_path)
                vae = load_klein_vae(vae_plan, support_path)
                plan.revalidate_components()
            pipeline_kwargs: dict[str, Any] = {
                "transformer": transformer,
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": plan.low_cpu_mem_usage,
                "use_safetensors": True,
                "local_files_only": True,
                "is_distilled": self._schedule(plan)["mode"] == "distilled",
            }
            if standalone_components:
                pipeline_kwargs.update(text_encoder=text_encoder, vae=vae)
            pipe = Flux2KleinPipeline.from_pretrained(support_path, **pipeline_kwargs)
            plan.revalidate_components()
            if pipe.transformer is not transformer:
                raise RuntimeError("Klein pipeline did not retain the bound stored FP8 transformer")
            pipe.text_encoder, text_hook = cpu_offload_with_hook(
                pipe.text_encoder,
                execution_device=plan.device,
            )
            pipe.vae, vae_hook = cpu_offload_with_hook(
                pipe.vae,
                execution_device=plan.device,
                prev_module_hook=text_hook,
            )
            self._dense_offload_hooks = {
                "text_encoder": text_hook,
                "vae": vae_hook,
            }
            backend = apply_attention_backend(pipe, "native")
            apply_vae_policy(pipe, tiling=plan.vae_tiling, slicing=plan.vae_slicing)
            self._pipeline_kit = {
                "attention_backend": backend,
                "compile_scope": None,
                "offload": "engine_staged",
                "vae_tiling": plan.vae_tiling,
                "vae_slicing": plan.vae_slicing,
                "stored_weight_contract": adapter_plan.artifact_contract,
                "component_topology": (
                    "comfy_standalone" if standalone_components else "diffusers_support"
                ),
            }
            return pipe
        except BaseException:
            self._dense_offload_hooks = {}
            for hook in (vae_hook, text_hook):
                if hook is not None:
                    try:
                        hook.offload()
                    except Exception:  # noqa: BLE001 - best-effort third-party hook teardown
                        cleanup_accelerator_memory()
            if pipe is not None:
                try:
                    from accelerate.hooks import remove_hook_from_module

                    for component in (pipe.vae, pipe.text_encoder):
                        remove_hook_from_module(component, recurse=True)
                except Exception:  # noqa: BLE001 - best-effort third-party hook teardown
                    cleanup_accelerator_memory()
            del transformer
            cleanup_accelerator_memory()
            raise

    @staticmethod
    def _schedule(plan: ResolvedRuntimePlan) -> dict[str, str | int | float]:
        parameters = dict(plan.pipeline_parameters)
        if "recipe_fingerprint" not in parameters:
            return {"mode": "distilled", "steps": KLEIN_DISTILLED_STEPS, "guidance_scale": 1.0}
        mode = str(parameters.get("recipe_mode"))
        steps = int(parameters.get("steps", 0))
        guidance = float(parameters.get("guidance_scale", -1))
        expected = {"distilled": (4, 1.0), "base": (20, 5.0)}.get(mode)
        if expected != (steps, guidance):
            raise ValueError("Klein recipe schedule differs from its immutable mode contract")
        return {"mode": mode, "steps": steps, "guidance_scale": guidance}

    def _offload_stored_text_encoder(self) -> None:
        hook = self._dense_offload_hooks.get("text_encoder")
        if hook is not None:
            hook.offload()

    def _offload_stored_vae(self) -> None:
        hook = self._dense_offload_hooks.get("vae")
        if hook is not None:
            hook.offload()

    def _offload_reference_vae(self, pipe: Any) -> None:
        """Release the VAE after reference encoding and before transformer onload."""

        self._offload_stored_vae()
        plan = self._active_plan or self.load_plan
        if (
            plan.model_format != "diffusers"
            or plan.quantization != "bf16"
            or plan.offload != "model"
        ):
            return

        from accelerate.hooks import UserCpuOffloadHook

        vae = getattr(pipe, "vae", None)
        for hook in getattr(pipe, "_all_hooks", ()):
            if isinstance(hook, UserCpuOffloadHook) and hook.model is vae:
                hook.offload()
                return

        raise RuntimeError("Klein model CPU offload did not install a VAE hook")

    def _offload_stored_dense_components(self) -> None:
        for name in ("vae", "text_encoder"):
            hook = self._dense_offload_hooks.get(name)
            if hook is not None:
                hook.offload()

    def residency_poisoned(self) -> bool:
        """Whether a failed CUDA barrier made this warm runtime unsafe to reuse."""

        transformer = getattr(self._pipeline, "transformer", None)
        return bool(
            transformer is not None
            and getattr(transformer, "_latentslate_klein_residency_poisoned", None)
        )
