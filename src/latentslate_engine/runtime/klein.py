from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ..config import Settings


@dataclass(frozen=True, slots=True)
class KleinSize:
    width: int | None
    height: int | None


KLEIN_SIZE_PRESETS: dict[str, KleinSize] = {
    "source": KleinSize(width=None, height=None),
    "768x768": KleinSize(width=768, height=768),
    "1024x1024": KleinSize(width=1024, height=1024),
    "1344x768": KleinSize(width=1344, height=768),
    "768x1344": KleinSize(width=768, height=1344),
    "1152x864": KleinSize(width=1152, height=864),
    "864x1152": KleinSize(width=864, height=1152),
}

KLEIN_DISTILLED_STEPS = 4


class KleinRuntime:
    """Lazy wrapper around the upstream Diffusers FLUX.2 Klein 9B pipeline."""

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
        seed: int,
        image_path: Path | None,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, Any]:
        with self._lock:
            check_cancelled()
            progress(0.02, "Loading FLUX.2 Klein 9B")
            pipe = self._load_pipeline()
            check_cancelled()

            import torch
            from diffusers.utils import load_image

            try:
                size = KLEIN_SIZE_PRESETS[size_name]
            except KeyError as exc:
                raise ValueError(f"Unknown image size {size_name!r}") from exc
            if size_name == "source" and image_path is None:
                raise ValueError("The source size preset requires a source image")

            source_image = load_image(str(image_path)) if image_path else None
            generator = torch.Generator(device="cpu").manual_seed(seed)

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
                "prompt": prompt,
                "num_inference_steps": KLEIN_DISTILLED_STEPS,
                "guidance_scale": 1.0,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if source_image is not None:
                kwargs["image"] = source_image
            if size.width is not None and size.height is not None:
                kwargs["width"] = size.width
                kwargs["height"] = size.height

            progress(0.10, "Generating image")
            result = pipe(**kwargs)
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
                "source_image": image_path is not None,
                "model_id": self.settings.klein_model_id,
                "profile": self.settings.klein_profile,
                "transformer_model_id": self.settings.klein_transformer_model_id,
                "text_encoder_model_id": self.settings.klein_text_encoder_model_id,
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
        profile = self.settings.klein_profile
        if profile == "consumer_nvfp4":
            self._pipeline = self._load_consumer_nvfp4()
        elif profile == "consumer_int8":
            self._pipeline = self._load_consumer_int8()
        elif profile == "bf16_model_offload":
            self._pipeline = self._load_bf16_model_offload()
        elif profile == "bf16_cuda":
            self._pipeline = self._load_bf16_cuda()
        else:
            raise RuntimeError(
                f"Unknown LATENTSLATE_KLEIN_PROFILE={profile!r}; expected "
                "consumer_nvfp4, consumer_int8, bf16_model_offload, or bf16_cuda"
            )
        return self._pipeline

    def _load_consumer_nvfp4(self) -> Any:
        try:
            import torch
            from diffusers import (
                Flux2KleinPipeline,
                Flux2Transformer2DModel,
                NVIDIAModelOptConfig,
            )
            from huggingface_hub import hf_hub_download
            from modelopt.torch.opt import enable_huggingface_checkpointing
        except ImportError as exc:
            raise RuntimeError(
                "The consumer_nvfp4 Klein profile requires the klein extra and NVIDIA "
                "ModelOpt. Linux/WSL2 is the recommended local runtime; set "
                "LATENTSLATE_KLEIN_PROFILE=consumer_int8 for the TorchAO fallback."
            ) from exc

        # ModelOpt's Hugging Face checkpoint patch is required to reconstruct the
        # quantized modules and buffers serialized in a pre-quantized checkpoint.
        enable_huggingface_checkpointing()
        transformer_path = hf_hub_download(
            repo_id=self.settings.klein_transformer_model_id,
            filename=self.settings.klein_transformer_filename,
        )
        transformer = Flux2Transformer2DModel.from_single_file(
            transformer_path,
            config=self.settings.klein_model_id,
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

        transformer = Flux2Transformer2DModel.from_pretrained(
            self.settings.klein_model_id,
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

        text_encoder = AutoModelForCausalLM.from_pretrained(
            self.settings.klein_text_encoder_model_id,
            torch_dtype=None,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.settings.klein_text_encoder_model_id)
        transformer.requires_grad_(False)
        text_encoder.requires_grad_(False)

        pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.klein_model_id,
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

        pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.klein_model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.enable_model_cpu_offload(device=self.settings.klein_device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_bf16_cuda(self) -> Any:
        import torch
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.klein_model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.to(self.settings.klein_device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
