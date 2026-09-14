"""Family-local lifecycle for the curated Qwen Image Edit 2511 native core."""

from dataclasses import dataclass
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import ProgressCallback, report_progress
from latentslate_engine.validation import validate_u64
from .contracts import Qwen2511Identity, validate_sampling
from .model import QwenImageTransformer2DModel
from .preprocessing import load_image, canvas_image, reference_images
from .sampling import sample
from .text import QwenTextEncoder
from .vae import load_vae
from .weights import Linear, RMSNorm, QwenWeights


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    width: int
    height: int
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    references_reused: bool
    reference_slots_reused: tuple[int, ...]
    positive_reused: bool
    negative_reused: bool


class Qwen2511Runtime:
    """Own one model identity and the latest request-derived conditioning."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = None
        self.model = self.weights = self.vae = self.text_encoder = None
        self.references = self.positive = self.negative = None

    def close(self):
        """Release the complete model context and every request-derived cache."""
        if self.weights is not None:
            self.weights.close()
        if self.text_encoder is not None:
            self.text_encoder.close()
        self.model = self.weights = self.vae = self.text_encoder = None
        self.references = self.positive = self.negative = None
        self.identity = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def ensure_identity(self, identity: Qwen2511Identity):
        """Reuse the exact identity or purge the entire previous context."""
        if identity == self.identity:
            return True
        self.close()
        self.identity = identity
        return False

    @torch.inference_mode()
    def generate(
        self, identity: Qwen2511Identity, prompt: str, seed: int, output: Path,
        *, image_1: Path, image_2: Path | None = None, image_3: Path | None = None,
        steps: int = 40, cfg: float = 4.0, shift: float = 3.1,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Edit the image-1 canvas using up to three independently numbered references."""
        validate_u64(seed, label="seed")
        validate_sampling(steps, cfg, shift)
        if not isinstance(prompt, str):
            raise TypeError("Qwen edit prompt must be text")
        if image_1 is None:
            raise ValueError("Qwen image edit requires image_1")
        started = time.perf_counter()
        input_key = tuple(
            (slot, FileContentIdentity.from_path(path))
            for slot, path in enumerate((image_1, image_2, image_3), start=1)
            if path is not None
        )
        reused = self.ensure_identity(identity) and self.model is not None
        references_reused = self.references is not None and self.references[0] == input_key
        reference_slots_reused = []
        timings = {}
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            stage = time.perf_counter()
            if self.vae is None:
                self.vae = load_vae(identity.vae.path, self.device)
            if not references_reused:
                report_progress(progress, 0.02, "Reference images")
                previous = {}
                if self.references is not None:
                    previous = dict(zip(self.references[0], zip(self.references[1], self.references[2])))
                visual, latents = [], []
                for slot, content in input_key:
                    cached = previous.get((slot, content))
                    if cached is not None:
                        vl_pixels, latent = cached
                        reference_slots_reused.append(slot)
                        if slot == 1:
                            width, height = self.references[3:5]
                    else:
                        pixels = load_image(content.path)
                        if slot == 1:
                            pixels = canvas_image(pixels)
                            height, width = pixels.shape[1:3]
                        vl_pixels, vae_pixels = reference_images(pixels)
                        latent = self.vae.encode(vae_pixels)
                    visual.append(vl_pixels)
                    latents.append(latent)
                self.references = (input_key, tuple(visual), tuple(latents), width, height)
                self.positive = self.negative = None
            else:
                reference_slots_reused = [slot for slot, _ in input_key]
            _, visual, latents, width, height = self.references
            timings["references"] = time.perf_counter() - stage
            positive_reused = self.positive is not None and self.positive[0] == prompt
            negative_reused = self.negative is not None
            stage = time.perf_counter()
            if not positive_reused or not negative_reused:
                report_progress(progress, 0.12, "Image and text conditioning")
                if self.text_encoder is None:
                    self.text_encoder = QwenTextEncoder(identity.text_encoder.path, identity.tokenizer, self.device)
                slots = tuple(slot for slot, _ in input_key)
                try:
                    if not positive_reused:
                        self.positive = (prompt, self.text_encoder.encode(prompt, visual, slots))
                    if not negative_reused:
                        self.negative = self.text_encoder.encode("", visual, slots)
                finally:
                    self.text_encoder.offload()
                    gc.collect()
                    torch.cuda.empty_cache()
            timings["conditioning"] = time.perf_counter() - stage
            stage = time.perf_counter()
            if self.model is None:
                report_progress(progress, 0.2, "Loading Qwen Image Edit")
                self.model = QwenImageTransformer2DModel(
                    device="meta", dtype=torch.bfloat16,
                    operations=SimpleNamespace(Linear=Linear, RMSNorm=RMSNorm, LayerNorm=torch.nn.LayerNorm),
                ).eval().requires_grad_(False)
                self.weights = QwenWeights(identity.diffusion.path, self.model, self.device, identity.adapters)
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            latent = sample(
                self.model, self.positive[1], self.negative, latents,
                seed, width, height, self.device,
                lambda step, total: report_progress(
                    progress, 0.25 + 0.6 * step / total, "Sampling",
                    stage_progress=step / total, detail=f"Step {step} of {total}",
                ),
                steps=steps, cfg=cfg, shift=shift,
            )
            timings["sampling"] = time.perf_counter() - stage
            stage = time.perf_counter()
            report_progress(progress, 0.87, "VAE decode")
            decoded = self.vae.decode(latent.to(device=self.device, dtype=torch.bfloat16))
            pixels = decoded.float().add_(1).div_(2).clamp_(0, 1).movedim(1, -1).cpu()[0, 0]
            timings["decode"] = time.perf_counter() - stage
            report_progress(progress, 0.97, "Artifact encoding")
            output = Path(output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pixels.numpy() * 255).clip(0, 255).astype(np.uint8)).save(output, format="PNG")
            report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
            return GenerationResult(
                output, width, height, time.perf_counter() - started, timings,
                reused, references_reused, tuple(reference_slots_reused), positive_reused, negative_reused,
            )
        except BaseException:
            self.close()
            raise
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction)
