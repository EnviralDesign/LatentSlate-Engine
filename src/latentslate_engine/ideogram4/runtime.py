"""Family-local Ideogram v4 state ownership and generation."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from latentslate_engine.progress import ProgressCallback, report_progress

from .contracts import Ideogram4Identity, validate_request
from .model import Ideogram4Transformer2DModel
from .recipes import compose_caption
from .sampling import sample
from .text import Ideogram4TextEncoder
from .vae import decode, load_decoder
from .weights import Ideogram4Weights


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    conditioning_reused: bool


class Ideogram4Runtime:
    """Own one immutable artifact identity and its last prompt's conditioning."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = self.model = self.weights = self.negative_model = (
            self.negative_weights
        ) = self.vae = self.conditioning = None

    def close(self):
        if self.negative_weights is not None:
            self.negative_weights.close()
        if self.weights is not None:
            self.weights.close()
        self.identity = self.model = self.weights = self.negative_model = (
            self.negative_weights
        ) = self.vae = self.conditioning = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _release_stage_scratch(self):
        # Comfy releases cast buffers and resets AIMDO limits between nodes.
        # Retaining LoRA/decode scratch can force every warm sample to repatch.
        from comfy_aimdo import model_vbar

        torch.cuda.synchronize(self.device)
        for weights in (self.weights, self.negative_weights):
            if weights is not None:
                weights.copy_buffers.clear()
        torch.cuda.empty_cache()
        model_vbar.vbars_reset_watermark_limits()

    @torch.inference_mode()
    def generate(
        self,
        identity: Ideogram4Identity,
        prompt: str,
        seed: int,
        output: Path,
        *,
        width=1024,
        height=1024,
        steps=20,
        mu=0.0,
        std=1.75,
        sampler="euler",
        background="",
        progress: ProgressCallback | None = None,
    ):
        validate_request(width, height, seed)
        if sampler != "euler":
            raise ValueError("Ideogram v4 sampler must be euler")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Ideogram v4 prompt must be nonempty text")
        if background is None:
            background = ""
        if not isinstance(background, str):
            raise ValueError("Ideogram v4 background must be text")
        prompt = compose_caption(prompt, background)
        started = time.perf_counter()
        reused = self.identity == identity and self.model is not None
        if self.identity != identity:
            self.close()
            self.identity = identity
        conditioning_reused = (
            self.conditioning is not None and self.conditioning[0] == prompt
        )
        timings = {}
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            if not conditioning_reused:
                report_progress(progress, 0.02, "Text conditioning")
                stage = time.perf_counter()
                encoder = Ideogram4TextEncoder(
                    identity.text_encoder.path, identity.tokenizer, self.device
                )
                try:
                    self.conditioning = (prompt, encoder.encode(prompt))
                finally:
                    encoder.close()
                    del encoder
                    gc.collect()
                    torch.cuda.empty_cache()
                timings["conditioning"] = time.perf_counter() - stage
            stage = time.perf_counter()
            if self.model is None:
                report_progress(progress, 0.2, "Loading Ideogram v4")
                self.model = (
                    Ideogram4Transformer2DModel(device="meta", dtype=torch.bfloat16)
                    .eval()
                    .requires_grad_(False)
                )
                self.weights = Ideogram4Weights(
                    identity.diffusion.path, self.model, self.device, identity.adapters
                )
                if identity.negative_diffusion is not None:
                    self.negative_model = (
                        Ideogram4Transformer2DModel(device="meta", dtype=torch.bfloat16)
                        .eval()
                        .requires_grad_(False)
                    )
                    self.negative_weights = Ideogram4Weights(
                        identity.negative_diffusion.path,
                        self.negative_model,
                        self.device,
                        identity.adapters,
                    )
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            # DualModelGuider prepares the negative model before the positive.
            if self.negative_weights is not None:
                self.negative_weights.activate()
            self.weights.activate()
            latent = sample(
                self.model,
                self.negative_model,
                self.conditioning[1],
                seed,
                width,
                height,
                self.device,
                steps=steps,
                mu=mu,
                std=std,
                progress=lambda step, total: report_progress(
                    progress,
                    0.25 + 0.6 * step / total,
                    "Sampling",
                    stage_progress=step / total,
                    detail=f"Step {step} of {total}",
                ),
            )
            self._release_stage_scratch()
            timings["sampling"] = time.perf_counter() - stage
            report_progress(progress, 0.87, "VAE decode")
            stage = time.perf_counter()
            if self.vae is None:
                self.vae = load_decoder(identity.vae.path, self.device)
            pixels = decode(self.vae, latent, self.device)[0]
            self._release_stage_scratch()
            timings["decode"] = time.perf_counter() - stage
            output = Path(output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            report_progress(progress, 0.97, "Artifact encoding")
            Image.fromarray((pixels.numpy() * 255).clip(0, 255).astype(np.uint8)).save(
                output, format="PNG"
            )
            report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
            return GenerationResult(
                output,
                time.perf_counter() - started,
                timings,
                reused,
                conditioning_reused,
            )
        except BaseException:
            self.close()
            raise
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction)
