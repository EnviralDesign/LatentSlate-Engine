"""Family-local Z-Image Turbo state ownership and generation."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from latentslate_engine.progress import ProgressCallback, report_progress

from .contracts import ZImageIdentity, validate_request
from .model import NextDiT
from .sampling import sample
from .text import ZImageTextEncoder
from .vae import decode, load_decoder
from .weights import ZImageWeights


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    conditioning_reused: bool


class ZImageRuntime:
    """Own one immutable artifact identity and its last prompt's conditioning."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = self.model = self.weights = self.vae = self.conditioning = None

    def close(self):
        if self.weights is not None:
            self.weights.close()
        self.identity = self.model = self.weights = self.vae = self.conditioning = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        identity: ZImageIdentity,
        prompt: str,
        seed: int,
        output: Path,
        *,
        width=1024,
        height=1024,
        progress: ProgressCallback | None = None,
    ):
        validate_request(width, height, seed)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Z-Image prompt must be nonempty text")
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
                encoder = ZImageTextEncoder(
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
                report_progress(progress, 0.2, "Loading Z-Image Turbo")
                self.model = NextDiT().eval().requires_grad_(False)
                self.weights = ZImageWeights(
                    identity.diffusion.path, self.model, self.device, identity.adapters
                )
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            latent = sample(
                self.model,
                self.conditioning[1],
                seed,
                width,
                height,
                self.device,
                lambda step, total: report_progress(
                    progress,
                    0.25 + 0.6 * step / total,
                    "Sampling",
                    stage_progress=step / total,
                    detail=f"Step {step} of {total}",
                ),
            )
            timings["sampling"] = time.perf_counter() - stage
            report_progress(progress, 0.87, "VAE decode")
            stage = time.perf_counter()
            if self.vae is None:
                self.vae = load_decoder(identity.vae.path, self.device)
            pixels = decode(self.vae, latent, self.device)[0]
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
