"""Family-local Krea Turbo lifecycle; no service or graph runtime dependencies."""

from dataclasses import dataclass
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from latentslate_engine.progress import ProgressCallback, report_progress

from .contracts import Krea2Identity, validate_request
from .model import SingleStreamDiT
from .sampling import sample
from .text import KreaTextEncoder
from .vae import load_vae
from .weights import KreaWeights, Linear


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    expanded_prompt: str
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    conditioning_reused: bool


class Krea2Runtime:
    """Own one immutable model identity and the last prompt's conditioning."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = None
        self.model = None
        self.weights = None
        self.vae = None
        self.conditioning = None

    def close(self):
        """Purge the entire identity, including mapped and virtual model state."""
        if self.weights is not None:
            self.weights.close()
        self.weights = None
        self.model = None
        self.vae = None
        self.conditioning = None
        self.identity = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def ensure_identity(self, identity: Krea2Identity) -> bool:
        """Reuse identical state; release all prior state before replacing it."""
        if self.identity == identity:
            return True
        self.close()
        self.identity = identity
        return False

    @torch.inference_mode()
    def generate(
        self,
        identity: Krea2Identity,
        prompt: str,
        seed: int,
        output: Path,
        *,
        width: int = 1024,
        height: int = 1024,
        prompt_suffix: str = "",
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Enhance, condition, sample, decode and save one ordinary RGB PNG."""
        validate_request(width, height, seed)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Krea prompt must be nonempty text")
        if not isinstance(prompt_suffix, str):
            raise TypeError("Krea prompt suffix must be text")
        started = time.perf_counter()
        reused = self.ensure_identity(identity) and self.model is not None
        conditioning_reused = (
            self.conditioning is not None and self.conditioning[0] == (prompt, prompt_suffix)
        )
        timings = {}
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            if not conditioning_reused:
                report_progress(progress, 0.02, "Prompt enhancement")
                stage = time.perf_counter()
                encoder = KreaTextEncoder(
                    identity.text_encoder.path, identity.tokenizer, self.device
                )
                try:
                    expanded = encoder.enhance(prompt)
                    timings["enhancement"] = time.perf_counter() - stage
                    report_progress(progress, 0.15, "Text conditioning")
                    stage = time.perf_counter()
                    if prompt_suffix:
                        expanded = f"{expanded}, {prompt_suffix}"
                    conditioning = encoder.encode(expanded)
                    self.conditioning = ((prompt, prompt_suffix), expanded, conditioning)
                    timings["conditioning"] = time.perf_counter() - stage
                finally:
                    encoder.close()
                    del encoder
                    gc.collect()
                    torch.cuda.empty_cache()
            _, expanded, conditioning = self.conditioning
            stage = time.perf_counter()
            if self.model is None:
                report_progress(progress, 0.2, "Loading Krea Turbo")
                self.model = (
                    SingleStreamDiT(
                        device="meta",
                        dtype=torch.bfloat16,
                        operations=SimpleNamespace(Linear=Linear),
                    )
                    .eval()
                    .requires_grad_(False)
                )
                self.weights = KreaWeights(
                    identity.diffusion.path, self.model, self.device, identity.adapters
                )
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            latent = sample(
                self.model,
                conditioning,
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
                self.vae = load_vae(identity.vae.path, self.device)
            decoded = self.vae.decode(
                latent.to(device=self.device, dtype=torch.bfloat16)
            )
            pixels = (
                decoded.float().add_(1).div_(2).clamp_(0, 1).movedim(1, -1).cpu()[0, 0]
            )
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
                expanded,
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
